"""MOE-2 共用（PREREG-18）：階段外殼、證據向量快取、累加式 ridge、γ 選擇、告訴任務／CIL 評估、各 seed 的主系統與 M3、K1–K3。

沿用既有程式，不重寫：
  AR／LIN8、σ、融合、主系統、K1   moe1_common（ar_stage、lin8_stage、sigma_b_table、eval_comps、main_eval、fold_ctx、paired、k1）
  四輪選片、一次取、彙整           selector.cil_ops（four_round、one_shot、mean_norm）
  aug（接常數 1、float64）         nc5_report.aug
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import moe1_common as M                                                   # noqa: E402
from moe1_common import C, CheckFailed, D64, N5, P, log, now              # noqa: E402
from selector.cil_ops import four_round, mean_norm, one_shot              # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402

SEED0 = M.SEED0
SEEDS = (42, 43, 44, 45, 46)
NEW = (43, 44, 45, 46)
ORDER_NAMES = M.ORDER_NAMES
GAMMAS = (1e-5, 1e-4, 1e-3, 1e-2)
BAL_GAMMAS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
G_K2 = 1e-3
NEW_KINDS = [f"s{s}" for s in NEW] + ["g0", "one42"]       # 本批新算的向量種類（train 另有 s42chk）


def kind_of(seed: int) -> str:
    return f"s{seed}"


# ── 階段外殼 ────────────────────────────────────────────────────────────────
class Run:
    """與 moe1_common.Run 相同的屬性（st、folds、full、tasks、root、ctx、pos、total、tick），moe1_common 的函式可直接吃。"""

    def __init__(self, stage: str):
        ap = argparse.ArgumentParser()
        ap.add_argument("--device", default="cpu")
        ap.add_argument("--folds", default="1-10")
        ap.add_argument("--out", default="moe2", help="outputs/navcil/<machine>/ 下的輸出子目錄（冒煙測試用 moe2_smoke）")
        a = ap.parse_args()
        if a.device != "cpu":
            raise SystemExit("MOE-2 規定 --device cpu")
        self.stage, self.folds = stage, P.parse_folds(a.folds)
        self.full = self.folds == C.FOLDS
        self.st = C.Store()
        torch.set_num_threads(int(self.st.b.cfg.get("threads", 8)))
        self.tasks = self.st.tasks
        self.base = self.st.b.out
        self.m1 = self.base / "moe1"
        self.root = self.base / a.out
        for d in (self.root, self.root / "cache", self.root / "progress"):
            d.mkdir(parents=True, exist_ok=True)
        self._ctx, self._n, self._tot = None, 0, 0
        self._vec, self._cos, self._K, self._base = {}, {}, {}, {}

    @property
    def ctx(self):
        if self._ctx is None:
            self._ctx = P.Ctx(torch.device("cpu"))
            self._ctx.timing_path, self._ctx.timing = self.root / "timing_ctx.json", {}
        return self._ctx

    def pos(self, order: str) -> list[int]:
        return [self.tasks.index(x) for x in M.ORDERS[order]]

    def total(self, n: int) -> None:
        self._n, self._tot = 0, n
        self._write()

    def tick(self, k: int = 1) -> None:
        self._n += k
        self._write()

    def _write(self) -> None:
        (self.root / "progress" / f"{self.stage}.txt").write_text(f"{self._n} {self._tot}\n")


def stage_main(stage: str, body) -> int:
    """跑一個階段：成功寫 <stage>.json 與 <stage>.done；失敗把 traceback 寫到 FAILED_<stage>.txt（不重試）。"""
    run = Run(stage)
    done, fail = run.root / f"{stage}.done", run.root / f"FAILED_{stage}.txt"
    if done.exists():
        log(f"{stage}: done, skip")
        return 0
    log(f"MOE-2 {stage} folds={run.folds} out={run.root.name} threads={torch.get_num_threads()}")
    t0 = time.perf_counter()
    try:
        res = body(run)
    except Exception:
        fail.write_text(f"[{now()}] 階段 {stage} 失敗（folds={run.folds}）\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3
    res = {"stage": stage, "folds": run.folds, "wall_s": round(time.perf_counter() - t0, 1), "finished": now(), **res}
    (run.root / f"{stage}.json").write_text(json.dumps(res, indent=1, default=M._default, ensure_ascii=False))
    done.write_text(now() + "\n")
    log(f"{stage}: 完成（{res['wall_s']:.0f}s）")
    return 0


# ── head 權重 ───────────────────────────────────────────────────────────────
def head_path(run: Run, seed: int, fold: int, task: str) -> Path:
    root = run.base / "i6" / "r2" if seed == SEED0 else run.m1 / f"i6_seed{seed}"
    return root / f"fold{fold}_{task}.pt"


def check_heads(run: Run) -> dict:
    """細則 2：5 個 seed × 折 × 4 任務的權重都要在；缺任何一個就停下（不重訓）。"""
    miss = [str(head_path(run, s, f, t).relative_to(run.base)) for s in SEEDS for f in run.folds for t in run.tasks
            if not head_path(run, s, f, t).exists()]
    if miss:
        raise CheckFailed(f"缺少 head 權重 {len(miss)} 個（不重訓）：{miss[:10]}")
    return {"n_files": len(SEEDS) * len(run.folds) * len(run.tasks), "missing": 0}


def heads_of(run: Run, fold: int, seed: int):
    return [M.load_head(head_path(run, seed, fold, t)) for t in run.tasks]


# ── 證據向量快取（vec 階段）──────────────────────────────────────────────────
def vec_path(run: Run, split: str, f: int) -> Path:
    return run.root / "cache" / f"vec_{split}_fold{f}.pt"


@torch.no_grad()
def build_vec(run: Run, fold: int, split: str) -> None:
    """每張 slide 讀一次。train：自己任務的 head／文字；validation／test：四個任務都算 → [n, 4, 512]。
    v(s)（s = 43–46）：four_round(Z, s0 + g)；g0：four_round(Z, s0)；one42：one_shot(seed 42 的 s0 + g)；s42chk（只 train）：seed 42 的 v 重算。"""
    path = vec_path(run, split, fold)
    done = path.with_suffix(".done")
    if done.exists():
        return
    ctx, lam, t_all, out = run.ctx, run.ctx.lam(), time.perf_counter(), {}
    H = {s: heads_of(run, fold, s) for s in SEEDS}
    kinds = NEW_KINDS + (["s42chk"] if split == "train" else [])
    for j, task in enumerate(ctx.tasks):
        ds, shift = ctx.ds(fold, task, split)
        rows = {k: [] for k in kinds}
        sids, labels, t_read, t_comp, n_patch = [], [], [], [], []
        ps = [j] if split == "train" else list(range(4))
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            Z = rec.Z
            r = {k: [] for k in kinds}
            for p in ps:
                ft = ctx.f_task(p)
                for s in NEW:
                    r[f"s{s}"].append(mean_norm(Z, four_round(Z, H[s][p](Z, ft), lam)))
                s0, g = H[SEED0][p].parts(Z, ft)
                s42 = s0 + g
                r["g0"].append(mean_norm(Z, four_round(Z, s0, lam)))
                r["one42"].append(mean_norm(Z, one_shot(s42)))
                if split == "train":
                    r["s42chk"].append(mean_norm(Z, four_round(Z, s42, lam)))
            for k, v in r.items():
                rows[k].append(v[0] if split == "train" else torch.stack(v))
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
            sids.append(rec.sid); labels.append(rec.label); n_patch.append(Z.shape[0])
        out[task] = {"sids": sids, "labels": torch.tensor(labels), "n_patch": torch.tensor(n_patch),
                     "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp),
                     "v": {k: torch.stack(v) for k, v in rows.items()}}
    torch.save(out, path)
    done.write_text(now() + "\n")
    log(f"vec {path.name}: n={sum(len(v['sids']) for v in out.values())} {time.perf_counter() - t_all:.0f}s")


def vecs(run: Run, split: str, f: int) -> dict:
    """依 config 任務序串接：task、label、mv（mean_vec）、v[kind]（train [N, 512]；val／test [N, 4, 512]）。
    train 另有 per[p]（每任務切片），供依序累加。slide id 與標籤逐張對齊檢查（細則 4(e)）。"""
    k = (split, f)
    if k in run._vec:
        return run._vec[k]
    raw = torch.load(vec_path(run, split, f), map_location="cpu")
    out = {"v": {}}
    if split == "train":
        s42 = torch.load(run.m1 / "cache" / f"s42v_train_fold{f}.pt", map_location="cpu")
        per = []
        for p, t in enumerate(run.tasks):
            c = run.st.b.c(f, "train", t)
            if not (raw[t]["sids"] == s42[t]["sids"] == c["sids"]) or not torch.equal(raw[t]["labels"], s42[t]["labels"]):
                raise CheckFailed(f"fold {f} {t}: train slide id／標籤與 s42v_train 或 NC-8 快取不一致")
            v = dict(raw[t]["v"]); v["s42"] = s42[t]["v"]
            per.append({"label": raw[t]["labels"], "mv": c["mean_vec"], "v": v})
        out["per"] = per
        out["label"] = torch.cat([x["label"] for x in per])
        out["task"] = torch.cat([torch.full((len(x["label"]),), p) for p, x in enumerate(per)])
        out["mv"] = torch.cat([x["mv"] for x in per])
        for kind in per[0]["v"]:
            out["v"][kind] = torch.cat([x["v"][kind] for x in per])
    else:
        S = run.st.split(split, f)
        cells = M.load_cat(run, run.m1 / "cache" / f"s42cells_{split}_fold{f}.pt", split, f, ["v_four"])
        for t in run.tasks:
            if raw[t]["sids"] != S["per_task"][t]["sids"] or not torch.equal(raw[t]["labels"], S["per_task"][t]["labels"]):
                raise CheckFailed(f"fold {f} {split} {t}: slide id／標籤與 moe0 快取不一致")
        for kind in raw[run.tasks[0]]["v"]:
            out["v"][kind] = torch.cat([raw[t]["v"][kind] for t in run.tasks])
        out["v"]["s42"] = cells["v_four"]
        out.update({"task": S["task"], "label": S["labels"], "mv": S["mean_vec"]})
    out["raw"] = raw
    run._vec[k] = out
    return out


def cos_rows(V: torch.Tensor, Ft: torch.Tensor) -> torch.Tensor:
    """逐張 v @ Fᵀ（與原快取的算法相同：每個 [512] 向量各自乘；批次矩陣乘法的累加順序不同，會差到 ~1e-6）。"""
    flat = V.reshape(-1, V.shape[-1])
    return torch.stack([x @ Ft for x in flat]).reshape(*V.shape[:-1], Ft.shape[1])


def check_vec(run: Run, f: int) -> dict:
    """細則 4 (a)–(d)：本批新算的向量與既有快取比對；不過就停下。"""
    Ft = run.ctx.F.t()
    res = {}
    tr = vecs(run, "train", f)
    res["a_s42_train_maxabs"] = float((tr["v"]["s42chk"] - tr["v"]["s42"]).abs().max())
    b_max, b_arg, c_max, c_arg = 0.0, True, 0.0, True
    for split in ("val", "test"):
        D = vecs(run, split, f)
        for s in NEW:
            ref = M.load_cat(run, run.m1 / "cache" / f"seed{s}_{split}_fold{f}.pt", split, f, ["I6_cos8"])["I6_cos8"]
            cos = cos_rows(D["v"][f"s{s}"], Ft)
            b_max = max(b_max, float((cos - ref).abs().max()))
            for q in range(4):
                rows = [2 * q, 2 * q + 1]
                b_arg &= bool(torch.equal(cos[:, q][:, rows].argmax(-1), ref[:, q][:, rows].argmax(-1)))
        cells = M.load_cat(run, run.m1 / "cache" / f"s42cells_{split}_fold{f}.pt", split, f, ["cells_cos8"])["cells_cos8"][:, :, 0]
        cos = cos_rows(D["v"]["one42"], Ft)
        c_max = max(c_max, float((cos - cells).abs().max()))
        for q in range(4):
            rows = [2 * q, 2 * q + 1]
            c_arg &= bool(torch.equal(cos[:, q][:, rows].argmax(-1), cells[:, q][:, rows].argmax(-1)))
    D, S = vecs(run, "test", f), run.st.split("test", f)
    n = torch.arange(len(D["task"]))
    cos = cos_rows(D["v"]["g0"][n, D["task"]], Ft)
    d_max = float((cos - S["g0_cos8"]).abs().max())
    d_arg = all(bool(torch.equal(cos[D["task"] == q][:, [2 * q, 2 * q + 1]].argmax(-1),
                                 S["g0_cos8"][D["task"] == q][:, [2 * q, 2 * q + 1]].argmax(-1))) for q in range(4))
    res.update({"b_seed_cos_maxabs": b_max, "b_argmax_equal": b_arg, "c_one42_cos_maxabs": c_max, "c_argmax_equal": c_arg,
                "d_g0_cos_maxabs": d_max, "d_argmax_equal": d_arg})
    ok = res["a_s42_train_maxabs"] <= 1e-6 and b_max <= 1e-6 and b_arg and c_max <= 1e-6 and c_arg and d_max <= 1e-6 and d_arg
    res["pass"] = bool(ok)
    if not ok:
        raise CheckFailed(f"fold {f} 向量快取檢查不符（細則 4）：{json.dumps(res, ensure_ascii=False)}")
    return res


# ── 累加式 ridge ────────────────────────────────────────────────────────────
class Reader:
    """一種 ridge 判讀器（細則 6）。kind = 向量種類（s42…s46、g0、one42）；use_v／use_mv 決定輸入；bal = 類別加權。"""

    def __init__(self, run: Run, name: str, kind: str | None, use_v: bool = True, use_mv: bool = False,
                 bal: bool = False, grid=GAMMAS):
        self.run, self.name, self.kind, self.use_v, self.use_mv, self.bal, self.grid = run, name, kind, use_v, use_mv, bal, grid
        self._W = {}

    def x(self, v, mv) -> torch.Tensor:
        parts = ([v] if self.use_v else []) + ([mv] if self.use_mv else [])
        return N5.aug(parts[0] if len(parts) == 1 else torch.cat(parts, 1))

    def train(self, f: int):
        """每任務 (X, y, w)；不快取（只在解 W 時用到，W 會快取）。"""
        out = []
        for p, d in enumerate(vecs(self.run, "train", f)["per"]):
            X, y = self.x(d["v"][self.kind] if self.use_v else None, d["mv"]), d["label"]
            w = None
            if self.bal:
                w = torch.zeros(len(y), dtype=D64)
                for c in (2 * p, 2 * p + 1):
                    w[y == c] = 1.0 / int((y == c).sum())
            out.append((X, y, w))
        return out

    def W(self, f: int, o: str, t: int, gamma: float):
        """序 o 的階段 t：依序累加前 t 個任務；B 只含已學類別（固定 8 類序）。回傳 (W, 類別欄)。"""
        k = (f, o, t, gamma)
        if k not in self._W:
            tr = self.train(f)
            d = tr[0][0].shape[1]
            A, cols = torch.zeros(d, d, dtype=D64), {}
            for p in self.run.pos(o)[:t]:
                X, y, w = tr[p]
                if w is None:
                    A += X.t() @ X
                    for c in (2 * p, 2 * p + 1):
                        cols[c] = X[y == c].sum(0)
                else:
                    A += X.t() @ (X * w.unsqueeze(1))
                    for c in (2 * p, 2 * p + 1):
                        m = y == c
                        cols[c] = (X[m] * w[m].unsqueeze(1)).sum(0)
            cls = sorted(cols)
            B = torch.stack([cols[c] for c in cls], 1)
            self._W[k] = (torch.linalg.solve(A + gamma * torch.eye(d, dtype=D64), B), cls)
        return self._W[k]

    def scores(self, f: int, o: str, t: int, gamma: float, D: dict, p: torch.Tensor) -> torch.Tensor:
        """[N, 8]：v 取自 p 所指任務的 head（val／test 的 v 是 [N, 4, 512]）；未學類別為 NaN。"""
        W, cls = self.W(f, o, t, gamma)
        v = D["v"][self.kind][torch.arange(len(p)), p] if self.use_v else None
        out = torch.full((len(p), 8), float("nan"), dtype=D64)
        out[:, cls] = self.x(v, D["mv"]) @ W
        return out

    def diff(self, f, o, t, gamma, D, p) -> torch.Tensor:
        """p 所指任務兩欄的分數差（第一類 − 第二類）。"""
        return M.pick(M.d_cols(self.scores(f, o, t, gamma, D, p)), p)

    def d_all(self, f, o, t, gamma, D) -> torch.Tensor:
        """[N, 4]：第 q 欄 = 用任務 q 的 head 取 v、在 q 兩欄的分數差（融合用；只用於不依 head 的 GR-bal 或 t = 4）。"""
        return torch.stack([self.diff(f, o, t, gamma, D, torch.full((len(D["task"]),), q)) for q in range(4)], 1)


def pred_of(d: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """d ≥ 0 判第一類（細則 6）。"""
    return 2 * p + (~(d >= 0)).long()


def wp_of(pred, D) -> tuple[float, list[float]]:
    return M.acc4(pred, D["label"], D["task"])


def select_gamma(run: Run, rd: Reader) -> dict:
    """細則 8：validation、reverse 序、t = 4、告訴任務 WP（四任務等權）的十折平均最大者；同分取較大的 γ。"""
    per = {}
    for g in rd.grid:
        per[g] = []
        for f in run.folds:
            D = vecs(run, "val", f)
            per[g].append(wp_of(pred_of(rd.diff(f, "reverse", 4, g, D, D["task"]), D["task"]), D)[0])
    means = {g: C.mean(v) for g, v in per.items()}
    best = max(means.values())
    star = max(g for g, v in means.items() if v == best)
    log(f"{rd.name} γ* = {star}（validation WP {means}）")
    return {"grid": list(rd.grid), "val_wp": {str(g): v for g, v in means.items()},
            "val_wp_per_fold": {str(g): v for g, v in per.items()}, "star": star,
            "edge": star in (min(rd.grid), max(rd.grid))}


def K(run: Run, f: int, o: str) -> dict:
    k = (f, o)
    if k not in run._K:
        run._K[k] = M.fold_ctx(run, f, o)
    return run._K[k]


def evaluate(run: Run, rd: Reader, gamma: float) -> dict:
    """t = 4、test、每序：告訴任務 WP（= Masked ACC）、CIL ACC（TP = AR、v 取自 τ̂ 的 head）。私有欄位存逐張判定與 d。"""
    out = {}
    for o in ORDER_NAMES:
        out[o] = []
        for f in run.folds:
            D, Kc = vecs(run, "test", f), K(run, f, o)
            d_tell = rd.diff(f, o, 4, gamma, D, D["task"])
            tell = pred_of(d_tell, D["task"])
            cil = pred_of(rd.diff(f, o, 4, gamma, D, Kc["th"]), Kc["th"])
            w, w_t = wp_of(tell, D)
            c, c_t = wp_of(cil, D)
            out[o].append({"wp": w, "wp_t": w_t, "mk": w, "cil": c, "cil_t": c_t, "_tell": tell, "_cil": cil, "_d": d_tell})
    return out


def pub(ev: dict) -> dict:
    return {o: [M.public(x) for x in v] for o, v in ev.items()}


def col(recs, k):
    return [r[k] for r in recs]


def vs(a: list, b: list, keys=("wp", "cil")) -> dict:
    return {k: M.paired(col(a, k), col(b, k)) for k in keys}


def order_same(ev: dict) -> dict:
    """細則 10：兩序 t = 4 的 test 判定是否逐張相同。"""
    a, b = ev[ORDER_NAMES[0]], ev[ORDER_NAMES[1]]
    return {"tell_diff": sum(int((x["_tell"] != y["_tell"]).sum()) for x, y in zip(a, b)),
            "cil_diff": sum(int((x["_cil"] != y["_cil"]).sum()) for x, y in zip(a, b)),
            "d_maxabs": max(float((x["_d"] - y["_d"]).abs().max()) for x, y in zip(a, b)),
            "n": sum(len(x["_tell"]) for x in a)}


# ── 各 seed 的主系統與 M3 ───────────────────────────────────────────────────
def seed_cos8(run: Run, s: int, split: str, f: int) -> torch.Tensor:
    k = (s, split, f)
    if k not in run._cos:
        run._cos[k] = (run.st.split(split, f)["I6_cos8"] if s == SEED0 else
                       M.load_cat(run, run.m1 / "cache" / f"seed{s}_{split}_fold{f}.pt", split, f, ["I6_cos8"])["I6_cos8"])
    return run._cos[k]


def lt_parts(run: Run, s: int, f: int):
    """seed s 的 (d_a [N, 4]（test）, σ_a [4]（validation）)。"""
    V = run.st.split("val", f)
    return M.d_heads(seed_cos8(run, s, "test", f)), M.sig(M.d_heads(seed_cos8(run, s, "val", f)), V["task"])


def seed_base(run: Run, s: int) -> dict:
    """{序: {"main": [每折], "M3": [每折]}}（細則 11；含私有的逐張判定）。"""
    if s not in run._base:
        out = {}
        for o in ORDER_NAMES:
            main, m3 = [], []
            for f in run.folds:
                Kc = K(run, f, o)
                I6 = seed_cos8(run, s, "test", f)
                da, sa = lt_parts(run, s, f)
                main.append(M.main_eval(I6, Kc["ar"], Kc["task"], Kc["label"]))
                m3.append(M.eval_comps([(da, sa, 1.0), (Kc["db"], Kc["sb"], 1.0)], Kc))
            out[o] = {"main": main, "M3": m3}
        run._base[s] = out
    return run._base[s]


def m1_json(run: Run, name: str) -> dict:
    return json.loads((run.m1 / f"{name}.json").read_text())


def fold_idx(j: dict, f: int) -> int:
    return j["folds"].index(f)


# ── 一致性檢查 ──────────────────────────────────────────────────────────────
def k2(run: Run, lr42: Reader) -> dict:
    """細則 13：MOE-1 S5 的算法（reverse 累加的 W、γ = 1e-3）重算 seed 42 LR 的 test WP 與兩序 CIL ACC。"""
    s5 = m1_json(run, "s5")
    maxd, wp, cil = 0.0, [], {o: [] for o in ORDER_NAMES}
    for f in run.folds:
        D, i = vecs(run, "test", f), fold_idx(s5, f)
        w = wp_of(pred_of(lr42.diff(f, "reverse", 4, G_K2, D, D["task"]), D["task"]), D)[0]
        wp.append(w)
        maxd = max(maxd, abs(w - s5["splits"]["test"]["alone"]["LR"]["wp"][i]))
        for o in ORDER_NAMES:
            th = K(run, f, o)["th"]
            c = wp_of(pred_of(lr42.diff(f, "reverse", 4, G_K2, D, th), th), D)[0]
            cil[o].append(c)
            maxd = max(maxd, abs(c - s5["lr_cil"][o]["cil"][i]))
    res = {"max_abs_fold_diff": maxd, "wp_mean": C.mean(wp), "cil_mean": {o: C.mean(v) for o, v in cil.items()},
           "rounding_checked": run.full}
    ok = maxd <= 1e-9
    if run.full:
        ok &= round(res["wp_mean"], 4) == 0.9457 and all(round(v, 4) == 0.9252 for v in res["cil_mean"].values())
    res["pass"] = bool(ok)
    if not ok:
        raise CheckFailed(f"K2 不符：{json.dumps(res, ensure_ascii=False)}")
    log(f"K2 通過：逐折最大絕對差 {maxd:.2e}")
    return res


def k3(run: Run) -> dict:
    """細則 14：seed 43–46 的主系統兩序逐折 CIL ACC 對 moe1 的 s4.json（43、44）、s7.json（45、46）。"""
    src = {43: "s4", 44: "s4", 45: "s7", 46: "s7"}
    js = {n: m1_json(run, n) for n in set(src.values())}
    per, maxd = {}, 0.0
    for s in NEW:
        j = js[src[s]]
        per[str(s)] = {}
        for o in ORDER_NAMES:
            mine = [x["cil"] for x in seed_base(run, s)[o]["main"]]
            ref = [j["rows"][str(s)][o]["main"][fold_idx(j, f)]["cil"] for f in run.folds]
            dd = max(abs(a - b) for a, b in zip(mine, ref))
            per[str(s)][o] = {"max_abs": dd, "mean": C.mean(mine)}
            maxd = max(maxd, dd)
    res = {"max_abs_fold_diff": maxd, "per_seed": per, "source": {str(k): v for k, v in src.items()}, "pass": bool(maxd <= 1e-9)}
    if not res["pass"]:
        raise CheckFailed(f"K3 不符：{json.dumps(res, ensure_ascii=False)}")
    log(f"K3 通過：逐折最大絕對差 {maxd:.2e}")
    return res


def m3_vs_moe1(run: Run) -> dict:
    """細則 11：各 seed 的 M3 逐折 CIL ACC 與 s4.json／s7.json 的 M3 最大絕對差（只報，不是 K）。"""
    src = {43: "s4", 44: "s4", 45: "s7", 46: "s7", 42: "s4"}
    js = {n: m1_json(run, n) for n in set(src.values())}
    out = {}
    for s, n in src.items():
        j = js[n]
        out[str(s)] = max(abs(seed_base(run, s)[o]["M3"][k]["cil"] - j["rows"][str(s)][o]["M3"][fold_idx(j, f)]["cil"])
                          for o in ORDER_NAMES for k, f in enumerate(run.folds))
    return out


def gate(per_order: dict, thr: float) -> dict:
    """per_order：{序: 逐折值}；每序十折平均 ≥ thr 且 > 1e-12 的折數 ≥ 7；兩序都滿足才通過。"""
    out = {}
    for o, xs in per_order.items():
        m, w = C.mean(xs), sum(x > 1e-12 for x in xs)
        out[o] = {"per_fold": xs, "mean": m, "wins": w, "pass": bool(m >= thr and w >= 7)}
    return {"orders": out, "pass": bool(all(v["pass"] for v in out.values()))}
