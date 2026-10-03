"""MOE-3 共用（PREREG-19）：階段外殼、u_K 向量快取、累加式統計量與三種判讀器（TXT、RDG、ANC）、逐階段評估、門檻。

沿用既有程式，不重寫：
  既有向量快取（v0、v(42)、mean_vec）   moe2_common.vecs（唯讀 moe2／moe1／NC-8／moe0 的快取）
  AR、主系統、M3、K1、逐折相減           moe1_common（ar_stage、k1、paired…）、moe2_common（seed_base、K、pred_of）
  s0、一次取前 K、彙整                   selector.flat_selector.text_nav_feats、selector.i6_expert.zscore、
                                         selector.multiround.top_k_select、selector.cil_ops.mean_norm
  逐階段 ACC／Forgetting／BWT            nc5_report.cil_full（Table 1 同一個函式）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import moe2_common as E                                                   # noqa: E402
from moe2_common import C, CheckFailed, D64, M, N5, P, log, now           # noqa: E402
from selector.cil_ops import mean_norm                                    # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import text_nav_feats                         # noqa: E402
from selector.i6_expert import zscore                                     # noqa: E402
from selector.multiround import top_k_select                              # noqa: E402

ORDER_NAMES = M.ORDER_NAMES
KS = (16, 32, 64, 128, 256)
K_MAIN = 64
RDG_G = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
ANC_G = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ANC_A = (0.5, 1.0, 2.0, 4.0)
G_LR = 1e-5                                    # S-LR 的固定 γ（MOE-2 的選定值）
SEEDS = E.SEEDS
TXT = ("TXT",)


def ukind(k: int) -> str:
    return f"u{k}"


U_MAIN = ukind(K_MAIN)
# 各自選一次超參數的（判讀器、輸入）組合；RDG × s42 只作描述（細則 5）
COMBOS = [("RDG", U_MAIN), ("RDG", "g0"), ("RDG", "s42"), ("ANC", U_MAIN), ("ANC", "g0"), ("ANC", "s42")]


# ── 階段外殼 ────────────────────────────────────────────────────────────────
class _M2View:
    """給 moe2_common.vecs 用的唯讀視角：root 指向 moe2（既有向量快取）。"""

    def __init__(self, run):
        self.root, self.m1, self.st, self.tasks, self._vec = run.m2, run.m1, run.st, run.tasks, {}


class Run:
    """與 moe2_common.Run 相同的屬性，moe1_common／moe2_common 的函式可直接吃；另有 m2（moe2 輸出）與 v2（讀 moe2 快取的視角）。"""

    def __init__(self, stage: str):
        ap = argparse.ArgumentParser()
        ap.add_argument("--device", default="cpu")
        ap.add_argument("--folds", default="1-10")
        ap.add_argument("--out", default="moe3", help="outputs/navcil/<machine>/ 下的輸出子目錄（冒煙測試用 moe3_smoke）")
        a = ap.parse_args()
        if a.device != "cpu":
            raise SystemExit("MOE-3 規定 --device cpu")
        self.stage, self.folds = stage, P.parse_folds(a.folds)
        self.full = self.folds == C.FOLDS
        self.st = C.Store()
        torch.set_num_threads(int(self.st.b.cfg.get("threads", 8)))
        self.tasks = self.st.tasks
        self.base = self.st.b.out
        self.m1, self.m2 = self.base / "moe1", self.base / "moe2"
        self.root = self.base / a.out
        for d in (self.root, self.root / "cache", self.root / "progress"):
            d.mkdir(parents=True, exist_ok=True)
        self._ctx, self._n, self._tot = None, 0, 0
        self._cos, self._K, self._base = {}, {}, {}
        self._data, self._stats, self._T = {}, {}, None
        self.v2 = _M2View(self)

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

    def drop(self) -> None:
        """換折時清掉逐折的向量與統計量快取（513 × 513 的 float64 統計量很多份）。"""
        self._data.clear(); self.v2._vec.clear()
        for s in self._stats.values():
            s.clear()

    def json(self, name: str) -> dict:
        return json.loads((self.root / f"{name}.json").read_text())


def stage_main(stage: str, body) -> int:
    """跑一個階段：成功寫 <stage>.json 與 <stage>.done；失敗把 traceback 寫到 FAILED_<stage>.txt（不重試）。"""
    run = Run(stage)
    done, fail = run.root / f"{stage}.done", run.root / f"FAILED_{stage}.txt"
    if done.exists():
        log(f"{stage}: done, skip")
        return 0
    log(f"MOE-3 {stage} folds={run.folds} out={run.root.name} threads={torch.get_num_threads()}")
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


def fold_i(j: dict, f: int) -> int:
    return j["folds"].index(f)


# ── u_K 向量快取（vec 階段；細則 2）──────────────────────────────────────────
def u_path(run: Run, split: str, f: int) -> Path:
    return run.root / "cache" / f"u_{split}_fold{f}.pt"


@torch.no_grad()
def build_u(run: Run, fold: int, split: str) -> None:
    """每張 slide 讀一次。s0 = 該任務兩類文字的最大 cosine（slide 內 z-score）；一次取前 K（不扣冗餘）、等權平均、L2 正規化。
    train：自己任務的文字 → [n, 512]；validation／test：四個任務的文字都算 → [n, 4, 512]。五個 K 同時算。不載入 head。"""
    path = u_path(run, split, fold)
    done = path.with_suffix(".done")
    if done.exists():
        return
    ctx, t_all, out = run.ctx, time.perf_counter(), {}
    for j, task in enumerate(ctx.tasks):
        ds, shift = ctx.ds(fold, task, split)
        rows = {k: [] for k in KS}
        sids, labels, t_read, t_comp, n_patch = [], [], [], [], []
        ps = [j] if split == "train" else list(range(4))
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            Z = rec.Z
            r = {k: [] for k in KS}
            for p in ps:
                s0 = zscore(text_nav_feats(Z, ctx.f_task(p))[:, 0])
                for k in KS:
                    r[k].append(mean_norm(Z, top_k_select(s0, k)))
            for k, v in r.items():
                rows[k].append(v[0] if split == "train" else torch.stack(v))
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
            sids.append(rec.sid); labels.append(rec.label); n_patch.append(Z.shape[0])
        out[task] = {"sids": sids, "labels": torch.tensor(labels), "n_patch": torch.tensor(n_patch),
                     "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp),
                     "u": {k: torch.stack(v) for k, v in rows.items()}}
    torch.save(out, path)
    done.write_text(now() + "\n")
    log(f"vec {path.name}: n={sum(len(v['sids']) for v in out.values())} {time.perf_counter() - t_all:.0f}s")


def data(run: Run, split: str, f: int) -> dict:
    """一折一個 split 的全部向量。kind：u16…u256（本批）、g0（v0）、s42（v(42)）、mv（mean_vec）。
    train：per[p] = {label, v[kind] [n, 512]}；validation／test：task、label、mv、v[kind] [N, 4, 512]（mv 為 [N, 512]）。
    u 快取的 slide id 與標籤與既有快取逐張比對（細則 2）。"""
    k = (split, f)
    if k in run._data:
        return run._data[k]
    D2 = E.vecs(run.v2, split, f)
    raw = torch.load(u_path(run, split, f), map_location="cpu")
    for t in run.tasks:
        if raw[t]["sids"] != D2["raw"][t]["sids"] or not torch.equal(raw[t]["labels"], D2["raw"][t]["labels"]):
            raise CheckFailed(f"fold {f} {split} {t}: u 快取的 slide id／標籤與既有快取不一致")
    if split == "train":
        per = []
        for p, t in enumerate(run.tasks):
            d = D2["per"][p]
            v = {"g0": d["v"]["g0"], "s42": d["v"]["s42"], "mv": d["mv"]}
            v.update({ukind(K): raw[t]["u"][K] for K in KS})
            per.append({"label": d["label"], "v": v})
        out = {"per": per, "raw": raw}
    else:
        v = {"g0": D2["v"]["g0"], "s42": D2["v"]["s42"]}
        v.update({ukind(K): torch.cat([raw[t]["u"][K] for t in run.tasks]) for K in KS})
        out = {"task": D2["task"], "label": D2["label"], "mv": D2["mv"], "v": v, "raw": raw, "_cos": {}}
    run._data[k] = out
    return out


def vec_of(D: dict, kind: str, p: torch.Tensor) -> torch.Tensor:
    """[N, 512]：每張取 p 所指任務版本的向量（mean_vec 與任務無關）。"""
    return D["mv"] if kind == "mv" else D["v"][kind][torch.arange(len(p)), p]


# ── 判讀器（細則 3）─────────────────────────────────────────────────────────
def anchor_T(run: Run) -> torch.Tensor:
    """[513, 8]：第 c 欄前 512 列 = 類別 c 的文字向量（ctx.F，與 TXT 相同），第 513 列 = 0。"""
    if run._T is None:
        Ft = run.ctx.F.to(D64).t()
        run._T = torch.cat([Ft, torch.zeros(1, Ft.shape[1], dtype=D64)], 0)
    return run._T


class Stats:
    """一種輸入向量的累加式統計量（累加方式同 moe2_common.Reader.W）與 RDG／ANC 的封閉解。"""

    def __init__(self, run: Run, kind: str):
        self.run, self.kind, self._g, self._ab = run, kind, {}, {}

    def clear(self) -> None:
        self._g.clear(); self._ab.clear()

    def task_stats(self, f: int, p: int):
        """任務 p：(XᵀX, {類別: 該類 x 之和})。"""
        k = (f, p)
        if k not in self._g:
            d = data(self.run, "train", f)["per"][p]
            X, y = N5.aug(d["v"][self.kind]), d["label"]
            self._g[k] = (X.t() @ X, {c: X[y == c].sum(0) for c in (2 * p, 2 * p + 1)})
        return self._g[k]

    def AB(self, f: int, o: str, t: int):
        """序 o 的階段 t：依序累加前 t 個任務；B 只含已學類別（固定 8 類序）。回傳 (A, B, 類別欄)。"""
        k = (f, o, t)
        if k not in self._ab:
            A, cols = None, {}
            for p in self.run.pos(o)[:t]:
                G, cs = self.task_stats(f, p)
                if A is None:
                    A = torch.zeros_like(G)
                A += G
                cols.update(cs)
            cls = sorted(cols)
            self._ab[k] = (A, torch.stack([cols[c] for c in cls], 1), cls)
        return self._ab[k]

    def rdg(self, f: int, o: str, t: int, gamma: float):
        A, B, cls = self.AB(f, o, t)
        return torch.linalg.solve(A + gamma * torch.eye(A.shape[0], dtype=D64), B), cls

    def anc(self, f: int, o: str, t: int, gamma: float, alpha: float):
        A, B, cls = self.AB(f, o, t)
        T = anchor_T(self.run)[:, cls]
        return torch.linalg.solve(A + gamma * torch.eye(A.shape[0], dtype=D64), B + gamma * alpha * T), cls

    def W(self, f: int, o: str, t: int, rd: tuple):
        return self.rdg(f, o, t, rd[1]) if rd[0] == "RDG" else self.anc(f, o, t, rd[1], rd[2])


def stats(run: Run, kind: str) -> Stats:
    if kind not in run._stats:
        run._stats[kind] = Stats(run, kind)
    return run._stats[kind]


def ridge_d(W: torch.Tensor, cls: list[int], X: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """p 所指任務兩欄的分數差（第一類 − 第二類）；未學類別為 NaN。"""
    out = torch.full((len(p), 8), float("nan"), dtype=D64)
    out[:, cls] = X @ W
    return M.pick(M.d_cols(out), p)


def cos8(run: Run, D: dict, kind: str) -> torch.Tensor:
    """向量 · Fᵀ（float32、逐張相乘，同 moe0／MOE-2）：[N, 4, 8]（mean_vec 為 [N, 8]，先 L2 正規化）。"""
    if kind not in D["_cos"]:
        V = F.normalize(D["mv"], dim=-1) if kind == "mv" else D["v"][kind]
        D["_cos"][kind] = E.cos_rows(V, run.ctx.F.t())
    return D["_cos"][kind]


def txt_d(run: Run, D: dict, kind: str, p: torch.Tensor) -> torch.Tensor:
    """TXT：p 所指任務版本的向量與該任務兩類文字的 cosine 差（第一類 − 第二類）。"""
    c = cos8(run, D, kind)
    if kind != "mv":
        c = c[torch.arange(len(p)), p]
    return C.pair_diff(c, p)


def reader_fn(run: Run, f: int, o: str, D: dict, kind: str, rd: tuple):
    """回傳 dfn(t, p) → [N] 的分數差；rd = ("TXT",)、("RDG", γ) 或 ("ANC", γ, α)。W 依序 o 累加到階段 t。"""
    if rd[0] == "TXT":
        return lambda t, p: txt_d(run, D, kind, p)
    Ws, st = {}, stats(run, kind)

    def fn(t, p):
        if t not in Ws:
            Ws[t] = st.W(f, o, t, rd)
        return ridge_d(*Ws[t], N5.aug(vec_of(D, kind, p)), p)
    return fn


# ── 逐階段評估（細則 4）─────────────────────────────────────────────────────
KEEP = ("acc_t", "wp_t", "forgetting", "bwt", "abar", "wp_task_t", "R", "Rm")
CMP = ("acc_t", "wp_t", "forgetting", "bwt")


def ar_stages(run: Run, f: int, o: str, D: dict) -> list[torch.Tensor]:
    return [M.ar_stage(run.st, f, o, t, D["mv"]) for t in range(1, 5)]


def cl_row(run: Run, o: str, r: dict, wp_task=None) -> dict:
    """cil_full 的結果 → 本批的欄位：acc_t、wp_t（告訴任務、已學任務等權）、Forgetting、BWT、Ā、每任務 WP [t][任務位置]。"""
    pos = run.pos(o)
    wt = [[None] * 4 for _ in range(4)]
    for t in range(4):
        for j in range(t + 1):
            wt[t][pos[j]] = r["Rm"][t][j] if wp_task is None else wp_task[pos[j]]
    wp_t = r["masked_t"] if wp_task is None else [C.mean(wp_task[pos[j]] for j in range(t + 1)) for t in range(4)]
    return {"acc_t": r["acc_t"], "wp_t": wp_t, "forgetting": r["forgetting"], "bwt": r["bwt"],
            "abar": C.mean(r["acc_t"]), "wp_task_t": wt, "R": r["R"], "Rm": r["Rm"]}


def run_cl(run: Run, f: int, o: str, D: dict, dfn, ar=None) -> dict:
    """一個系統、一折、一序的四個階段。私有欄位：t = 4 的告訴任務判定、CIL 判定與告訴任務 d。"""
    task, label = D["task"], D["label"]
    ar = ar_stages(run, f, o, D) if ar is None else ar
    last = {}

    def fn(seen):
        t = len(seen)
        m = torch.isin(task, torch.tensor(seen))
        th = ar[t - 1].argmax(-1)
        d_tell = dfn(t, task)
        tell, cil = E.pred_of(d_tell, task), E.pred_of(dfn(t, th), th)
        if t == 4:
            last.update({"_tell": tell, "_cil": cil, "_d": d_tell})
        return cil[m], tell[m], {"labels": label[m], "task": task[m]}

    out = cl_row(run, o, N5.cil_full(fn, o, run.st.tasks))
    out.update(last)
    return out


def from_s2(run: Run, o: str, rec: dict, wt: list, main: bool) -> dict:
    """moe1/s2.json 的一折 → 本批的欄位。主系統的告訴任務 WP = 真實任務 head 的 2 類正確率（wp_task_t.main）。"""
    pos = run.pos(o)
    wp_t = [C.mean(wt[t][pos[j]] for j in range(t + 1)) for t in range(4)] if main else rec["masked_t"]
    return {"acc_t": rec["acc_t"], "wp_t": wp_t, "forgetting": rec["forgetting"], "bwt": rec["bwt"],
            "abar": C.mean(rec["acc_t"]), "wp_task_t": wt, "R": rec["R"], "Rm": rec["Rm"]}


def row_diff(a: dict, b: dict) -> float:
    """兩個逐階段結果在 CMP 欄位上的最大絕對差。"""
    d = 0.0
    for k in CMP:
        x, y = a[k], b[k]
        d = max(d, *([abs(p - q) for p, q in zip(x, y)] if isinstance(x, list) else [abs(x - y)]))
    return d


def order_same(rows: dict) -> dict:
    """細則 8：兩序 t = 4 的 test 判定是否逐張相同。rows = {序: [每折（含私有欄位）]}。"""
    a, b = rows[ORDER_NAMES[0]], rows[ORDER_NAMES[1]]
    return {"tell_diff": sum(int((x["_tell"] != y["_tell"]).sum()) for x, y in zip(a, b)),
            "cil_diff": sum(int((x["_cil"] != y["_cil"]).sum()) for x, y in zip(a, b)),
            "d_maxabs": max(float((x["_d"] - y["_d"]).abs().max()) for x, y in zip(a, b)),
            "n": sum(len(x["_tell"]) for x in a)}


def t4(rows: list, key: str) -> list[float]:
    return [r[key][3] for r in rows]


# ── 超參數（細則 5）─────────────────────────────────────────────────────────
def cands(reader: str) -> list[tuple]:
    return [(reader, g) for g in RDG_G] if reader == "RDG" else [(reader, g, a) for g in ANC_G for a in ANC_A]


def ckey(rd: tuple) -> str:
    return "|".join(f"{x:g}" for x in rd[1:])


def hp_key(reader: str, kind: str) -> str:
    return f"{reader}:{kind}"


def val_objective(run: Run, f: int, kind: str, reader: str) -> dict:
    """一折：每個候選在兩序 × t = 1…4 的 validation 告訴任務 WP_t（已學任務等權）的平均。"""
    D, st = data(run, "val", f), stats(run, kind)
    task, label = D["task"], D["label"]
    X = N5.aug(vec_of(D, kind, task))
    cs = cands(reader)
    acc = {ckey(c): 0.0 for c in cs}
    for o in ORDER_NAMES:
        pos = run.pos(o)
        for t in range(1, 5):
            for c in cs:
                ok = E.pred_of(ridge_d(*st.W(f, o, t, c), X, task), task) == label
                acc[ckey(c)] += C.mean(ok[task == q].float().mean().item() for q in pos[:t]) / 8
    return acc


def select(reader: str, obj: dict) -> dict:
    """目標值最大者；完全相等才算同分，同分先取較大的 γ，再取較小的 α。"""
    cs = cands(reader)
    best = max(obj.values())
    star = max((c for c in cs if obj[ckey(c)] == best), key=lambda c: (c[1], -(c[2] if len(c) > 2 else 0.0)))
    out = {"gamma": star[1], "edge_gamma": star[1] in ((min(RDG_G), max(RDG_G)) if reader == "RDG" else (min(ANC_G), max(ANC_G)))}
    if reader == "ANC":
        out.update({"alpha": star[2], "edge_alpha": star[2] in (min(ANC_A), max(ANC_A))})
    return out


def star_rd(hp: dict, reader: str, kind: str) -> tuple:
    s = hp["combos"][hp_key(reader, kind)]["star"]
    return ("RDG", s["gamma"]) if reader == "RDG" else ("ANC", s["gamma"], s["alpha"])


def load_hp(run: Run) -> dict:
    hp = run.json("hp")
    if hp["folds"] != run.folds:
        raise CheckFailed(f"hp.json 的 folds {hp['folds']} 與本階段 {run.folds} 不同")
    return hp


def systems(hp: dict) -> dict:
    """本批重算的系統：{名稱: (輸入向量, 判讀器)}（細則 6）。"""
    return {"S-LR": ("s42", ("RDG", G_LR)),
            "S-T0": (U_MAIN, TXT),
            "S-R0": (U_MAIN, star_rd(hp, "RDG", U_MAIN)),
            "S-A0": (U_MAIN, star_rd(hp, "ANC", U_MAIN)),
            "S-R0f": ("g0", star_rd(hp, "RDG", "g0")),
            "S-A0f": ("g0", star_rd(hp, "ANC", "g0")),
            "S-Ah": ("s42", star_rd(hp, "ANC", "s42"))}


SYS_ORDER = ["S-main", "S-M3", "S-LR", "S-T0", "S-R0", "S-A0", "S-R0f", "S-A0f", "S-Ah"]


# ── 門檻 ────────────────────────────────────────────────────────────────────
def gate(per_order: dict, thr: float, wins: bool = True) -> dict:
    """per_order：{序: 逐折值}；每序十折平均 ≥ thr（且 wins 時 > 1e-12 的折數 ≥ 7）；兩序都滿足才通過。"""
    out = {}
    for o, xs in per_order.items():
        m, w = C.mean(xs), sum(x > 1e-12 for x in xs)
        out[o] = {"per_fold": xs, "mean": m, "wins": w, "pass": bool(m >= thr and (w >= 7 or not wins))}
    return {"orders": out, "pass": bool(all(v["pass"] for v in out.values()))}
