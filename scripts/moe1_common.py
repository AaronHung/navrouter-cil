"""MOE-1 共用（PREREG-17）：階段外殼、各階段的 AR／LIN8 分數、σ、融合判定、K1／K3、訓練與推論。

沿用既有程式，不重寫：
  AR／LIN8 封閉解   nc8_report.B8.W（float64；統計量只含該序前 t 個任務）
  z = d/σ 的算術    moe0_common.zapply(center=False)
  主系統判定         moe0_common.main_preds、nc2_report.hard
  訓練               selector.cil_ops.train_selector（呼叫方式同 nc7_i6.train_one，seed 為參數）
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
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import moe0_common as C                                                   # noqa: E402
import nc1_pipeline as P                                                  # noqa: E402
import nc5_report as N5                                                   # noqa: E402
from nc8_report import ROWS                                               # noqa: E402
from selector.cil_ops import ORDERS, four_round, mean_norm, one_shot, train_selector  # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.i6_expert import I6Expert                                   # noqa: E402

log = P.log
SEED0 = 42
G_AR, G_LIN8, G_LO = 1e-3, 0.01, 1e-3
ORDER_NAMES = list(ORDERS)
D64 = torch.float64
ROW6 = ROWS[5]                                 # NC-8 主表第 6 列（主系統）
CELLS = ["一次取 64／等權", "一次取 64／softmax", "四輪／等權（現行）", "四輪／softmax"]
CELL_NOW = 2


class CheckFailed(RuntimeError):
    """一致性檢查不過（PREREG-17：停下該階段並記錄，不重試、不改期望值）。"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ── 階段外殼 ────────────────────────────────────────────────────────────────
class Run:
    def __init__(self, stage: str):
        ap = argparse.ArgumentParser()
        ap.add_argument("--device", default="cpu")
        ap.add_argument("--folds", default="1-10")
        ap.add_argument("--out", default="moe1", help="outputs/navcil/<machine>/ 下的輸出子目錄（冒煙測試用 moe1_smoke）")
        ap.add_argument("--epochs", type=int, default=P.EPOCHS, help="新訓練的 epoch 數（冒煙測試用 1；K3 一律 5）")
        a = ap.parse_args()
        if a.device != "cpu":
            raise SystemExit("MOE-1 規定 --device cpu")
        self.stage, self.folds, self.epochs = stage, P.parse_folds(a.folds), a.epochs
        self.full = self.folds == C.FOLDS
        self.st = C.Store()
        torch.set_num_threads(int(self.st.b.cfg.get("threads", 8)))
        self.tasks = self.st.tasks
        self.root = self.st.b.out / a.out
        for d in (self.root, self.root / "cache", self.root / "progress"):
            d.mkdir(parents=True, exist_ok=True)
        self._ctx, self._n, self._tot = None, 0, 0

    @property
    def ctx(self):
        """讀特徵檔才需要（文字特徵、資料集、λ*）。不呼叫 record_time，不寫既有目錄。"""
        if self._ctx is None:
            self._ctx = P.Ctx(torch.device("cpu"))
            self._ctx.timing_path, self._ctx.timing = self.root / "timing_ctx.json", {}
        return self._ctx

    def pos(self, order: str) -> list[int]:
        return [self.tasks.index(x) for x in ORDERS[order]]

    def total(self, n: int) -> None:
        self._n, self._tot = 0, n
        self._write()

    def tick(self, k: int = 1) -> None:
        self._n += k
        self._write()

    def _write(self) -> None:
        (self.root / "progress" / f"{self.stage}.txt").write_text(f"{self._n} {self._tot}\n")


def _default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def stage_main(stage: str, body) -> int:
    """跑一個階段：成功寫 <stage>.json 與 <stage>.done；失敗把 traceback 寫到 FAILED_<stage>.txt（不重試）。"""
    run = Run(stage)
    done, fail = run.root / f"{stage}.done", run.root / f"FAILED_{stage}.txt"
    if done.exists():
        log(f"{stage}: done, skip")
        return 0
    log(f"MOE-1 {stage} folds={run.folds} out={run.root.name} threads={torch.get_num_threads()} epochs={run.epochs}")
    t0 = time.perf_counter()
    try:
        res = body(run)
    except Exception:
        fail.write_text(f"[{now()}] 階段 {stage} 失敗（folds={run.folds}）\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3
    res = {"stage": stage, "folds": run.folds, "epochs": run.epochs, "wall_s": round(time.perf_counter() - t0, 1),
           "finished": now(), **res}
    (run.root / f"{stage}.json").write_text(json.dumps(res, indent=1, default=_default, ensure_ascii=False))
    done.write_text(now() + "\n")
    log(f"{stage}: 完成（{res['wall_s']:.0f}s）")
    return 0


# ── 各階段的分數 ────────────────────────────────────────────────────────────
def ar_stage(st, f: int, o: str, t: int, mv: torch.Tensor) -> torch.Tensor:
    """[N, 4]：階段 t 的 AR（γ = 1e-3）任務分數，欄 = config 任務位置；未學任務為 −inf。"""
    W, pos = st.b.W(f, o, t, G_AR)
    out = torch.full((len(mv), 4), float("-inf"), dtype=D64)
    out[:, pos] = N5.aug(mv) @ W
    return out


def lin8_stage(st, f: int, o: str, t: int, mv: torch.Tensor, gamma: float) -> torch.Tensor:
    """[N, 8]：階段 t 的 LIN8 logits，欄 = 固定 8 類序；未學任務的兩欄為 NaN。"""
    W, pos = st.b.W(f, o, t, gamma, lin8=True)
    out = torch.full((len(mv), 8), float("nan"), dtype=D64)
    out[:, [2 * p + c for p in pos for c in (0, 1)]] = N5.aug(mv) @ W
    return out


def d_heads(I6: torch.Tensor) -> torch.Tensor:
    """I6 [N, 4, 8] → [N, 4]：第 q 欄 = 任務 q 的 head 在 q 兩類的 cosine 差（第一類 − 第二類）。"""
    return torch.stack([I6[:, q, 2 * q] - I6[:, q, 2 * q + 1] for q in range(4)], 1)


def d_cols(x8: torch.Tensor) -> torch.Tensor:
    """[N, 8] → [N, 4]：第 q 欄 = 任務 q 兩類的分數差（第一類 − 第二類）。"""
    return torch.stack([x8[:, 2 * q] - x8[:, 2 * q + 1] for q in range(4)], 1)


def pick(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return x.gather(1, p.unsqueeze(1)).squeeze(1)


def sig(d_all: torch.Tensor, task: torch.Tensor) -> list[float]:
    """每任務的樣本標準差（ddof = 1）：任務 q 的 slides 上第 q 欄。未學任務（NaN）回 NaN。"""
    return [d_all[task == q, q].std().item() for q in range(4)]


def sigma_b_table(run: Run, f: int, o: str, V: dict, gamma: float):
    """LIN8 的 σ_b：重算值 [t − 1][q]（t = 1…4）與固定值 [q]（= 學任務 q 那個階段的重算值）。"""
    pos = run.pos(o)
    re = [sig(d_cols(lin8_stage(run.st, f, o, t, V["mean_vec"], gamma)), V["task"]) for t in range(1, 5)]
    return re, [re[pos.index(q)][q] for q in range(4)]


# ── 融合 ────────────────────────────────────────────────────────────────────
def zs(d: torch.Tensor, p: torch.Tensor, sg: list[float]) -> torch.Tensor:
    """z = d/σ（不減平均）；算術同 MOE-0 的 z′。"""
    return C.zapply(d, p, [(0.0, s) for s in sg], center=False)


def fused(comps, p: torch.Tensor):
    """comps：[(d_all [N, 4], σ [4], 權重)]；p [N] = 用哪個任務的 head／欄位／σ。
    回傳 (f, 是否判第一類)。f = 0 時看第一項：≥ 0 判第一類（PREREG-17 細則 4）。"""
    f = None
    for d_all, sg, w in comps:
        z = zs(pick(d_all, p), p, sg)
        if w != 1.0:
            z = w * z
        f = z if f is None else f + z
    return f, (f > 0) | ((f == 0) & (pick(comps[0][0], p) >= 0))


def fused_pred(comps, p: torch.Tensor) -> torch.Tensor:
    return 2 * p + (~fused(comps, p)[1]).long()


def acc4(pred: torch.Tensor, label: torch.Tensor, task: torch.Tensor):
    t = C.task_mean(pred == label, task)
    return C.eq4(t), t


def main_eval(I6: torch.Tensor, ar: torch.Tensor, task: torch.Tensor, label: torch.Tensor) -> dict:
    """主系統（t = 4）：WP（告訴任務）、Masked ACC（Table 1 欄）、CIL ACC。"""
    th, pred, wp, mk = C.main_preds(I6, ar, task)
    w, w_t = acc4(wp, label, task)
    c, c_t = acc4(pred, label, task)
    return {"wp": w, "wp_t": w_t, "mk": acc4(mk, label, task)[0], "cil": c, "cil_t": c_t,
            "tp": C.eq4(C.task_mean(th == task, task)), "_wp_pred": wp, "_pred": pred, "_th": th}


def eval_comps(comps, K: dict, th: torch.Tensor | None = None) -> dict:
    """融合系統（t = 4）：WP = Masked ACC = 告訴任務的融合判定；CIL = τ̂ 的 head／欄位／σ。"""
    tell = fused_pred(comps, K["task"])
    cil = fused_pred(comps, K["th"] if th is None else th)
    w, w_t = acc4(tell, K["label"], K["task"])
    c, c_t = acc4(cil, K["label"], K["task"])
    return {"wp": w, "wp_t": w_t, "mk": w, "cil": c, "cil_t": c_t, "_wp_pred": tell, "_pred": cil}


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def fold_ctx(run: Run, f: int, o: str) -> dict:
    """t = 4、序 o：test 的任務／標籤／AR 分派，LIN8（γ = 0.01）的 d_b 與 σ_b 固定值。"""
    S, V = run.st.split("test", f), run.st.split("val", f)
    ar = ar_stage(run.st, f, o, 4, S["mean_vec"])
    return {"S": S, "V": V, "task": S["task"], "label": S["labels"], "ar": ar, "th": ar.argmax(-1),
            "db": d_cols(lin8_stage(run.st, f, o, 4, S["mean_vec"], G_LIN8)),
            "sb": sigma_b_table(run, f, o, V, G_LIN8)[1]}


def paired(a, b) -> dict:
    """逐折相減 a − b：平均、贏折數（差 > 1e-12）、Wilcoxon 雙尾 p（全為 0 或折數不足時為 None）。"""
    d = [x - y for x, y in zip(a, b)]
    p = None
    if len(d) > 1 and any(v != 0 for v in d):
        p = float(stats.wilcoxon(d).pvalue)
    return {"mean": C.mean(d), "wins": sum(v > 1e-12 for v in d), "losses": sum(v < -1e-12 for v in d), "p": p, "per_fold": d}


# ── 一致性檢查 ──────────────────────────────────────────────────────────────
def nc8_main(run: Run) -> dict:
    return json.loads((run.st.b.out / "nc8" / "per_fold.json").read_text())["rows"][ROW6]


def k1(run: Run, i6_fn=None) -> dict:
    """K1：主系統 seed 42（t = 4、兩序）逐折 CIL ACC 與每任務 WP 對 nc8/per_fold.json 第 6 列，最大絕對差 ≤ 1e-9；
    十折全跑時平均四捨五入須為 WP 0.9340、CIL ACC 0.9128。i6_fn(f) 可換成重算的 I6_cos8（S3 的現行格）。"""
    rows, maxd, wp, cil = nc8_main(run), 0.0, {}, {}
    for o in ORDER_NAMES:
        wp[o], cil[o] = [], []
        for f in run.folds:
            S = run.st.split("test", f)
            I6 = S["I6_cos8"] if i6_fn is None else i6_fn(f)
            m = main_eval(I6, ar_stage(run.st, f, o, 4, S["mean_vec"]), S["task"], S["labels"])
            j = rows[o][f - 1]
            maxd = max(maxd, abs(m["cil"] - j["acc"]), *[abs(a - b) for a, b in zip(m["wp_t"], j["wp_task"])])
            wp[o].append(m["wp"]); cil[o].append(m["cil"])
    res = {"max_abs_fold_diff": maxd, "wp_mean": {o: C.mean(v) for o, v in wp.items()},
           "cil_mean": {o: C.mean(v) for o, v in cil.items()}, "rounding_checked": run.full, "folds": run.folds}
    ok = maxd <= 1e-9
    if run.full:
        ok &= all(round(v, 4) == 0.9340 for v in res["wp_mean"].values())
        ok &= all(round(v, 4) == 0.9128 for v in res["cil_mean"].values())
    res["pass"] = bool(ok)
    if not ok:
        raise CheckFailed(f"K1 不符：{json.dumps(res, ensure_ascii=False)}")
    log(f"K1 通過：逐折最大絕對差 {maxd:.2e}")
    return res


# ── 訓練（同 nc7_i6.train_one，seed 為參數；idx = 只用 train split 的一部分）──
def train_head(run: Run, fold: int, task: str, seed: int, epochs: int, idx=None):
    ctx = run.ctx
    p = ctx.tasks.index(task)
    ds, shift = ctx.ds(fold, task, "train")
    sub = list(range(len(ds))) if idx is None else [int(i) for i in idx]
    torch.manual_seed(seed)
    model = I6Expert(2)
    log(f"I6 r=2 seed {seed} fold {fold} {task}: train n={len(sub)}／{len(ds)} params={model.n_params()} epochs={epochs}")

    def slides(g):
        for k in torch.randperm(len(sub), generator=g).tolist():
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, sub[k])
            yield time.perf_counter() - t0, rec.sid, rec.Z, rec.label - shift

    return train_selector(slides, ctx.f_task(p), ctx.ls, epochs=epochs, lr=P.LR, weight_decay=P.WD,
                          seed=seed, log=log, model=model)


def k3(run: Run) -> dict:
    """K3：seed 42 重訓 (fold 1, tcga_esca)、5 epochs，state_dict 與既有檔案逐位元相同。"""
    model, _ = train_head(run, 1, "tcga_esca", SEED0, P.EPOCHS)
    ref = torch.load(run.st.b.out / "i6" / "r2" / "fold1_tcga_esca.pt", map_location="cpu")
    sd = model.state_dict()
    same = {k: bool(k in sd and torch.equal(sd[k], ref[k])) for k in ref}
    maxd = max(float((sd[k] - ref[k]).abs().max()) for k in ref if k in sd)
    res = {"bitwise_equal": same, "max_abs_diff": maxd, "pass": bool(all(same.values()) and set(sd) == set(ref))}
    if not res["pass"]:
        raise CheckFailed(f"K3 不符：{json.dumps(res, ensure_ascii=False)}")
    log("K3 通過：權重逐位元相同")
    return res


def seed_dir(run: Run, seed: int) -> Path:
    return run.root / f"i6_seed{seed}"


def train_seed(run: Run, seed: int) -> None:
    """每（折、任務）訓練一個 I6(r = 2)；完成寫 .done，重跑自動跳過。每步的梯度／參數有限性檢查在 train_selector 內。"""
    root = seed_dir(run, seed)
    root.mkdir(parents=True, exist_ok=True)
    for fold in run.folds:
        for task in run.tasks:
            part, done = root / f"fold{fold}_{task}.pt", root / f"fold{fold}_{task}.done"
            if not done.exists():
                t0 = time.perf_counter()
                model, hist = train_head(run, fold, task, seed, run.epochs)
                torch.save(model.state_dict(), part)
                (root / f"fold{fold}_{task}_train.json").write_text(json.dumps({"wall_s": time.perf_counter() - t0, "epochs": hist}))
                done.write_text(now() + "\n")
            run.tick()


def load_head(path: Path) -> I6Expert:
    m = I6Expert(2)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def heads_of(run: Run, fold: int, seed: int) -> list[I6Expert]:
    root = run.st.b.out / "i6" / "r2" if seed == SEED0 else seed_dir(run, seed)
    return [load_head(root / f"fold{fold}_{t}.pt") for t in run.tasks]


# ── 推論快取 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(run: Run, heads, fold: int, split: str, path: Path, mode: str) -> None:
    """每張 slide 讀一次（逐張記錄 t_read_s、t_compute_s）。
    mode = "cos"    I6_cos8 [N, 4, 8]：四個 head 各自四輪、等權平均的 8 類 cosine（同 moe0_infer）
    mode = "cells"  cells_cos8 [N, 4 head, 4 格, 8]（格序見 CELLS）＋ v_four [N, 4, 512]（四輪等權向量）＋ n_patch
    mode = "own_v"  v [N, 512]：自己任務的 head 四輪等權向量（train split）"""
    done = path.with_suffix(".done")
    if done.exists():
        return
    ctx, lam, t_all, out = run.ctx, run.ctx.lam(), time.perf_counter(), {}
    for j, task in enumerate(ctx.tasks):
        ds, shift = ctx.ds(fold, task, split)
        rows, sids, labels, t_read, t_comp, n_patch = {}, [], [], [], [], []
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            Z = rec.Z
            if mode == "cos":
                r = {"I6_cos8": torch.stack([mean_norm(Z, four_round(Z, m(Z, ctx.f_task(p)), lam)) @ ctx.F.t()
                                             for p, m in enumerate(heads)])}
            elif mode == "cells":
                cells, vs = [], []
                for p, m in enumerate(heads):
                    s = m(Z, ctx.f_task(p))
                    i1, i4 = one_shot(s), four_round(Z, s, lam)
                    v4 = mean_norm(Z, i4)
                    vv = [mean_norm(Z, i1), mean_norm(Z, i1, F.softmax(s.index_select(0, i1), dim=0)),
                          v4, mean_norm(Z, i4, F.softmax(s.index_select(0, i4), dim=0))]
                    cells.append(torch.stack([v @ ctx.F.t() for v in vv])); vs.append(v4)
                r = {"cells_cos8": torch.stack(cells), "v_four": torch.stack(vs)}
            elif mode == "own_v":
                r = {"v": mean_norm(Z, four_round(Z, heads[j](Z, ctx.f_task(j)), lam))}
            else:
                raise ValueError(mode)
            for k, v in r.items():
                rows.setdefault(k, []).append(v)
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
            sids.append(rec.sid); labels.append(rec.label); n_patch.append(Z.shape[0])
        out[task] = {"sids": sids, "labels": torch.tensor(labels), "n_patch": torch.tensor(n_patch),
                     "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp),
                     **{k: torch.stack(v) for k, v in rows.items()}}
    torch.save(out, path)
    done.write_text(now() + "\n")
    log(f"infer {path.name}: n={sum(len(v['sids']) for v in out.values())} {time.perf_counter() - t_all:.0f}s")


def load_cat(run: Run, path: Path, split: str, f: int, keys) -> dict:
    """讀推論快取並依 config 任務序串接；val／test 檢查 slide id 與 moe0 快取逐張相同。"""
    raw = torch.load(path, map_location="cpu")
    if split in ("val", "test"):
        ref = run.st.split(split, f)["per_task"]
        for t in run.tasks:
            if raw[t]["sids"] != ref[t]["sids"] or not torch.equal(raw[t]["labels"], ref[t]["labels"]):
                raise CheckFailed(f"{path.name} {t}: slide id／labels 與 moe0 快取不一致")
    out = {k: torch.cat([raw[t][k] for t in run.tasks]) for k in keys}
    out["per_task"] = raw
    return out


def seed_cos8(run: Run, seed: int, split: str, f: int) -> torch.Tensor:
    """[N, 4, 8]：該 seed 四個 head 的四輪 8 類 cosine（seed 42 取 moe0 快取）。"""
    if seed == SEED0:
        return run.st.split(split, f)["I6_cos8"]
    return load_cat(run, run.root / "cache" / f"seed{seed}_{split}_fold{f}.pt", split, f, ["I6_cos8"])["I6_cos8"]


def infer_seed(run: Run, seed: int) -> None:
    for fold in run.folds:
        heads = heads_of(run, fold, seed)
        for split in ("val", "test"):
            infer(run, heads, fold, split, run.root / "cache" / f"seed{seed}_{split}_fold{fold}.pt", "cos")
            run.tick()


def cells42(run: Run, split: str, f: int) -> dict:
    """seed 42 的 2×2 四格與四輪等權向量（S3、S5 共用；誰先跑誰建）。"""
    path = run.root / "cache" / f"s42cells_{split}_fold{f}.pt"
    infer(run, heads_of(run, f, SEED0), f, split, path, "cells")
    return load_cat(run, path, split, f, ["cells_cos8", "v_four", "n_patch"])


# ── 每個 seed 的主系統與 M3（S4、S7 共用）──────────────────────────────────
def seed_rows(run: Run, seeds) -> dict:
    """每個 seed、兩序、逐折：主系統與 M3（σ 固定版，σ_a 用該 seed 的 head）的 WP／Masked ACC／CIL ACC 與逐折相減。"""
    out = {}
    for s in seeds:
        out[str(s)] = {}
        for o in ORDER_NAMES:
            main, m3 = [], []
            for f in run.folds:
                K = fold_ctx(run, f, o)
                I6, I6v = seed_cos8(run, s, "test", f), seed_cos8(run, s, "val", f)
                main.append(public(main_eval(I6, K["ar"], K["task"], K["label"])))
                m3.append(public(eval_comps([(d_heads(I6), sig(d_heads(I6v), K["V"]["task"]), 1.0), (K["db"], K["sb"], 1.0)], K)))
            out[str(s)][o] = {"main": main, "M3": m3,
                              "paired": {k: paired([x[k] for x in m3], [x[k] for x in main]) for k in ("wp", "mk", "cil")}}
    return out
