#!/usr/bin/env python3
"""MOE-1 S5：視角 × 判讀 2×2（PREREG-17 S5、細則 23–26；seed 42，告訴任務，validation 與 test）。

四個單獨的判讀器（都只輸出該任務兩類的分數差，第一類 − 第二類）：
  GT  mean_vec 與兩類文字的 cosine 差（不訓練）          GR  LIN8（γ = 0.01）= (b)
  LT  主系統 = (a)                                        LR  四輪等權向量 v 接常數 1 的累加式 ridge（8 類一欄）
LR 的 γ 以十折 validation 平均 WP 選（同分取小），選定後才算 test。

    NAVCIL_MACHINE=mac python scripts/moe1_s5.py --device cpu [--folds 1-10] [--out moe1]
輸出：outputs/navcil/<machine>/<out>/s5.json、s5.done；cache/s42v_train_fold{f}.pt、s42cells_{val,test}_fold{f}.pt
"""
from __future__ import annotations

from itertools import combinations

import torch

import moe1_common as M
import nc5_report as N5
from moe1_common import C

NAMES = ["GT", "GR", "LT", "LR"]
LR_GAMMAS = (1e-3, 1e-2, 1e-1)
COMBOS = [c for c in combinations(range(4), 2)] + [(0, 1, 2, 3)]


def lr_weights(run: M.Run, f: int) -> dict:
    """LR：A += VᵀV、每類一欄（依 reverse 序累加四個任務的 train slides；float64）。回傳 {γ: W [513, 8]}。"""
    path = run.root / "cache" / f"s42v_train_fold{f}.pt"
    M.infer(run, M.heads_of(run, f, M.SEED0), f, "train", path, "own_v")
    raw = torch.load(path, map_location="cpu")
    A, cols = torch.zeros(513, 513, dtype=M.D64), [None] * 8
    for p in run.pos("reverse"):
        t = run.tasks[p]
        if raw[t]["sids"] != run.st.b.c(f, "train", t)["sids"]:
            raise M.CheckFailed(f"fold {f} {t}: train slide 順序與 NC-8 快取不一致")
        X, y = N5.aug(raw[t]["v"]), raw[t]["labels"]
        A += X.t() @ X
        for c in (2 * p, 2 * p + 1):
            cols[c] = X[y == c].sum(0)
    B = torch.stack(cols, 1)
    return {g: torch.linalg.solve(A + g * torch.eye(513, dtype=M.D64), B) for g in LR_GAMMAS}


def lr_diff(W: torch.Tensor, v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """v [N, 512]（p 所指任務的 head 取的向量）→ p 的兩欄分數差。"""
    return M.pick(M.d_cols(N5.aug(v) @ W), p)


def first_of(d: torch.Tensor) -> torch.Tensor:
    return d >= 0


def body(run: M.Run) -> dict:
    st, Ft = run.st, run.ctx.F
    k1 = M.k1(run)
    n = len(run.folds)
    run.total(n * 3)
    W, D = {}, {"val": {}, "test": {}}                      # D[split][f] = 資料
    for f in run.folds:
        W[f] = lr_weights(run, f)
        run.tick()
        for split in ("val", "test"):
            S = st.split(split, f)
            task, label = S["task"], S["labels"]
            c = M.cells42(run, split, f)
            idx = torch.arange(len(task))
            D[split][f] = {"task": task, "first": (label - 2 * task) == 0, "label": label,
                           "v_true": c["v_four"][idx, task], "v_four": c["v_four"],
                           "GT": M.pick(M.d_cols(S["mean_vec"] @ Ft.t()), task),
                           "GR": M.pick(M.d_cols(M.lin8_stage(st, f, "reverse", 4, S["mean_vec"], M.G_LIN8)), task),
                           "LT": M.pick(M.d_heads(S["I6_cos8"]), task)}
            run.tick()

    def wp(first_pred, d):
        return C.eq4(C.task_mean(first_pred == d["first"], d["task"]))

    # 選 LR 的 γ：只讀 validation
    val_wp = {g: [wp(first_of(lr_diff(W[f][g], D["val"][f]["v_true"], D["val"][f]["task"])), D["val"][f]) for f in run.folds]
              for g in LR_GAMMAS}
    means = {g: C.mean(v) for g, v in val_wp.items()}
    g_star = min(g for g, v in means.items() if v == max(means.values()))
    M.log(f"S5 LR γ* = {g_star}（validation WP {means}）")

    res = {}
    for split in ("val", "test"):
        alone = {k: {"wp": [], "wp_t": [], "correct": 0, "n": 0} for k in NAMES}
        pairs = {f"{NAMES[a]}+{NAMES[b]}": {"both": 0, "only_first": 0, "only_second": 0, "neither": 0, "union_fold": []}
                 for a, b in COMBOS[:6]}
        fus = {"+".join(NAMES[i] for i in c): [] for c in COMBOS}
        for f in run.folds:
            d, dv = D[split][f], D["val"][f]
            d["LR"] = lr_diff(W[f][g_star], d["v_true"], d["task"])
            dv["LR"] = lr_diff(W[f][g_star], dv["v_true"], dv["task"])
            ok = {}
            for k in NAMES:
                ok[k] = first_of(d[k]) == d["first"]
                t = C.task_mean(ok[k], d["task"])
                alone[k]["wp"].append(C.eq4(t)); alone[k]["wp_t"].append(t)
                alone[k]["correct"] += int(ok[k].sum()); alone[k]["n"] += len(ok[k])
            for a, b in COMBOS[:6]:
                p = pairs[f"{NAMES[a]}+{NAMES[b]}"]
                x, y = ok[NAMES[a]], ok[NAMES[b]]
                p["both"] += int((x & y).sum()); p["only_first"] += int((x & ~y).sum())
                p["only_second"] += int((~x & y).sum()); p["neither"] += int((~x & ~y).sum())
                p["union_fold"].append(C.eq4(C.task_mean(x | y, d["task"])))
            # 融合：各自除以該折該任務 validation 上的 σ 後等權相加；f = 0 看第一項
            sg = {k: [dv[k][dv["task"] == q].std().item() for q in range(4)] for k in NAMES}
            for c in COMBOS:
                fz = None
                for i in c:
                    z = M.zs(d[NAMES[i]], d["task"], sg[NAMES[i]])
                    fz = z if fz is None else fz + z
                first = (fz > 0) | ((fz == 0) & (d[NAMES[c[0]]] >= 0))
                fus["+".join(NAMES[i] for i in c)].append(wp(first, d))
        res[split] = {"alone": alone, "pairs": pairs,
                      "fusion": {k: {"wp": v, "vs_LT": M.paired(v, alone["LT"]["wp"])} for k, v in fus.items()}}

    # LR 在 CIL 下：TP = AR；v 取自 τ̂ 的 head；在 τ̂ 的兩欄內判
    lr_cil = {}
    for o in M.ORDER_NAMES:
        cil, mainc = [], []
        for f in run.folds:
            S, d = st.split("test", f), D["test"][f]
            ar = M.ar_stage(st, f, o, 4, S["mean_vec"])
            th = ar.argmax(-1)
            v = d["v_four"][torch.arange(len(th)), th]
            pred = 2 * th + (~first_of(lr_diff(W[f][g_star], v, th))).long()
            cil.append(M.acc4(pred, d["label"], d["task"])[0])
            mainc.append(M.main_eval(S["I6_cos8"], ar, d["task"], d["label"])["cil"])
        lr_cil[o] = {"cil": cil, "main_cil": mainc, "vs_main": M.paired(cil, mainc)}
    return {"K1": k1, "names": NAMES, "lr_gamma": {"grid": list(LR_GAMMAS), "val_wp_per_fold": {str(g): v for g, v in val_wp.items()},
                                                    "val_wp": {str(g): v for g, v in means.items()}, "star": g_star},
            "splits": res, "lr_cil": lr_cil}


if __name__ == "__main__":
    raise SystemExit(M.stage_main("s5", body))
