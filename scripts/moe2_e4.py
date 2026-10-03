#!/usr/bin/env python3
"""MOE-2 E4：與整片向量合併（PREREG-18 E4、細則 20；門檻 G-CAT）。

  LRG(s)：輸入 [v(s); mean_vec; 1]（1025），自己選 γ（seed 42 validation）；seed 42–46 的 WP、CIL ACC；LRG − 同 seed LR（兩序）
  G-CAT（每序）：每折 seed 43–46 的 [WP(LRG(s)) − WP(LR(s))] 平均
  分數層級相加（seed 42、t = 4、告訴任務，validation 與 test）：LR＋GR、LT＋LR＋GR（各自除以該任務 validation 的 σ）

    NAVCIL_MACHINE=mac python scripts/moe2_e4.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e4.json、e4.done
"""
from __future__ import annotations

import moe2_common as E
from moe2_common import C, M

SUMS = (("LR", "GR"), ("LT", "LR", "GR"))


def score_sums(run: E.Run, lr42: E.Reader, g: float) -> dict:
    """細則 20：GR = LIN8（γ = 0.01、reverse、t = 4）；LR = LR(42)（reverse、γ*）；LT = 主系統的 d_a。f = 0 看第一項。"""
    st, out = run.st, {}
    d = {}
    for split in ("val", "test"):
        d[split] = {}
        for f in run.folds:
            S, D = st.split(split, f), E.vecs(run, split, f)
            task = S["task"]
            d[split][f] = {"task": task, "first": (S["labels"] - 2 * task) == 0,
                           "LT": M.pick(M.d_heads(S["I6_cos8"]), task),
                           "GR": M.pick(M.d_cols(M.lin8_stage(st, f, "reverse", 4, S["mean_vec"], M.G_LIN8)), task),
                           "LR": lr42.diff(f, "reverse", 4, g, D, task)}
    for split in ("val", "test"):
        alone = {k: [] for k in ("LT", "GR", "LR")}
        fus = {"+".join(c): [] for c in SUMS}
        for f in run.folds:
            x, xv = d[split][f], d["val"][f]
            for k in alone:
                alone[k].append(C.eq4(C.task_mean((x[k] >= 0) == x["first"], x["task"])))
            sg = {k: [xv[k][xv["task"] == q].std().item() for q in range(4)] for k in alone}
            for c in SUMS:
                fz = None
                for k in c:
                    z = M.zs(x[k], x["task"], sg[k])
                    fz = z if fz is None else fz + z
                first = (fz > 0) | ((fz == 0) & (x[c[0]] >= 0))
                fus["+".join(c)].append(C.eq4(C.task_mean(first == x["first"], x["task"])))
        out[split] = {"alone": alone, "fusion": {k: {"wp": v, "vs_LT": M.paired(v, alone["LT"])} for k, v in fus.items()}}
    return out


def body(run: E.Run) -> dict:
    run.total(len(E.SEEDS) + 2)
    k1 = M.k1(run)
    lr = {s: E.Reader(run, f"LR({s})", E.kind_of(s)) for s in E.SEEDS}
    lrg = {s: E.Reader(run, f"LRG({s})", E.kind_of(s), use_mv=True) for s in E.SEEDS}
    g_lr, g_lrg = E.select_gamma(run, lr[42]), E.select_gamma(run, lrg[42])
    run.tick()
    seeds, wpd = {}, {}
    for s in E.SEEDS:
        a, b = E.evaluate(run, lrg[s], g_lrg["star"]), E.evaluate(run, lr[s], g_lr["star"])
        seeds[str(s)] = {"order_same_LRG": E.order_same(a),
                         "orders": {o: {"LRG": [M.public(x) for x in a[o]], "LR": [M.public(x) for x in b[o]],
                                        "vs_LR": E.vs(a[o], b[o])} for o in E.ORDER_NAMES}}
        wpd[s] = {o: seeds[str(s)]["orders"][o]["vs_LR"]["wp"]["per_fold"] for o in E.ORDER_NAMES}
        run.tick()
    g_cat = E.gate({o: [C.mean(wpd[s][o][i] for s in E.NEW) for i in range(len(run.folds))] for o in E.ORDER_NAMES}, 0.005)
    g_cat["valid"] = run.full
    sums = score_sums(run, lr[42], g_lr["star"])
    run.tick()
    return {"K1": k1, "gamma_LR": g_lr, "gamma_LRG": g_lrg, "seeds": seeds, "G-CAT": g_cat, "score_sums": sums}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e4", body))
