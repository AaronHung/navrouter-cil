#!/usr/bin/env python3
"""MOE-2 E5：類別加權（PREREG-18 E5、細則 21；不設門檻）。

  每張 train slide 權重 1/n_c；A += Σ w·x xᵀ、B_c += Σ w·x；γ ∈ {1e-8, …, 1e-4}，各自選（seed 42 validation）
  LR-bal（輸入同 LR）、GR-bal（輸入同 LIN8）；M3-bal(s) = LT(seed s) + GR-bal 的 σ 固定版相加（β = 1）
  報 seed 42–46 的 LR-bal、M3-bal 的 WP、CIL ACC（兩序）；GR-bal 單獨；LR-bal − LR、M3-bal − M3（描述）

    NAVCIL_MACHINE=mac python scripts/moe2_e5.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e5.json、e5.done
"""
from __future__ import annotations

import moe2_common as E
from moe2_common import M


def grbal_parts(run: E.Run, grb: E.Reader, g: float, f: int, o: str):
    """GR-bal 的 d_b [N, 4]（test、t = 4、序 o）與 σ_b 固定值 [4]（學任務 q 那個階段、任務 q 的 validation）。"""
    Dv, Dt = E.vecs(run, "val", f), E.vecs(run, "test", f)
    pos = run.pos(o)
    sb = [grb.diff(f, o, pos.index(q) + 1, g, Dv, Dv["task"])[Dv["task"] == q].std().item() for q in range(4)]
    return grb.d_all(f, o, 4, g, Dt), sb


def m3bal_eval(run: E.Run, grb: E.Reader, g: float, s: int) -> dict:
    """{序: [每折 eval_comps 結果]}（含私有逐張判定）。"""
    out = {}
    for o in E.ORDER_NAMES:
        out[o] = []
        for f in run.folds:
            db, sb = grbal_parts(run, grb, g, f, o)
            da, sa = E.lt_parts(run, s, f)
            out[o].append(M.eval_comps([(da, sa, 1.0), (db, sb, 1.0)], E.K(run, f, o)))
    return out


def body(run: E.Run) -> dict:
    run.total(len(E.SEEDS) + 2)
    k1 = M.k1(run)
    lr = {s: E.Reader(run, f"LR({s})", E.kind_of(s)) for s in E.SEEDS}
    lrb = {s: E.Reader(run, f"LR-bal({s})", E.kind_of(s), bal=True, grid=E.BAL_GAMMAS) for s in E.SEEDS}
    grb = E.Reader(run, "GR-bal", None, use_v=False, use_mv=True, bal=True, grid=E.BAL_GAMMAS)
    g_lr, g_lrb, g_grb = E.select_gamma(run, lr[42]), E.select_gamma(run, lrb[42]), E.select_gamma(run, grb)
    run.tick()
    ev_grb = E.evaluate(run, grb, g_grb["star"])
    run.tick()
    seeds = {}
    for s in E.SEEDS:
        a, b = E.evaluate(run, lrb[s], g_lrb["star"]), E.evaluate(run, lr[s], g_lr["star"])
        mb, base = m3bal_eval(run, grb, g_grb["star"], s), E.seed_base(run, s)
        seeds[str(s)] = {"order_same_LRbal": E.order_same(a), "orders": {}}
        for o in E.ORDER_NAMES:
            m3 = [M.public(x) for x in base[o]["M3"]]
            seeds[str(s)]["orders"][o] = {"LR-bal": [M.public(x) for x in a[o]], "M3-bal": [M.public(x) for x in mb[o]],
                                          "vs_LR": E.vs(a[o], b[o]), "M3bal_vs_M3": E.vs(mb[o], m3),
                                          "LR": [M.public(x) for x in b[o]], "M3": m3}
        run.tick()
    return {"K1": k1, "gamma_LR": g_lr, "gamma_LRbal": g_lrb, "gamma_GRbal": g_grb, "GR-bal": E.pub(ev_grb), "seeds": seeds}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e5", body))
