#!/usr/bin/env python3
"""MOE-2 E3：一次取 64（PREREG-18 E3、細則 19；seed 42；不設門檻）。

  LR1 = v1(42)（seed 42 的 head 的 s 前 64、等權平均）的 ridge，自己選 γ；WP、CIL ACC；LR1 − LR(42) 的逐折差（兩序）

    NAVCIL_MACHINE=mac python scripts/moe2_e3.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e3.json、e3.done
"""
from __future__ import annotations

import moe2_common as E
from moe2_common import M


def body(run: E.Run) -> dict:
    run.total(2)
    k1 = M.k1(run)
    lr = E.Reader(run, "LR(42)", "s42")
    lr1 = E.Reader(run, "LR1", "one42")
    g_lr, g1 = E.select_gamma(run, lr), E.select_gamma(run, lr1)
    run.tick()
    ev, ev1 = E.evaluate(run, lr, g_lr["star"]), E.evaluate(run, lr1, g1["star"])
    run.tick()
    return {"K1": k1, "gamma_LR": g_lr, "gamma_LR1": g1, "LR1": E.pub(ev1), "LR": E.pub(ev),
            "vs_LR": {o: E.vs(ev1[o], ev[o]) for o in E.ORDER_NAMES}, "order_same_LR1": E.order_same(ev1)}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e3", body))
