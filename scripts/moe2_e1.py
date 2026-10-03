#!/usr/bin/env python3
"""MOE-2 E1：LR 多 seed 確認（PREREG-18 E1、細則 10–14、17；門檻 G-LR）。

  K1、K2（seed 42 LR、γ = 1e-3 對 moe1/s5.json）、K3（seed 43–46 主系統 CIL 對 s4.json／s7.json）
  LR 的 γ（seed 42 validation）→ seed 42–46 的 LR：WP、CIL ACC、每任務 WP；LR − 主系統、LR − M3（兩序）；兩序是否逐張相同

    NAVCIL_MACHINE=mac python scripts/moe2_e1.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e1.json、e1.done
"""
from __future__ import annotations

import moe2_common as E
from moe2_common import M


def body(run: E.Run) -> dict:
    run.total(len(E.SEEDS) + 3)
    k1 = M.k1(run)
    lr = {s: E.Reader(run, f"LR({s})", E.kind_of(s)) for s in E.SEEDS}
    k2 = E.k2(run, lr[42])
    k3 = E.k3(run)
    run.tick()
    gam = E.select_gamma(run, lr[42])
    g = gam["star"]
    run.tick()
    seeds = {}
    for s in E.SEEDS:
        ev = E.evaluate(run, lr[s], g)
        base = E.seed_base(run, s)
        seeds[str(s)] = {"order_same": E.order_same(ev), "orders": {}}
        for o in E.ORDER_NAMES:
            main, m3 = [M.public(x) for x in base[o]["main"]], [M.public(x) for x in base[o]["M3"]]
            seeds[str(s)]["orders"][o] = {"LR": [M.public(x) for x in ev[o]], "main": main, "M3": m3,
                                          "vs_main": E.vs(ev[o], main), "vs_M3": E.vs(ev[o], m3)}
        run.tick()
    g_lr = {"conditions": [], "valid": run.full}
    for s in E.NEW:
        for o in E.ORDER_NAMES:
            p = seeds[str(s)]["orders"][o]["vs_main"]["cil"]
            g_lr["conditions"].append({"seed": s, "order": o, "mean": p["mean"], "wins": p["wins"],
                                       "pass": bool(p["mean"] >= 0.007 and p["wins"] >= 7)})
    g_lr["pass"] = bool(all(c["pass"] for c in g_lr["conditions"]))
    run.tick()
    return {"K1": k1, "K2": k2, "K3": k3, "gamma": gam, "seeds": seeds, "G-LR": g_lr, "m3_vs_moe1_max_abs": E.m3_vs_moe1(run)}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e1", body))
