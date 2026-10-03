#!/usr/bin/env python3
"""MOE-3 f6：S-A0 的超參數敏感度（PREREG-19 F6、細則 25；不設門檻）。validation 網格與選定值在 hp.json。

  S-A0 的 (γ*, α*) 與 γ 上下各一格（α = α*）、α 上下各一格（γ = γ*）：test t = 4 的 CIL ACC 與 Ā（兩序）。
  超出網格的那一格不算（不延伸）。

    NAVCIL_MACHINE=mac python scripts/moe3_f6.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/f6.json、f6.done
"""
from __future__ import annotations

import moe3_common as T
from moe3_common import M


def neighbours(g: float, a: float) -> list[dict]:
    """[{"where", "gamma", "alpha"}]；網格外的為 None。"""
    gi, ai = T.ANC_G.index(g), T.ANC_A.index(a)

    def at(grid, i):
        return grid[i] if 0 <= i < len(grid) else None

    cells = [("選定值", g, a), ("γ 下一格", at(T.ANC_G, gi - 1), a), ("γ 上一格", at(T.ANC_G, gi + 1), a),
             ("α 下一格", g, at(T.ANC_A, ai - 1)), ("α 上一格", g, at(T.ANC_A, ai + 1))]
    return [{"where": w, "gamma": x, "alpha": y} for w, x, y in cells]


def body(run: T.Run) -> dict:
    run.total(len(run.folds) + 1)
    k1 = M.k1(run)
    hp = T.load_hp(run)
    _, g, a = T.star_rd(hp, "ANC", T.U_MAIN)
    cells = neighbours(g, a)
    live = [c for c in cells if c["gamma"] is not None and c["alpha"] is not None]
    for c in live:
        c["orders"] = {o: [] for o in T.ORDER_NAMES}
    run.tick()
    for f in run.folds:
        D = T.data(run, "test", f)
        for o in T.ORDER_NAMES:
            ar = T.ar_stages(run, f, o, D)
            for c in live:
                r = T.run_cl(run, f, o, D, T.reader_fn(run, f, o, D, T.U_MAIN, ("ANC", c["gamma"], c["alpha"])), ar)
                c["orders"][o].append({"cil": r["acc_t"][3], "abar": r["abar"]})
        run.drop()
        run.tick()
    return {"K1": k1, "star": {"gamma": g, "alpha": a}, "cells": cells}


if __name__ == "__main__":
    raise SystemExit(T.stage_main("f6", body))
