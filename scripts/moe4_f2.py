#!/usr/bin/env python3
"""MOE-4 f2：H1（逐階段）、H2（t = 4 逐折相減、兩序是否逐張相同）、門檻 G-CONF、G-TRAJ、G-ONE（PREREG-20；細則 17–25）。

  P-F、P-4、P-T1、P-main：seed 42–46 各一列，兩序，t = 1…4；另列五個 seed 平均（先對 seed 平均再算十折）
  P-0：一列（兩序）
  G-CONF／G-TRAJ 的 P-4 版本只報告、不判定；seed 42 只作描述

    NAVCIL_MACHINE=mac python scripts/moe4_f2.py --device cpu [--folds 1-10] [--out moe4]
輸出：outputs/navcil/<machine>/<out>/f2.json、f2.done
"""
from __future__ import annotations

import moe4_common as X
from moe4_common import C, T


def body(run: T.Run) -> dict:
    X.require_chk(run)
    specs = X.system_specs()
    run.total(len(run.folds) + 1)
    rows, priv = {}, {}
    for key, *_ in specs:
        for o in X.ORDERS:
            rows[(key, o)], priv[(key, o)] = [], []
    for f in run.folds:
        D = X.data4(run, "test", f)
        res = X.eval_fold(run, f, D, specs)
        for (key, o), r in res.items():
            rows[(key, o)].append(X.pub_row(r))
            priv[(key, o)].append({"_tell": r["_tell"], "_cil": r["_cil"], "_d": r["_d"]})
        run.drop()
        run.tick()

    # H1
    h1 = {key: {o: X.summarize(rows[(key, o)]) for o in X.ORDERS} for key, *_ in specs}
    h1_avg = {name: {o: X.summarize_avg([rows[(f"{name}({s})", o)] for s in X.SEEDS]) for o in X.ORDERS}
              for name in ("P-F", "P-4", "P-T1", "P-main")}

    # H2：t = 4 逐折相減（WP 與 CIL ACC）、兩序是否逐張相同
    def col(key, o, k):
        return [r[k] for r in rows[(key, o)]]

    h2 = {}
    for s in X.SEEDS:
        kF = f"P-F({s})"
        same = T.order_same({o: priv[(kF, o)] for o in X.ORDERS})
        diffs = {}
        for label, other in ((f"P-F({s}) − P-main({s})", f"P-main({s})"), (f"P-F({s}) − P-4({s})", f"P-4({s})"),
                             (f"P-F({s}) − P-0", "P-0"), (f"P-F({s}) − P-T1({s})", f"P-T1({s})")):
            diffs[label] = {k: {o: X.pairs(col(kF, o, k), col(other, o, k)) for o in X.ORDERS} for k in ("wp4", "cil4")}
        h2[str(s)] = {"wp4": {o: X.mean_sd(col(kF, o, "wp4")) for o in X.ORDERS},
                      "cil4": {o: X.mean_sd(col(kF, o, "cil4")) for o in X.ORDERS},
                      "same": same, "diffs": diffs}

    # 門檻
    def gate_row(xs):
        m = C.mean(xs)
        w = sum(x > X.TOL_GATE for x in xs)
        return {"per_fold": xs, "mean": m, "wins": w}

    G = {"G-CONF": {}, "G-TRAJ": {}, "G-ONE": {}, "P4-report": {"G-CONF": {}, "G-TRAJ": {}}}
    for s in X.SEEDS:
        for o in X.ORDERS:
            tag = f"{s}|{o}"
            dc = [a - b for a, b in zip(col(f"P-F({s})", o, "cil4"), col(f"P-main({s})", o, "cil4"))]
            dt = [a - b for a, b in zip(col(f"P-F({s})", o, "abar"), col(f"P-main({s})", o, "abar"))]
            gc, gt = gate_row(dc), gate_row(dt)
            gc.update({"pass": bool(gc["mean"] >= 0.007 and gc["wins"] >= 7)})
            gt.update({"pass": bool(gt["mean"] >= -0.005)})
            G["G-CONF"][tag] = gc
            G["G-TRAJ"][tag] = gt
            pc = [a - b for a, b in zip(col(f"P-4({s})", o, "cil4"), col(f"P-main({s})", o, "cil4"))]
            pt = [a - b for a, b in zip(col(f"P-4({s})", o, "abar"), col(f"P-main({s})", o, "abar"))]
            G["P4-report"]["G-CONF"][tag] = gate_row(pc)
            G["P4-report"]["G-TRAJ"][tag] = gate_row(pt)
    for o in X.ORDERS:
        one = []
        for i in range(len(run.folds)):
            one.append(C.mean([col(f"P-F({s})", o, "cil4")[i] - col(f"P-4({s})", o, "cil4")[i] for s in X.NEW]))
        g = gate_row(one)
        g.update({"pass": bool(g["mean"] >= -0.003)})
        G["G-ONE"][o] = g
    # 通過與否只看 seed 43–46（seed 42 的列只作描述）
    pass_all = {
        "G-CONF": all(G["G-CONF"][f"{s}|{o}"]["pass"] for s in X.NEW for o in X.ORDERS),
        "G-TRAJ": all(G["G-TRAJ"][f"{s}|{o}"]["pass"] for s in X.NEW for o in X.ORDERS),
        "G-ONE": all(G["G-ONE"][o]["pass"] for o in X.ORDERS),
    }
    run.tick()
    return {"folds": run.folds, "full": run.full, "h1": h1, "h1_avg": h1_avg, "h2": h2, "gates": G,
            "gates_pass": pass_all,
            "per_fold": {f"{k[0]}|{k[1]}": {"cil4": col(k[0], k[1], "cil4"), "wp4": col(k[0], k[1], "wp4"),
                                            "abar": col(k[0], k[1], "abar")} for k in rows}}


if __name__ == "__main__":
    raise SystemExit(X.stage_main("f2", body))
