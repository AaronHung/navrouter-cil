#!/usr/bin/env python3
"""MOE-3 f4：每類別（PREREG-19 F4、細則 23；test、告訴任務、t = 4、reverse、十折合計；不設門檻）。

  S-main、S-M3、S-LR、S-R0、S-A0：8 類各自的張數、正確張數、正確率；每任務 balanced accuracy 與四任務平均

    NAVCIL_MACHINE=mac python scripts/moe3_f4.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/f4.json、f4.done
"""
from __future__ import annotations

import moe2_e6 as E6
import moe3_common as T
from moe3_common import C, E, M

O = "reverse"
NAMES = ["S-main", "S-M3", "S-LR", "S-R0", "S-A0"]


def body(run: T.Run) -> dict:
    run.total(len(run.folds) + 1)
    k1 = M.k1(run)
    hp = T.load_hp(run)
    sysm = T.systems(hp)
    base = E.seed_base(run, 42)[O]
    preds = {"S-main": [x["_wp_pred"] for x in base["main"]], "S-M3": [x["_wp_pred"] for x in base["M3"]],
             "S-LR": [], "S-R0": [], "S-A0": []}
    labels = []
    run.tick()
    for f in run.folds:
        D = T.data(run, "test", f)
        labels.append(D["label"])
        for name in ("S-LR", "S-R0", "S-A0"):
            kind, rd = sysm[name]
            preds[name].append(E.pred_of(T.reader_fn(run, f, O, D, kind, rd)(4, D["task"]), D["task"]))
        run.drop()
        run.tick()
    return {"K1": k1, "order": O, "systems": NAMES, "class_names": C.LABEL,
            "readers": {k: {"kind": sysm[k][0], "reader": list(sysm[k][1])} for k in ("S-LR", "S-R0", "S-A0")},
            "table": {k: E6.per_class(preds[k], labels) for k in NAMES}}


if __name__ == "__main__":
    raise SystemExit(T.stage_main("f4", body))
