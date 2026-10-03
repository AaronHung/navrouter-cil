#!/usr/bin/env python3
"""MOE-4 f3：H3 每類別（PREREG-20 細則 26；test、告訴任務、t = 4、reverse 序、十折；不設門檻）。

  P-F、P-4、P-main：seed 42 一份、五個 seed 合計一份；P-0 一份（與 seed 無關，列十折合計）
  每系統：8 類各自的張數、正確張數、正確率；每任務 balanced accuracy 與四任務平均

    NAVCIL_MACHINE=mac python scripts/moe4_f3.py --device cpu [--folds 1-10] [--out moe4]
輸出：outputs/navcil/<machine>/<out>/f3.json、f3.done
"""
from __future__ import annotations

import moe2_e6 as E6
import moe4_common as X
from moe4_common import T


def body(run: T.Run) -> dict:
    X.require_chk(run)
    keys = [(f"P-F({s})", "P-F", s) for s in X.SEEDS] + [(f"P-4({s})", "P-4", s) for s in X.SEEDS] \
        + [(f"P-main({s})", "P-main", s) for s in X.SEEDS] + [("P-0", "P-0", None)]
    specs = [sp for sp in X.system_specs() if sp[0] in {k for k, *_ in keys}]
    run.total(len(run.folds) + 1)
    tell = {k: [] for k, *_ in keys}
    labels = []
    o = "reverse"
    for f in run.folds:
        D = X.data4(run, "test", f)
        labels.append(D["label"])
        ar = T.ar_stages(run, f, o, D)
        for key, _name, _s, kind, reader in specs:
            r = T.run_cl(run, f, o, D, X.dfn_of(run, f, o, D, kind, reader), ar)
            tell[key].append(r["_tell"])
        run.drop()
        run.tick()

    out = {}
    for name in ("P-F", "P-4", "P-main"):
        seeded = [f"{name}({s})" for s in X.SEEDS]
        out[name] = {"seed42": E6.per_class(tell[f"{name}(42)"], labels),
                     "pooled5": E6.per_class([p for k in seeded for p in tell[k]], labels * len(X.SEEDS))}
    out["P-0"] = {"pooled5": E6.per_class(tell["P-0"], labels)}
    run.tick()
    return {"folds": run.folds, "order": o, "class_names": ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"],
            "table": out, "note": "P-0 與 seed 無關，只列十折合計一份"}


if __name__ == "__main__":
    raise SystemExit(X.stage_main("f3", body))
