#!/usr/bin/env python3
"""MOE-3 hp：超參數的選擇（PREREG-19「超參數的選法」、細則 5；只用 validation）。

  每個（判讀器、輸入）組合各自選一次：RDG × {u_64, v0, v(42)（只作描述）}、ANC × {u_64, v0, v(42)}
  目標 = 兩序 × t = 1…4 的 validation 告訴任務 WP（已學任務等權）的平均，再取十折平均；同分先取較大的 γ，再取較小的 α。

    NAVCIL_MACHINE=mac python scripts/moe3_hp.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/hp.json、hp.done
"""
from __future__ import annotations

import moe3_common as T
from moe3_common import C


def body(run: T.Run) -> dict:
    run.total(len(run.folds))
    per = {T.hp_key(r, k): {T.ckey(c): [] for c in T.cands(r)} for r, k in T.COMBOS}
    for f in run.folds:
        for r, k in T.COMBOS:
            obj = T.val_objective(run, f, k, r)
            for c, v in obj.items():
                per[T.hp_key(r, k)][c].append(v)
        run.drop()
        run.tick()
    combos = {}
    for r, k in T.COMBOS:
        key = T.hp_key(r, k)
        obj = {c: C.mean(v) for c, v in per[key].items()}
        combos[key] = {"reader": r, "kind": k, "obj": obj, "obj_per_fold": per[key], "star": T.select(r, obj)}
        T.log(f"{key}：選定 {combos[key]['star']}")
    return {"grids": {"RDG_gamma": list(T.RDG_G), "ANC_gamma": list(T.ANC_G), "ANC_alpha": list(T.ANC_A)}, "combos": combos}


if __name__ == "__main__":
    raise SystemExit(T.stage_main("hp", body))
