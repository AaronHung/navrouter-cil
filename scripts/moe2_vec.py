#!/usr/bin/env python3
"""MOE-2 vec：證據向量快取（PREREG-18 細則 2–4）。

  1. 檢查 5 個 seed 的 head 權重都在（缺就停，不重訓）
  2. 每折、train／validation／test：v(43–46)、v0（g = 0）、v1(42)（一次取 64），train 另重算 v(42) 供比對
     → <out>/cache/vec_{split}_fold{f}.pt（每張 slide 記錄 t_read_s、t_compute_s）
  3. 細則 4 的快取檢查（每折）

    NAVCIL_MACHINE=mac python scripts/moe2_vec.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/vec.json、vec.done
"""
from __future__ import annotations

import moe2_common as E


def body(run: E.Run) -> dict:
    heads = E.check_heads(run)
    run.total(len(run.folds) * 3)
    checks, n, t_read, t_comp = {}, {}, 0.0, 0.0
    for f in run.folds:
        for split in ("train", "val", "test"):
            E.build_vec(run, f, split)
            raw = E.vecs(run, split, f)["raw"]
            n[f"{split}_fold{f}"] = sum(len(v["sids"]) for v in raw.values())
            t_read += sum(float(v["t_read_s"].sum()) for v in raw.values())
            t_comp += sum(float(v["t_compute_s"].sum()) for v in raw.values())
            run.tick()
        checks[str(f)] = E.check_vec(run, f)
        E.log(f"fold {f} 快取檢查通過：{checks[str(f)]}")
        run._vec.clear()
    worst = {k: max(c[k] for c in checks.values()) for k in ("a_s42_train_maxabs", "b_seed_cos_maxabs", "c_one42_cos_maxabs", "d_g0_cos_maxabs")}
    return {"heads": heads, "checks": checks, "worst": worst, "n_slides": n, "t_read_s": t_read, "t_compute_s": t_comp}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("vec", body))
