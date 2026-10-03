#!/usr/bin/env python3
"""MOE-3 vec：u_K 向量快取（PREREG-19 細則 2）。

  每折、train／validation／test：u_K（g = 0、依 s0 一次取前 K、等權平均、L2 正規化），K ∈ {16, 32, 64, 128, 256}
  → <out>/cache/u_{split}_fold{f}.pt（每張 slide 讀一次；記錄 t_read_s、t_compute_s）
  對齊檢查：slide id／標籤與既有快取逐張相同；向量有限、範數與 1 的差 ≤ 1e-5

    NAVCIL_MACHINE=mac python scripts/moe3_vec.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/vec.json、vec.done
"""
from __future__ import annotations

import torch

import moe3_common as T


def body(run: T.Run) -> dict:
    run.total(len(run.folds) * 3)
    n, small, t_read, t_comp, norm_dev = {}, {k: 0 for k in T.KS}, 0.0, 0.0, 0.0
    for f in run.folds:
        for split in ("train", "val", "test"):
            T.build_u(run, f, split)
            raw = T.data(run, split, f)["raw"]                    # 讀取時比對 slide id／標籤
            n[f"{split}_fold{f}"] = sum(len(v["sids"]) for v in raw.values())
            for v in raw.values():
                t_read += float(v["t_read_s"].sum()); t_comp += float(v["t_compute_s"].sum())
                for k in T.KS:
                    small[k] += int((v["n_patch"] < k).sum())
                    u = v["u"][k]
                    if not bool(torch.isfinite(u).all()):
                        raise T.CheckFailed(f"fold {f} {split}: u_{k} 有非有限值")
                    norm_dev = max(norm_dev, float((u.norm(dim=-1) - 1).abs().max()))
            run.tick()
        run.drop()
    if norm_dev > 1e-5:
        raise T.CheckFailed(f"u 向量的範數與 1 的最大差 {norm_dev:.2e} > 1e-5")
    T.log(f"vec：範數最大差 {norm_dev:.2e}；patch 數 < K 的張數 {small}")
    return {"n_slides": n, "n_patch_lt_K": {str(k): v for k, v in small.items()}, "norm_max_dev": norm_dev,
            "t_read_s": t_read, "t_compute_s": t_comp, "sid_label_aligned": True}


if __name__ == "__main__":
    raise SystemExit(T.stage_main("vec", body))
