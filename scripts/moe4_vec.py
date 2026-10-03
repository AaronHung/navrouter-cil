#!/usr/bin/env python3
"""MOE-4 vec：新算 w(43…46)（一次取 64、head 分數、原始 Z 等權、L2）；每張 slide 讀一次，逐張記錄 t_read_s、t_compute_s（PREREG-20 細則 2–3）。

    NAVCIL_MACHINE=mac python scripts/moe4_vec.py --device cpu [--folds 1-10] [--out moe4]
輸出：outputs/navcil/<machine>/<out>/cache/w_{split}_fold{f}.pt（每（split、折）一個 .done）、vec.json、vec.done
"""
from __future__ import annotations

import torch

import moe4_common as X


def body(run: X.T.Run) -> dict:
    run.total(len(run.folds) * 3)
    align = {}
    for f in run.folds:
        for split in ("train", "val", "test"):
            X.build_w(run, f, split)
            align[f"{split}_fold{f}"] = X.check_w_align(run, f, split)
            run.tick()
        run.drop()
    # 讀檔與計算秒數的彙總（描述；AGENTS.md 可攜規則 9）
    timing = {}
    for split in ("train", "val", "test"):
        rd, cp, n = [], [], 0
        for f in run.folds:
            raw = torch.load(X.w_path(run, split, f), map_location="cpu")
            for t in run.tasks:
                rd.append(raw[t]["t_read_s"]); cp.append(raw[t]["t_compute_s"]); n += len(raw[t]["sids"])
        rd, cp = torch.cat(rd), torch.cat(cp)
        timing[split] = {"n_slides": n, "t_read_s_total": float(rd.sum()), "t_compute_s_total": float(cp.sum()),
                         "t_read_s_mean": float(rd.mean()), "t_compute_s_mean": float(cp.mean())}
    return {"align": align, "timing": timing}


if __name__ == "__main__":
    raise SystemExit(X.stage_main("vec", body))
