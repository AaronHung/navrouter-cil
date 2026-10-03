#!/usr/bin/env python3
"""MOE-1 S3：訓練／推論落差 2×2（PREREG-17 S3、細則 16–17；seed 42，只推論，validation 與 test）。

選片 ∈ {一次取 s 前 64、四輪各 16（λ*，現行）} × 彙整 ∈ {等權（現行）、softmax(s[idx])}；四個 head 都算。
現行格須與 moe0 快取的 I6_cos8 相符（≤ 1e-6、2 類內 argmax 全同）並重現 K1。

    NAVCIL_MACHINE=mac python scripts/moe1_s3.py --device cpu [--folds 1-10] [--out moe1]
輸出：outputs/navcil/<machine>/<out>/s3.json、s3.done；cache/s42cells_{val,test}_fold{f}.pt（與 S5 共用）
"""
from __future__ import annotations

import torch

import moe1_common as M
from moe1_common import C


def body(run: M.Run) -> dict:
    st = run.st
    run.total(len(run.folds) * 2)
    cells, chk = {}, {"max_abs_cos_diff": 0.0, "argmax_all_equal": True}
    t_read = t_comp = 0.0
    for f in run.folds:
        for split in ("val", "test"):
            c = M.cells42(run, split, f)
            cells[(split, f)] = c["cells_cos8"]                               # [N, 4 head, 4 格, 8]
            now, ref = c["cells_cos8"][:, :, M.CELL_NOW], st.split(split, f)["I6_cos8"]
            chk["max_abs_cos_diff"] = max(chk["max_abs_cos_diff"], float((now - ref).abs().max()))
            for q in range(4):
                rows = torch.tensor([2 * q, 2 * q + 1])
                chk["argmax_all_equal"] &= bool(torch.equal(now[:, q][:, rows].argmax(-1), ref[:, q][:, rows].argmax(-1)))
            t_read += sum(float(v["t_read_s"].sum()) for v in c["per_task"].values())
            t_comp += sum(float(v["t_compute_s"].sum()) for v in c["per_task"].values())
            run.tick()
    chk["pass"] = bool(chk["max_abs_cos_diff"] <= 1e-6 and chk["argmax_all_equal"])
    if not chk["pass"]:
        raise M.CheckFailed(f"S3 現行格與 moe0 快取不符：{chk}")
    k1 = M.k1(run, i6_fn=lambda f: cells[("test", f)][:, :, M.CELL_NOW])     # 現行格（由特徵檔重算）重現 K1
    res = {}
    for split in ("val", "test"):
        res[split] = {}
        for o in M.ORDER_NAMES:
            per = [[] for _ in M.CELLS]
            for f in run.folds:
                S = st.split(split, f)
                ar = M.ar_stage(st, f, o, 4, S["mean_vec"])
                for c in range(4):
                    per[c].append(M.public(M.main_eval(cells[(split, f)][:, :, c], ar, S["task"], S["labels"])))
            res[split][o] = {"cells": per,
                             "vs_now": [{k: M.paired([x[k] for x in per[c]], [x[k] for x in per[M.CELL_NOW]]) for k in ("wp", "cil")}
                                        for c in range(4)]}
    return {"K1": k1, "check_now_cell": chk, "cell_names": M.CELLS, "now": M.CELL_NOW, "splits": res,
            "t_read_s": t_read, "t_compute_s": t_comp, "n_val": sum(len(st.split("val", f)["labels"]) for f in run.folds),
            "n_test": sum(len(st.split("test", f)["labels"]) for f in run.folds)}


if __name__ == "__main__":
    raise SystemExit(M.stage_main("s3", body))
