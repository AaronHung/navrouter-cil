#!/usr/bin/env python3
"""EXT-1 D2（PREREG-21 細則 24）：FINAL(seed 42) 的推論成本（fold 1 全部 test slides，CPU）與訓練成本。

推論（逐張記錄 t_read_s、t_compute_s）：讀檔 → mean_vec → AR 判任務 τ̂ → τ̂ 的 head 對全部 patch 評分 →
四輪各 16 選 64 → 等權平均、L2 正規化 → [v; 1] 的 ridge 在 τ̂ 兩類內判。AR 與 ridge 的 W（t = 4、reverse）在計時前解好。
判定與 C 的 FINAL（fold 1、reverse）逐張比對，不同就停。

訓練：head 訓練秒數讀既有 `*_train.json` 的 wall_s；ridge 累加與求解秒數在 fold 1 量測（向量取自既有快取）。

    NAVCIL_MACHINE=mac python scripts/ext1_d2.py --device cpu
輸出：outputs/navcil/<machine>/ext1/d2.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_c as CC                                                       # noqa: E402
from ext1_c import D64, M, N5, X                                          # noqa: E402
from selector.cil_ops import four_round, mean_norm                        # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402

FOLD, ORDER, SEED = 1, "reverse", 42


def stat(xs) -> dict:
    xs = [float(x) for x in xs]
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs), "sum": sum(xs), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.device != "cpu":
        raise SystemExit("D2 規定 --device cpu")
    sys.argv = sys.argv[:1]
    cx = CC.Ctx()
    r3, r4, ctx = cx.r3, cx.r4, cx.r4.ctx
    lam = ctx.lam()
    heads = [M.load_head(X.head_path(r4, SEED, FOLD, t)) for t in cx.tasks]
    W_ar, pos = cx.st.b.W(FOLD, ORDER, 4, M.G_AR)
    col = [pos.index(p) for p in range(4)]                                   # AR 欄 → config 任務位置
    W, cls = X.acc_of(r4, f"s{SEED}").W(FOLD, ORDER, 4, CC.G)
    assert cls == list(range(8))
    one = torch.ones(1, dtype=D64)

    n_patch, n_sel, t_read, t_comp, preds, labels = [], [], [], [], [], []
    with torch.no_grad():
        for task in cx.tasks:
            ds, shift = ctx.ds(FOLD, task, "test")
            for i in range(len(ds)):
                t0 = time.perf_counter()
                rec = read_slide(ds, shift, i)
                t1 = time.perf_counter()
                Z = rec.Z
                mv = mean_norm(Z)
                th = int((torch.cat([mv.to(D64), one]) @ W_ar)[col].argmax())
                idx = four_round(Z, heads[th](Z, ctx.f_task(th)), lam)
                v = mean_norm(Z, idx)
                s = torch.cat([v.to(D64), one]) @ W
                pred = 2 * th + int(not bool(s[2 * th] - s[2 * th + 1] >= 0))
                t2 = time.perf_counter()
                n_patch.append(Z.shape[0]); n_sel.append(len(idx)); t_read.append(t1 - t0); t_comp.append(t2 - t1)
                preds.append(pred); labels.append(rec.label)
    ok = [int(p == y) for p, y in zip(preds, labels)]
    ref = json.loads(cx.path("FINAL", ORDER, FOLD).read_text())["ok4"]
    n_diff = sum(x != y for x, y in zip(ok, ref))
    if len(ok) != len(ref) or n_diff:
        raise M.CheckFailed(f"D2 的逐張判定與 C 的 FINAL（fold {FOLD} {ORDER}）不同：{n_diff} 張")

    # ridge 累加與求解（fold 1；每任務：A += XᵀX、B 兩欄、解 W；AR 另計）
    tr = X.data4(r4, "train", FOLD)["per"]
    acc = []
    for o in CC.ORDERS:
        A = torch.zeros(513, 513, dtype=D64); cols = {}
        A2 = torch.zeros(513, 513, dtype=D64); cols2 = []
        for p in r4.pos(o):
            d = tr[p]
            t0 = time.perf_counter()
            Xv = N5.aug(d["v"][f"s{SEED}"])
            A += Xv.t() @ Xv
            for c in (2 * p, 2 * p + 1):
                cols[c] = Xv[d["label"] == c].sum(0)
            t1 = time.perf_counter()
            torch.linalg.solve(A + CC.G * torch.eye(513, dtype=D64), torch.stack([cols[c] for c in sorted(cols)], 1))
            t2 = time.perf_counter()
            Xm = N5.aug(d["mv"])
            A2 += Xm.t() @ Xm
            cols2.append(Xm.sum(0))
            torch.linalg.solve(A2 + M.G_AR * torch.eye(513, dtype=D64), torch.stack(cols2, 1))
            t3 = time.perf_counter()
            acc.append({"order": o, "task": cx.tasks[p], "n_train": len(d["label"]), "readout_accumulate_s": t1 - t0,
                        "readout_solve_s": t2 - t1, "ar_accumulate_and_solve_s": t3 - t2})

    # head 訓練秒數（既有紀錄）
    head = {}
    for s in (43, 44, 45, 46):
        per_task = {t: [json.loads((cx.base / "moe1" / f"i6_seed{s}" / f"fold{f}_{t}_train.json").read_text())["wall_s"]
                        for f in range(1, 11)] for t in cx.tasks}
        head[str(s)] = {t: stat(v) for t, v in per_task.items()}
    head_avg = {t: statistics.fmean(head[s][t]["mean"] for s in head) for t in cx.tasks}
    # 取向量的成本（train slide 的 head 四輪選片；MOE-2 vec 快取的逐張紀錄，含 seed 43–46 四個 head 與 v0、一次取 64）
    res = {"machine": "mac", "device": "cpu", "threads": torch.get_num_threads(), "fold": FOLD, "order": ORDER, "seed": SEED,
           "loadavg": list(os.getloadavg()), "n_slides": len(ok), "acc_slide_level": sum(ok) / len(ok),
           "pred_diff_vs_C": n_diff, "n_patch": stat(n_patch), "n_scored_by_head": stat(n_patch), "n_selected": stat(n_sel),
           "t_read_s": stat(t_read), "t_compute_s": stat(t_comp), "t_total_s": stat([x + y for x, y in zip(t_read, t_comp)]),
           "ridge": acc, "head_train_wall_s": head, "head_train_wall_s_4seed_mean_per_task": head_avg,
           "head_train_note": "seed 42 的 i6/r2/*_train.json 沒有 wall_s 欄；只有 seed 43–46（四個 seed）"}
    (cx.base / "ext1" / "d2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("n_slides", "loadavg", "n_patch", "n_selected", "t_read_s", "t_compute_s",
                                           "head_train_wall_s_4seed_mean_per_task")}, indent=1, ensure_ascii=False))
    print(json.dumps(acc[:4], indent=1))


if __name__ == "__main__":
    main()
