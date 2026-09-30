#!/usr/bin/env python3
"""NC-15 步驟 1：LoRA 對照版 L1(r = 3) 的訓練（PREREG-15 操作定義 1）。

沿用 `nc2_lora.py --tag lora_v2` 的設定：直接呼叫其 `train_order_fold`、`eval_order_fold`（不改既有腳本）。
輸入（只讀）：--src-out 下的 bank/、cache/、nc1/lambda.json。
輸出：outputs/navcil/<machine>/nc15/lora_r3/{order}/（worktree 內）；每（序、折、任務）寫 .done。
eval_order_fold 附帶的 test 評估照常產生，但在 nc15/selection.json commit 之前不得讀取。

    NAVCIL_MACHINE=mac python scripts/nc15_lora_train.py --device cpu --src-out <main>/outputs/navcil/mac
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc1_pipeline as P                                                  # noqa: E402
import nc2_lora as L                                                      # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402

log = P.log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--orders", default="reverse,paper")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--src-out", required=True, help="既有產物目錄（bank/、cache/、nc1/），只讀")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-15 規定 --device cpu")
    src = Path(args.src_out).resolve()
    ctx = P.Ctx(torch.device("cpu"))                   # 設定 machine 檔的固定執行緒數（8）
    ctx.bank_dir, ctx.cache_dir, ctx.nc1 = src / "bank", src / "cache", src / "nc1"
    root = ctx.out / "nc15" / f"lora_r{args.r}"
    ctx.timing_path = ctx.out / "nc15" / f"timing_train_r{args.r}.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    log(f"NC-15 L1 r={args.r} orders={args.orders} threads={torch.get_num_threads()} "
        f"λ*={ctx.lam()} src={src} → {root}")
    t_run = time.perf_counter()
    for order in args.orders.split(","):
        out = root / order
        out.mkdir(parents=True, exist_ok=True)
        if (root / f"{order}.done").exists():
            log(f"{order} r={args.r}: done, skip")
            continue
        t_o = time.perf_counter()
        for fold in P.parse_folds(args.folds):
            bank = SelectorBank.load(str(ctx.bank_dir / f"fold{fold}.pt"))
            experts = L.train_order_fold(ctx, args.r, order, fold, bank, out)
            L.eval_order_fold(ctx, args.r, order, fold, bank, experts, out)
        if args.folds == "1-10":
            (root / f"{order}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        ctx.record_time(f"order/r{args.r}/{order}", time.perf_counter() - t_o)
        log(f"{order} r={args.r} 完成（{time.perf_counter() - t_o:.0f}s）")
    ctx.record_time(f"run/r{args.r}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
