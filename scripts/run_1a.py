#!/usr/bin/env python3
"""1a — 重現（只推論，不訓練）。

reference/v9/skill_bank_reverse_f1.pt 的 per-task selector、cache/text 的乾淨文字頭
（8 類依任務序疊放）、one-shot top-K（預設 64）、softmax(top-K 分數) 權重；
另記等權聚合（selection-only）作對照。每張 slide 記錄讀檔秒數與計算秒數。

    NAVCIL_MACHINE=mac python scripts/run_1a.py [--device auto|cpu|mps|cuda]
        [--limit N] [--tag 1a] [--probe-train]

--limit N       只跑此關卡依任務序的前 N 張（跨任務累計），供裝置測速。
--probe-train   每張另計一步「訓練樣式」前向＋反向＋更新的秒數（用暫時複本，
                不影響推論結果），供估算 1b。

輸出：outputs/navcil/<machine>/<tag>/fold<k>/<task>.json 與 <task>.done；
已有 done 的（關卡、fold、任務）自動跳過。彙總寫到 <tag>/summary_fold<k>.json。
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.classifier import conch_classify                    # noqa: E402
from selector.device import get_device, setup_mps                 # noqa: E402
from selector.evaluate import (read_slide, score_based_indices,   # noqa: E402
                               select_and_classify, slide_dataset)
from selector.flat_selector import SelectorBank                   # noqa: E402
from selector.text_encoder import build_f_txt, load_config        # noqa: E402

BANK_PATH = REPO_ROOT / "reference" / "v9" / "skill_bank_reverse_f1.pt"


def _sync(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize()


def _probe_train_step(selector, opt, Z, label, f_txt, logit_scale, budget, device):
    """一步訓練樣式的計算（前向、top-K、softmax 權重、分類、CE、反向、更新）。"""
    _sync(device)
    t0 = time.perf_counter()
    with torch.enable_grad():
        scores = selector(Z, f_txt)
        idx = score_based_indices(scores.detach(), budget)
        w = F.softmax(scores.index_select(0, idx), dim=0)
        logits = conch_classify(Z.index_select(0, idx), w, f_txt, logit_scale)
        loss = F.cross_entropy(logits, torch.tensor([label], device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    _sync(device)
    return time.perf_counter() - t0


@torch.no_grad()
def run_task(cfg, bank, f_txt, logit_scale, task, task_pos, device, budget,
             limit, probe_train):
    selector = bank.build_selector(task_pos, device)
    probe = opt = None
    if probe_train:
        probe = copy.deepcopy(selector).train()
        opt = torch.optim.Adam(probe.parameters(), lr=1e-4)
    ds, shift = slide_dataset(cfg, task, task_pos, split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    records = []
    for i in range(n):
        t0 = time.perf_counter()
        rec = read_slide(ds, shift, i)
        t_read = time.perf_counter() - t0

        _sync(device)
        t0 = time.perf_counter()
        Z = rec.Z.to(device)
        scores = selector(Z, f_txt)
        idx = score_based_indices(scores, budget)
        pred_sm, _ = select_and_classify(Z, idx, f_txt, logit_scale,
                                         scores=scores, weighting="softmax")
        pred_eq, _ = select_and_classify(Z, idx, f_txt, logit_scale,
                                         weighting="uniform")
        _sync(device)
        t_compute = time.perf_counter() - t0

        r = {"slide_id": rec.sid, "n_patch": int(rec.Z.shape[0]), "true": rec.label,
             "pred_softmax": pred_sm, "pred_uniform": pred_eq,
             "t_read_s": round(t_read, 6), "t_compute_s": round(t_compute, 6)}
        if probe is not None:
            r["t_train_step_s"] = round(_probe_train_step(
                probe, opt, Z, rec.label, f_txt, logit_scale, budget, device), 6)
        records.append(r)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="auto|cpu|mps|cuda（預設讀 machine 檔）")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="1a")
    ap.add_argument("--probe-train", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    machine = cfg.get("machine")
    if not machine:
        raise SystemExit("請設定 NAVCIL_MACHINE")
    setup_mps()
    device = get_device(args.device or cfg.get("device", "auto"))
    torch.manual_seed(0)

    f_txt = torch.cat([build_f_txt(t, cfg, device=device).f_txt for t in cfg["tasks"]], 0)
    logit_scale = build_f_txt(cfg["tasks"][0], cfg, device=device).logit_scale
    bank = SelectorBank.load(str(BANK_PATH), map_location="cpu")
    assert f_txt.dtype == torch.float32 and f_txt.shape == (8, cfg["feat_dim"])

    out = REPO_ROOT / "outputs" / "navcil" / machine / args.tag / f"fold{cfg['fold']}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} torch={torch.__version__} budget={args.budget} "
          f"limit={args.limit} → {out}", flush=True)

    remaining = args.limit
    summary = []
    for task_pos, task in enumerate(cfg["tasks"]):
        done, res_path = out / f"{task}.done", out / f"{task}.json"
        if remaining is not None and remaining <= 0:
            break
        if done.exists():
            blob = json.loads(res_path.read_text())
            print(f"  skip {task}（done）", flush=True)
        else:
            t0 = time.perf_counter()
            recs = run_task(cfg, bank, f_txt, logit_scale, task, task_pos, device,
                            args.budget, remaining, args.probe_train)
            blob = {"task": task, "task_pos": task_pos, "fold": cfg["fold"],
                    "device": str(device), "budget": args.budget,
                    "wall_s": round(time.perf_counter() - t0, 3), "records": recs}
            res_path.write_text(json.dumps(blob, indent=1))
            done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
        recs = blob["records"]
        if remaining is not None:
            remaining -= len(recs)
        row = {"task": task, "n": len(recs),
               "correct_softmax": sum(r["pred_softmax"] == r["true"] for r in recs),
               "correct_uniform": sum(r["pred_uniform"] == r["true"] for r in recs),
               "t_read_mean_s": sum(r["t_read_s"] for r in recs) / len(recs),
               "t_compute_mean_s": sum(r["t_compute_s"] for r in recs) / len(recs),
               "n_patch_mean": sum(r["n_patch"] for r in recs) / len(recs)}
        if recs and "t_train_step_s" in recs[0]:
            row["t_train_step_mean_s"] = sum(r["t_train_step_s"] for r in recs) / len(recs)
        summary.append(row)
        print(f"  {task:10s} softmax {row['correct_softmax']}/{row['n']}  "
              f"uniform {row['correct_uniform']}/{row['n']}  "
              f"read {row['t_read_mean_s']:.4f}s  compute {row['t_compute_mean_s']:.4f}s",
              flush=True)

    mean_acc = sum(r["correct_softmax"] / r["n"] for r in summary) / len(summary)
    (out.parent / f"summary_fold{cfg['fold']}.json").write_text(json.dumps({
        "device": str(device), "torch": torch.__version__, "python": platform.python_version(),
        "budget": args.budget, "limit": args.limit, "mean_acc_softmax": mean_acc,
        "tasks": summary}, indent=1))
    print(f"  mean acc (softmax) = {mean_acc:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
