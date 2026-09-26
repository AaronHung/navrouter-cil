#!/usr/bin/env python3
"""NC-8 同一批重算（PREREG-8 操作定義 1–4）：每折讀一次 train、一次 test，不讀先前快取。

  train  每張 slide 的 mean_vec（全部 patch 平均正規化）與 label
  test   mean_vec；每個已見子集的 zero-shot 8 類 top-64 向量之 8 類 cosine；
         L0（4 個 expert）、L1(r=2) v2（每序 4 個 expert）、I6(r=2)（4 個 expert）的四輪 8 類 cosine

    NAVCIL_MACHINE=mac python scripts/nc8_batch.py --device cpu [--folds 1-10]
輸出：outputs/navcil/<machine>/cache/nc8_fold{f}_{split}_{task}.pt（+ .done）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc1_pipeline as P                                                  # noqa: E402
from selector.cil_ops import ORDERS, four_round, mean_norm, one_shot, task_rows  # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402
from selector.i6_expert import I6Expert                                   # noqa: E402
from selector.lora_expert import LowRankExpert                            # noqa: E402

log = P.log


def experts_for(ctx, fold):
    bank = SelectorBank.load(str(ctx.bank_dir / f"fold{fold}.pt"))
    L0 = [bank.build_selector(p) for p in range(4)]
    L1 = {}
    for o, names in ORDERS.items():
        first = ctx.tasks.index(names[0])
        ex = []
        for p, t in enumerate(ctx.tasks):
            if p == first:
                ex.append(L0[p])
            else:
                m = LowRankExpert(L0[first], 2)
                m.load_state_dict(torch.load(ctx.out / "lora_v2" / "r2" / o / f"fold{fold}_{t}.pt", map_location="cpu"))
                ex.append(m.eval())
        L1[o] = ex
    I6 = []
    for t in ctx.tasks:
        m = I6Expert(2)
        m.load_state_dict(torch.load(ctx.out / "i6" / "r2" / f"fold{fold}_{t}.pt", map_location="cpu"))
        I6.append(m.eval())
    return L0, L1, I6


@torch.no_grad()
def test_slide(ctx, Z, L0, L1, I6, lam) -> dict:
    C = F.normalize(Z, dim=-1) @ ctx.F.t()
    mv = mean_norm(Z)
    c8 = lambda s: mean_norm(Z, four_round(Z, s, lam)) @ ctx.F.t()          # noqa: E731
    zs = []
    for key in ctx.subsets:
        rows = [rr for p in (int(x) for x in key.split(",")) for rr in task_rows(p)]
        zs.append(mean_norm(Z, one_shot(C[:, rows].amax(-1))) @ ctx.F.t())
    return {"mean_vec": mv, "zs8_cos8": torch.stack(zs),
            "L0_cos8": torch.stack([c8(m(Z, ctx.f_task(p))) for p, m in enumerate(L0)]),
            "L1_cos8": torch.stack([torch.stack([c8(m(Z, ctx.f_task(p))) for p, m in enumerate(L1[o])])
                                    for o in ORDERS]),
            "I6_cos8": torch.stack([c8(m(Z, ctx.f_task(p))) for p, m in enumerate(I6)])}


def run(ctx, fold, task, split, fn):
    tag = f"nc8_fold{fold}_{split}_{task}"
    path, done = ctx.cache_dir / f"{tag}.pt", ctx.cache_dir / f"{tag}.done"
    if done.exists():
        return
    ds, shift = ctx.ds(fold, task, split)
    rows, sids, labels, t_read, t_comp = [], [], [], [], []
    t_all = time.perf_counter()
    for i in range(len(ds)):
        t0 = time.perf_counter()
        rec = read_slide(ds, shift, i)
        t1 = time.perf_counter()
        rows.append(fn(rec.Z))
        t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
        sids.append(rec.sid); labels.append(rec.label)
    out = {"sids": sids, "labels": torch.tensor(labels), "subsets": ctx.subsets, "orders": list(ORDERS),
           "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp)}
    for k in rows[0]:
        out[k] = torch.stack([r[k] for r in rows])
    torch.save(out, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"nc8/{tag}", time.perf_counter() - t_all)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-8 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))
    ctx.timing_path = ctx.out / "nc8" / "timing_batch.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    lam = ctx.lam()
    log(f"NC-8 batch threads={torch.get_num_threads()} λ*={lam}")
    t_run = time.perf_counter()
    for fold in P.parse_folds(args.folds):
        L0, L1, I6 = experts_for(ctx, fold)
        for task in ctx.tasks:
            run(ctx, fold, task, "train", lambda Z: {"mean_vec": mean_norm(Z)})
            run(ctx, fold, task, "test", lambda Z: test_slide(ctx, Z, L0, L1, I6, lam))
        log(f"fold {fold} NC-8 batch 完成")
    ctx.record_time(f"run/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
