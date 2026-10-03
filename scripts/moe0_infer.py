#!/usr/bin/env python3
"""MOE-0 B 段所需的新快取（PREREG-16 操作定義 4）：只推論，不訓練，不寫既有目錄。

每折、每任務、每個 split 讀一次特徵檔（t_read_s、t_compute_s 逐張記錄）：
  val   mean_vec；四個任務 expert 各自的四輪 8 類 cosine（I6_cos8 [N, 4, 8]）
  test  mean_vec；I6_cos8 [N, 4, 8]（一致性檢查用）；
        cross_cos8 [N, 4, 8]：以任務 j（slide 的真實任務）兩類文字算 s0、u，head 換成任務 k 的，
                              其四輪 8 類 cosine（k = j 時與 I6_cos8[:, j] 相同）；
        g0_cos8 [N, 8]：g = 0（只用 s0）的四輪 8 類 cosine

    NAVCIL_MACHINE=mac python scripts/moe0_infer.py --device cpu [--folds 1-10]
輸出：outputs/navcil/<machine>/moe0/{val,test}_fold{f}.pt（+ .done）
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
from selector.cil_ops import four_round, mean_norm                        # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.i6_expert import I6Expert                                   # noqa: E402

log = P.log


def load_heads(ctx, fold):
    heads = []
    for t in ctx.tasks:
        m = I6Expert(2)
        m.load_state_dict(torch.load(ctx.out / "i6" / "r2" / f"fold{fold}_{t}.pt", map_location="cpu"))
        heads.append(m.eval())
    return heads


@torch.no_grad()
def run_split(ctx, root, heads, fold, split, lam):
    path, done = root / f"{split}_fold{fold}.pt", root / f"{split}_fold{fold}.done"
    if done.exists():
        return
    t_all = time.perf_counter()
    out = {}
    for j, task in enumerate(ctx.tasks):
        ds, shift = ctx.ds(fold, task, split)
        rows = {"mean_vec": [], "I6_cos8": [], "cross_cos8": [], "g0_cos8": []}
        sids, labels, t_read, t_comp = [], [], [], []
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            Z = rec.Z
            c8 = lambda s: mean_norm(Z, four_round(Z, s, lam)) @ ctx.F.t()          # noqa: E731
            rows["mean_vec"].append(mean_norm(Z))
            rows["I6_cos8"].append(torch.stack([c8(m(Z, ctx.f_task(p))) for p, m in enumerate(heads)]))
            if split == "test":
                s0j = None
                cross = []
                for k, m in enumerate(heads):
                    s0, g = m.parts(Z, ctx.f_task(j))
                    s0j = s0
                    cross.append(c8(s0 + g))
                rows["cross_cos8"].append(torch.stack(cross))
                rows["g0_cos8"].append(c8(s0j))
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
            sids.append(rec.sid); labels.append(rec.label)
        rec_out = {"sids": sids, "labels": torch.tensor(labels), "t_read_s": torch.tensor(t_read),
                   "t_compute_s": torch.tensor(t_comp)}
        for k, v in rows.items():
            if v:
                rec_out[k] = torch.stack(v)
        out[task] = rec_out
    torch.save(out, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"moe0/{split}_fold{fold}", time.perf_counter() - t_all)
    log(f"fold {fold} {split}: n={sum(len(v['sids']) for v in out.values())} {time.perf_counter() - t_all:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("MOE-0 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))
    root = ctx.out / "moe0"
    root.mkdir(parents=True, exist_ok=True)
    ctx.timing_path = root / "timing.json"
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    lam = ctx.lam()
    log(f"MOE-0 infer threads={torch.get_num_threads()} λ*={lam} splits={args.splits}")
    t_run = time.perf_counter()
    for fold in P.parse_folds(args.folds):
        heads = load_heads(ctx, fold)
        for split in args.splits.split(","):
            run_split(ctx, root, heads, fold, split, lam)
    ctx.record_time(f"run/{args.splits}/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
