#!/usr/bin/env python3
"""NC-2 R 線快取：十折 validation／test／train slides 在各 expert 下的四輪選片（λ*）。

每張 slide 只讀一次：
  val    全部 patch 平均、各 expert 四輪選片索引、z̄（平均正規化，512 維）、8 類 cosine、
         patch-vote（各已見子集）
  test   各 expert 四輪 z̄（512 維）；選片索引必須與第一關 test 快取完全相同，否則停下（exit 4）
  train  自家 expert 四輪 z̄（evidence-proto 用）；8 類 cosine 必須與第一關 train 快取相同

    NAVCIL_MACHINE=mac python scripts/nc2_r_cache.py --device cpu [--folds 1-10]
輸出：outputs/navcil/<machine>/cache/nc2_fold{f}_{split}_{task}.pt（+ .done）
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc1_pipeline as P                                                  # noqa: E402
from selector.cil_ops import four_round, mean_norm, task_rows             # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402

log = P.log


class Mismatch(RuntimeError):
    pass


@torch.no_grad()
def val_slide(ctx, Z, sels, lam) -> dict:
    Zn = F.normalize(Z, dim=-1)
    C = Zn @ ctx.F.t()
    n = Z.shape[0]
    mv = mean_norm(Z)
    idx, zb = [], []
    for p, sel in enumerate(sels):
        i = four_round(Z, sel(Z, ctx.f_task(p)), lam)
        idx.append(P.pad(i, -1)); zb.append(mean_norm(Z, i))
    zb = torch.stack(zb)
    votes, vsum = [], []
    k = max(1, math.ceil(P.VOTE_FRAC * n))
    for key in ctx.subsets:
        seen = [int(x) for x in key.split(",")]
        rows = torch.tensor([rr for p in seen for rr in task_rows(p)])
        m, arg = C[:, rows].max(-1)
        owner = rows[arg] // 2
        top = torch.topk(m, k).indices
        votes.append(torch.bincount(owner[top], minlength=len(ctx.tasks)))
        vsum.append(torch.zeros(len(ctx.tasks)).index_add_(0, owner[top], m[top]))
    return {"mean_vec": mv, "mean_cos8": mv @ ctx.F.t(), "four_idx": torch.stack(idx).to(torch.int32),
            "four_zbar": zb, "four_cos8_uni": zb @ ctx.F.t(),
            "vote_counts": torch.stack(votes), "vote_msum": torch.stack(vsum)}


def run_split(ctx, fold, task, split, fn):
    tag = f"nc2_fold{fold}_{split}_{task}"
    path, done = ctx.cache_dir / f"{tag}.pt", ctx.cache_dir / f"{tag}.done"
    if done.exists():
        log(f"  {tag}: done, skip")
        return
    ds, shift = ctx.ds(fold, task, split)
    rows, sids, labels, t_read, t_comp = [], [], [], [], []
    t_all = time.perf_counter()
    for i in range(len(ds)):
        t0 = time.perf_counter()
        rec = read_slide(ds, shift, i)
        t1 = time.perf_counter()
        rows.append(fn(i, rec))
        t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
        sids.append(rec.sid); labels.append(rec.label)
    out = {"sids": sids, "labels": torch.tensor(labels), "t_read_s": torch.tensor(t_read),
           "t_compute_s": torch.tensor(t_comp), "subsets": ctx.subsets, "lambda": ctx.lam()}
    for k in rows[0]:
        out[k] = torch.stack([r[k] for r in rows])
    torch.save(out, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"nc2/cache/{tag}", time.perf_counter() - t_all)
    log(f"  {tag}: n={len(ds)} read={sum(t_read):.1f}s compute={sum(t_comp):.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-2 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))
    ctx.timing_path = ctx.out / "nc2" / "timing_rcache.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = {}
    lam = ctx.lam()
    log(f"R cache: λ*={lam} threads={torch.get_num_threads()}")
    t_run = time.perf_counter()
    try:
        for fold in P.parse_folds(args.folds):
            bank = SelectorBank.load(str(ctx.bank_dir / f"fold{fold}.pt"))
            sels = [bank.build_selector(p) for p in range(len(ctx.tasks))]
            for p, task in enumerate(ctx.tasks):
                run_split(ctx, fold, task, "val", lambda i, rec: val_slide(ctx, rec.Z, sels, lam))

                ref = torch.load(ctx.cache_dir / f"fold{fold}_test_{task}.pt", map_location="cpu")

                @torch.no_grad()
                def test_fn(i, rec, ref=ref):
                    if rec.sid != ref["sids"][i]:
                        raise Mismatch(f"test sid 順序不同：{rec.sid} vs {ref['sids'][i]}")
                    idx, zb = [], []
                    for q, sel in enumerate(sels):
                        j = four_round(rec.Z, sel(rec.Z, ctx.f_task(q)), lam)
                        if not torch.equal(P.pad(j, -1).to(torch.int32), ref["four_idx"][i, q]):
                            raise Mismatch(f"fold{fold} test {rec.sid} expert {q} 四輪選片與第一關不同")
                        idx.append(P.pad(j, -1)); zb.append(mean_norm(rec.Z, j))
                    return {"four_idx": torch.stack(idx).to(torch.int32), "four_zbar": torch.stack(zb)}
                run_split(ctx, fold, task, "test", test_fn)

                ref_tr = torch.load(ctx.cache_dir / f"fold{fold}_train_{task}.pt", map_location="cpu")

                @torch.no_grad()
                def train_fn(i, rec, ref=ref_tr, p=p):
                    if rec.sid != ref["sids"][i]:
                        raise Mismatch(f"train sid 順序不同：{rec.sid}")
                    j = four_round(rec.Z, sels[p](rec.Z, ctx.f_task(p)), lam)
                    zb = mean_norm(rec.Z, j)
                    if not torch.equal(zb @ ctx.F.t(), ref["four_cos8_uni"][i]):
                        raise Mismatch(f"fold{fold} train {rec.sid} 四輪 cos8 與第一關不同")
                    return {"four_idx": P.pad(j, -1).to(torch.int32), "four_zbar": zb}
                run_split(ctx, fold, task, "train", train_fn)
            log(f"fold {fold} R cache 完成")
    except Mismatch as e:
        log(f"⚠️ {e} —— 停下回報。")
        return 4
    ctx.record_time(f"nc2/run/r_cache/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
