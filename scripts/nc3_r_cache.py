#!/usr/bin/env python3
"""NC-3 第五關快取：R6 patch-codebook 與 R7 patch 對角高斯（PREREG-3 操作定義 2–4）。

每折：
  keys   每任務從 train slides 取樣（每張最多 200 個 patch，Generator seed 0，依 split 順序），
         R6 = KMeans(k=8, n_init=10, random_state=0) centroid 正規化；R7 = 逐維平均與變異數
  val／test  每張 slide 只讀一次，對每個已見子集算 R6、R7 的 patch 投票（前 10% patch）

    NAVCIL_MACHINE=mac python scripts/nc3_r_cache.py --device cpu [--folds 1-10]
輸出：outputs/navcil/<machine>/cache/nc3_fold{f}_keys.pt、nc3_fold{f}_{split}_{task}.pt（+ .done）
"""
from __future__ import annotations

import argparse
import json
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
from selector.evaluate import read_slide                                  # noqa: E402

N_SAMPLE, K_CODE, VAR_FLOOR = 200, 8, 1e-6
log = P.log


def build_keys(ctx, fold) -> dict:
    tag = f"nc3_fold{fold}_keys"
    path, done = ctx.cache_dir / f"{tag}.pt", ctx.cache_dir / f"{tag}.done"
    if done.exists():
        return torch.load(path, map_location="cpu")
    from sklearn.cluster import KMeans
    code, mu, var, n_samp, t_read, t_comp = [], [], [], [], [], []
    t_all = time.perf_counter()
    for task in ctx.tasks:
        ds, shift = ctx.ds(fold, task, "train")
        g = torch.Generator().manual_seed(0)
        pool = []
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            pool.append(rec.Z[torch.randperm(rec.Z.shape[0], generator=g)[:N_SAMPLE]])
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
        X = torch.cat(pool)
        km = KMeans(n_clusters=K_CODE, n_init=10, random_state=0).fit(X.numpy())
        code.append(F.normalize(torch.from_numpy(km.cluster_centers_).float(), dim=-1))
        mu.append(X.mean(0)); var.append(X.var(0, unbiased=False).clamp_min(VAR_FLOOR))
        n_samp.append(int(X.shape[0]))
        log(f"  fold {fold} {task}: keys from {len(ds)} slides, {X.shape[0]} patches")
    keys = {"codewords": torch.stack(code), "mu": torch.stack(mu), "var": torch.stack(var),
            "n_sampled": n_samp, "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp)}
    torch.save(keys, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"nc3/keys/fold{fold}", time.perf_counter() - t_all)
    return keys


@torch.no_grad()
def votes(ctx, Z, keys) -> dict:
    n = Z.shape[0]
    k = max(1, math.ceil(0.1 * n))
    T = len(ctx.tasks)
    cos = F.normalize(Z, dim=-1) @ keys["codewords"].reshape(T * K_CODE, -1).t()   # [n, T*8]
    cos_t = cos.reshape(n, T, K_CODE).amax(-1)                                     # [n, T]
    dist = torch.stack([((Z - keys["mu"][t]) ** 2 / keys["var"][t]).sum(-1) for t in range(T)], -1)
    out = {k_: [] for k_ in ("r6_counts", "r6_msum", "r7_counts", "r7_dsum")}
    for key in ctx.subsets:
        seen = torch.tensor([int(x) for x in key.split(",")])
        m, a = cos_t[:, seen].max(-1)
        own = seen[a]
        top = torch.topk(m, k).indices
        out["r6_counts"].append(torch.bincount(own[top], minlength=T))
        out["r6_msum"].append(torch.zeros(T, dtype=torch.float64).index_add_(0, own[top], m[top].double()))
        dmin, a = dist[:, seen].min(-1)
        own = seen[a]
        top = torch.topk(dmin, k, largest=False).indices
        out["r7_counts"].append(torch.bincount(own[top], minlength=T))
        out["r7_dsum"].append(torch.zeros(T, dtype=torch.float64).index_add_(0, own[top], dmin[top].double()))
    return {k_: torch.stack(v) for k_, v in out.items()}


def cache_split(ctx, fold, task, split, keys) -> None:
    tag = f"nc3_fold{fold}_{split}_{task}"
    path, done = ctx.cache_dir / f"{tag}.pt", ctx.cache_dir / f"{tag}.done"
    if done.exists():
        return
    ref = torch.load(ctx.cache_dir / f"nc2_fold{fold}_{split}_{task}.pt", map_location="cpu")
    ds, shift = ctx.ds(fold, task, split)
    rows, t_read, t_comp = [], [], []
    t_all = time.perf_counter()
    for i in range(len(ds)):
        t0 = time.perf_counter()
        rec = read_slide(ds, shift, i)
        t1 = time.perf_counter()
        if rec.sid != ref["sids"][i]:
            raise SystemExit(f"slide 順序與 nc2 快取不同：{tag} {rec.sid}")
        rows.append(votes(ctx, rec.Z, keys))
        t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
    out = {"sids": ref["sids"], "labels": ref["labels"], "subsets": ctx.subsets,
           "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp)}
    for k in rows[0]:
        out[k] = torch.stack([r[k] for r in rows])
    torch.save(out, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"nc3/cache/{tag}", time.perf_counter() - t_all)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-3 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))
    ctx.timing_path = ctx.out / "nc3" / "timing_rcache.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    log(f"NC-3 R cache threads={torch.get_num_threads()} subsets={ctx.subsets}")
    t_run = time.perf_counter()
    for fold in P.parse_folds(args.folds):
        keys = build_keys(ctx, fold)
        for task in ctx.tasks:
            for split in ("val", "test"):
                cache_split(ctx, fold, task, split, keys)
        log(f"fold {fold} NC-3 cache 完成")
    ctx.record_time(f"run/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
