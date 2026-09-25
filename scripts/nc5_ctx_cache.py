#!/usr/bin/env python3
"""NC-5 快取：背景向量與腫瘤半向量（PREREG-5 A、B；操作定義 1–3、6、7）。

每折、每張 slide 只讀一次，算出三種腫瘤度 u 的來源：
  z  cos(x, k_tumor) − cos(x, k_normal)（與已見集合無關）
  c  對已見任務全部類別文字的最大 cosine
  e  已見任務 L0 expert 的 one-shot 分數，slide 內 z-score 後取最大
並對每個來源、每個需要的已見子集，存 q ∈ {0.25, 0.5, 0.75} 的背景向量與腫瘤半向量。
  train  c、e 的子集 = 該任務在兩序中的前綴（key 用）
  val    c、e 只用全部 4 個任務（t = 4）
  test   c、e 用所有階段子集；另存 q = 0.5 背景集合含真實任務 L0 expert 四輪 64 張的比例、u 統計

    NAVCIL_MACHINE=mac python scripts/nc5_ctx_cache.py --device cpu [--folds 1-10]
輸出：outputs/navcil/<machine>/cache/nc5_fold{f}_{split}_{task}.pt（+ .done）
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
from selector.cil_eval import subset_key                                  # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402

QS = (0.25, 0.5, 0.75)
ALL = "0,1,2,3"
log = P.log


def vecs(Z: torch.Tensor, u: torch.Tensor):
    """(背景向量 [3, D]、腫瘤半向量 [D]、q = 0.5 的背景索引)。"""
    n = Z.shape[0]
    bg, idx05 = [], None
    for q in QS:
        idx = torch.topk(u, math.ceil(q * n), largest=False).indices
        bg.append(F.normalize(Z[idx].mean(0), dim=-1))
        if q == 0.5:
            idx05 = idx
    th = torch.topk(u, math.ceil(0.5 * n)).indices
    return torch.stack(bg), F.normalize(Z[th].mean(0), dim=-1), idx05


class Slide:
    def __init__(self, ctx, keys_tn, sels):
        self.ctx, self.kt, self.kn, self.sels = ctx, keys_tn[0], keys_tn[1], sels

    @torch.no_grad()
    def __call__(self, Z, subsets, four_idx=None) -> dict:
        ctx = self.ctx
        Zn = F.normalize(Z, dim=-1)
        uz = Zn @ self.kt - Zn @ self.kn
        C = Zn @ ctx.F.t()
        E = torch.stack([P.zscore(sel(Z, ctx.f_task(p))) for p, sel in enumerate(self.sels)])  # [4, n]
        r = {}
        bg, th, i05 = vecs(Z, uz)
        r["z_bg"], r["z_th"] = bg, th
        bg_idx = {"z": i05}
        ustat = {"z": uz}
        for src in ("c", "e"):
            bgs, ths = [], []
            for key in subsets:
                seen = [int(x) for x in key.split(",")]
                if src == "c":
                    u = C[:, [rr for p in seen for rr in task_rows(p)]].amax(-1)
                else:
                    u = E[seen].amax(0)
                b, t, i = vecs(Z, u)
                bgs.append(b); ths.append(t)
                if key == ALL:
                    bg_idx[src], ustat[src] = i, u
            r[f"{src}_bg"], r[f"{src}_th"] = torch.stack(bgs), torch.stack(ths)
        if four_idx is not None:                                   # test：背景集合比例、u 統計
            sel_set = set(four_idx[four_idx >= 0].tolist())
            r["frac_bg05"] = torch.tensor([len(sel_set & set(bg_idx[s].tolist())) / len(sel_set)
                                           for s in ("z", "c", "e")])
            r["u_stat"] = torch.tensor([[ustat[s].mean(), ustat[s].std()] for s in ("z", "c", "e")])
        return r


def run(ctx, fold, task, split, fn, subsets, ref_sids=None):
    tag = f"nc5_fold{fold}_{split}_{task}"
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
        if ref_sids is not None and rec.sid != ref_sids[i]:
            raise SystemExit(f"slide 順序與既有快取不同：{tag} {rec.sid}")
        rows.append(fn(i, rec.Z))
        t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
        sids.append(rec.sid); labels.append(rec.label)
    out = {"sids": sids, "labels": torch.tensor(labels), "subsets": subsets, "qs": QS,
           "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp)}
    for k in rows[0]:
        out[k] = torch.stack([r[k] for r in rows])
    torch.save(out, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"nc5/{tag}", time.perf_counter() - t_all)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-5 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))
    ctx.timing_path = ctx.out / "nc5" / "timing_cache.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    keys_tn = torch.load(REPO_ROOT / "cache" / "text" / "tumor_normal_keys.pt", map_location="cpu")["features"]
    log(f"NC-5 ctx cache threads={torch.get_num_threads()} subsets={ctx.subsets}")
    t_run = time.perf_counter()
    for fold in P.parse_folds(args.folds):
        bank = SelectorBank.load(str(ctx.bank_dir / f"fold{fold}.pt"))
        S = Slide(ctx, keys_tn, [bank.build_selector(p) for p in range(len(ctx.tasks))])
        for p, task in enumerate(ctx.tasks):
            pre = []
            for names in ORDERS.values():
                k = subset_key([ctx.tasks.index(x) for x in names[:names.index(task) + 1]])
                if k not in pre:
                    pre.append(k)
            tr = torch.load(ctx.cache_dir / f"fold{fold}_train_{task}.pt", map_location="cpu")["sids"]
            run(ctx, fold, task, "train", lambda i, Z, pre=pre: S(Z, pre), pre, tr)
            va = torch.load(ctx.cache_dir / f"nc2_fold{fold}_val_{task}.pt", map_location="cpu")["sids"]
            run(ctx, fold, task, "val", lambda i, Z: S(Z, [ALL]), [ALL], va)
            te = torch.load(ctx.cache_dir / f"fold{fold}_test_{task}.pt", map_location="cpu")
            run(ctx, fold, task, "test", lambda i, Z, te=te, p=p: S(Z, ctx.subsets, te["four_idx"][i, p]),
                ctx.subsets, te["sids"])
        log(f"fold {fold} NC-5 cache 完成")
    ctx.record_time(f"run/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
