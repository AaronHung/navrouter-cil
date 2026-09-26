#!/usr/bin/env python3
"""NC-7 A：I6 expert 的訓練與評估（PREREG-7；每折每任務訓練一次，與順序無關）。

  1. 訓練：r ∈ {1, 2, 3} 依序；每（r、折、任務）用與 L 線 v2 相同的 train_selector 設定
     （seed 42、5 epochs、lr 5e-4、wd 1e-4；每步有限性檢查）。
  2. 評估：每折的 val／test slides 各讀一次，同時算三個 r：
     val   自家任務 expert 的四輪選片 8 類 cosine（選 r 用）
     test  四個任務 expert 各自的四輪 8 類 cosine（D3 用）；自家任務另存 g／s0 標準差比、
           與 zero-shot top-64、L0 四輪的 Jaccard

    NAVCIL_MACHINE=mac python scripts/nc7_i6.py --device cpu [--rs 1,2,3] [--folds 1-10]
輸出：outputs/navcil/<machine>/i6/r{r}/fold{f}_{task}.pt、eval_fold{f}_{split}.pt（+ .done）
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
from selector.cil_ops import four_round, mean_norm, one_shot, task_rows, train_selector  # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.i6_expert import I6Expert                                   # noqa: E402

log = P.log


def jaccard(a, b) -> float:
    sa, sb = set(a[a >= 0].tolist()), set(b[b >= 0].tolist())
    return len(sa & sb) / len(sa | sb)


def train_one(ctx, root, r, fold, task):
    part, done = root / f"r{r}" / f"fold{fold}_{task}.pt", root / f"r{r}" / f"fold{fold}_{task}.done"
    if done.exists():
        return
    part.parent.mkdir(parents=True, exist_ok=True)
    p = ctx.tasks.index(task)
    ds, shift = ctx.ds(fold, task, "train")
    torch.manual_seed(P.SEED)
    model = I6Expert(r)
    log(f"I6 r={r} fold {fold} {task}: train n={len(ds)} params={model.n_params()}")

    def slides(g):
        for i in torch.randperm(len(ds), generator=g).tolist():
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            yield time.perf_counter() - t0, rec.sid, rec.Z, rec.label - shift

    t0 = time.perf_counter()
    model, hist = train_selector(slides, ctx.f_task(p), ctx.ls, epochs=P.EPOCHS, lr=P.LR,
                                 weight_decay=P.WD, seed=P.SEED, log=log, model=model)
    ctx.record_time(f"train/r{r}/fold{fold}/{task}", time.perf_counter() - t0)
    torch.save(model.state_dict(), part)
    (part.parent / f"fold{fold}_{task}_train.json").write_text(json.dumps(hist))
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


def load(root, r, fold, task):
    m = I6Expert(r)
    m.load_state_dict(torch.load(root / f"r{r}" / f"fold{fold}_{task}.pt", map_location="cpu"))
    return m.eval()


@torch.no_grad()
def eval_fold(ctx, root, rs, fold):
    done = root / f"eval_fold{fold}.done"
    if done.exists():
        return
    lam = ctx.lam()
    ex = {r: [load(root, r, fold, t) for t in ctx.tasks] for r in rs}
    t_all = time.perf_counter()
    out = {}
    for split in ("val", "test"):
        res = {}
        for p, task in enumerate(ctx.tasks):
            ds, shift = ctx.ds(fold, task, split)
            ref = (torch.load(ctx.cache_dir / f"fold{fold}_test_{task}.pt", map_location="cpu")
                   if split == "test" else None)
            rows = {r: {"cos8": [], "idx": [], "ratio": [], "jz": [], "jl0": []} for r in rs}
            labels, t_read, t_comp = [], [], []
            for i in range(len(ds)):
                t0 = time.perf_counter()
                rec = read_slide(ds, shift, i)
                t1 = time.perf_counter()
                Z = rec.Z
                if ref is not None:
                    assert rec.sid == ref["sids"][i]
                    zs = one_shot((F.normalize(Z, dim=-1) @ ctx.F[task_rows(p)].t()).amax(-1))
                for r in rs:
                    experts = ex[r] if split == "test" else [ex[r][p]]
                    qs = range(len(ctx.tasks)) if split == "test" else [p]
                    c8 = []
                    for q, m in zip(qs, experts):
                        s0, g = m.parts(Z, ctx.f_task(q))
                        j = four_round(Z, s0 + g, lam)
                        c8.append(mean_norm(Z, j) @ ctx.F.t())
                        if q == p:
                            rows[r]["idx"].append(P.pad(j, -1))
                            if ref is not None:
                                rows[r]["ratio"].append(float(g.std() / s0.std()))
                                rows[r]["jz"].append(jaccard(j, zs))
                                rows[r]["jl0"].append(jaccard(j, ref["four_idx"][i, p]))
                    rows[r]["cos8"].append(torch.stack(c8))
                t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
                labels.append(rec.label)
            res[task] = {"labels": torch.tensor(labels), "t_read_s": torch.tensor(t_read),
                         "t_compute_s": torch.tensor(t_comp),
                         **{f"r{r}": {"cos8": torch.stack(v["cos8"]), "idx": torch.stack(v["idx"]).to(torch.int32),
                                      "ratio": torch.tensor(v["ratio"]), "jaccard_zs": torch.tensor(v["jz"]),
                                      "jaccard_l0": torch.tensor(v["jl0"])} for r, v in rows.items()}}
        out[split] = res
    torch.save(out, root / f"eval_fold{fold}.pt")
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"eval/fold{fold}", time.perf_counter() - t_all)
    log(f"I6 fold {fold}: eval {time.perf_counter() - t_all:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rs", default="1,2,3")
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-7 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))
    root = ctx.out / "i6"
    root.mkdir(parents=True, exist_ok=True)
    ctx.timing_path = ctx.out / "nc7" / "timing_i6.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    rs = [int(x) for x in args.rs.split(",")]
    folds = P.parse_folds(args.folds)
    log(f"I6 rs={rs} threads={torch.get_num_threads()} λ*={ctx.lam()}")
    t_run = time.perf_counter()
    for r in rs:
        t_r = time.perf_counter()
        for fold in folds:
            for task in ctx.tasks:
                train_one(ctx, root, r, fold, task)
        ctx.record_time(f"train_total/r{r}", time.perf_counter() - t_r)
        log(f"I6 r={r} 訓練完成（{time.perf_counter() - t_r:.0f}s）")
    for fold in folds:
        eval_fold(ctx, root, rs, fold)
    ctx.record_time(f"run/{args.rs}/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
