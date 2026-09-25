#!/usr/bin/env python3
"""NC-2 L 線：L1(r) 低秩增量 expert 的訓練與評估（PREREG-2 操作定義 8–12）。

每個（序、r）：
  每折：底座 = 第一關 bank 中該序第一個任務的 expert（凍結、直接使用）；其餘任務各自從底座
  訓練 rank-r 增量（seed 42、5 epochs、lr 5e-4、wd 1e-4）。之後每張 test slide 只讀一次，算：
    - L1 各 expert 的四輪選片（λ*）與 8 類 cosine（WP、完整系統列用）
    - 自家任務：L1 與 L0 的 patch 分數 Pearson、四輪 64 張 Jaccard
    - L2：第 t 階段的合併 expert（底座 + 前 t 個任務增量）四輪選片、8 類 cosine；t = 4 時與自家
      L1 expert 選片的 Jaccard

    NAVCIL_MACHINE=mac python scripts/nc2_lora.py --r 4 [--orders reverse,paper] [--folds 1-10]
輸出：outputs/navcil/<machine>/lora/r{r}/{order}/…；每折、每（序、r）寫 .done。
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
from selector.cil_ops import ORDERS, four_round, mean_norm, task_rows, train_selector  # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402
from selector.lora_expert import LowRankExpert, merged_expert             # noqa: E402

log = P.log


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa, sb = set(a[a >= 0].tolist()), set(b[b >= 0].tolist())
    return len(sa & sb) / len(sa | sb)


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(); y = y - y.mean()
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-12))


def train_order_fold(ctx, r, order, fold, bank, out) -> dict:
    tasks = ORDERS[order]
    first = ctx.tasks.index(tasks[0])
    base = bank.build_selector(first)
    experts = {first: base}
    for task in tasks[1:]:
        p = ctx.tasks.index(task)
        part, done = out / f"fold{fold}_{task}.pt", out / f"fold{fold}_{task}.done"
        torch.manual_seed(P.SEED)
        model = LowRankExpert(base, r)
        if done.exists():
            model.load_state_dict(torch.load(part, map_location="cpu"))
        else:
            ds, shift = ctx.ds(fold, task, "train")
            log(f"{order} r={r} fold {fold} {task}: train n={len(ds)} "
                f"trainable={model.n_trainable()}")

            def slides(g, ds=ds, shift=shift):
                for i in torch.randperm(len(ds), generator=g).tolist():
                    t0 = time.perf_counter()
                    rec = read_slide(ds, shift, i)
                    yield time.perf_counter() - t0, rec.sid, rec.Z, rec.label - shift

            t0 = time.perf_counter()
            model, hist = train_selector(slides, ctx.f_task(p), ctx.ls, epochs=P.EPOCHS,
                                         lr=P.LR, weight_decay=P.WD, seed=P.SEED, log=log,
                                         model=model)
            ctx.record_time(f"train/r{r}/{order}/fold{fold}/{task}", time.perf_counter() - t0)
            torch.save(model.state_dict(), part)
            (out / f"fold{fold}_{task}_train.json").write_text(json.dumps(hist))
            done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        experts[p] = model.eval()
    return experts


@torch.no_grad()
def eval_order_fold(ctx, r, order, fold, bank, experts, out) -> None:
    done = out / f"fold{fold}_eval.done"
    if done.exists():
        log(f"{order} r={r} fold {fold}: eval done, skip")
        return
    lam = ctx.lam()
    tasks = ORDERS[order]
    pos = [ctx.tasks.index(t) for t in tasks]
    base = experts[pos[0]]
    merged = [merged_expert(base, [experts[p] for p in pos[1:t]]) for t in range(1, 5)]
    seen_rows = [[rr for p in pos[:t] for rr in task_rows(p)] for t in range(1, 5)]
    l0 = [bank.build_selector(q) for q in range(len(ctx.tasks))]
    t_all = time.perf_counter()
    res = {}
    for p, task in enumerate(ctx.tasks):
        ref = torch.load(ctx.cache_dir / f"fold{fold}_test_{task}.pt", map_location="cpu")
        ds, shift = ctx.ds(fold, task, "test")
        l1_cos8, l1_idx, prs, jl0, m_cos8, m_idx, jm, t_read, t_comp = ([] for _ in range(9))
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            Z = rec.Z
            assert rec.sid == ref["sids"][i]
            c8, ix = [], []
            s_own = None
            for q in range(len(ctx.tasks)):
                s = experts[q](Z, ctx.f_task(q))
                j = four_round(Z, s, lam)
                c8.append(mean_norm(Z, j) @ ctx.F.t()); ix.append(P.pad(j, -1))
                if q == p:
                    s_own = s
            s0 = l0[p](Z, ctx.f_task(p))
            prs.append(pearson(s_own, s0))
            jl0.append(jaccard(ix[p], ref["four_idx"][i, p]))
            mc, mi = [], []
            for t in range(4):
                if p not in pos[:t + 1]:
                    mc.append(torch.full((8,), float("nan"))); mi.append(torch.full((64,), -1))
                    continue
                s = merged[t](Z, ctx.F[seen_rows[t]])
                j = four_round(Z, s, lam)
                mc.append(mean_norm(Z, j) @ ctx.F.t()); mi.append(P.pad(j, -1))
            jm.append(jaccard(mi[3], ix[p]))
            l1_cos8.append(torch.stack(c8)); l1_idx.append(torch.stack(ix))
            m_cos8.append(torch.stack(mc)); m_idx.append(torch.stack(mi))
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
        res[task] = {"sids": ref["sids"], "labels": ref["labels"],
                     "l1_cos8": torch.stack(l1_cos8), "l1_idx": torch.stack(l1_idx).to(torch.int32),
                     "pearson_l0": torch.tensor(prs), "jaccard_l0": torch.tensor(jl0),
                     "m_cos8": torch.stack(m_cos8), "m_idx": torch.stack(m_idx).to(torch.int32),
                     "jaccard_m_l1": torch.tensor(jm),
                     "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp)}
    fro = {ctx.tasks[p]: float(experts[p].delta_W1().norm()) for p in pos[1:]}
    n_tr = {ctx.tasks[p]: experts[p].n_trainable() for p in pos[1:]}
    torch.save({"tasks": res, "fro_BA": fro, "n_trainable": n_tr, "r": r, "order": order,
                "fold": fold, "lambda": lam}, out / f"fold{fold}_eval.pt")
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"eval/r{r}/{order}/fold{fold}", time.perf_counter() - t_all)
    log(f"{order} r={r} fold {fold}: eval {time.perf_counter() - t_all:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--orders", default="reverse,paper")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="lora", help="輸出子目錄；AMENDMENT-1 後的 L 線 v2 用 lora_v2")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-2 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))                   # 設定 machine 檔的固定執行緒數
    if args.threads and args.threads != torch.get_num_threads():
        raise SystemExit(f"AMENDMENT-1：執行緒數固定為 {torch.get_num_threads()}，不得以 --threads 覆寫")
    root = ctx.out / args.tag / f"r{args.r}"
    ctx.timing_path = ctx.out / "nc2" / f"timing_{args.tag}_r{args.r}.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    log(f"L line {args.tag} r={args.r} orders={args.orders} threads={torch.get_num_threads()} λ*={ctx.lam()}")
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
            experts = train_order_fold(ctx, args.r, order, fold, bank, out)
            eval_order_fold(ctx, args.r, order, fold, bank, experts, out)
        if args.folds == "1-10":
            (root / f"{order}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        ctx.record_time(f"order/r{args.r}/{order}", time.perf_counter() - t_o)
        log(f"{order} r={args.r} 完成（{time.perf_counter() - t_o:.0f}s）")
    ctx.record_time(f"run/r{args.r}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
