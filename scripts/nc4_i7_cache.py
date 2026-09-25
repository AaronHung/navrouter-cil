#!/usr/bin/env python3
"""NC-4 A（I7）快取：各容量候選的四輪選片證據（PREREG-4 操作定義 1–3）。

每折、每張 slide 只讀一次：
  val   該 slide 自己的任務 τ：0z；每個 τ 不是底座的序：0b 與 L 線 v2 的 r = 1、2、4、8
  test  每個任務 τ 的 0z；每個序的 0b（底座 expert 以 τ 的文字計算）
各候選存 8 類 cosine；test 上 v2 expert 的證據沿用 lora_v2 評估檔。

    NAVCIL_MACHINE=mac python scripts/nc4_i7_cache.py --device cpu [--folds 1-10]
輸出：outputs/navcil/<machine>/cache/nc4_fold{f}_{split}_{task}.pt（+ .done）
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
from selector.cil_ops import ORDERS, four_round, mean_norm, task_rows     # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402
from selector.lora_expert import LowRankExpert                            # noqa: E402

R_LIST = (1, 2, 4, 8)
log = P.log


def load_experts(ctx, fold, bank) -> dict:
    """{(order, task_pos): {"0b": base, "r1": …}}，只含非底座任務。"""
    out = {}
    for order, names in ORDERS.items():
        first = ctx.tasks.index(names[0])
        base = bank.build_selector(first)
        for t in names[1:]:
            p = ctx.tasks.index(t)
            ex = {"0b": base}
            for r in R_LIST:
                m = LowRankExpert(base, r)
                m.load_state_dict(torch.load(ctx.out / "lora_v2" / f"r{r}" / order / f"fold{fold}_{t}.pt",
                                             map_location="cpu"))
                ex[f"r{r}"] = m.eval()
            out[(order, p)] = ex
    return out


@torch.no_grad()
def c8(ctx, Z, score, lam):
    return mean_norm(Z, four_round(Z, score, lam)) @ ctx.F.t()


def run(ctx, fold, task, split, fn):
    tag = f"nc4_fold{fold}_{split}_{task}"
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
    out = {"sids": sids, "labels": torch.tensor(labels), "t_read_s": torch.tensor(t_read),
           "t_compute_s": torch.tensor(t_comp), "lambda": ctx.lam()}
    for k in rows[0]:
        out[k] = torch.stack([r[k] for r in rows])
    torch.save(out, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"nc4/i7/{tag}", time.perf_counter() - t_all)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-4 規定 --device cpu")
    ctx = P.Ctx(torch.device("cpu"))
    ctx.timing_path = ctx.out / "nc4" / "timing_i7.json"
    ctx.timing_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    lam = ctx.lam()
    log(f"NC-4 I7 cache threads={torch.get_num_threads()} λ*={lam}")
    t_run = time.perf_counter()
    for fold in P.parse_folds(args.folds):
        bank = SelectorBank.load(str(ctx.bank_dir / f"fold{fold}.pt"))
        ex = load_experts(ctx, fold, bank)
        bases = {o: bank.build_selector(ctx.tasks.index(n[0])) for o, n in ORDERS.items()}
        for p, task in enumerate(ctx.tasks):
            f_p = ctx.f_task(p)

            def val_fn(Z, p=p, f_p=f_p):
                C = F.normalize(Z, dim=-1) @ ctx.F.t()
                r = {"0z": c8(ctx, Z, C[:, task_rows(p)].amax(-1), lam)}
                for (o, q), cands in ex.items():
                    if q == p:
                        for name, m in cands.items():
                            r[f"{name}|{o}"] = c8(ctx, Z, m(Z, f_p), lam)
                return r

            def test_fn(Z):
                C = F.normalize(Z, dim=-1) @ ctx.F.t()
                r = {"0z": torch.stack([c8(ctx, Z, C[:, task_rows(q)].amax(-1), lam)
                                        for q in range(len(ctx.tasks))])}
                for o, b in bases.items():
                    r[f"0b|{o}"] = torch.stack([c8(ctx, Z, b(Z, ctx.f_task(q)), lam)
                                                for q in range(len(ctx.tasks))])
                return r

            run(ctx, fold, task, "val", val_fn)
            run(ctx, fold, task, "test", test_fn)
        log(f"fold {fold} I7 cache 完成")
    ctx.record_time(f"run/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
