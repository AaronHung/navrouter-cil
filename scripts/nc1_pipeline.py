#!/usr/bin/env python3
"""NC-1 主流程（Mac CPU）：第一關訓練 + 各關所需的快取。

每折：
  1. train   每任務訓練一個新的 EvidenceSelector → bank/fold{f}.pt（兩序共用）
  2. (fold 1) val 快取（λ 網格）→ 選 λ*（nc1/lambda.json），之後十折固定
  3. test 快取：每張 test slide 只讀一次，算出所有導覽器的選片索引、分數，以及
     第一～三關需要的全部 8 類 cosine（見 process_eval_slide）
  4. train 快取：每張 train slide 的全部 patch 平均（proto）與自家導覽器四輪 λ* 的
     8 類 cosine（nav-cal 的校準統計）
  5. (fold 1) 除錯檢查：印出 (a)–(e)；(d) 或 (e) 任一任務比 (b) 低超過 0.10 就停（exit 3）

    NAVCIL_MACHINE=mac python scripts/nc1_pipeline.py --device cpu [--folds 1-10]

每個（關卡、fold、任務）完成即寫 .done，重跑自動跳過。
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

from selector.cil_eval import all_subsets, stage1_task                   # noqa: E402
from selector.cil_ops import (BUDGET, LAMBDAS, four_round, mean_norm,    # noqa: E402
                              one_shot, task_rows, train_selector)
from selector.device import get_device                                   # noqa: E402
from selector.evaluate import read_slide, slide_dataset                  # noqa: E402
from selector.flat_selector import SelectorBank                          # noqa: E402
from selector.text_encoder import build_f_txt, load_config               # noqa: E402

SEED, EPOCHS, LR, WD = 42, 5, 5e-4, 1e-4
N_RAND = 5
VOTE_FRAC = 0.10


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_folds(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


class Ctx:
    def __init__(self, device):
        self.cfg = load_config()
        if self.cfg.get("threads"):                    # AMENDMENT-1：固定執行緒數
            torch.set_num_threads(int(self.cfg["threads"]))
        self.machine = self.cfg["machine"]
        self.tasks = list(self.cfg["tasks"])
        self.device = device
        self.F = torch.cat([build_f_txt(t, self.cfg).f_txt for t in self.tasks], 0)
        self.ls = build_f_txt(self.tasks[0], self.cfg).logit_scale
        self.out = REPO_ROOT / "outputs" / "navcil" / self.machine
        self.bank_dir = self.out / "bank"
        self.cache_dir = self.out / "cache"
        self.nc1 = self.out / "nc1"
        for d in (self.bank_dir, self.cache_dir, self.nc1):
            d.mkdir(parents=True, exist_ok=True)
        self.subsets = all_subsets(self.tasks)
        self.timing_path = self.nc1 / "timing.json"
        self.timing = (json.loads(self.timing_path.read_text())
                       if self.timing_path.exists() else {})

    def f_task(self, p: int) -> torch.Tensor:
        return self.F[task_rows(p)]

    def ds(self, fold: int, task: str, split: str):
        cfg = dict(self.cfg, fold=fold)
        return slide_dataset(cfg, task, self.tasks.index(task), split)

    def record_time(self, key: str, seconds: float) -> None:
        self.timing[key] = round(seconds, 2)
        self.timing_path.write_text(json.dumps(self.timing, indent=1))

    def lam(self) -> float:
        return json.loads((self.nc1 / "lambda.json").read_text())["lambda_star"]


def cos8(ctx, v: torch.Tensor) -> torch.Tensor:
    return v @ ctx.F.t()


def zscore(s: torch.Tensor) -> torch.Tensor:
    return (s - s.mean()) / (s.std() + 1e-6)


def pad(x: torch.Tensor, fill) -> torch.Tensor:
    out = torch.full((BUDGET,), fill, dtype=x.dtype)
    out[:x.numel()] = x
    return out


# ── 1. 訓練 ──────────────────────────────────────────────────────────────────
def train_fold(ctx, fold: int) -> SelectorBank:
    bank_path = ctx.bank_dir / f"fold{fold}.pt"
    if bank_path.exists():
        log(f"fold {fold}: bank exists, skip training")
        return SelectorBank.load(str(bank_path))
    bank = SelectorBank()
    t_fold = time.perf_counter()
    for p, task in enumerate(ctx.tasks):
        part = ctx.bank_dir / f"fold{fold}_{task}.pt"
        done = ctx.bank_dir / f"fold{fold}_{task}.done"
        if done.exists():
            bank.add_skill(p, torch.load(part, map_location="cpu"))
            log(f"fold {fold} {task}: done, skip")
            continue
        ds, shift = ctx.ds(fold, task, "train")
        log(f"fold {fold} {task}: train n={len(ds)}")

        def slides(g, ds=ds, shift=shift):
            for i in torch.randperm(len(ds), generator=g).tolist():
                t0 = time.perf_counter()
                rec = read_slide(ds, shift, i)
                yield time.perf_counter() - t0, rec.sid, rec.Z, rec.label - shift

        t0 = time.perf_counter()
        sel, hist = train_selector(slides, ctx.f_task(p), ctx.ls, epochs=EPOCHS, lr=LR,
                                   weight_decay=WD, seed=SEED, log=log)
        ctx.record_time(f"train/fold{fold}/{task}", time.perf_counter() - t0)
        torch.save(sel.state_dict(), part)
        (ctx.nc1 / "train_logs").mkdir(exist_ok=True)
        (ctx.nc1 / "train_logs" / f"fold{fold}_{task}.json").write_text(json.dumps(hist))
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        bank.add_skill(p, sel)
    bank.save(str(bank_path))
    ctx.record_time(f"train/fold{fold}/total", time.perf_counter() - t_fold)
    return bank


# ── 2/3. 評估用快取 ─────────────────────────────────────────────────────────
@torch.no_grad()
def process_eval_slide(ctx, Z, sels, lam, lam_grid=None, full=True) -> dict:
    """一張 slide 的所有離線所需量。full=False（val）只算平均與導覽器四輪。"""
    Zn = F.normalize(Z, dim=-1)
    C = Zn @ ctx.F.t()                                       # [n, 8]
    n = Z.shape[0]
    mv = mean_norm(Z)
    r = {"mean_vec": mv, "mean_cos8": cos8(ctx, mv), "n_patch": n}
    scores = [sel(Z, ctx.f_task(p)) for p, sel in enumerate(sels)]
    if lam_grid is not None:                                  # val：λ 網格
        r["four_grid_cos8"] = torch.stack([torch.stack([
            cos8(ctx, mean_norm(Z, four_round(Z, s, l))) for s in scores])
            for l in lam_grid])                               # [L, 4, 8]
    if not full:                                              # val：T* 取網格中 λ* 那一格
        return r

    # (b) zero-shot top-64（分數 = 對任務 τ 兩類文字的最大 cosine）
    r["zs_task_cos8"] = torch.stack([
        cos8(ctx, mean_norm(Z, one_shot(C[:, task_rows(p)].amax(-1))))
        for p in range(len(ctx.tasks))])
    # (c) random-64，5 seeds
    r["rand_cos8"] = torch.stack([
        cos8(ctx, mean_norm(Z, torch.randperm(n, generator=torch.Generator().manual_seed(s))[:BUDGET]))
        for s in range(N_RAND)])
    # (d)(e) 各導覽器：一次 top-64 與四輪 λ*，等權與 softmax 聚合
    for name, fn in (("one", lambda s: one_shot(s)), ("four", lambda s: four_round(Z, s, lam))):
        idx_l, sc_l, uni, sm = [], [], [], []
        for s in scores:
            idx = fn(s)
            w = F.softmax(s.index_select(0, idx), dim=0)
            idx_l.append(pad(idx, -1)); sc_l.append(pad(s.index_select(0, idx), float("nan")))
            uni.append(cos8(ctx, mean_norm(Z, idx)))
            sm.append(cos8(ctx, mean_norm(Z, idx, w)))
        r[f"{name}_idx"] = torch.stack(idx_l).to(torch.int32)
        r[f"{name}_scores"] = torch.stack(sc_l)
        r[f"{name}_cos8_uni"] = torch.stack(uni)
        r[f"{name}_cos8_sm"] = torch.stack(sm)
    # 已見任務子集：zero-shot 8 類 top-64、patch-vote、E-max
    zs8, votes, vsum, emax = [], [], [], []
    zs = [zscore(s) for s in scores]
    k = max(1, math.ceil(VOTE_FRAC * n))
    for key in ctx.subsets:
        seen = [int(x) for x in key.split(",")]
        rows = torch.tensor([rr for p in seen for rr in task_rows(p)])
        m, arg = C[:, rows].max(-1)
        zs8.append(cos8(ctx, mean_norm(Z, one_shot(m))))
        owner = rows[arg] // 2
        top = torch.topk(m, k).indices
        votes.append(torch.bincount(owner[top], minlength=len(ctx.tasks)))
        vsum.append(torch.zeros(len(ctx.tasks)).index_add_(0, owner[top], m[top]))
        comb = torch.stack([zs[p] for p in seen]).amax(0)
        emax.append(cos8(ctx, mean_norm(Z, four_round(Z, comb, lam))))
    r["zs8_top64_cos8"] = torch.stack(zs8)
    r["vote_counts"] = torch.stack(votes)
    r["vote_msum"] = torch.stack(vsum)
    r["emax_cos8"] = torch.stack(emax)
    return r


def cache_split(ctx, fold, task, split, sels, lam, lam_grid=None, full=True) -> None:
    tag = f"fold{fold}_{split}_{task}"
    path, done = ctx.cache_dir / f"{tag}.pt", ctx.cache_dir / f"{tag}.done"
    if done.exists():
        log(f"  cache {tag}: done, skip")
        return
    ds, shift = ctx.ds(fold, task, split)
    rows, sids, labels, t_read, t_comp = [], [], [], [], []
    t_all = time.perf_counter()
    for i in range(len(ds)):
        t0 = time.perf_counter()
        rec = read_slide(ds, shift, i)
        t1 = time.perf_counter()
        rows.append(process_eval_slide(ctx, rec.Z, sels, lam, lam_grid, full))
        t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
        sids.append(rec.sid); labels.append(rec.label)
    out = {"sids": sids, "labels": torch.tensor(labels),
           "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp),
           "subsets": ctx.subsets, "lambda": lam}
    for k in rows[0]:
        v = [r[k] for r in rows]
        out[k] = torch.tensor(v) if isinstance(v[0], int) else torch.stack(v)
    torch.save(out, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"cache/{tag}", time.perf_counter() - t_all)
    log(f"  cache {tag}: n={len(ds)} read={sum(t_read):.1f}s compute={sum(t_comp):.1f}s")


@torch.no_grad()
def cache_train(ctx, fold, task, sel, lam) -> None:
    """train slides：全部 patch 平均（proto）＋ 自家導覽器四輪 λ* 的 8 類 cosine（nav-cal）。"""
    tag = f"fold{fold}_train_{task}"
    path, done = ctx.cache_dir / f"{tag}.pt", ctx.cache_dir / f"{tag}.done"
    if done.exists():
        log(f"  cache {tag}: done, skip")
        return
    p = ctx.tasks.index(task)
    ds, shift = ctx.ds(fold, task, "train")
    mv, c8, labels, t_read, t_comp, sids = [], [], [], [], [], []
    t_all = time.perf_counter()
    for i in range(len(ds)):
        t0 = time.perf_counter()
        rec = read_slide(ds, shift, i)
        t1 = time.perf_counter()
        s = sel(rec.Z, ctx.f_task(p))
        mv.append(mean_norm(rec.Z))
        c8.append(cos8(ctx, mean_norm(rec.Z, four_round(rec.Z, s, lam))))
        t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
        labels.append(rec.label); sids.append(rec.sid)
    torch.save({"sids": sids, "labels": torch.tensor(labels), "mean_vec": torch.stack(mv),
                "four_cos8_uni": torch.stack(c8), "t_read_s": torch.tensor(t_read),
                "t_compute_s": torch.tensor(t_comp), "lambda": lam}, path)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"cache/{tag}", time.perf_counter() - t_all)
    log(f"  cache {tag}: n={len(ds)} read={sum(t_read):.1f}s compute={sum(t_comp):.1f}s")


def load_cache(ctx, fold, split, task) -> dict:
    return torch.load(ctx.cache_dir / f"fold{fold}_{split}_{task}.pt", map_location="cpu")


def choose_lambda(ctx) -> float:
    path = ctx.nc1 / "lambda.json"
    if path.exists():
        return ctx.lam()
    from selector.cil_eval import masked_correct
    table = {}
    for li, l in enumerate(LAMBDAS):
        accs = {}
        for p, task in enumerate(ctx.tasks):
            c = load_cache(ctx, 1, "val", task)
            accs[task] = masked_correct(c["four_grid_cos8"][:, li, p], c["labels"], p
                                        ).float().mean().item()
        table[str(l)] = {"per_task": accs, "mean": sum(accs.values()) / len(accs)}
    best = max(table[str(l)]["mean"] for l in LAMBDAS)
    lam = min(l for l in LAMBDAS if table[str(l)]["mean"] == best)
    path.write_text(json.dumps({"lambda_star": lam, "fold": 1, "split": "val",
                                "grid": table}, indent=1))
    for l in LAMBDAS:
        log(f"  λ={l}: val mean masked ACC {table[str(l)]['mean']:.4f}  "
            + " ".join(f"{t[5:]}={a:.4f}" for t, a in table[str(l)]["per_task"].items()))
    log(f"  λ* = {lam}")
    return lam


def debug_check(ctx) -> bool:
    ok = True
    log("fold 1 除錯檢查（test，Masked ACC）")
    keys = ["a_zs_all", "b_zs_top64", "c_random64", "d_oneshot", "e_fourround",
            "d_oneshot_softmax", "e_fourround_softmax"]
    log("  task       " + " ".join(f"{k[:12]:>12s}" for k in keys))
    for p, task in enumerate(ctx.tasks):
        r = stage1_task(load_cache(ctx, 1, "test", task), p)
        log(f"  {task:10s} " + " ".join(f"{r[k]:12.4f}" for k in keys))
        for k in ("d_oneshot", "e_fourround"):
            if r[k] < r["b_zs_top64"] - 0.10:
                log(f"  ⚠️ {task} {k}={r[k]:.4f} 比 (b)={r['b_zs_top64']:.4f} 低超過 0.10")
                ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    device = get_device(args.device)
    if device.type != "cpu":
        raise SystemExit("NC-1 規定 --device cpu")
    torch.set_grad_enabled(True)
    ctx = Ctx(device)
    log(f"machine={ctx.machine} device={device} torch={torch.__version__} "
        f"threads={torch.get_num_threads()} subsets={ctx.subsets}")
    t_run = time.perf_counter()
    for fold in parse_folds(args.folds):
        t_f = time.perf_counter()
        bank = train_fold(ctx, fold)
        sels = [bank.build_selector(p) for p in range(len(ctx.tasks))]
        if fold == 1:
            for task in ctx.tasks:
                cache_split(ctx, 1, task, "val", sels, None, lam_grid=LAMBDAS, full=False)
            lam = choose_lambda(ctx)
        lam = ctx.lam()
        for task in ctx.tasks:
            cache_split(ctx, fold, task, "test", sels, lam)
        if fold == 1 and not debug_check(ctx):
            log("除錯檢查未通過 —— 停下回報。")
            return 3
        for p, task in enumerate(ctx.tasks):
            cache_train(ctx, fold, task, sels[p], lam)
        ctx.record_time(f"fold{fold}/total", time.perf_counter() - t_f)
        log(f"fold {fold} 完成（{time.perf_counter() - t_f:.0f}s）")
    ctx.record_time(f"run/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
