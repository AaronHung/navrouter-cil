#!/usr/bin/env python3
"""NC-15 步驟 2：LoRA 對照版 L1(r) 在 validation 上的四輪評估（PREREG-15 操作定義 2、6）。

每折的 validation slides 各讀一次，同時算：
  L0     四個任務各自的 L0 expert（第一關 bank）四輪 8 類 cosine（與 NC-2 validation 快取逐位比對）
  L1(r)  r ∈ {1, 2, 3}、兩序：該序中「自家任務」的 expert（序中第一個任務 = 底座 L0 expert）四輪 8 類 cosine
只讀 validation；不讀任何 test 檔。

輸入（只讀）：--src-out 下的 bank/、cache/nc2_*_val_*、nc1/lambda.json、lora_v2/r{1,2}/；
            r = 3 權重：本 worktree 的 outputs/navcil/<machine>/nc15/lora_r3/。
輸出：outputs/navcil/<machine>/nc15/val/fold{f}.pt（+ .done）

    NAVCIL_MACHINE=mac python scripts/nc15_lora_val.py --device cpu --src-out <main>/outputs/navcil/mac
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
from selector.cil_ops import ORDERS, four_round, mean_norm                # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402
from selector.lora_expert import LowRankExpert                            # noqa: E402

log = P.log
RS = (1, 2, 3)


def weight_dir(src: Path, own: Path, r: int, order: str) -> Path:
    return (own / f"lora_r{r}" / order) if r == 3 else (src / "lora_v2" / f"r{r}" / order)


def load_experts(ctx, bank, src, own, fold) -> dict:
    """{(r, order): {task_pos: expert}}；序中第一個任務為底座（L0 expert）。"""
    ex = {}
    for r in RS:
        for order, tasks in ORDERS.items():
            first = ctx.tasks.index(tasks[0])
            base = bank.build_selector(first)
            m = {first: base}
            for task in tasks[1:]:
                p = ctx.tasks.index(task)
                torch.manual_seed(P.SEED)
                e = LowRankExpert(base, r)
                e.load_state_dict(torch.load(weight_dir(src, own, r, order) / f"fold{fold}_{task}.pt",
                                             map_location="cpu"))
                m[p] = e.eval()
            ex[(r, order)] = m
    return ex


@torch.no_grad()
def eval_fold(ctx, src, own, out, fold) -> None:
    done = out / f"fold{fold}.done"
    if done.exists():
        log(f"fold {fold}: val done, skip")
        return
    lam = ctx.lam()
    bank = SelectorBank.load(str(ctx.bank_dir / f"fold{fold}.pt"))
    l0 = [bank.build_selector(q) for q in range(len(ctx.tasks))]
    ex = load_experts(ctx, bank, src, own, fold)
    t_all = time.perf_counter()
    res = {}
    for p, task in enumerate(ctx.tasks):
        ref = torch.load(ctx.cache_dir / f"nc2_fold{fold}_val_{task}.pt", map_location="cpu")
        ds, shift = ctx.ds(fold, task, "val")
        if len(ds) != len(ref["sids"]):
            raise SystemExit(f"fold {fold} {task}: val 張數 {len(ds)} ≠ 快取 {len(ref['sids'])}")
        l0_c8, l0_ix, labels, sids, t_read, t_comp = [], [], [], [], [], []
        c8 = {k: [] for k in ex}
        ix = {k: [] for k in ex}
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            Z = rec.Z
            if rec.sid != ref["sids"][i]:
                raise SystemExit(f"fold {fold} {task} #{i}: slide id 與 NC-2 validation 快取不一致")
            j0 = four_round(Z, l0[p](Z, ctx.f_task(p)), lam)
            v0 = mean_norm(Z, j0) @ ctx.F.t()
            l0_c8.append(v0); l0_ix.append(P.pad(j0, -1))
            for k, m in ex.items():
                if p == ctx.tasks.index(ORDERS[k[1]][0]):
                    c8[k].append(v0); ix[k].append(P.pad(j0, -1))      # 底座 = L0 expert，結果相同
                    continue
                j = four_round(Z, m[p](Z, ctx.f_task(p)), lam)
                c8[k].append(mean_norm(Z, j) @ ctx.F.t()); ix[k].append(P.pad(j, -1))
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
            labels.append(rec.label); sids.append(rec.sid)
        labels = torch.tensor(labels)
        l0_c8 = torch.stack(l0_c8)
        # PREREG-15 操作定義 6（AMENDMENT-4）：L0 的四輪 index 逐位相同、cosine 差 ≤ 1e-6、2 類內 argmax 全同
        l0_ix = torch.stack(l0_ix).to(torch.int32)
        rc, rr = ref["four_cos8_uni"][:, p], [2 * p, 2 * p + 1]
        d = (l0_c8 - rc).abs().max().item()
        if not torch.equal(labels, ref["labels"]):
            raise SystemExit(f"fold {fold} {task}: labels 與 NC-2 validation 快取不一致")
        if not torch.equal(l0_ix, ref["four_idx"][:, p]):
            raise SystemExit(f"fold {fold} {task}: L0 四輪 patch index 與 NC-2 快取不同")
        if d > 1e-6:
            raise SystemExit(f"fold {fold} {task}: L0 validation cosine 最大差 {d:.3e} > 1e-6")
        if not torch.equal(l0_c8[:, rr].argmax(-1), rc[:, rr].argmax(-1)):
            raise SystemExit(f"fold {fold} {task}: L0 2 類內 argmax 與 NC-2 快取不同")
        res[task] = {"sids": sids, "labels": labels, "l0_cos8": l0_c8, "l0_idx": l0_ix,
                     "check_l0_cos_maxabs": d,
                     "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp),
                     **{f"r{r}": {o: {"cos8": torch.stack(c8[(r, o)]),
                                      "idx": torch.stack(ix[(r, o)]).to(torch.int32)} for o in ORDERS}
                        for r in RS}}
        log(f"fold {fold} {task}: val n={len(ds)}，L0 與 NC-2 快取一致（index 相同、cosine 最大差 {d:.1e}）")
    torch.save({"tasks": res, "fold": fold, "lambda": lam, "rs": list(RS)}, out / f"fold{fold}.pt")
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"val/fold{fold}", time.perf_counter() - t_all)
    log(f"fold {fold}: val eval {time.perf_counter() - t_all:.0f}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--src-out", required=True, help="既有產物目錄（bank/、cache/、nc1/、lora_v2/），只讀")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("NC-15 規定 --device cpu")
    src = Path(args.src_out).resolve()
    ctx = P.Ctx(torch.device("cpu"))
    ctx.bank_dir, ctx.cache_dir, ctx.nc1 = src / "bank", src / "cache", src / "nc1"
    own = ctx.out / "nc15"
    if not all((own / "lora_r3" / f"{o}.done").exists() for o in ORDERS):
        raise SystemExit("r = 3 訓練尚未完成（nc15/lora_r3/{order}.done 不存在）")
    out = own / "val"
    out.mkdir(parents=True, exist_ok=True)
    ctx.timing_path = own / "timing_val.json"
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    log(f"NC-15 val rs={list(RS)} threads={torch.get_num_threads()} λ*={ctx.lam()} src={src}")
    t_run = time.perf_counter()
    for fold in P.parse_folds(args.folds):
        eval_fold(ctx, src, own, out, fold)
    ctx.record_time(f"run/{args.folds}", time.perf_counter() - t_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
