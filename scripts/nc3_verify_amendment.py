#!/usr/bin/env python3
"""AMENDMENT-1 驗證。

  --part i  --threads N   fold 1 reverse RCC、r = 1，完整 5 epochs（每步有限性檢查）
  --part ii               新舊寫法各訓練 50 步（fold 1 reverse RCC；r = 2，另加 r = 1），
                          參數最大絕對差；r = 2 超過 1e-5 以 exit 5 停下
  --part ref64            r = 1 以 float64 訓練同樣 50 步當參考，量新舊 float32 寫法各自與它的差

輸出：outputs/navcil/<machine>/nc3/amendment_{i_t<N>|ii}.json
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
from selector.cil_ops import NonFiniteError, train_selector               # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import SelectorBank                           # noqa: E402
from selector.lora_expert import LowRankExpert                            # noqa: E402

TOL = 1e-5


class LowRankExpertV1(LowRankExpert):
    """NC-2（commit 3a9e622）的原寫法：所有 r 都用 (u @ Aᵀ) @ Bᵀ。"""

    def lora(self, u):
        return (u @ self.A.t()) @ self.B.t()


def rcc_slides(ctx, limit=None, dtype=torch.float32):
    ds, shift = ctx.ds(1, "tcga_rcc", "train")

    def slides(g):
        order = torch.randperm(len(ds), generator=g).tolist()
        for i in (order if limit is None else order[:limit]):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            yield time.perf_counter() - t0, rec.sid, rec.Z.to(dtype), rec.label - shift
    return slides


def train(ctx, cls, r, base, epochs, limit=None):
    torch.manual_seed(P.SEED)
    model = cls(base, r)
    return train_selector(rcc_slides(ctx, limit), ctx.f_task(1), ctx.ls, epochs=epochs,
                          lr=P.LR, weight_decay=P.WD, seed=P.SEED, log=P.log, model=model)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("i", "ii", "ref64"), required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    ctx = P.Ctx(torch.device("cpu"))
    if args.threads:
        torch.set_num_threads(args.threads)          # 只有驗證 (i) 刻意改變執行緒數
    out = ctx.out / "nc3"
    out.mkdir(parents=True, exist_ok=True)
    base = SelectorBank.load(str(ctx.bank_dir / "fold1.pt")).build_selector(0)   # reverse 底座 = ESCA
    th = torch.get_num_threads()

    if args.part == "i":
        t0 = time.perf_counter()
        res = {"threads": th, "r": 1, "fold": 1, "order": "reverse", "task": "tcga_rcc"}
        try:
            model, hist = train(ctx, LowRankExpert, 1, base, P.EPOCHS)
            res.update(finite=True, epoch_loss=[h["mean_loss"] for h in hist],
                       params_finite=all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        except NonFiniteError as e:
            res.update(finite=False, error=str(e))
        res["seconds"] = round(time.perf_counter() - t0, 1)
        (out / f"amendment_i_t{th}.json").write_text(json.dumps(res, indent=1))
        P.log(json.dumps(res))
        return 0 if res["finite"] else 6

    if args.part == "ref64":
        res = {"threads": th, "steps": 50, "r": 1}
        torch.manual_seed(P.SEED)
        m64 = LowRankExpert(base, 1).double()
        ref, _ = train_selector(rcc_slides(ctx, 50, torch.float64), ctx.f_task(1).double(),
                                ctx.ls.double(), epochs=1, lr=P.LR, weight_decay=P.WD,
                                seed=P.SEED, log=P.log, model=m64)
        for name, cls in (("new_fp32", LowRankExpert), ("old_fp32", LowRankExpertV1)):
            m, _ = train(ctx, cls, 1, base, 1, limit=50)
            d = {k: float((m.state_dict()[k].double() - ref.state_dict()[k]).abs().max())
                 for k in m.state_dict()}
            res[name] = {"max_abs_diff_vs_fp64": max(d.values()), "per_param": d}
        (out / "amendment_ref64.json").write_text(json.dumps(res, indent=1))
        P.log(json.dumps(res))
        return 0

    res = {"threads": th, "steps": 50, "tol": TOL, "cases": {}}
    for r in (2, 1):
        case = {}
        try:
            new, _ = train(ctx, LowRankExpert, r, base, 1, limit=50)
            old, _ = train(ctx, LowRankExpertV1, r, base, 1, limit=50)
            diffs = {k: float((new.state_dict()[k] - old.state_dict()[k]).abs().max())
                     for k in new.state_dict()}
            case.update(max_abs_diff=max(diffs.values()), per_param=diffs,
                        pass_=max(diffs.values()) <= TOL)
        except NonFiniteError as e:
            case.update(error=str(e), pass_=False)
        res["cases"][f"r{r}"] = case
        P.log(f"r={r}: {case}")
    (out / "amendment_ii.json").write_text(json.dumps(res, indent=1))
    return 0 if res["cases"]["r2"]["pass_"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
