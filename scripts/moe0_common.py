"""MOE-0 共用：讀 moe0 快取、AR／LIN8 分數、分數差與正確率的小工具（PREREG-16）。

AR／LIN8 一律沿用 nc8_report.B8.W（NC-8 同一份封閉解，float64）；本檔不重寫求解。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc5_report as N5                                                   # noqa: E402
from nc8_report import B8                                                 # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402

FOLDS = list(range(1, 11))
ORDER_NAMES = list(ORDERS)                     # reverse, paper（= forward）
LABEL = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]
TASK_SHORT = ["ESCA", "RCC", "BRCA", "LUNG"]
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)
TEMPS = (0.01, 0.03, 0.1, 0.3, 1.0)
GAMMA_AR, GAMMA_LIN8 = 1e-3, 0.01


class Store:
    def __init__(self):
        self.b = B8()
        self.tasks = self.b.tasks
        self.root = self.b.out / "moe0"
        self._s, self._w = {}, {}

    def split(self, split: str, f: int) -> dict:
        """四個任務串接（config 任務序）：sids、labels、task（位置）、mv、I6 [N,4,8]，test 另有 cross、g0。"""
        k = (split, f)
        if k not in self._s:
            raw = torch.load(self.root / f"{split}_fold{f}.pt", map_location="cpu")
            out = {"sids": [], "per_task": raw}
            keys = ["labels", "mean_vec", "I6_cos8"] + (["cross_cos8", "g0_cos8"] if split == "test" else [])
            for key in keys:
                out[key] = torch.cat([raw[t][key] for t in self.tasks])
            for t in self.tasks:
                out["sids"] += raw[t]["sids"]
            out["task"] = torch.cat([torch.full((len(raw[t]["labels"]),), p) for p, t in enumerate(self.tasks)])
            self._s[k] = out
        return self._s[k]

    def ar(self, f: int, order: str, mv: torch.Tensor) -> torch.Tensor:
        """[N, 4]：AR（γ = 1e-3）任務分數，欄位依 config 任務序（esca, rcc, brca, lung）。"""
        W, pos = self.b.W(f, order, 4, GAMMA_AR)
        sc = N5.aug(mv) @ W
        return sc[:, [pos.index(p) for p in range(4)]]

    def lin8(self, f: int, order: str, mv: torch.Tensor) -> torch.Tensor:
        """[N, 8]：LIN8（γ = 0.01）logits，欄位依固定 8 類序。"""
        W, pos = self.b.W(f, order, 4, GAMMA_LIN8, lin8=True)
        lg = N5.aug(mv) @ W
        cols = [2 * pos.index(p) + c for p in range(4) for c in (0, 1)]
        return lg[:, cols]


def pair_diff(x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """x [N, 8]、p [N]（任務位置）→ 第一類 − 第二類 [N]。"""
    return x.gather(1, (2 * p).unsqueeze(1)).squeeze(1) - x.gather(1, (2 * p + 1).unsqueeze(1)).squeeze(1)


def task_mean(ok: torch.Tensor, task: torch.Tensor) -> list[float]:
    return [ok[task == p].float().mean().item() for p in range(4)]


def eq4(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs)


def sd(xs) -> float:
    xs = list(xs)
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0


def zparams(d: torch.Tensor, task: torch.Tensor) -> list[tuple[float, float]]:
    """每任務 validation 的 (平均, 樣本標準差 ddof=1)。"""
    return [(d[task == p].mean().item(), d[task == p].std().item()) for p in range(4)]


def zapply(d: torch.Tensor, p: torch.Tensor, par: list[tuple[float, float]], center: bool = True) -> torch.Tensor:
    mu = torch.tensor([par[i][0] for i in range(4)])[p] if center else 0.0
    sg = torch.tensor([par[i][1] for i in range(4)])[p]
    return (d - mu) / sg


def fused_masked_ok(da, db, task, label, pa, pb, lam, center=True):
    """告訴任務：融合分數 > 0 → 第一類。回傳逐張是否正確。"""
    f = zapply(da, task, pa, center) + lam * zapply(db, task, pb, center)
    return (f > 0) == ((label - 2 * task) == 0)


def fused_cil_ok(da_all, db_all, th, label, pa, pb, lam, center=True):
    """TP = th（AR 分派）。da_all／db_all [N,4]：每任務的兩類分數差（任務 q expert／LIN8 在 q 的兩類）。"""
    da = da_all.gather(1, th.unsqueeze(1)).squeeze(1)
    db = db_all.gather(1, th.unsqueeze(1)).squeeze(1)
    f = zapply(da, th, pa, center) + lam * zapply(db, th, pb, center)
    pred = 2 * th + (~(f > 0)).long()
    return pred == label


def all_task_diffs(I6: torch.Tensor, lin8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """I6 [N,4,8]、lin8 [N,8] → ([N,4] (a) 差、[N,4] (b) 差)：第 q 欄 = 任務 q expert／LIN8 在 q 兩類的第一 − 第二。"""
    da = torch.stack([I6[:, q, 2 * q] - I6[:, q, 2 * q + 1] for q in range(4)], 1)
    db = torch.stack([lin8[:, 2 * q] - lin8[:, 2 * q + 1] for q in range(4)], 1)
    return da, db


def main_preds(I6: torch.Tensor, ar: torch.Tensor, task: torch.Tensor):
    """主系統：th = AR argmax；pred = th 的 expert 在 th 兩類內 argmax；wp = 真實任務 expert 在真實任務兩類內 argmax；
    mk = th 的 expert 證據在真實任務兩類內 argmax（Table 1 的 Masked ACC 欄）。"""
    th = ar.argmax(-1)
    n = torch.arange(len(th))

    def inpair(z, p):   # z [N,8]
        return 2 * p + (z.gather(1, (2 * p + 1).unsqueeze(1)).squeeze(1) > z.gather(1, (2 * p).unsqueeze(1)).squeeze(1)).long()

    z_th = I6[n, th]
    z_true = I6[n, task]
    return th, inpair(z_th, th), inpair(z_true, task), inpair(z_th, task)


def soft_gate_pred(I6: torch.Tensor, ar: torch.Tensor, T: float, ls: float) -> torch.Tensor:
    """B5：π = softmax(AR/T)；q = softmax(ls × 兩類 cosine)；類別 c 的分數 = π[task(c)] × q[c]；8 類取最大。"""
    pi = torch.softmax(ar.to(torch.float64) / T, dim=-1)                                  # [N, 4]
    lg = torch.stack([I6[:, q, 2 * q:2 * q + 2] for q in range(4)], 1).to(torch.float64) * ls   # [N, 4, 2]
    sc = pi.unsqueeze(-1) * torch.softmax(lg, dim=-1)                                      # [N, 4, 2]
    return sc.reshape(len(sc), 8).argmax(-1)
