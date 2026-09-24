"""NC-1 共用運算：per-task selector 訓練、四輪選片、聚合。

8 類固定依 configs/base.yaml 的 tasks 疊放（esca、rcc、brca、lung），任務 p 佔第
2p、2p+1 列；任務序（reverse / paper）只決定「第 t 階段看過哪些任務」。
"""
from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F

from .classifier import conch_classify
from .flat_selector import EvidenceSelector
from .multiround import ObserveConfig, SequentialBudgetedObserver, top_k_select

ORDERS = {
    "reverse": ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"],
    "paper": ["tcga_lung", "tcga_brca", "tcga_rcc", "tcga_esca"],
}
BUDGET = 64
STEP = 16
LAMBDAS = (0.0, 0.5, 1.0, 1.5, 2.0)


def task_rows(task_pos: int) -> list[int]:
    return [2 * task_pos, 2 * task_pos + 1]


def mean_norm(Z: torch.Tensor, idx: Optional[torch.Tensor] = None,
              w: Optional[torch.Tensor] = None) -> torch.Tensor:
    """所選 patch 的（加權）平均後 L2 正規化 → [D]。w=None 為等權。"""
    X = Z if idx is None else Z.index_select(0, idx)
    v = X.mean(0) if w is None else (w.reshape(-1, 1) * X).sum(0)
    return F.normalize(v, dim=-1)


def four_round(Z: torch.Tensor, base: torch.Tensor, lam: float,
               budget: int = BUDGET, step: int = STEP) -> torch.Tensor:
    """四輪（每輪 step 張）選片，第 2 輪起扣 λ·max cos；回傳依選取順序的 index。"""
    obs = SequentialBudgetedObserver(ObserveConfig(
        budget=budget, step_size=step, redundancy_weight=float(lam),
        normalize_base=True, redundancy_mode="maxsim"))
    return obs.observe(Z, base, lambda Zs, idx: Zs.new_zeros(1)).selected


def one_shot(base: torch.Tensor, budget: int = BUDGET) -> torch.Tensor:
    return top_k_select(base, budget)


def train_selector(slides, f_txt_task: torch.Tensor, logit_scale, *, epochs: int,
                   lr: float, weight_decay: float, seed: int, budget: int = BUDGET,
                   log=print):
    """訓練一個 EvidenceSelector（只看該任務自己的 2 類文字）。

    slides: 可重複呼叫的 callable(order) → iterator of (t_read, sid, Z, local_label)。
    每步一張 slide：分數 → top-K → softmax(top-K 分數) 加權聚合 → CE。
    回傳 (selector, per-epoch 記錄)。
    """
    torch.manual_seed(seed)
    sel = EvidenceSelector()
    opt = torch.optim.Adam(sel.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for ep in range(epochs):
        sel.train()
        tot, n, t_read, t_comp, per_slide = 0.0, 0, 0.0, 0.0, []
        g = torch.Generator().manual_seed(seed + ep)
        for tr, sid, Z, y in slides(g):
            t0 = time.perf_counter()
            s = sel(Z, f_txt_task)
            idx = top_k_select(s.detach(), budget)
            w = F.softmax(s.index_select(0, idx), dim=0)
            logits = conch_classify(Z.index_select(0, idx), w, f_txt_task, logit_scale)
            loss = F.cross_entropy(logits, torch.tensor([y]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tc = time.perf_counter() - t0
            tot += float(loss.detach()); n += 1; t_read += tr; t_comp += tc
            per_slide.append([sid, round(tr, 6), round(tc, 6)])
        rec = {"epoch": ep + 1, "n": n, "mean_loss": tot / max(n, 1),
               "t_read_s": t_read, "t_compute_s": t_comp, "per_slide": per_slide}
        history.append(rec)
        log(f"    epoch {ep + 1}/{epochs} n={n} loss={rec['mean_loss']:.4f} "
            f"read={t_read:.1f}s compute={t_comp:.1f}s")
    sel.eval()
    return sel, history
