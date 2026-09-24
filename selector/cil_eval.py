"""NC-1 離線評估：只讀 outputs/navcil/<machine>/cache/ 的快取，不碰特徵檔。

快取欄位（每個 fold、split、task 一個檔，見 scripts/nc1_pipeline.py）都已存成
「對 8 類文字的 cosine」[.., 8]，所有分類都在這些 cosine 上取 argmax。
"""
from __future__ import annotations

import math

import torch

from .cil_ops import ORDERS, task_rows

N_RAND = 5


def masked_correct(cos8: torch.Tensor, labels: torch.Tensor, task_pos: int) -> torch.Tensor:
    rows = torch.tensor(task_rows(task_pos))
    return rows[cos8[..., rows].argmax(-1)] == labels


def seen_correct(cos8: torch.Tensor, labels: torch.Tensor, seen_pos: list[int]) -> torch.Tensor:
    rows = torch.tensor([r for p in seen_pos for r in task_rows(p)])
    return rows[cos8[..., rows].argmax(-1)] == labels


def stage1_task(c: dict, task_pos: int) -> dict:
    """單一任務 test set 的 (a)–(e) Masked ACC 與 oracle 8 類 ACC。"""
    y, p = c["labels"], task_pos
    mc = lambda x: masked_correct(x, y, p).float().mean().item()    # noqa: E731
    all8 = lambda x: (x.argmax(-1) == y).float().mean().item()       # noqa: E731
    rand = torch.stack([masked_correct(c["rand_cos8"][:, s], y, p).float().mean()
                        for s in range(N_RAND)]).mean().item()
    return {
        "a_zs_all": mc(c["mean_cos8"]),
        "b_zs_top64": mc(c["zs_task_cos8"][:, p]),
        "c_random64": rand,
        "d_oneshot": mc(c["one_cos8_uni"][:, p]),
        "e_fourround": mc(c["four_cos8_uni"][:, p]),
        "d_oneshot_softmax": mc(c["one_cos8_sm"][:, p]),
        "e_fourround_softmax": mc(c["four_cos8_sm"][:, p]),
        "d_oracle8": all8(c["one_cos8_uni"][:, p]),
        "e_oracle8": all8(c["four_cos8_uni"][:, p]),
        "d_oracle8_softmax": all8(c["one_cos8_sm"][:, p]),
        "e_oracle8_softmax": all8(c["four_cos8_sm"][:, p]),
        "n": int(y.numel()),
    }


def mean_sd(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def zrow(x: torch.Tensor) -> torch.Tensor:
    """候選任務之間 z-normalize（最後一維）；std = 0 時回 0。"""
    if x.shape[-1] < 2:
        return torch.zeros_like(x)
    sd = x.std(-1, unbiased=False, keepdim=True)
    return torch.where(sd > 0, (x - x.mean(-1, keepdim=True)) / sd.clamp_min(1e-12),
                       torch.zeros_like(x))


def nav_scores(four_cos8: torch.Tensor, seen_pos: list[int]) -> torch.Tensor:
    """four_cos8 [N, 4(τ), 8] → nav 分數 [N, len(seen)]。

    max_{c∈C_τ} cos(z̄_τ, e_c) − max_{c∈已見, c∉C_τ} cos(z̄_τ, e_c)；只有一個候選時第二項為 0。
    """
    cols = []
    for tau in seen_pos:
        own = four_cos8[:, tau][:, task_rows(tau)].amax(-1)
        other = [r for p in seen_pos if p != tau for r in task_rows(p)]
        oth = four_cos8[:, tau][:, other].amax(-1) if other else torch.zeros_like(own)
        cols.append(own - oth)
    return torch.stack(cols, -1)


def subset_key(seen_pos: list[int]) -> str:
    return ",".join(str(p) for p in sorted(seen_pos))


def all_subsets(tasks: list[str]) -> list[str]:
    keys = []
    for order in ORDERS.values():
        for t in range(1, len(order) + 1):
            k = subset_key([tasks.index(x) for x in order[:t]])
            if k not in keys:
                keys.append(k)
    return keys
