"""Router：把 slide 分派到某個任務的 expert（任務預測，TP）。

命名：每任務選片器稱 expert（程式沿用 selector / SelectorBank），分派到 expert 的
模組稱 router。本檔的分數函式只讀離線快取（見 scripts/nc1_pipeline.py），不碰特徵檔。

`d` 是資料提供者，需有 `.organ["features"]`、`.proto(fold)`、`.tasks`、`.c(fold, split, task)`；
`g` 是 gather() 疊好的 slide 欄位（mean_vec、mean_cos8、four、vote_counts、vote_msum、task）。
"""
from __future__ import annotations

import torch

from .cil_eval import nav_scores, zrow
from .cil_ops import task_rows


def tp_scores(d, fold: int, g: dict, seen: list[int], variant: str) -> torch.Tensor:
    """[N, len(seen)] 每個候選任務的分數（依 seen 的順序）。"""
    if variant == "oracle":
        return (g["task"].unsqueeze(1) == torch.tensor(seen)).float()
    if variant == "text-class":
        return torch.stack([g["mean_cos8"][:, task_rows(p)].amax(-1) for p in seen], -1)
    if variant == "text-organ":
        return g["mean_vec"] @ d.organ["features"][seen].t()
    if variant == "patch-vote":
        return (g["vote_counts"][:, seen].double() * 1e6 + g["vote_msum"][:, seen].double())
    if variant == "proto":
        return g["mean_vec"] @ d.proto(fold)[seen].t()
    if variant == "nav":
        return nav_scores(g["four"], seen)
    if variant == "nav-cal":
        raw = nav_scores(g["four"], seen)
        cols = []
        for j, p in enumerate(seen):
            tr = d.c(fold, "train", d.tasks[p])["four_cos8_uni"]
            X = torch.zeros(tr.shape[0], len(d.tasks), 8)
            X[:, p] = tr
            s_tr = nav_scores(X, seen)[:, j]
            cols.append((raw[:, j] - s_tr.mean()) / s_tr.std().clamp_min(1e-12))
        return torch.stack(cols, -1)
    if variant == "fuse":
        return zrow(tp_scores(d, fold, g, seen, "text-class")) + zrow(nav_scores(g["four"], seen))
    raise ValueError(variant)


def tp_pred(d, fold, g, seen, variant) -> torch.Tensor:
    return torch.tensor(seen)[tp_scores(d, fold, g, seen, variant).argmax(-1)]
