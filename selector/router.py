"""Router：把 slide 分派到某個任務的 expert（任務預測，TP）。

命名：每任務選片器稱 expert（程式沿用 selector / SelectorBank），分派到 expert 的
模組稱 router。本檔的分數函式只讀離線快取（見 scripts/nc1_pipeline.py），不碰特徵檔。

`d` 是資料提供者，需有 `.organ["features"]`、`.proto(fold)`、`.tasks`、`.c(fold, split, task)`；
`g` 是 gather() 疊好的 slide 欄位（mean_vec、mean_cos8、four、vote_counts、vote_msum、task）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

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


# ── NC-2 R 線：R0–R5（PREREG-2）──────────────────────────────────────────────
# key 皆為 [任務數, D] 或每任務 [k, D]，只用該任務自己的 train split 建立。
R_VARIANTS = ("R0", "R1", "R2", "R3", "R4", "R5")


def key_text_proto(f_txt_all: torch.Tensor, n_tasks: int = 4) -> torch.Tensor:
    """R0：該任務 2 個類別文字平均再正規化（與 T-Hard 原規則相同）。"""
    return F.normalize(f_txt_all.reshape(n_tasks, 2, -1).mean(1), dim=-1)


def key_proto(train_mean_vecs: list[torch.Tensor]) -> torch.Tensor:
    """R2：每任務 train slides 的全部 patch 平均正規化 → 再平均正規化。"""
    return torch.stack([F.normalize(v.mean(0), dim=-1) for v in train_mean_vecs])


def key_multi_proto(train_mean_vecs: list[torch.Tensor], k: int) -> list[torch.Tensor]:
    """R3：每任務 train slides 平均向量的 k-means centroid（正規化）。"""
    from sklearn.cluster import KMeans
    out = []
    for v in train_mean_vecs:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(v.numpy())
        out.append(F.normalize(torch.from_numpy(km.cluster_centers_).float(), dim=-1))
    return out


def key_evidence_proto(train_zbar: list[torch.Tensor]) -> torch.Tensor:
    """R5：expert τ 在 τ 自己 train slides 上四輪 64 張的平均正規化 → 再平均正規化。"""
    return torch.stack([F.normalize(z.mean(0), dim=-1) for z in train_zbar])


def r_scores(variant: str, g: dict, seen: list[int], keys: dict) -> torch.Tensor:
    """[N, len(seen)] 候選任務分數（依 seen 順序）。

    g：mean_vec [N, D]、zbar [N, T, D]（各 expert 四輪平均正規化）、
       vote_counts / vote_msum [N, T]（該已見子集的 patch-vote）。
    """
    if variant == "R0":
        return g["mean_vec"] @ keys["text_proto"][seen].t()
    if variant == "R1":
        return g["vote_counts"][:, seen].double() * 1e6 + g["vote_msum"][:, seen].double()
    if variant == "R2":
        return g["mean_vec"] @ keys["proto"][seen].t()
    if variant == "R3":
        return torch.stack([(g["mean_vec"] @ keys["multi_proto"][p].t()).amax(-1)
                            for p in seen], -1)
    if variant == "R4":
        return (zrow(g["mean_vec"] @ keys["proto"][seen].t())
                + zrow(g["vote_counts"][:, seen].float()))
    if variant == "R5":
        return torch.stack([g["zbar"][:, p] @ keys["evidence_proto"][p] for p in seen], -1)
    raise ValueError(variant)
