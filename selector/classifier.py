"""CONCH-style zero-shot classifier — the single classification head.

訓練與評估共用同一條路徑（舊 code 兩邊不一致，是 bug）：

    Z_w    = L2normalize( Σ_i w_i · z_i )          # 加權聚合已選 patch
    logits = logit_scale · (Z_w @ f_txt.T)         # [1, C]

其中 f_txt 來自 selector.text_encoder（CONCH text tower），logit_scale 亦取自
CONCH，不另設常數。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def conch_classify(Z_selected: torch.Tensor,
                   weights: Optional[torch.Tensor],
                   f_txt: torch.Tensor,
                   logit_scale: torch.Tensor | float) -> torch.Tensor:
    """CONCH text-alignment classifier.

    Z_selected:  [k, D]  已選 patch 的 CONCH embedding
    weights:     [k] 或 [k, 1]；None 表示等權（等價於平均）
    f_txt:       [C, D]  類別文字特徵（已 L2 normalize）
    logit_scale: scalar
    returns:     logits [1, C]
    """
    if Z_selected.dim() != 2:
        raise ValueError(f"Z_selected must be [k, D], got {tuple(Z_selected.shape)}")
    if weights is None:
        w = Z_selected.new_full((Z_selected.shape[0], 1), 1.0 / Z_selected.shape[0])
    else:
        w = weights.reshape(-1, 1).to(Z_selected.dtype)
        if w.shape[0] != Z_selected.shape[0]:
            raise ValueError(f"weights {tuple(weights.shape)} vs "
                             f"Z_selected {tuple(Z_selected.shape)}")
    Z_w = F.normalize((w * Z_selected).sum(0, keepdim=True), dim=-1)     # [1, D]
    return logit_scale * (Z_w @ f_txt.to(Z_w.dtype).t())                 # [1, C]


def make_predict_fn(f_txt: torch.Tensor, logit_scale: torch.Tensor | float,
                    scores: Optional[torch.Tensor] = None,
                    weighting: str = "softmax",
                    weights: Optional[torch.Tensor] = None):
    """包成觀察迴圈需要的 predict_fn(Z_subset, idx=None) -> logits。

    weighting（主線 = "softmax"，與訓練一致）
      "softmax"  權重 = softmax(選中 patch 的 selector 分數)。需要 `scores`
                 （整張 slide 的 [n] 分數向量）與呼叫端傳入的 `idx`。
                 主線選它是因為**訓練一致性**，不是因為數值比較好：等權聚合會把
                 counterfactual utility 結構性稀釋 —— 多看一個 patch 只把均值挪動
                 1/|E_t|，budget 越大訊號越平，正好壓掉本方法要量的東西。
      "uniform"  等權平均。保留為 **selection-only ablation**：只看「選得準不準」，
                 不讓分數再參與聚合。

    weights
      固定權重張量，優先於 weighting。這是讓訓練路徑與評估路徑在相同 indices 下
      位元相同的唯一安全做法 —— 在子集上重算 selector 分數不是位元穩定的
      （同一列在 batch=n 與 batch=k 下走不同 BLAS 分塊，差 ~1e-9）。
    """
    if weighting not in ("softmax", "uniform"):
        raise ValueError(f"unknown weighting: {weighting}")

    def predict_fn(Z_subset: torch.Tensor,
                   idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weights is not None:
            w = weights
        elif weighting == "uniform":
            w = None
        else:
            if scores is None or idx is None:
                raise ValueError(
                    'weighting="softmax" 需要 scores（整張 slide 的分數）與 idx；'
                    '若呼叫端拿不到分數，請明確選 weighting="uniform" 或給 weights。')
            w = F.softmax(scores.index_select(0, idx), dim=0)
        return conch_classify(Z_subset, w, f_txt, logit_scale)

    return predict_fn


def softmax_weights(scores: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """主線權重：softmax(選中 patch 的分數)。單一定義，訓練與評估共用。"""
    return F.softmax(scores.index_select(0, idx), dim=0)
