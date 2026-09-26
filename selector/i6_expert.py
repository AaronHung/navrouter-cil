"""I6 expert：以 zero-shot 分數為底座的低秩 expert（PREREG-7 A；操作定義 1–3）。

    score(x) = s0(x) + g(u)
    s0 = text_nav_feats(Z, f_task) 第 1 維（對該任務 2 類文字的最大 cosine），slide 內 z-score；不訓練
    u  = [Z ; text_nav_feats(Z, f_task)]（514 維）
    g  = w2ᵀ GELU(A u + b1) + b2；A kaiming_uniform(a=√5)，b1、w2、b2 為 0 → 訓練開始時 score = s0

參數 516r + 1。避開 BLAS gemv（AMENDMENT-1）：r = 1 的 A u 與所有 r 的 w2ᵀ(·) 都用逐元素乘法加總。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flat_selector import text_nav_feats


def zscore(s: torch.Tensor) -> torch.Tensor:
    return (s - s.mean()) / (s.std() + 1e-6)


class I6Expert(nn.Module):
    def __init__(self, r: int, in_dim: int = 514):
        super().__init__()
        self.r = r
        self.A = nn.Parameter(torch.empty(r, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.b1 = nn.Parameter(torch.zeros(r))
        self.w2 = nn.Parameter(torch.zeros(r))
        self.b2 = nn.Parameter(torch.zeros(()))

    def parts(self, Z: torch.Tensor, f_txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """回傳 (s0, g)，皆為 [n]。"""
        tf = text_nav_feats(Z, f_txt)                              # [n, 2]
        s0 = zscore(tf[:, 0])
        u = torch.cat([Z, tf], dim=-1)                             # [n, 514]
        if self.r == 1:
            a = (u * self.A[0]).sum(-1, keepdim=True)              # [n, 1]
        else:
            a = u @ self.A.t()                                     # [n, r]（gemm）
        g = (F.gelu(a + self.b1) * self.w2).sum(-1) + self.b2
        return s0, g

    def forward(self, Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
        s0, g = self.parts(Z, f_txt)
        return s0 + g

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
