"""L 線：以凍結底座 expert 為基礎的低秩增量 expert（PREREG-2 操作定義 8、9）。

    W1 = W1_base + B·A      A ∈ R^{r×514}（kaiming_uniform a=√5）、B ∈ R^{256×r}（0）
    b1、第二層 Linear(256→1) 以底座數值初始化，可訓練；W1_base 凍結。

與 EvidenceSelector 相同的前向：[Z ; text_nav_feats(Z, f_txt)] → Linear → GELU → Linear。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flat_selector import EvidenceSelector, text_nav_feats


class LowRankExpert(nn.Module):
    def __init__(self, base: EvidenceSelector, r: int):
        super().__init__()
        W1 = base.mlp[0].weight.detach().clone()                  # [256, 514]
        self.register_buffer("W1_base", W1)
        self.A = nn.Parameter(torch.empty(r, W1.shape[1]))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(W1.shape[0], r))
        self.b1 = nn.Parameter(base.mlp[0].bias.detach().clone())
        self.fc2 = nn.Linear(W1.shape[0], 1)
        self.fc2.load_state_dict(base.mlp[2].state_dict())
        self.r = r

    def forward(self, Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
        u = torch.cat([Z, text_nav_feats(Z, f_txt)], dim=-1)       # [n, 514]
        h = F.linear(u, self.W1_base, self.b1) + (u @ self.A.t()) @ self.B.t()
        return self.fc2(F.gelu(h)).squeeze(-1)

    def delta(self) -> dict:
        """相對底座的增量（合併用）。"""
        return {"W1": (self.B @ self.A).detach(),
                "b1": self.b1.detach(), "W2": self.fc2.weight.detach(),
                "b2": self.fc2.bias.detach()}

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def merged_expert(base: EvidenceSelector, experts: list[LowRankExpert]) -> EvidenceSelector:
    """底座 + Σ 各任務增量（b1、第二層取相對底座的差再相加）。"""
    m = EvidenceSelector(feat_dim=base.feat_dim, hidden=base.hidden)
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    for e in experts:
        d = e.delta()
        sd["mlp.0.weight"] += d["W1"]
        sd["mlp.0.bias"] += d["b1"] - base.mlp[0].bias.detach()
        sd["mlp.2.weight"] += d["W2"] - base.mlp[2].weight.detach()
        sd["mlp.2.bias"] += d["b2"] - base.mlp[2].bias.detach()
    m.load_state_dict(sd)
    return m.eval()
