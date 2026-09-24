"""EvidenceSelector — CONCH text-image alignment only, no external method deps.

ZeroSlide-inspired: uses max cosine similarity between patch embeddings
and class text embeddings (CONCH) as the selection foundation.

Per-patch summary is 2-D [max_txt, txt_ent]; the selector MLP therefore takes
D+2 = 514-D input.  No auxiliary prototype/backbone features are ever used.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def text_nav_feats(Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
    """2-D per-patch text-image alignment summary (ZeroSlide-inspired).

    Uses CONCH text-patch cosine similarity only.

    Z:     [n, D=512]  CONCH patch embeddings
    f_txt: [C, D=512]  CONCH class text embeddings
    returns [n, 2]: [max_text_sim, entropy(text_sim_distribution)]
    """
    z = F.normalize(Z, dim=-1)
    t = F.normalize(f_txt, dim=-1)
    txt = z @ t.t()                                               # [n, C]
    txt_prob = F.softmax(txt, dim=-1)
    txt_ent = -(txt_prob * (txt_prob + 1e-9).log()).sum(-1, keepdim=True)  # [n,1]
    return torch.cat([txt.amax(-1, keepdim=True), txt_ent], dim=-1)        # [n, 2]


def similarity_score(Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
    """ZeroSlide zero-shot patch importance score.

    max cosine similarity between CONCH patch embeddings and class text
    embeddings — analogous to ZeroSlide's patch scoring with TITAN/CONCH.
    No training required.

    Z: [n, D], f_txt: [C, D]  →  score [n]
    """
    z = F.normalize(Z, dim=-1)
    t = F.normalize(f_txt, dim=-1)
    return (z @ t.t()).amax(-1)                                   # [n]


class EvidenceSelector(nn.Module):
    """Learned patch-evidence selector.

    ZeroSlide-inspired: CONCH patch embedding + 2-D text-alignment summary
    → learned importance score.

    Architecture: Linear(D+2 → 256) → GELU → Linear(256 → 1)
    Default D=512 → 514-D input, ~130 K params.

    forward(Z, f_txt).
    """

    def __init__(self, feat_dim: int = 512, hidden: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z: torch.Tensor, f_txt: torch.Tensor) -> torch.Tensor:
        """
        Z:     [n, D]  CONCH patch embeddings
        f_txt: [C, D]  CONCH class text embeddings
        returns score [n].
        """
        s = text_nav_feats(Z, f_txt)          # [n, 2]
        u = torch.cat([Z, s], dim=-1)         # [n, D+2]
        return self.mlp(u).squeeze(-1)        # [n]


class SelectorBank:
    """task_id → EvidenceSelector state_dict.

    ⚠️ 這是 **per-task oracle upper bound**，不是我們的 CL 方法：
    每個 task 各存一份獨立訓練好的 selector，評估時用 ground-truth task_id
    直接取出對應的那一份。它的作用是給出「若 task identity 完全已知」的上界，
    供對照用；持續學習方法本身不得依賴這個 bank。
    """

    def __init__(self, feat_dim: int = 512, hidden: int = 256):
        self.feat_dim = feat_dim
        self.hidden = hidden
        self._skills: dict[int, dict] = {}

    # ── skill management ────────────────────────────────────────────────────

    def add_skill(self, task_id: int, selector_or_state) -> None:
        state = (selector_or_state.state_dict()
                 if hasattr(selector_or_state, "state_dict") else selector_or_state)
        self._skills[int(task_id)] = copy.deepcopy(state)

    def has(self, task_id: int) -> bool:
        return int(task_id) in self._skills

    def task_ids(self) -> list[int]:
        return sorted(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def build_selector(self, task_id: int, device=None) -> EvidenceSelector:
        selector = EvidenceSelector(feat_dim=self.feat_dim, hidden=self.hidden)
        selector.load_state_dict(self._skills[int(task_id)])
        if device is not None:
            selector = selector.to(device)
        return selector.eval()

    def weight_vectors(self) -> dict[int, torch.Tensor]:
        """Flat parameter vector per task (for cosine similarity analysis)."""
        return {
            int(tid): torch.cat([v.reshape(-1) for v in state.values()])
            for tid, state in self._skills.items()
        }

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "type": "selector_bank_v1",
            "feat_dim": self.feat_dim,
            "hidden": self.hidden,
            "skills": self._skills,
        }, path)
        print(f"[SelectorBank] saved {len(self)} skills → {path}")

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "SelectorBank":
        blob = torch.load(path, map_location=map_location)
        bank = cls(feat_dim=blob["feat_dim"], hidden=blob["hidden"])
        bank._skills = blob["skills"]
        print(f"[SelectorBank] loaded {len(bank)} skills from {path}")
        return bank
