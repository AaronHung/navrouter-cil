"""只存累加統計量的 AR 分派器（PREREG-9）。

    X_j = [mean_vec_j ; 1]（float64，513 維）
    AR     ：A = Σ_j X_jᵀX_j，       b_j = X_jᵀ1
    AR-bal ：A = Σ_j w_j X_jᵀX_j，   b_j = w_j X_jᵀ1，w_j = 1/n_j
    W = solve(A + γI, [b_1 … b_t])

狀態只有 A、b_1…b_t 與任務 id 清單；add_task 只接收當前任務的 train mean_vec，結束後不保留它。
運算次序與 scripts/nc8_report.py:74-87 的從頭加總相同（A 從 0 起依任務序加），故 W 逐位元相同。
AMENDMENT-2：513 × 513 的累加與求解在 CPU 上用 float64。
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch

D64 = torch.float64


def aug(X: torch.Tensor) -> torch.Tensor:
    """mean_vec 轉 float64 後接常數 1（與 scripts/nc5_report.py:151-154 相同）。"""
    X = X.to(D64)
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=D64)], 1)


class IncrementalRidge:
    def __init__(self, gamma: float, balanced: bool = False, dim: int = 513):
        self.gamma, self.balanced, self.dim = float(gamma), bool(balanced), int(dim)
        self.A = torch.zeros(dim, dim, dtype=D64)
        self.b: list[torch.Tensor] = []
        self.task_ids: list = []

    def add_task(self, task_id, mean_vec: torch.Tensor) -> None:
        """只用當前任務的 train mean_vec [n, 512] 更新 A 與 B；不保留 mean_vec 的任何引用。"""
        if task_id in self.task_ids:
            raise ValueError(f"任務 {task_id!r} 已加入過")
        X = aug(mean_vec)
        w = 1.0 / X.shape[0] if self.balanced else 1.0
        self.A += w * (X.t() @ X)
        self.b.append(w * X.sum(0))
        self.task_ids.append(task_id)

    def solve(self) -> torch.Tensor:
        """W [513, t]，第 i 欄對應 task_ids[i]。"""
        return torch.linalg.solve(self.A + self.gamma * torch.eye(self.dim, dtype=D64), torch.stack(self.b, 1))

    def scores(self, mean_vec: torch.Tensor) -> torch.Tensor:
        return aug(mean_vec) @ self.solve()

    def save(self, path) -> int:
        """存統計量（.pt）；回傳檔案大小（bytes）。"""
        path = Path(path)
        torch.save({"A": self.A, "b": torch.stack(self.b) if self.b else torch.zeros(0, self.dim, dtype=D64),
                    "task_ids": list(self.task_ids), "gamma": self.gamma, "balanced": self.balanced,
                    "dim": self.dim}, path)
        return path.stat().st_size

    @classmethod
    def load(cls, path) -> "IncrementalRidge":
        s = torch.load(path, map_location="cpu")
        m = cls(s["gamma"], s["balanced"], s["dim"])
        m.A = s["A"]
        m.b = list(s["b"].unbind(0))
        m.task_ids = list(s["task_ids"])
        return m


def stream(task_ids: Iterable, load_train: Callable[[object], torch.Tensor], gamma: float,
           balanced: bool = False) -> Iterator[IncrementalRidge]:
    """依任務序逐階段累加：第 t 階段只呼叫 load_train(任務 t) 一次，yield 更新後的統計量。"""
    m = IncrementalRidge(gamma, balanced)
    for tid in task_ids:
        m.add_task(tid, load_train(tid))
        yield m
