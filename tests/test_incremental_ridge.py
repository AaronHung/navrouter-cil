"""IncrementalRidge（PREREG-9 判準 6）：與從頭加總相同、第 t 階段只讀任務 t、與加入順序無關。"""
from __future__ import annotations

import gc
import os
import sys
import weakref
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from selector.incremental_ridge import D64, IncrementalRidge, aug, stream

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {"AR": (1e-3, False), "AR-bal": (1e-4, True)}
SIZES = (150, 300, 250, 200)          # 合計 900 > 513，與實際資料一樣 A 為滿秩


def synth() -> dict[int, torch.Tensor]:
    """四個任務的假 mean_vec（float32、L2 正規化，各任務有自己的平均方向）。"""
    g = torch.Generator().manual_seed(0)
    out = {}
    for p, n in enumerate(SIZES):
        mu = torch.randn(512, generator=g)
        out[p] = F.normalize(2.0 * mu + torch.randn(n, 512, generator=g), dim=-1)
    return out


def from_scratch(Xs: list[torch.Tensor], gamma: float, bal: bool) -> torch.Tensor:
    """原本的從頭加總（scripts/nc8_report.py:74-87 的非 LIN8 分支）。"""
    A = torch.zeros(513, 513, dtype=D64)
    cols = []
    for X in Xs:
        X = aug(X)
        w = 1.0 / X.shape[0] if bal else 1.0
        A += w * (X.t() @ X)
        cols.append(w * X.sum(0))
    return torch.linalg.solve(A + gamma * torch.eye(513, dtype=D64), torch.stack(cols, 1))


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_matches_from_scratch_every_stage(variant, tmp_path):
    gamma, bal = VARIANTS[variant]
    data = synth()
    order = [3, 2, 1, 0]
    for t, m in enumerate(stream(order, data.__getitem__, gamma, bal), 1):
        W = m.solve()
        ref = from_scratch([data[p] for p in order[:t]], gamma, bal)
        assert m.task_ids == order[:t]
        assert (W - ref).abs().max().item() <= 1e-10
        # 存檔再載入後 W 不變
        m.save(tmp_path / "s.pt")
        assert torch.equal(IncrementalRidge.load(tmp_path / "s.pt").solve(), W)


@pytest.mark.parametrize("variant", VARIANTS)
def test_b_stage_t_reads_only_task_t(variant):
    gamma, bal = VARIANTS[variant]
    data = synth()
    order = [0, 1, 2, 3]
    reads, refs = [], []

    def loader(p):
        reads.append(p)
        X = data[p].clone()
        refs.append(weakref.ref(X))
        return X

    for t, m in enumerate(stream(order, loader, gamma, bal), 1):
        assert reads == [order[t - 1]], f"第 {t} 階段讀了 {reads}"
        reads.clear()
        gc.collect()
        assert all(r() is None for r in refs), "add_task 之後仍保留 mean_vec 的引用"
        assert set(vars(m)) == {"gamma", "balanced", "dim", "A", "b", "task_ids"}


@pytest.mark.parametrize("variant", VARIANTS)
def test_c_order_independent(variant):
    gamma, bal = VARIANTS[variant]
    data = synth()
    Ws = []
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
        *_, m = stream(order, data.__getitem__, gamma, bal)
        W = m.solve()
        Ws.append(W[:, [m.task_ids.index(p) for p in range(4)]])
    for W in Ws[1:]:
        assert (W - Ws[0]).abs().max().item() <= 1e-10


def _nc8_cache_ready() -> bool:
    if os.environ.get("NAVCIL_MACHINE") != "mac":
        return False
    return (REPO_ROOT / "outputs/navcil/mac/cache/nc8_fold1_train_tcga_lung.pt").exists()


@pytest.mark.skipif(not _nc8_cache_ready(), reason="需要 Mac 上的 NC-8 快取（NAVCIL_MACHINE=mac）")
@pytest.mark.parametrize("variant", VARIANTS)
def test_a_matches_nc8_report_fold1(variant):
    """實際資料 fold 1：與原本的 nc8_report.B8.W 比對（兩序、四階段）。"""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import nc8_report as N8
    from selector.cil_ops import ORDERS

    gamma, bal = VARIANTS[variant]
    b = N8.B8()
    for o, names in ORDERS.items():
        pos = [b.tasks.index(x) for x in names]
        load = lambda p: torch.load(b.cache / f"nc8_fold1_train_{b.tasks[p]}.pt", map_location="cpu")["mean_vec"]  # noqa: E731
        for t, m in enumerate(stream(pos, load, gamma, bal), 1):
            W_ref, pos_ref = b.W(1, o, t, gamma, bal)
            assert pos_ref == m.task_ids
            assert (m.solve() - W_ref).abs().max().item() <= 1e-10
