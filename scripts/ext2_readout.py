#!/usr/bin/env python3
"""EXT-2 收尾：FINAL-B 讀出 ridge 的 B、W（fold 1、reverse 序、t = 4）存檔並印出 shape；核對判類規則。只讀快取，不訓練。

A、B、W 在既有產物中沒有存檔（每次由快取的 train 向量累加、求解）；本檔把這一格存成
outputs/navcil/<machine>/ext2/readout_fold1_reverse_t4.pt（.pt 不進版控），再從該檔讀回來印。

    NAVCIL_MACHINE=mac python scripts/ext2_readout.py
輸出：ext2/readout_fold1_reverse_t4.pt、ext2/readout.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext2_a as EA                                                       # noqa: E402
from ext1_c import C, M, N5, T                                            # noqa: E402

FOLD, ORDER = 1, "reverse"


def main() -> int:
    sys.argv = sys.argv[:1]
    cx = EA.Ctx()
    r3, root = cx.r3, cx.base / "ext2"
    gB = json.loads((root / "a.json").read_text())["gamma_B"]
    st = T.stats(r3, EA.KIND_B)
    stages = {}
    for t in range(1, 5):
        A, B, cls = st.AB(FOLD, ORDER, t)
        W, _ = st.W(FOLD, ORDER, t, ("RDG", gB))
        stages[t] = {"B_shape": list(B.shape), "W_shape": list(W.shape), "classes": [C.LABEL[c] for c in cls]}
    path = root / f"readout_fold{FOLD}_{ORDER}_t4.pt"
    torch.save({"A": A, "B": B, "W": W, "classes": cls, "gamma": gB, "fold": FOLD, "order": ORDER, "t": 4}, path)

    L = torch.load(path, map_location="cpu")                                # 從存檔讀回
    A, B, W, cls = L["A"], L["B"], L["W"], L["classes"]
    W_ar, pos = cx.st.b.W(FOLD, ORDER, 4, M.G_AR)

    # 判類規則核對（fold 1 全部 test、t = 4）：τ̂ 兩欄內 argmax 與「d = 第一類 − 第二類 ≥ 0 判第一類」是否逐張相同
    D = T.data(r3, "test", FOLD)
    th = T.ar_stages(r3, FOLD, ORDER, D)[3].argmax(-1)
    S = N5.aug(T.vec_of(D, EA.KIND_B, th)) @ W                              # [N, 8]
    n = torch.arange(len(th))
    two = torch.stack([S[n, 2 * th], S[n, 2 * th + 1]], 1)
    d = two[:, 0] - two[:, 1]
    p_arg, p_rule = 2 * th + two.argmax(1), 2 * th + (~(d >= 0)).long()
    ref = json.loads(cx.path("FINALB", ORDER, FOLD).read_text())["cil4"]
    res = {"file": str(path.relative_to(REPO_ROOT)), "fold": FOLD, "order": ORDER, "t": 4, "gamma": gB,
           "A_shape": list(A.shape), "B_shape": list(B.shape), "W_shape": list(W.shape), "dtype": str(B.dtype),
           "columns": [C.LABEL[c] for c in cls],
           "B_last_row": B[-1].tolist(), "B_last_row_sum": float(B[-1].sum()), "A_last": float(A[-1, -1]),
           "stages": stages, "AR_W_shape": list(W_ar.shape), "AR_columns": [cx.tasks[p] for p in pos],
           "check": {"n_test": len(th), "argmax_vs_d_rule_diff": int((p_arg != p_rule).sum()), "n_ties": int((d == 0).sum()),
                     "pred_diff_vs_A_stage": int((p_rule != torch.tensor(ref)).sum()),
                     "global8_argmax_outside_tau_pair": int(((S.argmax(1) // 2) != th).sum())}}
    (root / "readout.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
