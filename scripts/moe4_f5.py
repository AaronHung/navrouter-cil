#!/usr/bin/env python3
"""MOE-4 f5：H5 儲存（PREREG-20 細則 28）。由實際張量形狀與 head 參數數計算（d = 512、x = 513；fp32）；不設門檻。

  P-F（定案候選）與 P-4 為主表；P-main、P-T1、P-0 列對照。
  共用：ridge 的 A（XᵀX）、AR 的 A_mv；每任務：I6 head、兩類文字特徵、B（ridge）、B_ar（AR）。
  CONCH 骨幹、patch 特徵、超參數純量列出但不計入。

    NAVCIL_MACHINE=mac python scripts/moe4_f5.py --device cpu [--folds 1-10] [--out moe4]
輸出：outputs/navcil/<machine>/<out>/f5.json、f5.done
"""
from __future__ import annotations

import torch

import moe4_common as X
from moe4_common import M, N5, T


def body(run: T.Run) -> dict:
    X.require_chk(run)
    f = run.folds[0]
    D = X.data4(run, "train", f)["per"][0]
    x = N5.aug(torch.zeros(1, D["mv"].shape[-1], dtype=torch.float64)).shape[1]        # 513
    A_w = X.acc_of(run, "w42").AB(f, "reverse", 4)[0]
    A_u = X.acc_of(run, "u64").AB(f, "reverse", 4)[0]
    B_w = X.acc_of(run, "w42").AB(f, "reverse", 1)[1]                                  # [513, 2]
    B_u = X.acc_of(run, "u64").AB(f, "reverse", 1)[1]                                  # [513, 2]
    B_ar = torch.zeros(x, 1, dtype=torch.float64)                                       # AR：該任務一欄
    head = M.load_head(X.head_path(run, 42, f, run.tasks[0]))
    n_head = sum(p.numel() for p in head.parameters())
    n_text = 2 * run.ctx.F.shape[1]                                                    # 兩類文字特徵
    items = {
        "A_w": {"what": "ridge 的 A（[w(s); 1] 的 XᵀX，所有任務共用）", "shape": list(A_w.shape), "n_float": A_w.numel()},
        "A_mv": {"what": "AR 的 A（[mean_vec; 1] 的 XᵀX，所有任務共用）", "shape": [x, x], "n_float": x * x},
        "A_u": {"what": "ridge 的 A（[u_64; 1]，P-0）", "shape": list(A_u.shape), "n_float": A_u.numel()},
        "head": {"what": "I6(r = 2) head 參數（每任務一個）", "shape": [n_head], "n_float": n_head},
        "text": {"what": "兩類文字特徵（2 × 512；head 的 s0、TXT 與 u_64 的 s0 都要用）", "shape": [2, 512], "n_float": n_text},
        "B_w": {"what": "ridge 的 B：該任務兩欄（[w(s); 1]）", "shape": list(B_w.shape), "n_float": B_w.numel()},
        "B_u": {"what": "ridge 的 B：該任務兩欄（[u_64; 1]，P-0）", "shape": list(B_u.shape), "n_float": B_u.numel()},
        "B_ar": {"what": "AR 的 B：該任務一欄（[mean_vec; 1] 之和）", "shape": list(B_ar.shape), "n_float": B_ar.numel()},
    }
    T_TASKS = 4
    sysdef = {
        "P-F": (["A_w", "A_mv"], ["head", "text", "B_w", "B_ar"]),
        "P-4": (["A_w", "A_mv"], ["head", "text", "B_w", "B_ar"]),
        "P-main": (["A_mv"], ["head", "text", "B_ar"]),
        "P-T1": (["A_mv"], ["head", "text", "B_ar"]),
        "P-0": (["A_u", "A_mv"], ["text", "B_u", "B_ar"]),
    }
    systems = {}
    for name, (shared, per) in sysdef.items():
        sh = sum(items[i]["n_float"] for i in shared)
        pt = sum(items[i]["n_float"] for i in per)
        systems[name] = {"shared": shared, "per_task": per, "shared_bytes": sh * 4, "per_task_bytes": pt * 4,
                         "T4_bytes": (sh + T_TASKS * pt) * 4, "note": "不存 head" if "head" not in per else ""}
    for it in items.values():
        it["fp32_bytes"] = it["n_float"] * 4
    return {"folds": run.folds, "dim_x": x, "T": T_TASKS, "items": items, "systems": systems,
            "not_counted": "CONCH 骨幹（影像與文字編碼器）、patch 特徵、超參數純量（γ = 1e-3、α 不用）",
            "note": "P-F 的 W 由 solve(A + γI, B) 求得，推論時不另存 W；B 為 2 欄（該任務兩類）"}


if __name__ == "__main__":
    raise SystemExit(X.stage_main("f5", body))
