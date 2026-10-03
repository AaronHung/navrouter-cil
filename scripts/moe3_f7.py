#!/usr/bin/env python3
"""MOE-3 f7：儲存（PREREG-19 F7、細則 26；格式同 moe2 E8；不設門檻；不依賴其他階段）。

  S-R0、S-A0 各自「所有任務共用」與「每任務」存了什麼、fp32 的 bytes、隨任務數 T 怎麼增加（S + T·p）；
  同表列主系統與 S-LR 供對照。ridge 存累加統計量 A、B（W 由 A、B 解出，不另存）。尺寸由實際張量形狀計算。

    NAVCIL_MACHINE=mac python scripts/moe3_f7.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/f7.json、f7.done
"""
from __future__ import annotations

import moe3_common as T
from selector.i6_expert import I6Expert

FP32 = 4


def body(run: T.Run) -> dict:
    run.total(1)
    F = run.ctx.F
    d = int(F.shape[1])
    x = int(T.anchor_T(run).shape[0])                           # [向量; 1] = 513
    if x != d + 1:
        raise T.CheckFailed(f"輸入維度 {x} ≠ d + 1 = {d + 1}")
    n_head = I6Expert(2).n_params()
    n_txt = int(F[0:2].numel())                                 # 每任務兩類文字特徵 2 × d
    item = {
        "A_mv": ("AR 的 A（[mean_vec; 1] 的 Σ x xᵀ）", x * x),
        "A_v": ("S-LR 的 A（[v(42); 1]）", x * x),
        "A_u": ("S-R0／S-A0 的 A（[u_64; 1]）", x * x),
        "head": ("I6(r = 2) head 參數", n_head),
        "text": ("兩類文字特徵（2 × 512；s0 選片、TXT、ANC 的 T 共用）", n_txt),
        "B_ar": ("AR 的 B：該任務一欄（[mean_vec; 1] 之和）", x),
        "B_v": ("S-LR 的 B：該任務兩類各一欄", 2 * x),
        "B_u": ("S-R0／S-A0 的 B：該任務兩類各一欄", 2 * x),
    }
    systems = {
        "主系統": {"shared": ["A_mv"], "per_task": ["head", "text", "B_ar"]},
        "S-LR": {"shared": ["A_mv", "A_v"], "per_task": ["head", "text", "B_ar", "B_v"]},
        "S-R0": {"shared": ["A_mv", "A_u"], "per_task": ["text", "B_ar", "B_u"], "note": "不存 head"},
        "S-A0": {"shared": ["A_mv", "A_u"], "per_task": ["text", "B_ar", "B_u"],
                 "note": "不存 head；T 就是每任務的文字特徵，不另存；γ、α 兩個純量不計"},
    }
    out = {}
    for name, s in systems.items():
        S = sum(item[k][1] for k in s["shared"]) * FP32
        p = sum(item[k][1] for k in s["per_task"]) * FP32
        out[name] = {**s, "shared_bytes": S, "per_task_bytes": p, "T4_bytes": S + 4 * p}
    run.tick()
    return {"d": d, "items": {k: {"what": v[0], "n_float": v[1], "fp32_bytes": v[1] * FP32} for k, v in item.items()},
            "systems": out,
            "not_counted": "CONCH 骨幹（影像與文字編碼器）、patch 特徵、超參數純量（γ、α、K）：不計入",
            "solved_W_note": "W = solve(A + γI, B［＋ γ·α·T］) 可隨時由 A、B（與文字特徵）解出；推論時若快取 W，大小 = B 的大小"}


if __name__ == "__main__":
    raise SystemExit(T.stage_main("f7", body))
