#!/usr/bin/env python3
"""MOE-2 E8：儲存（PREREG-18 E8、細則 24；不設門檻；不依賴 vec）。

  主系統、M3、LR、LRG 各自「所有任務共用」與「每任務」存了什麼、fp32 的 bytes、隨任務數 T 怎麼增加（S + T·p）。
  ridge 存累加統計量 A、B（W 由 A、B 解出，不另存）。尺寸由實際張量形狀計算。

    NAVCIL_MACHINE=mac python scripts/moe2_e8.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e8.json、e8.done
"""
from __future__ import annotations

import torch

import moe2_common as E
from selector.i6_expert import I6Expert

FP32 = 4


def body(run: E.Run) -> dict:
    run.total(1)
    F = run.ctx.F
    d = int(F.shape[1])
    xm, xv, xg = d + 1, d + 1, 2 * d + 1                       # [mean_vec; 1]、[v; 1]、[v; mean_vec; 1]
    n_head = I6Expert(2).n_params()
    sd = torch.load(E.head_path(run, 42, run.folds[0], run.tasks[0]), map_location="cpu")
    if sum(v.numel() for v in sd.values()) != n_head:
        raise E.CheckFailed("head 權重的參數數與 I6Expert(2) 不符")
    n_txt = int(F[0:2].numel())                                # 每任務兩類文字特徵 2 × d
    item = {
        "A_mv": ("AR／LIN8 的 A（[mean_vec; 1] 的 Σ x xᵀ）", xm * xm),
        "A_v": ("LR 的 A（[v; 1]）", xv * xv),
        "A_g": ("LRG 的 A（[v; mean_vec; 1]）", xg * xg),
        "head": ("I6(r = 2) head 參數", n_head),
        "text": ("兩類文字特徵（2 × 512）", n_txt),
        "B_ar": ("AR 的 B：該任務一欄（[mean_vec; 1] 之和）", xm),
        "B_lin8": ("LIN8 的 B：該任務兩類各一欄", 2 * xm),
        "B_v": ("LR 的 B：該任務兩類各一欄", 2 * xv),
        "B_g": ("LRG 的 B：該任務兩類各一欄", 2 * xg),
        "sigma": ("σ_a、σ_b（2 個純量）", 2),
    }
    systems = {
        "主系統": {"shared": ["A_mv"], "per_task": ["head", "text", "B_ar"]},
        "M3": {"shared": ["A_mv"], "per_task": ["head", "text", "B_lin8", "sigma"],
               "note": "AR 的欄 = LIN8 該任務兩欄之和（同一個 A；γ 只在求解時用），不另存"},
        "LR": {"shared": ["A_mv", "A_v"], "per_task": ["head", "text", "B_ar", "B_v"]},
        "LRG": {"shared": ["A_mv", "A_g"], "per_task": ["head", "text", "B_ar", "B_g"],
                "note": "A_mv 是 A_g 的子區塊、B_ar 是 B_g 兩欄之和的 mean_vec 部分；省去重複時共用只需 A_g、每任務不需 B_ar"},
    }
    out = {}
    for name, s in systems.items():
        S = sum(item[k][1] for k in s["shared"]) * FP32
        p = sum(item[k][1] for k in s["per_task"]) * FP32
        out[name] = {**s, "shared_bytes": S, "per_task_bytes": p, "T4_bytes": S + 4 * p}
    lrg_min_S, lrg_min_p = item["A_g"][1] * FP32, (n_head + n_txt + 2 * xg) * FP32
    out["LRG"]["dedup"] = {"shared_bytes": lrg_min_S, "per_task_bytes": lrg_min_p, "T4_bytes": lrg_min_S + 4 * lrg_min_p}
    run.tick()
    return {"d": d, "items": {k: {"what": v[0], "n_float": v[1], "fp32_bytes": v[1] * FP32} for k, v in item.items()},
            "systems": out,
            "not_counted": "CONCH 骨幹（影像與文字編碼器）、patch 特徵、λ*、logit scale：各系統相同，不計入",
            "solved_W_note": "W = solve(A + γI, B) 可隨時由 A、B 解出；推論時若快取 W，大小 = B 的大小"}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e8", body))
