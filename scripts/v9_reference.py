#!/usr/bin/env python3
"""唯讀：讀 reference/v9 存檔裡的既有 key 名。

這是本 repo 唯一允許出現舊方法名稱的程式檔，理由很窄：v9 的 JSON 是**已封存的
資料**，它的 key 就叫那個名字，讀它就得指名。本檔**不寫任何檔案**、不 import
selector、不參與任何 pipeline —— 純粹把存檔翻譯成中性欄位交出去。

de-QPMIL 的對照條件（v9 用 QPMIL 的 CFE 類別特徵與 QPMIL 完整前向）也記在這裡，
讓寫報告的腳本不必再指名。
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V9_EVAL_DIR = REPO_ROOT / "reference" / "v9" / "eval"

# 存檔 JSON 的 key：λ=0 那一列的 one-shot（selector + 單次 top-K）。
# one-shot 不受 λ 影響，各 λ 皆同值。
_LAMBDA_KEY = "lambda_0.00"
_ONESHOT_KEY = "zeronav_oneshot"

#: 報告用的對照條件（v9 側）。中性描述，供 verify 腳本直接引用。
V9_CONDITIONS = {
    "f_txt": "QPMIL CFE 增強類別特徵",
    "classifier": "QPMIL 完整前向 (aggregate_and_predict)",
    "weights": "QPMIL 內部 bag aggregation",
    "source_key": f"{_LAMBDA_KEY}.{_ONESHOT_KEY}",
}


def accuracy(task_pos: int, task: str) -> tuple[float, int]:
    """回傳 (v9 accuracy, v9 n_slides)。"""
    blob = json.loads((V9_EVAL_DIR / f"task{task_pos}_{task}_reverse_f1.json").read_text())
    res = blob["results"]
    return res[_LAMBDA_KEY][_ONESHOT_KEY], res["n_slides"]


def task_order() -> list[str]:
    """存檔檔名裡的任務序（task0..task3）。"""
    names = {}
    for p in V9_EVAL_DIR.glob("task*_reverse_f1.json"):
        pos = int(p.name[4:p.name.index("_")])
        names[pos] = p.name[p.name.index("_") + 1:-len("_reverse_f1.json")]
    return [names[i] for i in sorted(names)]
