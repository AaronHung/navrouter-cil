#!/usr/bin/env python3
"""V2 + P-A — 對照 reference/v9 的偏移診斷與翻轉分析（純推論，不重訓）。

用 reference/v9/skill_bank_reverse_f1.pt 的 per-task selector，在 budget=64、
單次 top-K（不用 multiround）下重跑四個 task 的 test split。

主線權重 = softmax(top-K selector 分數)，**依訓練一致性原則選定，非依數值選定**
（等權會把 counterfactual utility 結構性稀釋：多看一個 patch 只把均值挪動
1/|E_t|，budget 越大訊號越平）。等權保留為 selection-only ablation。

輸出：
  outputs/verify/DELTA_v9.md      accuracy 對照表
  outputs/verify/FLIPS_v9.md      翻轉分析（P-A）
  outputs/verify/per_slide_v9.json 逐 slide 預測（供後續分析）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts import v9_reference                                    # noqa: E402
from selector.evaluate import (iter_test_slides, score_based_indices,  # noqa: E402
                               select_and_classify)
from selector.flat_selector import SelectorBank                     # noqa: E402
from selector.text_encoder import build_f_txt, load_config          # noqa: E402

BUDGET = 64
BANK_PATH = REPO_ROOT / "reference" / "v9" / "skill_bank_reverse_f1.pt"
OUT_DIR = REPO_ROOT / "outputs" / "verify"
MAINLINE = "softmax"
ABLATION = "uniform"


def build_f_txt_all(cfg, device) -> torch.Tensor:
    """8-way label space：依任務序串接每個 task 的 [2, 512]。見 P-B 對齊測試。"""
    return torch.cat([build_f_txt(t, cfg, device=device).f_txt for t in cfg["tasks"]], 0)


@torch.no_grad()
def eval_one_task(selector, f_txt, logit_scale, cfg, task, task_pos):
    """單次 top-K + conch_classify，argmax over 全部 8 類。逐 slide 記錄。"""
    records = []
    for rec in iter_test_slides(cfg, task, task_pos):
        scores = selector(rec.Z, f_txt)
        idx = score_based_indices(scores, BUDGET)
        pred_main, _ = select_and_classify(rec.Z, idx, f_txt, logit_scale,
                                           scores=scores, weighting=MAINLINE)
        pred_abl, _ = select_and_classify(rec.Z, idx, f_txt, logit_scale,
                                          weighting=ABLATION)
        records.append({"slide_id": rec.sid, "task": task, "task_pos": task_pos,
                        "n_patch": int(rec.Z.shape[0]), "true": rec.label,
                        "pred_softmax": pred_main, "pred_uniform": pred_abl})
    return records


def acc(records, key) -> float:
    return sum(r[key] == r["true"] for r in records) / len(records)


def main() -> int:
    cfg = load_config()
    device = torch.device("cpu")          # CPU：可重現，slide 規模下也夠快
    torch.manual_seed(42)

    f_txt = build_f_txt_all(cfg, device)
    logit_scale = build_f_txt(cfg["tasks"][0], cfg, device=device).logit_scale
    bank = SelectorBank.load(str(BANK_PATH))
    print(f"f_txt {tuple(f_txt.shape)}  logit_scale {float(logit_scale):.4f}  "
          f"bank tasks {bank.task_ids()}")

    rows, per_slide, over = [], [], []
    for task_pos, task in enumerate(cfg["tasks"]):
        selector = bank.build_selector(task_pos, device)
        recs = eval_one_task(selector, f_txt, logit_scale, cfg, task, task_pos)
        per_slide += recs
        a_main, a_abl = acc(recs, "pred_softmax"), acc(recs, "pred_uniform")
        ref_acc, ref_n = v9_reference.accuracy(task_pos, task)
        delta = (a_main - ref_acc) * 100
        rows.append(dict(task=task, task_pos=task_pos, n=len(recs), ref_n=ref_n,
                         ref=ref_acc, new=a_main, new_ablation=a_abl, delta=delta,
                         records=recs))
        print(f"  task{task_pos} {task:10s} n={len(recs):4d}(v9 {ref_n})  "
              f"v9={ref_acc:.4f}  new={a_main:.4f}  delta={delta:+.2f}pp  "
              f"(selection-only ablation {a_abl:.4f})", flush=True)
        if abs(delta) > 10:
            over.append(task)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_slide_v9.json").write_text(json.dumps(per_slide, indent=1))
    write_delta(rows, over)
    write_flips(rows)
    print(f"\n→ {OUT_DIR / 'DELTA_v9.md'}\n→ {OUT_DIR / 'FLIPS_v9.md'}"
          f"\n→ {OUT_DIR / 'per_slide_v9.json'}")
    if over:
        print(f"\n⚠️  delta 超過 10 pp：{', '.join(over)} — 停下來，不要繼續往下做。")
        return 1
    return 0


def write_delta(rows, over) -> None:
    m_ref = sum(r["ref"] for r in rows) / len(rows)
    m_new = sum(r["new"] for r in rows) / len(rows)
    m_abl = sum(r["new_ablation"] for r in rows) / len(rows)
    c = v9_reference.V9_CONDITIONS
    L = [
        "# DELTA_v9 — 拔掉舊方法造成的偏移",
        "",
        "純推論診斷，不重訓。`reference/v9/skill_bank_reverse_f1.pt` 的 per-task "
        "selector，budget=64、單次 top-K（不用 multiround）、reverse order、fold 1、"
        "8-way label space（疊放順序已由 `tests/test_label_space_alignment.py` 釘死）。",
        "",
        "**權重政策：主線用 softmax(top-K selector 分數)，依訓練一致性原則選定，"
        "非依數值選定。** 等權（selection-only）在下方單獨列為 ablation。",
        "",
        "| task | v9 accuracy | new accuracy | delta (pp) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['task']} | {r['ref']:.4f} | {r['new']:.4f} | {r['delta']:+.2f} |")
    L += [
        f"| **mean** | **{m_ref:.4f}** | **{m_new:.4f}** | "
        f"**{(m_new - m_ref) * 100:+.2f}** |",
        "",
        "## 對照條件",
        "",
        "| | v9 | new |",
        "|---|---|---|",
        f"| f_txt | {c['f_txt']} | CONCH text tower 原生 |",
        f"| 分類器 | {c['classifier']} | `conch_classify` |",
        "| selector | reference/v9 skill bank | 同左（未重訓） |",
        f"| 聚合權重 | {c['weights']} | softmax(top-K 分數) |",
        "| budget / 選法 | 64 / 單次 top-K | 同左 |",
        "| slide 集合 | fold 1 test split | 同左 |",
        "",
        f"v9 欄取自 `reference/v9/eval/task*_reverse_f1.json` 的 `{c['source_key']}`"
        "（one-shot 不受 λ 影響，各 λ 皆同值）。",
        "",
        "## slide 數對照",
        "",
        "| task | v9 n_slides | new n_slides |",
        "|---|---|---|",
    ]
    for r in rows:
        flag = "" if r["n"] == r["ref_n"] else "  ⚠️ 不一致"
        L.append(f"| {r['task']} | {r['ref_n']} | {r['n']}{flag} |")
    L += [
        "",
        "## Ablation：selection-only（等權聚合）",
        "",
        "同一組 top-K、同一個 `conch_classify`，只把權重換成等權。**這是 ablation，"
        "不是候選主線** —— 主線已依訓練一致性定為 softmax。",
        "",
        "| task | 主線 softmax | selection-only 等權 |",
        "|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['task']} | {r['new']:.4f} | {r['new_ablation']:.4f} |")
    L += [
        f"| **mean** | **{m_new:.4f}** | **{m_abl:.4f}** |",
        "",
        "## 判定",
        "",
        (f"⚠️ **有 task 的 |delta| > 10 pp：{', '.join(over)}**。依約定停下來。"
         if over else "所有 task 的 |delta| 都在 10 pp 以內。"),
        "",
        "重跑：`python scripts/verify_v9_delta.py`",
    ]
    (OUT_DIR / "DELTA_v9.md").write_text("\n".join(L) + "\n")


def write_flips(rows) -> None:
    L = [
        "# FLIPS_v9 — 翻轉分析（P-A）",
        "",
        "## ⚠️ 能算什麼、不能算什麼",
        "",
        "`reference/v9/` 只封存了**每個 task 的彙總 accuracy**"
        "（`eval/*.json` 是 λ × policy 的 accuracy，另一個彙總檔是 4×4 cross-task 矩陣），"
        "**沒有逐 slide 預測**。因此「v9 對→new 錯」這種 gross flip 無法從存檔算出來，"
        "只能從淨值推出區間。要拿到真值，唯一辦法是把 v9 的 backbone 重跑一次推論"
        "（需要 timm/torchvision 與舊方法的模型程式），那已超出目前授權範圍。",
        "",
        "下面提供兩件**算得出來**的事：",
        "1. 對 v9 的 gross flip **區間**（由淨值與正確張數推得）。",
        "2. 主線 softmax ↔ selection-only 等權之間的**逐張 gross flip**（真值）。",
        "",
        "## 1. 對 v9 的 gross flip 區間",
        "",
        "令 `f_cw` = v9 對→new 錯，`f_wc` = v9 錯→new 對，則 "
        "`f_wc - f_cw = new_correct - v9_correct`（淨值）。",
        "單靠淨值無法定出 gross，只能給上界：`f_cw ≤ min(v9_correct, new_wrong)`、"
        "`f_wc ≤ min(v9_wrong, new_correct)`。",
        "",
        "| task | n | v9 correct | new correct | 淨 | f_cw 上界 | f_wc 上界 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = r["n"]
        v9_c = round(r["ref"] * r["ref_n"])
        new_c = round(r["new"] * n)
        L.append(f"| {r['task']} | {n} | {v9_c} | {new_c} | {new_c - v9_c:+d} | "
                 f"{min(v9_c, n - new_c)} | {min(n - v9_c, new_c)} |")
    L += [
        "",
        "所以「esca / brca 淨值為 0」**不能**推論成零翻轉：以 brca 為例，最多可能有 "
        f"{min(round(rows[2]['ref'] * rows[2]['ref_n']), rows[2]['n'] - round(rows[2]['new'] * rows[2]['n']))}"
        " 張互相抵消。這個問題目前無解，需要 v9 逐 slide 預測。",
        "",
        "## 2. 主線 softmax ↔ selection-only 等權：逐張 gross flip（真值）",
        "",
        "同一組 top-K、同一個分類器，只換權重政策。",
        "",
        "| task | n | softmax 對 | 等權 對 | 淨 | 等權對→softmax錯 | 等權錯→softmax對 | gross |",
        "|---|---|---|---|---|---|---|---|",
    ]
    detail = []
    for r in rows:
        recs = r["records"]
        cw = [x for x in recs if x["pred_uniform"] == x["true"] != x["pred_softmax"]]
        wc = [x for x in recs if x["pred_softmax"] == x["true"] != x["pred_uniform"]]
        n_s = sum(x["pred_softmax"] == x["true"] for x in recs)
        n_u = sum(x["pred_uniform"] == x["true"] for x in recs)
        L.append(f"| {r['task']} | {len(recs)} | {n_s} | {n_u} | {n_s - n_u:+d} | "
                 f"{len(cw)} | {len(wc)} | {len(cw) + len(wc)} |")
        if cw or wc:
            detail.append((r["task"], cw, wc))
    L += ["", "### 翻轉 slide 明細", ""]
    if not detail:
        L.append("（無翻轉）")
    for task, cw, wc in detail:
        L += [f"#### {task}", "",
              "| slide id | true | 等權 pred | softmax pred | 方向 |",
              "|---|---|---|---|---|"]
        for x in cw:
            L.append(f"| {x['slide_id']} | {x['true']} | {x['pred_uniform']} | "
                     f"{x['pred_softmax']} | 等權對 → softmax錯 |")
        for x in wc:
            L.append(f"| {x['slide_id']} | {x['true']} | {x['pred_uniform']} | "
                     f"{x['pred_softmax']} | 等權錯 → softmax對 |")
        L.append("")
    L += [
        "## 3. 預測落點（P-B 的經驗佐證）",
        "",
        "每個 task 佔 8-way label space 的第 2p、2p+1 列。若疊放順序錯位，預測會"
        "大量落到別的 task 的列上。",
        "",
        "| task | 自己的列 | 落在自己列 | 落到別的 task |",
        "|---|---|---|---|",
    ]
    for r in rows:
        owned = {2 * r["task_pos"], 2 * r["task_pos"] + 1}
        inside = sum(x["pred_softmax"] in owned for x in r["records"])
        L.append(f"| {r['task']} | {sorted(owned)} | {inside}/{r['n']} | "
                 f"{r['n'] - inside} |")
    L += ["", "逐 slide 原始預測：`outputs/verify/per_slide_v9.json`", ""]
    (OUT_DIR / "FLIPS_v9.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
