#!/usr/bin/env python3
"""NC-15 步驟 3：依 validation WP 選 LoRA 對照版的 r（PREREG-15 操作定義 3、4）。

只讀：本 worktree 的 nc15/val/fold{f}.pt、--src-out 下 NC-2 validation 快取（L0）。不讀任何 test 檔。
輸出：outputs/navcil/<machine>/nc15/selection.json（commit 之後才可執行 nc15_test.py）。

    NAVCIL_MACHINE=mac python scripts/nc15_select.py --src-out <main>/outputs/navcil/mac
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

FOLDS = list(range(1, 11))
RS = (1, 2, 3)
L0_VAL_EXPECTED = 0.9290                     # REPORT_stage9.md:19
LITERAL_THRESHOLD = 0.9190                   # PREREG-15：L0 validation − 0.01


def masked(c8, labels, p) -> float:
    """與 scripts/nc7_report.py:37-39 相同。"""
    rr = torch.tensor(task_rows(p))
    return (rr[c8[:, rr].argmax(-1)] == labels).float().mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-out", required=True, help="既有產物目錄（cache/），只讀")
    args = ap.parse_args()
    src = Path(args.src_out).resolve()
    cfg = load_config()
    tasks = list(cfg["tasks"])
    own = REPO_ROOT / "outputs" / "navcil" / cfg["machine"] / "nc15"
    val = {f: torch.load(own / "val" / f"fold{f}.pt", map_location="cpu") for f in FOLDS}

    # 操作定義 3：L0 validation WP（NC-2 validation 快取，同 nc7_report.py:51-55）
    l0 = []
    for f in FOLDS:
        acc = []
        for p, t in enumerate(tasks):
            c = torch.load(src / "cache" / f"nc2_fold{f}_val_{t}.pt", map_location="cpu")
            acc.append(masked(c["four_cos8_uni"][:, p], c["labels"], p))
        l0.append(sum(acc) / 4)
    l0_m = mean_sd(l0)[0]
    if round(l0_m, 4) != L0_VAL_EXPECTED:
        raise SystemExit(f"L0 validation WP 重算 {l0_m:.6f} ≠ {L0_VAL_EXPECTED}（REPORT_stage9.md:19），停止")
    thr = l0_m - 0.01

    # 操作定義 2：L1(r) validation WP（各序、十折、四任務）
    wp = {}
    for r in RS:
        for o in ORDERS:
            per_fold, per_task = [], {t: [] for t in tasks}
            for f in FOLDS:
                acc = []
                for p, t in enumerate(tasks):
                    e = val[f]["tasks"][t]
                    a = masked(e[f"r{r}"][o]["cos8"], e["labels"], p)
                    acc.append(a); per_task[t].append(a)
                per_fold.append(sum(acc) / 4)
            m, sd = mean_sd(per_fold)
            wp[f"r{r}.{o}"] = {"per_fold": per_fold, "mean": m, "sd": sd, "per_task": per_task,
                               "pass": m >= thr, "pass_literal": m >= LITERAL_THRESHOLD}

    # 操作定義 3 末段：未四捨五入門檻與字面 0.9190 的判定不同即停
    diff = [k for k, v in wp.items() if v["pass"] != v["pass_literal"]]
    if diff:
        raise SystemExit(f"過門檻判定在未四捨五入門檻 {thr:.6f} 與 0.9190 之間不同：{diff}，停止")

    # 操作定義 4：兩序都要；最小 r；都不符合時取兩序較低者最高的 r（同分取小）
    ok = {r: all(wp[f"r{r}.{o}"]["pass"] for o in ORDERS) for r in RS}
    passing = [r for r in RS if ok[r]]
    if passing:
        r_star, meets = min(passing), True
    else:
        low = {r: min(wp[f"r{r}.{o}"]["mean"] for o in ORDERS) for r in RS}
        r_star, meets = max(RS, key=lambda r: (low[r], -r)), False
    base_check = {f: {t: val[f]["tasks"][t]["check_l0_cos_maxabs"] for t in tasks} for f in FOLDS}
    sel = {"l0_val_per_fold": l0, "l0_val_mean": l0_m, "l0_val_sd": mean_sd(l0)[1],
           "threshold": thr, "threshold_literal": LITERAL_THRESHOLD,
           "wp_val": wp, "r_pass": {str(r): ok[r] for r in RS}, "passing_r": passing,
           "r_star": r_star, "r_star_meets": meets,
           "check6_l0_cos_maxabs": max(x for v in base_check.values() for x in v.values()),
           "note": "只讀 validation（nc15/val、NC-2 validation 快取）；未讀任何 test 檔。"}
    (own / "selection.json").write_text(json.dumps(sel, indent=1))
    for r in RS:
        print(f"r={r}: " + "  ".join(f"{o} {wp[f'r{r}.{o}']['mean']:.4f} ± {wp[f'r{r}.{o}']['sd']:.4f}"
                                     f"{' ✓' if wp[f'r{r}.{o}']['pass'] else ' ✗'}" for o in ORDERS)
              + f"  → {'符合' if ok[r] else '不符合'}")
    print(f"L0 validation WP = {l0_m:.4f}；門檻 {thr:.4f}；r* = {r_star}{'' if meets else '（未符合）'}")
    print(f"→ {own / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
