#!/usr/bin/env python3
"""NC-15 報告：REPORT_stage17.md（讀 nc15/selection.json、result.json、facts.json、timing_*.json）。

    NAVCIL_MACHINE=mac python scripts/nc15_report.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.cil_ops import ORDERS                                       # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

TASKS = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]


def commit_of(rel: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), "log", "-1", "--format=%h", "--", rel],
                          capture_output=True, text=True).stdout.strip()


def main() -> int:
    cfg = load_config()
    base = REPO_ROOT / "outputs" / "navcil" / cfg["machine"]
    own = base / "nc15"
    S = json.loads((own / "selection.json").read_text())
    R = json.loads((own / "result.json").read_text())
    facts = json.loads((own / "facts.json").read_text())
    line = {x["id"]: i + 2 for i, x in enumerate(facts)}

    def fid(k):
        return f"`{k}`（nc15/facts.json:{line[k]}）"

    rs = R["r_star"]
    thr = S["threshold"]
    V = R["values"]
    ms = lambda k: f"{V[k]['mean']:.4f} ± {V[k]['sd']:.4f}"                  # noqa: E731
    tm_tr = json.loads((own / "timing_train_r3.json").read_text())
    tm_va = json.loads((own / "timing_val.json").read_text())
    tr_s = sum(v for k, v in tm_tr.items() if k.startswith("train/"))
    ev_s = sum(v for k, v in tm_tr.items() if k.startswith("eval/"))
    va_s = sum(v for k, v in tm_va.items() if k.startswith("val/"))

    out = ["# REPORT — NC-15：LoRA 對照版（L1 v2）的 r 改以 validation 選定（Mac CPU，十折兩序）", "",
           f"機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；所有數字來自同一台。"
           f"判準見 `PREREG-15.md`（{commit_of('PREREG-15.md')}），檢查 6 的比對方式見 `AMENDMENT-4.md`"
           f"（{commit_of('AMENDMENT-4.md')}）。選 r 只讀 validation；`nc15/selection.json` 於 {R['selection_commit']} "
           "commit 之後才讀 test。r = 1、2 的權重沿用 `lora_v2/`（NC-3 批次），r = 3 為本輪新訓練"
           "（`nc15/lora_r3/`，log：`logs/nc15_lora_r3.log`）；validation 評估 log：`logs/nc15_val.log`。"
           "task-inferred = AR 分派（γ = 0.001，float64），task-known = oracle 分派；表二的修正頭數字在本輪以同一程式重算。"
           "數字附 fact-id 與所在 facts.json 行號（路徑相對於 `outputs/navcil/mac/`）。", ""]

    # ── 判準落點 ──
    out += ["## 判準落點", "", "| 項目 | 數值 | 結果 |", "|---|---|---|",
            f"| L0 validation WP 重算 = 0.9290（REPORT_stage9.md:19） | {S['l0_val_mean']:.4f} ± {S['l0_val_sd']:.4f}"
            f"（{fid('nc15.val.l0.wp')}） | 符合 |",
            f"| 門檻 = L0 validation − 0.01 | {thr:.6f}（四捨五入 0.9190；{fid('nc15.val.threshold')}） | — |",
            f"| 未四捨五入門檻與字面 0.9190 的判定是否一致 | 三個候選、兩序皆一致 | 符合 |",
            f"| validation 評估檢查 6（AMENDMENT-4）：L0 四輪 index 逐位相同、cosine 最大差 ≤ 1e-6、2 類內 argmax 全同 | "
            f"十折四任務皆通過；cosine 最大差 {S['check6_l0_cos_maxabs']:.1e} | 符合 |",
            f"| 選出的 r\\* | r\\* = {rs}（{fid('nc15.rstar')}） | {'符合門檻' if S['r_star_meets'] else '**未符合**（取兩序較低者最高的 r）'} |"]
    for c in R["checks"]:
        extra = f"；與 nc8/per_fold.json 逐折最大差 {c['nc8_per_fold_maxabs']:.1e}" if "nc8_per_fold_maxabs" in c else ""
        out.append(f"| 一致性：{c['key']} = {c['expected']:.4f}（{c['source']}） | {c['got']:.4f}{extra} | "
                   f"{'符合' if c['ok'] else '**不符**'} |")
    out.append("")

    # ── 表一 ──
    out += ["## 表一 validation WP 與選 r（十折四任務平均，mean ± sd）", "",
            f"門檻 = {thr:.4f}（L0 validation {S['l0_val_mean']:.4f} − 0.01）；兩序都 ≥ 門檻才算過（PREREG-15 操作定義 4）。", "",
            "| r | reverse validation WP | paper validation WP | reverse ≥ 0.9190 | paper ≥ 0.9190 | 是否過門檻 | fact-id |",
            "|---|---|---|---|---|---|---|"]
    for r in (1, 2, 3):
        w = {o: S["wp_val"][f"r{r}.{o}"] for o in ORDERS}
        star = " ★" if r == rs else ""
        out.append(f"| {r}{star} | {w['reverse']['mean']:.4f} ± {w['reverse']['sd']:.4f} | "
                   f"{w['paper']['mean']:.4f} ± {w['paper']['sd']:.4f} | {'✓' if w['reverse']['pass'] else '✗'} | "
                   f"{'✓' if w['paper']['pass'] else '✗'} | {'過' if S['r_pass'][str(r)] else '未過'} | "
                   f"{fid(f'nc15.val.r{r}.reverse.wp')}；{fid(f'nc15.val.r{r}.paper.wp')} |")
    out += ["", f"★ = r\\* = {rs}（符合門檻者中最小的 r；同分取較小的 r）。" if S["r_star_meets"] else
            f"★ = r\\* = {rs}（三個候選都未過門檻，取兩序較低者最高的 r，標記未符合）。", "",
            "每任務 validation WP（十折平均）：", "",
            "| r | 序 | " + " | ".join(TASKS) + " |", "|---|---|" + "---|" * 4]
    for r in (1, 2, 3):
        for o in ORDERS:
            pt = S["wp_val"][f"r{r}.{o}"]["per_task"]
            out.append(f"| {r} | {o} | " + " | ".join(f"{sum(pt[t]) / len(pt[t]):.4f}" for t in TASKS) + " |")
    out += ["", "各序第一個任務（reverse：esca；paper：lung）使用凍結的底座（L0），該任務的數值與 r 無關。", ""]

    # ── 表二 ──
    out += [f"## 表二 LoRA 對照版 L1(r\\* = {rs}) 與修正頭（test、t = 4、十折 mean ± sd）", ""]
    if rs == 2:
        out += ["**r = 2 已由 validation 選定。** 既有數字不變：task-known 0.9401／0.9339（REPORT_stage5.md:43、:44）、"
                "task-inferred 0.9176／0.9132（REPORT_stage10.md:15、:28），本輪重算四位相同（見判準落點）。", ""]
    out += ["差距 = 修正頭 − LoRA 對照版；容許範圍：差距 ≥ −0.005（PREREG-15 操作定義 10）。", "",
            "| 指標 | 序 | LoRA 對照版 L1(r\\*) | 修正頭 I6(r = 2) | 差距 | 在 0.005 內 | 修正頭較高折數 | fact-id |",
            "|---|---|---|---|---|---|---|---|"]
    for kind, name in (("known", "task-known WP"), ("inferred", "task-inferred ACC")):
        for o in ORDERS:
            g = R["gap"][f"{kind}.{o}"]
            out.append(f"| {name} | {o} | {ms(f'lora.r{rs}.{kind}.{o}')} | {ms(f'head.{kind}.{o}')} | {g['gap']:+.4f} | "
                       f"{'是' if g['within_tol'] else '否'} | {g['head_higher']}/10（平 {g['ties']}） | "
                       f"{fid(f'nc15.lora.r{rs}.{kind}.{o}.acc')}；{fid(f'nc15.head.{kind}.{o}.acc')}；"
                       f"{fid(f'nc15.gap.{kind}.{o}')} |")
    if rs != 2:
        out += ["", "參考（既有 r = 2，本輪重算）：", "",
                "| 指標 | 序 | L1(r = 2) | fact-id |", "|---|---|---|---|"]
        for kind, name in (("known", "task-known WP"), ("inferred", "task-inferred ACC")):
            for o in ORDERS:
                out.append(f"| {name} | {o} | {ms(f'lora.r2.{kind}.{o}')} | {fid(f'nc15.lora.r2.{kind}.{o}.acc')} |")
    out += ["", "逐折差距（修正頭 − LoRA 對照版 L1(r\\*)）：", "",
            "| 指標 | 序 | " + " | ".join(str(f) for f in range(1, 11)) + " |", "|---|---|" + "---|" * 10]
    for kind, name in (("known", "task-known"), ("inferred", "task-inferred")):
        for o in ORDERS:
            out.append(f"| {name} | {o} | " + " | ".join(f"{x:+.4f}" for x in R["gap"][f"{kind}.{o}"]["per_fold"]) + " |")
    out += ["", f"L1(r\\*) 每任務 task-known test WP（十折平均）：", "",
            "| 序 | " + " | ".join(TASKS) + " |", "|---|" + "---|" * 4]
    pf = json.loads((own / "per_fold.json").read_text())
    for o in ORDERS:
        kt = pf[f"lora.r{rs}.known_task.{o}"]
        out.append(f"| {o} | " + " | ".join(f"{sum(x[p] for x in kt) / len(kt):.4f}" for p in range(4)) + " |")

    # ── 耗時 ──
    out += ["", "## 耗時（秒，wall clock；執行緒 8）", "", "| 項目 | 秒 |", "|---|---|",
            f"| L1(r = 3) 訓練（兩序 × 十折 × 3 任務 × 5 epochs） | {tr_s:,.0f} |",
            f"| L1(r = 3) 附帶 test 評估（兩序 × 十折） | {ev_s:,.0f} |",
            f"| validation 評估（十折，r = 1、2、3 × 兩序同時） | {va_s:,.0f} |",
            f"| test 讀取與重算（nc15_test.py） | {R['seconds']:,.0f} |", "",
            "逐張讀檔／計算秒數：`nc15/val/fold*.pt`、`nc15/lora_r3/*/fold*_eval.pt` 的 `t_read_s`、`t_compute_s`（本機，未 commit）。", ""]
    (base / "REPORT_stage17.md").write_text("\n".join(out))
    print(f"→ {base / 'REPORT_stage17.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
