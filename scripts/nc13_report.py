#!/usr/bin/env python3
"""NC-13 報告（PREREG-13）：修正頭的貢獻（g = 0，只重新推論）。

只讀 nc13/fold*.pt（scripts/nc13_head_contrib.py 的輸出）、bottleneck_fold1.json，以及 NC-8 快取的 test／train mean_vec
（AR 分派器；selector/incremental_ridge.py）。CIL 定義同 scripts/nc8_report.py（nc5_report.cil_full、nc2_report.hard）。
一致性檢查（PREREG-13）不符即停，不寫報告。

    NAVCIL_MACHINE=mac python scripts/nc13_report.py --cache-dir <主工作樹>/outputs/navcil/mac/cache
輸出：outputs/navcil/<machine>/REPORT_stage15.md、nc13/result.json、nc13/facts.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
import nc5_report as N5                                                   # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS                                       # noqa: E402
from selector.incremental_ridge import IncrementalRidge                   # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

FOLDS = list(range(1, 11))
ARMS = ("a", "b", "c", "d")
ARM_NAME = {"a": "(a) s0 + g，4 輪 × 16 去重（現行）", "b": "(b) g = 0，4 輪 × 16 去重",
            "c": "(c) s0 + g，一次 top-64", "d": "(d) g = 0，一次 top-64"}
MODES = {"known": "task-known（oracle 分派）", "inferred": "task-inferred（AR 分派）"}
NC8_ROW = {"known": "8 oracle＋I6(r=2)（上限）", "inferred": "6 D3：AR＋I6(r=2)（主系統）"}
REPORT10_LINE = {("known", "reverse"): 18, ("known", "paper"): 31, ("inferred", "reverse"): 16, ("inferred", "paper"): 29}
NC11_COMMIT = "bf488ff"                    # ws3-ablation：REPORT_stage13（尚未合併；git show 唯讀）
NC11_EXPECT = {"known": "0.9372", "inferred": "0.9160"}
GAMMA = 1e-3
OVER_PAIRS = ("ab", "cd", "ac", "bd")


def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"


def stop(msg: str):
    raise SystemExit(f"一致性檢查不符，停止：{msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    args = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_config()
    tasks = list(cfg["tasks"])
    mac = REPO_ROOT / "outputs" / "navcil" / cfg["machine"]
    out = mac / "nc13"
    recs = {f: torch.load(out / f"fold{f}.pt", map_location="cpu") for f in FOLDS}
    te = {(f, t): torch.load(args.cache_dir / f"nc8_fold{f}_test_{t}.pt", map_location="cpu") for f in FOLDS for t in tasks}
    tr = {(f, t): torch.load(args.cache_dir / f"nc8_fold{f}_train_{t}.pt", map_location="cpu")["mean_vec"] for f in FOLDS for t in tasks}
    for (f, t), c in te.items():
        if not torch.equal(recs[f]["tasks"][t]["labels"], c["labels"]):
            stop(f"fold {f} {t} 的 labels 與 NC-8 快取不同")

    # ── 分派（與 arm 無關，先算好）：AR 第 t 階段只讀任務 t 的 train mean_vec ──
    th_cache = {}

    def dispatch(f, seen, mode, task_vec, mv):
        if mode == "known":
            return task_vec
        key = (f, tuple(seen))
        if key not in th_cache:
            ridge = IncrementalRidge(GAMMA)
            for p in seen:
                ridge.add_task(p, tr[(f, tasks[p])])
            th_cache[key] = torch.tensor(ridge.task_ids)[ridge.scores(mv).argmax(-1)]
        return th_cache[key]

    def stage_fn(f, arm, mode):
        def stage(seen):
            parts = [recs[f]["tasks"][tasks[p]] for p in seen]
            labels = torch.cat([x["labels"] for x in parts])
            task_vec = torch.cat([torch.full((len(x["labels"]),), p) for p, x in zip(seen, parts)])
            four = torch.cat([x["cos8"][arm] for x in parts])
            mv = torch.cat([te[(f, tasks[p])]["mean_vec"] for p in seen])
            th = dispatch(f, seen, mode, task_vec, mv)
            pred, mk = N2.hard(four, th, task_vec)
            return pred, mk, {"labels": labels, "task": task_vec}
        return stage

    R = {a: {m: {o: [N5.cil_full(stage_fn(f, a, m), o, tasks) for f in FOLDS] for o in ORDERS} for m in MODES} for a in ARMS}
    acc = lambda a, m, o: [x["acc"] for x in R[a][m][o]]          # noqa: E731

    # ── 一致性檢查 ──
    maxabs = max(recs[f]["tasks"][t]["cos8_a_vs_nc8_maxabs"] for f in FOLDS for t in tasks)
    if maxabs > 1e-5:
        stop(f"(a) 8 類 cosine 與 NC-8 快取最大差 {maxabs:.2e}")
    nc8 = json.loads((mac / "nc8" / "per_fold.json").read_text())["rows"]
    repro = {}
    for m in MODES:
        for o in ORDERS:
            ref = [x["acc"] for x in nc8[NC8_ROW[m]][o]]
            now = acc("a", m, o)
            if any(round(x, 4) != round(y, 4) for x, y in zip(now, ref)):
                stop(f"(a) {m} {o} 逐折與 nc8/per_fold.json 不同")
            want = {"known": "0.9340 ± 0.0202", "inferred": "0.9128 ± 0.0258"}[m]
            if fmt(*mean_sd(now)) != want:
                stop(f"(a) {m} {o} = {fmt(*mean_sd(now))}，應為 {want}")
            repro[f"a|{m}|{o}"] = {"base": f"REPORT_stage10.md:{REPORT10_LINE[(m, o)]}", "base_value": want, "now": fmt(*mean_sd(now)),
                                  "per_fold_4dp_equal": 10, "max_abs_diff": max(abs(x - y) for x, y in zip(now, ref))}
    nc11 = json.loads(subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{NC11_COMMIT}:outputs/navcil/mac/nc11/select/result.json"],
                                     check=True, capture_output=True, text=True).stdout)["arms"]["oneshot64"]
    for m in MODES:
        for o in ORDERS:
            ref = [x["acc"] for x in nc11[m][o]]
            now = acc("c", m, o)
            if any(round(x, 4) != round(y, 4) for x, y in zip(now, ref)):
                stop(f"(c) {m} {o} 逐折與 REPORT_stage13（{NC11_COMMIT} nc11/select/result.json）不同")
            if f"{mean_sd(now)[0]:.4f}" != NC11_EXPECT[m]:
                stop(f"(c) {m} {o} = {mean_sd(now)[0]:.4f}，應為 {NC11_EXPECT[m]}")
            repro[f"c|{m}|{o}"] = {"base": f"REPORT_stage13.md:215（{NC11_COMMIT}）", "base_value": NC11_EXPECT[m],
                                  "now": fmt(*mean_sd(now)), "per_fold_4dp_equal": 10,
                                  "max_abs_diff": max(abs(x - y) for x, y in zip(now, ref))}

    # ── 配對與主表 ──
    def compare(x, y):
        d = [p - q for p, q in zip(x, y)]
        w = sum(v > 0 for v in d); ties = sum(v == 0 for v in d); lo = sum(v < 0 for v in d)
        pw = float(stats.wilcoxon(x, y).pvalue) if any(d) else float("nan")
        pt = float(stats.ttest_rel(x, y).pvalue) if any(d) else float("nan")
        return {"diff_per_fold": d, "mean_diff": sum(d) / len(d), "wins": w, "ties": ties, "losses": lo, "p_wilcoxon": pw, "p_ttest": pt}

    main_tab = {a: {m: {o: {"per_fold": acc(a, m, o), "mean": mean_sd(acc(a, m, o))[0], "sd": mean_sd(acc(a, m, o))[1],
                            "vs_a": compare(acc(a, m, o), acc("a", m, o))} for o in ORDERS} for m in MODES} for a in ARMS}
    paired = {f"{x}_vs_{y}": {m: {o: compare(acc(x, m, o), acc(y, m, o)) for o in ORDERS} for m in MODES} for x, y in (("a", "b"), ("c", "d"))}
    prediction = paired["a_vs_b"]["known"]["reverse"]["wins"] >= 8
    overlap = {}
    for k in OVER_PAIRS:
        per = [sum(float(recs[f]["tasks"][t]["overlap"][k].sum()) for t in tasks) /
               sum(len(recs[f]["tasks"][t]["overlap"][k]) for t in tasks) for f in FOLDS]
        overlap[k] = {"per_fold": per, "mean": mean_sd(per)[0], "sd": mean_sd(per)[1]}
    bott = json.loads((out / "bottleneck_fold1.json").read_text())
    # 另報（非判準）：(d) 的 task-known 與 NC-1 (b) zero-shot 一次 top-64 的逐折值（REPORT_stage1-3.md T1-c）比對
    r13 = (mac / "REPORT_stage1-3.md").read_text().splitlines()
    nc1_b = {}
    for ln, line in enumerate(r13, 1):
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 5 and c[0].isdigit() and c[4] in ("✓", "✗") and int(c[0]) in FOLDS and int(c[0]) not in nc1_b:
            nc1_b[int(c[0])] = (float(c[1]), ln)
    d_known = main_tab["d"]["known"]["reverse"]["per_fold"]
    nc1_match = {"n_equal": sum(round(d_known[f - 1], 4) == nc1_b[f][0] for f in FOLDS),
                 "lines": f"{min(v[1] for v in nc1_b.values())}-{max(v[1] for v in nc1_b.values())}"}
    timing = json.loads((out / "timing.json").read_text())
    n_slides = sum(len(recs[f]["tasks"][t]["labels"]) for f in FOLDS for t in tasks)
    t_read = sum(float(recs[f]["tasks"][t]["t_read_s"].sum()) for f in FOLDS for t in tasks)
    t_comp = sum(float(recs[f]["tasks"][t]["t_compute_s"].sum()) for f in FOLDS for t in tasks)

    # ── facts.json（一行一條，報告以行號引用）──
    facts = []

    def fact(fid, value, definition, sd=None):
        facts.append({"id": fid, "value": value, **({"sd": sd} if sd is not None else {}), "definition": definition})
        return fid
    for a in ARMS:
        for m in MODES:
            for o in ORDERS:
                t_ = main_tab[a][m][o]
                fact(f"nc13.{a}.{m}.{o}.acc", round(t_["mean"], 6), f"{ARM_NAME[a]}；{MODES[m]}；{o}；t = 4 ACC 十折平均", round(t_["sd"], 6))
    for pk, pv in paired.items():
        for m in MODES:
            for o in ORDERS:
                c = pv[m][o]
                fact(f"nc13.paired.{pk}.{m}.{o}.mean_diff", round(c["mean_diff"], 6), f"{pk}，{MODES[m]}，{o}：逐折差平均")
                fact(f"nc13.paired.{pk}.{m}.{o}.p_wilcoxon", c["p_wilcoxon"], f"{pk}，{MODES[m]}，{o}：Wilcoxon 雙尾 p")
                fact(f"nc13.paired.{pk}.{m}.{o}.p_ttest", c["p_ttest"], f"{pk}，{MODES[m]}，{o}：paired t 雙尾 p")
                fact(f"nc13.paired.{pk}.{m}.{o}.wins", c["wins"], f"{pk}，{MODES[m]}，{o}：前者較高的折數")
    for k, v in overlap.items():
        fact(f"nc13.overlap.{k}", round(v["mean"], 4), f"自家修正頭：arm {k[0]} 與 arm {k[1]} 所選 64 個 patch 的重疊數（每折平均後十折平均）", round(v["sd"], 4))
    for t, b in bott.items():
        k = t.replace("tcga_", "")
        fact(f"nc13.bottleneck.{k}.pearson", round(b["pearson_h1_h2"], 4), f"fold 1 {t} 修正頭：全部 test patch 上 h1 與 h2 的 Pearson 相關")
        fact(f"nc13.bottleneck.{k}.frac_h1_gt3", round(b["frac_h1_gt3"], 4), f"fold 1 {t}：h1 > 3 的 patch 比例")
        fact(f"nc13.bottleneck.{k}.frac_h2_gt3", round(b["frac_h2_gt3"], 4), f"fold 1 {t}：h2 > 3 的 patch 比例")
        fact(f"nc13.bottleneck.{k}.frac_both_gt3", round(b["frac_both_gt3"], 4), f"fold 1 {t}：h1、h2 同時 > 3 的 patch 比例")
    fact("nc13.check.cos8_a_vs_nc8_maxabs", maxabs, "(a) 的 8 類 cosine 與 NC-8 快取 I6_cos8 的逐張最大絕對差")
    lines = ["["] + [json.dumps(x, ensure_ascii=False) + ("," if i < len(facts) - 1 else "") for i, x in enumerate(facts)] + ["]"]
    (out / "facts.json").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LN = {x["id"]: i + 2 for i, x in enumerate(facts)}
    ref = lambda fid: f"`{fid}`（nc13/facts.json:{LN[fid]}）"          # noqa: E731

    result = {"prereg": "PREREG-13.md", "arms": {a: {m: {o: R[a][m][o] for o in ORDERS} for m in MODES} for a in ARMS},
              "main": main_tab, "paired": paired, "prediction_met": prediction, "overlap": overlap, "bottleneck": bott,
              "repro": repro, "nc1_b_vs_d_known": nc1_match, "cos8_a_vs_nc8_maxabs": maxabs, "n_slides": n_slides, "t_read_s": t_read, "t_compute_s": t_comp}
    (out / "result.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float) + "\n")
    write(mac, main_tab, paired, prediction, overlap, bott, repro, maxabs, timing, n_slides, t_read, t_comp, ref,
          time.perf_counter() - t0, nc1_match)
    print(f"→ {mac / 'REPORT_stage15.md'}")
    return 0


def pp(x):
    return f"{x * 100:+.2f}"


def pval(p):
    return "—" if p != p else (f"{p:.4f}" if p >= 1e-3 else f"{p:.2e}")


def write(mac, T, P, prediction, O, B, repro, maxabs, timing, n_slides, t_read, t_comp, ref, t_rep, nc1_match):
    rv = "reverse"
    g_known = T["a"]["known"][rv]["mean"] - T["b"]["known"][rv]["mean"]
    g_inf = T["a"]["inferred"][rv]["mean"] - T["b"]["inferred"][rv]["mean"]
    g1_known = T["c"]["known"][rv]["mean"] - T["d"]["known"][rv]["mean"]
    g1_inf = T["c"]["inferred"][rv]["mean"] - T["d"]["inferred"][rv]["mean"]
    ab = P["a_vs_b"]["known"][rv]
    colinear = [t.replace("tcga_", "").upper() for t, b in B.items() if abs(b["pearson_h1_h2"]) >= 0.95]
    other = [f"{t.replace('tcga_', '').upper()} {b['pearson_h1_h2']:+.2f}" for t, b in B.items() if abs(b["pearson_h1_h2"]) < 0.95]
    both = "、".join(f"{t.replace('tcga_', '').upper()} {b['frac_both_gt3'] * 100:.0f}%" for t, b in B.items())
    out = [
        "# REPORT — NC-13：修正頭的貢獻（修正量設為 0，只重新推論；Mac CPU，十折兩序）", "",
        "機器：mac（Apple M1 Pro）、CPU、`torch.set_num_threads(8)`、torch 2.11.0；四個 arm 與附加分析在同一次執行中完成"
        "（log：`logs/nc13_head.log`）。判準與定義見 `PREREG-13.md`。不訓練：修正頭沿用既有 I6(r = 2) 權重，分派器為 AR（γ = 1e-3，"
        "累加統計量）。task-known = oracle 分派、task-inferred = AR 分派，指標為 t = 4 CIL ACC（四任務等權平均），十折。"
        "數字附 fact-id 與其所在的 facts.json 行號（路徑相對於 `outputs/navcil/mac/`）。另一個 session 同時在本機執行，耗時僅供參考。", "",
        "## 0 白話摘要", "",
        f"在現行的挑法（4 輪 × 16、去重）下，把修正量拿掉（只用 zero-shot 分數 s0）後，task-known 由 {T['a']['known'][rv]['mean']:.4f} "
        f"變為 {T['b']['known'][rv]['mean']:.4f}，也就是修正頭帶來 {pp(g_known)} pp（{ref('nc13.a.known.reverse.acc')}、"
        f"{ref('nc13.b.known.reverse.acc')}）；task-inferred 帶來 {pp(g_inf)} pp。十折中加修正量較高的折數為 {ab['wins']}／10"
        f"（平 {ab['ties']}、低 {ab['losses']}），事先登記的方向性預測（至少 8 折）{'成立' if prediction else '不成立'}。"
        f"改用一次挑 top-64 時，修正頭帶來 task-known {pp(g1_known)} pp、task-inferred {pp(g1_inf)} pp。",
        "",
        f"瓶頸的兩個維度（h1、h2）：fold 1 各任務修正頭在全部 test patch 上的相關係數見 T6（以 |r| ≥ 0.95 描述「幾乎同向」，"
        "這是描述用的門檻，不是事先登記的判準）。"
        f"{'、'.join(colinear) if colinear else '沒有任務'} 的 |r| ≥ 0.95，兩個維度幾乎同向，實際上接近只用一個方向；"
        f"{'、'.join(other) if other else '其餘任務'}{'，兩個維度並未重合。' if other else ''}"
        f"另外，h1、h2 同時大於 3 的 patch 比例為 {both}；在這個區間 GELU 近似恆等，修正量近似為 u 的單一線性函數"
        "（(w2₁A₁ + w2₂A₂)· u 加上常數），也就是說不論相關係數多少，這些 patch 上的修正量都只沿一個方向變化（由 GELU 定義推導）。",
        "",
        "## 1 一致性檢查（PREREG-13）", "",
        "| 項目 | 基準 | 基準值 | 本輪 | 結果 |", "|---|---|---|---|---|",
    ]
    for k, v in repro.items():
        a, m, o = k.split("|")
        out.append(f"| ({a}) {MODES[m]}，{o} | {v['base']} | {v['base_value']} | {v['now']}；逐折四位相同 10/10（最大差 "
                   f"{v['max_abs_diff']:.1e}） | 符合 |")
    out += ["", f"- (a) 的 8 類 cosine 與 NC-8 快取 `I6_cos8` 的逐張最大絕對差：{maxabs:.2e}（{ref('nc13.check.cos8_a_vs_nc8_maxabs')}）。",
            f"- REPORT_stage13 位於分支 `ws3-ablation` @ {NC11_COMMIT}（尚未合併）；(c) 的逐折基準以 `git show` 讀取該 commit 的 "
            "`nc11/select/result.json`（`oneshot64`）。",
            f"- 另報（非判準）：(d) g = 0、一次 top-64 的 task-known 逐折值與 NC-1 第一關 (b) zero-shot top-64 在小數點後四位相同 "
            f"{nc1_match['n_equal']}/10（REPORT_stage1-3.md:{nc1_match['lines']}，十折平均 REPORT_stage1-3.md:38）。s0 是最大 cosine 的片內 "
            "z-score，排序與最大 cosine 相同，所以一次 top-64 選到同一組 patch。", "",
            "## 2 主表（t = 4、十折 mean ± sd；reverse 序，t = 4 時兩序逐折相同，兩序分列見 T2）", "",
            "| arm | task-known | task-inferred | 與 (a) 的差（pp，known／inferred） | 高於／低於 (a) 的折數（known；inferred） | fact-id |",
            "|---|---|---|---|---|---|"]
    for a in ARMS:
        k, i = T[a]["known"][rv], T[a]["inferred"][rv]
        out.append(f"| {ARM_NAME[a]} | {fmt(k['mean'], k['sd'])} | {fmt(i['mean'], i['sd'])} | "
                   f"{pp(k['mean'] - T['a']['known'][rv]['mean'])}／{pp(i['mean'] - T['a']['inferred'][rv]['mean'])} | "
                   f"{k['vs_a']['wins']}／{k['vs_a']['losses']}；{i['vs_a']['wins']}／{i['vs_a']['losses']} | "
                   f"{ref(f'nc13.{a}.known.reverse.acc')}；{ref(f'nc13.{a}.inferred.reverse.acc')} |")
    out += ["", "## T2 兩序分列", "", "| arm | known reverse | known paper | inferred reverse | inferred paper |", "|---|---|---|---|---|"]
    for a in ARMS:
        out.append(f"| {ARM_NAME[a]} | " + " | ".join(fmt(T[a][m][o]["mean"], T[a][m][o]["sd"]) for m in MODES for o in ORDERS) + " |")
    out += ["", "## T3 每折值（reverse）", ""]
    for m in MODES:
        out += [f"**{MODES[m]}**", "", "| 折 | " + " | ".join(f"({a})" for a in ARMS) + " | (a) − (b) | (c) − (d) |",
                "|" + "---|" * (len(ARMS) + 3)]
        for i, f in enumerate(FOLDS):
            v = [T[a][m][rv]["per_fold"][i] for a in ARMS]
            out.append(f"| {f} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {v[0] - v[1]:+.4f} | {v[2] - v[3]:+.4f} |")
        out.append("")
    out += ["## T4 配對比較（Wilcoxon signed-rank 雙尾、paired t 雙尾）", "",
            "`scipy.stats.wilcoxon`（zero_method = \"wilcox\"，n = 10 無平手時為精確檢定，最小可能 p = 2／1024 ≈ 0.0020）、"
            "`scipy.stats.ttest_rel`。共 8 次檢定（2 組比較 × 2 種分派 × 2 序），未做多重比較校正；t = 4 時兩序逐折相同，兩序各列一次。", "",
            "| 比較 | 分派 | 序 | 平均差（pp） | 前者 勝／平／負 | Wilcoxon p | paired t p | fact-id |", "|---|---|---|---|---|---|---|---|"]
    for pk in P:
        x, y = pk.split("_vs_")
        for m in MODES:
            for o in ORDERS:
                c = P[pk][m][o]
                out.append(f"| ({x}) vs ({y}) | {MODES[m]} | {o} | {pp(c['mean_diff'])} | {c['wins']}／{c['ties']}／{c['losses']} | "
                           f"{pval(c['p_wilcoxon'])} | {pval(c['p_ttest'])} | {ref(f'nc13.paired.{pk}.{m}.{o}.p_wilcoxon')}；"
                           f"{ref(f'nc13.paired.{pk}.{m}.{o}.p_ttest')} |")
    out += ["", f"方向性預測（PREREG-13）：(a) 的 task-known 高於 (b) 的折數 = {ab['wins']}／10（門檻 ≥ 8）→ "
            f"**{'成立' if prediction else '不成立'}**（{ref('nc13.paired.a_vs_b.known.reverse.wins')}）。", "",
            "## T5 附加分析 1：修正量對選取的影響（自家修正頭，所選 64 個 patch 的重疊數）", "",
            "每折先對該折全部 test slides（四任務）平均，再取十折 mean ± sd。", "",
            "| 比較 | 重疊數 | fact-id |", "|---|---|---|"]
    lab = {"ab": "(a) vs (b)：4 輪 × 16，有／無修正量", "cd": "(c) vs (d)：一次 top-64，有／無修正量",
           "ac": "(a) vs (c)：有修正量，兩種挑法", "bd": "(b) vs (d)：無修正量，兩種挑法"}
    for k, v in O.items():
        out.append(f"| {lab[k]} | {v['mean']:.2f} ± {v['sd']:.2f}（共 64） | {ref(f'nc13.overlap.{k}')} |")
    out += ["", "## T6 附加分析 2：瓶頸是否退化（fold 1，自家修正頭，該任務全部 test slides 的所有 patch）", "",
            "h = A u + b1（`selector/i6_expert.py:35-45`）；GELU 線性區以 h > 3 計（GELU(3) ≈ 2.996）。", "",
            "| 任務 | patch 數 | Pearson r(h1, h2) | h1 > 3 | h2 > 3 | 兩者皆 > 3 | w2 | fact-id |", "|---|---|---|---|---|---|---|---|"]
    for t, b in B.items():
        k = t.replace("tcga_", "")
        out.append(f"| {t} | {b['n_patches']:,} | {b['pearson_h1_h2']:+.4f} | {b['frac_h1_gt3']:.3f} | {b['frac_h2_gt3']:.3f} | "
                   f"{b['frac_both_gt3']:.3f} | ({b['w2'][0]:+.4f}, {b['w2'][1]:+.4f}) | {ref(f'nc13.bottleneck.{k}.pearson')} |")
    out += ["", "## T7 耗時（秒，wall clock；與其他工作同時執行）", "", "| 項目 | 秒 |", "|---|---|",
            f"| 推論與附加分析（十折，{n_slides:,} 張 test slide） | {timing.get('run/1-10', float('nan')):.0f} |",
            f"| 其中讀檔（逐張加總） | {t_read:.0f} |", f"| 其中計算（逐張加總） | {t_comp:.0f} |", f"| 報告 | {t_rep:.0f} |", ""]
    (mac / "REPORT_stage15.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
