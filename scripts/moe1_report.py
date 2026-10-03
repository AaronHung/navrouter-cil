#!/usr/bin/env python3
"""MOE-1 報告（PREREG-17 細則 35）：只讀各階段的 <階段>.json，不重算；只含已完成的階段。

    NAVCIL_MACHINE=mac python scripts/moe1_report.py [--out moe1]
輸出：--out moe1 → outputs/navcil/<machine>/REPORT_moe1.md；其他（冒煙測試）→ <out>/REPORT_smoke.md（不是結果）。
每一節各自 try：某一節格式化失敗只影響該節。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.text_encoder import load_config                             # noqa: E402

TASK = ["ESCA", "RCC", "BRCA", "LUNG"]
ORDERS = ["reverse", "paper"]
STAGES = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]
TITLE = {"s1": "S1 修好與弄壞", "s2": "S2 CL 流程", "s3": "S3 訓練／推論落差 2×2", "s4": "S4 多 seed 與 ensemble 對照",
         "s5": "S5 視角 × 判讀 2×2", "s6": "S6 gate", "s7": "S7 seed 45／46 與隨機特徵"}
SYS = {"main": "主系統", "M3": "M3（σ 固定版）", "M3re": "σ 重算版", "G1": "G1（每任務 β）", "G2": "G2（每張 slide 的 ρ）"}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def sd(xs):
    xs = list(xs)
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0


def ms(xs) -> str:
    return f"{mean(xs):.4f} ± {sd(xs):.4f}"


def f4(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def f0(x) -> str:
    return "—" if x is None else f"{x:.0f}"


def fp(p) -> str:
    return "不適用" if p is None else f"{p:.4f}"


def pf(xs) -> str:
    return " ".join(f"{x:+.4f}" for x in xs)


def yes(b) -> str:
    return "滿足" if b else "不滿足"


def col(recs, k):
    return [r[k] for r in recs]


def edge(g) -> str:
    """選到網格邊界時註明。"""
    return "（選在網格邊界）" if g["star"] in (min(g["grid"]), max(g["grid"])) else ""


def prow(name, p, n) -> str:
    return f"| {name} | {p['mean']:+.4f} | {p['wins']}/{n} | {p['losses']}/{n} | {fp(p['p'])} | {pf(p['per_fold'])} |"


PHEAD = ["| 比較 | 平均差 | 贏折數（差 > 0） | 輸折數 | Wilcoxon 雙尾 p | 逐折差 |", "|---|---|---|---|---|---|"]


# ── 門檻與 K ────────────────────────────────────────────────────────────────
def gates(J) -> list[str]:
    out = ["## 門檻（PREREG-17「門檻」；細則 21、22、31）", ""]
    s4, s6 = J.get("s4"), J.get("s6")
    out += ["### G-M3c（S4；seed 43、44 × 兩序，四個條件都滿足才通過）", ""]
    if s4:
        g = s4["G-M3c"]
        out += ["| seed | 序 | CIL ACC(M3) − CIL ACC(同 seed 主系統) 十折平均 | 差 > 0 的折數 | 門檻 | 結果 |", "|---|---|---|---|---|---|"]
        for c in g["conditions"]:
            out.append(f"| {c['seed']} | {c['order']} | {c['mean']:+.4f} | {c['wins']} | ≥ +0.007 且 ≥ 7 折 | {yes(c['pass'])} |")
        out += [f"| **G-M3c** | | | | 四個條件 | **{'通過' if g['pass'] else '未通過'}**{'' if g['valid'] else '（非十折，不作判定）'} |", ""]
    else:
        out += ["S4 尚未完成或失敗：G-M3c 無結果。", ""]
    out += ["### G-ENS（S4；每折 D = 三組 (s, s′) 的 [WP(LG(s)) − WP(LL(s, s′))] 平均；兩序都滿足才通過）", ""]
    if s4:
        g = s4["G-ENS"]
        out += ["| 序 | D 的十折平均 | D > 0 的折數 | 門檻 | 結果 | 逐折 D |", "|---|---|---|---|---|---|"]
        for o in ORDERS:
            x = g["orders"][o]
            out.append(f"| {o} | {x['mean']:+.4f} | {x['wins']} | ≥ +0.005 且 ≥ 7 折 | {yes(x['pass'])} | {pf(x['per_fold'])} |")
        out += [f"| **G-ENS** | | | 兩序 | **{'通過' if g['pass'] else '未通過'}**{'' if g['valid'] else '（非十折，不作判定）'} | |", ""]
    else:
        out += ["S4 尚未完成或失敗：G-ENS 無結果。", ""]
    out += ["### G-GATE（S6；代表 = 訓練資料上平均正確率較高者，同分取 G1；兩序都滿足才通過）", ""]
    if s6:
        g = s6["G-GATE"]
        out += ["| 序 | 訓練正確率 G1 | 訓練正確率 G2 | 代表 | test CIL ACC(代表) − CIL ACC(M3) 十折平均 | 差 > 0 的折數 | 門檻 | 結果 |",
                "|---|---|---|---|---|---|---|---|"]
        for o in ORDERS:
            x = g["orders"][o]
            out.append(f"| {o} | {x['train_acc']['G1']:.4f} | {x['train_acc']['G2']:.4f} | {x['representative']} | {x['mean']:+.4f} | "
                       f"{x['wins']} | ≥ +0.005 且 ≥ 7 折 | {yes(x['pass'])} |")
        out += [f"| **G-GATE** | | | | | | 兩序 | **{'通過' if g['pass'] else '未通過'}**{'' if g['valid'] else '（非十折，不作判定）'} |", ""]
    else:
        out += ["S6 尚未完成或失敗：G-GATE 無結果。", ""]
    out += ["S1、S2、S3、S5、S7 不設門檻。seed 42 的 test 數字在 MOE-0 已看過，本批 seed 42 的 M3 結果只作描述，不作確認。", ""]
    return out


def kchecks(J) -> list[str]:
    out = ["## 一致性檢查 K1–K3（PREREG-17「一致性檢查」；細則 6–8）", "",
           "| 檢查 | 階段 | 結果 | 判準 | 通過 |", "|---|---|---|---|---|"]
    for s in STAGES:
        k = (J.get(s) or {}).get("K1")
        if k:
            out.append(f"| K1 主系統 seed 42 重算 | {s.upper()}{'（現行格，由特徵檔重算）' if s == 's3' else ''} | 逐折最大絕對差 {k['max_abs_fold_diff']:.2e}；"
                       f"WP {k['wp_mean']['reverse']:.4f}／{k['wp_mean']['paper']:.4f}；CIL ACC {k['cil_mean']['reverse']:.4f}／{k['cil_mean']['paper']:.4f} | "
                       f"≤ 1e-9；0.9340；0.9128 | {'是' if k['pass'] else '否'} |")
    k = (J.get("s2") or {}).get("K2")
    if k:
        out.append(f"| K2 σ 重算版 t = 4 對 moe0 的 z′、λ = 1.0 | S2 | 逐折最大絕對差 {k['max_abs_fold_diff']:.2e}；Masked ACC "
                   f"{k['masked_mean']['reverse']:.4f}／{k['masked_mean']['paper']:.4f}；CIL ACC {k['cil_mean']['reverse']:.4f}／{k['cil_mean']['paper']:.4f} | "
                   f"≤ 1e-9；0.9448；0.9226 | {'是' if k['pass'] else '否'} |")
    for s in ("s4", "s6", "s7"):
        k = (J.get(s) or {}).get("K3")
        if k:
            out.append(f"| K3 seed 42 重訓 (fold 1, tcga_esca) | {s.upper()} | 逐位元相同：{'、'.join(f'{a} {b}' for a, b in k['bitwise_equal'].items())}；"
                       f"最大絕對差 {k['max_abs_diff']:.2e} | 每個張量 torch.equal | {'是' if k['pass'] else '否'} |")
    out += ["", "兩個數字並列者為 reverse／paper。K 不過的階段不會出現在上表（見「失敗的階段」）。", ""]
    return out


# ── S1 ──────────────────────────────────────────────────────────────────────
def sec_s1(r) -> list[str]:
    g = r["groups"]
    out = ["落點：PREREG-17 S1、細則 11–12。seed 42、test、告訴任務；M3 = σ 重算版（t = 4、reverse 序）。", "",
           f"張數 {r['n_slides']}；主系統錯 {r['main_wrong_total']}、M3 錯 {r['m3_wrong_total']}。CSV 中 d_a = 0 的列數 {r['csv_zero_da_rows']}；"
           f"以快取未捨入值重算的判定與 CSV 不同的張數：主系統 {r['mismatch_vs_cache']['main']}、M3 {r['mismatch_vs_cache']['M3']}。", "",
           "T1.1 每個類別的張數（十折合計）：", "", "| 類別 | n | " + " | ".join(g) + " |", "|---|---|---|---|---|---|"]
    for c, name in enumerate(r["class_names"]):
        x = r["counts_by_class"][c]
        out.append(f"| {name} | {sum(x)} | " + " | ".join(str(v) for v in x) + " |")
    tot = [sum(r["counts_by_class"][c][i] for c in range(8)) for i in range(4)]
    out += [f"| 合計 | {sum(tot)} | " + " | ".join(str(v) for v in tot) + " |", "", "每任務：", "",
            "| 任務 | n | " + " | ".join(g) + " |", "|---|---|---|---|---|---|"]
    for p in range(4):
        x = r["counts_by_task"][p]
        out.append(f"| {TASK[p]} | {sum(x)} | " + " | ".join(str(v) for v in x) + " |")
    out += ["", "T1.2 四組的統計（中位數；σ = t = 4 的 validation 重算值）：", "",
            "| 組 | 張數 | patch 數 N | \\|d_a/σ_a\\| | \\|d_b/σ_b\\| | AR 第一名 − 第二名 | TP 正確比例 |", "|---|---|---|---|---|---|---|"]
    for s in r["group_stats"]:
        out.append(f"| {s['group']} | {s['n']} | {f0(s['median_n_patch'])} | {f4(s['median_abs_za'])} | "
                   f"{f4(s['median_abs_zb'])} | {f4(s['median_ar_margin'])} | {f4(s['tp_correct'])} |")
    return out + [""]


# ── S2 ──────────────────────────────────────────────────────────────────────
def sec_s2(r) -> list[str]:
    out = ["落點：PREREG-17 S2、細則 13–15。seed 42，只推論。ACC 只在已學類別中判（TP = 該階段的 AR）；"
           "Masked ACC：主系統 = Table 1 的 Masked ACC 欄，融合系統 = 告訴任務的融合判定（細則 5）。"
           f"主系統的 `cil_full` 結果（ACC、Masked、Forgetting、BWT、逐階段 ACC）與 `nc8/per_fold.json` 逐折最大絕對差 {r['main_cil_full_vs_nc8_max_diff']:.2e}。", ""]
    for o in ORDERS:
        d = r["orders"][o]
        out += [f"### {o}", "", "T2.1 逐階段 ACC／Masked ACC（mean ± sd）與 Forgetting、BWT：", "",
                "| 系統 | 指標 | t = 1 | t = 2 | t = 3 | t = 4 | Forgetting | BWT |", "|---|---|---|---|---|---|---|---|"]
        for k in ("main", "M3", "M3re"):
            x = d[k]
            out.append(f"| {SYS[k]} | ACC | " + " | ".join(ms([v["acc_t"][t] for v in x]) for t in range(4)) +
                       f" | {ms(col(x, 'forgetting'))} | {ms(col(x, 'bwt'))} |")
            out.append(f"| {SYS[k]} | Masked ACC | " + " | ".join(ms([v["masked_t"][t] for v in x]) for t in range(4)) + " | | |")
        out += ["", "T2.2 每任務 WP（告訴任務；mean ± sd；未學為 —）：", "", "| 系統 | 任務 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|"]
        for k in ("main", "M3", "M3re"):
            w = d["wp_task_t"][k]
            for p in range(4):
                cells = [ms([x[t][p] for x in w]) if w[0][t][p] is not None else "—" for t in range(4)]
                out.append(f"| {SYS[k]} | {TASK[p]} | " + " | ".join(cells) + " |")
        out += ["", "T2.3 舊任務的整片 expert：(b) 單獨的告訴任務正確率，與 σ_b 重算值／固定值（mean ± sd；t < 學習階段為 —）：", "",
                "| 任務 | 量 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|"]
        for p in range(4):
            for key, name in (("b_alone_acc", "(b) 單獨正確率"), ("sigma_ratio", "σ_b 重算值／固定值")):
                cells = [ms([x[key][p][t] for x in d["drift"]]) if d["drift"][0][key][p][t] is not None else "—" for t in range(4)]
                out.append(f"| {TASK[p]} | {name} | " + " | ".join(cells) + " |")
        g = d["gamma"]
        out += ["", "T2.4 γ 統一：", "",
                "(i) |AR_j − (LIN8_{2j} + LIN8_{2j+1})|（兩者 γ = 1e-3；test 全部 slide、已學任務；十折最大值）："
                + "、".join(f"t = {t + 1}：{max(x['ar_eq_lin8sum_maxabs_t'][t] for x in g):.2e}" for t in range(4)) + "（預期 < 1e-8）。", "",
                "(ii) t = 4：", "", "| LIN8 的 γ | (b) 單獨 WP | M3 WP（= Masked ACC） | M3 CIL ACC |", "|---|---|---|---|"]
        for key, name in (("g0.01", "0.01（現行）"), ("g0.001", "1e-3")):
            out.append(f"| {name} | {ms([x[key]['b_alone_wp'] for x in g])} | {ms([x[key]['M3']['wp'] for x in g])} | {ms([x[key]['M3']['cil'] for x in g])} |")
        out += ["", "用 LIN8（γ = 0.01）兩類分數和當 TP（t = 4）：", "", "| TP | TP 正確率 | 主系統 CIL ACC | M3 CIL ACC |", "|---|---|---|---|",
                f"| AR（γ = 1e-3；現行） | {ms([x['tp_lin8sum']['tp_ar'] for x in g])} | {ms(col(d['main'], 'acc'))} | {ms(col(d['M3'], 'acc'))} |",
                f"| LIN8（γ = 0.01）兩類分數和 | {ms([x['tp_lin8sum']['tp'] for x in g])} | {ms([x['tp_lin8sum']['cil_main'] for x in g])} | "
                f"{ms([x['tp_lin8sum']['cil_M3'] for x in g])} |", ""]
    return out


# ── S3 ──────────────────────────────────────────────────────────────────────
def sec_s3(r) -> list[str]:
    c = r["check_now_cell"]
    n = len(r["folds"])
    out = ["落點：PREREG-17 S3、細則 16–17。seed 42，只推論；四個 head 都由特徵檔重算。"
           f"現行格（四輪／等權）與 moe0 快取的 8 類 cosine 最大絕對差 {c['max_abs_cos_diff']:.2e}、2 類內 argmax {'全同' if c['argmax_all_equal'] else '不同'}；"
           f"現行格重現 K1（見上）。validation {r['n_val']} 張、test {r['n_test']} 張；逐張記錄加總：讀檔 {r['t_read_s']:.0f} 秒、計算 {r['t_compute_s']:.0f} 秒。", ""]
    for split in ("val", "test"):
        out += [f"### {'validation' if split == 'val' else 'test'}", "",
                "| 選片／彙整 | WP（告訴任務） | CIL ACC reverse | CIL ACC paper | WP − 現行格：平均（贏折數） | CIL − 現行格 reverse | CIL − 現行格 paper |",
                "|---|---|---|---|---|---|---|"]
        R, Pp = r["splits"][split]["reverse"], r["splits"][split]["paper"]
        for i, name in enumerate(r["cell_names"]):
            v, vp = R["vs_now"][i], Pp["vs_now"][i]
            out.append(f"| {name} | {ms(col(R['cells'][i], 'wp'))} | {ms(col(R['cells'][i], 'cil'))} | {ms(col(Pp['cells'][i], 'cil'))} | "
                       f"{v['wp']['mean']:+.4f}（{v['wp']['wins']}/{n}） | {v['cil']['mean']:+.4f}（{v['cil']['wins']}/{n}） | "
                       f"{vp['cil']['mean']:+.4f}（{vp['cil']['wins']}/{n}） |")
        out += ["", "逐折差（該格 − 現行格；WP；reverse 序的 CIL ACC）：", "", "| 選片／彙整 | WP 逐折差 | CIL ACC 逐折差（reverse） |", "|---|---|---|"]
        for i, name in enumerate(r["cell_names"]):
            out.append(f"| {name} | {pf(R['vs_now'][i]['wp']['per_fold'])} | {pf(R['vs_now'][i]['cil']['per_fold'])} |")
        out.append("")
    return out


# ── S4／S7 的每 seed 列 ─────────────────────────────────────────────────────
def seed_tables(rows, seeds, n) -> list[str]:
    out = []
    for o in ORDERS:
        out += [f"### {o}", "", "| seed | 系統 | WP（告訴任務） | Masked ACC | CIL ACC |", "|---|---|---|---|---|"]
        for s in seeds:
            x = rows[str(s)][o]
            out.append(f"| {s} | 主系統 | {ms(col(x['main'], 'wp'))} | {ms(col(x['main'], 'mk'))} | {ms(col(x['main'], 'cil'))} |")
            out.append(f"| {s} | M3（σ 固定版） | {ms(col(x['M3'], 'wp'))} | {ms(col(x['M3'], 'mk'))} | {ms(col(x['M3'], 'cil'))} |")
        out += ["", "M3 − 同 seed 主系統（逐折相減）：", ""] + PHEAD
        for s in seeds:
            p = rows[str(s)][o]["paired"]
            out += [prow(f"seed {s}：WP", p["wp"], n), prow(f"seed {s}：Masked ACC", p["mk"], n), prow(f"seed {s}：CIL ACC", p["cil"], n)]
        out.append("")
    return out


def sec_s4(r) -> list[str]:
    n = len(r["folds"])
    out = ["落點：PREREG-17 S4、細則 18–22。seed 43、44 以與 nc7_i6.py 相同的設定重訓（K3 見上）；seed 42 取 moe0 快取。"
           "M3 的 Masked ACC = 其告訴任務的 WP（細則 5）。seed 42 的列只作描述。", "", "T4.1 每個 seed 的主系統與 M3（t = 4；mean ± sd）：", ""]
    out += seed_tables(r["rows"], r["seeds"], n)
    out += ["T4.2 ensemble 對照（t = 4、σ 固定版；增益 = 該列 − seed s 的主系統）：", ""]
    for o in ORDERS:
        out += [f"### {o}", "", "| 組合 | 相對的主系統 | WP | CIL ACC | WP 增益：平均（贏折數） | CIL 增益：平均（贏折數） | WP 增益逐折 | CIL 增益逐折 |",
                "|---|---|---|---|---|---|---|---|"]
        for e in r["ensemble"][o]:
            g = e["gain"]
            out.append(f"| {e['name']} | seed {e['vs_seed']} | {ms(col(e['abs'], 'wp'))} | {ms(col(e['abs'], 'cil'))} | "
                       f"{g['wp']['mean']:+.4f}（{g['wp']['wins']}/{n}） | {g['cil']['mean']:+.4f}（{g['cil']['wins']}/{n}） | "
                       f"{pf(g['wp']['per_fold'])} | {pf(g['cil']['per_fold'])} |")
        out.append("")
    d = r["disagree"]
    out += ["T4.3 三個 seed 的主系統彼此判定不同的張數（告訴任務、test、十折合計）：", "",
            f"共 {d['n']} 張；三者不全相同 **{d['any']}** 張；兩兩不同：" + "、".join(f"{k} {v} 張" for k, v in d["pairs"].items()) + "。", ""]
    return out


# ── S5 ──────────────────────────────────────────────────────────────────────
def sec_s5(r) -> list[str]:
    n = len(r["folds"])
    g = r["lr_gamma"]
    out = ["落點：PREREG-17 S5、細則 23–26。seed 42、告訴任務。GT = mean_vec 與文字的 cosine 差；GR = LIN8（γ = 0.01）；LT = 主系統；LR = 四輪等權向量的 ridge。", "",
           "T5.1 LR 的 γ（十折 validation 平均 WP 選，同分取小）：" + "、".join(f"γ = {k}：{v:.4f}" for k, v in g["val_wp"].items()) + f"；選定 γ\\* = **{g['star']}**{edge(g)}。", ""]
    for split in ("val", "test"):
        s = r["splits"][split]
        out += [f"### {'validation' if split == 'val' else 'test'}", "", "T5.2 四者單獨的 WP：", "",
                "| 判讀器 | WP（十折 mean ± sd） | 正確張數／總張數（十折合計） | ESCA | RCC | BRCA | LUNG |", "|---|---|---|---|---|---|---|"]
        for k in r["names"]:
            a = s["alone"][k]
            out.append(f"| {k} | {ms(a['wp'])} | {a['correct']}／{a['n']} | " + " | ".join(f4(mean(x[p] for x in a["wp_t"])) for p in range(4)) + " |")
        out += ["", "T5.3 兩兩組合的張數（十折合計）：", "",
                "| 前者＋後者 | 都對 | 只有前者對 | 只有後者對 | 都錯 | 至少一個對（合計比例） | 至少一個對（每折每任務計、四任務等權、十折平均） |",
                "|---|---|---|---|---|---|---|"]
        for name, p in s["pairs"].items():
            tot = p["both"] + p["only_first"] + p["only_second"] + p["neither"]
            out.append(f"| {name} | {p['both']} | {p['only_first']} | {p['only_second']} | {p['neither']} | {(tot - p['neither']) / tot:.4f} | {mean(p['union_fold']):.4f} |")
        out += ["", "T5.4 融合（各自除以該任務 validation 上的 σ 後等權相加）的 WP，與相對 LT 的逐折差：", "",
                "| 組合 | WP | − LT：平均 | 贏折數 | 逐折差 |", "|---|---|---|---|---|",
                f"| LT（單獨） | {ms(s['alone']['LT']['wp'])} | | | |"]
        for name, x in s["fusion"].items():
            v = x["vs_LT"]
            out.append(f"| {name} | {ms(x['wp'])} | {v['mean']:+.4f} | {v['wins']}/{n} | {pf(v['per_fold'])} |")
        out.append("")
    out += ["T5.5 LR 在 CIL 下（TP = AR；v 取自 τ̂ 的 head；在 τ̂ 的兩類內判；test）：", "",
            "| 序 | LR 的 CIL ACC | 主系統 CIL ACC | LR − 主系統：平均 | 贏折數 | 逐折差 |", "|---|---|---|---|---|---|"]
    for o in ORDERS:
        x = r["lr_cil"][o]
        out.append(f"| {o} | {ms(x['cil'])} | {ms(x['main_cil'])} | {x['vs_main']['mean']:+.4f} | {x['vs_main']['wins']}/{n} | {pf(x['vs_main']['per_fold'])} |")
    return out + [""]


# ── S6 ──────────────────────────────────────────────────────────────────────
def sec_s6(r) -> list[str]:
    n = len(r["folds"])
    out = ["落點：PREREG-17 S6、細則 27–32。seed 42；gate 的訓練資料 = 當前任務的 train slides（d_a 來自沒看過該張的 head、d_b 為 leave-one-out；"
           f"共 {r['n_gate_train_slides']} 張次）；test 用原本的 head。G1／G2 的 Masked ACC = 其告訴任務的 WP（細則 5）。", ""]
    for o in ORDERS:
        d = r["orders"][o]
        out += [f"### {o}", "", "T6.1 test（t = 4；mean ± sd）：", "", "| 系統 | WP（告訴任務） | Masked ACC | CIL ACC | gate 訓練資料上的正確率 |", "|---|---|---|---|---|"]
        for k in ("main", "M3", "G1", "G2"):
            x = d["systems"][k]
            out.append(f"| {SYS[k].replace('（σ 固定版）', '（固定 β = 1）')} | {ms(col(x, 'wp'))} | {ms(col(x, 'mk'))} | {ms(col(x, 'cil'))} | {d['train_acc'][k]:.4f} |")
        out += ["", "T6.2 相對 M3（固定 β = 1）的逐折相減：", ""] + PHEAD
        for k in ("G1", "G2"):
            for m, name in (("wp", "WP"), ("mk", "Masked ACC"), ("cil", "CIL ACC")):
                out.append(prow(f"{k} − M3：{name}", d["vs_M3"][k][m], n))
        for m, name in (("wp", "WP"), ("mk", "Masked ACC"), ("cil", "CIL ACC")):
            out.append(prow(f"（參考）M3 − 主系統：{name}", d["vs_M3"]["M3_vs_main"][m], n))
        out += ["", "T6.3 每任務選到的 β_j（十折中各值的次數）與 G2 的 c（十折平均）、test 上的 ρ：", "",
                "| 任務 | " + " | ".join(f"β = {b:g}" for b in r["betas"]) + " | c0 | c1 | c2 | ρ 第 25 百分位 | ρ 中位數 | ρ 第 75 百分位 |",
                "|---|" + "---|" * (len(r["betas"]) + 6)]
        for q in range(4):
            bc, cm, rq = d["beta_counts"][q], d["c_mean"][q], d["rho_test"][q]
            out.append(f"| {TASK[q]} | " + " | ".join(str(bc[str(b)]) for b in r["betas"]) +
                       f" | {cm[0]:+.4f} | {cm[1]:+.4f} | {cm[2]:+.4f} | {rq['q25']:.4f} | {rq['median']:.4f} | {rq['q75']:.4f} |")
        u = d["upper_LT_GR_union"]
        ev = [x["g2_evals"] for x in d["fit"]]
        out += ["", f"參考上限（S5 的 LT＋GR「至少一個對」；test、告訴任務）：十折合計比例 {u['total']:.4f}；每折每任務計、四任務等權、十折平均 {u['fold_task_mean']:.4f}。"
                f"G2 的 L-BFGS 函數評估次數：最大 {max(ev)}、中位 {sorted(ev)[len(ev) // 2]}。自行累加的 LIN8 權重與 `B8.W` 最大絕對差 {d['lin8_W_max_dev_vs_B8']:.2e}；"
                f"leave-one-out 的 h 最大值 {d['loo_h_max']:.4f}。", ""]
    return out


# ── S7 ──────────────────────────────────────────────────────────────────────
def sec_rf(rf, n) -> list[str]:
    g, b = rf["gamma"], rf["bytes"]
    out = ["T7.1 隨機特徵 ridge（φ = [ReLU(mean_vec · R); 1]，R 512 × 2048、N(0,1)、generator seed 42）：γ 以十折 validation 平均 WP 選（同分取小）："
           + "、".join(f"γ = {k}：{v:.4f}" for k, v in g["val_wp"].items()) + f"；選定 γ\\* = **{g['star']}**{edge(g)}。", "",
           f"統計矩陣大小：A {b['dim']} × {b['dim']} float64 = {b['A_float64']:,} bytes（fp32 換算 {b['A_fp32']:,}）；"
           f"B {b['dim']} × 8 float64 = {b['B_float64']:,} bytes（fp32 換算 {b['B_fp32']:,}）；R（由 seed 重建）float64 {b['R_float64']:,} bytes。", ""]
    for o in ORDERS:
        d = rf["orders"][o]
        out += [f"### {o}", "", "| 列 | WP（告訴任務） | CIL ACC（TP = AR） | TP 正確率 | CIL ACC（TP = 隨機特徵兩類分數和） |", "|---|---|---|---|---|",
                f"| 主系統 | {ms(col(d['main'], 'wp'))} | {ms(col(d['main'], 'cil'))} | AR {ms(d['tp_ar'])} | {ms(d['cil_main_rftp'])} |",
                f"| 隨機特徵 ridge 單獨 | {ms(d['alone'])} | — | 隨機特徵 {ms(d['tp_rf'])} | — |",
                f"| M3（(b) = LIN8，σ 固定版） | {ms(col(d['M3'], 'wp'))} | {ms(col(d['M3'], 'cil'))} | | |",
                f"| M3（(b) 換成隨機特徵，σ 固定版） | {ms(col(d['M3rf'], 'wp'))} | {ms(col(d['M3rf'], 'cil'))} | | {ms(d['cil_M3rf_rftp'])} |", ""] + PHEAD
        out += [prow("M3（隨機特徵）− 主系統：WP", d["M3rf_vs_main"]["wp"], n), prow("M3（隨機特徵）− 主系統：CIL ACC", d["M3rf_vs_main"]["cil"], n),
                prow("M3（隨機特徵）− M3（LIN8）：WP", d["M3rf_vs_M3"]["wp"], n), prow("M3（隨機特徵）− M3（LIN8）：CIL ACC", d["M3rf_vs_M3"]["cil"], n), ""]
    return out


def sec_s7(r) -> list[str]:
    n = len(r["folds"])
    out = ["落點：PREREG-17 S7、細則 33–34。不設門檻。", ""] + sec_rf(r["rf"], n)
    out += ["T7.2 seed 45、46 的主系統與 M3（同 S4 的 T4.1）：", ""]
    if r["seed_part"]["ran"]:
        out += seed_tables(r["rows"], r["seeds"], n)
    else:
        out += [f"未執行：{r['seed_part']['reason']}", ""]
    return out


SEC = {"s1": sec_s1, "s2": sec_s2, "s3": sec_s3, "s4": sec_s4, "s5": sec_s5, "s6": sec_s6, "s7": sec_s7}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="moe1")
    a = ap.parse_args()
    base = REPO_ROOT / "outputs" / "navcil" / load_config()["machine"]
    root = base / a.out
    smoke = a.out != "moe1"
    J, status = {}, {}
    for s in STAGES:
        if (root / f"{s}.done").exists() and (root / f"{s}.json").exists():
            J[s], status[s] = json.loads((root / f"{s}.json").read_text()), "完成"
        elif (root / f"FAILED_{s}.txt").exists():
            status[s] = "失敗"
        else:
            status[s] = "尚未完成"
    folds = sorted({tuple(v["folds"]) for v in J.values()})
    head = [("# REPORT（冒煙測試，不是結果）— " if smoke else "# REPORT — ") +
            "MOE-1：M3 確認、ensemble 對照、CL 流程、訓練／推論落差、視角 × 判讀 2×2、gate（Mac CPU，十折，reverse 與 paper 兩序）", "",
            "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；closed-form 一律 float64；四任務等權。"
            "所有數字來自同一台、同一批（`scripts/moe1_run_all.sh`；seed 42 的 I6 四輪 cosine 與 mean_vec 沿用 MOE-0 快取，S3 由特徵檔重算並比對）。"
            "判準與操作定義見 `PREREG-17.md`；前置 `AMENDMENT-5.md`。", "",
            "主系統 = AR（γ = 1e-3）＋ I6(r = 2)；M3：f = d_a/σ_a + d_b/σ_b（β = 1.0，σ 固定版）。", "",
            "各階段狀態：" + "、".join(f"{s.upper()} {status[s]}" for s in STAGES) +
            f"。已完成階段的 folds：{'；'.join(str(list(f)) for f in folds) if folds else '—'}。本報告只含已完成的階段。", ""]
    body = gates(J) + kchecks(J)
    for s in STAGES:
        body += [f"## {TITLE[s]}", ""]
        if s in J:
            try:
                body += SEC[s](J[s])
            except Exception:
                body += ["本節格式化失敗（數值仍在 `" + f"{a.out}/{s}.json" + "`）：", "", "```", traceback.format_exc().strip(), "```", ""]
        elif s == "s7" and (root / "s7_rf.json").exists():
            try:
                body += ["S7 尚未完成（seed 45／46 的部分未跑完）；隨機特徵的部分已完成：", ""] + \
                        sec_rf(json.loads((root / "s7_rf.json").read_text()), len(folds[0]) if folds else 10)
            except Exception:
                body += ["```", traceback.format_exc().strip(), "```", ""]
        else:
            body += [f"{status[s]}。", ""]
    # 失敗的階段
    body += ["## 失敗的階段", ""]
    fails = sorted(root.glob("FAILED_*.txt"))
    if not fails:
        body += ["無。", ""]
    for p in fails:
        lines = p.read_text().strip().splitlines()
        body += [f"### {p.name}", "", "```"] + lines[:3] + (["…"] if len(lines) > 9 else []) + lines[3:][-6:] + ["```", ""]
    # DECISIONS
    dec = base / "moe1" / "DECISIONS.md"
    body += ["## DECISIONS.md（執行前與執行中的判斷）", "", dec.read_text().strip() if dec.exists() else "（無）", ""]
    # 耗時
    body += ["## 各階段實際耗時", "", "| 階段 | 開始 | 結束 | 秒（run_all 計） | 結束碼 | 階段內計時（秒） |", "|---|---|---|---|---|---|"]
    tsv = root / "STAGE_TIMES.tsv"
    if tsv.exists():
        for line in tsv.read_text().strip().splitlines():
            x = line.split("\t")
            body.append(f"| {x[0].upper()} | {x[1]} | {x[2]} | {x[3]} | {x[4]} | {J[x[0]]['wall_s'] if x[0] in J else '—'} |")
    else:
        for s in J:
            body.append(f"| {s.upper()} | — | {J[s]['finished']} | — | 0 | {J[s]['wall_s']} |")
    body.append("")
    path = (root / "REPORT_smoke.md") if smoke else (base / "REPORT_moe1.md")
    path.write_text("\n".join(head + body))
    print("→", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
