#!/usr/bin/env python3
"""MOE-2 報告（PREREG-18 細則 27）：只讀各階段的 <階段>.json（E7 另讀 moe1/s2.json），不重算；只含已完成的階段。

    NAVCIL_MACHINE=mac python scripts/moe2_report.py [--out moe2]
輸出：--out moe2 → outputs/navcil/<machine>/REPORT_moe2.md；其他（冒煙測試）→ <out>/REPORT_smoke.md（不是結果）。
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from moe1_report import PHEAD, col, f4, mean, ms, pf, prow, sd, yes   # noqa: E402,F401
from selector.text_encoder import load_config                          # noqa: E402

TASK = ["ESCA", "RCC", "BRCA", "LUNG"]
ORDERS = ["reverse", "paper"]
SEEDS = [42, 43, 44, 45, 46]
STAGES = ["vec", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8"]
TITLE = {"vec": "向量快取（vec）", "e1": "E1 多 seed 確認", "e2": "E2 是否需要 head", "e3": "E3 一次取 64", "e4": "E4 與整片向量合併",
         "e5": "E5 類別加權", "e6": "E6 每類別", "e7": "E7 CL 流程", "e8": "E8 儲存"}
BASE = None                                     # outputs/navcil/<machine>


def gtext(g, name) -> str:
    edge = "（選在網格邊界；不延伸）" if g["edge"] else ""
    return (f"{name} 的 γ（seed 42、validation、reverse、t = 4、告訴任務 WP 十折平均；同分取較大）："
            + "、".join(f"γ = {k}：{v:.4f}" for k, v in g["val_wp"].items()) + f"；選定 γ\\* = **{g['star']:g}**{edge}。")


def pt(xs) -> str:
    return " | ".join(f4(mean(x[p] for x in xs)) for p in range(4))


def wl(p, n) -> str:
    return f"{p['mean']:+.4f}（{p['wins']}/{n}）"


def osame(x) -> str:
    return (f"告訴任務判定不同 {x['tell_diff']} 張、CIL 判定不同 {x['cil_diff']} 張（共 {x['n']} 張）；告訴任務 d 的最大絕對差 {x['d_maxabs']:.2e}")


# ── 門檻與 K ────────────────────────────────────────────────────────────────
def gates(J) -> list[str]:
    out = ["## 門檻（PREREG-18「門檻」；細則 10、17、18、20）", ""]
    e1, e2, e4 = J.get("e1"), J.get("e2"), J.get("e4")
    valid = lambda g: "" if g["valid"] else "（非十折，不作判定）"   # noqa: E731
    out += ["### G-LR（E1；seed 43–46 × 兩序，八個條件都滿足才通過）", ""]
    if e1:
        g = e1["G-LR"]
        out += ["| seed | 序 | CIL ACC(LR) − CIL ACC(同 seed 主系統) 十折平均 | 差 > 0 的折數 | 門檻 | 結果 |", "|---|---|---|---|---|---|"]
        for c in g["conditions"]:
            out.append(f"| {c['seed']} | {c['order']} | {c['mean']:+.4f} | {c['wins']} | ≥ +0.007 且 ≥ 7 折 | {yes(c['pass'])} |")
        out += [f"| **G-LR** | | | | 八個條件 | **{'通過' if g['pass'] else '未通過'}**{valid(g)} |", ""]
    else:
        out += ["E1 尚未完成或失敗：G-LR 無結果。", ""]
    for key, st, thr, name, desc in (("G-HEAD", e2, "+0.005", "E2", "D_head = 五個 seed 的 WP(LR(s)) 平均 − WP(LR0)"),
                                     ("G-CAT", e4, "+0.005", "E4", "每折 seed 43–46 的 [WP(LRG(s)) − WP(LR(s))] 平均")):
        out += [f"### {key}（{name}；{desc}；兩序都滿足才通過）", ""]
        if st:
            g = st[key]
            out += ["| 序 | 十折平均 | > 0 的折數 | 門檻 | 結果 | 逐折值 |", "|---|---|---|---|---|---|"]
            for o in ORDERS:
                x = g["orders"][o]
                out.append(f"| {o} | {x['mean']:+.4f} | {x['wins']} | ≥ {thr} 且 ≥ 7 折 | {yes(x['pass'])} | {pf(x['per_fold'])} |")
            out += [f"| **{key}** | | | 兩序 | **{'通過' if g['pass'] else '未通過'}**{valid(g)} | |", ""]
            if key == "G-HEAD":
                out += ["G-HEAD 通過代表 ridge 判讀下仍需要 head。", ""]
        else:
            out += [f"{name} 尚未完成或失敗：{key} 無結果。", ""]
    out += ["E3、E5、E6、E7、E8 不設門檻。seed 42 的 LR test 數字在 MOE-1 已看過，只作描述。", ""]
    return out


def kchecks(J) -> list[str]:
    out = ["## 一致性檢查 K1–K3（PREREG-18「一致性檢查」；細則 12–14）", "",
           "| 檢查 | 階段 | 結果 | 判準 | 通過 |", "|---|---|---|---|---|"]
    for s in STAGES:
        k = (J.get(s) or {}).get("K1")
        if k:
            out.append(f"| K1 主系統 seed 42 | {s.upper()} | 逐折最大絕對差 {k['max_abs_fold_diff']:.2e}；"
                       f"WP {k['wp_mean']['reverse']:.4f}／{k['wp_mean']['paper']:.4f}；CIL ACC {k['cil_mean']['reverse']:.4f}／{k['cil_mean']['paper']:.4f} | "
                       f"≤ 1e-9；0.9340；0.9128 | {'是' if k['pass'] else '否'} |")
    e1 = J.get("e1") or {}
    if "K2" in e1:
        k = e1["K2"]
        out.append(f"| K2 LR（seed 42、γ = 1e-3）對 moe1/s5.json | E1 | 逐折最大絕對差 {k['max_abs_fold_diff']:.2e}；WP {k['wp_mean']:.4f}；"
                   f"CIL ACC {k['cil_mean']['reverse']:.4f}／{k['cil_mean']['paper']:.4f} | ≤ 1e-9；0.9457；0.9252 | {'是' if k['pass'] else '否'} |")
    if "K3" in e1:
        k = e1["K3"]
        det = "；".join(f"seed {s}（{k['source'][s]}.json）{v['reverse']['max_abs']:.0e}／{v['paper']['max_abs']:.0e}" for s, v in k["per_seed"].items())
        out.append(f"| K3 seed 43–46 主系統 CIL ACC 對 moe1 | E1 | 逐折最大絕對差 {k['max_abs_fold_diff']:.2e}（{det}） | ≤ 1e-9 | {'是' if k['pass'] else '否'} |")
    v = J.get("vec")
    if v:
        w = v["worst"]
        out.append(f"| 向量快取檢查（細則 4 (a)–(e)） | VEC | (a) {w['a_s42_train_maxabs']:.2e}；(b) {w['b_seed_cos_maxabs']:.2e}；(c) {w['c_one42_cos_maxabs']:.2e}；"
                   f"(d) {w['d_g0_cos_maxabs']:.2e}；argmax 全同；slide id 全同 | ≤ 1e-6；全同 | 是 |")
    out += ["", "兩個數字並列者為 reverse／paper。K 不過的階段不會出現在上表（見「失敗的階段」）。", ""]
    return out


# ── 各節 ────────────────────────────────────────────────────────────────────
def sec_vec(r) -> list[str]:
    n = r["n_slides"]
    tot = {s: sum(v for k, v in n.items() if k.startswith(s)) for s in ("train", "val", "test")}
    return ["落點：PREREG-18 細則 2–4。seed 43–46 的 v(s)、v0、v1(42) 由特徵檔新算；seed 42 的 v 沿用 MOE-1 快取。", "",
            f"head 權重：{r['heads']['n_files']} 個檔案都在（缺 0）。張數（各折合計）：train {tot['train']}、validation {tot['val']}、test {tot['test']}；"
            f"逐張記錄加總：讀檔 {r['t_read_s']:.0f} 秒、計算 {r['t_compute_s']:.0f} 秒。", "",
            "| 檢查（細則 4） | 各折最大值 | 判準 |", "|---|---|---|",
            f"| (a) train 的 v(42) 重算 vs `s42v_train` | {r['worst']['a_s42_train_maxabs']:.2e} | ≤ 1e-6 |",
            f"| (b) seed 43–46 的 v·Fᵀ vs moe1 `seed{{s}}` 快取的 `I6_cos8` | {r['worst']['b_seed_cos_maxabs']:.2e}（argmax 全同） | ≤ 1e-6、全同 |",
            f"| (c) v1(42)·Fᵀ vs `s42cells` 一次取 64／等權格 | {r['worst']['c_one42_cos_maxabs']:.2e}（argmax 全同） | ≤ 1e-6、全同 |",
            f"| (d) test 的 v0·Fᵀ vs moe0 `g0_cos8` | {r['worst']['d_g0_cos_maxabs']:.2e}（argmax 全同） | ≤ 1e-6、全同 |", ""]


def sec_e1(r) -> list[str]:
    n = len(r["folds"])
    S = r["seeds"]
    out = ["落點：PREREG-18 E1、細則 7–11、17。test、t = 4；每序用該序自己的累加與 AR。M3 = σ 固定版（WP = Masked ACC）。seed 42 的列只作描述。", "",
           gtext(r["gamma"], "LR"), "",
           "T1.1 LR、主系統、M3（mean ± sd）：", "",
           "| seed | 序 | LR WP | LR CIL ACC | 主系統 WP | 主系統 CIL ACC | M3 WP | M3 CIL ACC |", "|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        for o in ORDERS:
            x = S[str(s)]["orders"][o]
            out.append(f"| {s} | {o} | {ms(col(x['LR'], 'wp'))} | {ms(col(x['LR'], 'cil'))} | {ms(col(x['main'], 'wp'))} | "
                       f"{ms(col(x['main'], 'cil'))} | {ms(col(x['M3'], 'wp'))} | {ms(col(x['M3'], 'cil'))} |")
    out += ["", "T1.2 LR 的每任務 WP（十折平均）：", "", "| seed | 序 | ESCA | RCC | BRCA | LUNG |", "|---|---|---|---|---|---|"]
    for s in SEEDS:
        for o in ORDERS:
            out.append(f"| {s} | {o} | {pt(col(S[str(s)]['orders'][o]['LR'], 'wp_t'))} |")
    for o in ORDERS:
        out += ["", f"T1.3 逐折相減（{o}）：", ""] + PHEAD
        for s in SEEDS:
            x = S[str(s)]["orders"][o]
            out += [prow(f"seed {s}：LR − 主系統 WP", x["vs_main"]["wp"], n), prow(f"seed {s}：LR − 主系統 CIL ACC", x["vs_main"]["cil"], n),
                    prow(f"seed {s}：LR − M3 WP", x["vs_M3"]["wp"], n), prow(f"seed {s}：LR − M3 CIL ACC", x["vs_M3"]["cil"], n)]
    out += ["", "T1.4 LR 在 t = 4 的兩序結果是否逐張相同（細則 10）：", "", "| seed | 比較 |", "|---|---|"]
    for s in SEEDS:
        out.append(f"| {s} | {osame(S[str(s)]['order_same'])} |")
    out += ["", "M3 的逐折 CIL ACC 與 moe1 的 s4.json／s7.json 最大絕對差（細則 11，不是 K）：" +
            "、".join(f"seed {s} {v:.2e}" for s, v in r["m3_vs_moe1_max_abs"].items()) + "。", ""]
    return out


def sec_e2(r) -> list[str]:
    n = len(r["folds"])
    L = r["LR0"]
    out = ["落點：PREREG-18 E2、細則 18。v0 = g = 0（只用 s0）的四輪等權向量；CIL 時取 τ̂ 的 s0。", "", gtext(r["gamma_LR0"], "LR0"), "",
           f"LR 的 γ\\* = {r['gamma_LR']['star']:g}（同 E1）。", "",
           "T2.1 LR0（test、t = 4；mean ± sd；每任務為十折平均）：", "",
           "| 序 | WP | CIL ACC | ESCA | RCC | BRCA | LUNG |", "|---|---|---|---|---|---|---|"]
    for o in ORDERS:
        out.append(f"| {o} | {ms(col(L[o], 'wp'))} | {ms(col(L[o], 'cil'))} | {pt(col(L[o], 'wp_t'))} |")
    out += ["", f"LR0 兩序：{osame(r['order_same_LR0'])}。", "",
            "T2.2 D_head（每折 = 五個 seed 的 WP(LR(s)) 平均 − WP(LR0)）：", "",
            "| 序 | 列 | " + " | ".join(f"折 {f}" for f in r["folds"]) + " | 平均 | > 0 的折數 |", "|---|---|" + "---|" * (n + 2)]
    for o in ORDERS:
        avg = [mean(r["wp_LR"][str(s)][o][i] for s in SEEDS) for i in range(n)]
        lr0 = col(L[o], "wp")
        dh = r["D_head"][o]
        out += [f"| {o} | 五個 seed 的 WP(LR) 平均 | " + " | ".join(f"{x:.4f}" for x in avg) + f" | {mean(avg):.4f} | |",
                f"| {o} | WP(LR0) | " + " | ".join(f"{x:.4f}" for x in lr0) + f" | {mean(lr0):.4f} | |",
                f"| {o} | D_head | " + " | ".join(f"{x:+.4f}" for x in dh) + f" | {mean(dh):+.4f} | {sum(x > 1e-12 for x in dh)}/{n} |"]
    g = r["LT_g0"]
    out += ["", "T2.3 LT 在 g = 0（test、真實任務的 v0 與兩類文字的 cosine、2 類 argmax）：", "",
            "| 列 | WP（十折 mean ± sd） | ESCA | RCC | BRCA | LUNG |", "|---|---|---|---|---|---|",
            f"| LT（g = 0；本批重算） | {ms(g['wp'])} | " + " | ".join(f4(x) for x in g["wp_t_mean"]) + " |",
            f"| MOE-0 B2「g = 0（只用 s0）」列（results.json） | — | " + " | ".join(f4(x) for x in g["b2_ref"]) + " |",
            "", f"每任務十折平均與 B2 的最大絕對差 {g['b2_max_abs']:.2e}" + ("" if g["b2_comparable"] else "（非十折，不可比）") +
            "（描述性比對，不是 K；cosine 層級的檢查見細則 4(d)）。", ""]
    return out


def sec_e3(r) -> list[str]:
    n = len(r["folds"])
    out = ["落點：PREREG-18 E3、細則 19。seed 42；v1 = head 的 s 前 64（不扣冗餘）、等權平均。不設門檻。", "",
           gtext(r["gamma_LR1"], "LR1"), "", f"LR 的 γ\\* = {r['gamma_LR']['star']:g}。", "",
           "| 序 | LR1 WP | LR1 CIL ACC | LR(42) WP | LR(42) CIL ACC |", "|---|---|---|---|---|"]
    for o in ORDERS:
        out.append(f"| {o} | {ms(col(r['LR1'][o], 'wp'))} | {ms(col(r['LR1'][o], 'cil'))} | {ms(col(r['LR'][o], 'wp'))} | {ms(col(r['LR'][o], 'cil'))} |")
    out += ["", "LR1 − LR(42)（逐折相減）：", ""] + PHEAD
    for o in ORDERS:
        out += [prow(f"{o}：WP", r["vs_LR"][o]["wp"], n), prow(f"{o}：CIL ACC", r["vs_LR"][o]["cil"], n)]
    out += ["", f"LR1 兩序：{osame(r['order_same_LR1'])}。", ""]
    return out


def sec_e4(r) -> list[str]:
    n = len(r["folds"])
    S = r["seeds"]
    out = ["落點：PREREG-18 E4、細則 20。LRG 的輸入 [v(s); mean_vec; 1]（1025 維；1025 × 1025 的 float64 封閉解超出 AGENTS.md 可攜規則 3 的 513 × 513 例外，依本批指令執行）。", "",
           gtext(r["gamma_LRG"], "LRG"), "", f"LR 的 γ\\* = {r['gamma_LR']['star']:g}。", "",
           "T4.1 LRG 與 LR（test、t = 4；mean ± sd）：", "",
           "| seed | 序 | LRG WP | LRG CIL ACC | LR WP | LR CIL ACC | LRG − LR WP：平均（贏折數） | LRG − LR CIL：平均（贏折數） |", "|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        for o in ORDERS:
            x = S[str(s)]["orders"][o]
            out.append(f"| {s} | {o} | {ms(col(x['LRG'], 'wp'))} | {ms(col(x['LRG'], 'cil'))} | {ms(col(x['LR'], 'wp'))} | {ms(col(x['LR'], 'cil'))} | "
                       f"{wl(x['vs_LR']['wp'], n)} | {wl(x['vs_LR']['cil'], n)} |")
    for o in ORDERS:
        out += ["", f"T4.2 LRG − LR 逐折相減（{o}）：", ""] + PHEAD
        for s in SEEDS:
            x = S[str(s)]["orders"][o]["vs_LR"]
            out += [prow(f"seed {s}：WP", x["wp"], n), prow(f"seed {s}：CIL ACC", x["cil"], n)]
    out += ["", "LRG 兩序：" + "；".join(f"seed {s} {osame(S[str(s)]['order_same_LRG'])}" for s in SEEDS) + "。", "",
            "T4.3 分數層級的相加（seed 42、t = 4、告訴任務；GR = LIN8 γ = 0.01、LR = LR(42)，皆 reverse；各自除以該任務 validation 的 σ；只作描述）：", ""]
    for split in ("val", "test"):
        x = r["score_sums"][split]
        out += [f"{'validation' if split == 'val' else 'test'}：", "", "| 列 | WP（mean ± sd） | − LT：平均 | 贏折數 | 逐折差 |", "|---|---|---|---|---|"]
        for k in ("LT", "GR", "LR"):
            out.append(f"| {k}（單獨） | {ms(x['alone'][k])} | | | |")
        for k, v in x["fusion"].items():
            out.append(f"| {k.replace('+', '＋')} | {ms(v['wp'])} | {v['vs_LT']['mean']:+.4f} | {v['vs_LT']['wins']}/{n} | {pf(v['vs_LT']['per_fold'])} |")
        out.append("")
    return out


def sec_e5(r) -> list[str]:
    n = len(r["folds"])
    S = r["seeds"]
    out = ["落點：PREREG-18 E5、細則 21。權重 1/n_c（n_c = 該折 train 中類別 c 的張數）。M3-bal = LT(seed s) ＋ GR-bal（σ 固定版，β = 1）；TP 仍為 AR。不設門檻。", "",
           gtext(r["gamma_LRbal"], "LR-bal"), "", gtext(r["gamma_GRbal"], "GR-bal"), "",
           f"LR 的 γ\\* = {r['gamma_LR']['star']:g}。", "",
           "T5.1 GR-bal 單獨（test、t = 4）：", "", "| 序 | WP | CIL ACC | ESCA | RCC | BRCA | LUNG |", "|---|---|---|---|---|---|---|"]
    for o in ORDERS:
        x = r["GR-bal"][o]
        out.append(f"| {o} | {ms(col(x, 'wp'))} | {ms(col(x, 'cil'))} | {pt(col(x, 'wp_t'))} |")
    out += ["", "T5.2 LR-bal、M3-bal（mean ± sd）與相對 LR、M3：", "",
            "| seed | 序 | LR-bal WP | LR-bal CIL ACC | M3-bal WP | M3-bal CIL ACC | LR-bal − LR WP | LR-bal − LR CIL | M3-bal − M3 WP | M3-bal − M3 CIL |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        for o in ORDERS:
            x = S[str(s)]["orders"][o]
            out.append(f"| {s} | {o} | {ms(col(x['LR-bal'], 'wp'))} | {ms(col(x['LR-bal'], 'cil'))} | {ms(col(x['M3-bal'], 'wp'))} | {ms(col(x['M3-bal'], 'cil'))} | "
                       f"{wl(x['vs_LR']['wp'], n)} | {wl(x['vs_LR']['cil'], n)} | {wl(x['M3bal_vs_M3']['wp'], n)} | {wl(x['M3bal_vs_M3']['cil'], n)} |")
    out += ["", "（差的格式：平均（贏折數／" + str(n) + "）。）", "",
            "LR-bal 兩序：" + "；".join(f"seed {s} {osame(S[str(s)]['order_same_LRbal'])}" for s in SEEDS) + "。", ""]
    return out


def sec_e6(r) -> list[str]:
    cn, T = r["class_names"], r["table"]
    out = ["落點：PREREG-18 E6、細則 22。seed 42（另列 LR 五個 seed 合計）、test、告訴任務、t = 4、reverse；十折合計。", "",
           "γ\\*：" + "、".join(f"{k} {v['star']:g}" for k, v in r["gamma"].items()) + "。", "",
           "T6.1 每類別：正確張數／張數（正確率）：", "", "| 系統 | " + " | ".join(cn) + " |", "|---|" + "---|" * 8]
    for k in r["systems"]:
        x = T[k]
        out.append(f"| {k} | " + " | ".join(f"{x['correct'][c]}/{x['n'][c]}（{x['acc'][c]:.4f}）" for c in range(8)) + " |")
    out += ["", "T6.2 每任務 balanced accuracy（兩類正確率的平均；十折合計）與四任務平均：", "",
            "| 系統 | ESCA | RCC | BRCA | LUNG | 四任務平均 | 全部張數的正確率 |", "|---|---|---|---|---|---|---|"]
    for k in r["systems"]:
        x = T[k]
        out.append(f"| {k} | " + " | ".join(f"{v:.4f}" for v in x["bal_acc"]) + f" | {x['bal_acc_mean']:.4f} | {x['acc_all']:.4f} |")
    g = r["groups"]
    out += ["", "T6.3 LR(42) 相對主系統（每類別，十折合計）：", "", "| 類別 | n | " + " | ".join(g) + " |", "|---|---|---|---|---|---|"]
    for c in range(8):
        x = r["fix_break_by_class"][c]
        out.append(f"| {cn[c]} | {sum(x)} | " + " | ".join(str(v) for v in x) + " |")
    tot = [sum(r["fix_break_by_class"][c][i] for c in range(8)) for i in range(4)]
    out += [f"| 合計 | {sum(tot)} | " + " | ".join(str(v) for v in tot) + " |", "", "每任務：", "", "| 任務 | n | " + " | ".join(g) + " |", "|---|---|---|---|---|---|"]
    for p in range(4):
        x = r["fix_break_by_task"][p]
        out.append(f"| {TASK[p]} | {sum(x)} | " + " | ".join(str(v) for v in x) + " |")
    return out + [""]


def sec_e7(r) -> list[str]:
    s2 = json.loads((BASE / "moe1" / "s2.json").read_text())
    d2 = r["s2_max_abs_diff"]
    out = ["落點：PREREG-18 E7、細則 23。seed 42；ACC 只在已學類別中判（TP = 該階段的 AR）；ridge 判讀器的 Masked ACC = 告訴任務判定。"
           "主系統與 M3 的數值取自 `moe1/s2.json`；本批照 `moe1_s2.py` 重算，與 s2.json 的最大絕對差："
           f"主系統 {d2['main']:.2e}、M3 {d2['M3']:.2e}（ACC、Masked、Forgetting、BWT、逐階段 ACC／Masked ACC）。", "",
           "γ\\*：" + "、".join(f"{k} {v['star']:g}" for k, v in r["gamma"].items()) + "。", ""]
    name = {"main": "主系統（s2.json）", "M3": "M3（σ 固定版；s2.json）", "LR": "LR", "LRG": "LRG", "LR-bal": "LR-bal"}
    for o in ORDERS:
        d, ref = r["orders"][o], s2["orders"][o]
        src = {"main": ref["main"], "M3": ref["M3"], "LR": d["LR"], "LRG": d["LRG"], "LR-bal": d["LR-bal"]}
        out += [f"### {o}", "", "T7.1 逐階段 ACC／Masked ACC（mean ± sd）與 Forgetting、BWT：", "",
                "| 系統 | 指標 | t = 1 | t = 2 | t = 3 | t = 4 | Forgetting | BWT |", "|---|---|---|---|---|---|---|---|"]
        for k, x in src.items():
            out.append(f"| {name[k]} | ACC | " + " | ".join(ms([v["acc_t"][t] for v in x]) for t in range(4)) +
                       f" | {ms(col(x, 'forgetting'))} | {ms(col(x, 'bwt'))} |")
            out.append(f"| {name[k]} | Masked ACC | " + " | ".join(ms([v["masked_t"][t] for v in x]) for t in range(4)) + " | | |")
        out += ["", "T7.2 每任務 WP（告訴任務；mean ± sd；未學為 —）：", "", "| 系統 | 任務 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|"]
        wsrc = {"main": ref["wp_task_t"]["main"], "M3": ref["wp_task_t"]["M3"], "LR": d["wp_task_t"]["LR"],
                "LRG": d["wp_task_t"]["LRG"], "LR-bal": d["wp_task_t"]["LR-bal"]}
        for k, w in wsrc.items():
            for p in range(4):
                cells = [ms([x[t][p] for x in w]) if w[0][t][p] is not None else "—" for t in range(4)]
                out.append(f"| {name[k]} | {TASK[p]} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def sec_e8(r) -> list[str]:
    it, S = r["items"], r["systems"]
    out = ["落點：PREREG-18 E8、細則 24。fp32 bytes；ridge 存累加統計量 A、B（W 可由 A、B 解出）。不設門檻。", "",
           "T8.1 項目：", "", "| 代號 | 內容 | float 個數 | fp32 bytes |", "|---|---|---|---|"]
    for k, v in it.items():
        out.append(f"| {k} | {v['what']} | {v['n_float']:,} | {v['fp32_bytes']:,} |")
    out += ["", "T8.2 各系統（總量 = S + T·p，T = 任務數）：", "",
            "| 系統 | 所有任務共用 | 每任務 | 共用 S（bytes） | 每任務 p（bytes） | T = 4 總量（bytes） | 備註 |", "|---|---|---|---|---|---|---|"]
    for k, v in S.items():
        out.append(f"| {k} | {' ＋ '.join(v['shared'])} | {' ＋ '.join(v['per_task'])} | {v['shared_bytes']:,} | {v['per_task_bytes']:,} | "
                   f"{v['T4_bytes']:,} | {v.get('note', '')} |")
    dd = S["LRG"]["dedup"]
    out += ["", f"LRG 省去重複（A_mv 取自 A_g 的子區塊、不存 B_ar）：S = {dd['shared_bytes']:,}、p = {dd['per_task_bytes']:,}、T = 4 總量 {dd['T4_bytes']:,} bytes。",
            "", f"不計入：{r['not_counted']}。{r['solved_W_note']}。A 為對稱矩陣，只存上三角約可減半（上表為完整矩陣）。", ""]
    return out


SEC = {"vec": sec_vec, "e1": sec_e1, "e2": sec_e2, "e3": sec_e3, "e4": sec_e4, "e5": sec_e5, "e6": sec_e6, "e7": sec_e7, "e8": sec_e8}


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="moe2")
    a = ap.parse_args()
    BASE = REPO_ROOT / "outputs" / "navcil" / load_config()["machine"]
    root = BASE / a.out
    smoke = a.out != "moe2"
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
            "MOE-2：證據向量的 closed-form 判讀（LR）— 多 seed 確認、是否需要 head、與整片向量合併、類別加權、CL 流程（Mac CPU，十折，reverse 與 paper 兩序）", "",
            "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；closed-form 一律 float64；四任務等權。"
            "所有數字來自同一台、同一批（`scripts/moe2_run_all.sh`）；沿用 MOE-1 的程式、快取與 head 權重（seed 42：`i6/r2/`；seed 43–46：`moe1/i6_seed{s}/`），不訓練任何 head。"
            "判準與操作定義見 `PREREG-18.md`。", "",
            "主系統 = AR（γ = 1e-3）＋ I6(r = 2)；M3 = PREREG-17 的 σ 固定版；LR = 證據向量 [v(s); 1] 的累加式 ridge；CIL 的 TP 一律 AR（γ = 1e-3）。", "",
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
        else:
            body += [f"{status[s]}。", ""]
    body += ["## 失敗的階段", ""]
    fails = sorted(root.glob("FAILED_*.txt"))
    if not fails:
        body += ["無。", ""]
    for p in fails:
        lines = p.read_text().strip().splitlines()
        body += [f"### {p.name}", "", "```"] + lines[:3] + (["…"] if len(lines) > 9 else []) + lines[3:][-6:] + ["```", ""]
    dec = BASE / "moe2" / "DECISIONS.md"
    body += ["## DECISIONS.md（執行前與執行中的判斷）", "", dec.read_text().strip() if dec.exists() else "（無）", ""]
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
    path = (root / "REPORT_smoke.md") if smoke else (BASE / "REPORT_moe2.md")
    path.write_text("\n".join(head + body))
    print("→", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
