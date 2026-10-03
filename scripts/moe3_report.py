#!/usr/bin/env python3
"""MOE-3 報告（PREREG-19 細則 29）：只讀各階段的 <階段>.json，不重算；只含已完成的階段。

    NAVCIL_MACHINE=mac python scripts/moe3_report.py [--out moe3]
輸出：--out moe3 → outputs/navcil/<machine>/REPORT_moe3.md；其他（冒煙測試）→ <out>/REPORT_smoke.md（不是結果）。
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

from moe1_report import PHEAD, col, f4, mean, ms, pf, prow, yes        # noqa: E402,F401
from selector.text_encoder import load_config                          # noqa: E402

TASK = ["ESCA", "RCC", "BRCA", "LUNG"]
ORDERS = ["reverse", "paper"]
STAGES = ["vec", "chk", "hp", "f2", "f4", "f5", "f6", "f7"]
SYS = ["S-main", "S-M3", "S-LR", "S-T0", "S-R0", "S-A0", "S-R0f", "S-A0f", "S-Ah"]
KN = {"u16": "u_16", "u32": "u_32", "u64": "u_64", "u128": "u_128", "u256": "u_256", "g0": "v0（四輪）", "s42": "v(42)（四輪）",
      "mv": "mean_vec（全部 patch）"}
BASE = None                                     # outputs/navcil/<machine>


def rdtext(rd) -> str:
    if rd[0] == "TXT":
        return "TXT"
    return f"RDG(γ = {rd[1]:g})" if rd[0] == "RDG" else f"ANC(γ = {rd[1]:g}, α = {rd[2]:g})"


def osame(x) -> str:
    return f"告訴任務判定不同 {x['tell_diff']} 張、CIL 判定不同 {x['cil_diff']} 張（共 {x['n']} 張）；告訴任務 d 的最大絕對差 {x['d_maxabs']:.2e}"


def t4(rows, key):
    return [r[key][3] for r in rows]


# ── 門檻與 K ────────────────────────────────────────────────────────────────
def gates(J) -> list[str]:
    out = ["## 門檻（PREREG-19「門檻」；細則 16–19）", ""]
    r = J.get("f2")
    if not r:
        return out + ["f2 尚未完成或失敗：G-FINAL、G-COLD、G-ANCHOR 無結果。", ""]
    G, note = r["gates"], "" if r["gates_valid"] else "（非十折，不作判定）"
    verdict = lambda g: f"**{'通過' if g['pass'] else '未通過'}**{note}"   # noqa: E731

    out += ["### G-FINAL（每折 D = CIL ACC(系統, t = 4) − seed 42–46 五個主系統的 CIL ACC 平均；兩序都滿足才通過）", "",
            "| 系統 | 序 | D 十折平均 | D > 0 的折數 | 門檻 | 結果 | 逐折 D |", "|---|---|---|---|---|---|---|"]
    for name in ("S-A0", "S-R0"):
        for o in ORDERS:
            x = G[name]["G-FINAL"]["orders"][o]
            res = yes(x["pass"]) if name == "S-A0" else "（只報告）"
            out.append(f"| {name} | {o} | {x['mean']:+.4f} | {x['wins']} | ≥ +0.007 且 ≥ 7 折 | {res} | {pf(x['per_fold'])} |")
        if name == "S-A0":
            out.append(f"| **G-FINAL** | | | | 兩序 | {verdict(G[name]['G-FINAL'])} | |")
    out += ["", f"主系統五個 seed 的逐折 CIL ACC 取自 `moe2/e1.json`；本批重算的最大絕對差 {r['main5_vs_recompute_max_abs']:.2e}。"
            "五個 seed 平均的十折平均：" + "、".join(f"{o} {mean(r['main5_cil'][o]):.4f}" for o in ORDERS) + "。", "",
            "### G-COLD（兩序各自：t = 1 的 ACC(系統) − ACC(S-main) 的十折平均 ≥ −0.010）", "",
            "| 系統 | 序 | 十折平均 | > 0 的折數 | 門檻 | 結果 | 逐折差 |", "|---|---|---|---|---|---|---|"]
    for name in ("S-A0", "S-R0"):
        for o in ORDERS:
            x = G[name]["G-COLD"]["orders"][o]
            res = yes(x["pass"]) if name == "S-A0" else "（只報告）"
            out.append(f"| {name} | {o} | {x['mean']:+.4f} | {x['wins']} | ≥ −0.010 | {res} | {pf(x['per_fold'])} |")
        if name == "S-A0":
            out.append(f"| **G-COLD** | | | | 兩序 | {verdict(G[name]['G-COLD'])} | |")
    out += ["", "### G-ANCHOR（兩序各自：Ā(S-A0) − Ā(S-R0) 的十折平均 ≥ +0.005 且 > 0 的折數 ≥ 7）", "",
            "| 系統 | 序 | 十折平均 | > 0 的折數 | 門檻 | 結果 | 逐折差 |", "|---|---|---|---|---|---|---|"]
    for o in ORDERS:
        x = G["S-A0"]["G-ANCHOR"]["orders"][o]
        out.append(f"| S-A0 | {o} | {x['mean']:+.4f} | {x['wins']} | ≥ +0.005 且 ≥ 7 折 | {yes(x['pass'])} | {pf(x['per_fold'])} |")
    out += [f"| **G-ANCHOR** | | | | 兩序 | {verdict(G['S-A0']['G-ANCHOR'])} | |", "| S-R0 | | | | | （留空） | |", "",
            "S-R0 的列是同一組數字把 S-A0 換成 S-R0，只報告、不判定。F4、F5、F6、F7 不設門檻。", ""]
    return out


def kchecks(J) -> list[str]:
    out = ["## 一致性檢查 K1–K5（PREREG-19「一致性檢查」；細則 9–13）", "",
           "| 檢查 | 階段 | 結果 | 判準 | 通過 |", "|---|---|---|---|---|"]
    ok = lambda k: "是" if k["pass"] else "否"   # noqa: E731
    for s in STAGES:
        k = (J.get(s) or {}).get("K1")
        if k:
            out.append(f"| K1 主系統 seed 42 | {s.upper()} | 逐折最大絕對差 {k['max_abs_fold_diff']:.2e}；"
                       f"WP {k['wp_mean']['reverse']:.4f}／{k['wp_mean']['paper']:.4f}；CIL ACC {k['cil_mean']['reverse']:.4f}／{k['cil_mean']['paper']:.4f} | "
                       f"≤ 1e-9；0.9340；0.9128 | {ok(k)} |")
    c = J.get("chk")
    if c:
        k = c["K2"]
        out.append(f"| K2 RDG(γ = 1e-3)、v0 對 moe2 的 LR0 | CHK | 逐折最大絕對差 {k['max_abs_fold_diff']:.2e}；WP {k['wp_mean']['reverse']:.4f}／{k['wp_mean']['paper']:.4f}；"
                   f"CIL ACC {k['cil_mean']['reverse']:.4f}／{k['cil_mean']['paper']:.4f} | ≤ 1e-9；0.9435；0.9214 | {ok(k)} |")
        k = c["K3"]
        out.append(f"| K3 ANC(γ, α = 0) 與 RDG(γ) 的 W | CHK | 最大絕對差 {k['max_abs']:.2e}（{KN[k['kind']]}、γ = {k['gamma']:g}、fold {k['fold']}、{k['order']}、t = 4） | ≤ 1e-10 | {ok(k)} |")
        k = c["K4"]
        out.append(f"| K4 ANC(γ = 1e8, α = 1)、v0 對 TXT（v0） | CHK | 告訴任務判定不同 {k['diff']['reverse']}／{k['diff']['paper']} 張（共 {k['n']} 張；t = 4）；"
                   f"TXT 的 \\|d\\| 最小值 {k['d_txt_min_abs']:.2e} | 0 張 | {ok(k)} |")
        k = c["K5"]
        b2 = f"{k['b2_max_abs']:.2e}" if k["b2_comparable"] else "非十折，不可比"
        out.append(f"| K5 TXT（v0）每任務 WP 對 MOE-0 B2「g = 0」列 | CHK | 十折平均最大絕對差 {b2}；每折每任務對 moe0 `g0_cos8` 的最大絕對差 {k['cache_max_abs']:.2e} | ≤ 1e-9 | {ok(k)} |")
    out += ["", "兩個數字並列者為 reverse／paper。K 不過的階段不會出現在上表（見「失敗的階段」）。", ""]
    return out


# ── 各節 ────────────────────────────────────────────────────────────────────
def sec_vec(r) -> list[str]:
    n = r["n_slides"]
    tot = {s: sum(v for k, v in n.items() if k.startswith(s)) for s in ("train", "val", "test")}
    return ["落點：PREREG-19「向量」、細則 2。u_K 由特徵檔新算（每張 slide 讀一次、五個 K 同時算、不載入 head）；v0、v(42)、mean_vec 沿用既有快取。", "",
            f"張數（各折合計）：train {tot['train']}、validation {tot['val']}、test {tot['test']}；逐張記錄加總：讀檔 {r['t_read_s']:.0f} 秒、計算 {r['t_compute_s']:.0f} 秒。"
            f"slide id／標籤與既有快取逐張相同；u 向量範數與 1 的最大差 {r['norm_max_dev']:.2e}（判準 ≤ 1e-5）。", "",
            "patch 數 < K 的張數（三個 split、各折合計；這些 slide 取全部 patch）：" +
            "、".join(f"K = {k}：{v}" for k, v in r["n_patch_lt_K"].items()) + "。", ""]


def sec_f1(J) -> list[str]:
    r, hp = J.get("f2"), J.get("hp")
    out = ["落點：PREREG-19 F1、細則 6、20。", ""]
    if not r:
        return out + ["f2 尚未完成或失敗。", ""]
    S = r["systems"]
    row = lambda n, src: f"| {n} | {KN[S[n]['kind']]} | {rdtext(S[n]['reader'])} | {src} |"   # noqa: E731
    d = r["recompute_max_abs"]
    out += ["| 系統 | 輸入向量 | 判讀器（選定的超參數） | 數值來源 |", "|---|---|---|---|",
            f"| S-main | seed 42 head 的四輪向量 | 主系統（AR ＋ I6(r = 2)；與兩類文字比 cosine） | `moe1/s2.json`（本批重算最大絕對差 {d['S-main']:.2e}） |",
            f"| S-M3 | 同上 ＋ mean_vec | M3（σ 固定版） | `moe1/s2.json`（本批重算最大絕對差 {d['S-M3']:.2e}） |",
            row("S-LR", f"`moe2/e7.json` 的 LR（本批重算最大絕對差 {d['S-LR']:.2e}）；γ 固定，不重選"),
            row("S-T0", "本批；不訓練"), row("S-R0", "本批"), row("S-A0", "本批"), row("S-R0f", "本批"), row("S-A0f", "本批"), row("S-Ah", "本批"), "",
            "CIL 的 TP 一律 AR（γ = 1e-3）。重算的最大絕對差涵蓋逐階段 ACC、逐階段 WP、Forgetting、BWT（只報，不是 K）。", ""]
    if hp:
        out += ["超參數的整張 validation 網格見 F6。", ""]
    return out


def sec_f2(r) -> list[str]:
    R = r["rows"]
    out = ["落點：PREREG-19 F2、細則 4、21。test；ACC 只在已學類別中判（TP = 該階段的 AR）；WP = 告訴任務、已學任務等權；Ā = 四個階段 ACC 的平均；"
           "Forgetting、BWT 由 `nc5_report.cil_full` 計。十折 mean ± sd。", ""]
    for o in ORDERS:
        out += [f"### {o}", "", "T2.1 逐階段 ACC／WP 與 Forgetting、BWT、Ā：", "",
                "| 系統 | 指標 | t = 1 | t = 2 | t = 3 | t = 4 | Forgetting | BWT | Ā |", "|---|---|---|---|---|---|---|---|---|"]
        for k in SYS:
            x = R[o][k]
            out.append(f"| {k} | ACC | " + " | ".join(ms([v["acc_t"][t] for v in x]) for t in range(4)) +
                       f" | {ms(col(x, 'forgetting'))} | {ms(col(x, 'bwt'))} | {ms(col(x, 'abar'))} |")
            out.append(f"| {k} | WP | " + " | ".join(ms([v["wp_t"][t] for v in x]) for t in range(4)) + " | | | |")
        out += ["", "T2.2 每任務 WP（告訴任務；mean ± sd；未學為 —）：", "", "| 系統 | 任務 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|"]
        for k in SYS:
            w = [v["wp_task_t"] for v in R[o][k]]
            for p in range(4):
                cells = [ms([x[t][p] for x in w]) if w[0][t][p] is not None else "—" for t in range(4)]
                out.append(f"| {k} | {TASK[p]} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def sec_f3(r) -> list[str]:
    n, R = len(r["folds"]), r["rows"]
    out = ["落點：PREREG-19 F3、細則 7、8、22。test、t = 4；S-main 的 WP = 真實任務 head 的 2 類正確率（四任務等權）。", ""]
    for o in ORDERS:
        out += [f"### {o}", "", "T3.1 t = 4 的 WP、CIL ACC（mean ± sd）與相對 S-main 的平均差（贏折數）：", "",
                "| 系統 | WP | CIL ACC | WP − S-main | CIL ACC − S-main |", "|---|---|---|---|---|"]
        for k in SYS:
            v = r["vs_main"][o].get(k)
            d = f"{v['wp']['mean']:+.4f}（{v['wp']['wins']}/{n}） | {v['cil']['mean']:+.4f}（{v['cil']['wins']}/{n}）" if v else "— | —"
            out.append(f"| {k} | {ms(t4(R[o][k], 'wp_t'))} | {ms(t4(R[o][k], 'acc_t'))} | {d} |")
        out += ["", "T3.2 相對 S-main(42) 的逐折相減：", ""] + PHEAD
        for k in SYS:
            if k != "S-main":
                v = r["vs_main"][o][k]
                out += [prow(f"{k} − S-main：WP", v["wp"], n), prow(f"{k} − S-main：CIL ACC", v["cil"], n)]
        out.append("")
    out += ["T3.3 兩序 t = 4 的結果是否逐張相同（細則 8）：", "", "| 系統 | 比較 |", "|---|---|"]
    for k in ("S-R0", "S-A0", "S-T0", "S-R0f", "S-A0f", "S-Ah", "S-LR"):
        out.append(f"| {k} | {osame(r['order_same'][k])} |")
    return out + [""]


def sec_f4(r) -> list[str]:
    cn, T = r["class_names"], r["table"]
    out = ["落點：PREREG-19 F4、細則 23。test、告訴任務、t = 4、reverse；十折合計。不設門檻。", "",
           "判讀器：" + "、".join(f"{k} = {KN[v['kind']]}／{rdtext(v['reader'])}" for k, v in r["readers"].items()) + "。", "",
           "T4.1 每類別：正確張數／張數（正確率）：", "", "| 系統 | " + " | ".join(cn) + " |", "|---|" + "---|" * 8]
    for k in r["systems"]:
        x = T[k]
        out.append(f"| {k} | " + " | ".join(f"{x['correct'][c]}/{x['n'][c]}（{x['acc'][c]:.4f}）" for c in range(8)) + " |")
    out += ["", "T4.2 每任務 balanced accuracy（兩類正確率的平均；十折合計）與四任務平均：", "",
            "| 系統 | ESCA | RCC | BRCA | LUNG | 四任務平均 | 全部張數的正確率 |", "|---|---|---|---|---|---|---|"]
    for k in r["systems"]:
        x = T[k]
        out.append(f"| {k} | " + " | ".join(f"{v:.4f}" for v in x["bal_acc"]) + f" | {x['bal_acc_mean']:.4f} | {x['acc_all']:.4f} |")
    return out + [""]


def sec_f5(r) -> list[str]:
    out = ["落點：PREREG-19 F5、細則 24。test；RDG、ANC 一律用 u_64 選定的超參數（不重選）；「全部 patch」= mean_vec。mean ± sd。不設門檻。", "",
           "判讀器：" + "、".join(f"{k} = {rdtext(v)}" for k, v in r["readers"].items()) + "。", ""]
    head = "| 系統的判讀器 | 指標 | " + " | ".join("全部 patch" if k == "mv" else f"K = {k[1:]}" for k in r["kinds"]) + " |"
    for o in ORDERS:
        out += [f"### {o}", "", head, "|---|---|" + "---|" * len(r["kinds"])]
        for n in r["readers"]:
            for key, lab in (("wp", "t = 4 WP"), ("cil", "t = 4 CIL ACC"), ("abar", "Ā")):
                out.append(f"| {n} | {lab} | " + " | ".join(ms(col(r["orders"][o][n][k], key)) for k in r["kinds"]) + " |")
        out.append("")
    return out


def sec_f6(J) -> list[str]:
    hp, r = J.get("hp"), J.get("f6")
    out = ["落點：PREREG-19「超參數的選法」、F6、細則 5、25。目標值 = 兩序 × t = 1…4 的 validation 告訴任務 WP（已學任務等權）平均的十折平均；"
           "同分先取較大的 γ，再取較小的 α；選在邊界照報，不延伸。", ""]
    if not hp:
        return out + ["hp 尚未完成或失敗。", ""]
    G = hp["grids"]
    for key, c in hp["combos"].items():
        s = c["star"]
        use = "（只作描述；S-LR 固定 γ = 1e-5，不用這個選擇）" if key == "RDG:s42" else ""
        if c["reader"] == "RDG":
            out += [f"T6 {c['reader']} × {KN[c['kind']]}{use}：", "", "| γ | " + " | ".join(f"{g:g}" for g in G["RDG_gamma"]) + " |",
                    "|---|" + "---|" * len(G["RDG_gamma"]),
                    "| 目標值 | " + " | ".join(f"{c['obj'][f'{g:g}']:.4f}" for g in G["RDG_gamma"]) + " |", "",
                    f"選定 γ\\* = **{s['gamma']:g}**；γ 在邊界：{'是' if s['edge_gamma'] else '否'}。", ""]
        else:
            out += [f"T6 {c['reader']} × {KN[c['kind']]}：", "", "| γ ＼ α | " + " | ".join(f"{a:g}" for a in G["ANC_alpha"]) + " |",
                    "|---|" + "---|" * len(G["ANC_alpha"])]
            for g in G["ANC_gamma"]:
                out.append(f"| {g:g} | " + " | ".join(f"{c['obj'][f'{g:g}|{a:g}']:.4f}" for a in G["ANC_alpha"]) + " |")
            out += ["", f"選定 γ\\* = **{s['gamma']:g}**、α\\* = **{s['alpha']:g}**；γ 在邊界：{'是' if s['edge_gamma'] else '否'}；"
                    f"α 在邊界：{'是' if s['edge_alpha'] else '否'}。", ""]
    if not r:
        return out + ["f6（敏感度）尚未完成或失敗。", ""]
    out += [f"T6.S S-A0 選定值（γ = {r['star']['gamma']:g}、α = {r['star']['alpha']:g}）附近的 test 敏感度（t = 4 CIL ACC 與 Ā；mean ± sd；網格外為 —）：", "",
            "| 格 | γ | α | CIL ACC（reverse） | Ā（reverse） | CIL ACC（paper） | Ā（paper） |", "|---|---|---|---|---|---|---|"]
    for c in r["cells"]:
        if "orders" in c:
            vals = " | ".join(f"{ms(col(c['orders'][o], 'cil'))} | {ms(col(c['orders'][o], 'abar'))}" for o in ORDERS)
            out.append(f"| {c['where']} | {c['gamma']:g} | {c['alpha']:g} | {vals} |")
        else:
            out.append(f"| {c['where']} | {'—' if c['gamma'] is None else format(c['gamma'], 'g')} | {'—' if c['alpha'] is None else format(c['alpha'], 'g')} | — | — | — | — |")
    return out + [""]


def sec_f7(r) -> list[str]:
    it, S = r["items"], r["systems"]
    out = ["落點：PREREG-19 F7、細則 26。fp32 bytes；ridge 存累加統計量 A、B（W 可由 A、B 解出）。不設門檻。", "",
           "T7.1 項目：", "", "| 代號 | 內容 | float 個數 | fp32 bytes |", "|---|---|---|---|"]
    for k, v in it.items():
        out.append(f"| {k} | {v['what']} | {v['n_float']:,} | {v['fp32_bytes']:,} |")
    out += ["", "T7.2 各系統（總量 = S + T·p，T = 任務數）：", "",
            "| 系統 | 所有任務共用 | 每任務 | 共用 S（bytes） | 每任務 p（bytes） | T = 4 總量（bytes） | 備註 |", "|---|---|---|---|---|---|---|"]
    for k, v in S.items():
        out.append(f"| {k} | {' ＋ '.join(v['shared'])} | {' ＋ '.join(v['per_task'])} | {v['shared_bytes']:,} | {v['per_task_bytes']:,} | "
                   f"{v['T4_bytes']:,} | {v.get('note', '')} |")
    out += ["", f"不計入：{r['not_counted']}。{r['solved_W_note']}。A 為對稱矩陣，只存上三角約可減半（上表為完整矩陣）。", ""]
    return out


SECTIONS = [("向量快取（vec）", "vec", sec_vec, False), ("F1 系統列表", "f2", sec_f1, True), ("F2 逐階段", "f2", sec_f2, False),
            ("F3 t = 4 與相對 S-main", "f2", sec_f3, False), ("F4 每類別", "f4", sec_f4, False), ("F5 K 的影響", "f5", sec_f5, False),
            ("F6 超參數", "hp", sec_f6, True), ("F7 儲存", "f7", sec_f7, False)]


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="moe3")
    a = ap.parse_args()
    BASE = REPO_ROOT / "outputs" / "navcil" / load_config()["machine"]
    root = BASE / a.out
    smoke = a.out != "moe3"
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
            "MOE-3：定案批 — 不用 head 的證據向量、以文字為起點的 ridge 判讀、逐階段結果（Mac CPU，十折，reverse 與 paper 兩序）", "",
            "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；closed-form 一律 float64；四任務等權。"
            "所有數字來自同一台、同一批（`scripts/moe3_run_all.sh`）；只算向量與 closed-form，不訓練；沿用 MOE-2 的程式與快取（v(42)、四輪的 v0）。"
            "判準與操作定義見 `PREREG-19.md`。", "",
            "u_K = 不用 head、依 s0 一次取前 K 個 patch 的等權平均向量；TXT = 與兩類文字比 cosine；RDG(γ)：W = (A + γI)⁻¹ B；"
            "ANC(γ, α)：W = (A + γI)⁻¹ (B + γ·α·T)；CIL 的 TP 一律 AR（γ = 1e-3）。", "",
            "各階段狀態：" + "、".join(f"{s.upper()} {status[s]}" for s in STAGES) +
            f"。已完成階段的 folds：{'；'.join(str(list(f)) for f in folds) if folds else '—'}。本報告只含已完成的階段。", ""]
    body = gates(J) + kchecks(J)
    for title, stage, fn, whole in SECTIONS:
        body += [f"## {title}", ""]
        if whole or stage in J:
            try:
                body += fn(J if whole else J[stage])
            except Exception:
                body += ["本節格式化失敗（數值仍在 `" + f"{a.out}/{stage}.json" + "`）：", "", "```", traceback.format_exc().strip(), "```", ""]
        else:
            body += [f"{stage} {status[stage]}。", ""]
    body += ["## 失敗的階段", ""]
    fails = sorted(root.glob("FAILED_*.txt"))
    if not fails:
        body += ["無。", ""]
    for p in fails:
        lines = p.read_text().strip().splitlines()
        body += [f"### {p.name}", "", "```"] + lines[:3] + (["…"] if len(lines) > 9 else []) + lines[3:][-6:] + ["```", ""]
    dec = BASE / "moe3" / "DECISIONS.md"
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
    path = (root / "REPORT_smoke.md") if smoke else (BASE / "REPORT_moe3.md")
    path.write_text("\n".join(head + body))
    print("→", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
