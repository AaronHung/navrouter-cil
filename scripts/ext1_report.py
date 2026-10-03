#!/usr/bin/env python3
"""EXT-1 報告（PREREG-21 細則 29）：由 ext1/*.json、ABLATION_full.csv 與 outputs/external/ 產生 REPORT_ext1.md。只給數字與表格。

外部對照的名稱、來源說明、K6 一律讀 outputs/external/<dir>/ 內的檔（meta.json、PROVENANCE.md、k6.json、PORT.md），本檔不寫死。

    NAVCIL_MACHINE=mac python scripts/ext1_report.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.text_encoder import load_config                             # noqa: E402

BASE = REPO_ROOT / "outputs" / "navcil" / load_config()["machine"]
EXT = REPO_ROOT / "outputs" / "external"
ORDERS = ("reverse", "paper")
TASKS = ["ESCA", "RCC", "BRCA", "LUNG"]
ROWS = [("ZS8", "zero-shot 8 類（top-64，無訓練）"), ("LIN8", "LIN8（mean_vec，ridge γ = 0.01，無 TP）"),
        ("MAIN", "主系統 P-main(42)"), ("FINAL", "FINAL＝P-4(42)"), ("NOHEAD", "拿掉 head（v0／RDG 1e-3）"),
        ("ONE64", "一次 top-64（P-F(42)）"), ("K32", "K = 32（u_K／RDG，不用 head）"), ("K64", "K = 64"), ("K128", "K = 128"),
        ("K256", "K = 256"), ("M1", "M1：任務間 soft gating（T\\* = 0.1）"), ("M2_head_esca", "M2：ESCA head 用於全部任務"),
        ("M2_head_rcc", "M2：RCC head"), ("M2_head_brca", "M2：BRCA head"), ("M2_head_lung", "M2：LUNG head"),
        ("M2_g0", "M2 對照：g = 0"), ("M3", "M3（σ 固定版，β = 1）"), ("G1", "M3 ＋ G1"), ("G2", "M3 ＋ G2"),
        ("ANC", "ANC × u_64"), ("ANC_v0", "ANC × v0"), ("ANC_v42", "ANC × v(42)"), ("CONCAT", "concat：LRG(42)，[v; mean_vec; 1]"),
        ("RF", "random features 取代 (b) 的 M3"), ("MAIN_5seed", "主系統，五 seed 平均"), ("FINAL_5seed", "FINAL，五 seed 平均"),
        ("ONE64_5seed", "一次 top-64，五 seed 平均")]


def J(p: Path):
    return json.loads(p.read_text())


def pm(x) -> str:
    return f"{x[0]:.4f} ± {x[1]:.4f}"


def ms(xs):
    return (statistics.fmean(xs), statistics.stdev(xs)) if xs else None


def sg(x: float) -> str:
    return f"{x:+.4f}"


def pv(p) -> str:
    return "不適用" if p is None else f"{p:.4f}"


def main() -> None:
    d, d2, k7 = J(BASE / "ext1" / "d.json"), J(BASE / "ext1" / "d2.json"), J(BASE / "ext1" / "k7.json")
    rows = list(csv.DictReader((BASE / "ABLATION_full.csv").open()))
    A = defaultdict(lambda: defaultdict(list))                            # (row, order, t) → 欄 → 逐折值
    for r in rows:
        for c in ("CIL_ACC", "WP", "MaskedACC_table1", "MaskedACC_oracle", "Forgetting", "BWT"):
            if r[c] != "":
                A[(r["row"], r["order"], int(r["t"]))][c].append(float(r[c]))
    store = {(r["row"], int(r["t"])): r["storage_bytes"] for r in rows}
    L = []
    w = L.append
    fails = sorted((BASE / "ext1").glob("FAILED_*.txt"))
    ext_dirs = sorted(p for p in EXT.iterdir() if p.is_dir()) if EXT.is_dir() else []
    metas = {p.name: (J(p / "meta.json") if (p / "meta.json").exists() else {"display_name": p.name}) for p in ext_dirs}

    w("# REPORT — EXT-1：外部對照的同折逐折配對；定稿表格全量重算（Mac CPU，十折，reverse 與 paper 兩序）")
    w("")
    w("機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0、closed-form float64。C、D 的數字來自同一台、同一批"
      "（`scripts/ext1_c.py`、`ext1_k7.py`、`ext1_d.py`、`ext1_d2.py`）；只讀既有快取與權重，沒有訓練任何模型，沒有改 FINAL 的任何定義或參數。"
      "判準與操作定義見 `PREREG-21.md`（commit 277339b）。外部對照的數字來自其他機器，來源見各自的 PROVENANCE。")
    w("")
    w("## 狀態")
    w("")
    w("| 階段 | 狀態 |")
    w("|---|---|")
    for p in ext_dirs:
        k6p = p / "k6.json"
        if k6p.exists():
            k6 = J(k6p)["batches"]
            s = "；".join(f"{b} K6 {v['n_same']}/{v['n_cells']}（{'通過' if v['pass'] else '不過'}）" for b, v in k6.items())
            w(f"| A {metas[p.name]['display_name']} 逐折匯入 | 完成。{s} |")
    w("| B MergeSlide 同折重跑 | **停在 B1**（補充 4 的停止條件成立；見 B 節）。B2、B3、B4、B5、K8 沒有執行 |")
    w(f"| C 全量重算 | 完成。K7：比對 {k7['cells']} 格，差 > {k7['tol']:g} 的 {k7['n_bad']} 格 |")
    w("| D 配對統計、成本、γ 敏感度、混淆矩陣 | 完成 |")
    w(f"| 失敗的階段 | {len(fails)} |")
    w("")

    # ── A ──
    for p in ext_dirs:
        name, meta = p.name, metas[p.name]
        disp = meta["display_name"]
        if not (p / "perfold.csv").exists():
            continue
        w(f"## A　{disp} 逐折數字與配對（PREREG-21 細則 4、7–11）")
        w("")
        k6 = J(p / "k6.json")
        w("### K6 折一致性（每格 = 一個 (fold, task, split) 的 slide ID 集合 sha256；對 navrouter-cil 實際載入的清單）")
        w("")
        w("| 批次 | 序 | 相同格數／總格數 | test 相同格數／40 | 判定 | 是否進配對表 |")
        w("|---|---|---|---|---|---|")
        prim = set(meta.get("primary", {}).values())
        for b, v in k6["batches"].items():
            w(f"| {b} | {v['order']} | {v['n_same']}／{v['n_cells']} | {v['test_cells_same']}／40 | {'通過' if v['pass'] else '不過'} | "
              f"{'是' if v['pass'] and b in prim else '否'} |")
        w("")
        bad = [(b, x) for b, v in k6["batches"].items() for x in v["diffs"]]
        if bad:
            w("不一致的格：")
            w("")
            w("| 批次 | 格（fold｜task｜split） | 差異 |")
            w("|---|---|---|")
            for b, x in bad:
                why = x.get("reason") or f"對方 {x['n_theirs']} 張、我方 {x['n_ours']} 張；只在對方 {len(x['only_theirs'])}、只在我方 {len(x['only_ours'])}"
                w(f"| {b} | {x['cell'].replace('|', '｜')} | {why} |")
            w("")
        e = d["EXT"].get(name)
        pub = meta.get("published", {})
        for o in ORDERS:
            if not e or not e.get(o):
                continue
            w(f"### 逐折配對：序 {o}（t = 4；十折；差 = 我方 − {disp}）")
            w("")
            w(f"| 我方系統 | 指標 | 我方 mean ± sd | {disp} mean ± sd | 平均差 | 我方較高／較低／平手的折數 | 我方較好的折數／10 | 符號檢定 p（雙尾） |")
            w("|---|---|---|---|---|---|---|---|")
            for sysname, cell in e[o].items():
                for mk, r in cell.items():
                    better = r["losses"] if mk == "Forgetting" else r["wins"]
                    w(f"| {sysname} | {mk} | {pm(r['a'])} | {pm(r['b'])} | {sg(r['mean_diff'])} | {r['wins']}／{r['losses']}／{r['ties']} | "
                      f"{better} | {pv(r['p_sign'])} |")
            w("")
            md = meta.get("machine_date", {}).get(o)
            if md:
                w(f"表註（機器與日期）：{disp} 的數字在 {md} 產生；我方 FINAL 與主系統在 {meta.get('ours', '—')} 產生。外部對照不受 AGENTS.md 紅線 4 約束（PI 2026-10-03 裁決）。")
                w("")
            if o in pub:
                q = pub[o]
                w(f"表註（發表值，非同折，不進配對）：{q['table']}　ACC {q['ACC']}、Masked ACC {q['MaskedACC']}、Forgetting {q['Forgetting']}。")
                w("")
        w(f"- 「我方較好」：ACC、Masked ACC、BWT、Ā 為差 > 0 的折數；Forgetting 為差 < 0 的折數。")
        w(f"- {disp} 的 Masked ACC 是告訴任務的版本，對應我方的 oracle 定義；我方 Table 1 定義的那一列是拿兩個不同定義相比，照列供主表使用。")
        w("- FINAL(5seed)、MAIN(5seed)：每折先對 seed 42–46 平均，再配對。")
        w("")
        w(f"### {disp} 逐折值（t = 4；`outputs/external/{name}/perfold.csv`）")
        w("")
        pr = list(csv.DictReader((p / "perfold.csv").open()))
        w("| 序 | fold | ACC | MaskedACC | Forgetting | BWT |")
        w("|---|---|---|---|---|---|")
        for r in pr:
            if r["t"] == "4":
                w(f"| {r['order']} | {r['fold']} | {float(r['ACC']):.4f} | {float(r['MaskedACC']):.4f} | {float(r['Forgetting']):.4f} | {float(r['BWT']):.4f} |")
        w("")
        w(f"### 來源（`outputs/external/{name}/PROVENANCE.md` 全文）")
        w("")
        w((p / "PROVENANCE.md").read_text().replace("\n# ", "\n#### ").replace("\n## ", "\n#### ").lstrip("# ").strip())
        w("")

    # ── B ──
    w("## B　MergeSlide 同折重跑")
    w("")
    w("| 項目 | 結果 |")
    w("|---|---|")
    w("| B1 clone 與 PORT.md | 完成。上游 commit `96e7d67`，clone 於 `~/research/03_mergeslide`，未改動任何檔案 |")
    w("| 補充 4 的停止條件 | **成立**：它微調與合併的是 TITAN 預訓練的 slide encoder（輸入 CONCH v1.5 的 768 維特徵與座標）；CONCH v1（512 維、無座標）沒有對應物 |")
    w("| B2 接特徵 | 沒有執行 |")
    w("| B3 計時與估時 | 沒有執行；沒有估計時數 |")
    w("| B4 十折兩序、K8、B5 配對 | 沒有執行（另需 PI 的「B4 go」） |")
    w("| 替代 backbone | 沒有使用 |")
    w("")
    w("主表的 MergeSlide 列目前沒有同折數字。可能做法與各自偏離原方法的地方見下方 PORT.md 的最後一節，等 PI 決定。")
    w("")
    port = EXT / "mergeslide" / "PORT.md"
    if port.exists():
        w("### PORT.md 全文（`outputs/external/mergeslide/PORT.md`；正本在 `~/research/03_mergeslide/PORT.md`）")
        w("")
        w(port.read_text().replace("\n## ", "\n#### ").replace("# PORT.md", "#### PORT.md", 1).strip())
        w("")

    # ── C ──
    w("## C　全量重算（seed 42；十折 mean ± sd；`ABLATION_full.csv`）")
    w("")
    w("- 每列每 (order, fold) 的逐折值與 done 標記在 `ext1/c/<row>/`；CSV 另含 FINAL、MAIN、ONE64 的 seed 43–46 各列。")
    w("- Masked ACC（Table 1 定義）：向量／證據取自 τ̂ 的 expert，在真實任務兩類內 argmax；主表用這個定義。Masked ACC（oracle 定義）＝告訴任務的 WP，CSV 兩欄同值，報告只列 WP。")
    w("- 空格（—）：該列的既有定義沒有這個量（見 DECISIONS D17、D20）。**融合系統（M3、G1、G2、RF）的 Masked ACC 不進任何表（PI 2026-10-03 裁決，DECISIONS D15 作廢）：CSV 兩個 Masked 欄對這四列留空，WP 與 CIL ACC 照列。**")
    w("")
    for o in ORDERS:
        w(f"### C-1　t = 4，序 {o}")
        w("")
        w("| row | 說明 | CIL ACC | WP（告訴任務） | Masked（Table 1） | Forgetting | BWT | 儲存 bytes（T = 4） |")
        w("|---|---|---|---|---|---|---|---|")
        for row, desc in ROWS:
            a = A[(row, o, 4)]
            f = lambda c: pm(ms(a[c])) if a.get(c) else "—"            # noqa: E731
            sb = store.get((row, 4), "")
            w(f"| {row} | {desc} | {f('CIL_ACC')} | {f('WP')} | {f('MaskedACC_table1')} | {f('Forgetting')} | {f('BWT')} | "
              f"{int(sb):,}" + " |" if sb else f"| {row} | {desc} | {f('CIL_ACC')} | {f('WP')} | {f('MaskedACC_table1')} | {f('Forgetting')} | {f('BWT')} | — |")
        w("")
    for o in ORDERS:
        w(f"### C-2　逐階段，序 {o}（t = 1 ／ 2 ／ 3 ／ 4）")
        w("")
        w("| row | CIL ACC | Masked（Table 1） | WP（告訴任務） |")
        w("|---|---|---|---|")
        for row, _ in ROWS:
            if row.startswith("M2"):
                continue
            f = lambda c: " ／ ".join((pm(ms(A[(row, o, t)][c])) if A[(row, o, t)].get(c) else "—") for t in range(1, 5))   # noqa: E731
            w(f"| {row} | {f('CIL_ACC')} | {f('MaskedACC_table1')} | {f('WP')} |")
        w("")
    w("### K7　重算一致性（與既有數字逐格差 ≤ 1e-4）")
    w("")
    w("| 比對來源 | 格數 | 最大絕對差 | 超出 1e-4 的格數 |")
    w("|---|---|---|---|")
    for g, v in k7["groups"].items():
        w(f"| {g} | {v['cells']} | {v['max_abs']:.2e} | {v['bad']} |")
    w(f"| **合計** | **{k7['cells']}** | | **{k7['n_bad']}** |")
    w("")
    if k7["bad"]:
        w("| 來源 | 列 | 序 | 折 | 指標 | 舊值 | 新值 | 差 |")
        w("|---|---|---|---|---|---|---|---|")
        for b in k7["bad"]:
            w(f"| {b['source']} | {b['row']} | {b['order']} | {b['fold']} | {b['metric']} | {b['old']:.6f} | {b['new']:.6f} | {b['diff']:+.6f} |")
    else:
        w("沒有差 > 1e-4 的格。**K7 通過。**")
    w("")
    w("K7 沒有涵蓋的格（沒有既有數字可比，本批第一次算出）：FINAL 與各 ridge 列的 Masked ACC（Table 1 定義）；M1、G1、G2、RF 的 t < 4、Forgetting、BWT；"
      "K32、K128、K256 的 t < 4、Forgetting、BWT。")
    w("")

    # ── D ──
    w("## D1　逐折配對（t = 4 CIL ACC；差 = FINAL − 對照）與 slide 層級 bootstrap")
    w("")
    w("| 序 | 比較 | FINAL mean ± sd | 對照 mean ± sd | 平均差 | FINAL 贏／輸／平手 | 符號檢定 p（雙尾） | bootstrap 95% CI（1000 次，seed 0） |")
    w("|---|---|---|---|---|---|---|---|")
    for o in ORDERS:
        for kname, r in d["D1"][o].items():
            w(f"| {o} | {kname} | {pm(r['a'])} | {pm(r['b'])} | {sg(r['mean_diff'])} | {r['wins']}／{r['losses']}／{r['ties']} | {pv(r['p_sign'])} | "
              f"[{sg(r['boot_ci95'][0])}, {sg(r['boot_ci95'][1])}] |")
    w("")
    w("- bootstrap：單位 = test slide；在每個（折、任務）層內有放回重抽；統計量 = 十折平均的四任務等權 CIL ACC 差；兩個系統用同一組索引。")
    w("- 5seed：每折先對 seed 42–46 平均；NOHEAD、ZS8、LIN8 與 seed 無關。十折 n = 10 時，符號檢定雙尾 p 的最小值為 0.0020（10／0）。")
    w("")
    w(f"## D2　成本（Mac CPU、8 執行緒；fold {d2['fold']}、{d2['order']}、seed {d2['seed']}）")
    w("")
    w("| 推論（FINAL，每張 slide） | 平均 | 中位數 | 最小 ～ 最大 |")
    w("|---|---|---|---|")
    for key, lab, f in (("n_patch", "patch 數", "{:.1f}"), ("n_scored_by_head", "head 評分的 patch 數（= 全部 patch）", "{:.1f}"),
                        ("n_selected", "選出張數", "{:.0f}"), ("t_read_s", "讀檔秒數", "{:.4f}"), ("t_compute_s", "計算秒數", "{:.4f}"),
                        ("t_total_s", "合計秒數", "{:.4f}")):
        s = d2[key]
        w(f"| {lab} | {f.format(s['mean'])} | {f.format(s['median'])} | {f.format(s['min'])} ～ {f.format(s['max'])} |")
    w("")
    w(f"- test slide 數 {d2['n_slides']}；逐張判定與 C 的 FINAL 不同的張數：{d2['pred_diff_vs_C']}。"
      f"量測時系統 load average {d2['loadavg'][0]:.1f}／{d2['loadavg'][1]:.1f}／{d2['loadavg'][2]:.1f}（非本批的系統程序）；讀檔秒數是檔案已在系統快取內的數字。")
    w("- 計算 = mean_vec → AR → τ̂ 的 head 對全部 patch 評分 → 四輪選 64 → ridge 判讀；AR 與 ridge 的 W 在計時前已解好。")
    w("")
    w("| 訓練：每任務 head 訓練 wall 秒數（seed 43–46 × 十折的平均；5 個 epoch） | ESCA | RCC | BRCA | LUNG |")
    w("|---|---|---|---|---|")
    h = d2["head_train_wall_s_4seed_mean_per_task"]
    w("| 四個 seed 平均 | " + " | ".join(f"{h[t]:.2f}" for t in ("tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung")) + " |")
    for s, v in d2["head_train_wall_s"].items():
        w(f"| seed {s} | " + " | ".join(f"{v[t]['mean']:.2f}" for t in ("tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung")) + " |")
    w("")
    w(f"{d2['head_train_note']}。來源：`moe1/i6_seed{{s}}/fold*_train.json` 的 `wall_s`。")
    w("")
    w("| ridge（fold 1）：序 | 任務 | train 張數 | readout 累加秒數 | readout 求解秒數 | AR 累加＋求解秒數 |")
    w("|---|---|---|---|---|---|")
    for r in d2["ridge"]:
        w(f"| {r['order']} | {r['task']} | {r['n_train']} | {r['readout_accumulate_s']:.4f} | {r['readout_solve_s']:.4f} | {r['ar_accumulate_and_solve_s']:.4f} |")
    w("")
    w("ridge 的秒數不含取向量（每張 train slide 讀檔、head 評分、四輪選片）；向量取自既有快取。")
    w("")
    w("## D3　γ 敏感度（事後分析，只放補充；γ 的選擇仍由 MOE-3 的逐階段 validation 決定，定案值 1e-3 不變）")
    w("")
    w("只改 ridge readout 的 γ；AR 的 γ 維持 1e-3。十折 mean ± sd。")
    w("")
    w("| 序 | γ | FINAL(42) CIL ACC（t = 4） | FINAL 五 seed CIL ACC（t = 4） | FINAL(42) Ā | FINAL 五 seed Ā |")
    w("|---|---|---|---|---|---|")
    for o in ORDERS:
        for g, r in d["D3"][o].items():
            w(f"| {o} | {float(g):g} | {pm(r['seed42_cil4'])} | {pm(r['5seed_cil4'])} | {pm(r['seed42_abar'])} | {pm(r['5seed_abar'])} |")
    w("")
    w("## D4　TP 混淆矩陣、逐任務 WP、FINAL 逐類正確率（t = 4）")
    w("")
    for o in ORDERS:
        w(f"TP（AR，γ = 1e-3）4 × 4，序 {o}；列 = 真實任務、欄 = τ̂；十折合計張數：")
        w("")
        w("| 真實＼τ̂ | " + " | ".join(TASKS) + " | 合計 | 每折 TP 正確率 mean ± sd |")
        w("|---|---|---|---|---|---|---|")
        for a, row in enumerate(d["D4"]["tp_confusion"][o]):
            w(f"| {TASKS[a]} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} | {pm(d['D4']['tp_task_acc'][o][a])} |")
        w("")
    w("FINAL 逐任務 WP（告訴任務；十折 mean ± sd）：")
    w("")
    w("| | " + " | ".join(TASKS) + " |")
    w("|---|---|---|---|---|")
    for lab in ("seed42", "5seed"):
        w(f"| {lab} | " + " | ".join(pm(x) for x in d["D4"]["final_wp_task"][lab]) + " |")
    w("")
    w("FINAL 逐類正確率（正確張數／張數；十折合計；5seed 為五個 seed 合計）：")
    w("")
    w("| 類別 | 告訴任務，seed 42 | 告訴任務，5seed | CIL，seed 42 | CIL，5seed |")
    w("|---|---|---|---|---|")
    pc = d["D4"]
    for c in pc["final_per_class_tell"]["seed42"]:
        cell = lambda x: f"{x[0]}／{x[1]}（{x[0] / x[1]:.4f}）"                # noqa: E731
        w(f"| {c} | {cell(pc['final_per_class_tell']['seed42'][c])} | {cell(pc['final_per_class_tell']['5seed'][c])} | "
          f"{cell(pc['final_per_class_cil']['seed42'][c])} | {cell(pc['final_per_class_cil']['5seed'][c])} |")
    w("")
    w("出處與核對：告訴任務的五 seed 逐類計數與 `REPORT_moe4.md` H3（`FINAL_RESULTS.md` D 表）相同；TP 每任務正確率與 `REPORT_moe0.md` T1 相同。"
      "CIL 的逐類計數為本批新算。" + pc["note"] + "。")
    w("")
    w("## 未完成項與待 PI 決定")
    w("")
    w("| 項目 | 原因 | 需要的決定 |")
    w("|---|---|---|")
    w("| B2–B5、K8（MergeSlide 同折數字） | 補充 4 的停止條件成立；不跑（`outputs/external/mergeslide/DECISION.md`） | 從做法 1–5 選一個；做法 1 需要約 2.6 TB 的原始 WSI 與一台 GPU pod（DECISION.md） |")
    w("| M2 的 CIL、t < 4 | 既有快取沒有 τ̂ 文字 × 換 head 的版本（DECISIONS D17） | 是否要重讀特徵檔新算 |")
    w("")
    w("已由 PI 裁決（2026-10-03，見 DECISIONS D2、D10、D15、D26）：M1 = MOE-0 B5、M2 = MOE-0 B2；外部對照不受紅線 4 約束、表註寫機器與日期；融合系統的 Masked ACC 不進任何表；外部方法評估器的 `acc@mid` 是不含 test 資訊的 argmax 正確率，不換欄。reverse 的外部批次取 b8（DECISIONS D8）未被推翻。")
    w("")
    w("## 失敗")
    w("")
    if fails:
        for p in fails:
            w(f"### {p.name}")
            w("")
            w("```")
            w(p.read_text().strip()[-3000:])
            w("```")
            w("")
    else:
        w("沒有失敗的階段。K7 通過；K6 見 A 節（一個不進配對表的批次不過）；K8 沒有執行。")
        w("")
    w("## DECISIONS.md（`ext1/DECISIONS.md` 全文）")
    w("")
    w((BASE / "ext1" / "DECISIONS.md").read_text().strip())
    w("")
    (BASE / "REPORT_ext1.md").write_text("\n".join(L))
    print(f"REPORT_ext1.md：{len(L)} 行")


if __name__ == "__main__":
    main()
