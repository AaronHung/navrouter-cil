#!/usr/bin/env python3
"""EXT-2 報告（PREREG-22 細則 26）：由 ext2/*.json、ext2/a/ 的逐折輸出與 outputs/external/ 產生
FINALB_RESULTS.md 與 REPORT_ext2.md。只給數字與表格。外部對照的名稱與表註一律讀 outputs/external/<dir>/meta.json，本檔不寫死。

    NAVCIL_MACHINE=mac python scripts/ext2_report.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.text_encoder import load_config                             # noqa: E402

BASE = REPO_ROOT / "outputs" / "navcil" / load_config()["machine"]
E2 = BASE / "ext2"
EXT = REPO_ROOT / "outputs" / "external"
ORDERS = ("reverse", "paper")
FOLDS = list(range(1, 11))
SEEDS = (42, 43, 44, 45, 46)
TASKS = ["ESCA", "RCC", "BRCA", "LUNG"]
TKEYS = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
LABEL = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]
PREREG_COMMIT = "0b49cc5"


def J(p: Path):
    return json.loads(p.read_text())


def ms(xs):
    xs = [float(x) for x in xs]
    return statistics.fmean(xs), statistics.stdev(xs)


def pm(x) -> str:
    return f"{x[0]:.4f} ± {x[1]:.4f}"


def sg(x: float) -> str:
    return f"{x:+.4f}"


def pv(p) -> str:
    return "不適用" if p is None else f"{p:.4f}"


def rows_of(row: str, o: str) -> list[dict]:
    return [J(E2 / "a" / row / f"{o}_fold{f}.json") for f in FOLDS]


def rows5(base: str, o: str) -> list[dict]:
    """每折先對 seed 42–46 平均。"""
    out = []
    for f in FOLDS:
        rs = [J(E2 / "a" / (base if s == 42 else f"{base}_s{s}") / f"{o}_fold{f}.json") for s in SEEDS]
        d = {}
        for k in ("acc_t", "wp_t", "mk1_t"):
            d[k] = [statistics.fmean(r[k][t] for r in rs) for t in range(4)]
        for k in ("forgetting", "bwt"):
            d[k] = statistics.fmean(r[k] for r in rs)
        out.append(d)
    return out


def t4(rs, k):
    return pm(ms([r[k][3] for r in rs]))


def stages(rs, k):
    return " ／ ".join(pm(ms([r[k][t] for r in rs])) for t in range(4))


def abar(rs):
    return pm(ms([sum(r["acc_t"]) / 4 for r in rs]))


def task_vec(rs, k, t):
    """每任務十折平均（未學為 —）。"""
    return "[" + " ".join("—" if rs[0][k][t][q] is None else f"{statistics.fmean(r[k][t][q] for r in rs):.3f}" for q in range(4)) + "]"


# ── A：FINAL-B 全套（FINALB_RESULTS.md 與 REPORT_ext2.md 共用）─────────────────
def sec_gamma(w, a) -> None:
    g = a["gamma"]
    w("| γ | " + " | ".join(f"{x:g}" for x in g["grid"]) + " |")
    w("|---|" + "---|" * len(g["grid"]))
    w("| validation 目標值（兩序 × t = 1…4 的告訴任務 WP 平均；十折平均） | " + " | ".join(
        (f"**{v:.4f}**" if float(k) == a["gamma_B"] else f"{v:.4f}") for k, v in g["obj"].items()) + " |")
    w("| `moe3/hp.json` 的 `RDG:g0`（MOE-3 當時的值） | " + " | ".join(f"{v:.4f}" for v in g["K10"]["moe3_obj"].values()) + " |")
    w("")
    edge = "在邊界" if g["star"]["edge_gamma"] else "不在邊界"
    w(f"選定 γ\\_B = **{a['gamma_B']:g}**（{edge}）；MOE-3 當時對同一種向量選到 {g['moe3_star']['gamma']:g}。"
      f"K10：兩列最大絕對差 {g['K10']['max_abs']:.1e}（{'通過' if g['K10']['pass'] else '不過'}）。"
      + ("γ\\_B 與 FINAL-A 的定案值 1e-3 相同，沒有第二組數字。" if a["gamma_B"] == 1e-3 else "γ\\_B ≠ 1e-3：另列 1e-3 的數字。"))
    w("")


def sec_A(w, a, b, cost) -> None:
    st = a["storage"]
    RB = {o: rows_of("FINALB", o) for o in ORDERS}
    w("### A-1　t = 4（十折 mean ± sd；test）")
    w("")
    w("| 序 | CIL ACC | WP（告訴任務） | Masked ACC（Table 1） | Forgetting | BWT | TP 正確率 | Ā | 儲存 bytes（T = 4） |")
    w("|---|---|---|---|---|---|---|---|---|")
    for o in ORDERS:
        rs = RB[o]
        w(f"| {o} | **{t4(rs, 'acc_t')}** | {t4(rs, 'wp_t')} | {t4(rs, 'mk1_t')} | {pm(ms([r['forgetting'] for r in rs]))} | "
          f"{pm(ms([r['bwt'] for r in rs]))} | {t4(rs, 'tp_t')} | {abar(rs)} | {st['systems']['FINALB']['T4_bytes']:,} |")
    w("")
    sm = a["order_same_t4"]
    w(f"t = 4 時兩序的判定逐張比對（{sm['n']} 張）：告訴任務不同 {sm['tell_diff']} 張、CIL 不同 {sm['cil_diff']} 張。")
    w("")
    w("### A-2　逐階段（t = 1 ／ 2 ／ 3 ／ 4；十折 mean ± sd）")
    w("")
    w("| 序 | 指標 | t = 1 ／ 2 ／ 3 ／ 4 |")
    w("|---|---|---|")
    for o in ORDERS:
        rs = RB[o]
        for lab, k in (("CIL ACC", "acc_t"), ("WP（告訴任務）", "wp_t"), ("Masked ACC（Table 1）", "mk1_t"), ("TP 正確率（已學任務等權）", "tp_t")):
            w(f"| {o} | {lab} | {stages(rs, k)} |")
        for lab, k in (("每任務 WP [ESCA RCC BRCA LUNG]", "wp_task_t"), ("每任務 CIL ACC", "cil_task_t"), ("每任務 TP 正確率", "tp_task_t")):
            w(f"| {o} | {lab}（十折平均） | " + " ／ ".join(task_vec(rs, k, t) for t in range(4)) + " |")
    w("")
    w("t = 1 只有一個任務，TP 正確率依定義為 1。")
    w("")
    w("### A-3　逐任務（t = 4；十折 mean ± sd；兩序相同）")
    w("")
    w("| | " + " | ".join(TASKS) + " | 四任務等權 |")
    w("|---|---|---|---|---|---|")
    rs = RB["reverse"]
    for lab, k, tot in (("WP（告訴任務）", "wp_task_t", "wp_t"), ("CIL ACC", "cil_task_t", "acc_t"), ("TP 正確率", "tp_task_t", "tp_t")):
        w(f"| {lab} | " + " | ".join(pm(ms([r[k][3][q] for r in rs])) for q in range(4)) + f" | {t4(rs, tot)} |")
    w("")
    w("### A-4　逐類正確率（t = 4；正確張數／張數，十折合計；reverse 序，兩序逐張相同）")
    w("")
    w("| 類別 | 告訴任務 | CIL |")
    w("|---|---|---|")
    cell = lambda x: f"{x[0]}／{x[1]}（{x[0] / x[1]:.4f}）"                    # noqa: E731
    for c in LABEL:
        w(f"| {c} | {cell(a['per_class_tell'][c])} | {cell(a['per_class_cil'][c])} |")
    w("")
    w("| 任務 | balanced accuracy（告訴任務） | balanced accuracy（CIL） |")
    w("|---|---|---|")
    bal = {"tell": [], "cil": []}
    for q in range(4):
        for key, src in (("tell", a["per_class_tell"]), ("cil", a["per_class_cil"])):
            bal[key].append(statistics.fmean(src[LABEL[2 * q + j]][0] / src[LABEL[2 * q + j]][1] for j in (0, 1)))
        w(f"| {TASKS[q]} | {bal['tell'][-1]:.4f} | {bal['cil'][-1]:.4f} |")
    w(f"| 四任務平均 | {statistics.fmean(bal['tell']):.4f} | {statistics.fmean(bal['cil']):.4f} |")
    tot = lambda src: sum(v[0] for v in src.values()) / sum(v[1] for v in src.values())   # noqa: E731
    w(f"| 全部張數正確率 | {tot(a['per_class_tell']):.4f} | {tot(a['per_class_cil']):.4f} |")
    w("")
    w("### A-5　TP（AR，γ = 1e-3）的 4 × 4 混淆矩陣（t = 4；列 = 真實任務、欄 = τ̂；十折合計張數）")
    w("")
    same = a["tp_confusion"]["reverse"] == a["tp_confusion"]["paper"]
    for o in (("reverse",) if same else ORDERS):
        w(f"序 {o}" + ("（paper 序的矩陣逐格相同）" if same else "") + "：")
        w("")
        w("| 真實＼τ̂ | " + " | ".join(TASKS) + " | 合計 | 正確率（合計） |")
        w("|---|---|---|---|---|---|---|")
        for i, row in enumerate(a["tp_confusion"][o]):
            w(f"| {TASKS[i]} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} | {row[i] / sum(row):.4f} |")
        w("")
    w("TP 只用 mean_vec，與 head 無關，所以這張矩陣與 FINAL-A 的相同（`REPORT_ext1.md` D4）。")
    w("")
    w("### A-6　儲存（fp32；由實際張量形狀計算）")
    w("")
    w("| 項目 | 內容 | 數值個數 | bytes |")
    w("|---|---|---|---|")
    for k, v in st["items"].items():
        w(f"| {k} | {v['what']} | {v['n_float']:,} | {v['fp32_bytes']:,} |")
    w("")
    w("| 系統 | 共用 S（bytes） | 每任務 p（bytes） | T = 4 總量 | S + T·p | 每任務參數（個） |")
    w("|---|---|---|---|---|---|")
    for key, lab in (("FINALB", "**FINAL-B**"), ("FINALA", "FINAL-A"), ("MAIN", "主系統"), ("LIN8", "LIN8"), ("ZS8", "zero-shot 8 類")):
        s = st["systems"][key]
        sh = " ＋ ".join(s["shared"]) or "—"
        pt = " ＋ ".join(s["per_task"]) or "—"
        w(f"| {lab} | {s['shared_bytes']:,}（{sh}） | {s['per_task_bytes']:,}（{pt}） | {s['T4_bytes']:,} | "
          f"{s['shared_bytes']:,} + {s['per_task_bytes']:,} T | {s['per_task_params']:,} |")
    w("")
    sh = st["shapes"]
    w(f"實測 shape（fold 1、t = 4）：A {tuple(sh['A'])}、B {tuple(sh['B_t4'])}、W {tuple(sh['W_t4'])}、AR 的 W {tuple(sh['W_ar_t4'])}。"
      f"每任務參數 ＝ 由該任務 train 資料得到的數值個數（head、B、B_ar），不含兩類文字特徵。{st['not_counted']}。")
    w("")
    inf = cost["inference"]
    w(f"### A-7　每張 slide 的推論秒數（fold {cost['fold']} 全部 test，{inf['FINALB']['n_slides']} 張；CPU、{cost['threads']} 執行緒）")
    w("")
    w("| | FINAL-B | FINAL-A(42) | FINAL-B 第二遍 |")
    w("|---|---|---|---|")
    for key, lab in (("t_read_s", "讀檔秒數"), ("t_compute_s", "計算秒數"), ("t_total_s", "合計秒數")):
        f = lambda s: f"平均 {s['mean']:.4f}／中位數 {s['median']:.4f}／最大 {s['max']:.4f}"       # noqa: E731
        w(f"| {lab} | {f(inf['FINALB'][key])} | {f(inf['FINALA'][key])} | {f(inf['FINALB_second_pass'][key])} |")
    npn = inf["FINALB"]["n_patch"]
    w(f"| patch 數（全部都要算 s0 或過 head） | 平均 {npn['mean']:.1f}／中位數 {npn['median']:.0f}／{npn['min']:.0f} ～ {npn['max']:.0f} | 同左 | 同左 |")
    w(f"| 選出張數 | {inf['FINALB']['n_selected']['mean']:.0f} | {inf['FINALA']['n_selected']['mean']:.0f} | — |")
    w("")
    la, lb = cost["loadavg_start"], cost["loadavg_end"]
    w(f"- 計算 = mean_vec → AR → τ̂ 的分數（FINAL-B：s0；FINAL-A：head）→ 四輪選 64 → ridge 判讀；AR 與 ridge 的 W 在計時前解好。"
      f"三遍依序執行（FINAL-B、FINAL-A、FINAL-B 第二遍），各自讀檔。")
    w(f"- load average：開始 {la[0]:.1f}／{la[1]:.1f}／{la[2]:.1f}，結束 {lb[0]:.1f}／{lb[1]:.1f}／{lb[2]:.1f}（非本批的系統程序）。"
      f"逐張判定與 A 階段不同的張數：FINAL-B {inf['FINALB']['ok_diff_vs_A_stage']}、FINAL-A {inf['FINALA']['ok_diff_vs_A_stage']}。")
    w("")
    w("### A-8　K ∈ {32, 64, 128, 256} 與「一次 top-64 對四輪」（t = 4；十折 mean ± sd）")
    w("")
    w("K 列 ＝ 依 s0 一次取前 K 個（不扣冗餘、不用 head）的向量 u_K，判讀器同 FINAL-B（ridge，γ\\_B）。FINAL-B 本身是四輪各 16。")
    w("")
    w("| 序 | 列 | CIL ACC | WP | Masked ACC（Table 1） | Forgetting | BWT | Ā |")
    w("|---|---|---|---|---|---|---|---|")
    for o in ORDERS:
        for lab, key in [("**FINAL-B（四輪 4 × 16）**", "FINALB")] + [(f"一次取前 {K}" + ("（一次 top-64）" if K == 64 else ""), f"FINALB_K{K}") for K in (32, 64, 128, 256)]:
            s = b["summary"][o][key]
            w(f"| {o} | {lab} | {pm(s['ACC'])} | {pm(s['WP'])} | {pm(s['MaskedACC_table1'])} | {pm(s['Forgetting'])} | {pm(s['BWT'])} | {pm(s['Abar'])} |")
    w("")
    w("逐折配對（差 = FINAL-B − 該列；t = 4 CIL ACC；bootstrap 1000 次、seed 0）：")
    w("")
    w("| 序 | 比較 | 平均差 | FINAL-B 贏／輸／平手 | exact binomial p（雙尾） | bootstrap 95% CI | WP 的平均差（贏／輸／平手） |")
    w("|---|---|---|---|---|---|---|")
    for o in ORDERS:
        for k, v in b["K"][o].items():
            r, r2 = v["ACC"], v["WP"]
            w(f"| {o} | {k} | {sg(r['mean_diff'])} | {r['wins']}／{r['losses']}／{r['ties']} | {pv(r['p_sign'])} | "
              f"[{sg(r['boot_ci95'][0])}, {sg(r['boot_ci95'][1])}] | {sg(r2['mean_diff'])}（{r2['wins']}／{r2['losses']}／{r2['ties']}） |")
    w("")
    w("四輪版本的 K ≠ 64 沒有既有定義（每輪幾張未定），沒有算（DECISIONS D5）。")
    w("")
    w("### A-9　γ 敏感度（test；事後描述，不改 γ\\_B）")
    w("")
    w("| 序 | γ | CIL ACC（t = 4） | WP（t = 4） | Ā |")
    w("|---|---|---|---|---|")
    for o in ORDERS:
        for g, v in a["sens"][o].items():
            mark = "（γ\\_B）" if float(g) == a["gamma_B"] else ""
            w(f"| {o} | {float(g):g}{mark} | {pm(ms(v['cil4']))} | {pm(ms(v['wp4']))} | {pm(ms(v['abar']))} |")
    w("")
    w("只改 ridge 判讀的 γ；AR 的 γ 維持 1e-3。γ\\_B 由上方的 validation 目標決定。")
    w("")


def pair_table(w, cells: dict, a_lab: str, b_lab: str, o: str) -> None:
    w(f"| 比較 | 指標 | {a_lab} mean ± sd | {b_lab} mean ± sd | 平均差 | 較高／較低／平手的折數 | {a_lab} 較好的折數／10 | exact binomial p（雙尾） | bootstrap 95% CI（1000 次，seed 0） |")
    w("|---|---|---|---|---|---|---|---|---|")
    for name, cell in cells.items():
        for mk, r in cell.items():
            better = r["losses"] if mk == "Forgetting" else r["wins"]
            ci = f"[{sg(r['boot_ci95'][0])}, {sg(r['boot_ci95'][1])}]" if r.get("boot_ci95") else "—"
            lab = {"ACC": "CIL ACC", "MaskedACC_table1": "Masked ACC（Table 1）", "Abar": "Ā"}.get(mk, mk)
            w(f"| {name} | {lab} | {pm(r['a'])} | {pm(r['b'])} | {sg(r['mean_diff'])} | {r['wins']}／{r['losses']}／{r['ties']} | {better} | {pv(r['p_sign'])} | {ci} |")
    w("")


def ext_notes(meta: dict, e: dict, o: str, disp: str) -> list[str]:
    """外部對照的表註：機器與日期（PI 2026-10-03）＋ PREREG-22 第一部分 0-1a 的兩句。"""
    out = []
    md = meta.get("machine_date", {}).get(o)
    if md:
        out.append(f"表註（機器與日期）：{disp} 的數字在 {md} 產生；我方的數字在 Mac M1 Pro（CPU），2026-10-03（EXT-2 批）產生。外部對照不受 AGENTS.md 紅線 4 約束（PI 2026-10-03 裁決）。")
    opt = e.get("theirs_optthr")
    mid = e["theirs"]["MaskedACC"]
    opt_s = f"（{o} 序十折平均：acc@mid {mid[0]:.4f}、test 最佳門檻的 acc {opt[0]:.4f}）" if opt else ""
    out.append(f"表註（批次與匯入欄）：reverse 序為 b8（論文設定）、paper 序為 b16。匯入欄為 acc@mid（argmax、不含 test 資訊），"
               f"該欄在兩序都高於 test 最佳門檻的 acc{opt_s}，對外部方法有利。")
    return out


def main() -> None:
    a, b, cost, d, ex = (J(E2 / f"{n}.json") for n in ("a", "b", "cost", "d", "example"))
    fails = sorted(E2.glob("FAILED_*.txt"))
    ext_dirs = sorted(p for p in EXT.iterdir() if p.is_dir()) if EXT.is_dir() else []
    metas = {p.name: (J(p / "meta.json") if (p / "meta.json").exists() else {"display_name": p.name}) for p in ext_dirs}
    k9, k11 = a["K9"], cost["K11"]
    head = ("機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0、closed-form float64。我方的數字全部來自同一台、同一批"
            "（EXT-2，2026-10-03：`scripts/ext2_a.py`、`ext2_cost.py`、`ext2_b.py`、`ext2_example.py`、`ext2_d.py`）；只讀既有快取、權重與特徵檔，"
            f"沒有訓練任何模型，沒有在 test 上調任何東西。判準與操作定義見 `PREREG-22.md`（commit {PREREG_COMMIT}）。")
    definition = [
        "| | FINAL-A（原 FINAL，定義不動） | FINAL-B |",
        "|---|---|---|",
        "| 任務判定（TP） | AR：[mean_vec; 1] 的累加式 ridge，γ = 1e-3 | 相同 |",
        "| 每個 patch 的分數 | s = s0 + g（I6(r = 2) head，每任務 1,033 個參數，seed 42–46） | **s = s0**（對該任務兩類文字 cosine 的最大值，單張 slide 內 z-score）；沒有 head、沒有 seed |",
        "| 選片 | 四輪各 16，λ = 1.5 | 相同 |",
        "| slide 向量 | 64 個原始 Z 等權平均後 L2 正規化，補 1 | 相同 |",
        f"| 判讀 | [v; 1] 累加式 8 類 ridge，γ = 1e-3，τ̂ 兩類內 argmax | 相同；γ\\_B = {a['gamma_B']:g}（本批用 validation 重選） |",
        "| 訓練 | 每任務訓練一顆 head（5 個 epoch）＋ closed-form | 只有 closed-form（累加 A、B） |",
        ""]

    # ══ FINALB_RESULTS.md ══
    L = []
    w = L.append
    w("# FINAL-B 結果（EXT-2，2026-10-03）")
    w("")
    w(head)
    w("")
    w("逐折值：`FINALB_perfold.csv`（欄位與 `ABLATION_full.csv` 相同；列 `FINALB`、`FINALB_K32…256`，以及本批重算的對照列 `ZS8`、`LIN8`、`MAIN*`、`FINALA*`）；"
      "TP 正確率與逐任務值在 `FINALB_perfold_extra.csv`；每列每 (order, fold) 的 JSON 在 `ext2/a/<row>/`。兩套並排、外部對照與總表在 `REPORT_ext2.md` 的 B 節；逐步實例在 `FINALB_EXAMPLE.md`。")
    w("")
    w("## 系統定義")
    w("")
    L.extend(definition)
    w("縮寫：CIL ACC ＝ 8 類 CIL 正確率；WP ＝ 告訴任務判定（已學任務等權）；Masked ACC（Table 1）＝ 向量取自 τ̂ 的版本、在真實任務兩類內 argmax；Ā ＝ 四階段 CIL ACC 平均。")
    w("")
    w("## γ 的重選（只用 validation；PREREG-19 細則 5 的逐階段目標，grid 同 MOE-3）")
    w("")
    sec_gamma(w, a)
    w("## 與既有「拿掉 head」列的關係")
    w("")
    w(f"FINAL-B 就是 `ABLATION_full.csv` 的 `NOHEAD` 列（MOE-3 的 S-R0f：v0／RDG γ = 1e-3）。本批由同一批快取重算一次，並做兩項確認："
      f"K9（本批重算的 {len(a['rows_ref'])} 列對 `ext1/c/` 的逐折值，{k9['cells']} 格）最大絕對差 {k9['max_abs']:.1e}；"
      f"K11（fold 1 不載入 head、由特徵檔重算）train 向量最大絕對差 {k11['a_train_v_max_abs']:.1e}、test 逐張判定不同 {k11['b_test_pred_diff']} 張。")
    w("")
    w("## A　FINAL-B 全套（十折兩序，t = 1…4）")
    w("")
    sec_A(w, a, b, cost)
    (BASE / "FINALB_RESULTS.md").write_text("\n".join(L) + "\n")
    n1 = len(L)

    # ══ REPORT_ext2.md ══
    L = []
    w = L.append
    w("# REPORT — EXT-2：FINAL-B（不訓練的變體）全套；FINAL-A 與 FINAL-B 並排；範例與質性圖數據（Mac CPU，十折，reverse 與 paper 兩序）")
    w("")
    w(head + "外部對照的數字來自其他機器，來源見其 PROVENANCE。")
    w("")
    w("## 狀態")
    w("")
    w("| 階段 | 狀態 |")
    w("|---|---|")
    w(f"| 0　PREREG-22 | commit {PREREG_COMMIT}（執行前）。γ 重選：γ\\_B = {a['gamma_B']:g}。「拿掉 head」列 ＝ FINAL-B 的定義：是，直接引用 |")
    w(f"| K9　本批重算 ＝ EXT-1 逐折值 | {k9['cells']} 格，最大絕對差 {k9['max_abs']:.1e}，超出 1e-9 的 {k9['n_bad']} 格（{'通過' if k9['pass'] else '不過'}） |")
    w(f"| K10　γ 的 validation 目標值 ＝ moe3/hp.json | 最大絕對差 {a['gamma']['K10']['max_abs']:.1e}（{'通過' if a['gamma']['K10']['pass'] else '不過'}） |")
    w(f"| K11　由特徵檔重算 ＝ 快取（fold 1） | (a) train 向量最大絕對差 {k11['a_train_v_max_abs']:.1e}；(b) test 逐張判定不同 {k11['b_test_pred_diff']} 張（{'通過' if k11['pass'] else '不過'}） |")
    w(f"| A　FINAL-B 全套 | 完成（`FINALB_RESULTS.md`、`FINALB_perfold.csv`、`FINALB_perfold_extra.csv`） |")
    w("| B　兩套並排、外部對照、總表 | 完成 |")
    w("| C　FINAL-B 範例 | 完成（`FINALB_EXAMPLE.md`） |")
    w(f"| D　質性圖數據 | D1 完成；D2、D3 **跳過**（{d['D2']['reason']}） |")
    w(f"| 失敗的階段 | {len(fails)} |")
    w("")
    w("## 0-1　承 EXT-1 的三個裁決（DECISIONS D1–D3）")
    w("")
    w("| 裁決 | 在本報告的落點 |")
    w("|---|---|")
    w("| a. 外部對照表註加兩句（reverse = b8、paper = b16；匯入欄 acc@mid 在兩序都高於 test 最佳門檻的 acc，對外部方法有利）；不重跑 paper 序 | B2、B3 每張表下方的表註；沒有重跑 |")
    w("| b. MergeSlide 做法 1–5 都不做；論文只放相關工作與表註 | 本批的表沒有 MergeSlide 列；成本見 `outputs/external/mergeslide/DECISION.md` |")
    w("| c. M2 只報 WP；CIL 與 t < 4 填「—」，不補算 | 本批沒有重算 M2；`ABLATION_full.csv` 的 M2 列維持原狀 |")
    w("")
    w("## 0-2　系統定義")
    w("")
    L.extend(definition)
    w("## 0-3　「拿掉 head」列是不是 FINAL-B 的定義")
    w("")
    w("結論：**是**，直接引用（`ABLATION_full.csv` 的 `NOHEAD` ＝ MOE-3 的 S-R0f）。逐項對照：")
    w("")
    w("| FINAL-B 定義的項目 | `NOHEAD` 列的實作 | 檔名：行號 | 相符 |")
    w("|---|---|---|---|")
    w("| s_i = s0_i（對該任務兩類文字的最大 cosine，z-score） | `s0, g = head.parts(Z, ft)` 取 s0，g 不用；s0 = `zscore(text_nav_feats(Z, f_txt)[:, 0])`，不含 head 的任何參數 | `scripts/moe2_common.py:161-163`；`selector/i6_expert.py:37-38`；`selector/flat_selector.py:28-33` | 是 |")
    w("| 四輪各 16、λ = 1.5 | `four_round(Z, s0, lam)`；BUDGET = 64、STEP = 16；λ 讀 `nc1/lambda.json`（1.5） | `scripts/moe2_common.py:163`；`selector/cil_ops.py:22-23`、`:39-45`；`scripts/nc1_pipeline.py:86-87` | 是 |")
    w("| 原始 Z 等權平均後 L2，再補 1 | `mean_norm(Z, idx)`：`X.mean(0)` 後 `F.normalize`；`aug` 接常數 1（float64） | `selector/cil_ops.py:31-36`；`scripts/nc5_report.py:151-154`；`scripts/moe3_common.py:242` | 是 |")
    w("| [v; 1] 累加式 ridge | 依序對已學任務 `A += XᵀX`；B 的類別欄 = 該類 x 之和；`solve(A + γI, B)` | `scripts/moe3_common.py:243`、`:250-258`、`:263` | 是 |")
    w("| γ | `moe3/hp.json` 的 `RDG:g0` = 1e-3（validation 選）；本批重選得同值 | `scripts/ext1_c.py:183`、`:39` | 是 |")
    w("| τ̂ 兩類內 argmax | 向量取 τ̂ 的版本；τ̂ 兩欄分數差 d ≥ 0 判第一類 | `scripts/ext1_c.py:100-101`、`:116-117`；`scripts/moe1_common.py:145-151` | 是 |")
    w("| TP | 階段 t 的 AR（γ = 1e-3）在已學任務中 argmax | `scripts/moe3_common.py:321-322`；`scripts/moe1_common.py:124-129` | 是 |")
    w("")
    w(f"數值確認：K11(a) fold 1 的 train {sum(v['n_train'] for v in cost['train_per_task'].values()):,} 張，不載入任何 head、由特徵檔依定義重算的 v 與 `NOHEAD` 所用的快取最大絕對差 "
      f"{k11['a_train_v_max_abs']:.1e}；K11(b) fold 1 的 test {cost['inference']['FINALB']['n_slides']} 張完整推論，逐張判定不同 {k11['b_test_pred_diff']} 張。"
      f"本批由快取算出的 `FINALB` 列與 `NOHEAD` 列逐折、逐階段相同（K9 最大絕對差 {k9['per_row_max_abs']['NOHEAD']:.1e}）。")
    w("")
    w("一個實作細節（兩套相同，不是差異）：`four_round` 在選片前會把傳入的分數在同一張 slide 內再 z-score 一次（`selector/multiround.py:146-150`）；s0 本身已是 z-score，這一步只差一個 1/(1 + 1e-6) 的倍數，不改排序。")
    w("")
    w("## 0-4　γ 的重選")
    w("")
    sec_gamma(w, a)
    w("## 0-5　程式確認（逐條附檔名與行號）")
    w("")
    w("| 問題 | 答案 | 檔名：行號 |")
    w("|---|---|---|")
    w("| TP 的 x 是「patch 平均後 L2 再補 1」還是「L2 後平均」 | **patch 平均後 L2，再補 1**。`mean_norm(Z)` 先 `X.mean(0)`（原始 patch 特徵，範數約 25，未先正規化）再 `F.normalize`；`aug` 轉 float64 並接常數 1 → 513 維 | "
      "`selector/cil_ops.py:31-36`；快取處 `scripts/nc8_batch.py:116`（train）、`scripts/moe0_infer.py:62`（validation／test）；`scripts/nc5_report.py:151-154`；呼叫 `scripts/nc8_report.py:79`、`scripts/moe1_common.py:128` |")
    w("| TP 的 γ | **1e-3**（`G_AR`） | `scripts/moe1_common.py:36`、`:126`；求解 `scripts/nc8_report.py:87` |")
    w("| TP（AR）的 y 是 one-hot {0,1} 還是 ±1 | **one-hot {0,1}**。B 的第 p 欄 = 任務 p 全部 train slide 的 x 之和（`X.sum(0)`，權重 1.0），等於 XᵀY、Y 為 one-hot；沒有 −1 | `scripts/nc8_report.py:80`、`:86` |")
    w(f"| TP（AR）的 B 的 shape | **[513, t]**（每任務一欄，每學一個任務多一欄；判任務時在已學任務的欄內 argmax，沒有閾值）；t = 4 時 [513, 4]。本批實測 AR 的 W shape {tuple(a['storage']['shapes']['W_ar_t4'])} | `scripts/nc8_report.py:76`、`:86-87` |")
    w("| TP（AR）的 A 是否跨任務共用累加 | **是**。單一個 513 × 513 的 A，對已學任務逐一 `A += XᵀX`；沒有每任務各自的 A。實作上每個（折、序、t）從快取的 train mean_vec 把前 t 個任務重新加總一次，數值等同逐任務累加 | `scripts/nc8_report.py:74-81` |")
    w("| s0 的 z-score 在哪個範圍內算 | **單張 slide 內**（該張的全部 patch）：`(s − s.mean()) / (s.std() + 1e-6)`，std 為樣本標準差；輸入是一張 slide 的 `text_nav_feats(Z, f_txt)[:, 0]`。不用 train 集的統計量 | "
      "`selector/i6_expert.py:21-22`、`:37-38`；v0 的產生處 `scripts/moe2_common.py:161-163`；u_K 的產生處 `scripts/moe3_common.py:171` |")
    w("")
    w("上表的 y、B、A 三列答的是 **TP（AR）** 的 ridge。**讀出（判讀器）的 ridge** 另列於下（EXT-2 收尾更正：原版把讀出的 ridge 只寫在表後的一句話裡，容易誤讀成「B 是每任務一欄」）。")
    w("")
    w("### 0-5b　讀出 ridge 的 B：定義、累加、判類（程式原文）")
    w("")
    w("結論：**每類一欄（t = 4 時 8 欄），在 τ̂ 的兩欄內 argmax**。不是「每任務一欄、閾值判類」。")
    w("")
    for title, fn, lo, hi in (("B 的每一欄的定義（`scripts/moe3_common.py:237-244`）", "moe3_common.py", 237, 244),
                              ("跨任務累加 A、B 與求解 W（`scripts/moe3_common.py:246-263`）", "moe3_common.py", 246, 263),
                              ("分數 = [v; 1] · W，放進固定 8 類序的欄（`scripts/ext1_c.py:121-131`）", "ext1_c.py", 121, 131),
                              ("任務 q 兩欄的分數差與取 τ̂ 的那一個（`scripts/moe1_common.py:145-151`）", "moe1_common.py", 145, 151),
                              ("判類（`scripts/ext1_c.py:114-118`；CIL 的呼叫在 `:100-101`）", "ext1_c.py", 114, 118),
                              ("CIL：向量與欄都取 τ̂（`scripts/ext1_c.py:100-101`）", "ext1_c.py", 100, 101)):
        src = (REPO_ROOT / "scripts" / fn).read_text().splitlines()
        w(f"{title}：")
        w("")
        w("```python")
        for n in range(lo, hi + 1):
            w(f"{n}: {src[n - 1]}")
        w("```")
        w("")
    rd = J(E2 / "readout.json")
    w("| 問題 | 答案 |")
    w("|---|---|")
    w("| B 的一欄是什麼 | 類別 c 的欄 = 該類全部 train slide 的 x = [v; 1] 之和（`X[y == c].sum(0)`），等於 XᵀY、Y 為 **one-hot {0,1}**（每類一欄）；沒有 ±1 |")
    w("| 累加 | 每學一個任務：`A += XᵀX`（單一個 A，跨任務共用），B 多出該任務的**兩欄**；欄依固定 8 類序排列 |")
    w("| 判類的確切規則 | 分數 s = [v; 1] · W（v 取 τ̂ 的文字所選的 64 個 patch）；只看 τ̂ 的兩欄 s[2τ̂]、s[2τ̂ + 1]，d = s[2τ̂] − s[2τ̂ + 1]，**d ≥ 0 判第一類、否則第二類**。這就是兩欄內 argmax（平手判第一類，與 argmax 取較小索引一致）；沒有另外的閾值，其他六欄不參與 |")
    w("| 告訴任務（WP） | 同上，把 τ̂ 換成真實任務 |")
    w("")
    ck = rd["check"]
    w(f"實際存檔（`{rd['file']}`；fold {rd['fold']}、{rd['order']} 序、t = {rd['t']}、γ = {rd['gamma']:g}；由 `scripts/ext2_readout.py` 存檔後讀回印出。"
      "既有產物原本沒有存 A、B、W，每次由快取的 train 向量累加求解；`.pt` 依 `.gitignore` 不進版控，數字在 `ext2/readout.json`）：")
    w("")
    w("| 量 | shape | 說明 |")
    w("|---|---|---|")
    w(f"| A | {tuple(rd['A_shape'])} | {rd['dtype']}；右下角 = {rd['A_last']:.0f}（train 張數） |")
    w(f"| **B** | **{tuple(rd['B_shape'])}** | 欄 = " + "、".join(rd["columns"]) + "；最後一列（常數 1 那一維）= 每類的 train 張數 "
      + "、".join(f"{x:.0f}" for x in rd["B_last_row"]) + f"（合計 {rd['B_last_row_sum']:.0f}） |")
    w(f"| **W** | **{tuple(rd['W_shape'])}** | solve(A + γI, B) |")
    w("| 逐階段的 B／W | " + "、".join(f"t = {t}：{tuple(v['B_shape'])}" for t, v in rd["stages"].items()) + " | 每學一個任務多兩欄 |")
    w(f"| 對照：TP（AR）的 W | {tuple(rd['AR_W_shape'])} | 每任務一欄（" + "、".join(rd["AR_columns"]) + "），在已學任務的欄內 argmax 得 τ̂ |")
    w("")
    w(f"規則核對（fold 1 全部 test {ck['n_test']} 張、t = 4）：「τ̂ 兩欄內 argmax」與「d ≥ 0 判第一類」逐張不同 {ck['argmax_vs_d_rule_diff']} 張；d = 0 的平手 {ck['n_ties']} 張；"
      f"與 A 階段的 FINAL-B 判定不同 {ck['pred_diff_vs_A_stage']} 張。另：若不限制在 τ̂ 兩欄、直接取 8 欄全域 argmax，落在 τ̂ 兩欄之外的有 {ck['global8_argmax_outside_tau_pair']} 張（只是這一折的觀察；系統的規則是限制在 τ̂ 兩欄內）。")
    w("")
    w("## A　FINAL-B 全套（十折兩序，t = 1…4）")
    w("")
    sec_A(w, a, b, cost)

    # ── B ──
    w("## B1　FINAL-A 對 FINAL-B 逐折配對（t = 4；差 = FINAL-A − FINAL-B；同折、同序）")
    w("")
    for o in ORDERS:
        w(f"序 {o}：")
        w("")
        pair_table(w, b["B1"][o], "FINAL-A", "FINAL-B", o)
    w("- 「較好」：CIL ACC、WP、Masked ACC、BWT、Ā 為差 > 0 的折數；Forgetting 為差 < 0 的折數。p 為 exact binomial（雙尾，平手不計）；十折時最小值 0.0020。")
    w("- bootstrap：單位 = test slide；每個（折、任務）層內有放回重抽；統計量 = 十折平均的四任務等權 CIL ACC 差；兩個系統用同一組索引；五 seed 版本每張的正確與否先對 seed 平均。只對 CIL ACC 做。")
    w("- FINAL-A(5seed)：每折先對 seed 42–46 平均，再配對。t = 4 的 CIL ACC、WP、Masked 與序無關，所以兩序的這三列相同；Forgetting、BWT、Ā 與序有關。")
    w("")
    w("附表：FINAL-B 對主系統、zero-shot、LIN8（t = 4 CIL ACC；差 = FINAL-B − 對照）：")
    w("")
    w("| 序 | 比較 | FINAL-B mean ± sd | 對照 mean ± sd | 平均差 | FINAL-B 贏／輸／平手 | exact binomial p（雙尾） | bootstrap 95% CI |")
    w("|---|---|---|---|---|---|---|---|")
    for o in ORDERS:
        for name, cell in b["B1_others"][o].items():
            r = cell["ACC"]
            w(f"| {o} | {name} | {pm(r['a'])} | {pm(r['b'])} | {sg(r['mean_diff'])} | {r['wins']}／{r['losses']}／{r['ties']} | {pv(r['p_sign'])} | "
              f"[{sg(r['boot_ci95'][0])}, {sg(r['boot_ci95'][1])}] |")
    w("")
    for p in ext_dirs:
        e = b["EXT"].get(p.name)
        if not e:
            continue
        meta, disp = metas[p.name], metas[p.name]["display_name"]
        w(f"## B2　FINAL-B 對 {disp} 逐折配對（`outputs/external/{p.name}/perfold.csv`；t = 4；十折；差 = FINAL-B − {disp}）")
        w("")
        for o in ORDERS:
            if not e.get(o):
                continue
            w(f"### 序 {o}")
            w("")
            w(f"| 我方系統 | 指標 | 我方 mean ± sd | {disp} mean ± sd | 平均差 | 我方較高／較低／平手的折數 | 我方較好的折數／10 | 符號檢定 p（雙尾） |")
            w("|---|---|---|---|---|---|---|---|")
            for mk, r in e[o]["FINAL-B"].items():
                better = r["losses"] if mk == "Forgetting" else r["wins"]
                w(f"| FINAL-B | {mk} | {pm(r['a'])} | {pm(r['b'])} | {sg(r['mean_diff'])} | {r['wins']}／{r['losses']}／{r['ties']} | {better} | {pv(r['p_sign'])} |")
            w("")
            for line in ext_notes(meta, e[o], o, disp):
                w(line)
                w("")
            pub = meta.get("published", {}).get(o)
            if pub:
                w(f"表註（發表值，非同折，不進配對）：{pub['table']}　ACC {pub['ACC']}、Masked ACC {pub['MaskedACC']}、Forgetting {pub['Forgetting']}。")
                w("")
        w(f"- 「我方較好」：ACC、Masked ACC、BWT、Ā 為差 > 0 的折數；Forgetting 為差 < 0 的折數。符號檢定 = exact binomial（雙尾，平手不計）。")
        w(f"- {disp} 的 Masked ACC 是告訴任務的版本，對應我方的 oracle 定義（＝ WP）；我方 Table 1 定義的那一列是拿兩個不同定義相比，照列供主表使用。")
        w(f"- {disp} 的逐張預測不在本機，沒有 slide 層級的 bootstrap。只用 K6 通過的批次（`outputs/external/{p.name}/k6.json`）。FINAL-A 對 {disp} 的同一張表在 `REPORT_ext1.md` A 節。")
        w("")

    w("## B3　總表（t = 4；十折 mean ± sd；兩序各一張）")
    w("")
    st = a["storage"]["systems"]
    tr = cost["train_seconds"]

    def secs(key):
        v = [tr[t][key] for t in TKEYS]
        return f"{statistics.fmean(v):.2f}（" + "／".join(f"{x:.2f}" for x in v) + "）"

    hd = cost["head_train_wall_s_recorded"]["four_seed_mean"]
    for o in ORDERS:
        w(f"### 序 {o}")
        w("")
        w("| 系統 | CIL ACC | WP | Masked ACC（Table 1） | Forgetting | BWT | 每任務參數 | 每任務儲存 bytes | 每任務訓練秒數（平均；ESCA／RCC／BRCA／LUNG） |")
        w("|---|---|---|---|---|---|---|---|---|")
        S = b["summary"][o]

        def line(lab, key, skey, sec):
            s = S[key]
            w(f"| {lab} | {pm(s['ACC'])} | {pm(s['WP'])} | {pm(s['MaskedACC_table1'])} | {pm(s['Forgetting'])} | {pm(s['BWT'])} | "
              f"{st[skey]['per_task_params']:,} | {st[skey]['per_task_bytes']:,} | {sec} |")

        line("zero-shot 8 類（top-64，無訓練）", "ZS8", "ZS8", "0")
        line("LIN8（mean_vec，ridge γ = 0.01）", "LIN8", "LIN8", secs("LIN8"))
        for p in ext_dirs:
            e = b["EXT"].get(p.name)
            if e and e.get(o):
                t = e[o]["theirs"]
                w(f"| {metas[p.name]['display_name']}（同折；註 1） | {pm(t['ACC'])} | — | {pm(t['MaskedACC'])}（告訴任務的版本） | {pm(t['Forgetting'])} | {pm(t['BWT'])} | — | — | — |")
        line("主系統，五 seed 平均", "MAIN_5seed", "MAIN", secs("MAIN") + " ＋ head 訓練（註 3）")
        line("主系統，seed 42", "MAIN", "MAIN", "同上")
        line("**FINAL-B**（無 seed）", "FINALB", "FINALB", secs("FINALB"))
        line("**FINAL-A，五 seed 平均**", "FINALA_5seed", "FINALA", secs("FINALA") + " ＋ head 訓練（註 3）")
        line("FINAL-A，seed 42", "FINALA", "FINALA", "同上")
        w("")
        for p in ext_dirs:
            e = b["EXT"].get(p.name)
            if e and e.get(o):
                disp = metas[p.name]["display_name"]
                notes = ext_notes(metas[p.name], e[o], o, disp)
                w(f"註 1（{disp}）：{notes[0].replace('表註（機器與日期）：', '')}{notes[1].replace('表註（批次與匯入欄）：', '')}"
                  f"**它的 Masked 欄為告訴任務定義**（告訴真實任務後在該任務兩類內判；對應我方的 WP 欄），不是 Table 1 定義（τ̂ 的證據、真實任務兩類內判），兩者不可直接當同一欄比較；WP 欄不另填。每任務參數、儲存、訓練秒數既有產物沒有，填「—」。")
                w("")
        w("註 2：zero-shot 與 LIN8 沒有 expert，Masked ACC 的兩種定義相同，WP 欄與 Masked 欄同值。每任務參數 ＝ 由該任務 train 資料得到的數值個數（head 1,033、判讀器 B 兩欄 1,026、AR 的 B 一欄 513；LIN8 為 B 兩欄 1,026），"
          "不含兩類文字特徵（1,024 個，計入儲存 bytes）。共用的 A 不在這兩欄內（見 A-6）。")
        w("")
        w(f"註 3：訓練秒數是本批在 fold 1 量到的部分（讀該任務的 train 特徵檔 ＋ mean_vec ＋ 該系統的 slide 向量 ＋ closed-form 累加與求解；CPU、8 執行緒），**不含 head 訓練**。"
          f"head 訓練沒有在本批重跑；MOE-1 批（2026-10-02，同一台）的既有紀錄為每顆 head 5 個 epoch 的 wall 秒數，四個 seed × 十折平均："
          + "、".join(f"{TASKS[i]} {hd[t]:.2f}" for i, t in enumerate(TKEYS)) + "（seed 42 沒有紀錄）。FINAL-B 與 LIN8、zero-shot 不需要這一項。")
        w("")
        kk = [v for k, v in b["K"][o].items() if "K64" in k][0]["ACC"]
        w(f"註 4：FINAL-B 用四輪各 16。改成一次 top-64（依 s0 一次取前 64 個）的 CIL ACC 為 {pm(S['FINALB_K64']['ACC'])}；四輪 − 一次 top-64 的差異**不顯著**："
          f"平均差 {sg(kk['mean_diff'])}，贏／輸／平手 {kk['wins']}／{kk['losses']}／{kk['ties']}，exact binomial p = {pv(kk['p_sign'])}，"
          f"bootstrap 95% CI [{sg(kk['boot_ci95'][0])}, {sg(kk['boot_ci95'][1])}]（跨 0）。FINAL-B 的定義維持四輪，一次 top-64 列為消融（A-8；DECISIONS D21）。")
        w("")
    w("訓練秒數的組成（fold 1；秒）：")
    w("")
    w("| 任務 | train 張數 | 讀檔 | mean_vec | FINAL-B 的向量（s0 ＋ 四輪） | FINAL-A(42) 的向量（head ＋ 四輪） | closed-form：AR | closed-form：FINAL-B 判讀 | closed-form：FINAL-A 判讀 | closed-form：LIN8 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    cf = {r["task"]: r for r in cost["closed_form"] if r["order"] == cost["order"]}
    for i, t in enumerate(TKEYS):
        v, c = cost["train_per_task"][t], cf[t]
        w(f"| {TASKS[i]} | {v['n_train']} | {v['read_s']:.2f} | {v['mean_vec_s']:.2f} | {v['vec_B_s']:.2f} | {v['vec_A_s']:.2f} | "
          f"{c['ar_accumulate_s'] + c['ar_solve_s']:.4f} | {c['B_accumulate_s'] + c['B_solve_s']:.4f} | {c['A_accumulate_s'] + c['A_solve_s']:.4f} | "
          f"{c['lin8_accumulate_s'] + c['lin8_solve_s']:.4f} |")
    w("")

    # ── C ──
    w("## C　FINAL-B 範例（全文在 `FINALB_EXAMPLE.md`；fold 1，與 `FINAL_EXAMPLE.md` 同兩張）")
    w("")
    w("| slide | 標籤 | TP 分數 [esca, rcc, brca, lung] → τ̂ | s0 的範圍（min／median／max） | 四輪 64 個在 s0 純排序的名次（min／median／max）；名次 > 64 的個數 | 與 FINAL-A(42) 的 64 個的交集 | ‖v‖ | τ̂ 兩類的 ridge 分數差 → FINAL-B 判定 | FINAL-A(42) 的差 → 判定 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for s in ex["slides"]:
        c = s["cil"]
        tp, q, rk = s["tp_scores"], c["s0_q3"], c["sel_rank_in_s0"]
        okB = "正確" if s["correct"]["FINALB_cil"] else "錯"
        okA = "正確" if s["correct"]["FINALA_cil"] else "錯"
        w(f"| {s['task']} test 第 {s['index']} 張（N = {s['n_patch']}） | {s['label']} | {tp[0]:.4f}、{tp[1]:.4f}、{tp[2]:.4f}、{tp[3]:.4f} → {s['tau_hat']} | "
          f"{q[0]:.4f}／{q[1]:.4f}／{q[2]:.4f} | {rk[0]}／{rk[1]:.0f}／{rk[2]}；{c['n_sel_rank_gt64']} 個 | {c['inter_total']}／64 | {c['v_norm']:.6f} | "
          f"{c['d']:+.6f} → {c['pred']}（{okB}） | {c['A_d']:+.6f} → {c['A_pred']}（{okA}） |")
    w("")
    w("檢查：重算的 v 對快取的 v0 最大絕對差 " + "、".join(f"{s['check']['v_max_abs_vs_cache_tau']:.1e}" for s in ex["slides"])
      + "；TP 分數對 `moe5_example/example.json` 最大絕對差 " + "、".join(f"{s['check']['tp_max_abs_vs_moe5']:.1e}" for s in ex["slides"]) + "。")
    w("")

    # ── D ──
    w("## D　質性圖數據")
    w("")
    w("### D1　特徵檔是否帶座標與倍率")
    w("")
    w("| 任務 | 特徵目錄 | 檔數 | 讀到的範例檔 | Python 型別 | shape | dtype | dict 鍵 | 同任務下的 .h5／coord 檔 | 任務目錄下的檔案類型（副檔名：數量） |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for t, v in d["tasks"].items():
        ec = "、".join(f"{k or '（無副檔名）'}：{n}" for k, n in v["file_ext_counts"].items())
        w(f"| {t} | `{v['feat_dir']}` | {v['n_feat_files']} | `{v['example_file']}` | {v['python_type']} | {tuple(v['shape'])} | {v['dtype']} | "
          f"{'、'.join(v['keys']) if v['keys'] else '無（不是 dict）'} | {'、'.join(v['coord_like_files']) or '無'} | {ec} |")
    w("")
    ms_ = d["magnification_strings"]
    w(f"- 座標：**沒有**。每個特徵檔是單一 tensor [N, 512]，沒有欄位；`can_dataset` 與 `~/research/WSI_data` 下也沒有 `.h5` 或檔名含 coord 的檔"
      f"（WSI_data 下找到 {len(d['other_coord_files_under_WSI_data'])} 個）。patch 在 tensor 內的順序對應哪個位置，無法由特徵檔得知。")
    w(f"- 倍率：特徵檔內**沒有**倍率欄位。只有檔名字樣：特徵目錄 `{ms_['feat_dir_name']}`（l1、s256）與標籤表 `{ms_['table_name']}`（x10）；不據此推測實際倍率。")
    w("")
    w("### D2、D3　跳過")
    w("")
    w("| 項目 | 狀態 | 不成立的條件 |")
    w("|---|---|---|")
    w(f"| D2 兩套各四輪的 64 個 patch 座標 CSV | 沒有做 | {d['D2']['reason']} |")
    w(f"| D3 縮圖疊框 PNG | 沒有做 | {d['D3']['reason']} |")
    w("| D4 不下載新切片 | 遵守 | 沒有下載任何檔案、沒有安裝任何套件 |")
    w("")
    w(f"本機切片與資料集的重疊（只比對檔名，沒有讀切片）：`~/research/WSI_data` 下共 {d['n_local_slides']} 個切片檔，其中 {d['n_local_in_feats']} 張的 slide ID 有對應的特徵檔，"
      f"{d['n_local_in_fold1_test']} 張在 fold 1 的 test。")
    w("")
    w("| 本機檔 | 大小（MB） | 有特徵檔的任務 | 在 fold 1 test | fold 1、reverse、t = 4 的 CIL 判定（FINAL-B／FINAL-A(42)） |")
    w("|---|---|---|---|---|")
    for x in d["local_slides"]:
        cc = x.get("fold1_cil_correct")
        cs = "—" if not cc else f"{'對' if cc['FINALB'] else '錯'}／{'對' if cc['FINALA_seed42'] else '錯'}"
        w(f"| `{Path(x['file']).name}` | {x['size_bytes'] / 1e6:.0f} | {'、'.join(x['in_feats']) or '無'} | {'、'.join(x['in_fold1_test']) or '否'} | {cs} |")
    w("")
    w("沒有座標時仍可交付的東西（本批沒有輸出，需要時可由 `scripts/ext2_example.py` 的同一段程式產生）：每張 slide 兩套各四輪的 64 個 **patch index** 與 s0、g、s。"
      "index 要對回座標，需要當初抽特徵時的座標檔或重抽一次。")
    w("")

    # ── 未完成、失敗、DECISIONS ──
    w("## 未完成項與原因")
    w("")
    w("| 項目 | 原因 | 需要的東西 |")
    w("|---|---|---|")
    w("| D2 座標 CSV、D3 PNG | 特徵檔沒有逐 patch 座標（DECISIONS D15） | 抽 `feats-l1-s256_CONCH` 時的座標檔（通常是同名 `.h5` 的 `coords`），或對要畫的 slide 重抽特徵並留座標；D3 另需安裝 openslide 或 tifffile |")
    w("| 四輪版本的 K ≠ 64 | 沒有既有定義（每輪幾張未定；DECISIONS D5） | PI 指定每輪張數後可算，需重讀全部特徵檔 |")
    w("| FINAL-A 與主系統的 head 訓練秒數（本批量測） | 本批不訓練 head；表註引用 MOE-1 批的紀錄（DECISIONS D8） | 若要同批數字，需重訓至少一個 seed |")
    w("| 外部對照的每任務參數、儲存、訓練秒數 | 既有產物沒有（DECISIONS D16） | 查其論文或由 PI 指定算法 |")
    w("| 每任務訓練秒數只量 fold 1 | 見 DECISIONS D13 | — |")
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
        w("沒有失敗的階段。K9、K10、K11 都通過。D 階段第一次執行時指令列的分隔字串打錯（shell 語法，程式沒有被執行到），改正後執行一次完成，沒有產生 FAILED 檔。")
        w("")
    w("## DECISIONS.md（`ext2/DECISIONS.md` 全文）")
    w("")
    w((E2 / "DECISIONS.md").read_text().strip())
    w("")
    (BASE / "REPORT_ext2.md").write_text("\n".join(L) + "\n")
    print(f"FINALB_RESULTS.md：{n1} 行；REPORT_ext2.md：{len(L)} 行")


if __name__ == "__main__":
    main()
