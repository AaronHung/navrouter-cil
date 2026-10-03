#!/usr/bin/env python3
"""MOE-4 報告：由各階段的 JSON 產生 REPORT_moe4.md（只給數字與表格；PREREG-20 第二部分細則 29–31）。

  --out moe4        → outputs/navcil/<machine>/REPORT_moe4.md
  --out moe4_smoke  → outputs/navcil/<machine>/moe4_smoke/REPORT_smoke.md（冒煙測試；門檻不作判定）

    NAVCIL_MACHINE=mac python scripts/moe4_report.py --out moe4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORD = ("reverse", "paper")
SEEDS = (42, 43, 44, 45, 46)
NEW = (43, 44, 45, 46)
CLS = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]
TASK_NAMES = ["ESCA", "RCC", "BRCA", "LUNG"]
TASK_NAMES_ORDER = ["ESCA", "RCC", "BRCA", "LUNG"]   # 任務位置 0–3（config 序）


def f4(x) -> str:
    return "—" if x is None else f"{x:.4f}"


def ms(x) -> str:
    if x is None:
        return "—"
    return f"{x[0]:.4f} ± {x[1]:.4f}"


def sg(x) -> str:
    return "—" if x is None else f"{x:+.4f}"


def pv(p) -> str:
    return "不適用" if p is None else f"{p:.3g}"


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def load(d: Path, name: str):
    p = d / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="moe4")
    ap.add_argument("--folds", default="1-10")
    a = ap.parse_args()
    machine = os.environ.get("NAVCIL_MACHINE", "mac")
    base = REPO / "outputs" / "navcil" / machine
    d = base / a.out
    smoke = a.out != "moe4"
    dest = d / "REPORT_smoke.md" if smoke else base / "REPORT_moe4.md"
    chk, f2, f3, f4j, f5, vec = (load(d, n) for n in ("chk", "f2", "f3", "f4", "f5", "vec"))
    full = bool(f2 and f2.get("full"))

    L: list[str] = []
    L.append(f"# REPORT — MOE-4：定案確認（head 一次取 64 ＋ ridge 判讀 γ = 1e-3，多 seed、逐階段）")
    L.append("")
    L.append(f"機器：{machine}（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、closed-form 一律 float64；"
             "所有數字來自同一台、同一批（`scripts/moe4_run_all.sh`）；不訓練任何 head。判準與操作定義見 `PREREG-20.md`。")
    if smoke:
        L.append("")
        L.append("**冒煙測試（fold 1）**：數字不作為結果；門檻不作判定。")
    L.append("")
    stages = {"vec": vec, "chk": chk, "f2": f2, "f3": f3, "f4": f4j, "f5": f5}
    L.append("各階段狀態：" + "、".join(f"{k.upper()} {'完成' if v else '未完成'}" for k, v in stages.items()))
    L.append(f"已完成階段的 folds：{(f2 or chk or vec or {}).get('folds', '—')}。")
    L.append("")

    # ── 開頭：門檻與 K1–K5 ─────────────────────────────────────────────────
    L.append("## 門檻（PREREG-20「門檻」；細則 17–23）")
    L.append("")
    if f2:
        gp = f2["gates_pass"]
        decide = "" if full else "（非十折：不作判定）"
        L.append(f"- **G-CONF**：{'通過' if gp['G-CONF'] else '未通過'}{decide}　"
                 "（seed 43–46 各自、兩序各自：十折平均 ≥ +0.007 且差 > 1e-12 的折數 ≥ 7）")
        L.append(f"- **G-TRAJ**：{'通過' if gp['G-TRAJ'] else '未通過'}{decide}　（seed 43–46 各自、兩序各自：Ā 差的十折平均 ≥ −0.005）")
        L.append(f"- **G-ONE**：{'通過' if gp['G-ONE'] else '未通過'}{decide}　（每折取 seed 43–46 平均的 CIL ACC 差，十折平均 ≥ −0.003；兩序都滿足）")
        L.append("")
        rows = []
        for s in SEEDS:
            for o in ORD:
                tag = f"{s}|{o}"
                gc, gt = f2["gates"]["G-CONF"][tag], f2["gates"]["G-TRAJ"][tag]
                note = "seed 42 只作描述" if s == 42 else ""
                if s == 42:
                    verdict_c = verdict_t = "（描述）"
                else:
                    verdict_c = "通過" if gc["pass"] else "未通過"
                    verdict_t = "通過" if gt["pass"] else "未通過"
                rows.append([str(s), o, sg(gc["mean"]), f"{gc['wins']}/10", verdict_c, note,
                             sg(gt["mean"]), verdict_t])
        L.append("G-CONF（CIL ACC(P-F) − CIL ACC(P-main)，t = 4）與 G-TRAJ（Ā(P-F) − Ā(P-main)）：")
        L.append("")
        L.append(table(["seed", "序", "G-CONF 十折平均", "差 > 0 的折數", "G-CONF 判定", "附註", "G-TRAJ 十折平均", "G-TRAJ 判定"], rows))
        L.append("")
        rows = []
        for s in SEEDS:
            for o in ORD:
                tag = f"{s}|{o}"
                pc = f2["gates"]["P4-report"]["G-CONF"][tag]
                pt = f2["gates"]["P4-report"]["G-TRAJ"][tag]
                rows.append([str(s), o, sg(pc["mean"]), f"{pc['wins']}/10", sg(pt["mean"])])
        L.append("P-4 版本（只報告、不判定；P-4(s) − P-main(s)）：")
        L.append("")
        L.append(table(["seed", "序", "G-CONF 式 十折平均", "差 > 0 的折數", "G-TRAJ 式 十折平均"], rows))
        L.append("")
        rows = [[o, sg(f2["gates"]["G-ONE"][o]["mean"]), "通過" if f2["gates"]["G-ONE"][o]["pass"] else "未通過"] for o in ORD]
        L.append("G-ONE（P-F − P-4，t = 4，seed 43–46 平均）：")
        L.append("")
        L.append(table(["序", "十折平均", "判定"], rows))
        L.append("")
    else:
        L.append("（f2 未完成）")
        L.append("")

    L.append("## 一致性檢查 K1–K5（PREREG-20；細則 10–14）")
    L.append("")
    if chk:
        k1, k2, k3, k4, k5 = chk["K1"], chk["K2"], chk["K3"], chk["K4"], chk["K5"]
        rows = [
            ["K1", "P-main(42)：WP、CIL ACC 逐折（nc8/per_fold.json）＋逐階段 ACC（moe1/s2.json）", f"{k1['max_abs_fold_diff']:.2e}／{k1['stage_acc_max_abs']:.2e}",
             "≤ 1e-9", "通過" if k1["pass"] else "不符"],
            ["K2", "P-4(42)：t = 4 的 WP、CIL ACC（moe1/s5.json LR，γ = 1e-3）", f"{k2['max_abs_fold_diff']:.2e}", "≤ 1e-9", "通過" if k2["pass"] else "不符"],
            ["K3", "P-0 逐階段 ACC、WP、Forgetting、BWT（moe3/f2.json S-R0）", f"{k3['max_abs_fold_diff']:.2e}", "≤ 1e-9", "通過" if k3["pass"] else "不符"],
            ["K4", "P-main(s)，s = 43…46：t = 4 CIL ACC（moe2/e1.json，逐折）", f"{k4['max_abs_fold_diff']:.2e}", "≤ 1e-9", "通過" if k4["pass"] else "不符"],
            ["K5", "w(42)·Fᵀ 對 moe1 s42cells「一次取 64／等權」格（val＋test）", f"{k5['max_abs_cos']:.2e}；argmax {'全同' if k5['argmax_equal'] else '不同'}",
             "≤ 1e-6、argmax 全同", "通過" if k5["pass"] else "不符"],
        ]
        L.append(table(["項", "內容", "最大絕對差", "判準", "結果"], rows))
        if full:
            L.append("")
            L.append(f"十折平均（四捨五入）：K1 WP {k1['wp_mean']['reverse']:.4f}、CIL {k1['cil_mean']['reverse']:.4f}；"
                     f"K2 WP {k2['wp_mean']['reverse']:.4f}、CIL {k2['cil_mean']['reverse']:.4f}。")
    else:
        L.append("（chk 未完成；後續階段不執行）")
    L.append("")

    # ── H1 ─────────────────────────────────────────────────────────────────
    L.append("## H1 逐階段（PREREG-20 「要報的」H1；細則 24）")
    L.append("")
    if f2:
        for o in ORD:
            L.append(f"### 序：{o}")
            L.append("")
            rows = []
            for name in ("P-F", "P-4", "P-T1", "P-main"):
                for s in SEEDS:
                    h = f2["h1"][f"{name}({s})"][o]
                    rows.append(row_stage(f"{name}({s})", h))
                rows.append(row_stage(f"{name}（五個 seed 平均）", f2["h1_avg"][name][o]))
            h0 = f2["h1"]["P-0"][o]
            rows.append(row_stage("P-0（不用 head，一列）", h0))
            L.append(table(["系統", "ACC（t = 1 ／ 2 ／ 3 ／ 4）", "WP（告訴任務；t = 1 ／ 2 ／ 3 ／ 4）",
                            "每任務 WP（各階段一組 [ESCA RCC BRCA LUNG]，依 t = 1…4 排列）", "Forgetting", "BWT", "Ā"], rows))
            L.append("")
        L.append("（每任務 WP 與 ACC、WP 的全部逐階段值也在 `f2.json` 的 `h1`、`h1_avg`。）")
        L.append("")
    else:
        L.append("（f2 未完成）")
        L.append("")

    # ── H2 ─────────────────────────────────────────────────────────────────
    L.append("## H2 t = 4：每個 seed 的 WP、CIL ACC；逐折相減（PREREG-20 「要報的」H2；細則 25）")
    L.append("")
    if f2:
        rows = []
        for s in SEEDS:
            h = f2["h2"][str(s)]
            rows.append([str(s), ms(h["wp4"]["reverse"]), ms(h["cil4"]["reverse"]), ms(h["wp4"]["paper"]), ms(h["cil4"]["paper"]),
                         f"{h['same']['tell_diff']}／{h['same']['cil_diff']}（d 最大差 {h['same']['d_maxabs']:.1e}）"])
        L.append("P-F(s) 的 t = 4（reverse／paper）：")
        L.append("")
        L.append(table(["seed", "WP（reverse）", "CIL ACC（reverse）", "WP（paper）", "CIL ACC（paper）", "兩序告訴任務判定不同／CIL 不同的張數"], rows))
        L.append("")
        for s in SEEDS:
            h = f2["h2"][str(s)]
            rows = []
            for label, dd in h["diffs"].items():
                for k, kn in (("cil4", "CIL ACC"), ("wp4", "WP")):
                    for o in ORD:
                        p = dd[k][o]
                        rows.append([label, kn, o, sg(p["mean"]), f"{p['wins']}/10", pv(p["p"]),
                                     ", ".join(f"{x:+.4f}" for x in p["per_fold"])])
            L.append(f"逐折相減（seed {s}）：")
            L.append("")
            L.append(table(["比較", "指標", "序", "平均差", "贏折數", "Wilcoxon p", "逐折值（fold 1…10）"], rows))
            L.append("")
    else:
        L.append("（f2 未完成）")
        L.append("")

    # ── H3 ─────────────────────────────────────────────────────────────────
    L.append("## H3 每類別（PREREG-20 「要報的」H3；細則 26；reverse、告訴任務、t = 4、test；不設門檻）")
    L.append("")
    if f3:
        T = f3["table"]
        L.append("五個 seed 合計（seed 42–46 × 十折的正確張數與正確率）：")
        L.append("")
        rows = []
        for i, c in enumerate(CLS):
            rows.append([c] + [cell(T[sn]["pooled5"], i) for sn in ("P-F", "P-4", "P-main")] + [cell(T["P-0"]["pooled5"], i)])
        rows.append(["ACC（全體）"] + [f"{T[sn]['pooled5']['acc_all']:.4f}" for sn in ("P-F", "P-4", "P-main")] + [f"{T['P-0']['pooled5']['acc_all']:.4f}"])
        rows.append(["四任務 balanced acc 平均"] + [f"{T[sn]['pooled5']['bal_acc_mean']:.4f}" for sn in ("P-F", "P-4", "P-main")] + [f"{T['P-0']['pooled5']['bal_acc_mean']:.4f}"])
        for q in range(4):
            rows.append([f"{TASK_NAMES[q]} balanced acc"] + [f"{T[sn]['pooled5']['bal_acc'][q]:.4f}" for sn in ("P-F", "P-4", "P-main")] + [f"{T['P-0']['pooled5']['bal_acc'][q]:.4f}"])
        L.append(table(["類別", "P-F", "P-4", "P-main", "P-0（一份，十折）"], rows))
        L.append("")
        L.append("seed 42 一份：")
        L.append("")
        rows = []
        for i, c in enumerate(CLS):
            rows.append([c] + [cell(T[sn]["seed42"], i) for sn in ("P-F", "P-4", "P-main")] + [cell(T["P-0"]["pooled5"], i)])
        rows.append(["ACC（全體）"] + [f"{T[sn]['seed42']['acc_all']:.4f}" for sn in ("P-F", "P-4", "P-main")] + [f"{T['P-0']['pooled5']['acc_all']:.4f}"])
        rows.append(["四任務 balanced acc 平均"] + [f"{T[sn]['seed42']['bal_acc_mean']:.4f}" for sn in ("P-F", "P-4", "P-main")] + [f"{T['P-0']['pooled5']['bal_acc_mean']:.4f}"])
        for q in range(4):
            rows.append([f"{TASK_NAMES[q]} balanced acc"] + [f"{T[sn]['seed42']['bal_acc'][q]:.4f}" for sn in ("P-F", "P-4", "P-main")] + [f"{T['P-0']['pooled5']['bal_acc'][q]:.4f}"])
        L.append(table(["類別", "P-F", "P-4", "P-main", "P-0（一份，十折）"], rows))
        L.append("")
        L.append("（P-0 與 seed 無關，因此五個 seed 合計與 seed 42 欄都以十折合計一份表示；格式：正確張數／張數（正確率）。）")
        L.append("")
    else:
        L.append("（f3 未完成）")
        L.append("")

    # ── H4 ─────────────────────────────────────────────────────────────────
    L.append("## H4 γ 敏感度（PREREG-20 「要報的」H4；細則 27；只作描述）")
    L.append("")
    if f4j:
        rows = []
        for o in ORD:
            for g in f4j["gammas"]:
                t = f4j["table"][f"{o}|{g:g}"]
                mark = "（定案）" if g == f4j["gamma_star"] else ""
                rows.append([o, f"{g:g}{mark}", ms(t["cil4"]), ms(t["abar"])])
        L.append(table(["序", "γ", "CIL ACC（t = 4，五個 seed 平均）", "Ā（五個 seed 平均）"], rows))
        L.append("")
    else:
        L.append("（f4 未完成）")
        L.append("")

    # ── H5 ─────────────────────────────────────────────────────────────────
    L.append("## H5 儲存（PREREG-20 「要報的」H5；細則 28；fp32；不設門檻）")
    L.append("")
    if f5:
        rows = [[k, str(v["shape"]), f"{v['n_float']:,}", f"{v['fp32_bytes']:,}", v["what"]] for k, v in f5["items"].items()]
        L.append(table(["項目", "形狀", "float 數", "bytes", "內容"], rows))
        L.append("")
        rows = []
        for k, v in f5["systems"].items():
            rows.append([k, ", ".join(v["shared"]), f"{v['shared_bytes']:,}", ", ".join(v["per_task"]), f"{v['per_task_bytes']:,}",
                         f"{v['T4_bytes']:,}", v.get("note", "")])
        L.append(table(["系統", "共用", "共用 bytes", "每任務", "每任務 bytes", "T = 4 總 bytes", "註"], rows))
        L.append("")
        L.append(f"{f5['not_counted']}。{f5['note']}")
        L.append("")
    else:
        L.append("（f5 未完成）")
        L.append("")

    # ── 失敗階段、DECISIONS、耗時 ─────────────────────────────────────────
    L.append("## 失敗的階段")
    L.append("")
    fails = sorted(d.glob("FAILED_*.txt"))
    if fails:
        for p in fails:
            L.append(f"- `{p.name}`：" + p.read_text().strip().splitlines()[0][:200])
    else:
        L.append("無。")
    L.append("")
    L.append("## DECISIONS.md")
    L.append("")
    dec = d / "DECISIONS.md"
    L.append(dec.read_text() if dec.exists() else "（無 DECISIONS.md）")
    L.append("")
    L.append("## 各階段耗時")
    L.append("")
    st = d / "STAGE_TIMES.tsv"
    if st.exists():
        rows = [line.split("\t") for line in st.read_text().strip().splitlines()]
        L.append(table(["階段", "開始", "結束", "秒", "結束碼"], rows))
    else:
        L.append("（無 STAGE_TIMES.tsv）")
    L.append("")
    dest.write_text("\n".join(L))
    print(f"wrote {dest}")
    return 0


def row_stage(label: str, h: dict) -> list[str]:
    """H1 的一列：ACC_t、WP_t 依 t = 1…4 以 ／ 分隔；每任務 WP 依階段列出（ESCA／RCC／BRCA／LUNG，未學為 —）。"""
    per = []
    for tt in range(4):
        per.append("[" + " ".join(("—" if x is None else f"{x[0]:.3f}") for x in h["wp_task_t"][tt]) + "]")
    return [label, " ／ ".join(ms(x) for x in h["acc_t"]), " ／ ".join(ms(x) for x in h["wp_t"]), " ／ ".join(per),
            ms(h["forgetting"]), ms(h["bwt"]), ms(h["abar"])]


def cell(t: dict, i: int) -> str:
    return f"{t['correct'][i]}／{t['n'][i]}（{t['acc'][i]:.4f}）"


if __name__ == "__main__":
    sys.exit(main())
