#!/usr/bin/env python3
"""MOE-0 報告：B1–B5（PREREG-16）。只推論；讀 moe0 快取與 NC-8 同批快取。

    NAVCIL_MACHINE=mac python scripts/moe0_report.py
啟動前檢查（PREREG-16 操作定義 24）：selection.json 已 commit 且未修改。
一致性檢查（操作定義 5a–5d）不符即停（exit 3）。
輸出：outputs/navcil/<machine>/REPORT_moe0.md、moe0/b3_per_slide.csv、moe0/results.json
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys

import torch
from scipy import stats

import moe0_common as C
from selector.text_encoder import build_f_txt

REVERSE, PAPER = "reverse", "paper"


def fail(msg: str):
    print("停止：" + msg)
    sys.exit(3)


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=C.REPO_ROOT, capture_output=True, text=True)


def f4(x) -> str:
    return f"{x:.4f}"


def ms(xs) -> str:
    return f"{C.mean(xs):.4f} ± {C.sd(xs):.4f}"


def paired(diffs: list[float]) -> dict:
    nz = any(d != 0 for d in diffs)
    return {"mean": C.mean(diffs), "wins": sum(d > 1e-12 for d in diffs), "losses": sum(d < -1e-12 for d in diffs),
            "p": float(stats.wilcoxon(diffs).pvalue) if nz else None, "per_fold": diffs}


def fp(p) -> str:
    return "不適用（全為 0）" if p is None else f"{p:.4f}"


def main() -> int:
    st = C.Store()
    sel_path = st.root / "selection.json"
    rel = str(sel_path.relative_to(C.REPO_ROOT))
    if git("ls-files", "--error-unmatch", rel).returncode != 0 or git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        fail(f"{rel} 尚未 commit 或已被修改（PREREG-16 操作定義 24）")
    sel = json.loads(sel_path.read_text())
    ls = sel["logit_scale"]
    commit_sel = git("log", "-1", "--format=%h", "--", rel).stdout.strip()
    nc8_pf = json.loads((st.b.out / "nc8" / "per_fold.json").read_text())["rows"]["6 D3：AR＋I6(r=2)（主系統）"]
    tasks = st.tasks

    # ── 一致性檢查 5a–5d + 逐折資料 ──
    chk = {"5a_max_cos": 0.0, "5b_max_mv": 0.0, "5c_max_fold_diff": 0.0, "5d_diag_max_cos": 0.0}
    R = {o: {} for o in C.ORDER_NAMES}
    for f in C.FOLDS:
        S = st.split("test", f)
        for p, t in enumerate(tasks):
            c, m = st.b.c(f, "test", t), S["per_task"][t]
            if c["sids"] != m["sids"] or not torch.equal(c["labels"], m["labels"]):
                fail(f"fold {f} {t}: sids／labels 與 NC-8 快取不一致")
            chk["5a_max_cos"] = max(chk["5a_max_cos"], (c["I6_cos8"] - m["I6_cos8"]).abs().max().item())
            chk["5b_max_mv"] = max(chk["5b_max_mv"], (c["mean_vec"] - m["mean_vec"]).abs().max().item())
            for q in range(4):
                rows = torch.tensor([2 * q, 2 * q + 1])
                if not torch.equal(c["I6_cos8"][:, q][:, rows].argmax(-1), m["I6_cos8"][:, q][:, rows].argmax(-1)):
                    fail(f"fold {f} {t} expert {q}: 2 類內 argmax 與 NC-8 快取不同")
            rows = torch.tensor([2 * p, 2 * p + 1])
            chk["5d_diag_max_cos"] = max(chk["5d_diag_max_cos"], (m["cross_cos8"][:, p] - m["I6_cos8"][:, p]).abs().max().item())
            if not torch.equal(m["cross_cos8"][:, p][:, rows].argmax(-1), m["I6_cos8"][:, p][:, rows].argmax(-1)):
                fail(f"fold {f} {t}: B2 對角線與 I6_cos8 的 2 類內 argmax 不同")
        label, task, I6 = S["labels"], S["task"], S["I6_cos8"]
        for o in C.ORDER_NAMES:
            ar = st.ar(f, o, S["mean_vec"])
            lin = st.lin8(f, o, S["mean_vec"])
            th, pred, wp_pred, mk_pred = C.main_preds(I6, ar, task)
            da_all, db_all = C.all_task_diffs(I6, lin)
            r = {"th": th, "pred": pred, "wp": wp_pred, "mk": mk_pred, "ar": ar, "lin": lin, "da_all": da_all, "db_all": db_all,
                 "tp_t": C.task_mean(th == task, task), "wp_t": C.task_mean(wp_pred == label, task),
                 "cil_t": C.task_mean(pred == label, task), "mk_t": C.task_mean(mk_pred == label, task)}
            r.update(tp=C.eq4(r["tp_t"]), wp4=C.eq4(r["wp_t"]), cil=C.eq4(r["cil_t"]), mk=C.eq4(r["mk_t"]))
            R[o][f] = r
            j = nc8_pf[o][f - 1]
            d = max(abs(r["cil"] - j["acc"]), abs(r["mk"] - j["masked"]),
                    *[abs(a - b) for a, b in zip(r["tp_t"], j["tp_task"])], *[abs(a - b) for a, b in zip(r["wp_t"], j["wp_task"])])
            chk["5c_max_fold_diff"] = max(chk["5c_max_fold_diff"], d)
    chk["main_cil_mean"] = {o: C.mean(R[o][f]["cil"] for f in C.FOLDS) for o in C.ORDER_NAMES}
    if chk["5a_max_cos"] > 1e-6:
        fail(f"5a 不符：{chk['5a_max_cos']:.2e}")
    if chk["5b_max_mv"] > 1e-6:
        fail(f"5b 不符：{chk['5b_max_mv']:.2e}")
    if chk["5c_max_fold_diff"] > 1e-9 or any(round(v, 4) != 0.9128 for v in chk["main_cil_mean"].values()):
        fail(f"5c 不符：{chk['5c_max_fold_diff']:.2e}、{chk['main_cil_mean']}")
    if chk["5d_diag_max_cos"] > 1e-6:
        fail(f"5d 不符：{chk['5d_diag_max_cos']:.2e}")

    out_md, res = [], {"checks": chk}
    folds = C.FOLDS
    # 兩序在 t = 4 是否逐張相同
    same_order = all(torch.equal(R[REVERSE][f]["pred"], R[PAPER][f]["pred"]) and torch.equal(R[REVERSE][f]["th"], R[PAPER][f]["th"])
                     for f in folds)
    res["orders_identical_pred"] = same_order

    # ── B1 ──
    n_task = [sum(len(st.split("test", f)["per_task"][t]["labels"]) for f in folds) for t in tasks]
    wp_err = [[0] * 2 for _ in range(4)]       # [任務][類]：錯誤張數
    n_cls = [[0] * 2 for _ in range(4)]
    for f in folds:
        S = st.split("test", f)
        r = R[REVERSE][f]
        for p in range(4):
            for c in range(2):
                m = (S["task"] == p) & (S["labels"] == 2 * p + c)
                n_cls[p][c] += int(m.sum()); wp_err[p][c] += int((r["wp"][m] != S["labels"][m]).sum())
    b1 = {}
    out_md += ["## T1（B1）每任務明細（test、t = 4、十折）", "",
               "TP／WP／CIL 為十折平均（每折內的任務正確率，與 9/30 Table 1 相同）；n 與錯誤張數為十折合計。"
               f"兩序在 t = 4 逐張相同：**{'是' if same_order else '否'}**（AR 分派與主系統判定）。", ""]
    for o in C.ORDER_NAMES:
        out_md += [f"### {o}", "", "| 任務 | test slides（十折合計） | TP 正確率 | WP 正確率（告訴任務） | CIL 正確率 | WP 錯誤張數 |",
                   "|---|---|---|---|---|---|"]
        for p, t in enumerate(tasks):
            e = sum(wp_err[p])
            out_md.append(f"| {C.TASK_SHORT[p]} | {n_task[p]} | {ms([R[o][f]['tp_t'][p] for f in folds])} | "
                          f"{ms([R[o][f]['wp_t'][p] for f in folds])} | {ms([R[o][f]['cil_t'][p] for f in folds])} | {e} |")
        out_md.append(f"| 四任務等權 | {sum(n_task)} | {ms([R[o][f]['tp'] for f in folds])} | {ms([R[o][f]['wp4'] for f in folds])} | "
                      f"{ms([R[o][f]['cil'] for f in folds])} | {sum(map(sum, wp_err))} |")
        out_md.append("")
    out_md += ["WP 錯誤張數按真實類別（十折合計；WP 與序無關）：", "", "| 任務 | 類別（第一類） | n | WP 錯 | 錯誤率 | 類別（第二類） | n | WP 錯 | 錯誤率 |",
               "|---|---|---|---|---|---|---|---|---|"]
    for p in range(4):
        out_md.append(f"| {C.TASK_SHORT[p]} | {C.LABEL[2 * p]} | {n_cls[p][0]} | {wp_err[p][0]} | {wp_err[p][0] / n_cls[p][0]:.4f} | "
                      f"{C.LABEL[2 * p + 1]} | {n_cls[p][1]} | {wp_err[p][1]} | {wp_err[p][1] / n_cls[p][1]:.4f} |")
    out_md.append("")
    res["B1"] = {"n_task": n_task, "wp_err": wp_err, "n_cls": n_cls}

    # ── B2 ──
    M = [[[] for _ in range(4)] for _ in range(5)]           # [head k 或 g0][真實任務 j] → 逐折正確率
    for f in folds:
        S = st.split("test", f)
        for j in range(4):
            m = S["task"] == j
            rows = torch.tensor([2 * j, 2 * j + 1])
            for k in range(4):
                z = S["cross_cos8"][m][:, k][:, rows]
                M[k][j].append((rows[z.argmax(-1)] == S["labels"][m]).float().mean().item())
            z = S["g0_cos8"][m][:, rows]
            M[4][j].append((rows[z.argmax(-1)] == S["labels"][m]).float().mean().item())
    Mm = [[C.mean(M[k][j]) for j in range(4)] for k in range(5)]
    names = [f"{C.TASK_SHORT[k]} head" for k in range(4)] + ["g = 0（只用 s0）"]
    wp_ref = [C.mean(R[REVERSE][f]["wp_t"][j] for f in folds) for j in range(4)]
    diag_dev = max(abs(Mm[j][j] - wp_ref[j]) for j in range(4))
    cn = {t: build_f_txt(t).class_names for t in tasks}
    out_md += ["## T2（B2）交叉 head 矩陣（test、告訴真實任務、2 類正確率、十折平均）", "",
               "列 = 使用的 head 權重（另加 g = 0 一列）、欄 = 真實任務 j；s0 與 head 的輸入一律用任務 j 的兩個亞型文字。"
               "（PREREG-16 操作定義 8 寫的是「列 = 真實任務、欄 = head」；為了容納 g = 0 這一列，本表轉置，內容相同；"
               "G-M2 的兩格 = 〔LUNG head、ESCA 欄〕與〔ESCA head、LUNG 欄〕。）", "",
               "| head＼真實任務 | " + " | ".join(C.TASK_SHORT) + " | 四欄等權 |", "|---|---|---|---|---|---|"]
    for k in range(5):
        out_md.append(f"| {names[k]} | " + " | ".join(f4(Mm[k][j]) for j in range(4)) + f" | {f4(C.eq4(Mm[k]))} |")
    out_md += ["", f"對角線與 B1 的 WP（十折平均）逐格最大絕對差 = {diag_dev:.2e}；B2 對角線與 I6_cos8 的 2 類內 argmax 全同、cosine 最大絕對差 {chk['5d_diag_max_cos']:.2e}（檢查 5d）。", "",
               "每格逐折 sd：", "", "| head＼真實任務 | " + " | ".join(C.TASK_SHORT) + " |", "|---|---|---|---|---|"]
    for k in range(5):
        out_md.append(f"| {names[k]} | " + " | ".join(f"{C.sd(M[k][j]):.4f}" for j in range(4)) + " |")
    order_rows = [(t, cn[t]) for t in tasks]
    esca_first, lung_first = cn[tasks[0]][0], cn[tasks[3]][0]
    aligned_already = (esca_first, lung_first) == ("ESAD", "LUAD")
    out_md += ["", "每個任務兩類的順序（第一類 = label 0 = 8 類中第 2p 列）：", "", "| 任務 | 第一類 | 第二類 | 資料表 label 欄 |", "|---|---|---|---|"]
    for t, c in order_rows:
        out_md.append(f"| {t} | {c[0]} | {c[1]} | {c[0]} = 0、{c[1]} = 1 |")
    out_md += ["", f"ESCA（{cn[tasks[0]][0]} 第一）與 LUNG（{cn[tasks[3]][0]} 第一）的腺癌／鱗癌順序**{'相同' if aligned_already else '不同'}**；"
               + ("順序已相同，不需對齊。" if aligned_already else "（未對齊；見下）"), ""]
    cell1, cell2 = Mm[3][0], Mm[0][3]          # (LUNG head, ESCA 欄)、(ESCA head, LUNG 欄)
    gm2 = {"cell_lung_head_on_esca": cell1, "diag_esca": Mm[0][0], "gap1": Mm[0][0] - cell1,
           "cell_esca_head_on_lung": cell2, "diag_lung": Mm[3][3], "gap2": Mm[3][3] - cell2}
    gm2["pass"] = bool(gm2["gap1"] < 0.01 and gm2["gap2"] < 0.01)
    res["B2"] = {"matrix": Mm, "gm2": gm2, "aligned_already": aligned_already}
    if not aligned_already:
        fail("ESCA 與 LUNG 的類別順序不同，需要對齊後重算（PREREG-16 操作定義 9），本腳本未實作")

    # ── B3（reverse 序；t = 4 兩序相同）──
    csv_path = st.root / "b3_per_slide.csv"
    cnt = [[0] * 4 for _ in range(4)]                       # [任務] → [兩對, 只a, 只b, 兩錯]
    fold_union = [[] for _ in range(4)]
    fold_a = [[] for _ in range(4)]
    fold_b = [[] for _ in range(4)]
    b_same_order = True
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fold", "slide_id", "task", "true_class", "d_a", "d_b"])
        for f in folds:
            S, r = st.split("test", f), R[REVERSE][f]
            task, label = S["task"], S["labels"]
            a_ok = r["wp"] == label
            b_pred = r["lin"].gather(1, torch.stack([2 * task, 2 * task + 1], 1)).argmax(-1) + 2 * task
            b_ok = b_pred == label
            b_same_order &= bool(torch.equal(b_pred, R[PAPER][f]["lin"].gather(1, torch.stack([2 * task, 2 * task + 1], 1)).argmax(-1) + 2 * task))
            da = C.pair_diff(S["I6_cos8"][torch.arange(len(task)), task], task)
            db = C.pair_diff(r["lin"], task)
            for p in range(4):
                m = task == p
                cnt[p][0] += int((a_ok & b_ok)[m].sum()); cnt[p][1] += int((a_ok & ~b_ok)[m].sum())
                cnt[p][2] += int((~a_ok & b_ok)[m].sum()); cnt[p][3] += int((~a_ok & ~b_ok)[m].sum())
                fold_union[p].append((a_ok | b_ok)[m].float().mean().item())
                fold_a[p].append(a_ok[m].float().mean().item()); fold_b[p].append(b_ok[m].float().mean().item())
            for i in range(len(task)):
                w.writerow([f, S["sids"][i], tasks[int(task[i])], C.LABEL[int(label[i])], f"{float(da[i]):.8f}", f"{float(db[i]):.8f}"])
    U = C.eq4([C.mean(x) for x in fold_union])
    WP4 = C.eq4([C.mean(R[REVERSE][f]["wp_t"][p] for f in folds) for p in range(4)])
    out_md += ["## T3（B3）整片統計（LIN8）與局部證據（主系統 WP）的錯誤重疊（test、告訴真實任務、十折合計）", "",
               "(a) = 主系統 WP 的判定（I6 expert 四輪證據，2 類內 argmax）；(b) = LIN8（γ = 0.01）logits 只在真實任務兩類內 argmax。"
               f"兩序在 t = 4 的 (b) 逐張相同：**{'是' if b_same_order else '否'}**（per-slide 檔只存 reverse 序一份）。", "",
               "| 任務 | n | 兩者都對 | 只有 (a) 對 | 只有 (b) 對 | 兩者都錯 | 至少一個對（合計） | (a) 正確率 | (b) 正確率 |",
               "|---|---|---|---|---|---|---|---|---|"]
    for p in range(4):
        n = sum(cnt[p])
        out_md.append(f"| {C.TASK_SHORT[p]} | {n} | {cnt[p][0]} | {cnt[p][1]} | {cnt[p][2]} | {cnt[p][3]} | "
                      f"{(cnt[p][0] + cnt[p][1] + cnt[p][2]) / n:.4f} | {(cnt[p][0] + cnt[p][1]) / n:.4f} | {(cnt[p][0] + cnt[p][2]) / n:.4f} |")
    tot = [sum(cnt[p][i] for p in range(4)) for i in range(4)]
    out_md += [f"| 合計 | {sum(tot)} | {tot[0]} | {tot[1]} | {tot[2]} | {tot[3]} | {(tot[0] + tot[1] + tot[2]) / sum(tot):.4f} | "
               f"{(tot[0] + tot[1]) / sum(tot):.4f} | {(tot[0] + tot[2]) / sum(tot):.4f} |", "",
               "「至少一個對」比例（每折每任務計、十折平均；用於 G-M3）：", "", "| 任務 | 至少一個對 | (a) 即 WP | 差 |", "|---|---|---|---|"]
    for p in range(4):
        out_md.append(f"| {C.TASK_SHORT[p]} | {ms(fold_union[p])} | {ms(fold_a[p])} | {C.mean(fold_union[p]) - C.mean(fold_a[p]):+.4f} |")
    out_md += [f"| 四任務等權 U | {f4(U)} | WP₄ = {f4(WP4)} | {U - WP4:+.4f} |", "",
               f"per-slide 檔：`outputs/navcil/mac/moe0/b3_per_slide.csv`（{sum(tot)} 列；欄位 fold、slide_id、task、true_class、d_a、d_b；差 = 第一類 − 第二類）。", ""]
    res["B3"] = {"counts": cnt, "U": U, "WP4": WP4, "b_same_order": b_same_order,
                 "b_acc4": C.eq4([C.mean(x) for x in fold_b])}

    # ── B4 ──
    def b4_variant(variant: str, center: bool):
        lam_star = sel["B4"][variant]["lambda_star"]
        tbl = {lam: {"masked": [], "cil": {o: [] for o in C.ORDER_NAMES}} for lam in C.LAMBDAS}
        for f in folds:
            V = st.split("val", f)
            vlin = st.lin8(f, REVERSE, V["mean_vec"])
            vda, vdb = C.all_task_diffs(V["I6_cos8"], vlin)
            vt = V["task"]
            pa = C.zparams(vda.gather(1, vt[:, None]).squeeze(1), vt)
            pb = C.zparams(vdb.gather(1, vt[:, None]).squeeze(1), vt)
            S = st.split("test", f)
            task, label = S["task"], S["labels"]
            for lam in C.LAMBDAS:
                for o in C.ORDER_NAMES:
                    r = R[o][f]
                    if o == REVERSE:
                        da = r["da_all"].gather(1, task[:, None]).squeeze(1)
                        db = r["db_all"].gather(1, task[:, None]).squeeze(1)
                        ok = C.fused_masked_ok(da, db, task, label, pa, pb, lam, center)
                        tbl[lam]["masked"].append(C.eq4(C.task_mean(ok, task)))
                    ok = C.fused_cil_ok(r["da_all"], r["db_all"], r["th"], label, pa, pb, lam, center)
                    tbl[lam]["cil"][o].append(C.eq4(C.task_mean(ok, task)))
        return lam_star, tbl

    wp_f = [R[REVERSE][f]["wp4"] for f in folds]
    mk_f = [R[REVERSE][f]["mk"] for f in folds]
    cil_f = {o: [R[o][f]["cil"] for f in folds] for o in C.ORDER_NAMES}
    res["B4"] = {}
    out_md += ["## T4（B4）簡單融合 f = z_a + λ·z_b（z 以該折該任務 validation slides 的平均與樣本標準差；TP = AR）", "",
               f"選定（只讀 validation；`selection.json` 見 commit {commit_sel}）：λ\\* = **{sel['B4']['z']['lambda_star']}**。"
               "主系統為 λ 網格之外的未置中版本（決策門檻 d_a = 0）；λ = 0 是「只用 (a)、減 validation 平均並除以標準差」的版本。", ""]
    for variant, center, title in (("z", True, "主要：z = (d − μ)/σ"), ("zprime", False, "附帶（不用於門檻）：z′ = d/σ，不減平均")):
        lam_star, tbl = b4_variant(variant, center)
        vm = sel["B4"][variant]["val_masked_by_lambda"]
        out_md += [f"### {title}（λ\\* = {lam_star}）", "",
                   "| λ | validation Masked ACC | test Masked ACC | test CIL ACC reverse | test CIL ACC paper |", "|---|---|---|---|---|"]
        out_md.append(f"| 主系統（未置中、無融合） | {f4(sel['checks']['5f']['val_wp_main'])} | {ms(wp_f)}（Table 1 Masked 欄 {ms(mk_f)}） | "
                      f"{ms(cil_f[REVERSE])} | {ms(cil_f[PAPER])} |")
        for lam in C.LAMBDAS:
            star = " ★" if lam == lam_star else ""
            out_md.append(f"| {lam}{star} | {f4(vm[str(lam)])} | {ms(tbl[lam]['masked'])} | {ms(tbl[lam]['cil'][REVERSE])} | {ms(tbl[lam]['cil'][PAPER])} |")
        t = tbl[lam_star]
        pm_wp, pm_mk = paired([a - b for a, b in zip(t["masked"], wp_f)]), paired([a - b for a, b in zip(t["masked"], mk_f)])
        pc = {o: paired([a - b for a, b in zip(t["cil"][o], cil_f[o])]) for o in C.ORDER_NAMES}
        out_md += ["", f"λ\\* = {lam_star} 與主系統逐折相減（融合 − 主系統；test；十折）：", "",
                   "| 比較 | 平均差 | 贏折數（差 > 0） | 輸折數 | Wilcoxon 雙尾 p | 逐折差（折 1–10） |", "|---|---|---|---|---|---|"]
        for name, pr in (("Masked ACC − 主系統告訴任務 WP（主要基準）", pm_wp), ("Masked ACC − Table 1 的 Masked ACC 欄（次要基準）", pm_mk),
                         ("CIL ACC − 主系統 CIL ACC（reverse）", pc[REVERSE]), ("CIL ACC − 主系統 CIL ACC（paper）", pc[PAPER])):
            out_md.append(f"| {name} | {pr['mean']:+.4f} | {pr['wins']}/10 | {pr['losses']}/10 | {fp(pr['p'])} | " +
                          " ".join(f"{x:+.4f}" for x in pr["per_fold"]) + " |")
        out_md.append("")
        res["B4"][variant] = {"lambda_star": lam_star, "test_masked": {str(l): C.mean(v["masked"]) for l, v in tbl.items()},
                              "test_cil": {str(l): {o: C.mean(v["cil"][o]) for o in C.ORDER_NAMES} for l, v in tbl.items()},
                              "paired": {"masked_vs_wp": pm_wp, "masked_vs_tablemasked": pm_mk, "cil": pc},
                              "val_masked": vm}

    # ── B5 ──
    out_md += ["## T5（B5）soft gating 基準：score(c) = π[task(c)] × q[c]", "",
               f"π = softmax(AR 分數／T)；q = softmax(logit_scale × 兩類 cosine)，logit_scale = {ls:.4f}；四個任務的 expert 都跑。"
               f"T 以 validation 十折平均 CIL ACC 選（`selection.json`，commit {commit_sel}）。", ""]
    res["B5"] = {}
    for o in C.ORDER_NAMES:
        T_star = sel["B5"][o]["T_star"]
        cil_T = {T: [] for T in C.TEMPS}
        chg = {"changed": 0, "wrong_to_right": 0, "right_to_wrong": 0, "wrong_to_other_wrong": 0}
        chg_by_task = [[0, 0, 0, 0] for _ in range(4)]
        for f in folds:
            S, r = st.split("test", f), R[o][f]
            for T in C.TEMPS:
                pred = C.soft_gate_pred(S["I6_cos8"], r["ar"], T, ls)
                cil_T[T].append(C.eq4(C.task_mean(pred == S["labels"], S["task"])))
                if T == T_star:
                    old_ok, new_ok = r["pred"] == S["labels"], pred == S["labels"]
                    ch = pred != r["pred"]
                    chg["changed"] += int(ch.sum()); chg["wrong_to_right"] += int((ch & ~old_ok & new_ok).sum())
                    chg["right_to_wrong"] += int((ch & old_ok & ~new_ok).sum()); chg["wrong_to_other_wrong"] += int((ch & ~old_ok & ~new_ok).sum())
                    for p in range(4):
                        m = S["task"] == p
                        chg_by_task[p][0] += int((ch & m).sum()); chg_by_task[p][1] += int((ch & m & ~old_ok & new_ok).sum())
                        chg_by_task[p][2] += int((ch & m & old_ok & ~new_ok).sum()); chg_by_task[p][3] += int((ch & m & ~old_ok & ~new_ok).sum())
        vc = sel["B5"][o]["val_cil_by_T"]
        out_md += [f"### {o}（T\\* = {T_star}）", "", "| T | validation CIL ACC | test CIL ACC |", "|---|---|---|",
                   f"| 主系統（hard routing） | — | {ms(cil_f[o])} |"]
        for T in C.TEMPS:
            out_md.append(f"| {T}{' ★' if T == T_star else ''} | {f4(vc[str(T)])} | {ms(cil_T[T])} |")
        pr = paired([a - b for a, b in zip(cil_T[T_star], cil_f[o])])
        out_md += ["", f"T\\* = {T_star} 與主系統逐折相減（soft − 主系統）：平均差 {pr['mean']:+.4f}、贏 {pr['wins']}/10、輸 {pr['losses']}/10、Wilcoxon 雙尾 p = {fp(pr['p'])}；"
                   "逐折差：" + " ".join(f"{x:+.4f}" for x in pr["per_fold"]), "",
                   f"判定改變的張數（十折合計、test {sum(n_task)} 張）：改變 {chg['changed']}；原錯→改對 **{chg['wrong_to_right']}**；原對→改錯 **{chg['right_to_wrong']}**；"
                   f"原錯→另一個錯 {chg['wrong_to_other_wrong']}。", "", "| 真實任務 | 改變 | 原錯→改對 | 原對→改錯 | 原錯→另一個錯 |", "|---|---|---|---|---|"]
        for p in range(4):
            out_md.append(f"| {C.TASK_SHORT[p]} | " + " | ".join(str(x) for x in chg_by_task[p]) + " |")
        out_md.append("")
        res["B5"][o] = {"T_star": T_star, "test_cil": {str(T): C.mean(v) for T, v in cil_T.items()}, "paired": pr, "changed": chg}

    # ── 門檻 ──
    lam_star = res["B4"]["z"]["lambda_star"]
    val_main = sel["checks"]["5f"]["val_wp_main"]
    val_fuse = sel["B4"]["z"]["val_masked_by_lambda"][str(lam_star)]
    g3_i, g3_ii = U - WP4, val_fuse - val_main
    gm3 = {"U": U, "WP4": WP4, "diff_i": g3_i, "pass_i": bool(g3_i >= 0.02), "lambda_star": lam_star,
           "val_fused": val_fuse, "val_main": val_main, "diff_ii": g3_ii, "pass_ii": bool(g3_ii >= 0.005)}
    gm3["pass"] = gm3["pass_i"] and gm3["pass_ii"]
    res["gates"] = {"G-M3": gm3, "G-M2": gm2}

    head = ["# REPORT — MOE-0：MoE 前置診斷 B1–B5（只推論；Mac CPU，十折，reverse 與 forward〔= repo 的 paper〕兩序）", "",
            "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0。所有數字來自同一台、同一批：NC-8 同批快取（train mean_vec、test I6_cos8）"
            f"＋本輪新推論（`scripts/moe0_infer.py`；validation 十折與 test 十折，見 T6）。判準與操作定義見 `PREREG-16.md`（commit 146344f）；"
            f"選 λ、T 只讀 validation（`selection.json`，commit {commit_sel}）。A 段稽核見 `AUDIT_wp.md`。",
            "", "主系統 = D3 = AR（γ = 1e-3）＋ I6(r = 2)；test ACC 0.9128（REPORT_stage10.md:359），本批重算 "
            f"reverse {chk['main_cil_mean']['reverse']:.4f}／paper {chk['main_cil_mean']['paper']:.4f}。", "",
            "## PREREG-16 落點與門檻", "",
            "| 項目 | PREREG-16 | 表 |", "|---|---|---|",
            "| B1 每任務明細 | 操作定義 6 | T1 |", "| B2 交叉 head 矩陣 | 操作定義 7–9 | T2 |", "| B3 錯誤重疊 | 操作定義 10–12 | T3 |",
            "| B4 簡單融合 | 操作定義 13–17 | T4 |", "| B5 soft gating | 操作定義 18–21 | T5 |",
            "| 一致性檢查 | 操作定義 5a–5f | T6 |", "| 重算範圍與耗時 | 操作定義 4 | T6 |", "",
            "### G-M3（操作定義 22）", "",
            "| 條件 | 數值 | 門檻 | 結果 |", "|---|---|---|---|",
            f"| (i) U（B3 至少一個對，四任務等權）− WP₄ | {gm3['U']:.4f} − {gm3['WP4']:.4f} = {gm3['diff_i']:+.4f} | ≥ +0.02 | {'滿足' if gm3['pass_i'] else '不滿足'} |",
            f"| (ii) λ\\* = {lam_star} 的 validation Masked ACC − 主系統 validation WP | {gm3['val_fused']:.4f} − {gm3['val_main']:.4f} = {gm3['diff_ii']:+.4f} | ≥ +0.005 | {'滿足' if gm3['pass_ii'] else '不滿足'} |",
            f"| **G-M3** | | (i) 且 (ii) | **{'通過' if gm3['pass'] else '未通過'}** |", "",
            f"參考（不改變判定）：(ii) 在附帶版本 z′（只除以 σ、不減平均；PREREG-16 操作定義 17）為 λ\\* = {sel['B4']['zprime']['lambda_star']} 的 validation "
            f"{sel['B4']['zprime']['val_masked_by_lambda'][str(sel['B4']['zprime']['lambda_star'])]:.4f} − {val_main:.4f} = "
            f"{sel['B4']['zprime']['val_masked_by_lambda'][str(sel['B4']['zprime']['lambda_star'])] - val_main:+.4f}；"
            f"主要版本 z 的 λ = 0（只置中、不融合）validation 為 {sel['B4']['z']['val_masked_by_lambda']['0.0']:.4f}（T4）。", "",
            "### G-M2（操作定義 23）", "",
            "| 格 | 該格 | 對角線 | 對角線 − 該格 | 門檻 | 結果 |", "|---|---|---|---|---|---|",
            f"| LUNG head 用在 ESCA slides | {gm2['cell_lung_head_on_esca']:.4f} | {gm2['diag_esca']:.4f}（ESCA head、ESCA） | {gm2['gap1']:+.4f} | < 0.01 | {'滿足' if gm2['gap1'] < 0.01 else '不滿足'} |",
            f"| ESCA head 用在 LUNG slides | {gm2['cell_esca_head_on_lung']:.4f} | {gm2['diag_lung']:.4f}（LUNG head、LUNG） | {gm2['gap2']:+.4f} | < 0.01 | {'滿足' if gm2['gap2'] < 0.01 else '不滿足'} |",
            f"| **G-M2** | | | | 兩格皆滿足 | **{'通過' if gm2['pass'] else '未通過'}** |", "",
            "B5 不設門檻。以上門檻只用來決定下一步做哪個設計，不是論文結果。", ""]

    # T6
    tr = sum(float(st.split(s, f)["per_task"][t]["t_read_s"].sum()) for s in ("val", "test") for f in folds for t in tasks)
    tc = sum(float(st.split(s, f)["per_task"][t]["t_compute_s"].sum()) for s in ("val", "test") for f in folds for t in tasks)
    timing = json.loads((st.root / "timing.json").read_text())
    nval = sum(len(st.split("val", f)["labels"]) for f in folds)
    tail = ["## T6 一致性檢查、重算範圍與耗時", "",
            "| 檢查 | 結果 | 判準 |", "|---|---|---|",
            f"| 5a test I6_cos8（4 expert）與 NC-8 快取 8 類 cosine 最大絕對差 | {chk['5a_max_cos']:.2e}；2 類內 argmax 全同 | ≤ 1e-6 |",
            f"| 5b test mean_vec 與 NC-8 快取最大絕對差 | {chk['5b_max_mv']:.2e} | ≤ 1e-6 |",
            f"| 5c 主系統逐折 ACC／Masked／TP／WP 與 `nc8/per_fold.json` 逐折最大絕對差；ACC 平均 | {chk['5c_max_fold_diff']:.2e}；"
            f"{chk['main_cil_mean']['reverse']:.4f}／{chk['main_cil_mean']['paper']:.4f} | ≤ 1e-9；0.9128 |",
            f"| 5d B2 對角線 vs I6_cos8 cosine 最大絕對差；argmax | {chk['5d_diag_max_cos']:.2e}；全同 | ≤ 1e-6；全同 |",
            f"| 5e validation 自家 expert vs `i6/eval_fold{{f}}.pt` cosine 最大絕對差；argmax | {sel['checks']['5e']['max_abs_cos_diff']:.2e}；"
            f"{'全同' if sel['checks']['5e']['argmax_all_equal'] else '不同'} | ≤ 1e-6；全同 |",
            f"| 5f 主系統 validation WP（四任務等權、十折平均） | {sel['checks']['5f']['val_wp_main']:.4f} | 0.9194 |", "",
            "重算範圍（既有快取不足的部分）：validation 10 折 " + f"{nval} 張（mean_vec 與 4 個 expert 的四輪 8 類 cosine；既有 `i6/eval_fold` 的 validation 只有自家 expert、沒有 mean_vec）；"
            f"test 10 折 {sum(n_task)} 張（B2 的 3 個非對角 head 與 g = 0 的四輪 8 類 cosine，另重算 4 個 expert 供檢查）。"
            f"PREREG 估計 ≤ 6 分鐘；實際 `moe0_infer` 全部 {timing.get('run/val,test/1-10', float('nan')):.0f} 秒（wall clock、執行緒 8）；"
            f"逐張記錄加總：讀檔 `t_read_s` {tr:.0f} 秒、計算 `t_compute_s` {tc:.0f} 秒。", "",
            "## 檔案", "", "| 檔案 | 內容 |", "|---|---|",
            "| `outputs/navcil/mac/moe0/b3_per_slide.csv` | B3 per-slide（fold、slide_id、task、true_class、d_a、d_b） |",
            "| `outputs/navcil/mac/moe0/selection.json` | validation 上選 λ、T 的結果與 5e／5f 檢查（先於 test 讀取 commit） |",
            "| `outputs/navcil/mac/moe0/results.json` | 本報告的數值（機器可讀） |",
            "| `outputs/navcil/mac/moe0/{val,test}_fold{f}.pt` | 新快取（不進版控） |", ""]
    (st.b.out / "REPORT_moe0.md").write_text("\n".join(head + out_md + tail))
    (st.root / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: o if not hasattr(o, "tolist") else o.tolist()))
    print("→", st.b.out / "REPORT_moe0.md")
    print(json.dumps({"G-M3": gm3, "G-M2": gm2}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
