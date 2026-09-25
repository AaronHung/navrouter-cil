#!/usr/bin/env python3
"""NC-3 報告：AMENDMENT-1 驗證、L 線 v2、第五關 router 機制、完整系統列 v2。

只讀快取、lora 評估檔與驗證 JSON，不碰特徵檔。
    NAVCIL_MACHINE=mac python scripts/nc3_report.py
輸出：outputs/navcil/<machine>/REPORT_stage5.md、nc3/metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
from selector.cil_eval import mean_sd, subset_key                         # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402

FOLDS = N2.FOLDS
QS = (0.0, 0.1, 0.2, 0.3, 0.5)
VARIANTS = ("基準", "R6", "R7", "R8")
BYTES = {"基準": 16384, "R6": 16384, "R7": 4096, "R8": 16384}
EXT = json.loads((REPO_ROOT / "reference" / "external_baselines.json").read_text())
fmt = N2.fmt


def gather3(d, fold, split, seen) -> dict:
    g = N2.gather_r(d, fold, split, seen)
    parts = [torch.load(d.cache / f"nc3_fold{fold}_{split}_{d.tasks[p]}.pt", map_location="cpu")
             for p in seen]
    si = parts[0]["subsets"].index(subset_key(seen))
    for k in ("r6_counts", "r6_msum", "r7_counts", "r7_dsum"):
        g[k] = torch.cat([c[k][:, si] for c in parts])
    assert torch.equal(torch.cat([c["labels"] for c in parts]), g["labels"])
    return g


def scores(d, fold, g, seen, variant) -> torch.Tensor:
    if variant in ("基準", "R8"):
        return N2.scores_for(d, fold, g, seen, "R3", 8).double()
    if variant == "R6":
        return g["r6_counts"][:, seen].double() * 1e12 + g["r6_msum"][:, seen]
    if variant == "R7":
        return g["r7_counts"][:, seen].double() * 1e12 - g["r7_dsum"][:, seen]
    raise ValueError(variant)


def predict(d, fold, g, seen, variant, four, theta=None):
    """回傳 (pred, masked_pred, 分派任務, 是否會診)。four = 各 expert 的 8 類 cosine [N, 4, 8]。"""
    sc = scores(d, fold, g, seen, variant)
    st = torch.tensor(seen)
    th = st[sc.argmax(-1)]
    pred, mk = N2.hard(four, th, g["task"])
    consult = torch.zeros(len(th), dtype=torch.bool)
    if variant == "R8" and len(seen) >= 2:
        top = sc.topk(2, dim=-1)
        consult = (top.values[:, 0] - top.values[:, 1]) < theta
        for i in consult.nonzero().flatten().tolist():
            t1, t2 = int(st[top.indices[i, 0]]), int(st[top.indices[i, 1]])
            rows = task_rows(t1) + task_rows(t2)
            vals = torch.stack([four[i, t1, rows[0]], four[i, t1, rows[1]],
                                four[i, t2, rows[2]], four[i, t2, rows[3]]])
            c = rows[int(vals.argmax())]
            pred[i] = c
            tp = c // 2
            th[i] = tp
            rt = task_rows(int(g["task"][i]))
            mk[i] = rt[int(four[i, tp, rt].argmax())]
    return pred, mk, th, consult


def margins(d, fold, g, seen) -> torch.Tensor:
    top = scores(d, fold, g, seen, "基準").topk(2, dim=-1).values
    return (top[:, 0] - top[:, 1]).float()


def t4_acc(g, pred, seen) -> float:
    ok = pred == g["labels"]
    return sum(ok[g["task"] == p].float().mean().item() for p in seen) / len(seen)


def cil_variant(d, variant, order, four_fn, theta_f) -> list:
    out = []
    for f in FOLDS:
        def stage(seen, f=f):
            g = gather3(d, f, "test", seen)
            pred, mk, _, _ = predict(d, f, g, seen, variant, four_fn(f, seen, g), theta_f.get(f))
            return pred, mk, g
        out.append(N2.cil_from(stage, order, d.tasks))
    return out


def main() -> int:
    t0 = time.perf_counter()
    d = N2.D2()
    M1 = json.loads((d.out / "nc1" / "metrics.json").read_text())
    nc3 = d.out / "nc3"
    M = {}

    # ── A ─────────────────────────────────────────────────────────────────
    A = {"i": {t: json.loads((nc3 / f"amendment_i_t{t}.json").read_text()) for t in (1, 3, 8)},
         "ii": json.loads((nc3 / "amendment_ii.json").read_text()),
         "ref64": json.loads((nc3 / "amendment_ref64.json").read_text())}
    A["pass_i"] = all(v["finite"] for v in A["i"].values())
    A["pass_ii"] = A["ii"]["cases"]["r2"]["max_abs_diff"] <= 1e-5
    M["amendment"] = A

    # ── B ─────────────────────────────────────────────────────────────────
    L0 = M1["stage1"]["fold_mean"]["e_fourround"]
    L0m = mean_sd(L0)[0]
    Lv2, gv2 = N2.compute_L(d, L0m, "lora_v2")
    Lv1, gv1 = N2.compute_L(d, L0m, "lora")
    M["gates_L_v2"], M["gates_L_v1"] = gv2, gv1
    r_star = gv2["r_star"]

    # ── C ─────────────────────────────────────────────────────────────────
    seen4 = list(range(4))
    four_L0 = lambda f, seen, g: g["four"]                                  # noqa: E731
    val = {v: [] for v in ("基準", "R6", "R7")}
    val_r8 = {q: [] for q in QS}
    theta = {q: {} for q in QS}
    for f in FOLDS:
        g = gather3(d, f, "val", seen4)
        for v in val:
            val[v].append(t4_acc(g, predict(d, f, g, seen4, v, g["four"])[0], seen4))
        mg = margins(d, f, g, seen4)
        for q in QS:
            theta[q][f] = float(torch.quantile(mg, q))
            val_r8[q].append(t4_acc(g, predict(d, f, g, seen4, "R8", g["four"], theta[q][f])[0], seen4))
    best_q = max(mean_sd(val_r8[q])[0] for q in QS)
    q_star = min(q for q in QS if mean_sd(val_r8[q])[0] == best_q)
    val["R8"] = val_r8[q_star]
    th_star = theta[q_star]
    best_v = max(mean_sd(val[v])[0] for v in VARIANTS)
    selected = next(v for v in VARIANTS if mean_sd(val[v])[0] == best_v)
    M["val"] = {"acc": val, "r8_by_q": {str(q): val_r8[q] for q in QS},
                "theta": {str(q): theta[q] for q in QS}, "q_star": q_star, "selected": selected}

    test = {}
    for v in VARIANTS:
        th_f = th_star if v == "R8" else {}
        res = {"cil": {o: cil_variant(d, v, o, four_L0, th_f) for o in ORDERS},
               "tp_task": {p: [] for p in seen4}, "tp_macro": [], "tp_micro": [],
               "esca_to_lung": 0, "lung_to_esca": 0, "consult": 0, "n": 0, "fixed": 0, "broken": 0}
        ie, il = d.tasks.index("tcga_esca"), d.tasks.index("tcga_lung")
        for f in FOLDS:
            g = gather3(d, f, "test", seen4)
            pred, _, tp, cons = predict(d, f, g, seen4, v, g["four"], th_f.get(f))
            base_pred = predict(d, f, g, seen4, "基準", g["four"])[0]
            ok_tp = tp == g["task"]
            accs = [ok_tp[g["task"] == p].float().mean().item() for p in seen4]
            for p in seen4:
                res["tp_task"][p].append(accs[p])
            res["tp_macro"].append(sum(accs) / 4)
            res["tp_micro"].append(ok_tp.float().mean().item())
            res["esca_to_lung"] += int(((g["task"] == ie) & (tp == il)).sum())
            res["lung_to_esca"] += int(((g["task"] == il) & (tp == ie)).sum())
            res["consult"] += int(cons.sum()); res["n"] += len(cons)
            ok, okb = pred == g["labels"], base_pred == g["labels"]
            res["fixed"] += int((ok & ~okb).sum()); res["broken"] += int((~ok & okb).sum())
        res["acc_t4"] = [x["acc"] for x in res["cil"]["reverse"]]
        test[v] = res
    for v, r in test.items():
        diffs = [a - b for a, b in zip(r["acc_t4"], test["基準"]["acc_t4"])]
        r["diff"] = {"per_fold": diffs, "mean": mean_sd(diffs)[0], "wins": sum(x > 0 for x in diffs)}
    sel = test[selected]
    r_mech = (selected != "基準" and sel["diff"]["mean"] >= 0.01 and sel["diff"]["wins"] >= 7
              and BYTES[selected] <= 16384)
    M["gate_R_mech"] = {"selected": selected, "pass": r_mech}

    # ── D ─────────────────────────────────────────────────────────────────
    router_D = selected if r_mech else "基準"
    full = {}
    if r_star is not None:
        for o in ORDERS:
            evs = {f: ev for f, ev in zip(FOLDS, Lv2[(r_star, o)]["evs"])}
            four_L1 = lambda f, seen, g, evs=evs: torch.cat(                  # noqa: E731
                [evs[f]["tasks"][d.tasks[p]]["l1_cos8"] for p in seen])
            cil = cil_variant(d, router_D, o, four_L1, th_star if router_D == "R8" else {})
            first = d.tasks.index(ORDERS[o][0])
            by = {t: (N2.FULL_PARAMS if p == first else N2.lora_params(r_star)) * 4 + BYTES[router_D]
                  for p, t in enumerate(d.tasks)}
            full[o] = {"cil": cil, "bytes": by}

    # ── metrics ─────────────────────────────────────────────────────────────
    timing = {p.stem: json.loads(p.read_text()) for p in sorted((d.out / "nc2").glob("timing_lora_v2_*.json"))}
    if (nc3 / "timing_rcache.json").exists():
        timing["nc3_rcache"] = json.loads((nc3 / "timing_rcache.json").read_text())
    M["timing"] = timing
    M["test"] = {v: {k: r[k] for k in r if k != "cil"} | {"cil": {o: [{k: x[k] for k in ("acc", "masked", "forgetting", "R")}
                                                                    for x in r["cil"][o]] for o in ORDERS}}
                 for v, r in test.items()}
    strip = lambda L: {f"r{r}|{o}": (None if v is None else {k: v[k] for k in ("n_nan", "wp", "wp_task", "pearson",  # noqa: E731
                        "jaccard_l0", "fro", "jaccard_m_l1")} | {"l2": [{k: x[k] for k in ("acc", "masked", "forgetting")}
                                                                    for x in v["l2"]]}) for (r, o), v in L.items()}
    M["L_v2"], M["L_v1"] = strip(Lv2), strip(Lv1)
    M["full_system_v2"] = {o: {"router": router_D, "r_star": r_star,
                               "acc": [x["acc"] for x in v["cil"]], "masked": [x["masked"] for x in v["cil"]],
                               "forgetting": [x["forgetting"] for x in v["cil"]], "bytes": v["bytes"]}
                           for o, v in full.items()}
    nc3.mkdir(exist_ok=True)
    (nc3 / "metrics.json").write_text(json.dumps(M, indent=1, default=str))
    write(d, M, A, Lv1, Lv2, gv1, gv2, L0, test, val, val_r8, full, router_D, time.perf_counter() - t0)
    print(f"→ {d.out / 'REPORT_stage5.md'}")
    return 0


def write(d, M, A, Lv1, Lv2, gv1, gv2, L0, test, val, val_r8, full, router_D, t_rep):
    ok = lambda b: "**通過**" if b else "**未通過**"                     # noqa: E731
    L0m = mean_sd(L0)[0]
    sel = M["val"]["selected"]
    gm = M["gate_R_mech"]
    rs = gv2["r_star"]
    out = ["# REPORT — NC-3：AMENDMENT-1、L 線 v2、第五關 router 機制（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；所有數字來自"
           "同一台、同一批。文件：`AMENDMENT-1.md`（7ca67e4）、`PREREG-2.md`（f79ce66）、`PREREG-3.md`（51b636f）。", "",
           "## 判準落點", "", "| 文件 | 判準 | 數值 | 所在表格 | 結果 |", "|---|---|---|---|---|",
           f"| AMENDMENT-1 | (i) fold 1 reverse RCC、r=1，1／3／8 條執行緒各 5 epochs 無 NaN | "
           + "、".join(f"{t} 條：{'有限' if v['finite'] else 'NaN'}" for t, v in A["i"].items())
           + f" | T-A | {ok(A['pass_i'])} |",
           f"| AMENDMENT-1 | (ii) r=2 新舊寫法 50 步參數最大差 ≤ 1e-5 | {A['ii']['cases']['r2']['max_abs_diff']:.2e} | T-A | "
           f"{ok(A['pass_ii'])} |",
           f"| PREREG-2（v2） | L-pass：最小 r 使兩序十折 WP ≥ L0 − 0.01（{L0m - 0.01:.4f}）且 r ≤ 4 | 符合的 r：{gv2['passing_r'] or '無'}"
           f"；NaN：{gv2['invalid_r'] or '無'} | T-B | {ok(gv2['L_pass'])}"
           + (f"（r\\* = {rs}）" if gv2["L_pass"] else "") + " |",
           f"| PREREG-3 | R8 的分位數 q（validation） | q\\* = {M['val']['q_star']} | T-C1 | — |",
           f"| PREREG-3 | 選出的 router（validation 平均 CIL） | **{sel}**（{mean_sd(val[sel])[0]:.4f}） | T-C1 | — |",
           f"| PREREG-3 | R-機制：選出者 ≠ 基準、test 差 ≥ +0.01、贏 ≥ 7/10、≤ 16 KB | "
           f"{sel}；差 {test[sel]['diff']['mean']:+.4f}、贏 {test[sel]['diff']['wins']}/10、{BYTES[sel]:,} bytes | T-C2 | "
           f"{ok(gm['pass'])} |", ""]

    # A
    out += ["## T-A AMENDMENT-1 驗證", "",
            "修法：r = 1 的 LoRA 分支改為 `(u ⊙ A[0]).sum(-1) ⊙ B[:,0]`（與合併用的 B·A 同樣改為廣播外積），"
            "前向與反向都不經 BLAS gemv；r ≥ 2 不變。每步檢查梯度與參數有限性；執行緒固定 8。", "",
            "| 驗證 | 執行緒 | 結果 | epoch loss（1–5） | 秒 |", "|---|---|---|---|---|"]
    for t, v in A["i"].items():
        out.append(f"| (i) r=1、fold 1 reverse RCC、5 epochs | {t} | {'全程有限' if v['finite'] else v.get('error')} | "
                   + ", ".join(f"{x:.4f}" for x in v.get("epoch_loss", [])) + f" | {v['seconds']} |")
    c = A["ii"]["cases"]
    out += ["", "| 驗證 | 參數最大絕對差 | 門檻 | 結果 |", "|---|---|---|---|",
            f"| (ii) r=2 新舊寫法各 50 步 | {c['r2']['max_abs_diff']:.2e} | ≤ 1e-5 | {ok(A['pass_ii'])} |",
            f"| (加做) r=1 新舊寫法各 50 步 | {c['r1']['max_abs_diff']:.2e}（{max(c['r1']['per_param'], key=c['r1']['per_param'].get)}） | "
            "不設門檻 | — |",
            f"| (加做) r=1 新寫法 vs float64 參考 | {A['ref64']['new_fp32']['max_abs_diff_vs_fp64']:.2e} | — | — |",
            f"| (加做) r=1 舊寫法 vs float64 參考 | {A['ref64']['old_fp32']['max_abs_diff_vs_fp64']:.2e} | — | — |", "",
            "r = 2 的程式路徑未改動，所以新舊逐位元相同。r = 1 新舊之間的差與兩者各自對 float64 參考的差同一量級，"
            "屬 float32 加總順序造成的捨入差。", ""]

    # B
    out += ["## T-B L 線 v2（AMENDMENT-1 後，依序重跑 r = 1、2、4、8）", "",
            f"L0 四任務平均 WP = {fmt(L0)}；L-pass 門檻 {L0m - 0.01:.4f}。v1 = NC-2（commit 3a9e622；r=1 無效）。", "",
            "| r | 序 | v2 WP | v1 WP | v2 − v1 | ≥ 門檻（v2） | v2 NaN 增量 | 非底座參數量 |", "|---|---|---|---|---|---|---|---|"]
    for r in N2.R_LIST:
        for o in ORDERS:
            a, b = Lv2.get((r, o)), Lv1.get((r, o))
            if a is None:
                out.append(f"| {r} | {o} | 未完成 | | | | | |")
                continue
            ma = mean_sd(a["wp"])[0]
            v1s = "無效（NaN）" if (b is None or b["n_nan"]) else fmt(b["wp"])
            dd = "—" if (b is None or b["n_nan"]) else f"{ma - mean_sd(b['wp'])[0]:+.4f}"
            out.append(f"| {r} | {o} | {fmt(a['wp'])} | {v1s} | {dd} | {'✓' if ma >= L0m - 0.01 else '✗'} | "
                       f"{a['n_nan']}/{a['n_inc']} | {N2.lora_params(r):,} |")
    out += ["", "v2 每任務 WP：", "", "| r | 序 | " + " | ".join(d.tasks) + " |", "|---|---|" + "---|" * 4]
    for r in N2.R_LIST:
        for o in ORDERS:
            a = Lv2.get((r, o))
            if a:
                out.append(f"| {r} | {o} | " + " | ".join(fmt(a["wp_task"][p]) for p in range(4)) + " |")
    out += ["", "v2 狀態／行為指標（非底座任務，十折平均）：", "",
            "| r | 序 | 任務 | Pearson vs L0 | Jaccard vs L0 | ‖BA‖_F |", "|---|---|---|---|---|---|"]
    for r in N2.R_LIST:
        for o in ORDERS:
            a = Lv2.get((r, o))
            if a:
                for t in ORDERS[o][1:]:
                    out.append(f"| {r} | {o} | {t} | {fmt(a['pearson'][t])} | {fmt(a['jaccard_l0'][t])} | {fmt(a['fro'][t])} |")
    out += ["", "v2 L2（合併 expert、不用 router、已見類別內判）：", "",
            "| r | 序 | ACC | Masked ACC | Forgetting |", "|---|---|---|---|---|"]
    for r in N2.R_LIST:
        for o in ORDERS:
            a = Lv2.get((r, o))
            if a:
                out.append(f"| {r} | {o} | {fmt([x['acc'] for x in a['l2']])} | {fmt([x['masked'] for x in a['l2']])} | "
                           f"{fmt([x['forgetting'] for x in a['l2']])} |")

    # C
    out += ["", "## T-C 第五關 router 機制（L0 expert、Hard）", "", "### T-C1 validation 選法（十折 validation、t = 4 CIL ACC）", "",
            "| 候選 | validation CIL ACC | 備註 |", "|---|---|---|"]
    for q in QS:
        out.append(f"| R8（q = {q}） | {fmt(val_r8[q])} | {'q\\* ' if q == M['val']['q_star'] else ''}"
                   f"{'（q = 0 即基準）' if q == 0 else ''} |")
    for v in VARIANTS:
        out.append(f"| {v}{'（q = ' + str(M['val']['q_star']) + '）' if v == 'R8' else ''} | {fmt(val[v])} | "
                   f"{'**選出**' if v == sel else ''} |")
    out += ["", "### T-C2 test 總表", "",
            "| 變體 | 儲存 bytes／任務 | TP macro | TP micro | ESCA→Lung | Lung→ESCA | reverse ACC | reverse Forgetting | "
            "paper ACC | paper Forgetting | Masked ACC | 對基準差 | 贏折數 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, r in test.items():
        out.append(f"| {v}{' ★' if v == sel else ''} | {BYTES[v]:,} | {fmt(r['tp_macro'])} | {fmt(r['tp_micro'])} | "
                   f"{r['esca_to_lung']} | {r['lung_to_esca']} | {fmt([x['acc'] for x in r['cil']['reverse']])} | "
                   f"{fmt([x['forgetting'] for x in r['cil']['reverse']])} | {fmt([x['acc'] for x in r['cil']['paper']])} | "
                   f"{fmt([x['forgetting'] for x in r['cil']['paper']])} | {fmt([x['masked'] for x in r['cil']['reverse']])} | "
                   f"{r['diff']['mean']:+.4f} | {r['diff']['wins']}/10 |")
    out += ["", "★ = validation 選出者。ESCA↔Lung 為 t = 4 十折合計張數；兩序在 t = 4 的 ACC 與 Masked ACC 相同。", "",
            "每任務 TP（t = 4）：", "", "| 變體 | " + " | ".join(d.tasks) + " |", "|---|" + "---|" * 4]
    for v, r in test.items():
        out.append(f"| {v} | " + " | ".join(fmt(r["tp_task"][p]) for p in range(4)) + " |")
    r8 = test["R8"]
    out += ["", f"R8（q\\* = {M['val']['q_star']}）：test t = 4 會診 {r8['consult']}/{r8['n']} 張"
            f"（{r8['consult'] / max(r8['n'], 1):.2%}）；相對基準改對 {r8['fixed']} 張、改錯 {r8['broken']} 張（十折合計）。"
            + ("q = 0 時 θ = 各折 validation 的最小 margin，validation 上不會會診（與基準相同）；test 上仍有 margin "
               "更小的 slide 會觸發會診，所以 test 的 R8 與基準不完全相同。validation 平均 CIL 同分時依 PREREG-3 "
               "操作定義 6 取基準。" if M["val"]["q_star"] == 0.0 else ""), "",
            "每折對基準的差（test，t = 4 ACC）：", "", "| 變體 | " + " | ".join(str(f) for f in FOLDS) + " |", "|---|" + "---|" * 10]
    for v in VARIANTS[1:]:
        out.append(f"| {v} | " + " | ".join(f"{x:+.4f}" for x in test[v]["diff"]["per_fold"]) + " |")

    # D
    out += ["", "## T-D 完整系統列 v2", "",
            f"router = {'R3(k=8)（基準）' if router_D == '基準' else router_D}"
            f"{'（第五關未通過，維持 R3(k=8)）' if not gm['pass'] else ''}；expert = L1(r = {rs}) v2；Hard。", "",
            "Masked ACC 的算法：在真實任務的 2 類中取 argmax，證據用**分派後的 expert**（router 選出的 τ̂，"
            "R8 會診時為最終預測類別所屬任務的 expert）四輪選片的 8 類 cosine；不是用真實任務的 expert。", "",
            "| 序 | ACC | Masked ACC | Forgetting | 每任務儲存 bytes（expert fp32 ＋ router key） |", "|---|---|---|---|---|"]
    for o, v in full.items():
        by = "、".join(f"{t.split('_')[1]} {b:,}" for t, b in v["bytes"].items())
        out.append(f"| {o} | {fmt([x['acc'] for x in v['cil']])} | {fmt([x['masked'] for x in v['cil']])} | "
                   f"{fmt([x['forgetting'] for x in v['cil']])} | {by}（平均 {sum(v['bytes'].values()) / 4:,.0f}） |")
    for o, lab in (("reverse", "reverse"), ("paper", "forward（paper）")):
        out.append(f"| *外部參考：{EXT['method']} {lab}（已發表值，非 paired）* | *{EXT['acc'][o]:.3f}* | "
                   f"*{EXT['masked_acc'][o]:.3f}* | *{EXT['forgetting'][o]:.3f}* | |")

    # T5
    tm = M["timing"]
    out += ["", "## T-E 實際耗時（秒，wall clock；執行緒 8，全部依序執行）", "", "| 項目 | 秒 |", "|---|---|",
            "| AMENDMENT-1 驗證 (i) 1／3／8 條 | " + "／".join(str(A["i"][t]["seconds"]) for t in (1, 3, 8)) + " |"]
    for r in N2.R_LIST:
        t = tm.get(f"timing_lora_v2_r{r}", {})
        tr = sum(v for k, v in t.items() if k.startswith("train/"))
        ev = sum(v for k, v in t.items() if k.startswith("eval/"))
        out.append(f"| L 線 v2 r={r}：訓練 {tr:,.0f}、評估 {ev:,.0f} | {t.get(f'run/r{r}', float('nan')):,.0f} |")
    rc = tm.get("nc3_rcache", {})
    out.append(f"| 第五關快取：key 建立 {sum(v for k, v in rc.items() if k.startswith('nc3/keys')):,.0f}、"
               f"val／test 投票 {sum(v for k, v in rc.items() if k.startswith('nc3/cache')):,.0f} | "
               f"{rc.get('run/1-10', float('nan')):,.0f} |")
    out.append(f"| 報告（離線計算，含 k-means） | {t_rep:,.0f} |")
    out += ["", "逐張讀檔／計算秒數：`lora_v2/r*/*/fold*_eval.pt`、`cache/nc3_*.pt` 的 `t_read_s`、`t_compute_s`；"
            "全部數值另存 `nc3/metrics.json`。", ""]
    (d.out / "REPORT_stage5.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
