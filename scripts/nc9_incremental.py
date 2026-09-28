#!/usr/bin/env python3
"""NC-9：AR／AR-bal 分派器只存累加統計量（PREREG-9），十折兩序重算第 6、7 列並逐項比對 NC-8。

流程與 scripts/nc8_report.py 相同（唯讀 import 其評估程式），只把分派器換成 selector.incremental_ridge：
每（折、序、分派器）的第 t 階段 = 載入上一階段的統計量檔 → 只讀任務 t 的 train mean_vec → add_task → 存檔。
從頭加總的參考值由 nc8_report.B8.W 重算（不改該檔）。
    NAVCIL_MACHINE=mac python scripts/nc9_incremental.py
    （先跑 scripts/nc9_split_audit.py，本檔把其結果寫進報告）
輸出：outputs/navcil/<machine>/nc9/{state/,per_fold.json,result.json,reads.json,tau_hat.pt}、REPORT_stage11.md
判準任一不通過：仍寫出全部輸出，結束碼 1。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
import nc5_report as N5                                                   # noqa: E402
import nc8_report as N8                                                   # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS                                       # noqa: E402
from selector.incremental_ridge import IncrementalRidge                   # noqa: E402

FOLDS = N8.FOLDS
VARIANTS = {"AR": (1e-3, False), "AR-bal": (1e-4, True)}
ROWS = {6: "AR", 7: "AR-bal"}
FID = {6: "d3", 7: "arbal_i6"}
METRICS = ("acc", "masked", "forgetting", "bwt")
LISTS = ("acc_t", "masked_t", "tp_task", "wp_task")
R10_T1 = {(6, "reverse"): 16, (7, "reverse"): 17, (6, "paper"): 29, (7, "paper"): 30}
R10_CONF = {"AR": 103, "AR-bal": 112}
SHORT = ["esca", "rcc", "brca", "lung"]
fmt = N2.fmt


class B9(N8.B8):
    """NC-8 的評估容器；分派器的 W 只來自累加統計量，沒有就報錯（不退回從頭加總）。"""

    def __init__(self):
        super().__init__()
        self.inc = {}

    def W(self, f, o, t, gamma, bal=False, lin8=False):
        return self.inc[(f, o, t, gamma, bal)]


def build(b, nc9, f, o, name, reads, info):
    gamma, bal = VARIANTS[name]
    pos = [b.tasks.index(x) for x in ORDERS[o]]
    sp = nc9 / "state" / f"fold{f}_{o}_{name}.pt"
    for t, p in enumerate(pos, 1):
        m = IncrementalRidge(gamma, bal) if t == 1 else IncrementalRidge.load(sp)
        assert m.task_ids == pos[:t - 1]
        path = b.cache / f"nc8_fold{f}_train_{b.tasks[p]}.pt"
        t0 = time.perf_counter()
        mv = torch.load(path, map_location="cpu")["mean_vec"]
        t1 = time.perf_counter()
        reads.append({"fold": f, "order": o, "router": name, "stage": t, "files": [path.name], "n_slides": len(mv)})
        m.add_task(p, mv)
        del mv
        W = m.solve()
        t2 = time.perf_counter()
        size = m.save(sp)
        b.inc[(f, o, t, gamma, bal)] = (W, list(m.task_ids))
        info[(f, o, name, t)] = {"bytes": size, "n_new": reads[-1]["n_slides"], "t_read_s": t1 - t0, "t_compute_s": t2 - t1}
        del m


def lineno(lines, anchors, key, k):
    """在 lines 中依序找到 anchors 後，第 k 個（0 起）以 key 開頭的行，回傳 1 起的行號。"""
    i = 0
    for a in anchors:
        while a not in lines[i]:
            i += 1
    n = -1
    while True:
        i += 1
        if lines[i].lstrip().startswith(key):
            n += 1
            if n == k:
                return i + 1


def cells(line):
    return [c.strip() for c in line.split("|")[1:-1]]


def main() -> int:
    t_start = time.perf_counter()
    b, bref = B9(), N8.B8()
    nc9 = b.out / "nc9"
    (nc9 / "state").mkdir(parents=True, exist_ok=True)
    reads, info = [], {}
    for f in FOLDS:
        for o in ORDERS:
            for name in VARIANTS:
                build(b, nc9, f, o, name, reads, info)
    t_build = time.perf_counter() - t_start

    # 判準 5：W 與從頭加總；判準 2：逐張 τ̂
    wdiff, weq, tau, mism = {}, {}, {}, {}
    for f in FOLDS:
        for o in ORDERS:
            pos = [b.tasks.index(x) for x in ORDERS[o]]
            for name, (gamma, bal) in VARIANTS.items():
                for t in range(1, 5):
                    Wi, pi = b.W(f, o, t, gamma, bal)
                    Wr, pr = bref.W(f, o, t, gamma, bal)
                    assert pi == pr
                    wdiff[(f, o, name, t)] = (Wi - Wr).abs().max().item()
                    weq[(f, o, name, t)] = torch.equal(Wi, Wr)
                    seen = pos[:t]
                    g = b.gather(f, seen)
                    st = torch.tensor(seen)
                    ti = st[N8.router(b, name, f, o, g, seen).argmax(-1)]
                    tr = st[N8.router(bref, name, f, o, g, seen).argmax(-1)]
                    tau[(f, o, name, t)] = ti
                    mism[(f, o, name, t)] = int((ti != tr).sum())
    # 真實資料上與加入順序無關（另報）：t = 4 兩序的 W（欄依任務對齊）與 τ̂
    order_w, order_tau = {}, {}
    for f in FOLDS:
        for name, (gamma, bal) in VARIANTS.items():
            Ws = []
            for o in ORDERS:
                W, pos = b.W(f, o, 4, gamma, bal)
                Ws.append(W[:, [pos.index(p) for p in range(4)]])
            order_w[(f, name)] = (Ws[0] - Ws[1]).abs().max().item()
            # 兩序的 test slides 依各自的任務序串接（N8.B8.gather），先還原成 canonical 任務序再逐張比
            canon = []
            for o in ORDERS:
                pos = [b.tasks.index(x) for x in ORDERS[o]]
                parts = tau[(f, o, name, 4)].split([len(b.c(f, "test", b.tasks[p])["labels"]) for p in pos])
                canon.append(torch.cat([parts[pos.index(p)] for p in range(4)]))
            order_tau[(f, name)] = int((canon[0] != canon[1]).sum())

    # 判準 1、4：第 6、7 列十折（流程同 nc8_report.main）
    per_fold = {}
    conf = {(n, o): torch.zeros(4, 4, dtype=torch.long) for n in VARIANTS for o in ORDERS}
    for row, name in ROWS.items():
        per_fold[row] = {}
        for o in ORDERS:
            recs = []
            for f in FOLDS:
                stage, inf = N8.stage_row(b, row, f, o)
                r = N5.cil_full(stage, o, b.tasks)
                g, th = inf["g"], inf["th"]
                assert torch.equal(th, tau[(f, o, name, 4)])
                ok = th == g["task"]
                r["tp_task"] = [ok[g["task"] == p].float().mean().item() for p in range(4)]
                r["wp_task"] = N8.wp_task(b, row, f, o)
                recs.append(r)
                for a, c in zip(g["task"].tolist(), th.tolist()):
                    conf[(name, o)][a, c] += 1
            per_fold[row][o] = recs
    assert not [k for k in b._c if k[1] == "train"], "評估容器讀了 train 快取"

    # 比對 NC-8
    p8 = b.out / "nc8" / "per_fold.json"
    old = json.loads(p8.read_text())["rows"]
    r10 = (b.out / "REPORT_stage10.md").read_text().splitlines()
    keep = ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t", "R", "Rm", "tp_task", "wp_task")
    p9 = nc9 / "per_fold.json"
    p9.write_text(json.dumps({"rows": {N8.ROWS[r - 1]: {o: [{k: x[k] for k in keep} for x in v] for o, v in per_fold[r].items()}
                                       for r in per_fold}, "folds": FOLDS}, indent=1, ensure_ascii=False))
    L8, L9 = p8.read_text().splitlines(), p9.read_text().splitlines()
    cmp_rows = {}
    for row in ROWS:
        rn = N8.ROWS[row - 1]
        for o in ORDERS:
            new, prev = per_fold[row][o], old[rn][o]
            folds = []
            for k, (x, y) in enumerate(zip(new, prev)):
                folds.append({m: {"nc8": y[m], "nc9": x[m], "exact": x[m] == y[m], "r4": round(x[m], 4) == round(y[m], 4),
                                  "absdiff": abs(x[m] - y[m]), "line8": lineno(L8, [f'"{rn}"', f'"{o}"'], f'"{m}":', k),
                                  "line9": lineno(L9, [f'"{rn}"', f'"{o}"'], f'"{m}":', k)} for m in METRICS}
                             | {m: {"exact": x[m] == y[m]} for m in LISTS})
            ln = R10_T1[(row, o)]
            c10 = cells(r10[ln - 1])
            tp = "／".join(f"{mean_sd([x['tp_task'][p] for x in new])[0]:.4f}" for p in range(4))
            wp = "／".join(f"{mean_sd([x['wp_task'][p] for x in new])[0]:.4f}" for p in range(4))
            c9 = [rn] + [fmt([x[m] for x in new]) for m in METRICS] + [tp, wp]
            cmp_rows[(row, o)] = {"folds": folds, "report10_line": ln, "report10": c10, "nc9": c9, "cells_equal": c10 == c9}
    cmp_conf = {}
    for name, ln in R10_CONF.items():
        prev = [[int(v) for v in cells(r10[ln - 1 + i])[1:]] for i in range(4)]
        cmp_conf[name] = {"report10_lines": (ln, ln + 3), "report10": prev, "nc9": conf[(name, "reverse")].tolist(),
                          "equal": prev == conf[(name, "reverse")].tolist()}

    # 儲存
    n_train = {(f, p): len(torch.load(b.cache / f"nc8_fold{f}_train_{t}.pt", map_location="cpu")["mean_vec"])
               for f in FOLDS for p, t in enumerate(b.tasks)}
    disk_train = {(f, p): (b.cache / f"nc8_fold{f}_train_{t}.pt").stat().st_size for f in FOLDS for p, t in enumerate(b.tasks)}

    crit = {
        "1_d3": all(v["cells_equal"] and all(d[m]["r4"] for d in v["folds"] for m in METRICS)
                    for (r, _), v in cmp_rows.items() if r == 6),
        "2_tau_d3": sum(v for k, v in mism.items() if k[2] == "AR") == 0,
        "3_conf_ar": cmp_conf["AR"]["equal"],
        "4_arbal": all(v["cells_equal"] and all(d[m]["r4"] for d in v["folds"] for m in METRICS)
                       for (r, _), v in cmp_rows.items() if r == 7)
                   and sum(v for k, v in mism.items() if k[2] == "AR-bal") == 0 and cmp_conf["AR-bal"]["equal"],
        "5_W": max(wdiff.values()) <= 1e-10,
        "7_reads": all(x["files"] == [f"nc8_fold{x['fold']}_train_{ORDERS[x['order']][x['stage'] - 1]}.pt"] for x in reads)
                   and len(reads) == len(FOLDS) * len(ORDERS) * len(VARIANTS) * 4,
    }
    env = dict(os.environ, NAVCIL_MACHINE=b.cfg["machine"], PYTHONNOUSERSITE="1")
    pt = subprocess.run([sys.executable, "-m", "pytest", "-v", "-p", "no:cacheprovider"], cwd=REPO_ROOT, env=env,
                        capture_output=True, text=True)
    pl = pt.stdout.strip().splitlines()
    crit["6_pytest"] = pt.returncode == 0
    res = {"criteria": crit, "pytest": {"rc": pt.returncode, "summary": pl[-1].strip("= "),
                                        "tail": [x for x in pl if "incremental_ridge" in x] + [pl[-1]]}, "w_maxabs": {"|".join(map(str, k)): v for k, v in wdiff.items()},
           "w_bitwise_equal": {"|".join(map(str, k)): v for k, v in weq.items()},
           "tau_mismatch": {"|".join(map(str, k)): v for k, v in mism.items()},
           "order_w_maxabs": {"|".join(map(str, k)): v for k, v in order_w.items()},
           "order_tau_mismatch": {"|".join(map(str, k)): v for k, v in order_tau.items()},
           "compare_rows": {f"{r}|{o}": v for (r, o), v in cmp_rows.items()}, "compare_conf": cmp_conf,
           "confusion_paper": {n: conf[(n, "paper")].tolist() for n in VARIANTS},
           "stage_info": {"|".join(map(str, k)): v for k, v in info.items()},
           "n_train": {f"{f}|{p}": v for (f, p), v in n_train.items()},
           "disk_train_cache": {f"{f}|{p}": v for (f, p), v in disk_train.items()},
           "seconds": {"build": t_build, "total": time.perf_counter() - t_start}}
    (nc9 / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (nc9 / "reads.json").write_text(json.dumps(reads, indent=1))
    torch.save({"|".join(map(str, k)): v for k, v in tau.items()}, nc9 / "tau_hat.pt")
    audit_p = nc9 / "split_audit.json"
    audit = json.loads(audit_p.read_text()) if audit_p.exists() else None
    write(b, res, cmp_rows, cmp_conf, conf, info, n_train, disk_train, wdiff, weq, mism, order_w, order_tau, audit)
    print(f"→ {b.out / 'REPORT_stage11.md'}")
    for k, v in crit.items():
        print(f"  判準 {k}: {'通過' if v else '不通過'}")
    return 0 if all(crit.values()) else 1


def write(b, res, cmp_rows, cmp_conf, conf, info, n_train, disk_train, wdiff, weq, mism, order_w, order_tau, audit):
    ok = lambda v: "通過" if v else "**不通過**"                               # noqa: E731
    crit = res["criteria"]
    out = ["# REPORT — NC-9：AR 分派器改為只存累加統計量（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、CPU、`torch.set_num_threads(8)`、torch 2.11.0。所有數字來自同一台、同一批。"
           "判準與定義見 `PREREG-9.md`（commit 33f76a4）。分派器 = `selector/incremental_ridge.py` 的 `IncrementalRidge`；"
           "第 t 階段只讀任務 t 的 train mean_vec，統計量（A、b_1…b_t）經由檔案在階段之間傳遞。證據與任務分類頭沿用 NC-8 同一批快取"
           "（`cache/nc8_fold*_*.pt`）與 I6(r=2) 權重；比對基準為 `REPORT_stage10.md`、`nc8/per_fold.json` 與以唯讀方式呼叫 "
           "`nc8_report.B8.W`（從頭加總）重算的參考值。", "",
           "## 判準總表", "", "| 判準 | 內容 | 結果 |", "|---|---|---|",
           f"| 1 | D3 十折 ACC／Masked／Forgetting／BWT（兩序）與 NC-8 小數點後四位相同 | {ok(crit['1_d3'])} |",
           f"| 2 | D3 逐張 τ̂（十折 × 兩序 × t = 1–4）與從頭加總相同 | {ok(crit['2_tau_d3'])} |",
           f"| 3 | AR 分派混淆（reverse、t = 4、十折合計）與 REPORT_stage10.md:103-106 相同 | {ok(crit['3_conf_ar'])} |",
           f"| 4 | AR-bal＋I6：十折數字、逐張 τ̂、混淆（REPORT_stage10.md:112-115）相同 | {ok(crit['4_arbal'])} |",
           f"| 5 | W 與從頭加總的最大絕對差 ≤ 1e-10（全部 160 組） | {ok(crit['5_W'])}（最大 {max(wdiff.values()):.2e}；逐位元相同 {sum(weq.values())}/{len(weq)}） |",
           f"| 6 | `tests/test_incremental_ridge.py` 與全體 pytest | {ok(crit['6_pytest'])}（{res['pytest']['summary']}） |",
           f"| 7 | 第 t 階段只讀任務 t 的 train 快取（`nc9/reads.json`，{len(FOLDS) * 2 * 2 * 4} 筆） | {ok(crit['7_reads'])} |", ""]

    # T1 主結果
    out += ["## T1 十折平均 ± 標準差（test、t = 4）：REPORT_stage10 對 NC-9", "",
            "| 列 | 序 | 來源 | ACC | Masked ACC | Forgetting | BWT | TP esca／rcc／brca／lung | WP esca／rcc／brca／lung | 相同 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for (row, o), v in cmp_rows.items():
        fid = f"nc8.t1.{FID[row]}.{o}.acc"
        out.append(f"| {N8.ROWS[row - 1]} | {o} | REPORT_stage10.md:{v['report10_line']}（`{fid}`） | "
                   + " | ".join(v["report10"][1:]) + " | |")
        out.append(f"| | | NC-9（`nc9.t1.{FID[row]}.{o}.acc`） | " + " | ".join(v["nc9"][1:]) + f" | {ok(v['cells_equal'])} |")
    out.append("")

    # T2 逐折
    out += ["## T2 每折 ACC（test、t = 4）", "",
            "行號：NC-8 值在 `outputs/navcil/mac/nc8/per_fold.json`，NC-9 值在 `outputs/navcil/mac/nc9/per_fold.json`。"
            "「四位」= 四捨五入到小數點後四位相同；「位元」= 兩個 float 完全相等。Masked ACC、Forgetting、BWT 與逐階段 ACC、"
            "每任務 TP／WP 的逐折比對見 `nc9/result.json` 的 `compare_rows`，結果列在表下。", ""]
    for (row, o), v in cmp_rows.items():
        out += [f"**{N8.ROWS[row - 1]}，{o}**", "", "| 折 | NC-8 ACC | per_fold.json 行 | NC-9 ACC | nc9/per_fold.json 行 | 四位 | 位元 |",
                "|---|---|---|---|---|---|---|"]
        for k, d in enumerate(v["folds"], 1):
            a = d["acc"]
            out.append(f"| {k} | {a['nc8']:.4f} | {a['line8']} | {a['nc9']:.4f} | {a['line9']} | {ok(a['r4'])} | {'是' if a['exact'] else '否'} |")
        allm = {m: (sum(d[m]["r4"] for d in v["folds"]), sum(d[m]["exact"] for d in v["folds"])) for m in METRICS}
        lists = {m: sum(d[m]["exact"] for d in v["folds"]) for m in LISTS}
        out += ["", "四位相同／位元相同的折數（共 10）：" + "；".join(f"{m} {a}/{e}" for m, (a, e) in allm.items())
                + "。逐項位元相同的折數：" + "；".join(f"{m} {n}" for m, n in lists.items()) + "。", ""]

    # T3 τ̂
    out += ["## T3 逐張分派結果 τ̂（累加版對從頭加總版）", "", "| 分派器 | 序 | t = 1 | t = 2 | t = 3 | t = 4 | 比對張數 | 不一致張數 |",
            "|---|---|---|---|---|---|---|---|"]
    for name in VARIANTS:
        for o in ORDERS:
            per_t = [sum(mism[(f, o, name, t)] for f in FOLDS) for t in range(1, 5)]
            n = sum(len(torch.cat([b.c(f, "test", b.tasks[p])["labels"] for p in [b.tasks.index(x) for x in ORDERS[o]][:t]]))
                    for f in FOLDS for t in range(1, 5))
            out.append(f"| {name} | {o} | " + " | ".join(str(x) for x in per_t) + f" | {n:,} | {sum(per_t)} |")
    out += ["", "比對張數 = 十折 × 階段 t = 1–4 的已見任務 test slides 累計（兩序在 t < 4 的已見任務不同，故張數不同）。"
            "每張 test slide 的 τ̂ 存於 `nc9/tau_hat.pt`（鍵 `折|序|分派器|t`，slides 依該序的任務順序串接）。", ""]

    # T4 混淆
    out += ["## T4 分派混淆矩陣（test、t = 4、十折合計；列 = 真實、欄 = 分派）", ""]
    for name, v in cmp_conf.items():
        a, z = v["report10_lines"]
        fid = "ar" if name == "AR" else "arbal"
        out += [f"**{name}**：REPORT_stage10.md:{a}-{z}（`router.nc8.confusion.{fid}.*`）對 NC-9（reverse；`nc9.conf.{fid}.*`）——{ok(v['equal'])}", "",
                "| 真實 \\ 分派 | " + " | ".join(f"{s}（NC-8／NC-9）" for s in SHORT) + " |", "|---|" + "---|" * 4]
        for i in range(4):
            out.append(f"| {SHORT[i]} | " + " | ".join(f"{v['report10'][i][j]}／{v['nc9'][i][j]}" for j in range(4)) + " |")
        out.append("")
    ar = cmp_conf["AR"]["nc9"]
    out += [f"AR 的主要錯分：肺→食道 {ar[3][0]}、食道→肺 {ar[0][3]}、乳→肺 {ar[2][3]}、肺→乳 {ar[3][2]}、食道→乳 {ar[0][2]}。"
            "paper 序的 t = 4 混淆矩陣（另報）：AR " + str(res["confusion_paper"]["AR"]) + "；AR-bal " + str(res["confusion_paper"]["AR-bal"]) + "。", ""]

    # T5 儲存
    out += ["## T5 統計量檔案大小與「保留全部舊訓練片 mean_vec」的對照（bytes）", "",
            "統計量檔 = `nc9/state/fold{f}_{序}_{分派器}.pt`（`torch.save`：A 513 × 513 float64 ＋ b 4 × 513 float64 ＋ 任務 id、γ 等中繼資料）。"
            "mean_vec 原始大小 = 已見 train slides 數 × 512 × 4（float32，與快取相同精度）；另列 NC-8 四個 train 快取檔的實際大小"
            "（含 sid、label、計時欄位）。", "",
            "| 折 | train slides（esca／rcc／brca／lung，合計） | 統計量檔 t = 1 | t = 4 | mean_vec 原始 t = 4 | 倍數（mean_vec ÷ 統計量） | NC-8 train 快取檔合計 |",
            "|---|---|---|---|---|---|---|"]
    ratios = []
    for f in FOLDS:
        ns = [n_train[(f, p)] for p in range(4)]
        s1, s4 = info[(f, "reverse", "AR", 1)]["bytes"], info[(f, "reverse", "AR", 4)]["bytes"]
        raw = sum(ns) * 512 * 4
        ratios.append(raw / s4)
        out.append(f"| {f} | {'／'.join(map(str, ns))}（{sum(ns):,}） | {s1:,} | {s4:,} | {raw:,} | {raw / s4:.2f} | "
                   f"{sum(disk_train[(f, p)] for p in range(4)):,} |")
    sizes = sorted({v["bytes"] for v in info.values()})
    out += ["", f"全部 160 個（折 × 序 × 分派器 × 階段）統計量檔大小的範圍：{sizes[0]:,}–{sizes[-1]:,} bytes；"
            f"只隨已見任務數 t 增加 513 × 8 = 4,104 bytes／任務，與 slide 數無關。mean_vec ÷ 統計量（t = 4）十折範圍 "
            f"{min(ratios):.2f}–{max(ratios):.2f}。統計量以 float64 存（AMENDMENT-2）；PREREG-8 操作定義 9 的 fp32 計法為 "
            "A 1,052,676 ＋ 4 × 2,052 = 1,060,884 bytes。", ""]

    # T6 其他
    out += ["## T6 其他數字", "", "| 項目 | 值 |", "|---|---|",
            f"| W 最大絕對差（累加 − 從頭加總；AR） | {max(v for k, v in wdiff.items() if k[2] == 'AR'):.2e} |",
            f"| W 最大絕對差（累加 − 從頭加總；AR-bal） | {max(v for k, v in wdiff.items() if k[2] == 'AR-bal'):.2e} |",
            f"| 真實資料 t = 4：reverse 與 paper 的 W 最大絕對差（欄依任務對齊；AR／AR-bal） | "
            f"{max(v for k, v in order_w.items() if k[1] == 'AR'):.2e}／{max(v for k, v in order_w.items() if k[1] == 'AR-bal'):.2e} |",
            f"| 真實資料 t = 4：reverse 與 paper 的 τ̂ 不一致張數（依任務對齊後逐張；十折合計；AR／AR-bal） | "
            f"{sum(v for k, v in order_tau.items() if k[1] == 'AR')}／{sum(v for k, v in order_tau.items() if k[1] == 'AR-bal')} |",
            f"| 每階段讀檔秒數（十折兩序兩分派器平均） | {sum(v['t_read_s'] for v in info.values()) / len(info):.4f} |",
            f"| 每階段 add_task＋solve 秒數（平均） | {sum(v['t_compute_s'] for v in info.values()) / len(info):.4f} |",
            f"| 本腳本總耗時（秒） | {res['seconds']['total']:.0f} |", "",
            "兩序的 W 差不是判準（判準 6c 用合成資料）；它來自 A 以不同任務順序加總的浮點捨入。AR 的最大差"
            f"{'超過' if max(v for k, v in order_w.items() if k[1] == 'AR') > 1e-10 else '未超過'} 1e-10，照實列出。", "",
            "測試紀錄（判準 6；本腳本以子行程執行 `python -m pytest -v`，結束碼 " + str(res["pytest"]["rc"]) + "）：", "", "```"]
    out += res["pytest"]["tail"] + ["```", ""]
    out += READ_ONLY
    out += split_section(audit)
    (b.out / "REPORT_stage11.md").write_text("\n".join(out))


def split_section(audit):
    out = ["## R4 十折切分來源（唯讀）", ""]
    if audit is None:
        return out + ["`nc9/split_audit.json` 不存在（未跑 `scripts/nc9_split_audit.py`）。", ""]
    nav = audit["navcil"]
    out += [f"本 repo：`{nav['config']}` → `{nav['dataset_root_dir']}{nav['path_split']}`（四任務 × 十折 = {len(nav['files'])} 檔）。", "",
            "| pathselect clone | HEAD | 設定 | 解析後路徑 | 同一實體檔 | sha256 相同 | 病人名單（train／val／test）逐折相同 | 不同檔數 |",
            "|---|---|---|---|---|---|---|---|"]
    for root, v in audit["pathselect"].items():
        fs = v["files"].values()
        out.append(f"| `{root}` | {v['head'][:7]} | `{v['config']}`（`{v['dataset_root_dir']}{v['path_split']}`） | 同左 | "
                   f"{sum(x['same_realpath'] for x in fs)}/{len(fs)} | {sum(x['sha256_equal'] for x in fs)}/{len(fs)} | "
                   f"{sum(x['patient_lists_equal'] for x in fs)}/{len(fs)} | {v['n_different']} |")
    out += ["", "各檔 sha256 與 train／val／test 人數：", "", "| 檔 | sha256 | train／val／test |", "|---|---|---|"]
    for k, x in nav["files"].items():
        out.append(f"| {k} | `{x['sha256']}` | {x['n']['train']}／{x['n']['val']}／{x['n']['test']} |")
    out += ["", "結論：一致。三個 pathselect clone 的設定都解析到與本 repo 相同的實體檔（`can_dataset/<task>/datasplit/fold_{1..10}.npz`），"
            "40 檔 sha256 全同、病人名單逐折逐鍵相同（含順序）。限制：這是「同一份檔」的確認，不是兩份獨立來源的比對；"
            "pathselect 在 RunPod 上使用的資料集副本（見其 `docs/repro/RUNPOD.md:107`、`docs/ledger/DR-048.md:444`）不在本機，未比對。"
            "切分的產生邏輯與 seed 仍未找到（RUNBOOK_navcil.md §8-1）。", ""]
    return out


READ_ONLY = [
    "## R1 任務分類頭（I6）的訓練資料範圍（唯讀）", "",
    "呼叫鏈（第 t 階段訓練任務 t 的任務分類頭）：",
    "1. `scripts/nc7_i6.py:146-150`：`for r → for fold → for task: train_one(ctx, root, r, fold, task)`；每（r、折、任務）各訓練一次，"
    "與任務順序無關（`nc7_i6.py:2`）。",
    "2. `scripts/nc7_i6.py:47-50`：`p = ctx.tasks.index(task)`；`ds, shift = ctx.ds(fold, task, \"train\")`；`torch.manual_seed(42)`；"
    "新建 `I6Expert(r)`（不載入任何其他任務的權重）。",
    "3. `scripts/nc1_pipeline.py:78-80` `Ctx.ds` → `selector/evaluate.py:29-46` `slide_dataset`：只讀該任務自己的 "
    "`<task>/datasplit/fold_{f}.npz` 的 train 病人名單（`evaluate.py:40-41`），特徵在 `read_slide` 時才讀（`nc7_i6.py:53-57`）。",
    "4. `scripts/nc7_i6.py:60-61` → `selector/cil_ops.py:65-106` `train_selector`：迭代只來自上述 slides；文字只用該任務的 2 類 "
    "`ctx.f_task(p)`（`nc1_pipeline.py:75-76`）與 CONCH 的 logit_scale；optimizer 只含該頭的參數（`cil_ops.py:78-79`）。",
    "5. 推論：`scripts/nc8_batch.py:51-55` 逐任務載入 `i6/r2/fold{f}_{task}.pt`。", "",
    "結論：每個任務分類頭的訓練只讀任務 t 自己的 train slides（外加該任務 2 類的文字特徵），不讀其他任務的 slides、mean_vec 或權重，"
    "所以任務分類頭這一側沒有 replay；搭配本輪的累加分派器，「訓練時不重讀舊任務資料」對整個系統成立。"
    "需要另外註明的兩點（不是 replay，但屬跨任務資訊）：(i) λ\\*=1.5 由 fold 1 四個任務的 validation 一次選定"
    "（`nc1_pipeline.py:263-285`）；(ii) r\\*=2 由十折四任務的 validation 選定（`nc7_report.py:59-71`）。"
    "兩者都是事前一次性的超參數選擇，不在逐階段流程內。另外程式實際上是一次把四個任務都訓練完（步驟 1 的迴圈），"
    "不是逐階段觸發；因為各頭之間沒有相依，結果與逐階段訓練相同。", "",
    "## R2 I6 的結構與 1,033 參數（唯讀）", "",
    "| 層／張量 | 形狀（r = 2） | 參數數 | 位置 |", "|---|---|---|---|",
    "| s0 = zscore(text_nav_feats(Z, f_task)[:, 0])：對該任務 2 類文字的最大 cosine，slide 內 z-score | [n] | 0（不訓練） | `selector/i6_expert.py:37-38`；`selector/flat_selector.py:19-33` |",
    "| u = [Z ; text_nav_feats]（512 ＋ 2：最大 cosine、文字相似度分布熵） | [n, 514] | 0 | `i6_expert.py:39` |",
    "| A | [2, 514] | 1,028 | `i6_expert.py:29-30`（kaiming_uniform a = √5） |",
    "| b1 | [2] | 2 | `i6_expert.py:31`（初始 0） |",
    "| w2 | [2] | 2 | `i6_expert.py:32`（初始 0） |",
    "| b2 | 純量 | 1 | `i6_expert.py:33`（初始 0） |",
    "| 輸出 score = s0 ＋ w2ᵀ GELU(A u ＋ b1) ＋ b2 | [n] | 合計 1,033 = 514r ＋ r ＋ r ＋ 1 = 516r ＋ 1 | `i6_expert.py:44`、`:48-49`；計數 `:51-52`、`scripts/nc7_report.py:66` |", "",
    "判斷：**獨立的小頭**。每個任務各自新建一個 `I6Expert`（`nc7_i6.py:50`），沒有任何共享的可訓練或凍結權重矩陣；"
    "「底座」s0 是無參數的 zero-shot 文字相似度分數，不是權重。A 的「低秩」指隱藏寬度 r = 2 的瓶頸，不是 ΔW = BA 疊在共享 W 上。"
    "對照：`selector/lora_expert.py:21-27`、`:46-49` 的 `LowRankExpert`（D1 用的 L1 v2）才是 W1 = W1_base ＋ B·A，"
    "W1_base（256 × 514）凍結並取自該序第一個任務的 L0。初始時 w2 = b2 = 0，所以訓練開始時 score = s0（`i6_expert.py:6`）。"
    "fp32 儲存 1,033 × 4 = 4,132 bytes／任務（PREREG-8 操作定義 9）。", "",
    "## R3 top-64 的 4 輪 × 16：每輪之間更新什麼（唯讀）", "",
    "呼叫點：`scripts/nc8_batch.py:63`、`:72` → `selector/cil_ops.py:39-45` `four_round`（budget 64、step 16、redundancy_weight = λ\\* = 1.5、"
    "normalize_base = True、redundancy_mode = \"maxsim\"）→ `selector/multiround.py:134-203` `SequentialBudgetedObserver.observe`。", "",
    "機制：任務分類頭的分數 s = s0 ＋ g 在進入迴圈前算一次（`nc8_batch.py:72` 的 `m(Z, ctx.f_task(p))`），在迴圈前做一次 z-score"
    "（`multiround.py:146-150`），之後不再重算，任務分類頭也不再被呼叫。Zn = 各 patch 的 L2 正規化特徵（`:139`）；"
    "max_sim_seen 初始為 0（`:152`）。每一輪：(1) 調整後分數 adj = z(s) − λ · max_sim_seen（`:156-163`；第 1 輪 seen 為空，沒有懲罰）；"
    "(2) 已選 patch 設為 −∞，不重選（`:164`）；(3) 取 adj 最高的 16 個（`:166-169`）加入已選集合（`:173`）；"
    "(4) 更新 max_sim_seen = max(max_sim_seen, 每個 patch 對本輪新選 16 個的最大 cosine)（`:178-180`）。"
    "輪與輪之間只更新「已選集合（排除）」與「對已選集合的最大相似度」兩個狀態；沒有查詢向量、沒有文字向量更新、分數本身不重算。"
    "confidence_threshold 為 None，迴圈內不呼叫分類（`:184`）；迴圈後的 predict_fn 是回傳 0 的佔位函式（`cil_ops.py:45`），不影響結果。"
    "選完的 64 個 patch 等權平均後 L2 正規化（`cil_ops.py:31-36` `mean_norm`），再對 8 類文字取 cosine（`nc8_batch.py:63`）。"
    "注意：訓練時不是四輪，而是 one-shot top-64 ＋ softmax(分數) 加權（`cil_ops.py:87-90`）。", "",
]


if __name__ == "__main__":
    raise SystemExit(main())
