#!/usr/bin/env python3
"""NC-7 報告：I6 expert（T1、T3）、D3 vs D1（T2）、分派不確定時轉交（T4）。

只讀 i6 評估檔、lora_v2 評估檔與既有快取；AR router 沿用 NC-6（mean_vec、γ = 0.001、float64）。
    NAVCIL_MACHINE=mac python scripts/nc7_report.py
輸出：outputs/navcil/<machine>/REPORT_stage9.md、nc7/metrics.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
import nc5_report as N5                                                   # noqa: E402
import nc6_report as N6                                                   # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402

FOLDS = N2.FOLDS
RS = (1, 2, 3)
GAMMA = 1e-3
QS = (0.0, 0.01, 0.02, 0.05, 0.10)
D1_BYTES = 553_024
fmt = N2.fmt
LABEL = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]


def masked(c8, labels, p) -> float:
    rr = torch.tensor(task_rows(p))
    return (rr[c8[:, rr].argmax(-1)] == labels).float().mean().item()


def main() -> int:
    t0 = time.perf_counter()
    d = N2.D2()
    C5 = N5.Ctx5(d)
    M1 = json.loads((d.out / "nc1" / "metrics.json").read_text())
    root = d.out / "i6"
    ev = {f: torch.load(root / f"eval_fold{f}.pt", map_location="cpu") for f in FOLDS}

    # ── T1：WP 與選 r ────────────────────────────────────────────────────────
    l0_val = []
    for f in FOLDS:
        l0_val.append(sum(masked(d.n2(f, "val", t)["four_cos8_uni"][:, p], d.n2(f, "val", t)["labels"], p)
                          for p, t in enumerate(d.tasks)) / 4)
    l0_val_m = mean_sd(l0_val)[0]
    l0_test = M1["stage1"]["fold_mean"]["e_fourround"]
    wp = {}
    for r in RS:
        val, test, task = [], [], {p: [] for p in range(4)}
        for f in FOLDS:
            v = [masked(ev[f]["val"][t][f"r{r}"]["cos8"][:, 0], ev[f]["val"][t]["labels"], p) for p, t in enumerate(d.tasks)]
            te = [masked(ev[f]["test"][t][f"r{r}"]["cos8"][:, p], ev[f]["test"][t]["labels"], p) for p, t in enumerate(d.tasks)]
            val.append(sum(v) / 4); test.append(sum(te) / 4)
            for p in range(4):
                task[p].append(te[p])
        wp[r] = {"val": val, "test": test, "task": task, "params": 516 * r + 1}
    ok_r = [r for r in RS if mean_sd(wp[r]["val"])[0] >= l0_val_m - 0.01]
    if ok_r:
        r_star, r_ok = min(ok_r), True
    else:
        r_star, r_ok = max(RS, key=lambda r: mean_sd(wp[r]["val"])[0]), False
    i6_pass = r_ok and mean_sd(wp[r_star]["test"])[0] >= 0.9299 and wp[r_star]["params"] <= 2053

    # ── T2：D3 vs D1 ─────────────────────────────────────────────────────────
    arx = N6.ARX(d, C5)
    fn = arx.fn(GAMMA)
    D3, D1, evs1 = {}, {}, {}
    for o in ORDERS:
        four3 = lambda f, o_, seen: torch.cat([ev[f]["test"][d.tasks[p]][f"r{r_star}"]["cos8"] for p in seen])  # noqa: E731
        D3[o] = N5.evaluate(d, C5, fn, "AR*", four3)[o]
        evs1[o] = {f: torch.load(d.out / "lora_v2" / "r2" / o / f"fold{f}_eval.pt", map_location="cpu") for f in FOLDS}
        four1 = lambda f, o_, seen, e=evs1[o]: torch.cat([e[f]["tasks"][d.tasks[p]]["l1_cos8"] for p in seen])  # noqa: E731
        D1[o] = N5.evaluate(d, C5, fn, "AR*", four1)[o]
    cmp = {}
    for o in ORDERS:
        a = [x["acc"] for x in D3[o]["cil"]]; b = [x["acc"] for x in D1[o]["cil"]]
        diffs = [x - y for x, y in zip(a, b)]
        cmp[o] = {"per_fold": diffs, "mean": mean_sd(diffs)[0], "wins": sum(x > 0 for x in diffs),
                  "secondary": mean_sd(a)[0] >= mean_sd(b)[0] - 0.005}

    # ── T3：狀態與行為 ─────────────────────────────────────────────────────
    beh = {}
    for r in RS:
        beh[r] = {k: {p: [] for p in range(4)} for k in ("ratio", "jz", "jl0")}
        for f in FOLDS:
            for p, t in enumerate(d.tasks):
                e = ev[f]["test"][t][f"r{r}"]
                beh[r]["ratio"][p].append(e["ratio"].mean().item())
                beh[r]["jz"][p].append(e["jaccard_zs"].mean().item())
                beh[r]["jl0"][p].append(e["jaccard_l0"].mean().item())

    # ── T4：轉交 ─────────────────────────────────────────────────────────────
    seen4 = list(range(4))
    defer = {o: {q: [] for q in QS} for o in ORDERS}
    comp = {q: {"task": [0] * 4, "class": [0] * 8, "n": 0} for q in QS}
    mis_total, mis_deferred5 = 0, 0
    for o in ORDERS:
        for f in FOLDS:
            g = N2.gather_r(d, f, "test", seen4)
            sc = fn(f, o, "test", g, seen4)
            top = sc.topk(2, dim=-1)
            margin = top.values[:, 0] - top.values[:, 1]
            th = torch.tensor(seen4)[sc.argmax(-1)]
            four = torch.cat([evs1[o][f]["tasks"][t]["l1_cos8"] for t in d.tasks])
            pred, _ = N2.hard(four, th, g["task"])
            ok = pred == g["labels"]
            N = len(margin)
            for q in QS:
                k = math.ceil(q * N)
                keep = torch.ones(N, dtype=torch.bool)
                if k > 0:
                    idx = torch.topk(margin, k, largest=False).indices
                    keep[idx] = False
                accs = [ok[keep & (g["task"] == p)].float().mean().item() for p in seen4]
                defer[o][q].append(sum(accs) / 4)
                if o == "reverse":
                    dd = ~keep
                    comp[q]["n"] += int(dd.sum())
                    for p in seen4:
                        comp[q]["task"][p] += int((dd & (g["task"] == p)).sum())
                    for c in range(8):
                        comp[q]["class"][c] += int((dd & (g["labels"] == c)).sum())
                    if q == 0.05:
                        mis = th != g["task"]
                        mis_total += int(mis.sum()); mis_deferred5 += int((mis & dd).sum())

    strip = lambda R: {o: {"acc": [x["acc"] for x in R[o]["cil"]], "masked": [x["masked"] for x in R[o]["cil"]],  # noqa: E731
                           "forgetting": [x["forgetting"] for x in R[o]["cil"]], "bwt": [x["bwt"] for x in R[o]["cil"]],
                           "acc_t": [x["acc_t"] for x in R[o]["cil"]], "masked_t": [x["masked_t"] for x in R[o]["cil"]]}
                       for o in ORDERS}
    M = {"l0_val": l0_val, "l0_test": l0_test, "wp": {str(r): v for r, v in wp.items()}, "r_star": r_star,
         "r_star_meets": r_ok, "i6_pass": i6_pass, "D3": strip(D3), "D1": strip(D1), "cmp": cmp,
         "behavior": {str(r): v for r, v in beh.items()},
         "defer": {o: {str(q): v for q, v in x.items()} for o, x in defer.items()},
         "composition": {str(q): v for q, v in comp.items()},
         "misrouted_deferred_at_5pct": {"deferred": mis_deferred5, "misrouted": mis_total}}
    tp = d.out / "nc7" / "timing_i6.json"
    M["timing"] = json.loads(tp.read_text()) if tp.exists() else {}
    (d.out / "nc7").mkdir(exist_ok=True)
    (d.out / "nc7" / "metrics.json").write_text(json.dumps(M, indent=1, default=str))
    write(d, M, l0_val_m, time.perf_counter() - t0)
    print(f"→ {d.out / 'REPORT_stage9.md'}")
    return 0


def write(d, M, l0_val_m, t_rep):
    ok = lambda b: "**通過**" if b else "**未通過**"                     # noqa: E731
    wp, rs_ = M["wp"], M["r_star"]
    w = wp[str(rs_)]
    out = ["# REPORT — NC-7：I6 expert（zero-shot 分數為底座）與分派不確定時轉交（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；所有數字來自同一台、同一批。"
           "判準見 `PREREG-7.md`（commit 3d70d29）。router = AR（mean_vec、γ = 0.001、float64，與 NC-6 相同）。", "",
           "## PREREG-7 判準落點", "", "| 判準 | 數值 | 所在表格 | 結果 |", "|---|---|---|---|",
           f"| 選 r：validation WP ≥ L0 validation WP − 0.01（{l0_val_m:.4f} − 0.01 = {l0_val_m - 0.01:.4f}）的最小 r | "
           f"r\\* = {rs_}（validation {mean_sd(w['val'])[0]:.4f}）{'' if M['r_star_meets'] else '，**未符合**'} | T1 | "
           f"{'符合' if M['r_star_meets'] else '未符合'} |",
           f"| I6-pass：test WP ≥ 0.9299 | {mean_sd(w['test'])[0]:.4f} | T1 | {ok(mean_sd(w['test'])[0] >= 0.9299)} |",
           f"| I6-pass：每任務參數 ≤ 2,053 | {w['params']:,} | T1 | {ok(w['params'] <= 2053)} |",
           f"| **I6-pass 整體** | | T1 | {ok(M['i6_pass'])} |"]
    for o in ORDERS:
        a = mean_sd(M["D3"][o]["acc"])[0]; b = mean_sd(M["D1"][o]["acc"])[0]
        out.append(f"| D3-次要（{o}）：D3 test ACC ≥ D1 − 0.005 | {a:.4f} vs {b:.4f} − 0.005 = {b - 0.005:.4f} | T2 | "
                   f"{ok(M['cmp'][o]['secondary'])} |")
    out += ["| B 轉交分析 | 不設門檻，必報 | T4 | 已報 |", ""]

    # T1
    out += ["## T1 I6 各 r 的 WP 與參數（四輪、Masked ACC、十折）", "",
            f"L0：validation {fmt(M['l0_val'])}、test {fmt(M['l0_test'])}（每任務 132,097 參數）。", "",
            "| r | 每任務參數 | validation WP | test WP | " + " | ".join(f"test WP {t.split('_')[1]}" for t in d.tasks) + " |",
            "|---|---|---|---|" + "---|" * 4]
    for r in RS:
        x = wp[str(r)]
        out.append(f"| {r}{' ★' if r == rs_ else ''} | {x['params']:,} | {fmt(x['val'])} | {fmt(x['test'])} | "
                   + " | ".join(fmt(x["task"][str(p)] if str(p) in x["task"] else x["task"][p]) for p in range(4)) + " |")
    out += ["", "★ = r\\*。I6 每折每任務訓練一次，與順序無關（兩序共用）。", ""]

    # T2
    out += ["## T2 D3 = AR ＋ I6(r\\*) 與 D1 = AR ＋ L1(r=2) v2（Hard）", "",
            "| 系統 | 序 | ACC | Masked ACC | Forgetting | BWT |", "|---|---|---|---|---|---|"]
    for name in ("D3", "D1"):
        for o in ORDERS:
            r = M[name][o]
            out.append(f"| {name} | {o} | {fmt(r['acc'])} | {fmt(r['masked'])} | {fmt(r['forgetting'])} | {fmt(r['bwt'])} |")
    out += ["", "| 系統 | 序 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|"]
    for name in ("D3", "D1"):
        for o in ORDERS:
            r = M[name][o]
            out.append(f"| {name} | {o} | " + " | ".join(fmt([x[t] for x in r["acc_t"]]) for t in range(4)) + " |")
    out += ["", "逐折差（D3 − D1，test t = 4 ACC）：", "", "| 序 | " + " | ".join(str(f) for f in FOLDS) + " | 平均 | 贏 |",
            "|---|" + "---|" * 12]
    for o in ORDERS:
        c = M["cmp"][o]
        out.append(f"| {o} | " + " | ".join(f"{x:+.4f}" for x in c["per_fold"]) + f" | {c['mean']:+.4f} | {c['wins']}/10 |")

    # T3
    b = M["behavior"]
    out += ["", "## T3 狀態與行為（test，該任務 slides，逐張平均後十折平均）", "",
            "| r | 指標 | " + " | ".join(d.tasks) + " |", "|---|---|" + "---|" * 4]
    for r in RS:
        x = b[str(r)]
        g = lambda k, p: x[k][str(p)] if str(p) in x[k] else x[k][p]         # noqa: E731
        out.append(f"| {r} | std(g)／std(s0) | " + " | ".join(fmt(g("ratio", p)) for p in range(4)) + " |")
        out.append(f"| {r} | Jaccard vs zero-shot top-64 | " + " | ".join(fmt(g("jz", p)) for p in range(4)) + " |")
        out.append(f"| {r} | Jaccard vs L0 四輪 | " + " | ".join(fmt(g("jl0", p)) for p in range(4)) + " |")
    out += ["", "四個任務 expert 合計 bytes（fp32）：", "", "| 系統 | bytes |", "|---|---|"]
    for r in RS:
        out.append(f"| I6 r = {r}（4 × {516 * r + 1:,} 參數） | {4 * (516 * r + 1) * 4:,} |")
    out += [f"| D1（L1 r=2 v2：底座 528,388 ＋ 3 × 8,212） | {D1_BYTES:,} |", ""]

    # T4
    out += ["## T4 分派不確定時轉交（router = AR γ = 0.001、expert = D1；test t = 4）", "",
            "| 轉交比例 | ACC reverse | ACC paper | 轉交張數（十折合計） |", "|---|---|---|---|"]
    for q in QS:
        out.append(f"| {q:.0%} | {fmt(M['defer']['reverse'][str(q)])} | {fmt(M['defer']['paper'][str(q)])} | "
                   f"{M['composition'][str(q)]['n']} |")
    out += ["", "被轉交片子的組成（十折合計；AR 分派在 t = 4 兩序相同）：", "",
            "| 轉交比例 | " + " | ".join(d.tasks) + " | " + " | ".join(LABEL) + " |", "|---|" + "---|" * 12]
    for q in QS[1:]:
        c = M["composition"][str(q)]
        out.append(f"| {q:.0%} | " + " | ".join(str(x) for x in c["task"]) + " | " + " | ".join(str(x) for x in c["class"]) + " |")
    m5 = M["misrouted_deferred_at_5pct"]
    out += ["", f"轉交 5% 時，原本分派錯的 {m5['misrouted']} 張中有 {m5['deferred']} 張被轉交"
            f"（{m5['deferred'] / max(m5['misrouted'], 1):.1%}，十折合計）。", ""]

    tm = M["timing"]
    tr = sum(v for k, v in tm.items() if k.startswith("train/"))
    evs = sum(v for k, v in tm.items() if k.startswith("eval/"))
    out += ["## T5 實際耗時（秒，wall clock；執行緒 8）", "", "| 項目 | 秒 |", "|---|---|",
            f"| I6 訓練（3 個 r × 十折 × 4 任務 × 5 epochs） | {tr:,.0f} |",
            f"| I6 評估（每折 val／test 各讀一次，三個 r 同時） | {evs:,.0f} |",
            f"| 報告 | {t_rep:,.0f} |", ""]
    (d.out / "REPORT_stage9.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
