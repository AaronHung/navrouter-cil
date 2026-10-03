#!/usr/bin/env python3
"""EXT-1 D1、D3、D4 與外部對照的逐折配對（PREREG-21 細則 4、23、25、26）。只讀既有快取與 C 的逐折輸出，不訓練。

D1  FINAL − {主系統, NOHEAD, ZS8, LIN8}：t = 4 CIL ACC 的逐折配對（符號檢定）＋ slide 層級 bootstrap 95% CI
D3  FINAL 的 ridge γ ∈ {1e-5 … 1e-1}（AR 的 γ 不變）：t = 4 CIL ACC、Ā（事後描述）
D4  TP（AR、t = 4）的 4 × 4 混淆矩陣、逐任務 WP、FINAL 逐類正確率
EXT 外部對照（outputs/external/<name>/perfold.csv，若存在）與 FINAL、主系統的逐折配對

    NAVCIL_MACHINE=mac python scripts/ext1_d.py
輸出：outputs/navcil/<machine>/ext1/d.json
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import torch
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_c as CC                                                       # noqa: E402
from ext1_c import D64, M, N5, T, X, C                                    # noqa: E402

ORDERS, FOLDS, SEEDS = CC.ORDERS, list(range(1, 11)), CC.SEEDS
GAMMAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
LABEL = C.LABEL
N_BOOT, BOOT_SEED, TIE = 1000, 0, 1e-12


def ms(xs):
    xs = [float(x) for x in xs]
    return [statistics.fmean(xs), statistics.stdev(xs)]


def sign_pair(a, b) -> dict:
    """逐折配對 a − b（細則 4）：平均差、贏／輸／平手折數、exact binomial 雙尾 p（平手不計）。"""
    d = [x - y for x, y in zip(a, b)]
    w, l = sum(v > TIE for v in d), sum(v < -TIE for v in d)
    p = float(binomtest(w, w + l, 0.5, alternative="two-sided").pvalue) if w + l else None
    return {"a": ms(a), "b": ms(b), "mean_diff": statistics.fmean(d), "wins": w, "losses": l, "ties": len(d) - w - l,
            "p_sign": p, "per_fold": d}


def row(cx, name, o, f):
    return json.loads(cx.path(name, o, f).read_text())


def series(cx, name, o, fn, five=False):
    """逐折的值；five = 每折先對 seed 42–46 平均。"""
    out = []
    for f in FOLDS:
        rs = [row(cx, name if s == 42 else f"{name}_s{s}", o, f) for s in (SEEDS if five else (42,))]
        out.append(statistics.fmean(fn(r) for r in rs))
    return out


METRICS = {"ACC": lambda r: r["acc_t"][3], "MaskedACC_table1": lambda r: r["mk1_t"][3], "MaskedACC_oracle": lambda r: r["wp_t"][3],
           "Forgetting": lambda r: r["forgetting"], "BWT": lambda r: r["bwt"], "Abar": lambda r: sum(r["acc_t"]) / 4}


def bootstrap(cx, tasks_per_fold, a_ok, b_ok) -> list[float]:
    """slide 層級、(折, 任務) 分層、配對重抽；統計量 = 十折平均的四任務等權 CIL ACC 差。"""
    g = torch.Generator().manual_seed(BOOT_SEED)
    stats = torch.zeros(N_BOOT, dtype=D64)
    for task, xa, xb in zip(tasks_per_fold, a_ok, b_ok):
        d = (xa - xb).to(D64)
        for q in range(4):
            dq = d[task == q]
            idx = torch.randint(len(dq), (N_BOOT, len(dq)), generator=g)
            stats += dq[idx].mean(1) / (4 * len(tasks_per_fold))
    qs = torch.quantile(stats, torch.tensor([0.025, 0.975], dtype=D64))
    return [float(qs[0]), float(qs[1])]


def main() -> None:
    sys.argv = sys.argv[:1]
    cx = CC.Ctx()
    out = {}

    # ── D1 ──
    tasks_pf = [cx.st.split("test", f)["task"] for f in FOLDS]

    def ok(name, o, five=False):
        res = []
        for f in FOLDS:
            xs = [torch.tensor(row(cx, name if s == 42 else f"{name}_s{s}", o, f)["ok4"], dtype=D64) for s in (SEEDS if five else (42,))]
            res.append(torch.stack(xs).mean(0))
        return res

    d1 = {}
    for o in ORDERS:
        d1[o] = {}
        for label, five in (("seed42", False), ("5seed", True)):
            fin, fin_ok = series(cx, "FINAL", o, METRICS["ACC"], five), ok("FINAL", o, five)
            for other, seedless in (("MAIN", False), ("NOHEAD", True), ("ZS8", True), ("LIN8", True)):
                b = series(cx, other, o, METRICS["ACC"], five and not seedless)
                r = sign_pair(fin, b)
                r["boot_ci95"] = bootstrap(cx, tasks_pf, fin_ok, ok(other, o, five and not seedless))
                d1[o][f"FINAL({label}) − {other}"] = r
    out["D1"] = d1

    # ── 外部對照的逐折配對 ──
    ext = {}
    ext_root = REPO_ROOT / "outputs" / "external"
    for name in sorted(d.name for d in ext_root.iterdir() if d.is_dir()) if ext_root.is_dir() else []:
        p = ext_root / name / "perfold.csv"
        if not p.exists():
            ext[name] = None
            continue
        rows = list(csv.DictReader(p.open()))
        ext[name] = {}
        for o in ORDERS:
            rr = {int(q["fold"]): {int(x["t"]): x for x in rows if x["order"] == o and x["fold"] == q["fold"]}
                  for q in rows if q["order"] == o}
            if sorted(rr) != FOLDS:
                ext[name][o] = None
                continue
            theirs = {"ACC": [float(rr[f][4]["ACC"]) for f in FOLDS], "MaskedACC": [float(rr[f][4]["MaskedACC"]) for f in FOLDS],
                      "Forgetting": [float(rr[f][4]["Forgetting"]) for f in FOLDS], "BWT": [float(rr[f][4]["BWT"]) for f in FOLDS],
                      "Abar": [statistics.fmean(float(rr[f][t]["ACC"]) for t in range(1, 5)) for f in FOLDS]}
            ext[name][o] = {}
            for sysname in ("FINAL", "MAIN"):
                for label, five in (("seed42", False), ("5seed", True)):
                    cell = {}
                    for mk, key in (("ACC", "ACC"), ("MaskedACC（我方 Table 1 定義）", "MaskedACC_table1"),
                                    ("MaskedACC（我方 oracle 定義）", "MaskedACC_oracle"), ("Forgetting", "Forgetting"),
                                    ("BWT", "BWT"), ("Abar", "Abar")):
                        tk = "MaskedACC" if mk.startswith("MaskedACC") else mk
                        cell[mk] = sign_pair(series(cx, sysname, o, METRICS[key], five), theirs[tk])
                    ext[name][o][f"{sysname}({label})"] = cell
    out["EXT"] = ext

    # ── D3、D4：每折重算一次（γ 網格；TP 混淆矩陣；告訴任務與 CIL 的逐類正確張數）──
    d3 = {o: {str(g): {"cil4": {s: [] for s in SEEDS}, "abar": {s: [] for s in SEEDS}} for g in GAMMAS} for o in ORDERS}
    conf = {o: torch.zeros(4, 4, dtype=torch.long) for o in ORDERS}
    per_class = {k: {s: torch.zeros(8, 2, dtype=torch.long) for s in SEEDS} for k in ("tell", "cil")}
    wp_task = {s: [] for s in SEEDS}
    tp_task = {o: [] for o in ORDERS}
    for f in FOLDS:
        D4 = X.data4(cx.r4, "test", f)
        task, label = D4["task"], D4["label"]
        for o in ORDERS:
            ar = T.ar_stages(cx.r3, f, o, D4)
            th = ar[3].argmax(-1)
            for a in range(4):
                for b in range(4):
                    conf[o][a, b] += int(((task == a) & (th == b)).sum())
            tp_task[o].append(C.task_mean(th == task, task))
            for s in SEEDS:
                kind = f"s{s}"
                for g in GAMMAS:
                    first = CC.ridge_first(CC.ridge_scores(lambda t, g=g: X.acc_of(cx.r4, kind).W(f, o, t, g),
                                                           lambda pv: T.vec_of(D4, kind, pv)))
                    r = CC.cl_eval(first, ar, task, label, o, cx.tasks)
                    d3[o][str(g)]["cil4"][s].append(r["acc_t"][3]); d3[o][str(g)]["abar"][s].append(sum(r["acc_t"]) / 4)
                    if g == CC.G and o == "reverse":
                        tell = 2 * task + (~first(4, task, task)).long()
                        cil = 2 * th + (~first(4, th, th)).long()
                        wp_task[s].append(C.task_mean(tell == label, task))
                        for c in range(8):
                            m = label == c
                            per_class["tell"][s][c] += torch.tensor([int((tell[m] == c).sum()), int(m.sum())])
                            per_class["cil"][s][c] += torch.tensor([int((cil[m] == c).sum()), int(m.sum())])
        cx.r3.drop(); cx.r4.drop()

    out["D3"] = {o: {g: {"seed42_cil4": ms(v["cil4"][42]), "seed42_abar": ms(v["abar"][42]),
                         "5seed_cil4": ms([statistics.fmean(v["cil4"][s][i] for s in SEEDS) for i in range(10)]),
                         "5seed_abar": ms([statistics.fmean(v["abar"][s][i] for s in SEEDS) for i in range(10)])}
                     for g, v in d3[o].items()} for o in ORDERS}

    def cls_table(P):
        return {"seed42": {LABEL[c]: P[42][c].tolist() for c in range(8)},
                "5seed": {LABEL[c]: sum(P[s][c] for s in SEEDS).tolist() for c in range(8)}}

    out["D4"] = {"tp_confusion": {o: conf[o].tolist() for o in ORDERS},
                 "tp_task_acc": {o: [ms([x[q] for x in tp_task[o]]) for q in range(4)] for o in ORDERS},
                 "final_wp_task": {"seed42": [ms([x[q] for x in wp_task[42]]) for q in range(4)],
                                   "5seed": [ms([statistics.fmean(wp_task[s][i][q] for s in SEEDS) for i in range(10)]) for q in range(4)]},
                 "final_per_class_tell": cls_table(per_class["tell"]), "final_per_class_cil": cls_table(per_class["cil"]),
                 "note": "告訴任務與逐類計數取 reverse 序、t = 4（告訴任務判定與序無關，同 MOE-4 D10）；格式 [正確張數, 張數]"}
    (cx.base / "ext1" / "d.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=M._default))
    print("D：完成")


if __name__ == "__main__":
    main()
