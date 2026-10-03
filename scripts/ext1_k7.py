#!/usr/bin/env python3
"""EXT-1 K7（PREREG-21 細則 21）：C 的重算值與既有 JSON（逐折）及 FINAL_RESULTS.md 印出值（十折 mean ± sd）逐格比對。

只讀；不覆寫任何既有報告。輸出 outputs/navcil/<machine>/ext1/k7.json。

    NAVCIL_MACHINE=mac python scripts/ext1_k7.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from selector.text_encoder import load_config                             # noqa: E402

TOL = 1e-4
ORDERS = ("reverse", "paper")
FOLDS = list(range(1, 11))
BASE = REPO_ROOT / "outputs" / "navcil" / load_config()["machine"]
ROW1, ROW2, ROW6 = "1 zero-shot 8 類 top-64", "2 LIN8（γ = 0.01）", "6 D3：AR＋I6(r=2)（主系統）"
TASKS = ["esca", "rcc", "brca", "lung"]


def J(rel: str) -> dict:
    return json.loads((BASE / rel).read_text())


def mine(row: str, o: str, f: int) -> dict:
    return json.loads((BASE / "ext1" / "c" / row / f"{o}_fold{f}.json").read_text())


class K7:
    def __init__(self):
        self.cells, self.bad, self.groups = 0, [], {}

    def cmp(self, group, row, o, fold, metric, new, old):
        """new／old 可為數或等長 list。"""
        pairs = list(zip(new, old)) if isinstance(new, list) else [(new, old)]
        g = self.groups.setdefault(group, {"cells": 0, "max_abs": 0.0, "bad": 0})
        for i, (a, b) in enumerate(pairs):
            if a is None or b is None:
                continue
            d = abs(a - b)
            self.cells += 1
            g["cells"] += 1
            g["max_abs"] = max(g["max_abs"], d)
            if d > TOL:
                g["bad"] += 1
                self.bad.append({"source": group, "row": row, "order": o, "fold": fold,
                                 "metric": metric + (f"[t={i + 1}]" if isinstance(new, list) else ""), "old": b, "new": a, "diff": a - b})


def ms(xs):
    return statistics.fmean(xs), statistics.stdev(xs)


def main() -> None:
    k = K7()
    nc8, s2, s6, rf = J("nc8/per_fold.json")["rows"], J("moe1/s2.json"), J("moe1/s6.json"), J("moe1/s7_rf.json")
    e7, m3, f5, m4, r0 = J("moe2/e7.json"), J("moe3/f2.json"), J("moe3/f5.json"), J("moe4/f2.json"), J("moe0/results.json")
    M = {(row, o, f): mine(row, o, f) for row in
         ["ZS8", "LIN8", "MAIN", "FINAL", "NOHEAD", "ONE64", "K32", "K64", "K128", "K256", "M1", "M3", "G1", "G2", "ANC",
          "ANC_v0", "ANC_v42", "CONCAT", "RF", "M2_g0"] + [f"M2_head_{t}" for t in TASKS]
         + [f"{n}_s{s}" for n in ("FINAL", "MAIN", "ONE64") for s in (43, 44, 45, 46)]
         for o in ORDERS for f in FOLDS}

    def full(group, row, o, f, ref, mk="mk1_t", ref_mk="masked_t", acc="acc_t"):
        m = M[(row, o, f)]
        k.cmp(group, row, o, f, "CIL_ACC", m["acc_t"], ref[acc])
        if ref_mk in ref:
            k.cmp(group, row, o, f, {"mk1_t": "MaskedACC_table1", "wp_t": "WP"}[mk], m[mk], ref[ref_mk])
        k.cmp(group, row, o, f, "Forgetting", m["forgetting"], ref["forgetting"])
        k.cmp(group, row, o, f, "BWT", m["bwt"], ref["bwt"])

    for o in ORDERS:
        for i, f in enumerate(FOLDS):
            full("nc8/per_fold.json", "ZS8", o, f, nc8[ROW1][o][i])
            full("nc8/per_fold.json", "LIN8", o, f, nc8[ROW2][o][i])
            full("nc8/per_fold.json", "MAIN", o, f, nc8[ROW6][o][i])
            full("moe1/s2.json", "MAIN", o, f, s2["orders"][o]["main"][i])
            full("moe1/s2.json", "M3", o, f, s2["orders"][o]["M3"][i], mk="wp_t")
            full("moe2/e7.json", "CONCAT", o, f, e7["orders"][o]["LRG"][i], mk="wp_t")
            for row, name in (("NOHEAD", "S-R0f"), ("K64", "S-R0"), ("ANC", "S-A0"), ("ANC_v0", "S-A0f"), ("ANC_v42", "S-Ah")):
                full("moe3/f2.json", row, o, f, m3["rows"][o][name][i], mk="wp_t", ref_mk="wp_t")
            for K in (32, 64, 128, 256):
                ref, m = f5["orders"][o]["S-R0"][f"u{K}"][i], M[(f"K{K}", o, f)]
                k.cmp("moe3/f5.json", f"K{K}", o, f, "CIL_ACC[t=4]", m["acc_t"][3], ref["cil"])
                k.cmp("moe3/f5.json", f"K{K}", o, f, "WP[t=4]", m["wp_t"][3], ref["wp"])
            for g in ("G1", "G2"):
                ref, m = s6["orders"][o]["systems"][g][i], M[(g, o, f)]
                k.cmp("moe1/s6.json", g, o, f, "CIL_ACC[t=4]", m["acc_t"][3], ref["cil"])
                k.cmp("moe1/s6.json", g, o, f, "WP[t=4]", m["wp_t"][3], ref["wp"])
            ref, m = rf["orders"][o]["M3rf"][i], M[("RF", o, f)]
            k.cmp("moe1/s7_rf.json", "RF", o, f, "CIL_ACC[t=4]", m["acc_t"][3], ref["cil"])
            k.cmp("moe1/s7_rf.json", "RF", o, f, "WP[t=4]", m["wp_t"][3], ref["wp"])
            # M1：moe0 只存 (soft gating − 主系統) 的逐折差
            d = r0["B5"][o]["paired"]["per_fold"][i]
            k.cmp("moe0/results.json B5", "M1", o, f, "CIL_ACC[t=4] − MAIN", M[("M1", o, f)]["acc_t"][3] - M[("MAIN", o, f)]["acc_t"][3], d)
            # moe4 的逐折值（t = 4 的 CIL、WP、Ā）
            for name, sysname in (("FINAL", "P-4"), ("MAIN", "P-main"), ("ONE64", "P-F")):
                for s in (42, 43, 44, 45, 46):
                    row = name if s == 42 else f"{name}_s{s}"
                    pf, m = m4["per_fold"][f"{sysname}({s})|{o}"], M[(row, o, f)]
                    k.cmp("moe4/f2.json per_fold", row, o, f, "CIL_ACC[t=4]", m["acc_t"][3], pf["cil4"][i])
                    k.cmp("moe4/f2.json per_fold", row, o, f, "WP[t=4]", m["wp_t"][3], pf["wp4"][i])
                    if "abar" in pf:
                        k.cmp("moe4/f2.json per_fold", row, o, f, "Abar", sum(m["acc_t"]) / 4, pf["abar"][i])

    # 十折 mean ± sd：moe4 h1（逐階段）、h1_avg（五 seed）
    def agg(rows, key, t=None):
        xs = [(r[key][t] if t is not None else r[key]) for r in rows]
        return ms(xs)

    for o in ORDERS:
        for name, sysname in (("FINAL", "P-4"), ("MAIN", "P-main"), ("ONE64", "P-F")):
            per_seed = {s: [M[(name if s == 42 else f"{name}_s{s}", o, f)] for f in FOLDS] for s in (42, 43, 44, 45, 46)}
            for s, rows in per_seed.items():
                h = m4["h1"][f"{sysname}({s})"][o]
                for t in range(4):
                    k.cmp("moe4/f2.json h1", f"{name}(s{s})", o, "十折 mean／sd", f"CIL_ACC[t={t + 1}]", list(agg(rows, "acc_t", t)), h["acc_t"][t])
                    k.cmp("moe4/f2.json h1", f"{name}(s{s})", o, "十折 mean／sd", f"WP[t={t + 1}]", list(agg(rows, "wp_t", t)), h["wp_t"][t])
                k.cmp("moe4/f2.json h1", f"{name}(s{s})", o, "十折 mean／sd", "Forgetting", list(agg(rows, "forgetting")), h["forgetting"])
                k.cmp("moe4/f2.json h1", f"{name}(s{s})", o, "十折 mean／sd", "BWT", list(agg(rows, "bwt")), h["bwt"])
            ha = m4["h1_avg"][sysname][o]
            for key, lab in (("acc_t", "CIL_ACC"), ("wp_t", "WP")):
                for t in range(4):
                    xs = [statistics.fmean(per_seed[s][i][key][t] for s in per_seed) for i in range(10)]
                    k.cmp("moe4/f2.json h1_avg", f"{name}_5seed", o, "十折 mean／sd", f"{lab}[t={t + 1}]", list(ms(xs)), ha[key][t])
            for key, lab in (("forgetting", "Forgetting"), ("bwt", "BWT")):
                xs = [statistics.fmean(per_seed[s][i][key] for s in per_seed) for i in range(10)]
                k.cmp("moe4/f2.json h1_avg", f"{name}_5seed", o, "十折 mean／sd", lab, list(ms(xs)), ha[key])

    # M2：B2 的 4 × 4（列 = head，欄 = 真實任務）＋ g = 0 列；十折平均
    for hi, name in enumerate(TASKS + ["g0"]):
        row = "M2_g0" if name == "g0" else f"M2_head_{name}"
        got = [statistics.fmean(M[(row, "reverse", f)]["wp_task4"][q] for f in FOLDS) for q in range(4)]
        k.cmp("moe0/results.json B2", row, "reverse", "十折平均", "每任務 WP", got, r0["B2"]["matrix"][hi])

    # FINAL_RESULTS.md 印出的四位小數（A 表與 B 表；mean, sd）。None = 報告該格沒有數字。
    P = {  # (row, order, metric): (mean, sd)
        ("ZS8", "reverse", "CIL"): (0.8270, 0.0311), ("ZS8", "reverse", "MK1"): (0.9044, 0.0156),
        ("ZS8", "reverse", "F"): (0.0700, 0.0219), ("ZS8", "paper", "F"): (0.0247, 0.0112),
        ("ZS8", "reverse", "BWT"): (-0.0696, 0.0225), ("ZS8", "paper", "BWT"): (-0.0244, 0.0107),
        ("LIN8", "reverse", "CIL"): (0.8566, 0.0294), ("LIN8", "reverse", "MK1"): (0.9243, 0.0200),
        ("LIN8", "reverse", "F"): (0.0907, 0.0321), ("LIN8", "paper", "F"): (0.0101, 0.0086),
        ("LIN8", "reverse", "BWT"): (-0.0780, 0.0315), ("LIN8", "paper", "BWT"): (-0.0062, 0.0111),
        ("MAIN", "reverse", "CIL"): (0.9128, 0.0258), ("MAIN", "reverse", "WP"): (0.9340, 0.0202),
        ("MAIN", "reverse", "MK1"): (0.9312, 0.0215), ("MAIN", "reverse", "F"): (0.0224, 0.0172),
        ("MAIN", "paper", "F"): (0.0041, 0.0055), ("MAIN", "reverse", "BWT"): (-0.0220, 0.0170),
        ("MAIN", "paper", "BWT"): (-0.0034, 0.0049),
        ("FINAL", "reverse", "CIL"): (0.9252, 0.0210), ("FINAL", "reverse", "WP"): (0.9457, 0.0162),
        ("FINAL", "paper", "CIL"): (0.9252, 0.0210), ("FINAL", "reverse", "F"): (0.0179, 0.0209),
        ("FINAL", "paper", "F"): (0.0081, 0.0058), ("FINAL", "reverse", "BWT"): (-0.0049, 0.0281),
        ("FINAL", "paper", "BWT"): (-0.0041, 0.0057),
        ("FINAL_5seed", "reverse", "CIL"): (0.9252, 0.0194), ("FINAL_5seed", "reverse", "WP"): (0.9466, 0.0140),
        ("FINAL_5seed", "reverse", "F"): (0.0207, 0.0186), ("FINAL_5seed", "paper", "F"): (0.0074, 0.0038),
        ("FINAL_5seed", "reverse", "BWT"): (-0.0097, 0.0234), ("FINAL_5seed", "paper", "BWT"): (-0.0042, 0.0050),
        ("MAIN_5seed", "reverse", "CIL"): (0.9130, 0.0223), ("MAIN_5seed", "reverse", "WP"): (0.9338, 0.0170),
        ("MAIN_5seed", "reverse", "F"): (0.0222, 0.0172), ("MAIN_5seed", "paper", "F"): (0.0038, 0.0049),
        ("MAIN_5seed", "reverse", "BWT"): (-0.0218, 0.0169), ("MAIN_5seed", "paper", "BWT"): (-0.0031, 0.0045),
        ("ONE64_5seed", "reverse", "CIL"): (0.9238, 0.0196), ("ONE64_5seed", "reverse", "WP"): (0.9451, 0.0145),
        ("NOHEAD", "reverse", "CIL"): (0.9214, 0.0217), ("NOHEAD", "reverse", "WP"): (0.9435, 0.0150),
        ("K32", "reverse", "CIL"): (0.9201, 0.0221), ("K32", "reverse", "WP"): (0.9423, 0.0146),
        ("K64", "reverse", "CIL"): (0.9199, 0.0222), ("K64", "reverse", "WP"): (0.9420, 0.0153),
        ("K128", "reverse", "CIL"): (0.9206, 0.0219), ("K128", "reverse", "WP"): (0.9427, 0.0124),
        ("K256", "reverse", "CIL"): (0.9209, 0.0219), ("K256", "reverse", "WP"): (0.9427, 0.0116),
        ("M3", "reverse", "CIL"): (0.9233, 0.0223), ("M3", "reverse", "WP"): (0.9439, 0.0146),
        ("M3", "paper", "CIL"): (0.9217, 0.0236), ("M3", "paper", "WP"): (0.9439, 0.0154),
        ("G1", "reverse", "CIL"): (0.9229, 0.0193), ("G1", "reverse", "WP"): (0.9435, 0.0126),
        ("G1", "paper", "CIL"): (0.9252, 0.0209), ("G1", "paper", "WP"): (0.9470, 0.0127),
        ("G2", "reverse", "CIL"): (0.9187, 0.0231), ("G2", "reverse", "WP"): (0.9396, 0.0169),
        ("G2", "paper", "CIL"): (0.9194, 0.0224), ("G2", "paper", "WP"): (0.9402, 0.0163),
        ("M1", "reverse", "CIL"): (0.9131, 0.0258), ("M1", "paper", "CIL"): (0.9131, 0.0258),
        ("ANC", "reverse", "CIL"): (0.9199, 0.0222), ("ANC", "reverse", "WP"): (0.9420, 0.0153),
        ("ANC_v0", "reverse", "CIL"): (0.9214, 0.0217), ("ANC_v0", "reverse", "WP"): (0.9435, 0.0150),
        ("ANC_v42", "reverse", "CIL"): (0.9252, 0.0210), ("ANC_v42", "reverse", "WP"): (0.9457, 0.0162),
        ("CONCAT", "reverse", "CIL"): (0.9200, 0.0201), ("CONCAT", "reverse", "WP"): (0.9418, 0.0147),
        ("RF", "reverse", "CIL"): (0.9249, 0.0240), ("RF", "reverse", "WP"): (0.9457, 0.0172),
        ("RF", "paper", "CIL"): (0.9251, 0.0240), ("RF", "paper", "WP"): (0.9460, 0.0170),
    }

    def tenfold(row, o, metric):
        seeds = (42, 43, 44, 45, 46)
        if row.endswith("_5seed"):
            base = row[:-6]
            rows = [[M[(base if s == 42 else f"{base}_s{s}", o, f)] for s in seeds] for f in FOLDS]
        else:
            rows = [[M[(row, o, f)]] for f in FOLDS]
        get = {"CIL": lambda r: r["acc_t"][3], "WP": lambda r: r["wp_t"][3], "MK1": lambda r: r["mk1_t"][3],
               "F": lambda r: r["forgetting"], "BWT": lambda r: r["bwt"]}[metric]
        return ms([statistics.fmean(get(r) for r in rs) for rs in rows])

    for (row, o, metric), (pm, psd) in P.items():
        m, sd = tenfold(row, o, metric)
        k.cmp("FINAL_RESULTS.md（印出值，四位小數）", row, o, "十折 mean", metric, round(m, 4), pm)
        k.cmp("FINAL_RESULTS.md（印出值，四位小數）", row, o, "十折 sd", metric, round(sd, 4), psd)

    res = {"tol": TOL, "cells": k.cells, "n_bad": len(k.bad), "pass": not k.bad, "groups": k.groups, "bad": k.bad}
    (BASE / "ext1" / "k7.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"K7：比對 {k.cells} 格，差 > {TOL:g} 的 {len(k.bad)} 格")
    for g, v in k.groups.items():
        print(f"  {g}: {v['cells']} 格，最大絕對差 {v['max_abs']:.3e}，超出 {v['bad']}")
    for b in k.bad[:40]:
        print("  ", json.dumps(b, ensure_ascii=False))


if __name__ == "__main__":
    main()
