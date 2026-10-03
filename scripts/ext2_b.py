#!/usr/bin/env python3
"""EXT-2 B（PREREG-22 細則 14、18–21）：兩套並排。只讀 A 階段的逐折輸出（ext2/a/）與 outputs/external/<dir>/perfold.csv。

B1  FINAL-A（seed 42、五 seed 平均）− FINAL-B：逐折配對（符號檢定）＋ t = 4 CIL ACC 的 slide 層級 bootstrap 95% CI
    另列 FINAL-B − {主系統, ZS8, LIN8} 與「四輪 − 一次 top-K」（FINAL-B − FINALB_K*）
B2  FINAL-B 對外部對照的逐折配對（外部目錄名在執行時列出，本檔不寫死）
B3  總表用的十折 mean ± sd

    NAVCIL_MACHINE=mac python scripts/ext2_b.py
輸出：outputs/navcil/<machine>/ext2/b.json
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_d as DD                                                       # noqa: E402
import ext2_a as EA                                                       # noqa: E402
from ext1_c import D64, M                                                 # noqa: E402

ORDERS, FOLDS, SEEDS = EA.ORDERS, DD.FOLDS, EA.SEEDS
METRICS = [("ACC", "ACC"), ("WP", "MaskedACC_oracle"), ("MaskedACC_table1", "MaskedACC_table1"), ("Forgetting", "Forgetting"),
           ("BWT", "BWT"), ("Abar", "Abar")]
SEEDED = ("FINAL", "MAIN")


def series(cx, name, o, key, five=False):
    return DD.series(cx, name, o, DD.METRICS[key], five and name in SEEDED)


def ok(cx, name, o, five=False):
    out = []
    for f in FOLDS:
        seeds = SEEDS if five and name in SEEDED else (42,)
        xs = [torch.tensor(DD.row(cx, name if s == 42 else f"{name}_s{s}", o, f)["ok4"], dtype=D64) for s in seeds]
        out.append(torch.stack(xs).mean(0))
    return out


def pair(cx, tasks_pf, a, b, o, five, boot=True) -> dict:
    """a − b：六個指標的逐折配對；t = 4 CIL ACC 另加 bootstrap。"""
    cell = {mk: DD.sign_pair(series(cx, a, o, key, five), series(cx, b, o, key, five)) for mk, key in METRICS}
    if boot:
        cell["ACC"]["boot_ci95"] = DD.bootstrap(cx, tasks_pf, ok(cx, a, o, five), ok(cx, b, o, five))
    return cell


def run(cx: EA.Ctx) -> dict:
    tasks_pf = [cx.st.split("test", f)["task"] for f in FOLDS]
    out = {"B1": {}, "B1_others": {}, "K": {}, "summary": {}}
    for o in ORDERS:
        out["B1"][o] = {f"FINAL-A({lab}) − FINAL-B": pair(cx, tasks_pf, "FINAL", "FINALB", o, five)
                        for lab, five in (("seed42", False), ("5seed", True))}
        oth = {}
        for other, labs in (("MAIN", (("seed42", False), ("5seed", True))), ("ZS8", (("", False),)), ("LIN8", (("", False),))):
            for lab, five in labs:
                oth[f"FINAL-B − {other}" + (f"({lab})" if lab else "")] = pair(cx, tasks_pf, "FINALB", other, o, five)
        out["B1_others"][o] = oth
        out["K"][o] = {f"FINAL-B（四輪）− K{K}（一次取前 {K}）": pair(cx, tasks_pf, "FINALB", f"FINALB_K{K}", o, False) for K in EA.KS}
        summ = {}
        for label, name, five in (("ZS8", "ZS8", False), ("LIN8", "LIN8", False), ("MAIN_5seed", "MAIN", True), ("MAIN", "MAIN", False),
                                  ("FINALB", "FINALB", False), ("FINALA_5seed", "FINAL", True), ("FINALA", "FINAL", False)) + tuple(
                                      (f"FINALB_K{K}", f"FINALB_K{K}", False) for K in EA.KS):
            summ[label] = {mk: DD.ms(series(cx, name, o, key, five)) for mk, key in METRICS}
        out["summary"][o] = summ

    # ── B2：外部對照 ──
    ext, ext_root = {}, REPO_ROOT / "outputs" / "external"
    for name in sorted(d.name for d in ext_root.iterdir() if d.is_dir()) if ext_root.is_dir() else []:
        p = ext_root / name / "perfold.csv"
        if not p.exists():
            ext[name] = None
            continue
        rows = list(csv.DictReader(p.open()))
        ext[name] = {}
        for o in ORDERS:
            rr = {int(q["fold"]): {int(x["t"]): x for x in rows if x["order"] == o and x["fold"] == q["fold"]} for q in rows if q["order"] == o}
            if sorted(rr) != FOLDS:
                ext[name][o] = None
                continue
            theirs = {"ACC": [float(rr[f][4]["ACC"]) for f in FOLDS], "MaskedACC": [float(rr[f][4]["MaskedACC"]) for f in FOLDS],
                      "Forgetting": [float(rr[f][4]["Forgetting"]) for f in FOLDS], "BWT": [float(rr[f][4]["BWT"]) for f in FOLDS],
                      "Abar": [statistics.fmean(float(rr[f][t]["ACC"]) for t in range(1, 5)) for f in FOLDS]}
            if "MaskedACC_optthr" in rows[0]:
                theirs_opt = [float(rr[f][4]["MaskedACC_optthr"]) for f in FOLDS if rr[f][4]["MaskedACC_optthr"] != ""]
            else:
                theirs_opt = []
            cell = {}
            for mk, key in (("ACC", "ACC"), ("MaskedACC（我方 Table 1 定義）", "MaskedACC_table1"), ("MaskedACC（我方 oracle 定義）", "MaskedACC_oracle"),
                            ("Forgetting", "Forgetting"), ("BWT", "BWT"), ("Abar", "Abar")):
                tk = "MaskedACC" if mk.startswith("MaskedACC") else mk
                cell[mk] = DD.sign_pair(series(cx, "FINALB", o, key), theirs[tk])
            ext[name][o] = {"FINAL-B": cell, "theirs": {k: DD.ms(v) for k, v in theirs.items()},
                            "theirs_optthr": DD.ms(theirs_opt) if len(theirs_opt) > 1 else None}
    out["EXT"] = ext
    return out


def main() -> int:
    sys.argv = sys.argv[:1]
    cx = EA.Ctx()
    root = cx.base / "ext2"
    try:
        res = run(cx)
    except Exception:
        (root / "FAILED_b.txt").write_text(f"[{M.now()}] 階段 b 失敗\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3
    res.update({"stage": "b", "finished": M.now(), "n_boot": DD.N_BOOT, "boot_seed": DD.BOOT_SEED})
    (root / "b.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._default))
    for o in ORDERS:
        for k, v in res["B1"][o].items():
            r = v["ACC"]
            print(o, k, f"{r['mean_diff']:+.4f}", f"{r['wins']}／{r['losses']}／{r['ties']}", r["p_sign"], r["boot_ci95"])
    print("B：完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
