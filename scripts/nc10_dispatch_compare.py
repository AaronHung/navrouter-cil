#!/usr/bin/env python3
"""NC-10：分派器同折、同任務分類頭比較（PREREG-10）。

oracle、R0（T-Hard 規則）、text-organ、R3(k=8)、AR（γ = 1e-3，累加統計量）都接同一批 I6(r=2)，Hard；
同一次執行、NC-8 同一批快取、十折、兩序、每階段 t = 1–4 只比較已見任務的 key。
評估沿用 scripts/nc8_report.py（唯讀 import）；AR 由 selector.incremental_ridge 逐階段累加（第 t 階段只讀任務 t）。
    NAVCIL_MACHINE=mac python scripts/nc10_dispatch_compare.py [--cache-dir <NC-8 快取目錄>]
輸出：outputs/navcil/<machine>/nc10/{per_fold,confusion,result}.json、REPORT_stage12.md
一致性檢查任一不符：仍寫出全部輸出，結束碼 1。
"""
from __future__ import annotations

import argparse
import json
import os
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
from nc9_incremental import cells, lineno                                  # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS                                       # noqa: E402
from selector.incremental_ridge import stream                             # noqa: E402
from selector.router import key_text_proto, r_scores, tp_scores           # noqa: E402

FOLDS = N8.FOLDS
DISPATCH = ["oracle", "R0", "text-organ", "R3", "AR"]
LABEL = {"oracle": "oracle（直接給任務）", "R0": "R0（T-Hard 規則）", "text-organ": "text-organ", "R3": "R3(k=8)",
         "AR": "AR（γ = 1e-3）"}
FID = {"oracle": "oracle", "R0": "r0", "text-organ": "text_organ", "R3": "r3", "AR": "ar"}
NATURE = {"oracle": "上限；直接給真實任務；不需儲存",
          "R0": "零訓練；key = 各任務兩個亞型文字 embedding 的平均（與第二站共用的文字，不另存）",
          "text-organ": "零訓練；key = 器官名稱文字 embedding；2,048 bytes／任務（fp32）",
          "R3": "用各任務 train mean_vec 分群（k = 8）；16,384 bytes／任務（fp32）",
          "AR": "用各任務 train mean_vec 累加閉式解；A 513 × 513 共用 ＋ 每任務 513 維（float64：2,105,352 ＋ 4,104 bytes／任務）"}
AR_GAMMA = 1e-3
SHORT = ["esca", "rcc", "brca", "lung"]
IE, IL = 0, 3
fmt = N2.fmt


class B10(N8.B8):
    """NC-8 評估容器 ＋ 文字 key；AR 的 W 只來自累加統計量。"""

    def __init__(self, cache_dir=None):
        super().__init__()
        if cache_dir:
            self.cache = Path(cache_dir)
        self.F = torch.cat([torch.load(REPO_ROOT / "cache" / "text" / f"f_txt_{t}.pt", map_location="cpu")["f_txt"]
                            for t in self.tasks])
        self.r0_keys = {"text_proto": key_text_proto(self.F)}
        self.organ = torch.load(REPO_ROOT / "cache" / "text" / "organ_keys.pt", map_location="cpu")
        assert self.organ["tasks"] == self.tasks
        self.inc, self.reads = {}, []

    def build_ar(self, f, o):
        pos = [self.tasks.index(x) for x in ORDERS[o]]

        def load(p):
            self.reads.append((f, o, p))
            return torch.load(self.cache / f"nc8_fold{f}_train_{self.tasks[p]}.pt", map_location="cpu")["mean_vec"]
        for t, m in enumerate(stream(pos, load, AR_GAMMA), 1):
            self.inc[(f, o, t)] = (m.solve(), list(m.task_ids))

    def W(self, f, o, t, gamma, bal=False, lin8=False):
        assert gamma == AR_GAMMA and not bal and not lin8
        return self.inc[(f, o, t)]


def dispatch(b, name, f, o, g, seen):
    """τ̂（canonical 任務 index）[N]。"""
    if name == "oracle":
        return g["task"].clone()
    st = torch.tensor(seen)
    if name == "R0":
        sc = r_scores("R0", g, seen, b.r0_keys)
    elif name == "text-organ":
        sc = tp_scores(b, f, g, seen, "text-organ")
    else:
        sc = N8.router(b, name, f, o, g, seen)
    return st[sc.argmax(-1)]


def run_one(b, name, f, o, head="I6"):
    log = {}

    def stage(seen):
        g = b.gather(f, seen)
        th = dispatch(b, name, f, o, g, seen)
        pred, mk = N2.hard(g[head], th, g["task"])
        log[len(seen)] = (th, g["task"])
        return pred, mk, g
    r = N5.cil_full(stage, o, b.tasks)
    pos = [b.tasks.index(x) for x in ORDERS[o]]
    mic, mac = [], []
    for t in range(1, 5):
        th, task = log[t]
        ok = th == task
        mic.append(ok.float().mean().item())
        mac.append(sum(ok[task == p].float().mean().item() for p in pos[:t]) / t)
    th, task = log[4]
    ok = th == task
    conf = torch.zeros(4, 4, dtype=torch.long)
    for a, c in zip(task.tolist(), th.tolist()):
        conf[a, c] += 1
    rec = {k: r[k] for k in ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t", "R", "Rm")}
    rec |= {"tp_micro": mic[3], "tp_macro": mac[3], "tp_micro_t": mic, "tp_macro_t": mac,
            "tp_task": [ok[task == p].float().mean().item() for p in range(4)],
            "acc_task": [r["R"][3][pos.index(p)] for p in range(4)]}
    return rec, conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None, help="NC-8 快取目錄（預設：本工作樹的 outputs/navcil/<machine>/cache）")
    args = ap.parse_args()
    t0 = time.perf_counter()
    b = B10(os.path.expanduser(args.cache_dir) if args.cache_dir else None)
    for f in FOLDS:
        for o in ORDERS:
            b.build_ar(f, o)
    per_fold = {n: {o: [] for o in ORDERS} for n in DISPATCH}
    conf = {n: {o: torch.zeros(4, 4, dtype=torch.long) for o in ORDERS} for n in DISPATCH}
    for n in DISPATCH:
        for o in ORDERS:
            for f in FOLDS:
                rec, c = run_one(b, n, f, o)
                per_fold[n][o].append(rec)
                conf[n][o] += c
    r0l0 = [run_one(b, "R0", f, "reverse", head="L0")[0]["acc"] for f in FOLDS]      # 另報：R0 ＋ L0
    t_eval = time.perf_counter() - t0

    # 乘法框架（t = 4）
    mult = {n: {} for n in DISPATCH}
    for n in DISPATCH:
        for o in ORDERS:
            rows = []
            for x, y in zip(per_fold[n][o], per_fold["oracle"][o]):
                rows.append({"macro_x_oracle": x["tp_macro"] * y["acc"], "micro_x_oracle": x["tp_micro"] * y["acc"],
                             "sum_tp_wp": sum(tp * wp for tp, wp in zip(x["tp_task"], y["acc_task"])) / 4,
                             "measured": x["acc"]})
            mult[n][o] = {k: mean_sd([r[k] for r in rows])[0] for k in rows[0]}

    # AR 對 R0 逐折
    paired = {}
    for o in ORDERS:
        for m in ("tp_micro", "tp_macro"):
            d = [x[m] - y[m] for x, y in zip(per_fold["AR"][o], per_fold["R0"][o])]
            paired[(o, m)] = {"diff": d, "wins": sum(v > 0 for v in d), "ties": sum(v == 0 for v in d), "mean": mean_sd(d)[0]}
    prediction = all(paired[(o, "tp_micro")]["wins"] >= 8 for o in ORDERS)

    # 一致性檢查
    old = json.loads((b.out / "nc8" / "per_fold.json").read_text())["rows"]
    r10 = (b.out / "REPORT_stage10.md").read_text().splitlines()
    r6 = (b.out / "REPORT_stage6.md").read_text().splitlines()
    checks = {}
    for n, row, lines in (("AR", 6, {"reverse": 16, "paper": 29}), ("oracle", 8, {"reverse": 18, "paper": 31})):
        for o in ORDERS:
            new = [x["acc"] for x in per_fold[n][o]]
            prev = [x["acc"] for x in old[N8.ROWS[row - 1]][o]]
            cell = cells(r10[lines[o] - 1])[1]
            checks[f"{n}|{o}"] = {"report10_line": lines[o], "report10": cell, "nc10": fmt(new),
                                  "folds_r4_equal": sum(round(a, 4) == round(c, 4) for a, c in zip(new, prev)),
                                  "max_fold_absdiff": max(abs(a - c) for a, c in zip(new, prev)),
                                  "ok": cell == fmt(new) and all(round(a, 4) == round(c, 4) for a, c in zip(new, prev))}
    prev_r3 = [[int(v) for v in cells(r10[93 + i])[1:]] for i in range(4)]
    checks["R3|confusion"] = {"report10_lines": "94-97", "report10": prev_r3, "nc10": conf["R3"]["reverse"].tolist(),
                              "ok": prev_r3 == conf["R3"]["reverse"].tolist()}
    c6 = cells(r6[93])
    for o in ORDERS:
        mi = f"{mean_sd([x['tp_micro'] for x in per_fold['R0'][o]])[0]:.4f}"
        ma = f"{mean_sd([x['tp_macro'] for x in per_fold['R0'][o]])[0]:.4f}"
        checks[f"R0|tp|{o}"] = {"report6_line": 94, "report6": {"micro": c6[8], "macro": c6[7]}, "nc10": {"micro": mi, "macro": ma},
                                "ok": (mi, ma) == (c6[8], c6[7])}
    ok_all = all(v["ok"] for v in checks.values())

    # 另報比對
    extra = {}
    for n, ln in (("R0", 94), ("text-organ", 92)):
        c = cells(r6[ln - 1])
        recs = per_fold[n]["reverse"]
        cf = conf[n]["reverse"]
        extra[n] = {"report6_line": ln, "report6": {"tp_task": c[3:7], "macro": c[7], "micro": c[8], "esca_to_lung": c[9],
                                                    "lung_to_esca": c[10], "cil_l0": c[11]},
                    "nc10": {"tp_task": [f"{mean_sd([x['tp_task'][p] for x in recs])[0]:.4f}" for p in range(4)],
                             "macro": f"{mean_sd([x['tp_macro'] for x in recs])[0]:.4f}",
                             "micro": f"{mean_sd([x['tp_micro'] for x in recs])[0]:.4f}",
                             "esca_to_lung": str(int(cf[IE, IL])), "lung_to_esca": str(int(cf[IL, IE]))}}
    extra["R0"]["nc10"]["cil_l0"] = f"{mean_sd(r0l0)[0]:.4f}"
    for v in extra.values():
        v["equal"] = {k: v["report6"][k] == v["nc10"][k] for k in v["nc10"]}

    # 讀取紀錄：AR 第 t 階段只讀任務 t
    reads_ok = all(b.reads[i] == (f, o, [b.tasks.index(x) for x in ORDERS[o]][t])
                   for i, (f, o, t) in enumerate((f, o, t) for f in FOLDS for o in ORDERS for t in range(4)))

    nc10 = b.out / "nc10"
    nc10.mkdir(parents=True, exist_ok=True)
    pf = nc10 / "per_fold.json"
    pf.write_text(json.dumps({"dispatch": per_fold, "folds": FOLDS, "r0_l0_reverse_acc": r0l0}, indent=1, ensure_ascii=False))
    (nc10 / "confusion.json").write_text(json.dumps({n: {o: c.tolist() for o, c in v.items()} for n, v in conf.items()}, indent=1))
    res = {"checks": checks, "checks_ok": ok_all, "extra": extra, "prediction_holds": prediction,
           "paired": {f"{o}|{m}": v for (o, m), v in paired.items()}, "mult": mult, "ar_reads_only_task_t": reads_ok,
           "seconds": {"eval": t_eval, "total": time.perf_counter() - t0}}
    (nc10 / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    write(b, per_fold, conf, mult, paired, prediction, checks, extra, reads_ok, pf, res)
    print(f"→ {b.out / 'REPORT_stage12.md'}")
    for k, v in checks.items():
        print(f"  一致性 {k}: {'符合' if v['ok'] else '不符'}")
    print(f"  方向性預測（AR micro 勝 R0 ≥ 8/10，兩序）：{'成立' if prediction else '不成立'}")
    return 0 if ok_all else 1


def write(b, per_fold, conf, mult, paired, prediction, checks, extra, reads_ok, pf, res):
    yes = lambda v: "符合" if v else "**不符**"                              # noqa: E731
    out = ["# REPORT — NC-10：分派器同折、同任務分類頭比較（Mac CPU，十折兩序）", "",
           "**R0 的任務 key 定義**：每個任務的 key = 該任務兩個亞型文字 embedding 的平均，再 L2 正規化"
           "（`selector/router.py:56-58`；每個亞型文字 embedding 本身是多個同義名 × 22 個模板的平均，`cache/text/f_txt_<task>.pt`）；"
           "分派分數 = 整片 mean_vec 與各已見任務 key 的 cosine，取 argmax（`selector/router.py:87-88`），零訓練。"
           "**labmate 版 T-Hard 的 key 定義尚未確認**；本報告的 R0 是本 repo 的版本，若 labmate 版的 key 不同（例如器官文字或其他模板），"
           "數字可能不同。", "",
           "機器：mac（Apple M1 Pro）、CPU、torch 2.11.0；所有數字來自同一台、同一次執行（`scripts/nc10_dispatch_compare.py`）。"
           "判準與定義見 `PREREG-10.md`（commit d09c966）。所有分派器都接同一批 I6(r=2) 任務分類頭（第二站 Hard：在分派任務的 2 類內取 argmax），"
           "證據與 mean_vec 來自 NC-8 同一批快取；第 t 階段只比較前 t 個已見任務的 key。"
           f"AR 由累加統計量逐階段建立，第 t 階段只讀任務 t 的 train mean_vec（讀取紀錄核對：{'符合' if reads_ok else '不符'}）。"
           "另一個 session 同時在 Mac 上執行 WS3，耗時僅供參考。", ""]

    # 一致性
    out += ["## 一致性檢查（PREREG-10；任一不符即停）", "", "| 項目 | 基準 | 基準值 | NC-10 | 結果 |", "|---|---|---|---|---|"]
    for o in ORDERS:
        c = checks[f"AR|{o}"]
        out.append(f"| AR ＋ I6 CIL，{o} | REPORT_stage10.md:{c['report10_line']}（`nc8.t1.d3.{o}.acc`）；nc8/per_fold.json 第 6 列 | "
                   f"{c['report10']} | {c['nc10']}；逐折四位相同 {c['folds_r4_equal']}/10（最大差 {c['max_fold_absdiff']:.1e}） | {yes(c['ok'])} |")
    for o in ORDERS:
        c = checks[f"oracle|{o}"]
        out.append(f"| oracle ＋ I6 CIL，{o} | REPORT_stage10.md:{c['report10_line']}（`nc8.t1.oracle_i6.{o}.acc`）；第 8 列 | "
                   f"{c['report10']} | {c['nc10']}；逐折四位相同 {c['folds_r4_equal']}/10（最大差 {c['max_fold_absdiff']:.1e}） | {yes(c['ok'])} |")
    c = checks["R3|confusion"]
    out.append(f"| R3 混淆（reverse、t = 4、十折合計） | REPORT_stage10.md:94-97（`router.nc8.confusion.r3.*`） | {c['report10']} | {c['nc10']} | {yes(c['ok'])} |")
    for o in ORDERS:
        c = checks[f"R0|tp|{o}"]
        out.append(f"| R0 分派正確率 micro／macro（t = 4），{o} | REPORT_stage6.md:94（`router.nc4.table.r0_t_hard.tp_micro`／`.tp_macro`） | "
                   f"{c['report6']['micro']}／{c['report6']['macro']} | {c['nc10']['micro']}／{c['nc10']['macro']} | {yes(c['ok'])} |")
    out += ["", "另報（不作為停止條件；REPORT_stage6 的數字來自 NC-1／NC-2 的快取，t = 4 reverse）：", "",
            "| 項目 | REPORT_stage6 | NC-10 | 相同 |", "|---|---|---|---|"]
    for n, v in extra.items():
        for k in v["nc10"]:
            out.append(f"| {LABEL[n]} {k}（REPORT_stage6.md:{v['report6_line']}） | {v['report6'][k]} | {v['nc10'][k]} | {'是' if v['equal'][k] else '否'} |")
    out.append("")

    # 主表
    out += ["## T1 主表（test、t = 4、十折 mean ± sd；混淆為十折合計）", ""]
    for o in ORDERS:
        out += [f"### {o}", "", "| 分派器 | 分派正確率 micro | 分派正確率 macro | CIL 全程（task-inferred） | 肺→食道 | 食道→肺 | 性質 |",
                "|---|---|---|---|---|---|---|"]
        for n in DISPATCH:
            recs, cf = per_fold[n][o], conf[n][o]
            out.append(f"| {LABEL[n]} | {fmt([x['tp_micro'] for x in recs])} | {fmt([x['tp_macro'] for x in recs])} | "
                       f"{fmt([x['acc'] for x in recs])} | {int(cf[IL, IE])} | {int(cf[IE, IL])} | {NATURE[n]} |")
        out.append("")
    out += [f"fact-id：`nc10.t1.<分派器>.<序>.{{tp_micro,tp_macro,acc,lung_to_esca,esca_to_lung}}`，分派器代號 "
            + "、".join(f"{LABEL[n]} = `{FID[n]}`" for n in DISPATCH) + "。每折原始值：`nc10/per_fold.json`；混淆：`nc10/confusion.json`。", ""]
    out += ["Masked ACC、Forgetting、BWT（t = 4，十折 mean ± sd）：", "",
            "| 分派器 | 序 | Masked ACC | Forgetting | BWT |", "|---|---|---|---|---|"]
    for n in DISPATCH:
        for o in ORDERS:
            recs = per_fold[n][o]
            out.append(f"| {LABEL[n]} | {o} | {fmt([x['masked'] for x in recs])} | {fmt([x['forgetting'] for x in recs])} | {fmt([x['bwt'] for x in recs])} |")
    out += ["", "逐階段（十折平均；t = 已見任務數）：", "",
            "| 分派器 | 序 | 分派 micro t=1／2／3／4 | 分派 macro t=1／2／3／4 | CIL t=1／2／3／4 |", "|---|---|---|---|---|"]
    for n in DISPATCH:
        for o in ORDERS:
            recs = per_fold[n][o]
            s = lambda k: "／".join(f"{mean_sd([x[k][t] for x in recs])[0]:.4f}" for t in range(4))  # noqa: E731
            out.append(f"| {LABEL[n]} | {o} | {s('tp_micro_t')} | {s('tp_macro_t')} | {s('acc_t')} |")
    out += ["", "4 × 4 分派混淆（t = 4、十折合計；列 = 真實、欄 = 分派；esca／rcc／brca／lung）：", "",
            "| 分派器 | 序 | esca 列 | rcc 列 | brca 列 | lung 列 |", "|---|---|---|---|---|---|"]
    for n in DISPATCH:
        for o in ORDERS:
            out.append(f"| {LABEL[n]} | {o} | " + " | ".join("／".join(map(str, r)) for r in conf[n][o].tolist()) + " |")
    out.append("")

    # 逐折
    L = pf.read_text().splitlines()
    out += ["## T2 逐折：AR 對 R0 的分派正確率（t = 4）", "",
            f"方向性預測（PREREG-10）：「AR 的分派正確率 micro 高於 R0，十折中至少 8 折，兩序皆然」→ **{'成立' if prediction else '不成立'}**。"
            "行號為 `outputs/navcil/mac/nc10/per_fold.json` 中 `tp_micro` 所在行。", ""]
    for o in ORDERS:
        out += [f"### {o}", "", "| 折 | AR micro | 行 | R0 micro | 行 | 差（AR − R0） | AR macro | R0 macro | 差（macro） |", "|---|---|---|---|---|---|---|---|---|"]
        for k in range(10):
            a, r = per_fold["AR"][o][k], per_fold["R0"][o][k]
            la = lineno(L, ['"AR"', f'"{o}"'], '"tp_micro":', k)
            lr = lineno(L, ['"R0"', f'"{o}"'], '"tp_micro":', k)
            out.append(f"| {k + 1} | {a['tp_micro']:.4f} | {la} | {r['tp_micro']:.4f} | {lr} | {a['tp_micro'] - r['tp_micro']:+.4f} | "
                       f"{a['tp_macro']:.4f} | {r['tp_macro']:.4f} | {a['tp_macro'] - r['tp_macro']:+.4f} |")
        pm, pa = paired[(o, "tp_micro")], paired[(o, "tp_macro")]
        out += ["", f"AR 勝的折數：micro {pm['wins']}/10（平手 {pm['ties']}，平均差 {pm['mean']:+.4f}）；"
                f"macro {pa['wins']}/10（平手 {pa['ties']}，平均差 {pa['mean']:+.4f}）。", ""]

    # 乘法
    out += ["## T3 乘法框架核對（t = 4，十折平均）", "",
            "(a) 分派 macro × oracle CIL；(b) 分派 micro × oracle CIL；(c) Σ_p TP_p × WP_p ／ 4（WP_p = oracle ＋ I6 在任務 p 的正確率）。"
            "Hard 之下分派錯的片必定判錯，所以 (c) 在「同一任務內分派正確與第二站正確互相獨立」時等於實測；實測 − (c) > 0 表示分派錯的片"
            "多半也是第二站本來就會判錯的片。", ""]
    for o in ORDERS:
        out += [f"### {o}", "", "| 分派器 | (a) macro × oracle | (b) micro × oracle | (c) Σ TP_p·WP_p／4 | 實測 CIL | 實測 − (a) | 實測 − (c) |",
                "|---|---|---|---|---|---|---|"]
        for n in DISPATCH:
            m = mult[n][o]
            out.append(f"| {LABEL[n]} | {m['macro_x_oracle']:.4f} | {m['micro_x_oracle']:.4f} | {m['sum_tp_wp']:.4f} | {m['measured']:.4f} | "
                       f"{m['measured'] - m['macro_x_oracle']:+.4f} | {m['measured'] - m['sum_tp_wp']:+.4f} |")
        out.append("")
    out += ["## T4 耗時", "", f"評估 {res['seconds']['eval']:.0f} 秒；全部 {res['seconds']['total']:.0f} 秒（與 WS3 同時執行）。", ""]
    (b.out / "REPORT_stage12.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
