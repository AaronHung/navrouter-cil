#!/usr/bin/env python3
"""NC-8 報告：主表（8 列）、D3 逐折比較、圖的資料（CSV）、每折原始值（PREREG-8）。

只讀本批 nc8 快取（scripts/nc8_batch.py）；router 統計量由本批 train mean_vec 重建（AR 類 float64）。
    NAVCIL_MACHINE=mac python scripts/nc8_report.py
輸出：outputs/navcil/<machine>/REPORT_stage10.md、nc8/per_fold.json、nc8/fig/*.csv
"""
from __future__ import annotations

import csv
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
import nc4_report as N4                                                   # noqa: E402
import nc5_report as N5                                                   # noqa: E402
from selector.cil_eval import mean_sd, subset_key                         # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

FOLDS = list(range(1, 11))
D64 = torch.float64
QS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10)
LABEL = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]
ROWS = ["1 zero-shot 8 類 top-64", "2 LIN8（γ = 0.01）", "3 R3(k=8)＋L1(r=2) v2", "4 D2：AR＋L0",
        "5 D1：AR＋L1(r=2) v2", "6 D3：AR＋I6(r=2)（主系統）", "7 AR-bal（γ = 1e-4）＋I6(r=2)", "8 oracle＋I6(r=2)（上限）"]
EXPERT_BYTES = {1: None, 2: None, 3: 553_024, 4: 4 * 528_388, 5: 553_024, 6: 16_528, 7: 16_528, 8: 16_528}
ROUTER_BYTES = {1: "0", 2: "A 1,052,676 ＋ 4 × 4,104", 3: "4 × 16,384 = 65,536", 4: "A 1,052,676 ＋ 4 × 2,052",
                5: "A 1,052,676 ＋ 4 × 2,052", 6: "A 1,052,676 ＋ 4 × 2,052", 7: "A 1,052,676 ＋ 4 × 2,052", 8: "0"}
fmt = N2.fmt


class B8:
    def __init__(self):
        self.cfg = load_config()
        self.tasks = list(self.cfg["tasks"])
        self.out = REPO_ROOT / "outputs" / "navcil" / self.cfg["machine"]
        self.cache = self.out / "cache"
        self._c, self._keys, self._W = {}, {}, {}

    def c(self, f, split, t):
        k = (f, split, t)
        if k not in self._c:
            self._c[k] = torch.load(self.cache / f"nc8_fold{f}_{split}_{t}.pt", map_location="cpu")
        return self._c[k]

    def gather(self, f, seen):
        parts = [self.c(f, "test", self.tasks[p]) for p in seen]
        si = parts[0]["subsets"].index(subset_key(seen))
        cat = lambda k, sl=None: torch.cat([c[k] if sl is None else c[k][sl] for c in parts])   # noqa: E731
        return {"labels": cat("labels"), "mean_vec": cat("mean_vec"),
                "task": torch.cat([torch.full((len(c["labels"]),), p) for p, c in zip(seen, parts)]),
                "zs8": torch.cat([c["zs8_cos8"][:, si] for c in parts]),
                "L0": cat("L0_cos8"), "I6": cat("I6_cos8"),
                "L1": {o: torch.cat([c["L1_cos8"][:, i] for c in parts]) for i, o in enumerate(ORDERS)}}

    def r3_keys(self, f):
        if f not in self._keys:
            self._keys[f] = [N4.kmeans_keys(self.c(f, "train", t)["mean_vec"], 8) for t in self.tasks]
        return self._keys[f]

    def W(self, f, o, t, gamma, bal=False, lin8=False):
        k = (f, o, t, gamma, bal, lin8)
        if k not in self._W:
            pos = [self.tasks.index(x) for x in ORDERS[o]][:t]
            A = torch.zeros(513, 513, dtype=D64)
            cols = []
            for p in pos:
                c = self.c(f, "train", self.tasks[p])
                X = N5.aug(c["mean_vec"])
                w = 1.0 / X.shape[0] if bal else 1.0
                A += w * (X.t() @ X)
                if lin8:
                    y = c["labels"] - 2 * p
                    cols += [X[y == 0].sum(0), X[y == 1].sum(0)]
                else:
                    cols.append(w * X.sum(0))
            self._W[k] = (torch.linalg.solve(A + gamma * torch.eye(513, dtype=D64), torch.stack(cols, 1)), pos)
        return self._W[k]


def router(b, name, f, o, g, seen):
    if name == "R3":
        keys = b.r3_keys(f)
        return torch.stack([(g["mean_vec"] @ keys[p].t()).amax(-1) for p in seen], -1)
    gamma, bal = {"AR": (1e-3, False), "AR-bal": (1e-4, True)}[name]
    W, pos = b.W(f, o, len(seen), gamma, bal)
    return N5.aug(g["mean_vec"]) @ W[:, [pos.index(p) for p in seen]]


def stage_row(b, row, f, o):
    """回傳 stage_fn(seen) → (pred, masked, g)，另附 t = 4 的分派任務（供 TP）。"""
    info = {}

    def stage(seen):
        g = b.gather(f, seen)
        st = torch.tensor(seen)
        rows = torch.tensor([r for p in seen for r in task_rows(p)])
        rt = torch.stack([2 * g["task"], 2 * g["task"] + 1], -1)
        if row == 1:
            c8 = g["zs8"]
            pred = rows[c8[:, rows].argmax(-1)]
            mk = rt.gather(1, c8.gather(1, rt).argmax(-1, keepdim=True)).squeeze(-1)
            th = pred // 2
        elif row == 2:
            W, pos = b.W(f, o, len(seen), 0.01, lin8=True)
            logits = N5.aug(g["mean_vec"]) @ W
            cls = torch.tensor([r for p in pos for r in task_rows(p)])
            pred = cls[logits.argmax(-1)]
            col = torch.tensor([pos.index(int(p)) for p in g["task"]])
            mk = 2 * g["task"] + logits.gather(1, torch.stack([2 * col, 2 * col + 1], -1)).argmax(-1)
            th = pred // 2
        else:
            rname = {3: "R3", 4: "AR", 5: "AR", 6: "AR", 7: "AR-bal"}.get(row)
            four = {3: g["L1"][o], 4: g["L0"], 5: g["L1"][o], 6: g["I6"], 7: g["I6"], 8: g["I6"]}[row]
            th = g["task"].clone() if row == 8 else st[router(b, rname, f, o, g, seen).argmax(-1)]
            pred, mk = N2.hard(four, th, g["task"])
        if len(seen) == 4:
            info["th"], info["g"] = th, g
        return pred, mk, g
    return stage, info


def wp_task(b, row, f, o):
    if row in (1, 2):
        return None
    out = []
    for p, t in enumerate(b.tasks):
        c = b.c(f, "test", t)
        ev = {3: c["L1_cos8"][:, list(ORDERS).index(o)], 4: c["L0_cos8"], 5: c["L1_cos8"][:, list(ORDERS).index(o)],
              6: c["I6_cos8"], 7: c["I6_cos8"], 8: c["I6_cos8"]}[row][:, p]
        rr = torch.tensor(task_rows(p))
        out.append((rr[ev[:, rr].argmax(-1)] == c["labels"]).float().mean().item())
    return out


def main() -> int:
    t0 = time.perf_counter()
    b = B8()
    per_fold = {}
    conf = {n: torch.zeros(4, 4, dtype=torch.long) for n in ("R3", "AR", "AR-bal")}
    for row in range(1, 9):
        per_fold[row] = {}
        for o in ORDERS:
            recs = []
            for f in FOLDS:
                stage, info = stage_row(b, row, f, o)
                r = N5.cil_full(stage, o, b.tasks)
                g, th = info["g"], info["th"]
                ok = th == g["task"]
                r["tp_task"] = [ok[g["task"] == p].float().mean().item() for p in range(4)]
                r["wp_task"] = wp_task(b, row, f, o)
                recs.append(r)
                if o == "reverse" and row in (3, 4, 7):
                    n = {3: "R3", 4: "AR", 7: "AR-bal"}[row]
                    for a, c in zip(g["task"].tolist(), th.tolist()):
                        conf[n][a, c] += 1
            per_fold[row][o] = recs
    paired = {}
    for row in (1, 2, 3, 4, 5, 7):
        paired[row] = {}
        for o in ORDERS:
            diffs = [x["acc"] - y["acc"] for x, y in zip(per_fold[6][o], per_fold[row][o])]
            paired[row][o] = {"per_fold": diffs, "mean": mean_sd(diffs)[0], "wins": sum(v > 0 for v in diffs)}

    # 轉交曲線（AR ＋ I6(r=2)）
    curve = {o: {q: [] for q in QS} for o in ORDERS}
    comp = {q: {"n": 0, "mis": 0, "task": [0] * 4, "class": [0] * 8} for q in QS}
    seen4 = list(range(4))
    for o in ORDERS:
        for f in FOLDS:
            g = b.gather(f, seen4)
            sc = router(b, "AR", f, o, g, seen4)
            top = sc.topk(2, dim=-1)
            margin = top.values[:, 0] - top.values[:, 1]
            th = torch.tensor(seen4)[sc.argmax(-1)]
            pred, _ = N2.hard(g["I6"], th, g["task"])
            ok = pred == g["labels"]
            N = len(margin)
            for q in QS:
                k = math.ceil(q * N)
                keep = torch.ones(N, dtype=torch.bool)
                if k:
                    keep[torch.topk(margin, k, largest=False).indices] = False
                curve[o][q].append(sum(ok[keep & (g["task"] == p)].float().mean().item() for p in seen4) / 4)
                if o == "reverse":
                    dd = ~keep
                    comp[q]["n"] += int(dd.sum()); comp[q]["mis"] += int((dd & (th != g["task"])).sum())
                    for p in seen4:
                        comp[q]["task"][p] += int((dd & (g["task"] == p)).sum())
                    for c in range(8):
                        comp[q]["class"][c] += int((dd & (g["labels"] == c)).sum())

    # 與先前報告比對
    M1 = json.loads((b.out / "nc1" / "metrics.json").read_text())
    M3 = json.loads((b.out / "nc3" / "metrics.json").read_text())
    M5 = json.loads((b.out / "nc5" / "metrics.json").read_text())
    M6 = json.loads((b.out / "nc6" / "metrics.json").read_text())
    M7 = json.loads((b.out / "nc7" / "metrics.json").read_text())
    prev = {}
    for o in ORDERS:
        zs = M1["cil"][f"無 gate：zero-shot 8 類（top-64）|{o}"]
        prev[(1, o)] = ("REPORT_stage1-3 T3-a", [x["acc"] for x in zs], [x["masked"] for x in zs], [x["forgetting"] for x in zs])
        l8 = M5["lin8"]["test"][o]
        prev[(2, o)] = ("REPORT_stage7 T4-c", [x["acc"] for x in l8], [x["masked"] for x in l8], [x["forgetting"] for x in l8])
        fs = M3["full_system_v2"][o]
        prev[(3, o)] = ("REPORT_stage5 T-D", fs["acc"], fs["masked"], fs["forgetting"])
        for row, key, src in ((4, "D2", "REPORT_stage8 T4"), (5, "D1", "REPORT_stage8 T4")):
            x = M6[key][o]
            prev[(row, o)] = (src, x["acc"], x["masked"], x["forgetting"])
        x = M7["D3"][o]
        prev[(6, o)] = ("REPORT_stage9 T2", x["acc"], x["masked"], x["forgetting"])
    diffs = []
    for (row, o), (src, acc, mk, fg) in prev.items():
        now = per_fold[row][o]
        for name, old, new in (("ACC", acc, [x["acc"] for x in now]), ("Masked ACC", mk, [x["masked"] for x in now]),
                               ("Forgetting", fg, [x["forgetting"] for x in now])):
            dlt = mean_sd(new)[0] - mean_sd(old)[0]
            maxf = max(abs(a - c) for a, c in zip(new, old))
            diffs.append({"row": row, "order": o, "metric": name, "source": src, "prev": mean_sd(old)[0],
                          "now": mean_sd(new)[0], "diff": dlt, "max_fold_abs_diff": maxf})

    # 輸出
    nc8 = b.out / "nc8"
    fig = nc8 / "fig"
    fig.mkdir(parents=True, exist_ok=True)
    keep = ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t", "R", "Rm", "tp_task", "wp_task")
    (nc8 / "per_fold.json").write_text(json.dumps(
        {"rows": {ROWS[r - 1]: {o: [{k: x[k] for k in keep} for x in v] for o, v in per_fold[r].items()} for r in per_fold},
         "paired_D3_minus_row": {ROWS[r - 1]: v for r, v in paired.items()},
         "folds": FOLDS}, indent=1, ensure_ascii=False))
    with open(fig / "stage_acc.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["row", "order", "t", "acc_mean", "acc_sd", "masked_mean", "masked_sd"])
        for r in (3, 5, 6):
            for o in ORDERS:
                for t in range(4):
                    a = mean_sd([x["acc_t"][t] for x in per_fold[r][o]]); m = mean_sd([x["masked_t"][t] for x in per_fold[r][o]])
                    w.writerow([ROWS[r - 1], o, t + 1, f"{a[0]:.6f}", f"{a[1]:.6f}", f"{m[0]:.6f}", f"{m[1]:.6f}"])
    with open(fig / "deferral_curve.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["defer_frac", "acc_reverse_mean", "acc_reverse_sd", "acc_paper_mean", "acc_paper_sd",
                    "n_deferred", "n_deferred_misrouted", "frac_deferred_misrouted"])
        for q in QS:
            a, p_ = mean_sd(curve["reverse"][q]), mean_sd(curve["paper"][q])
            c = comp[q]
            w.writerow([q, f"{a[0]:.6f}", f"{a[1]:.6f}", f"{p_[0]:.6f}", f"{p_[1]:.6f}", c["n"], c["mis"],
                        f"{(c['mis'] / c['n']) if c['n'] else 0:.6f}"])
    with open(fig / "deferral_composition.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["defer_frac"] + b.tasks + LABEL)
        for q in QS:
            w.writerow([q] + comp[q]["task"] + comp[q]["class"])
    for n, cm in conf.items():
        with open(fig / f"confusion_{n}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["true\\pred"] + b.tasks)
            for i, t in enumerate(b.tasks):
                w.writerow([t] + cm[i].tolist())
    tp = nc8 / "timing_batch.json"
    timing = json.loads(tp.read_text()) if tp.exists() else {}
    write(b, per_fold, paired, curve, comp, conf, diffs, timing, time.perf_counter() - t0)
    print(f"→ {b.out / 'REPORT_stage10.md'}")
    return 0


def write(b, per_fold, paired, curve, comp, conf, diffs, timing, t_rep):
    out = ["# REPORT — NC-8：主表定稿（所有系統同一批重算）與圖的資料（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0。所有數字來自同一台、同一批："
           "每折讀一次 train（mean_vec）與一次 test，由已存權重重算全部證據，不讀先前快取、不讀 validation、不訓練、不做選擇。"
           "判準與定義見 `PREREG-8.md`（commit 6ff3315）。主系統 = D3 = AR（γ = 0.001）＋ I6(r = 2)（依 NC-7 事先訂的判準）。", "",
           "## T1 主表（test、t = 4、十折 mean ± sd）", ""]
    for o in ORDERS:
        out += [f"### {o}", "", "| 列 | ACC | Masked ACC | Forgetting | BWT | TP esca／rcc／brca／lung | WP esca／rcc／brca／lung |",
                "|---|---|---|---|---|---|---|"]
        for r in range(1, 9):
            recs = per_fold[r][o]
            tp = "／".join(f"{mean_sd([x['tp_task'][p] for x in recs])[0]:.4f}" for p in range(4))
            wp = ("—" if recs[0]["wp_task"] is None else
                  "／".join(f"{mean_sd([x['wp_task'][p] for x in recs])[0]:.4f}" for p in range(4)))
            out.append(f"| {ROWS[r - 1]} | {fmt([x['acc'] for x in recs])} | {fmt([x['masked'] for x in recs])} | "
                       f"{fmt([x['forgetting'] for x in recs])} | {fmt([x['bwt'] for x in recs])} | {tp} | {wp} |")
        out.append("")
    out += ["zero-shot 與 LIN8 的 TP 以預測類別所屬任務計；兩者沒有 expert，不報 WP。oracle 的 ACC 等於 Masked ACC。", "",
            "儲存（fp32 bytes，不含所有列共用的類別文字 f_txt）：", "", "| 列 | expert 合計 | router 統計量 |", "|---|---|---|"]
    for r in range(1, 9):
        eb = EXPERT_BYTES[r]
        out.append(f"| {ROWS[r - 1]} | {'—（無 expert）' if eb is None else f'{eb:,}'} | {ROUTER_BYTES[r]} |")
    out += ["", "## T2 D3 逐折比較（test t = 4 ACC，D3 − 該列）", "",
            "| 對照列 | 序 | " + " | ".join(str(f) for f in FOLDS) + " | 平均差 | D3 贏的折數 |", "|---|---|" + "---|" * 12]
    for r, v in paired.items():
        for o in ORDERS:
            p = v[o]
            out.append(f"| {ROWS[r - 1]} | {o} | " + " | ".join(f"{x:+.4f}" for x in p["per_fold"]) +
                       f" | {p['mean']:+.4f} | {p['wins']}/10 |")
    out += ["", "## T3 轉交曲線（router = AR γ = 0.001、expert = I6(r=2)、test t = 4）", "",
            "| 轉交比例 | ACC reverse | ACC paper | 轉交張數 | 其中原本分派錯 | 比例 |", "|---|---|---|---|---|---|"]
    for q in QS:
        c = comp[q]
        out.append(f"| {q:.0%} | {fmt(curve['reverse'][q])} | {fmt(curve['paper'][q])} | {c['n']} | {c['mis']} | "
                   f"{(c['mis'] / c['n']) if c['n'] else 0:.1%} |")
    out += ["", "轉交片子的組成（十折合計）：", "", "| 轉交比例 | " + " | ".join(b.tasks) + " | " + " | ".join(LABEL) + " |",
            "|---|" + "---|" * 12]
    for q in QS[1:]:
        c = comp[q]
        out.append(f"| {q:.0%} | " + " | ".join(str(x) for x in c["task"]) + " | " + " | ".join(str(x) for x in c["class"]) + " |")
    out += ["", "## T4 分派混淆矩陣（test t = 4、十折合計；列 = 真實、欄 = 分派）", ""]
    for n, cm in conf.items():
        out += [f"**{n}**", "", "| 真實 \\ 分派 | " + " | ".join(b.tasks) + " |", "|---|" + "---|" * 4]
        for i, t in enumerate(b.tasks):
            out.append(f"| {t} | " + " | ".join(str(int(x)) for x in cm[i]) + " |")
        out.append("")
    big = [x for x in diffs if abs(x["diff"]) > 0.001]
    out += ["## T5 與先前報告的比對（十折平均）", ""]
    if big:
        out += ["| 列 | 序 | 指標 | 來源 | 先前 | 本批 | 差 |", "|---|---|---|---|---|---|---|"]
        for x in big:
            out.append(f"| {ROWS[x['row'] - 1]} | {x['order']} | {x['metric']} | {x['source']} | {x['prev']:.4f} | "
                       f"{x['now']:.4f} | {x['diff']:+.4f} |")
    else:
        mx = max(abs(x["max_fold_abs_diff"]) for x in diffs)
        out.append(f"第 1–6 列的 ACC、Masked ACC、Forgetting 與先前報告相比，十折平均差全部 ≤ 0.001"
                   f"（逐折最大絕對差 {mx:.2e}）。第 7、8 列為新列（先前未報）。")
    out += ["", "## T6 圖的資料與每折原始值", "", "| 檔案 | 內容 |", "|---|---|",
            "| `nc8/fig/stage_acc.csv` | 第 3、5、6 列 t = 1–4 的 ACC 與 Masked ACC（兩序，十折 mean、sd） |",
            "| `nc8/fig/deferral_curve.csv` | 轉交比例 × 剩下片子 ACC（兩序）、轉交張數、其中分派錯的比例 |",
            "| `nc8/fig/deferral_composition.csv` | 轉交片子的任務與類別組成 |",
            "| `nc8/fig/confusion_{R3,AR,AR-bal}.csv` | 4 × 4 分派混淆矩陣 |",
            "| `nc8/per_fold.json` | 8 列 × 兩序 × 十折的 ACC、Masked ACC、Forgetting、BWT、逐階段 ACC、R、每任務 TP／WP，及 D3 配對差 |", "",
            "## T7 實際耗時（秒，wall clock；執行緒 8）", "", "| 項目 | 秒 |", "|---|---|",
            f"| 同一批重算（十折 train mean_vec ＋ test 全部證據） | {timing.get('run/1-10', float('nan')):,.0f} |",
            f"| 報告（router 統計量、8 列評估、CSV） | {t_rep:,.0f} |", ""]
    (b.out / "REPORT_stage10.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
