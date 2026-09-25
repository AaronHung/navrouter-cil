#!/usr/bin/env python3
"""NC-1 報告：只讀 outputs/navcil/<machine>/cache/ 與 bank/，離線算第一～三關。

    NAVCIL_MACHINE=mac python scripts/nc1_report.py [--folds 1-10]
輸出：outputs/navcil/<machine>/REPORT_stage1-3.md、nc1/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.cil_eval import mean_sd, stage1_task, subset_key          # noqa: E402
from selector.cil_ops import LAMBDAS, ORDERS, task_rows                  # noqa: E402
from selector.flat_selector import EvidenceSelector                      # noqa: E402
from selector.router import tp_pred, tp_scores                           # noqa: E402
from selector.text_encoder import _abs, build_f_txt, load_config         # noqa: E402

T_GRID = (0.01, 0.02, 0.05, 0.1)
TP_ALL = ["text-class", "text-organ", "patch-vote", "proto", "nav", "nav-cal", "fuse"]
TP_CIL = ["oracle", "text-class", "text-organ", "proto", "nav"]
EXT = json.loads((REPO_ROOT / "reference" / "external_baselines.json").read_text())
EXT_ACC = EXT["acc"]            # 第三關門檻 = 外部已發表值
S1_KEYS = [("a_zs_all", "(a) zero-shot 全部 patch"), ("b_zs_top64", "(b) zero-shot top-64"),
           ("c_random64", "(c) random-64（5 seeds）"), ("d_oneshot", "(d) 導覽器一次 top-64"),
           ("e_fourround", "(e) 導覽器四輪"),
           ("d_oneshot_softmax", "(d) softmax 聚合"), ("e_fourround_softmax", "(e) softmax 聚合")]
S1_ORACLE = [("d_oracle8", "(d) oracle 8 類"), ("e_oracle8", "(e) oracle 8 類"),
             ("d_oracle8_softmax", "(d) oracle 8 類 softmax"),
             ("e_oracle8_softmax", "(e) oracle 8 類 softmax")]


def fmt(xs) -> str:
    m, sd = mean_sd(list(xs))
    return f"{m:.4f} ± {sd:.4f}"


class Data:
    def __init__(self, folds):
        self.cfg = load_config()
        self.tasks = list(self.cfg["tasks"])
        self.out = REPO_ROOT / "outputs" / "navcil" / self.cfg["machine"]
        self.cache = self.out / "cache"
        self.folds = folds
        self.lam_info = json.loads((self.out / "nc1" / "lambda.json").read_text())
        self.lam = self.lam_info["lambda_star"]
        self.li = LAMBDAS.index(self.lam)
        self.ls = float(build_f_txt(self.tasks[0], self.cfg).logit_scale)
        self.organ = torch.load(_abs(self.cfg["f_txt_cache_dir"]) / "organ_keys.pt",
                                map_location="cpu")
        assert self.organ["tasks"] == self.tasks
        self._c = {}

    def c(self, fold, split, task) -> dict:
        k = (fold, split, task)
        if k not in self._c:
            self._c[k] = torch.load(self.cache / f"fold{fold}_{split}_{task}.pt",
                                    map_location="cpu")
        return self._c[k]

    def proto(self, fold) -> torch.Tensor:
        return torch.stack([F.normalize(self.c(fold, "train", t)["mean_vec"].mean(0), dim=-1)
                            for t in self.tasks])


def gather(d: Data, fold: int, split: str, seen: list[int]) -> dict:
    """把 seen 任務的 slides 疊起來。val 的四輪取 λ* 那一格。"""
    parts = [d.c(fold, split, d.tasks[p]) for p in seen]
    g = {"labels": torch.cat([c["labels"] for c in parts]),
         "task": torch.cat([torch.full((len(c["labels"]),), p) for p, c in zip(seen, parts)]),
         "mean_vec": torch.cat([c["mean_vec"] for c in parts]),
         "mean_cos8": torch.cat([c["mean_cos8"] for c in parts])}
    if split == "val":
        g["four"] = torch.cat([c["four_grid_cos8"][:, d.li] for c in parts])
    else:
        key = subset_key(seen)
        si = parts[0]["subsets"].index(key)
        g["four"] = torch.cat([c["four_cos8_uni"] for c in parts])
        for k in ("zs8_top64_cos8", "emax_cos8", "vote_counts", "vote_msum"):
            g[k] = torch.cat([c[k][:, si] for c in parts])
    return g


def cil_scores(d, fold, g, seen, method, T=None) -> torch.Tensor:
    """[N, 8] 類別分數；未見類別為 -inf。method = (tp, combo) 或無 gate 名稱。"""
    N = g["labels"].numel()
    seen_rows = [r for p in seen for r in task_rows(p)]
    out = torch.full((N, 8), float("-inf"))
    if method in ("zs8-all", "zs8-top64", "e-max"):
        src = {"zs8-all": "mean_cos8", "zs8-top64": "zs8_top64_cos8", "e-max": "emax_cos8"}[method]
        out[:, seen_rows] = g[src][:, seen_rows]
        return out
    tp, combo = method
    four = g["four"]                                   # [N, 4, 8]
    if combo in ("Hard", "Hard-8") or tp == "oracle":
        th = tp_pred(d, fold, g, seen, tp)
        z = four[torch.arange(N), th]                  # [N, 8]
        rows = torch.tensor(seen_rows) if combo == "Hard-8" else None
        if rows is None:
            for i in range(N):
                rr = task_rows(int(th[i]))
                out[i, rr] = z[i, rr]
        else:
            out[:, rows] = z[:, rows]
        return out
    pi = F.softmax(tp_scores(d, fold, g, seen, tp).float() / T, dim=-1)   # Soft
    for j, p in enumerate(seen):
        rr = task_rows(p)
        out[:, rr] = pi[:, j:j + 1] * F.softmax(d.ls * four[:, p][:, rr], dim=-1)
    return out


def cil_eval(d, fold, order, method, T=None) -> dict:
    tasks = [d.tasks.index(t) for t in ORDERS[order]]
    R = [[None] * 4 for _ in range(4)]
    masked = None
    for t in range(1, 5):
        seen = tasks[:t]
        g = gather(d, fold, "test", seen)
        sc = cil_scores(d, fold, g, seen, method, T)
        ok = sc.argmax(-1) == g["labels"]
        for j in range(t):
            R[t - 1][j] = ok[g["task"] == tasks[j]].float().mean().item()
        if t == 4:
            mk = []
            for p in tasks:
                m = g["task"] == p
                rr = torch.tensor(task_rows(p))
                mk.append((rr[sc[m][:, rr].argmax(-1)] == g["labels"][m]).float().mean().item())
            masked = sum(mk) / 4
    acc = sum(R[3]) / 4
    forg = sum(max(R[t][j] for t in range(j, 3)) - R[3][j] for j in range(3)) / 3
    return {"R": R, "acc": acc, "masked": masked, "forgetting": forg}


def choose_T(d) -> dict:
    """fold 1 validation、t = 4、Soft 的 ACC；同分取小。"""
    seen = list(range(4))
    g = gather(d, 1, "val", seen)
    out = {}
    for tp in TP_CIL[1:]:
        accs = {}
        for T in T_GRID:
            ok = cil_scores(d, 1, g, seen, (tp, "Soft"), T).argmax(-1) == g["labels"]
            accs[T] = sum(ok[g["task"] == p].float().mean().item() for p in seen) / 4
        best = max(accs.values())
        out[tp] = {"T_star": min(T for T in T_GRID if accs[T] == best),
                   "val_acc": {str(k): v for k, v in accs.items()}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1-10")
    a, _, b = ap.parse_args().folds.partition("-")
    folds = list(range(int(a), int(b or a) + 1))
    d = Data(folds)
    t_start = time.perf_counter()
    M = {"lambda": d.lam_info, "folds": folds}

    # ── T1 ──────────────────────────────────────────────────────────────────
    s1 = {f: [stage1_task(d.c(f, "test", t), p) for p, t in enumerate(d.tasks)] for f in folds}
    fold_mean = {k: [sum(r[k] for r in s1[f]) / 4 for f in folds]
                 for k, _ in S1_KEYS + S1_ORACLE}
    wins = sum(e > b for e, b in zip(fold_mean["e_fourround"], fold_mean["b_zs_top64"]))
    M["stage1"] = {"per_fold": s1, "fold_mean": fold_mean, "wins_e_over_b": wins}
    e_m, _ = mean_sd(fold_mean["e_fourround"])
    b_m, _ = mean_sd(fold_mean["b_zs_top64"])
    gate1 = {"e_mean": e_m, "b_mean": b_m, "diff": e_m - b_m, "wins": wins,
             "pass": e_m >= 0.90 and e_m - b_m >= 0.03 and wins >= 7}
    M["gate1"] = gate1

    # ── T2 ──────────────────────────────────────────────────────────────────
    tp = {}
    conf = {}
    per_task_tp4 = {}
    for order, names in ORDERS.items():
        tasks = [d.tasks.index(t) for t in names]
        for t in range(1, 5):
            seen = tasks[:t]
            for f in folds:
                g = gather(d, f, "test", seen)
                for v in TP_ALL:
                    pred = tp_pred(d, f, g, seen, v)
                    tp.setdefault((v, order, t), []).append(
                        (pred == g["task"]).float().mean().item())
                    if t == 4:
                        cm = conf.setdefault((v, order), torch.zeros(4, 4, dtype=torch.long))
                        for tt, pp in zip(g["task"].tolist(), pred.tolist()):
                            cm[tt, pp] += 1
                        for p in range(4):
                            m = g["task"] == p
                            per_task_tp4.setdefault((v, f), {})[p] = \
                                (pred[m] == p).float().mean().item()
    M["tp"] = {f"{v}|{o}|{t}": xs for (v, o, t), xs in tp.items()}
    M["tp_confusion_t4"] = {f"{v}|{o}": cm.tolist() for (v, o), cm in conf.items()}
    tp4 = {v: mean_sd(tp[(v, "reverse", 4)] + tp[(v, "paper", 4)])[0] for v in TP_ALL}
    tc, nv, pr = tp4["text-class"], tp4["nav"], tp4["proto"]
    if tc >= 0.97:
        g2 = "TP-text-class ≥ 0.97 → N1 降為消融，第四關主做 N2"
    elif tc <= 0.95 and nv >= tc + 0.03 and nv >= pr - 0.005:
        g2 = "TP-text-class ≤ 0.95、TP-nav ≥ TP-text-class + 0.03 且 TP-nav ≥ TP-proto − 0.005 → N1 為主線"
    else:
        g2 = "其他 → N1 列為消融，主做 N2"
    M["gate2"] = {"tp_t4": tp4, "verdict": g2}

    # ── T3 ──────────────────────────────────────────────────────────────────
    Tsel = choose_T(d)
    M["T_star"] = Tsel
    methods = []
    for v in TP_CIL:
        for combo in ("Hard", "Hard-8", "Soft"):
            if v == "oracle" and combo == "Soft":
                continue
            methods.append(((v, combo), f"{v} + {combo}"))
    methods += [("zs8-all", "無 gate：zero-shot 8 類（全部 patch）"),
                ("zs8-top64", "無 gate：zero-shot 8 類（top-64）"),
                ("e-max", "無 gate：E-max")]
    cil = {}
    for m, name in methods:
        T = Tsel[m[0]]["T_star"] if isinstance(m, tuple) and m[1] == "Soft" else None
        for order in ORDERS:
            cil[(name, order)] = [cil_eval(d, f, order, m, T) for f in folds]
    M["cil"] = {f"{n}|{o}": v for (n, o), v in cil.items()}
    g3 = {}
    for v in ("text-class", "nav"):
        g3[v] = {o: mean_sd([r["acc"] for r in cil[(f"{v} + Hard", o)]])[0] for o in ORDERS}
    gate3_pass = any(g3[v]["reverse"] >= EXT_ACC["reverse"] and g3[v]["paper"] >= EXT_ACC["paper"]
                     for v in g3)
    M["gate3"] = {"acc": g3, "pass": gate3_pass}

    # TP × WP vs 實測（t = 4；兩序在 t = 4 相同，取 reverse）
    txw = {}
    for v in TP_CIL:
        for i, f in enumerate(folds):
            wp = {p: s1[f][p]["e_fourround"] for p in range(4)}
            prod = sum(per_task_tp4[(v, f)][p] * wp[p] for p in range(4)) / 4 \
                if v != "oracle" else sum(wp.values()) / 4
            meas = cil[(f"{v} + Hard", "reverse")][i]["acc"]
            txw.setdefault(v, []).append((prod, meas))
    M["tp_x_wp"] = txw

    # ── T4 ──────────────────────────────────────────────────────────────────
    n_param = sum(p.numel() for p in EvidenceSelector().parameters())
    files = sorted((d.out / "bank").glob("fold1_tcga_*.pt"))
    M["storage"] = {"selector_params": n_param, "selector_bytes_fp32": n_param * 4,
                    "selector_file_bytes": {p.stem: p.stat().st_size for p in files}}
    timing = json.loads((d.out / "nc1" / "timing.json").read_text())
    M["timing"] = timing
    (d.out / "nc1" / "metrics.json").write_text(json.dumps(M, indent=1, default=str))

    write_report(d, M, s1, fold_mean, tp, conf, cil, methods, Tsel, txw, timing,
                 time.perf_counter() - t_start)
    print(f"→ {d.out / 'REPORT_stage1-3.md'}")
    return 0


def write_report(d, M, s1, fold_mean, tp, conf, cil, methods, Tsel, txw, timing, t_rep):
    g1, g2, g3 = M["gate1"], M["gate2"], M["gate3"]
    folds = d.folds
    ok = lambda b: "**通過**" if b else "**未通過**"                    # noqa: E731
    L = ["# REPORT — NC-1 第一～三關（Mac CPU，十折兩序）", "",
         f"機器：mac（Apple M1 Pro）、`--device cpu`、torch {torch.__version__}；"
         f"所有數字來自同一台、同一批。folds {folds[0]}–{folds[-1]}。"
         f"判準見 `PREREG.md`（commit e06a8e5）。", "",
         "## PREREG 判準落點", "",
         "| 判準 | 數值 | 所在表格 | 結果 |", "|---|---|---|---|",
         f"| 第一關：(e) 十折平均 ≥ 0.90 | {g1['e_mean']:.4f} | T1-a「四任務平均」列 (e) 欄 | "
         f"{ok(g1['e_mean'] >= 0.90)} |",
         f"| 第一關：(e) − (b) ≥ 0.03 | {g1['e_mean']:.4f} − {g1['b_mean']:.4f} = "
         f"{g1['diff']:+.4f} | T1-a「四任務平均」列 (b)(e) 欄 | {ok(g1['diff'] >= 0.03)} |",
         f"| 第一關：(e) 贏 (b) 的折數 ≥ 7/10 | {g1['wins']}/{len(folds)} | T1-c | "
         f"{ok(g1['wins'] >= 7)} |",
         f"| **第一關整體** | | | {ok(g1['pass'])} |",
         f"| 第二關：t = 4 TP（兩序十折平均） | text-class {g2['tp_t4']['text-class']:.4f}、"
         f"nav {g2['tp_t4']['nav']:.4f}、proto {g2['tp_t4']['proto']:.4f} | T2-a 的 t = 4 列 | "
         f"{g2['verdict']} |",
         f"| 第三關：text-class + Hard ACC | reverse {g3['acc']['text-class']['reverse']:.4f}"
         f"（≥ 0.859？）、paper {g3['acc']['text-class']['paper']:.4f}（≥ 0.890？） | "
         f"T3-a「text-class + Hard」列 | "
         f"{ok(g3['acc']['text-class']['reverse'] >= 0.859 and g3['acc']['text-class']['paper'] >= 0.890)} |",
         f"| 第三關：nav + Hard ACC | reverse {g3['acc']['nav']['reverse']:.4f}、"
         f"paper {g3['acc']['nav']['paper']:.4f} | T3-a「nav + Hard」列 | "
         f"{ok(g3['acc']['nav']['reverse'] >= 0.859 and g3['acc']['nav']['paper'] >= 0.890)} |",
         f"| **第三關整體**（兩者任一，零 replay） | | | {ok(g3['pass'])} |", ""]

    # T1
    L += ["## T1 第一關：Masked ACC（test，十折 mean ± sd）", "",
          f"λ\\* = **{d.lam}**（fold 1 validation，四任務 (e) Masked ACC 平均，同分取小）。", "",
          "| λ | val 四任務平均 | " + " | ".join(d.tasks) + " |",
          "|---|---|" + "---|" * 4]
    for l in LAMBDAS:
        row = M["lambda"]["grid"][str(l)]
        L.append(f"| {l} | {row['mean']:.4f} | " +
                 " | ".join(f"{row['per_task'][t]:.4f}" for t in d.tasks) + " |")
    L += ["", "### T1-a 主聚合（所選 patch 等權平均後正規化）與 softmax 聚合", "",
          "| 任務 | " + " | ".join(n for _, n in S1_KEYS) + " |",
          "|---|" + "---|" * len(S1_KEYS)]
    for p, t in enumerate(d.tasks):
        L.append(f"| {t} | " + " | ".join(fmt([s1[f][p][k] for f in folds])
                                          for k, _ in S1_KEYS) + " |")
    L.append("| **四任務平均** | " + " | ".join(fmt(fold_mean[k]) for k, _ in S1_KEYS) + " |")
    L += ["", "### T1-b oracle 選導覽器下的 8 類 ACC", "",
          "| 任務 | " + " | ".join(n for _, n in S1_ORACLE) + " |",
          "|---|" + "---|" * len(S1_ORACLE)]
    for p, t in enumerate(d.tasks):
        L.append(f"| {t} | " + " | ".join(fmt([s1[f][p][k] for f in folds])
                                          for k, _ in S1_ORACLE) + " |")
    L.append("| **四任務平均** | " + " | ".join(fmt(fold_mean[k]) for k, _ in S1_ORACLE) + " |")
    L += ["", "### T1-c 每折四任務平均：(e) 對 (b)", "",
          "| fold | (b) | (e) | (e) − (b) | 贏 |", "|---|---|---|---|---|"]
    for i, f in enumerate(folds):
        b, e = fold_mean["b_zs_top64"][i], fold_mean["e_fourround"][i]
        L.append(f"| {f} | {b:.4f} | {e:.4f} | {e - b:+.4f} | {'✓' if e > b else ''} |")
    L.append(f"| **贏折數** | | | | **{M['stage1']['wins_e_over_b']}/{len(folds)}** |")

    # T2
    L += ["", "## T2 第二關：TP 正確率（test，十折 mean ± sd）", "",
          "TP 正確率 = 前 t 個任務的 test slides 合併後，逐張判對任務的比例。", "",
          "### T2-a 各變體 × 階段 × 兩序", "",
          "| 變體 | 序 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|"]
    for v in TP_ALL:
        for o in ORDERS:
            L.append(f"| {v} | {o} | " + " | ".join(fmt(tp[(v, o, t)]) for t in range(1, 5)) + " |")
    L += ["", "t = 4 兩序、十折平均（第二關判定值）：" +
          "、".join(f"{v} {M['gate2']['tp_t4'][v]:.4f}" for v in TP_ALL), "",
          "### T2-b t = 4 混淆矩陣（十折合計張數；列 = 真實任務、欄 = 預測任務）", "",
          "兩序在 t = 4 的候選集合相同；下表為 reverse，paper 逐格相同者標 =。", ""]
    for v in TP_ALL:
        cm_r, cm_p = conf[(v, "reverse")], conf[(v, "paper")]
        same = "=" if torch.equal(cm_r, cm_p) else "（paper 不同，見 metrics.json）"
        L += [f"**{v}** {same}", "", "| 真實 \\ 預測 | " + " | ".join(d.tasks) + " |",
              "|---|" + "---|" * 4]
        for i, t in enumerate(d.tasks):
            L.append(f"| {t} | " + " | ".join(str(int(x)) for x in cm_r[i]) + " |")
        L.append("")

    # T3
    L += ["## T3 第三關：CIL（test，十折 mean ± sd，零 replay）", "",
          "T\\*（fold 1 validation、t = 4、Soft ACC，同分取小）：" +
          "、".join(f"{v} {Tsel[v]['T_star']}" for v in Tsel) + "。"
          "oracle 的 Soft 與 Hard 相同，不另列。", "",
          "### T3-a", "",
          "| 組合 | reverse ACC | reverse Masked ACC | reverse Forgetting | "
          "paper ACC | paper Masked ACC | paper Forgetting |",
          "|---|---|---|---|---|---|---|"]
    for _, name in methods:
        cells = []
        for o in ORDERS:
            rs = cil[(name, o)]
            cells += [fmt([r["acc"] for r in rs]), fmt([r["masked"] for r in rs]),
                      fmt([r["forgetting"] for r in rs])]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append(f"| *外部參考：{EXT['method']}（已發表值，非 paired）* | *{EXT_ACC['reverse']:.3f}* | | | "
             f"*{EXT_ACC['paper']:.3f}* | | |")
    L += ["", "### T3-b 每折 TP × WP 與實測 CIL（t = 4，Hard；WP = (e) Masked ACC）", "",
          "TP × WP = 平均_j(TP_j × WP_j)；兩序在 t = 4 相同。", "",
          "| fold | " + " | ".join(f"{v} TP×WP | {v} 實測" for v in TP_CIL) + " |",
          "|---|" + "---|---|" * len(TP_CIL)]
    for i, f in enumerate(folds):
        L.append(f"| {f} | " + " | ".join(f"{txw[v][i][0]:.4f} | {txw[v][i][1]:.4f}"
                                         for v in TP_CIL) + " |")
    L.append("| **mean** | " + " | ".join(
        f"{mean_sd([x[0] for x in txw[v]])[0]:.4f} | {mean_sd([x[1] for x in txw[v]])[0]:.4f}"
        for v in TP_CIL) + " |")

    # T4
    st = M["storage"]
    L += ["", "## T4 每任務參數量與儲存位元組數", "",
          "| 項目 | 每任務參數量 | 每任務位元組（fp32） | 備註 |", "|---|---|---|---|",
          f"| 導覽器 EvidenceSelector（Linear 514→256、GELU、Linear 256→1） | "
          f"{st['selector_params']:,} | {st['selector_bytes_fp32']:,} | state_dict 檔案大小："
          + "、".join(f"{k.split('_', 1)[1]} {v:,}" for k, v in st["selector_file_bytes"].items())
          + " |",
          "| 類別文字特徵 f_txt（2 × 512） | 0（凍結） | 4,096 | 分類頭本身需要，所有 TP 共用 |",
          "| TP text-class | 0 | 0 | 直接用 f_txt |",
          "| TP text-organ key（512） | 0 | 2,048 | 由文字產生，不需資料 |",
          "| TP proto key（512） | 0 | 2,048 | 需要該任務訓練 slides 的特徵平均 |",
          "| TP nav | 0 | 0 | 用導覽器與 f_txt |",
          "| TP nav-cal（每個已見集合：mean、sd） | 0 | 8 / 已見集合 | 需要訓練 slides 的分數統計 |",
          "| replay | — | 0 | 零 replay |"]

    # T5
    train_s = sum(v for k, v in timing.items() if k.startswith("train/") and not k.endswith("total"))
    cache_test = sum(v for k, v in timing.items() if k.startswith("cache/") and "_test_" in k)
    cache_train = sum(v for k, v in timing.items() if k.startswith("cache/") and "_train_" in k)
    cache_val = sum(v for k, v in timing.items() if k.startswith("cache/") and "_val_" in k)
    L += ["", "## T5 實際耗時（秒，wall clock）", "",
          "| 階段 | 秒 |", "|---|---|",
          f"| 第一關訓練（10 折 × 4 任務 × 5 epochs） | {train_s:,.0f} |",
          f"| fold 1 validation 快取（λ 網格） | {cache_val:,.0f} |",
          f"| test 快取（每張讀一次，各導覽器選片與全部離線量） | {cache_test:,.0f} |",
          f"| train 快取（proto、nav-cal） | {cache_train:,.0f} |"]
    for f in folds:
        if f"fold{f}/total" in timing:
            L.append(f"| fold {f} 合計 | {timing[f'fold{f}/total']:,.0f} |")
    runs = [v for k, v in timing.items() if k.startswith("run/")]
    if runs:
        L.append(f"| 主流程總計 | {sum(runs):,.0f} |")
    L.append(f"| 報告（第一～三關離線計算） | {t_rep:,.0f} |")
    L += ["", "逐張讀檔／計算秒數存在 `cache/*.pt` 的 `t_read_s`、`t_compute_s` 與 "
          "`nc1/train_logs/`；全部數值另存 `nc1/metrics.json`。", ""]
    (d.out / "REPORT_stage1-3.md").write_text("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
