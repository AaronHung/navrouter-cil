#!/usr/bin/env python3
"""NC-5 報告：背景分派 ctx、機制診斷、對照組 AR／LIN8、主系統逐階段指標。

只讀快取、lora_v2 評估檔與既有 metrics，不碰特徵檔。
    NAVCIL_MACHINE=mac python scripts/nc5_report.py
輸出：outputs/navcil/<machine>/REPORT_stage7.md、nc5/metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
import nc4_report as N4                                                   # noqa: E402
from selector.cil_eval import mean_sd, subset_key                         # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402

FOLDS = N2.FOLDS
QS = (0.25, 0.5, 0.75)
SOURCES = ("z", "c", "e")
GAMMAS = (0.01, 0.1, 1.0, 10.0)
ALL = "0,1,2,3"
CANDS = ["基準"] + [f"{s}{q}" for s in SOURCES for q in (0.75, 0.5, 0.25)]    # 同分順序
fmt = N2.fmt
IE, IL = 0, 3                                                             # esca、lung（canonical）
LABEL = {0: "ESAD", 1: "ESCC", 6: "LUAD", 7: "LUSC"}


def parse(c):
    return c[0], float(c[1:])


# ── 共用：CIL（含逐階段 Masked）──────────────────────────────────────────────
def cil_full(stage_fn, order, tasks_all) -> dict:
    pos = [tasks_all.index(t) for t in ORDERS[order]]
    R = [[None] * 4 for _ in range(4)]
    Rm = [[None] * 4 for _ in range(4)]
    for t in range(1, 5):
        pred, mk, g = stage_fn(pos[:t])
        ok, okm = pred == g["labels"], mk == g["labels"]
        for j in range(t):
            m = g["task"] == pos[j]
            R[t - 1][j] = ok[m].float().mean().item()
            Rm[t - 1][j] = okm[m].float().mean().item()
    acc_t = [sum(R[t][:t + 1]) / (t + 1) for t in range(4)]
    mk_t = [sum(Rm[t][:t + 1]) / (t + 1) for t in range(4)]
    forg = sum(max(R[t][j] for t in range(j, 3)) - R[3][j] for j in range(3)) / 3
    bwt = sum(R[3][j] - R[j][j] for j in range(3)) / 3
    return {"R": R, "Rm": Rm, "acc": acc_t[3], "masked": mk_t[3], "forgetting": forg, "bwt": bwt,
            "acc_t": acc_t, "masked_t": mk_t}


class Ctx5:
    def __init__(self, d):
        self.d, self._c, self._k = d, {}, {}

    def c(self, fold, split, task):
        k = (fold, split, task)
        if k not in self._c:
            self._c[k] = torch.load(self.d.cache / f"nc5_fold{fold}_{split}_{task}.pt", map_location="cpu")
        return self._c[k]

    def vec(self, fold, split, p, src, q, key, kind="bg"):
        """任務 p 的 slides 在已見子集 key 下的向量 [N, D]；q=None 表示腫瘤半向量。"""
        c = self.c(fold, split, self.d.tasks[p])
        if src == "z":
            return c["z_th"] if kind == "th" else c["z_bg"][:, QS.index(q)]
        si = c["subsets"].index(key)
        return c[f"{src}_th"][:, si] if kind == "th" else c[f"{src}_bg"][:, si, QS.index(q)]

    def keys(self, fold, order, src, q, kind="bg"):
        """{task_pos: 8 個正規化 centroid}。z 與順序無關。"""
        o = "any" if src == "z" else order
        k = (fold, o, src, q, kind)
        if k not in self._k:
            out = {}
            for p in range(4):
                names = ORDERS[order]
                key = subset_key([self.d.tasks.index(x) for x in names[:names.index(self.d.tasks[p]) + 1]])
                out[p] = N4.kmeans_keys(self.vec(fold, "train", p, src, q, key, kind), 8)
            self._k[k] = out
        return self._k[k]

    def scores(self, fold, order, split, seen, src, q, kind="bg"):
        key = subset_key(seen) if split == "test" else ALL
        keys = self.keys(fold, order, src, q, kind)
        V = torch.cat([self.vec(fold, split, p, src, q, key, kind) for p in seen])
        return torch.stack([(V @ keys[p].t()).amax(-1) for p in seen], -1)


def router_scores(d, C5, AR, name, fold, order, split, g, seen):
    if name == "基準":
        return N2.scores_for(d, fold, g, seen, "R3", 8)
    if name.startswith("AR"):
        return AR(fold, order, split, g, seen)
    if name.startswith("TH-"):
        return C5.scores(fold, order, split, seen, name[3], None, "th")
    s, q = parse(name)
    return C5.scores(fold, order, split, seen, s, q)


def tp_stats(d, th, g):
    ok = th == g["task"]
    accs = [ok[g["task"] == p].float().mean().item() for p in range(4)]
    e2l = (g["task"] == IE) & (th == IL)
    l2e = (g["task"] == IL) & (th == IE)
    return {"tp_task": accs, "macro": sum(accs) / 4, "micro": ok.float().mean().item(),
            "e2l": int(e2l.sum()), "l2e": int(l2e.sum()),
            "e2l_cls": {LABEL[c]: int((e2l & (g["labels"] == c)).sum()) for c in (0, 1)},
            "l2e_cls": {LABEL[c]: int((l2e & (g["labels"] == c)).sum()) for c in (6, 7)}}


def evaluate(d, C5, AR, name, four_fn=None) -> dict:
    """test：兩序 CIL（逐階段）與 t = 4 的 TP 診斷；以及 t = 4 的逐張預測（修好／弄壞用）。"""
    res = {}
    for o in ORDERS:
        cil, diag, per = [], [], []
        for f in FOLDS:
            def stage(seen, f=f, o=o):
                g = N2.gather_r(d, f, "test", seen)
                th = torch.tensor(seen)[router_scores(d, C5, AR, name, f, o, "test", g, seen).argmax(-1)]
                four = g["four"] if four_fn is None else four_fn(f, o, seen)
                pred, mk = N2.hard(four, th, g["task"])
                if len(seen) == 4:
                    diag.append(tp_stats(d, th, g)); per.append((th, pred == g["labels"], g))
                return pred, mk, g
            cil.append(cil_full(stage, o, d.tasks))
        res[o] = {"cil": cil, "diag": diag, "per": per}
    return res


def val_acc(d, C5, AR, name, order, fold) -> float:
    seen = list(range(4))
    g = N2.gather_r(d, fold, "val", seen)
    th = torch.tensor(seen)[router_scores(d, C5, AR, name, fold, order, "val", g, seen).argmax(-1)]
    pred, _ = N2.hard(g["four"], th, g["task"])
    ok = pred == g["labels"]
    return sum(ok[g["task"] == p].float().mean().item() for p in seen) / 4


# ── C：AR 與 LIN8 ───────────────────────────────────────────────────────────
def aug(X):
    return torch.cat([X, torch.ones(X.shape[0], 1)], 1)


class Ridge:
    """依序累加的解析式 router／8 類線性基線（float32）。"""

    def __init__(self, d, C5, inp=None):
        self.d, self.C5, self.inp = d, C5, inp        # inp = None（mean_vec）或 (src, q)
        self._W = {}

    def train_X(self, fold, order, p):
        if self.inp is None:
            return self.d.c(fold, "train", self.d.tasks[p])["mean_vec"]
        src, q = self.inp
        names = ORDERS[order]
        key = subset_key([self.d.tasks.index(x) for x in names[:names.index(self.d.tasks[p]) + 1]])
        return self.C5.vec(fold, "train", p, src, q, key)

    def W(self, fold, order, t, gamma, lin8=False):
        k = (fold, order, t, gamma, lin8)
        if k not in self._W:
            pos = [self.d.tasks.index(x) for x in ORDERS[order]]
            A = torch.zeros(513, 513)
            cols = []
            for p in pos[:t]:
                Xj = aug(self.train_X(fold, order, p))
                A += Xj.t() @ Xj
                if lin8:
                    y = self.d.c(fold, "train", self.d.tasks[p])["labels"] - 2 * p
                    cols += [Xj[y == 0].sum(0), Xj[y == 1].sum(0)]
                else:
                    cols.append(Xj.sum(0))
            self._W[k] = torch.linalg.solve(A + gamma * torch.eye(513), torch.stack(cols, 1))
        return self._W[k]

    def input(self, fold, order, split, g, seen):
        if self.inp is None:
            return aug(g["mean_vec"])
        src, q = self.inp
        key = subset_key(seen) if split == "test" else ALL
        return aug(torch.cat([self.C5.vec(fold, split, p, src, q, key) for p in seen]))

    def router(self, gamma):
        def f(fold, order, split, g, seen):
            pos = [self.d.tasks.index(x) for x in ORDERS[order]][:len(seen)]
            assert sorted(pos) == sorted(seen)
            W = self.W(fold, order, len(seen), gamma)                # 欄依該序的已見任務
            return self.input(fold, order, split, g, seen) @ W[:, [pos.index(p) for p in seen]]
        return f


def lin8_eval(d, RG, gamma, order, fold, split="test", stages=(1, 2, 3, 4)):
    pos = [d.tasks.index(x) for x in ORDERS[order]]
    out = {}
    for t in stages:
        seen = pos[:t]
        g = N2.gather_r(d, fold, split, seen)
        logits = aug(g["mean_vec"]) @ RG.W(fold, order, t, gamma, lin8=True)        # 欄 = seen 依序 × 2 類
        cls = torch.tensor([rr for p in seen for rr in task_rows(p)])
        pred = cls[logits.argmax(-1)]
        col_true = torch.stack([2 * torch.tensor([seen.index(int(p)) for p in g["task"]]),
                                2 * torch.tensor([seen.index(int(p)) for p in g["task"]]) + 1], -1)
        mk_local = logits.gather(1, col_true).argmax(-1)
        mk = 2 * g["task"] + mk_local
        out[t] = (pred, mk, g)
    return out


def main() -> int:
    t0 = time.perf_counter()
    d = N2.D2()
    C5 = Ctx5(d)
    M1 = json.loads((d.out / "nc1" / "metrics.json").read_text())
    M = {}

    # ── A：選法 ────────────────────────────────────────────────────────────
    val = {}
    for c in CANDS:
        val[c] = {o: [val_acc(d, C5, None, c, o, f) for f in FOLDS] for o in ORDERS}
    val_mean = {c: mean_sd(val[c]["reverse"] + val[c]["paper"])[0] for c in CANDS}
    best = max(val_mean.values())
    selected = next(c for c in CANDS if val_mean[c] == best)

    test = {c: evaluate(d, C5, None, c) for c in CANDS}
    for c in CANDS:
        for o in ORDERS:
            diffs = [a["acc"] - b["acc"] for a, b in zip(test[c][o]["cil"], test["基準"][o]["cil"])]
            test[c][o]["diff"] = {"per_fold": diffs, "mean": mean_sd(diffs)[0], "wins": sum(x > 0 for x in diffs)}
    ctx_pass = selected != "基準" and all(test[selected][o]["diff"]["mean"] >= 0.01 and
                                        test[selected][o]["diff"]["wins"] >= 7 for o in ORDERS)

    # ── B：機制診斷 ──────────────────────────────────────────────────────────
    fixed = {}
    if selected != "基準":
        for o in ORDERS:
            tf = tb = cf = cb = 0
            for (th_s, ok_s, g), (th_b, ok_b, _) in zip(test[selected][o]["per"], test["基準"][o]["per"]):
                ts, tbs = th_s == g["task"], th_b == g["task"]
                tf += int((ts & ~tbs).sum()); tb += int((~ts & tbs).sum())
                cf += int((ok_s & ~ok_b).sum()); cb += int((~ok_s & ok_b).sum())
            fixed[o] = {"tp_fixed": tf, "tp_broken": tb, "cil_fixed": cf, "cil_broken": cb}
    th_router = {f"TH-{s}": evaluate(d, C5, None, f"TH-{s}") for s in SOURCES}
    frac, ustat = {s: [] for s in SOURCES}, {s: [[], []] for s in SOURCES}
    for f in FOLDS:
        fr = torch.cat([C5.c(f, "test", t)["frac_bg05"] for t in d.tasks])
        us = torch.cat([C5.c(f, "test", t)["u_stat"] for t in d.tasks])
        for i, s in enumerate(SOURCES):
            frac[s].append(fr[:, i].mean().item())
            ustat[s][0].append(us[:, i, 0].mean().item()); ustat[s][1].append(us[:, i, 1].mean().item())

    # ── C：AR、2×2、LIN8 ─────────────────────────────────────────────────────
    RG = Ridge(d, C5)
    ar_val = {g_: [] for g_ in GAMMAS}
    for g_ in GAMMAS:
        fn = RG.router(g_)
        for o in ORDERS:
            for f in FOLDS:
                ar_val[g_].append(val_acc(d, C5, fn, "AR", o, f))
    ar_best = max(mean_sd(ar_val[g_])[0] for g_ in GAMMAS)
    ar_gamma = min(g_ for g_ in GAMMAS if mean_sd(ar_val[g_])[0] == ar_best)
    check = {}
    for g_ in GAMMAS:
        X = [aug(d.c(1, "train", t)["mean_vec"]) for t in d.tasks]
        Xa = torch.cat(X)
        W1 = torch.linalg.solve(Xa.t() @ Xa + g_ * torch.eye(513), torch.stack([x.sum(0) for x in X], 1))
        for o in ORDERS:
            pos = [d.tasks.index(x) for x in ORDERS[o]]
            Wi = RG.W(1, o, 4, g_)
            Wc = torch.zeros_like(Wi)
            for j, p in enumerate(pos):
                Wc[:, p] = Wi[:, j]
            check[f"{g_}|{o}"] = float((Wc - W1).abs().max())
    ar_check_ok = all(check[f"{ar_gamma}|{o}"] < 1e-4 for o in ORDERS)
    if not ar_check_ok:
        (d.out / "nc5").mkdir(exist_ok=True)
        (d.out / "nc5" / "ar_check_failed.json").write_text(json.dumps(check, indent=1))
        print(f"⚠️ AR 一致性檢查未達 < 1e-4（γ = {ar_gamma}）：{check} —— 停下回報。")
        return 7
    ar_test = evaluate(d, C5, RG.router(ar_gamma), "AR")
    two = {}
    if selected != "基準":
        s, q = parse(selected)
        RGb = Ridge(d, C5, (s, q))
        two_val = {g_: [val_acc(d, C5, RGb.router(g_), "AR", o, f) for o in ORDERS for f in FOLDS] for g_ in GAMMAS}
        gb = max(GAMMAS, key=lambda g_: (round(mean_sd(two_val[g_])[0], 12), -g_))
        two = {"gamma": gb, "val": two_val, "test": evaluate(d, C5, RGb.router(gb), "AR-bg")}

    l8_val = {g_: [] for g_ in GAMMAS}
    for g_ in GAMMAS:
        for o in ORDERS:
            for f in FOLDS:
                pred, _, g = lin8_eval(d, RG, g_, o, f, "val", (4,))[4]
                ok = pred == g["labels"]
                l8_val[g_].append(sum(ok[g["task"] == p].float().mean().item() for p in range(4)) / 4)
    l8_best = max(mean_sd(l8_val[g_])[0] for g_ in GAMMAS)
    l8_gamma = min(g_ for g_ in GAMMAS if mean_sd(l8_val[g_])[0] == l8_best)
    lin8 = {}
    for o in ORDERS:
        lin8[o] = []
        for f in FOLDS:
            st = lin8_eval(d, RG, l8_gamma, o, f)
            lin8[o].append(cil_full(lambda seen, st=st: st[len(seen)], o, d.tasks))

    # ── D：主系統（R3(k=8) ＋ L1(r=2) v2）───────────────────────────────────
    main_sys = {}
    for o in ORDERS:
        evs = {f: torch.load(d.out / "lora_v2" / "r2" / o / f"fold{f}_eval.pt", map_location="cpu") for f in FOLDS}
        cil = []
        for f in FOLDS:
            def stage(seen, f=f, o=o, evs=evs):
                g = N2.gather_r(d, f, "test", seen)
                th = torch.tensor(seen)[N2.scores_for(d, f, g, seen, "R3", 8).argmax(-1)]
                four = torch.cat([evs[f]["tasks"][d.tasks[p]]["l1_cos8"] for p in seen])
                pred, mk = N2.hard(four, th, g["task"])
                return pred, mk, g
            cil.append(cil_full(stage, o, d.tasks))
        main_sys[o] = cil
    bwt_check = {o: [abs(x["bwt"] + x["forgetting"]) < 1e-9 for x in main_sys[o]] for o in ORDERS}
    lin8_vs = {}
    for o in ORDERS:
        diffs = [a["acc"] - b["acc"] for a, b in zip(lin8[o], main_sys[o])]
        lin8_vs[o] = {"per_fold": diffs, "mean": mean_sd(diffs)[0], "wins": sum(x > 0 for x in diffs)}

    # ── metrics ────────────────────────────────────────────────────────────
    strip = lambda r: {o: {"cil": [{k: x[k] for k in ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t")}  # noqa: E731
                                  for x in v["cil"]], "diag": v["diag"],
                          **({"diff": v["diff"]} if "diff" in v else {})} for o, v in r.items()}
    M = {"val": val, "val_mean": val_mean, "selected": selected, "ctx_pass": ctx_pass,
         "test": {c: strip(test[c]) for c in CANDS}, "fixed_broken": fixed,
         "tumor_half": {k: strip(v) for k, v in th_router.items()}, "frac_bg05": frac, "u_stat": ustat,
         "ar": {"val": {str(k): v for k, v in ar_val.items()}, "gamma": ar_gamma, "check_fold1": check,
                "test": strip(ar_test)},
         "ar_bg": ({"gamma": two["gamma"], "val": {str(k): v for k, v in two["val"].items()},
                    "test": strip(two["test"])} if two else None),
         "lin8": {"val": {str(k): v for k, v in l8_val.items()}, "gamma": l8_gamma,
                  "test": {o: [{k: x[k] for k in ("acc", "masked", "forgetting", "acc_t")} for x in v] for o, v in lin8.items()},
                  "vs_main": lin8_vs},
         "main_system": {o: [{k: x[k] for k in ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t", "R")}
                             for x in v] for o, v in main_sys.items()}, "bwt_check": bwt_check}
    tp = d.out / "nc5" / "timing_cache.json"
    M["timing"] = json.loads(tp.read_text()) if tp.exists() else {}
    (d.out / "nc5").mkdir(exist_ok=True)
    (d.out / "nc5" / "metrics.json").write_text(json.dumps(M, indent=1, default=str))
    write(d, M, test, th_router, ar_test, two, lin8, main_sys, time.perf_counter() - t0)
    print(f"→ {d.out / 'REPORT_stage7.md'}")
    return 0


def acc_col(r, o, k="acc"):
    return [x[k] for x in r[o]["cil"]]


def write(d, M, test, th_router, ar_test, two, lin8, main_sys, t_rep):
    ok = lambda b: "**通過**" if b else "**未通過**"                     # noqa: E731
    sel = M["selected"]
    ar_g = M["ar"]["gamma"]
    out = ["# REPORT — NC-5：背景分派 ctx、機制診斷、對照組 AR／LIN8、主系統指標（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；所有數字來自同一台、"
           "同一批；全部只做推論。判準見 `PREREG-5.md`（commit a219456）。", "",
           "## PREREG-5 判準落點", "", "| 判準 | 數值 | 所在表格 | 結果 |", "|---|---|---|---|",
           f"| 選法（validation t = 4 CIL，c、e 取兩序平均） | 選出 **{sel}**（{M['val_mean'][sel]:.4f}；基準 "
           f"{M['val_mean']['基準']:.4f}） | T1 | — |"]
    if sel != "基準":
        for o in ORDERS:
            dd = test[sel][o]["diff"]
            out.append(f"| ctx-pass（{o}）：test 差 ≥ +0.01 且贏 ≥ 7/10 | {dd['mean']:+.4f}、{dd['wins']}/10 | T1 | "
                       f"{ok(dd['mean'] >= 0.01 and dd['wins'] >= 7)} |")
    out += [f"| ctx-pass：每任務儲存 ≤ 16,384 bytes | 16,384（z 另有共用文字向量 4,096，不計入） | T1 | 符合 |",
            f"| **ctx-pass 整體**（選出者 ≠ 基準、兩序皆成立） | | T1 | {ok(M['ctx_pass'])} |",
            f"| B、C、D | 不設門檻，必報 | T2–T5 | 已報 |",
            f"| C：AR 一致性檢查（fold 1、t = 4、γ = {ar_g}） | "
            + "、".join(f"{o} {M['ar']['check_fold1'][f'{ar_g}|{o}']:.2e}" for o in ORDERS)
            + " | T4 | < 1e-4 **符合** |",
            f"| D：每折 BWT = −Forgetting | reverse {sum(M['bwt_check']['reverse'])}/10、paper {sum(M['bwt_check']['paper'])}/10 | "
            f"T5 | {'全部成立' if all(M['bwt_check']['reverse'] + M['bwt_check']['paper']) else '有不成立的折'} |", ""]

    # T1
    out += ["## T1 候選總表（L0 expert、Hard）", "",
            "| 候選 | validation reverse | validation paper | validation 平均 | test ACC reverse | test ACC paper | "
            "對基準差 reverse（贏） | 對基準差 paper（贏） | Forgetting reverse／paper | 儲存 bytes／任務 |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CANDS:
        r = test[c]
        by = "16,384" + ("（＋共用 4,096）" if c.startswith("z") else "")
        out.append(f"| {c}{' ★' if c == sel else ''} | {fmt(M['val'][c]['reverse'])} | {fmt(M['val'][c]['paper'])} | "
                   f"{M['val_mean'][c]:.4f} | {fmt(acc_col(r, 'reverse'))} | {fmt(acc_col(r, 'paper'))} | "
                   f"{r['reverse']['diff']['mean']:+.4f}（{r['reverse']['diff']['wins']}/10） | "
                   f"{r['paper']['diff']['mean']:+.4f}（{r['paper']['diff']['wins']}/10） | "
                   f"{mean_sd(acc_col(r, 'reverse', 'forgetting'))[0]:.4f}／{mean_sd(acc_col(r, 'paper', 'forgetting'))[0]:.4f} | {by} |")
    out += ["", "候選名稱 = 來源 + q（例：c0.5 = 已見類別文字、q = 0.5）。★ = validation 選出者。", ""]

    # T2
    def diag_row(name, r, o):
        dg = r[o]["diag"]
        return (f"| {name} | {o} | " + " | ".join(fmt([x['tp_task'][p] for x in dg]) for p in range(4))
                + f" | {fmt([x['macro'] for x in dg])} | {fmt([x['micro'] for x in dg])} | {sum(x['e2l'] for x in dg)} | "
                f"{sum(x['l2e'] for x in dg)} | {fmt(acc_col(r, o))} |")
    out += ["## T2 每任務 TP 與 ESCA↔Lung 混淆（test、t = 4、十折）", "",
            "| router | 序 | " + " | ".join(f"TP {t.split('_')[1]}" for t in d.tasks)
            + " | TP macro | TP micro | ESCA→Lung | Lung→ESCA | CIL ACC |", "|---|---|" + "---|" * 4 + "---|---|---|---|---|"]
    for c in CANDS:
        for o in ORDERS:
            if o == "paper" and c[0] in ("基", "z"):
                continue
            out.append(diag_row(c, test[c], o))
    out += ["", "基準與 z 的 key、已見集合與順序無關，只列 reverse（paper 相同）。", ""]

    # T3
    def cls_counts(r, o):
        dg = r[o]["diag"]
        return ({k: sum(x["l2e_cls"][k] for x in dg) for k in ("LUAD", "LUSC")},
                {k: sum(x["e2l_cls"][k] for x in dg) for k in ("ESAD", "ESCC")})
    out += ["## T3 機制診斷", "", "### T3-a 混淆的真實類別（test、t = 4、十折合計）", "",
            "| router | 序 | Lung→ESCA：LUAD | LUSC | ESCA→Lung：ESAD | ESCC |", "|---|---|---|---|---|---|"]
    rows = [("基準", test["基準"])] + ([(sel, test[sel])] if sel != "基準" else [])
    for name, r in rows:
        for o in ORDERS:
            l, e = cls_counts(r, o)
            out.append(f"| {name} | {o} | {l['LUAD']} | {l['LUSC']} | {e['ESAD']} | {e['ESCC']} |")
    if M["fixed_broken"]:
        out += ["", f"### T3-b {sel} 相對基準修好／弄壞（test、t = 4、十折合計）", "",
                "| 序 | TP 修好 | TP 弄壞 | CIL 修好 | CIL 弄壞 |", "|---|---|---|---|---|"]
        for o, v in M["fixed_broken"].items():
            out.append(f"| {o} | {v['tp_fixed']} | {v['tp_broken']} | {v['cil_fixed']} | {v['cil_broken']} |")
    else:
        out += ["", "### T3-b 修好／弄壞", "", "選出者為基準，依操作定義 5 不報。"]
    out += ["", "### T3-c 腫瘤半向量當 router（key 規格同 ctx；test、t = 4）", "",
            "| 來源 | 序 | TP micro | ESCA→Lung | Lung→ESCA | CIL ACC |", "|---|---|---|---|---|---|"]
    for s in SOURCES:
        r = th_router[f"TH-{s}"]
        for o in ORDERS:
            if o == "paper" and s == "z":
                continue
            dg = r[o]["diag"]
            out.append(f"| {s} | {o} | {fmt([x['micro'] for x in dg])} | {sum(x['e2l'] for x in dg)} | "
                       f"{sum(x['l2e'] for x in dg)} | {fmt(acc_col(r, o))} |")
    out += ["", "### T3-d 真實任務 L0 expert 四輪 64 張落在 q = 0.5 背景集合的比例；u 統計（test、t = 4、十折平均）", "",
            "| 來源 | 落在背景的比例 | u 平均 | u 標準差 |", "|---|---|---|---|"]
    for s in SOURCES:
        out.append(f"| {s} | {fmt(M['frac_bg05'][s])} | {mean_sd(M['u_stat'][s][0])[0]:.4f} | {mean_sd(M['u_stat'][s][1])[0]:.4f} |")
    out += ["", "隨機選片的期望比例為 0.5。", ""]

    # T4
    ar = M["ar"]
    out += ["## T4 對照組", "", f"### T4-a AR 解析式 router（γ\\* = {ar['gamma']}；L0 expert、Hard）", "",
            "| γ | validation t = 4 CIL（兩序十折） | fold 1 一致性（reverse／paper） |", "|---|---|---|"]
    for g_ in GAMMAS:
        out.append(f"| {g_} | {fmt(ar['val'][str(g_)])} | {ar['check_fold1'][f'{g_}|reverse']:.2e}／{ar['check_fold1'][f'{g_}|paper']:.2e} |")
    out += ["", "| router | 序 | " + " | ".join(f"TP {t.split('_')[1]}" for t in d.tasks)
            + " | TP macro | TP micro | ESCA→Lung | Lung→ESCA | CIL ACC |", "|---|---|" + "---|" * 4 + "---|---|---|---|---|"]
    for o in ORDERS:
        out.append(diag_row("AR", ar_test, o))
    out += ["", f"儲存：A 共用 1,052,676 bytes；每任務 b_j 2,052 bytes。", ""]
    if two:
        out += [f"### T4-b 2 × 2（router × 輸入；test t = 4 CIL ACC，兩序）", "",
                f"AR ＋ 背景（{sel}）的 γ = {two['gamma']}（validation 選）。", "",
                "| router ＼ 輸入 | 全部平均 reverse | 全部平均 paper | 背景 reverse | 背景 paper |", "|---|---|---|---|---|",
                f"| R3(k=8) | {fmt(acc_col(test['基準'], 'reverse'))} | {fmt(acc_col(test['基準'], 'paper'))} | "
                f"{fmt(acc_col(test[sel], 'reverse'))} | {fmt(acc_col(test[sel], 'paper'))} |",
                f"| AR | {fmt(acc_col(ar_test, 'reverse'))} | {fmt(acc_col(ar_test, 'paper'))} | "
                f"{fmt(acc_col(two['test'], 'reverse'))} | {fmt(acc_col(two['test'], 'paper'))} |", "",
                "| AR ＋ 背景 | 序 | TP 與混淆 |", "|---|---|---|"]
        for o in ORDERS:
            dg = two["test"][o]["diag"]
            out.append(f"| AR ＋ {sel} | {o} | TP micro {fmt([x['micro'] for x in dg])}、ESCA→Lung "
                       f"{sum(x['e2l'] for x in dg)}、Lung→ESCA {sum(x['l2e'] for x in dg)} |")
        out.append("")
    else:
        out += ["### T4-b 2 × 2", "", "A 選出的是基準，依 PREREG-5 不做。", ""]
    l8 = M["lin8"]
    out += [f"### T4-c LIN8 8 類線性基線（γ\\* = {l8['gamma']}；不用 expert、不用文字頭）", "",
            "| γ | validation t = 4 ACC（兩序十折） |", "|---|---|"]
    for g_ in GAMMAS:
        out.append(f"| {g_} | {fmt(l8['val'][str(g_)])} |")
    out += ["", "| 系統 | 序 | ACC | Masked ACC | Forgetting | LIN8 − 完整系統（十折平均） | LIN8 贏的折數 |",
            "|---|---|---|---|---|---|---|"]
    for o in ORDERS:
        a, b = lin8[o], main_sys[o]
        v = l8["vs_main"][o]
        out.append(f"| LIN8 | {o} | {fmt([x['acc'] for x in a])} | {fmt([x['masked'] for x in a])} | "
                   f"{fmt([x['forgetting'] for x in a])} | {v['mean']:+.4f} | {v['wins']}/10 |")
        out.append(f"| 完整系統 R3(k=8)＋L1(r=2) v2 | {o} | {fmt([x['acc'] for x in b])} | {fmt([x['masked'] for x in b])} | "
                   f"{fmt([x['forgetting'] for x in b])} | | |")
    out += ["", "LIN8 每折差（test t = 4 ACC，LIN8 − 完整系統）：", "", "| 序 | " + " | ".join(str(f) for f in FOLDS) + " |",
            "|---|" + "---|" * 10]
    for o in ORDERS:
        out.append(f"| {o} | " + " | ".join(f"{x:+.4f}" for x in l8["vs_main"][o]["per_fold"]) + " |")

    # T5
    out += ["", "## T5 主系統 R3(k=8)＋L1(r=2) v2 逐階段（test，十折 mean ± sd）", "",
            "| 序 | 指標 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|"]
    for o in ORDERS:
        ms = main_sys[o]
        out.append(f"| {o} | ACC | " + " | ".join(fmt([x["acc_t"][t] for x in ms]) for t in range(4)) + " |")
        out.append(f"| {o} | Masked ACC | " + " | ".join(fmt([x["masked_t"][t] for x in ms]) for t in range(4)) + " |")
    out += ["", "| 序 | BWT | Forgetting | 每折 BWT = −Forgetting |", "|---|---|---|---|"]
    for o in ORDERS:
        ms = main_sys[o]
        out.append(f"| {o} | {fmt([x['bwt'] for x in ms])} | {fmt([x['forgetting'] for x in ms])} | "
                   f"{sum(M['bwt_check'][o])}/10 |")

    tm = M["timing"]
    out += ["", "## T6 實際耗時（秒，wall clock；執行緒 8）", "", "| 項目 | 秒 |", "|---|---|",
            f"| NC-5 快取（十折 train／val／test，三來源、三個 q、腫瘤半、比例與 u 統計） | {tm.get('run/1-10', float('nan')):,.0f} |",
            f"| 報告（key 的 KMeans、AR／LIN8 求解、所有評估） | {t_rep:,.0f} |", ""]
    (d.out / "REPORT_stage7.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
