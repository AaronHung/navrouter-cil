#!/usr/bin/env python3
"""NC-2 報告：R 線、L 線、完整系統列（只讀快取與 lora 評估檔，不碰特徵檔）。

    NAVCIL_MACHINE=mac python scripts/nc2_report.py
輸出：outputs/navcil/<machine>/REPORT_stage4.md、nc2/metrics.json
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

import nc1_report as N1                                                   # noqa: E402
from selector.cil_eval import mean_sd, subset_key                         # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402
from selector.router import (R_VARIANTS, key_evidence_proto, key_multi_proto,  # noqa: E402
                             key_proto, key_text_proto, r_scores, tp_pred)

K_GRID = (2, 4, 8)
R_LIST = (1, 2, 4, 8)
FOLDS = list(range(1, 11))
BYTES_KEY = {"R0": 0, "R1": 0, "R2": 2048, "R4": 2048, "R5": 2048}
FULL_PARAMS = 132_097
fmt = N1.fmt


def lora_params(r: int) -> int:
    return 770 * r + 513                     # A r×514 + B 256×r + b1 256 + fc2 257


class D2(N1.Data):
    def __init__(self):
        super().__init__(FOLDS)
        self._n2, self._keys = {}, {}
        self.F = torch.cat([torch.load(REPO_ROOT / "cache" / "text" / f"f_txt_{t}.pt",
                                       map_location="cpu")["f_txt"] for t in self.tasks])

    def n2(self, fold, split, task):
        k = (fold, split, task)
        if k not in self._n2:
            self._n2[k] = torch.load(self.cache / f"nc2_fold{fold}_{split}_{task}.pt",
                                     map_location="cpu")
        return self._n2[k]

    def keys(self, fold) -> dict:
        if fold not in self._keys:
            mv = [self.c(fold, "train", t)["mean_vec"] for t in self.tasks]
            zb = [self.n2(fold, "train", t)["four_zbar"] for t in self.tasks]
            self._keys[fold] = {
                "text_proto": key_text_proto(self.F), "proto": key_proto(mv),
                "evidence_proto": key_evidence_proto(zb),
                "multi_proto_k": {k: key_multi_proto(mv, k) for k in K_GRID}}
        return self._keys[fold]


def gather_r(d: D2, fold, split, seen) -> dict:
    si = None
    parts = []
    for p in seen:
        t = d.tasks[p]
        n2 = d.n2(fold, split, t)
        base = n2 if split == "val" else d.c(fold, "test", t)
        si = base["subsets"].index(subset_key(seen))
        parts.append((p, n2, base))
    cat = lambda f: torch.cat([f(p, n2, b) for p, n2, b in parts])      # noqa: E731
    return {"labels": cat(lambda p, n2, b: b["labels"]),
            "task": cat(lambda p, n2, b: torch.full((len(b["labels"]),), p)),
            "mean_vec": cat(lambda p, n2, b: b["mean_vec"]),
            "zbar": cat(lambda p, n2, b: n2["four_zbar"]),
            "four": cat(lambda p, n2, b: b["four_cos8_uni"]),
            "vote_counts": cat(lambda p, n2, b: b["vote_counts"][:, si]),
            "vote_msum": cat(lambda p, n2, b: b["vote_msum"][:, si])}


def scores_for(d, fold, g, seen, variant, k=None):
    keys = dict(d.keys(fold))
    if variant == "R3":
        keys["multi_proto"] = keys["multi_proto_k"][k]
    return r_scores(variant, g, seen, keys)


def hard(four: torch.Tensor, th: torch.Tensor, task: torch.Tensor):
    """Hard：選中 expert τ̂ 的 8 類 cosine，在 C_τ̂ 取 argmax；另回傳真實任務內的 masked 預測。"""
    z = four[torch.arange(len(th)), th]
    rows = torch.stack([2 * th, 2 * th + 1], -1)
    pred = rows.gather(1, z.gather(1, rows).argmax(-1, keepdim=True)).squeeze(-1)
    rt = torch.stack([2 * task, 2 * task + 1], -1)
    masked = rt.gather(1, z.gather(1, rt).argmax(-1, keepdim=True)).squeeze(-1)
    return pred, masked


def cil_from(stage_fn, order, tasks_all) -> dict:
    """stage_fn(seen) → (pred, masked, g)；回傳 R、ACC、Masked ACC、Forgetting。"""
    pos = [tasks_all.index(t) for t in ORDERS[order]]
    R = [[None] * 4 for _ in range(4)]
    masked = None
    for t in range(1, 5):
        pred, mk, g = stage_fn(pos[:t])
        ok = pred == g["labels"]
        for j in range(t):
            R[t - 1][j] = ok[g["task"] == pos[j]].float().mean().item()
        if t == 4:
            masked = sum((mk == g["labels"])[g["task"] == p].float().mean().item() for p in pos) / 4
    return {"R": R, "acc": sum(R[3]) / 4, "masked": masked,
            "forgetting": sum(max(R[t][j] for t in range(j, 3)) - R[3][j] for j in range(3)) / 3}


def margin_summary(scores: torch.Tensor, correct: torch.Tensor) -> dict:
    s = scores.float().sort(-1, descending=True).values
    m = s[:, 0] - s[:, 1]
    def q(x):
        if x.numel() == 0:
            return None
        return {"mean": x.mean().item(), "median": x.median().item(),
                "p10": x.quantile(0.1).item(), "p90": x.quantile(0.9).item(), "n": int(x.numel())}
    return {"all": q(m), "correct": q(m[correct]), "wrong": q(m[~correct])}


# ── T0(b) ────────────────────────────────────────────────────────────────────
def t0b(d: D2) -> dict:
    out = {}
    for v in N1.TP_ALL:
        per_task = {p: [] for p in range(4)}
        mac, mic = {}, {}
        for order, names in ORDERS.items():
            pos = [d.tasks.index(t) for t in names]
            for t in range(1, 5):
                seen = pos[:t]
                for f in FOLDS:
                    g = N1.gather(d, f, "test", seen)
                    ok = tp_pred(d, f, g, seen, v) == g["task"]
                    accs = [ok[g["task"] == p].float().mean().item() for p in seen]
                    mac.setdefault((order, t), []).append(sum(accs) / len(accs))
                    mic.setdefault((order, t), []).append(ok.float().mean().item())
                    if t == 4 and order == "reverse":
                        for p, a in zip(seen, accs):
                            per_task[p].append(a)
        out[v] = {"per_task_t4": per_task, "macro": mac, "micro": mic}
    tc = mean_sd(out["text-class"]["macro"][("reverse", 4)] + out["text-class"]["macro"][("paper", 4)])[0]
    nv = mean_sd(out["nav"]["macro"][("reverse", 4)] + out["nav"]["macro"][("paper", 4)])[0]
    pr = mean_sd(out["proto"]["macro"][("reverse", 4)] + out["proto"]["macro"][("paper", 4)])[0]
    if tc >= 0.97:
        verdict = "TP-text-class ≥ 0.97 → N1 降為消融，第四關主做 N2"
    elif tc <= 0.95 and nv >= tc + 0.03 and nv >= pr - 0.005:
        verdict = "TP-text-class ≤ 0.95、TP-nav ≥ TP-text-class + 0.03 且 TP-nav ≥ TP-proto − 0.005 → N1 為主線"
    else:
        verdict = "其他 → N1 列為消融，主做 N2"
    return {"tables": out, "macro_t4": {"text-class": tc, "nav": nv, "proto": pr},
            "macro_verdict": verdict}


def main() -> int:
    t0 = time.perf_counter()
    d = D2()
    M1 = json.loads((d.out / "nc1" / "metrics.json").read_text())
    M = {}

    # ── T0 ──────────────────────────────────────────────────────────────────
    T0b = t0b(d)
    M["t0b"] = {"macro_t4": T0b["macro_t4"], "macro_verdict": T0b["macro_verdict"],
                "nc1_micro_verdict": M1["gate2"]["verdict"]}
    thard = json.loads((d.out / "nc2" / "thard_check.json").read_text())

    # ── T1 R 線 ─────────────────────────────────────────────────────────────
    variants = [("R0", None), ("R1", None), ("R2", None)] + [("R3", k) for k in K_GRID] + \
               [("R4", None), ("R5", None)]
    name = lambda v, k: f"R3(k={k})" if v == "R3" else v                 # noqa: E731
    val_acc = {}
    for v, k in variants:
        accs = []
        for f in FOLDS:
            seen = list(range(4))
            g = gather_r(d, f, "val", seen)
            th = torch.tensor(seen)[scores_for(d, f, g, seen, v, k).argmax(-1)]
            pred, _ = hard(g["four"], th, g["task"])
            ok = pred == g["labels"]
            accs.append(sum(ok[g["task"] == p].float().mean().item() for p in seen) / 4)
        val_acc[name(v, k)] = accs
    k_star = max(K_GRID, key=lambda k: (round(mean_sd(val_acc[f"R3(k={k})"])[0], 12), -k))
    cand = ["R0", "R1", "R2", f"R3(k={k_star})", "R4", "R5"]
    best_val = max(mean_sd(val_acc[c])[0] for c in cand)
    main_router = next(c for c in cand if mean_sd(val_acc[c])[0] == best_val)
    r01 = "R0" if mean_sd(val_acc["R0"])[0] >= mean_sd(val_acc["R1"])[0] else "R1"
    M["val_selection"] = {"val_acc": val_acc, "k_star": k_star, "main_router": main_router,
                          "better_R0_R1": r01}

    test = {}
    for v, k in variants:
        nm = name(v, k)
        res = {"cil": {}, "tp_t4_task": {p: [] for p in range(4)}, "tp_macro": [], "tp_micro": [],
               "conf": torch.zeros(4, 4, dtype=torch.long), "acc_t4": []}
        margins_s, margins_ok = [], []
        for order in ORDERS:
            per_fold = []
            for f in FOLDS:
                def stage(seen, f=f):
                    g = gather_r(d, f, "test", seen)
                    th = torch.tensor(seen)[scores_for(d, f, g, seen, v, k).argmax(-1)]
                    pred, mk = hard(g["four"], th, g["task"])
                    return pred, mk, g
                per_fold.append(cil_from(stage, order, d.tasks))
            res["cil"][order] = per_fold
        for f in FOLDS:
            seen = list(range(4))
            g = gather_r(d, f, "test", seen)
            sc = scores_for(d, f, g, seen, v, k)
            th = sc.argmax(-1)
            ok = th == g["task"]
            accs = [ok[g["task"] == p].float().mean().item() for p in seen]
            for p in seen:
                res["tp_t4_task"][p].append(accs[p])
            res["tp_macro"].append(sum(accs) / 4)
            res["tp_micro"].append(ok.float().mean().item())
            for a, b in zip(g["task"].tolist(), th.tolist()):
                res["conf"][a, b] += 1
            res["acc_t4"].append(res["cil"]["reverse"][f - 1]["acc"])
            ms = (g["vote_counts"].float() if v == "R1" else sc)
            margins_s.append(ms); margins_ok.append(ok)
        res["margin"] = margin_summary(torch.cat(margins_s), torch.cat(margins_ok))
        res["bytes"] = 2048 * k if v == "R3" else BYTES_KEY[v]
        test[nm] = res
    for nm, res in test.items():
        diffs = [a - b for a, b in zip(res["acc_t4"], test["R2"]["acc_t4"])]
        res["paired_vs_R2"] = {"diffs": diffs, "mean": mean_sd(diffs)[0],
                               "wins": sum(x > 0 for x in diffs)}
    zs = [r["acc"] for r in M1["cil"]["無 gate：zero-shot 8 類（top-64）|reverse"]]
    sel = test[main_router]
    r_pass = sel["bytes"] <= 16 * 1024 and mean_sd(sel["acc_t4"])[0] >= 0.890
    r_imp = sel["paired_vs_R2"]["mean"] >= 0.01 and sel["paired_vs_R2"]["wins"] >= 7
    ns_diff = [a - b for a, b in zip(test[r01]["acc_t4"], zs)]
    ns_pass = mean_sd(ns_diff)[0] >= 0.02 and sum(x > 0 for x in ns_diff) >= 7
    M["gates_R"] = {"main_router": main_router, "R_pass": r_pass, "R_improve": r_imp,
                    "no_storage": {"router": r01, "diffs": ns_diff, "pass": ns_pass,
                                   "zs8_top64": zs}}

    # ── T2/T3 L 線 ──────────────────────────────────────────────────────────
    L0 = M1["stage1"]["fold_mean"]["e_fourround"]
    L0_mean = mean_sd(L0)[0]
    L = {}
    for r in R_LIST:
        for order in ORDERS:
            files = [d.out / "lora" / f"r{r}" / order / f"fold{f}_eval.pt" for f in FOLDS]
            if not all(p.exists() for p in files):
                L[(r, order)] = None
                continue
            first = d.tasks.index(ORDERS[order][0])
            wp, wp_task, prs, jac, fro, jm = [], {p: [] for p in range(4)}, {}, {}, {}, {}
            l2 = []
            evs = [torch.load(p, map_location="cpu") for p in files]
            for ev in evs:
                accs = []
                for p, t in enumerate(d.tasks):
                    e = ev["tasks"][t]
                    rr = torch.tensor(task_rows(p))
                    a = (rr[e["l1_cos8"][:, p][:, rr].argmax(-1)] == e["labels"]).float().mean().item()
                    accs.append(a); wp_task[p].append(a)
                    if p != first:
                        prs.setdefault(t, []).append(e["pearson_l0"].mean().item())
                        jac.setdefault(t, []).append(e["jaccard_l0"].mean().item())
                    jm.setdefault(t, []).append(e["jaccard_m_l1"].mean().item())
                wp.append(sum(accs) / 4)
                for t, x in ev["fro_BA"].items():
                    fro.setdefault(t, []).append(x)

                def stage(seen, ev=ev):
                    ti = len(seen) - 1
                    labs = torch.cat([ev["tasks"][d.tasks[p]]["labels"] for p in seen])
                    task = torch.cat([torch.full((len(ev["tasks"][d.tasks[p]]["labels"]),), p)
                                      for p in seen])
                    c8 = torch.cat([ev["tasks"][d.tasks[p]]["m_cos8"][:, ti] for p in seen])
                    rows = torch.tensor([x for p in seen for x in task_rows(p)])
                    pred = rows[c8[:, rows].argmax(-1)]
                    rt = torch.stack([2 * task, 2 * task + 1], -1)
                    mk = rt.gather(1, c8.gather(1, rt).argmax(-1, keepdim=True)).squeeze(-1)
                    return pred, mk, {"labels": labs, "task": task}
                l2.append(cil_from(stage, order, d.tasks))
            n_nan = sum(1 for ev in evs for x in ev["fro_BA"].values() if x != x)
            L[(r, order)] = {"n_nan": n_nan, "n_inc": sum(len(ev["fro_BA"]) for ev in evs),
                             "wp": wp, "wp_task": wp_task, "pearson": prs, "jaccard_l0": jac,
                             "fro": fro, "jaccard_m_l1": jm, "l2": l2, "evs": evs,
                             "params": {t: (FULL_PARAMS if d.tasks.index(t) == first
                                            else lora_params(r)) for t in d.tasks}}
    passing = [r for r in R_LIST if all(L.get((r, o)) for o in ORDERS)
               and all(mean_sd(L[(r, o)]["wp"])[0] >= L0_mean - 0.01 for o in ORDERS)]
    l_pass_r = min(passing) if passing else None
    l_pass = l_pass_r is not None and l_pass_r <= 4
    if l_pass:
        r_star, r_star_note = l_pass_r, "通過 L-pass"
    else:
        done_r = [r for r in R_LIST if all(L.get((r, o)) for o in ORDERS)]
        r_star = max(done_r, key=lambda r: min(mean_sd(L[(r, o)]["wp"])[0] for o in ORDERS)) \
            if done_r else None
        r_star_note = "未通過 L-pass（操作定義 11 的備援）"
    invalid = [r for r in R_LIST if any(L.get((r, o)) and L[(r, o)]["n_nan"] for o in ORDERS)]
    provisional = any(r < (r_star or 99) for r in invalid)
    if provisional:
        r_star_note += f"；**暫定**：r = {invalid} 的訓練出現 NaN、結果無效，較小的 r 未能判定"
    M["gates_L"] = {"L0_mean": L0_mean, "passing_r": passing, "L_pass": l_pass,
                    "r_star": r_star, "note": r_star_note, "invalid_r": invalid,
                    "provisional": provisional}

    # ── T4 完整系統列 ────────────────────────────────────────────────────────
    full = {}
    mr_v, mr_k = (("R3", k_star) if main_router.startswith("R3") else (main_router, None))
    if r_star is not None:
        for order in ORDERS:
            evs = L[(r_star, order)]["evs"]
            per_fold = []
            for f, ev in zip(FOLDS, evs):
                def stage(seen, f=f, ev=ev):
                    g = gather_r(d, f, "test", seen)
                    th = torch.tensor(seen)[scores_for(d, f, g, seen, mr_v, mr_k).argmax(-1)]
                    four = torch.cat([ev["tasks"][d.tasks[p]]["l1_cos8"] for p in seen])
                    pred, mk = hard(four, th, g["task"])
                    return pred, mk, g
                per_fold.append(cil_from(stage, order, d.tasks))
            first = d.tasks.index(ORDERS[order][0])
            bytes_task = {t: (FULL_PARAMS if p == first else lora_params(r_star)) * 4
                          + test[main_router]["bytes"] for p, t in enumerate(d.tasks)}
            full[order] = {"cil": per_fold, "bytes": bytes_task}
    M["full_system"] = {o: {"acc": [x["acc"] for x in v["cil"]], "masked": [x["masked"] for x in v["cil"]],
                            "forgetting": [x["forgetting"] for x in v["cil"]], "bytes": v["bytes"]}
                        for o, v in full.items()}

    # ── 寫檔 ───────────────────────────────────────────────────────────────
    timing = {}
    for p in sorted((d.out / "nc2").glob("timing_*.json")):
        timing[p.stem] = json.loads(p.read_text())
    M["timing"] = timing
    M["test_R"] = {nm: {"acc_t4": r["acc_t4"], "tp_macro": r["tp_macro"], "tp_micro": r["tp_micro"],
                        "tp_t4_task": r["tp_t4_task"], "conf": r["conf"].tolist(),
                        "margin": r["margin"], "bytes": r["bytes"], "paired_vs_R2": r["paired_vs_R2"],
                        "cil": {o: [{k: x[k] for k in ("acc", "masked", "forgetting", "R")} for x in v]
                                for o, v in r["cil"].items()}} for nm, r in test.items()}
    M["L"] = {f"r{r}|{o}": (None if v is None else
                           {k: v[k] for k in ("n_nan", "n_inc", "wp", "wp_task", "pearson", "jaccard_l0", "fro",
                                              "jaccard_m_l1", "params")} |
                           {"l2": [{k: x[k] for k in ("acc", "masked", "forgetting", "R")} for x in v["l2"]]})
              for (r, o), v in L.items()}
    (d.out / "nc2" / "metrics.json").write_text(json.dumps(M, indent=1, default=str))
    write(d, M, T0b, thard, val_acc, test, L, full, L0, time.perf_counter() - t0)
    print(f"→ {d.out / 'REPORT_stage4.md'}")
    return 0


def write(d, M, T0b, thard, val_acc, test, L, full, L0, t_rep):
    ok = lambda b: "**通過**" if b else "**未通過**"                     # noqa: E731
    gR, gL = M["gates_R"], M["gates_L"]
    vs = M["val_selection"]
    mr = gR["main_router"]
    sel = test[mr]
    L0m = gL["L0_mean"]
    out = ["# REPORT — NC-2 第四關：R 線與 L 線（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、`--device cpu`、torch 2.11.0；所有數字來自同一台、同一批。"
           "判準見 `PREREG-2.md`（commit f79ce66）。expert = 第一關 bank，四輪 K=64、每輪 16、λ\\*=1.5，Hard。", "",
           "## PREREG-2 判準落點", "", "| 判準 | 數值 | 所在表格 | 結果 |", "|---|---|---|---|",
           f"| R3 的 k（validation 平均 CIL） | k\\* = {vs['k_star']} | T1-a | — |",
           f"| 主 router（validation 平均 CIL） | **{mr}**（{mean_sd(val_acc[mr])[0]:.4f}） | T1-a | — |",
           f"| R-pass：儲存 ≤ 16 KB／任務、不存 slide 資料 | {sel['bytes']:,} bytes／任務；只存平均向量 | T1-b「儲存」欄 | "
           f"{ok(sel['bytes'] <= 16384)} |",
           f"| R-pass：test CIL ACC 十折平均 ≥ 0.890 | {mean_sd(sel['acc_t4'])[0]:.4f} | T1-b「{mr}」列 | "
           f"{ok(mean_sd(sel['acc_t4'])[0] >= 0.890)} |",
           f"| **R-pass 整體** | | | {ok(gR['R_pass'])} |",
           f"| R-改進：相對 R2 每折 paired 差 ≥ +0.01 且贏 ≥ 7/10 | 平均 {sel['paired_vs_R2']['mean']:+.4f}、"
           f"贏 {sel['paired_vs_R2']['wins']}/10 | T1-c | {ok(gR['R_improve'])} |",
           f"| 無儲存 router：{gR['no_storage']['router']}（R0、R1 中 validation 較好者）比 zero-shot 8 類 top-64 "
           f"高 ≥ 0.02 且贏 ≥ 7/10 | 平均 {mean_sd(gR['no_storage']['diffs'])[0]:+.4f}、贏 "
           f"{sum(x > 0 for x in gR['no_storage']['diffs'])}/10 | T1-d | {ok(gR['no_storage']['pass'])} |",
           f"| L-pass：最小 r 使兩序十折 WP ≥ L0 − 0.01（{L0m:.4f} − 0.01 = {L0m - 0.01:.4f}）且 r ≤ 4 | "
           f"符合 WP 條件的 r：{gL['passing_r'] or '無'}；無效（NaN）的 r：{gL['invalid_r'] or '無'} | T2-a | "
           f"{ok(gL['L_pass'])}" + (f"（r\\* = {gL['r_star']}）" if gL["L_pass"] else "")
           + ("，**暫定**" if gL["provisional"] else "") + " |",
           f"| 完整系統列（不設門檻） | {mr} + L1(r={gL['r_star']})；{gL['note']} | T4 | — |", ""]

    # T0
    out += ["## T0 前置", "", "### (a) 命名", "",
            "- `tests/test_no_banned_deps.py`：禁用清單移除 `router`（模組名 regex 同步移除），清單上其他禁用字不變。",
            "- 本 repo 原本**沒有**名為 `task_gate` 的模組；任務分派邏輯在 `scripts/nc1_report.py` 的 "
            "`tp_scores`／`tp_pred`，已原樣搬到 `selector/router.py`（NC-2 的 R0–R5 也在此）。搬移後重跑 NC-1 報告，"
            "`nc1/metrics.json` 與 `REPORT_stage1-3.md` 逐位元相同。",
            "- `AGENTS.md` 新增「命名原則」：expert（程式沿用 selector）、router；方法名稱不用 navigation、zero。",
            "- pytest：" + ((d.out / "nc2" / "pytest.txt").read_text().strip()
                           if (d.out / "nc2" / "pytest.txt").exists() else "（未記錄）") + "。", "",
            "### (b) 第二關 TP：每任務、macro、micro（test，十折 mean ± sd）", "",
            "t = 4 每任務 TP（兩序相同）：", "",
            "| 變體 | " + " | ".join(d.tasks) + " | macro | micro |", "|---|" + "---|" * 6]
    for v in N1.TP_ALL:
        tb = T0b["tables"][v]
        out.append(f"| {v} | " + " | ".join(fmt(tb["per_task_t4"][p]) for p in range(4)) +
                   f" | {fmt(tb['macro'][('reverse', 4)])} | {fmt(tb['micro'][('reverse', 4)])} |")
    out += ["", "各階段 macro／micro：", "",
            "| 變體 | 序 | t=2 macro | t=2 micro | t=3 macro | t=3 micro | t=4 macro | t=4 micro |",
            "|---|---|---|---|---|---|---|---|"]
    for v in N1.TP_ALL:
        tb = T0b["tables"][v]
        for o in ORDERS:
            out.append(f"| {v} | {o} | " + " | ".join(
                f"{fmt(tb['macro'][(o, t)])} | {fmt(tb['micro'][(o, t)])}" for t in (2, 3, 4)) + " |")
    mt = T0b["macro_t4"]
    out += ["", f"若第二關判讀改用 macro：text-class {mt['text-class']:.4f}、nav {mt['nav']:.4f}、"
            f"proto {mt['proto']:.4f} → 「{T0b['macro_verdict']}」；原判讀（micro）為「{M['t0b']['nc1_micro_verdict']}」→ "
            + ("**落點不變**" if T0b["macro_verdict"] == M["t0b"]["nc1_micro_verdict"] else "**落點改變**")
            + "。PREREG.md 未修改。", "",
            "### (c) T-hard PROVENANCE.md 的 SHA256 核對", "",
            f"`{thard['repo']}` @ `{thard['commit'][:7]}`，`{thard['file']}`。", "",
            "| 檔案 | 本 repo SHA256 | T-hard 所列 | 相同 |", "|---|---|---|---|"]
    for r in thard["f_txt"]:
        out.append(f"| cache/text/f_txt_{r['task']}.pt | `{r['local_sha256']}` | `{r['t_hard_sha256']}` | "
                   f"{'✓' if r['equal'] else '✗'} |")
    cp = thard["class_prompts"]
    out.append(f"| data/class_prompts.json | `{cp['local']}` | `{cp['t_hard']}` | {'✓' if cp['equal'] else '✗'} |")

    # T1
    out += ["", "## T1 R 線", "", "### T1-a validation 選法（十折 validation、t = 4、Hard 的 CIL ACC）", "",
            "| 變體 | validation CIL ACC | 備註 |", "|---|---|---|"]
    for nm, xs in val_acc.items():
        note = []
        if nm == f"R3(k={vs['k_star']})":
            note.append("R3 選定的 k")
        if nm == mr:
            note.append("**主 router**")
        if nm == vs["better_R0_R1"]:
            note.append("R0、R1 中較好者")
        out.append(f"| {nm} | {fmt(xs)} | {'；'.join(note)} |")
    out += ["", "### T1-b test 總表（每個變體一列）", "",
            "| 變體 | 儲存 bytes／任務 | TP macro（t=4） | TP micro（t=4） | reverse ACC | reverse Masked | "
            "reverse Forgetting | paper ACC | paper Masked | paper Forgetting | vs R2 paired 差 | 贏折數 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, r in test.items():
        cells = []
        for o in ORDERS:
            cells += [fmt([x["acc"] for x in r["cil"][o]]), fmt([x["masked"] for x in r["cil"][o]]),
                      fmt([x["forgetting"] for x in r["cil"][o]])]
        pr = r["paired_vs_R2"]
        out.append(f"| {nm}{' ★' if nm == mr else ''} | {r['bytes']:,} | {fmt(r['tp_macro'])} | {fmt(r['tp_micro'])} | "
                   + " | ".join(cells) + f" | {pr['mean']:+.4f} | {pr['wins']}/10 |")
    out += ["", "★ = 主 router。所有變體都只存對 train slides 平均（或 k-means）後的向量，不存逐張 slide 資料。",
            "", "每任務 TP（t = 4，十折 mean ± sd）：", "",
            "| 變體 | " + " | ".join(d.tasks) + " |", "|---|" + "---|" * 4]
    for nm, r in test.items():
        out.append(f"| {nm} | " + " | ".join(fmt(r["tp_t4_task"][p]) for p in range(4)) + " |")
    out += ["", "t = 4 混淆矩陣（十折合計；列 = 真實、欄 = 預測；**粗體** = ESCA↔Lung）：", ""]
    ie, il = d.tasks.index("tcga_esca"), d.tasks.index("tcga_lung")
    for nm, r in test.items():
        out += [f"**{nm}**", "", "| 真實 \\ 預測 | " + " | ".join(d.tasks) + " |", "|---|" + "---|" * 4]
        for i, t in enumerate(d.tasks):
            cells = []
            for j in range(4):
                x = int(r["conf"][i, j])
                cells.append(f"**{x}**" if {i, j} == {ie, il} else str(x))
            out.append(f"| {t} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["分派 margin（t = 4，第一名 − 第二名候選分數，各變體自己的單位；R1 以票數計）：", "",
            "| 變體 | 全部 mean | median | p10 | p90 | 判對 median | 判錯 median | 判錯張數 |",
            "|---|---|---|---|---|---|---|---|"]
    for nm, r in test.items():
        m = r["margin"]
        w = m["wrong"]
        out.append(f"| {nm} | {m['all']['mean']:.4f} | {m['all']['median']:.4f} | {m['all']['p10']:.4f} | "
                   f"{m['all']['p90']:.4f} | {m['correct']['median']:.4f} | "
                   f"{(w['median'] if w else float('nan')):.4f} | {(w['n'] if w else 0)} |")
    out += ["", "### T1-c 主 router 相對 R2 的每折 paired 差（test，t = 4 ACC）", "",
            "| fold | " + " | ".join(str(f) for f in FOLDS) + " | mean | 贏 |", "|---|" + "---|" * 12,
            f"| {mr} − R2 | " + " | ".join(f"{x:+.4f}" for x in sel["paired_vs_R2"]["diffs"]) +
            f" | {sel['paired_vs_R2']['mean']:+.4f} | {sel['paired_vs_R2']['wins']}/10 |", "",
            "### T1-d 無儲存 router 對 zero-shot 8 類 top-64（test，t = 4 ACC）", "",
            "| fold | " + " | ".join(str(f) for f in FOLDS) + " | mean | 贏 |", "|---|" + "---|" * 12,
            f"| {gR['no_storage']['router']} − zero-shot top-64 | " +
            " | ".join(f"{x:+.4f}" for x in gR["no_storage"]["diffs"]) +
            f" | {mean_sd(gR['no_storage']['diffs'])[0]:+.4f} | {sum(x > 0 for x in gR['no_storage']['diffs'])}/10 |"]

    # T2
    out += ["", "## T2 L 線", "", f"L0（完整 expert，第一關 (e)）四任務平均 WP = {fmt(L0)}。"
            "底座 = 該序第一個任務的完整 expert（reverse：ESCA；paper：LUNG）。", "",
            "### T2-a WP（test Masked ACC，四輪，十折 mean ± sd）", "",
            "| r | 序 | 四任務平均 WP | ≥ L0 − 0.01 | " + " | ".join(d.tasks) + " | 非底座任務參數量／任務 | NaN 增量數 |",
            "|---|---|---|---|" + "---|" * 4 + "---|---|"]
    for r in R_LIST:
        for o in ORDERS:
            v = L[(r, o)]
            if v is None:
                out.append(f"| {r} | {o} | 未完成 | | | | | | |")
                continue
            m = mean_sd(v["wp"])[0]
            out.append(f"| {r} | {o} | {fmt(v['wp'])} | {'✓' if m >= L0m - 0.01 else '✗'} | "
                       + " | ".join(fmt(v["wp_task"][p]) for p in range(4)) + f" | {lora_params(r):,} | "
                       + (f"**{v['n_nan']}/{v['n_inc']}**" if v["n_nan"] else f"0/{v['n_inc']}") + " |")
    out += ["", f"完整 expert 每任務 {FULL_PARAMS:,} 參數。", ""]
    if gL["invalid_r"]:
        out += [f"⚠️ **r = {gL['invalid_r']} 的結果無效**：訓練中增量參數變成 NaN（見「NaN 增量數」欄），"
                "該列的 WP、行為指標與 L2 都不可用。診斷（重播 fold 1 reverse RCC、r = 1）：NaN 只出現在 "
                "A 梯度的第 512、513 欄（text_nav_feats 的 2 維摘要輸入），同一步的 loss、上游梯度與手動 "
                "da^T·u 皆為有限值；是否出現取決於 CPU 執行緒數（預設 8 條：第 263 步；3 條：前 320 步無，原 run 用 3 條、在第 2 個 epoch 出現；"
                "1 條：第 215 步），屬 rank-1 反向傳播的數值運算問題，不是訓練發散。依規則未重跑、未修改。", ""]
    out += [

            "### T2-b 狀態／行為指標（非底座任務，十折平均）", "",
            "| r | 序 | 任務 | Pearson（L1 vs L0 patch 分數） | Jaccard（四輪 64 張 vs L0） | ‖BA‖_F |",
            "|---|---|---|---|---|---|"]
    for r in R_LIST:
        for o in ORDERS:
            v = L[(r, o)]
            if v is None:
                continue
            for t in ORDERS[o][1:]:
                out.append(f"| {r} | {o} | {t} | {fmt(v['pearson'][t])} | {fmt(v['jaccard_l0'][t])} | "
                           f"{fmt(v['fro'][t])} |")

    # T3
    out += ["", "## T3 L2（各任務增量相加的合併 expert，不用 router，已見類別內判）", "",
            "| r | 序 | ACC | Masked ACC | Forgetting | " +
            " | ".join(f"Jaccard vs L1 {t}" for t in d.tasks) + " |", "|---|---|---|---|---|" + "---|" * 4]
    for r in R_LIST:
        for o in ORDERS:
            v = L[(r, o)]
            if v is None:
                continue
            out.append(f"| {r} | {o} | {fmt([x['acc'] for x in v['l2']])} | {fmt([x['masked'] for x in v['l2']])} | "
                       f"{fmt([x['forgetting'] for x in v['l2']])} | "
                       + " | ".join(fmt(v["jaccard_m_l1"][t]) for t in d.tasks) + " |")

    # T4
    out += ["", "## T4 完整系統列", "",
            f"router = {mr}；expert = L1(r = {gL['r_star']})（{gL['note']}）；Hard。", "",
            "| 序 | ACC | Masked ACC | Forgetting | 每任務儲存 bytes（expert 參數 fp32 ＋ router key） |",
            "|---|---|---|---|---|"]
    for o, v in full.items():
        cil = v["cil"]
        by = "、".join(f"{t.split('_')[1]} {b:,}" for t, b in v["bytes"].items())
        out.append(f"| {o} | {fmt([x['acc'] for x in cil])} | {fmt([x['masked'] for x in cil])} | "
                   f"{fmt([x['forgetting'] for x in cil])} | {by}（平均 {sum(v['bytes'].values()) / 4:,.0f}） |")

    # T5
    out += ["", "## T5 實際耗時（秒，wall clock）", "", "| 項目 | 秒 |", "|---|---|"]
    tm = M["timing"]
    rc = tm.get("timing_rcache", {})
    out.append(f"| R 線快取（十折 val／test／train） | {sum(v for k, v in rc.items() if k.startswith('nc2/cache/')):,.0f} |")
    for r in R_LIST:
        t = tm.get(f"timing_lora_r{r}", {})
        tr = sum(v for k, v in t.items() if k.startswith("train/"))
        ev = sum(v for k, v in t.items() if k.startswith("eval/"))
        tot = t.get(f"run/r{r}")
        out.append(f"| L 線 r={r}：訓練 {tr:,.0f}、評估 {ev:,.0f} | {tot if tot is not None else float('nan'):,.0f} |")
    out.append(f"| 報告（離線計算，含 k-means） | {t_rep:,.0f} |")
    out += ["", "R 線快取與 L 線 r=4 同時執行；r = 1、2、8 同時執行（各限 CPU 執行緒），耗時含彼此競爭。",
            "逐張讀檔／計算秒數：`cache/nc2_*.pt`、`lora/r*/*/fold*_eval.pt` 的 `t_read_s`、`t_compute_s`；"
            "全部數值另存 `nc2/metrics.json`。", ""]
    (d.out / "REPORT_stage4.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
