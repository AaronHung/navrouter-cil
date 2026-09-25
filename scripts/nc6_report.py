#!/usr/bin/env python3
"""NC-6 報告：主 router 改 AR、AR 的兩個調整、TSP、機制檢查、新主系統（PREREG-6）。

只讀既有快取（NC-1 mean_vec、NC-2 R3 key、NC-5 背景／腫瘤半向量、lora_v2 評估檔），不讀 slide、
不訓練 expert。AR 類的累加、求解、白化一律 float64（CPU）。
    NAVCIL_MACHINE=mac python scripts/nc6_report.py
輸出：outputs/navcil/<machine>/REPORT_stage8.md、nc6/metrics.json
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
import nc5_report as N5                                                   # noqa: E402
from selector.cil_eval import mean_sd, subset_key                         # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402

FOLDS = N2.FOLDS
GAMMAS = (1e-4, 1e-3, 1e-2, 1e-1)
fmt = N2.fmt
D64 = torch.float64
NC5_VAL_R3 = 0.8942                     # REPORT_stage7：R3(k=8) validation（設計決定 0 的依據，僅引用）


def order_pos(d, order):
    return [d.tasks.index(x) for x in ORDERS[order]]


def prefix_key(d, order, p):
    names = ORDERS[order]
    return subset_key([d.tasks.index(x) for x in names[:names.index(d.tasks[p]) + 1]])


def feat(d, C5, kind, fold, split, p, order, seen) -> torch.Tensor:
    t = d.tasks[p]
    if kind == "mean":
        if split == "train":
            return d.c(fold, "train", t)["mean_vec"]
        return d.n2(fold, "val", t)["mean_vec"] if split == "val" else d.c(fold, "test", t)["mean_vec"]
    src, q, k = {"e0.75": ("e", 0.75, "bg"), "z0.75": ("z", 0.75, "bg"), "e-th": ("e", None, "th")}[kind]
    key = prefix_key(d, order, p) if split == "train" else (N5.ALL if split == "val" else subset_key(seen))
    return C5.vec(fold, split, p, src, q, key, k)


def proj(d, seen, mode) -> torch.Tensor:
    Fx = d.F.to(D64)
    if mode == "diff":
        U = torch.stack([Fx[2 * p + 1] - Fx[2 * p] for p in seen], 1)
    else:
        U = torch.stack([Fx[r] for p in seen for r in task_rows(p)], 1)
    Q, _ = torch.linalg.qr(U)
    return torch.eye(Fx.shape[1], dtype=D64) - Q @ Q.t()


class ARX:
    """AR／AR-bal（可選輸入種類與 TSP 投影）；float64。"""

    def __init__(self, d, C5, kind="mean", bal=False, mode=None):
        self.d, self.C5, self.kind, self.bal, self.mode = d, C5, kind, bal, mode
        self._A, self._W = {}, {}

    def vec(self, fold, split, p, order, seen):
        v = feat(self.d, self.C5, self.kind, fold, split, p, order, seen).to(D64)
        if self.mode:
            v = F.normalize(v @ proj(self.d, seen, self.mode), dim=-1)
        return v

    def A(self, fold, order, t):
        k = (fold, order, t)
        if k not in self._A:
            seen = order_pos(self.d, order)[:t]
            A = torch.zeros(513, 513, dtype=D64)
            cols, N = [], 0
            for p in seen:
                Xj = N5.aug(self.vec(fold, "train", p, order, seen))
                w = 1.0 / Xj.shape[0] if self.bal else 1.0
                A += w * (Xj.t() @ Xj)
                cols.append(w * Xj.sum(0))
                N += Xj.shape[0]
            self._A[k] = (A, torch.stack(cols, 1), N)
        return self._A[k]

    def W(self, fold, order, t, gamma):
        k = (fold, order, t, gamma)
        if k not in self._W:
            A, B, _ = self.A(fold, order, t)
            self._W[k] = torch.linalg.solve(A + gamma * torch.eye(513, dtype=D64), B)
        return self._W[k]

    def fn(self, gamma):
        def f(fold, order, split, g, seen):
            pos = order_pos(self.d, order)[:len(seen)]
            assert sorted(pos) == sorted(seen)
            X = torch.cat([N5.aug(self.vec(fold, split, p, order, seen)) for p in seen])
            return X @ self.W(fold, order, len(seen), gamma)[:, [pos.index(p) for p in seen]]
        return f

    def one_shot_W(self, fold, gamma):
        X = [N5.aug(self.vec(fold, "train", p, "reverse", list(range(4)))) for p in range(4)]
        w = [1.0 / x.shape[0] if self.bal else 1.0 for x in X]
        Xa = torch.cat(X)
        wa = torch.cat([torch.full((x.shape[0],), wi, dtype=D64) for x, wi in zip(X, w)])
        A = (Xa * wa[:, None]).t() @ Xa
        B = torch.stack([wi * x.sum(0) for x, wi in zip(X, w)], 1)
        return torch.linalg.solve(A + gamma * torch.eye(513, dtype=D64), B)


def r3u_fn(d, mode):
    def f(fold, order, split, g, seen):
        P = proj(d, seen, mode)
        keys = d.keys(fold)["multi_proto_k"][8]
        x = F.normalize(g["mean_vec"].to(D64) @ P, dim=-1)
        return torch.stack([(x @ F.normalize(keys[p].to(D64) @ P, dim=-1).t()).amax(-1) for p in seen], -1)
    return f


def white_r2_fn(d, arx, gamma):
    def f(fold, order, split, g, seen):
        t = len(seen)
        A, _, N = arx.A(fold, order, t)
        S = A[:512, :512] / N
        lam, V = torch.linalg.eigh(S + gamma * torch.eye(512, dtype=D64))
        Wh = V @ torch.diag(lam.clamp_min(1e-300) ** -0.5) @ V.t()
        mu = torch.stack([d.c(fold, "train", d.tasks[p])["mean_vec"].to(D64).mean(0) for p in seen])
        keys = F.normalize(mu @ Wh, dim=-1)
        x = F.normalize(g["mean_vec"].to(D64) @ Wh, dim=-1)
        return x @ keys.t()
    return f


def r2_fn(d):
    def f(fold, order, split, g, seen):
        return N2.scores_for(d, fold, g, seen, "R2")
    return f


def run(d, C5, fn, four_fn=None):
    return N5.evaluate(d, C5, fn, "AR*", four_fn)


def val_mean(d, C5, fn):
    xs = [N5.val_acc(d, C5, fn, "AR*", o, f) for o in ORDERS for f in FOLDS]
    return mean_sd(xs)[0], xs


def cil(r, o, k="acc"):
    return [x[k] for x in r[o]["cil"]]


def summ(r, o):
    dg = r[o]["diag"]
    return {"acc": cil(r, o), "masked": cil(r, o, "masked"), "forgetting": cil(r, o, "forgetting"),
            "bwt": cil(r, o, "bwt"), "micro": [x["micro"] for x in dg], "macro": [x["macro"] for x in dg],
            "tp_task": [[x["tp_task"][p] for x in dg] for p in range(4)],
            "e2l": sum(x["e2l"] for x in dg), "l2e": sum(x["l2e"] for x in dg),
            "l2e_cls": {k: sum(x["l2e_cls"][k] for x in dg) for k in ("LUAD", "LUSC")},
            "e2l_cls": {k: sum(x["e2l_cls"][k] for x in dg) for k in ("ESAD", "ESCC")}}


def paired(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    return {"per_fold": diffs, "mean": mean_sd(diffs)[0], "wins": sum(x > 0 for x in diffs)}


def main() -> int:
    t0 = time.perf_counter()
    d = N2.D2()
    C5 = N5.Ctx5(d)
    M1 = json.loads((d.out / "nc1" / "metrics.json").read_text())
    M = {}

    # ── A ────────────────────────────────────────────────────────────────
    ars = {"AR": ARX(d, C5), "AR-bal": ARX(d, C5, bal=True)}
    candA = [(v, g) for v in ("AR", "AR-bal") for g in GAMMAS]
    valA = {f"{v}|{g}": val_mean(d, C5, ars[v].fn(g)) for v, g in candA}
    best = max(x[0] for x in valA.values())
    order_pref = sorted(candA, key=lambda c: (c[0] != "AR", -c[1]))         # AR 優先、γ 大者優先
    selA = next(c for c in order_pref if valA[f"{c[0]}|{c[1]}"][0] == best)
    selv, selg = selA
    arx = ars[selv]
    W1 = arx.one_shot_W(1, selg)
    check = {}
    for o in ORDERS:
        pos = order_pos(d, o)
        Wi = arx.W(1, o, 4, selg)
        Wc = torch.zeros_like(Wi)
        for j, p in enumerate(pos):
            Wc[:, p] = Wi[:, j]
        check[o] = float((Wc - W1).abs().max())
    if not all(v < 1e-4 for v in check.values()):
        (d.out / "nc6").mkdir(exist_ok=True)
        (d.out / "nc6" / "check_failed.json").write_text(json.dumps({"selected": selA, "check": check}, indent=1))
        print(f"⚠️ 一致性檢查未達 < 1e-4：{check} —— 停下回報。")
        return 7
    testA = {f"{v}|{g}": run(d, C5, ars[v].fn(g)) for v, g in candA}
    kA = f"{selv}|{selg}"
    sel_res = testA[kA]

    # ── B：TSP ───────────────────────────────────────────────────────────────
    tsp = {"R3+U_diff": r3u_fn(d, "diff"), "R3+U_all": r3u_fn(d, "all"),
           "AR+U_diff": ARX(d, C5, bal=(selv == "AR-bal"), mode="diff").fn(selg),
           "AR+U_all": ARX(d, C5, bal=(selv == "AR-bal"), mode="all").fn(selg)}
    valB = {k: val_mean(d, C5, f) for k, f in tsp.items()}
    testB = {k: run(d, C5, f) for k, f in tsp.items()}
    pick = lambda a, b: a if valB[a][0] >= valB[b][0] else b                # noqa: E731
    s1, s2 = pick("R3+U_diff", "R3+U_all"), pick("AR+U_diff", "AR+U_all")
    tsp1 = all(mean_sd(cil(testB[s1], o))[0] >= mean_sd(cil(sel_res, o))[0] - 0.005 for o in ORDERS)
    pB2 = {o: paired(cil(testB[s2], o), cil(sel_res, o)) for o in ORDERS}
    tsp2 = all(pB2[o]["mean"] >= 0.01 and pB2[o]["wins"] >= 7 for o in ORDERS)
    cosLE = {"before": [], "U_diff": [], "U_all": []}
    for f in FOLDS:
        lu = d.c(f, "train", "tcga_lung"); es = d.c(f, "train", "tcga_esca")
        a = lu["mean_vec"][lu["labels"] == 7].to(D64); b = es["mean_vec"][es["labels"] == 1].to(D64)
        cosLE["before"].append(float(F.normalize(a.mean(0), dim=0) @ F.normalize(b.mean(0), dim=0)))
        for mode, key in (("diff", "U_diff"), ("all", "U_all")):
            P = proj(d, list(range(4)), mode)
            ma = F.normalize(F.normalize(a @ P, dim=-1).mean(0), dim=0)
            mb = F.normalize(F.normalize(b @ P, dim=-1).mean(0), dim=0)
            cosLE[key].append(float(ma @ mb))

    # ── C ────────────────────────────────────────────────────────────────────
    C1 = {k: run(d, C5, ARX(d, C5, kind=k, bal=(selv == "AR-bal")).fn(selg)) for k in ("e0.75", "z0.75", "e-th")}
    C3 = {"whitened R2": run(d, C5, white_r2_fn(d, arx, selg)), "R2（原始）": run(d, C5, r2_fn(d))}

    # ── D ────────────────────────────────────────────────────────────────────
    def four_l1(o):
        evs = {f: torch.load(d.out / "lora_v2" / "r2" / o / f"fold{f}_eval.pt", map_location="cpu") for f in FOLDS}
        return lambda f, o_, seen: torch.cat([evs[f]["tasks"][d.tasks[p]]["l1_cos8"] for p in seen]), evs
    D1, D2, wp1 = {}, {}, {}
    for o in ORDERS:
        ffn, evs = four_l1(o)
        D1[o] = N5.evaluate(d, C5, arx.fn(selg), "AR*", lambda f, o_, seen, ffn=ffn: ffn(f, o_, seen))[o]
        wp1[o] = [[masked_acc(evs[f]["tasks"][d.tasks[p]]["l1_cos8"][:, p], evs[f]["tasks"][d.tasks[p]]["labels"], p)
                   for f in FOLDS] for p in range(4)]
        D2[o] = sel_res[o]
    wp0 = [[M1["stage1"]["per_fold"][str(f)][p]["e_fourround"] for f in FOLDS] for p in range(4)]
    D1_tsp = {}
    if tsp1 or tsp2:
        sname = s2 if tsp2 else s1
        for o in ORDERS:
            ffn, _ = four_l1(o)
            D1_tsp[o] = N5.evaluate(d, C5, tsp[sname], "AR*", lambda f, o_, seen, ffn=ffn: ffn(f, o_, seen))[o]

    # ── metrics ──────────────────────────────────────────────────────────────
    M = {"A": {"val": {k: v[0] for k, v in valA.items()}, "selected": kA, "check_fold1": check,
               "test": {k: {o: summ(r, o) for o in ORDERS} for k, r in testA.items()}},
         "B": {"val": {k: v[0] for k, v in valB.items()}, "picked": [s1, s2], "tsp1": tsp1, "tsp2": tsp2,
               "paired_AR_U": pB2, "test": {k: {o: summ(r, o) for o in ORDERS} for k, r in testB.items()},
               "cos_LUSC_ESCC": cosLE},
         "C1": {k: {o: summ(r, o) for o in ORDERS} for k, r in C1.items()},
         "C3": {k: {o: summ(r, o) for o in ORDERS} for k, r in C3.items()},
         "D1": {o: summ({o: D1[o]}, o) | {"acc_t": [x["acc_t"] for x in D1[o]["cil"]],
                                         "masked_t": [x["masked_t"] for x in D1[o]["cil"]]} for o in ORDERS},
         "D2": {o: summ({o: D2[o]}, o) | {"acc_t": [x["acc_t"] for x in D2[o]["cil"]],
                                         "masked_t": [x["masked_t"] for x in D2[o]["cil"]]} for o in ORDERS},
         "wp_L1r2": wp1, "wp_L0": wp0,
         "D1_tsp": {o: summ({o: v}, o) for o, v in D1_tsp.items()}}
    (d.out / "nc6").mkdir(exist_ok=True)
    (d.out / "nc6" / "metrics.json").write_text(json.dumps(M, indent=1, default=str))
    write(d, M, selv, selg, time.perf_counter() - t0)
    print(f"→ {d.out / 'REPORT_stage8.md'}")
    return 0


def masked_acc(c8, labels, p):
    rr = torch.tensor(task_rows(p))
    return (rr[c8[:, rr].argmax(-1)] == labels).float().mean().item()


def write(d, M, selv, selg, t_rep):
    ok = lambda b: "**通過**" if b else "**未通過**"                     # noqa: E731
    A, B = M["A"], M["B"]
    kA = A["selected"]
    sA = A["test"][kA]
    out = ["# REPORT — NC-6：主 router 改 AR、AR 的兩個調整、TSP、機制檢查、新主系統（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；所有數字來自同一台、同一批；"
           "全部只做推論、不讀 slide（只用既有快取）。AR 類的累加、求解、白化一律 float64（AMENDMENT-2 例外）。"
           "判準見 `PREREG-6.md`（commit d9eba4d）。", "",
           "## PREREG-6 判準落點", "", "| 判準 | 數值 | 所在表格 | 結果 |", "|---|---|---|---|",
           f"| 0. 主 router 改 AR（依 NC-5 validation：AR 0.9166 vs R3(k=8) {NC5_VAL_R3}） | 設計決定 | — | 已採用 |",
           f"| A 選法（validation t = 4 CIL，十折 × 兩序） | 選出 **{selv}、γ = {selg:g}**（{A['val'][kA]:.4f}） | T1 | — |",
           f"| A 選出者的 fold 1 一致性檢查 < 1e-4 | reverse {A['check_fold1']['reverse']:.2e}、paper {A['check_fold1']['paper']:.2e} | T1 | "
           f"{ok(all(v < 1e-4 for v in A['check_fold1'].values()))} |"]
    s1, s2 = B["picked"]
    for o in ORDERS:
        a = mean_sd(B["test"][s1][o]["acc"])[0]; b = mean_sd(sA[o]["acc"])[0]
        out.append(f"| TSP-pass-1（{o}）：{s1} test CIL ≥ A 選出者 − 0.005 | {a:.4f} vs {b:.4f} − 0.005 = {b - 0.005:.4f} | T2 | "
                   f"{ok(a >= b - 0.005)} |")
    out.append(f"| **TSP-pass-1 整體**（每任務儲存 16,384 bytes） | | T2 | {ok(B['tsp1'])} |")
    for o in ORDERS:
        p_ = B["paired_AR_U"][o]
        out.append(f"| TSP-pass-2（{o}）：{s2} − A 選出者 ≥ +0.01 且贏 ≥ 7/10 | {p_['mean']:+.4f}、{p_['wins']}/10 | T2 | "
                   f"{ok(p_['mean'] >= 0.01 and p_['wins'] >= 7)} |")
    out += [f"| **TSP-pass-2 整體** | | T2 | {ok(B['tsp2'])} |",
            "| C、D | 不設門檻，必報 | T3、T4 | 已報 |", ""]

    # T1
    out += ["## T1 A 段：AR／AR-bal × γ（L0 expert、Hard）", "",
            "| 候選 | validation CIL | test CIL reverse | test CIL paper | TP micro | Lung→ESCA |", "|---|---|---|---|---|---|"]
    for k in A["val"]:
        r = A["test"][k]
        out.append(f"| {k.replace('|', '、γ = ')}{' ★' if k == kA else ''} | {A['val'][k]:.4f} | {fmt(r['reverse']['acc'])} | "
                   f"{fmt(r['paper']['acc'])} | {fmt(r['reverse']['micro'])} | {r['reverse']['l2e']} |")
    out += ["", f"★ = validation 選出者。t = 4 的 AR 在兩序數學上相同。", "",
            f"選出者（{selv}、γ = {selg:g}）細節（test、t = 4）：", "",
            "| 序 | " + " | ".join(f"TP {t.split('_')[1]}" for t in d.tasks) + " | TP macro | TP micro | ESCA→Lung（ESAD／ESCC） | "
            "Lung→ESCA（LUAD／LUSC） | CIL |", "|---|" + "---|" * 4 + "---|---|---|---|---|"]
    for o in ORDERS:
        r = sA[o]
        out.append(f"| {o} | " + " | ".join(fmt(r["tp_task"][p]) for p in range(4)) + f" | {fmt(r['macro'])} | {fmt(r['micro'])} | "
                   f"{r['e2l']}（{r['e2l_cls']['ESAD']}／{r['e2l_cls']['ESCC']}） | {r['l2e']}（{r['l2e_cls']['LUAD']}／{r['l2e_cls']['LUSC']}） | "
                   f"{fmt(r['acc'])} |")

    # T2
    out += ["", "## T2 B 段：TSP 文字引導的組織型方向移除", "",
            "| 候選 | validation CIL | test CIL reverse | test CIL paper | TP micro（reverse） | ESCA→Lung | Lung→ESCA | 儲存 |",
            "|---|---|---|---|---|---|---|---|"]
    for k in ("R3+U_diff", "R3+U_all", "AR+U_diff", "AR+U_all"):
        r = B["test"][k]
        tag = " ★" if k in (s1, s2) else ""
        by = "16,384（R3 key）" if k.startswith("R3") else "同 A 選出者"
        out.append(f"| {k}{tag} | {B['val'][k]:.4f} | {fmt(r['reverse']['acc'])} | {fmt(r['paper']['acc'])} | {fmt(r['reverse']['micro'])} | "
                   f"{r['reverse']['e2l']} | {r['reverse']['l2e']} | {by} |")
    out += ["", "★ = 各組 validation 選出者。AR＋U 在各階段以新投影重算舊任務統計量（PREREG-6 操作定義 5 揭露），"
            "t = 4 CIL 為判定值。R3 key 以 NC-2 的同一程式與種子（KMeans(k=8, n_init=10, random_state=0)）在本批重算、"
            "未另存檔；R3(k=8) 基準在 NC-3／NC-4／NC-5 的重算結果與 NC-2 相同（test t = 4 CIL 0.8987）。", "",
            "LUSC 與 ESCC 的 train slide 平均向量 cosine（十折 mean ± sd）：", "",
            "| 投影 | cosine |", "|---|---|",
            f"| 投影前 | {fmt(B['cos_LUSC_ESCC']['before'])} |", f"| U_diff | {fmt(B['cos_LUSC_ESCC']['U_diff'])} |",
            f"| U_all | {fmt(B['cos_LUSC_ESCC']['U_all'])} |", ""]

    # T3
    def row(name, r, o="reverse"):
        x = r[o]
        return (f"| {name} | {fmt(x['micro'])} | {x['e2l']} | {x['l2e']} | {fmt(x['acc'])} | {fmt(r['paper']['acc'])} |")
    out += ["## T3 C 段：機制檢查", "", f"### C1 輸入改為 NC-5 向量（{selv}、γ = {selg:g}）", "",
            "| 輸入 | TP micro（reverse） | ESCA→Lung | Lung→ESCA | CIL reverse | CIL paper |", "|---|---|---|---|---|---|",
            row("全部 patch 平均（A 選出者）", sA)]
    for k, lab in (("e0.75", "e0.75 背景"), ("z0.75", "z0.75 背景"), ("e-th", "e 腫瘤半")):
        out.append(row(lab, M["C1"][k]))
    out += ["", "### C2 γ 趨勢（test、reverse、t = 4）", "", "| 變體 | " + " | ".join(f"γ = {g:g}" for g in GAMMAS) + " |",
            "|---|" + "---|" * 4]
    for v in ("AR", "AR-bal"):
        out.append(f"| {v} TP micro | " + " | ".join(f"{mean_sd(A['test'][f'{v}|{g}']['reverse']['micro'])[0]:.4f}" for g in GAMMAS) + " |")
        out.append(f"| {v} Lung→ESCA | " + " | ".join(str(A['test'][f'{v}|{g}']['reverse']['l2e']) for g in GAMMAS) + " |")
    out += ["", f"### C3 重新加權（白化後 R2）vs 判別式 AR（γ = {selg:g}）", "",
            "| router | TP micro（reverse） | ESCA→Lung | Lung→ESCA | CIL reverse | CIL paper |", "|---|---|---|---|---|---|",
            row(f"{selv}（判別式）", sA), row("白化後 R2", M["C3"]["whitened R2"]), row("R2（原始）", M["C3"]["R2（原始）"]), ""]

    # T4
    out += ["## T4 D 段：新主系統（兩序）", "", f"router = {selv}、γ = {selg:g}；D1 expert = L1(r=2) v2；D2 expert = L0；Hard。", "",
            "| 系統 | 序 | ACC | Masked ACC | Forgetting | BWT |", "|---|---|---|---|---|---|"]
    for name, key in (("D1", "D1"), ("D2", "D2")):
        for o in ORDERS:
            r = M[key][o]
            out.append(f"| {name} | {o} | {fmt(r['acc'])} | {fmt(r['masked'])} | {fmt(r['forgetting'])} | {fmt(r['bwt'])} |")
    out += ["", "| 系統 | 序 | 指標 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|---|---|"]
    for name in ("D1", "D2"):
        for o in ORDERS:
            r = M[name][o]
            out.append(f"| {name} | {o} | ACC | " + " | ".join(fmt([x[t] for x in r["acc_t"]]) for t in range(4)) + " |")
            out.append(f"| {name} | {o} | Masked ACC | " + " | ".join(fmt([x[t] for x in r["masked_t"]]) for t in range(4)) + " |")
    out += ["", "每任務 TP（router，t = 4）與 WP（真實任務 expert 的 Masked ACC）：", "",
            "| 項目 | 序 | " + " | ".join(d.tasks) + " |", "|---|---|" + "---|" * 4]
    for o in ORDERS:
        out.append(f"| TP | {o} | " + " | ".join(fmt(M['D1'][o]['tp_task'][p]) for p in range(4)) + " |")
        out.append(f"| WP D1（L1 r=2 v2） | {o} | " + " | ".join(fmt(M['wp_L1r2'][o][p]) for p in range(4)) + " |")
    out.append(f"| WP D2（L0） | 兩序 | " + " | ".join(fmt(M['wp_L0'][p]) for p in range(4)) + " |")
    lp = N2.lora_params(2)
    out += ["", "儲存分項（bytes；fp32／fp64）：", "", "| 項目 | fp32 | fp64 | 備註 |", "|---|---|---|---|",
            f"| D1 expert：底座任務（完整） | {N2.FULL_PARAMS * 4:,} | {N2.FULL_PARAMS * 8:,} | 132,097 參數 |",
            f"| D1 expert：其他每任務（r = 2 增量） | {lp * 4:,} | {lp * 8:,} | {lp:,} 參數 |",
            f"| D2 expert：每任務（L0 完整） | {N2.FULL_PARAMS * 4:,} | {N2.FULL_PARAMS * 8:,} | |",
            f"| router：A（共用） | {513 * 513 * 4:,} | {513 * 513 * 8:,} | 513 × 513 |",
            f"| router：每任務 b_j | {513 * 4:,} | {513 * 8:,} | {'w_j 已乘入' if selv == 'AR-bal' else ''} |",
            "| 類別文字 f_txt（每任務） | 4,096 | 8,192 | 分類頭本身 |", "",
            "不存任何 slide 或 patch 特徵（PREREG-6 設計決定 0）。", ""]
    if M["D1_tsp"]:
        out += ["### TSP 版本的 D1", "", "| 序 | ACC | Masked ACC | Forgetting |", "|---|---|---|---|"]
        for o, r in M["D1_tsp"].items():
            out.append(f"| {o} | {fmt(r['acc'])} | {fmt(r['masked'])} | {fmt(r['forgetting'])} |")
        out.append("")
    else:
        out += ["TSP-pass-1、TSP-pass-2 皆未通過，不列 TSP 版本的 D1。", ""]
    out += ["## T5 實際耗時", "", f"報告（全部離線計算：R3 key 的 KMeans 重算、AR／AR-bal 求解、白化、TSP 投影與所有評估）："
            f"{t_rep:,.0f} 秒（執行緒 8）。不讀 slide、不訓練 expert。", ""]
    (d.out / "REPORT_stage8.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
