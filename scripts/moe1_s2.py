#!/usr/bin/env python3
"""MOE-1 S2：CL 流程（PREREG-17 S2、細則 13–15；seed 42，只推論，只讀快取）。

每序、每階段 t = 1…4：主系統、M3（σ 固定版）、σ 重算版的 ACC／Masked ACC／每任務 WP，Forgetting、BWT 用
nc5_report.cil_full（Table 1 同一個函式）。另：(b) 單獨的正確率與 σ_b 重算值／固定值、γ 統一、K1、K2。

    NAVCIL_MACHINE=mac python scripts/moe1_s2.py --device cpu [--folds 1-10] [--out moe1]
輸出：outputs/navcil/<machine>/<out>/s2.json、s2.done
"""
from __future__ import annotations

import json

import torch

import moe1_common as M
import nc2_report as N2
import nc5_report as N5
from moe1_common import C

KEEP = ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t", "R", "Rm")


def b_alone(db_q: torch.Tensor, first: torch.Tensor) -> float:
    """(b) 單獨：d_b ≥ 0 判第一類。"""
    return ((db_q >= 0) == first).float().mean().item()


def body(run: M.Run) -> dict:
    st = run.st
    k1 = M.k1(run)
    nc8 = M.nc8_main(run)
    z = json.loads((st.root / "results.json").read_text())["B4"]["zprime"]
    if z["lambda_star"] != 1.0:
        raise M.CheckFailed(f"moe0/results.json 的 z′ λ* = {z['lambda_star']}，不是 1.0")
    out = {o: {"main": [], "M3": [], "M3re": [], "wp_task_t": {"main": [], "M3": [], "M3re": []}, "drift": [], "gamma": []}
           for o in M.ORDER_NAMES}
    k2 = {"max_abs_fold_diff": 0.0, "masked": {o: [] for o in M.ORDER_NAMES}, "cil": {o: [] for o in M.ORDER_NAMES}}
    extra = 0.0                                             # 主系統 cil_full 與 nc8 的其他欄位（資訊用）
    run.total(len(run.folds) * 2)
    for f in run.folds:
        S, V = st.split("test", f), st.split("val", f)
        task, label, I6, mv = S["task"], S["labels"], S["I6_cos8"], S["mean_vec"]
        first = (label - 2 * task) == 0
        da, sa = M.d_heads(I6), M.sig(M.d_heads(V["I6_cos8"]), V["task"])
        wp_main = C.task_mean(C.main_preds(I6, M.ar_stage(st, f, "reverse", 4, mv), task)[2] == label, task)
        for o in M.ORDER_NAMES:
            pos = run.pos(o)
            ar = [M.ar_stage(st, f, o, t, mv) for t in range(1, 5)]
            lin = {g: [M.lin8_stage(st, f, o, t, mv, g) for t in range(1, 5)] for g in (M.G_LIN8, M.G_LO)}
            db = {g: [M.d_cols(x) for x in v] for g, v in lin.items()}
            sre, sfix = {}, {}
            for g in db:
                sre[g], sfix[g] = M.sigma_b_table(run, f, o, V, g)

            def stage_fn(kind, g=M.G_LIN8):
                def fn(seen):
                    t = len(seen)
                    m = torch.isin(task, torch.tensor(seen))
                    tk, th = task[m], ar[t - 1][m].argmax(-1)
                    if kind == "main":
                        pred, mk = N2.hard(I6[m], th, tk)
                    else:
                        comps = [(da[m], sa, 1.0), (db[g][t - 1][m], sfix[g] if kind == "M3" else sre[g][t - 1], 1.0)]
                        pred, mk = M.fused_pred(comps, th), M.fused_pred(comps, tk)
                    return pred, mk, {"labels": label[m], "task": tk}
                return fn

            for kind in ("main", "M3", "M3re"):
                r = N5.cil_full(stage_fn(kind), o, st.tasks)
                out[o][kind].append({k: r[k] for k in KEEP})
                # 每任務 WP（欄 = config 任務位置）：主系統 = 真實任務 head 的 2 類正確率；融合 = 告訴任務的融合判定（= Rm）
                wt = [[None] * 4 for _ in range(4)]
                for t in range(4):
                    for j in range(t + 1):
                        wt[t][pos[j]] = wp_main[pos[j]] if kind == "main" else r["Rm"][t][j]
                out[o]["wp_task_t"][kind].append(wt)
                if kind == "main":
                    j8 = nc8[o][f - 1]
                    extra = max(extra, abs(r["acc"] - j8["acc"]), abs(r["masked"] - j8["masked"]),
                                abs(r["forgetting"] - j8["forgetting"]), abs(r["bwt"] - j8["bwt"]),
                                *[abs(a - b) for a, b in zip(r["acc_t"], j8["acc_t"])])
                if kind == "M3re":                           # K2
                    j8 = nc8[o][f - 1]
                    e_mk = C.eq4(j8["wp_task"]) + z["paired"]["masked_vs_wp"]["per_fold"][f - 1]
                    e_cil = j8["acc"] + z["paired"]["cil"][o]["per_fold"][f - 1]
                    k2["max_abs_fold_diff"] = max(k2["max_abs_fold_diff"], abs(r["masked"] - e_mk), abs(r["acc"] - e_cil))
                    k2["masked"][o].append(r["masked"]); k2["cil"][o].append(r["acc"])

            # 舊任務的整片 expert：(b) 單獨的正確率、σ_b 重算值／固定值；[q][t − 1]，t < t_q 為 None
            g0 = M.G_LIN8
            acc_b, ratio = [[None] * 4 for _ in range(4)], [[None] * 4 for _ in range(4)]
            for q in range(4):
                mq = task == q
                for t in range(pos.index(q) + 1, 5):
                    acc_b[q][t - 1] = b_alone(db[g0][t - 1][mq, q], first[mq])
                    ratio[q][t - 1] = sre[g0][t - 1][q] / sfix[g0][q]
            out[o]["drift"].append({"b_alone_acc": acc_b, "sigma_ratio": ratio, "sigma_b_fixed": sfix[g0], "sigma_a": sa})

            # γ 統一
            eq_t = []
            for t in range(1, 5):
                seen = pos[:t]
                d = torch.stack([(ar[t - 1][:, q] - (lin[M.G_LO][t - 1][:, 2 * q] + lin[M.G_LO][t - 1][:, 2 * q + 1])).abs() for q in seen])
                eq_t.append(float(d.max()))
            K = {"task": task, "label": label, "th": ar[3].argmax(-1)}
            gam = {"ar_eq_lin8sum_maxabs_t": eq_t}
            for g, name in ((M.G_LIN8, "g0.01"), (M.G_LO, "g0.001")):
                bt = [b_alone(db[g][3][task == q, q], first[task == q]) for q in range(4)]
                gam[name] = {"b_alone_wp_t": bt, "b_alone_wp": C.eq4(bt),
                             "M3": M.public(M.eval_comps([(da, sa, 1.0), (db[g][3], sfix[g], 1.0)], K))}
            th2 = torch.stack([lin[M.G_LIN8][3][:, 2 * q] + lin[M.G_LIN8][3][:, 2 * q + 1] for q in range(4)], 1).argmax(-1)
            comps = [(da, sa, 1.0), (db[M.G_LIN8][3], sfix[M.G_LIN8], 1.0)]
            gam["tp_lin8sum"] = {"tp": C.eq4(C.task_mean(th2 == task, task)), "tp_t": C.task_mean(th2 == task, task),
                                 "cil_main": M.acc4(N2.hard(I6, th2, task)[0], label, task)[0],
                                 "cil_M3": M.acc4(M.fused_pred(comps, th2), label, task)[0],
                                 "tp_ar": C.eq4(C.task_mean(K["th"] == task, task))}
            out[o]["gamma"].append(gam)
            run.tick()
    k2["masked_mean"] = {o: C.mean(v) for o, v in k2["masked"].items()}
    k2["cil_mean"] = {o: C.mean(v) for o, v in k2["cil"].items()}
    ok = k2["max_abs_fold_diff"] <= 1e-9
    if run.full:
        ok &= all(round(v, 4) == 0.9448 for v in k2["masked_mean"].values())
        ok &= all(round(v, 4) == 0.9226 for v in k2["cil_mean"].values())
    k2.update({"pass": bool(ok), "rounding_checked": run.full})
    if not ok:
        raise M.CheckFailed(f"K2 不符：{json.dumps(k2, ensure_ascii=False)}")
    M.log(f"K2 通過：逐折最大絕對差 {k2['max_abs_fold_diff']:.2e}")
    return {"K1": k1, "K2": k2, "main_cil_full_vs_nc8_max_diff": extra, "orders": out}


if __name__ == "__main__":
    raise SystemExit(M.stage_main("s2", body))
