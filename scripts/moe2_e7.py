#!/usr/bin/env python3
"""MOE-2 E7：CL 流程（PREREG-18 E7、細則 23；seed 42；不設門檻）。

  LR、LRG、LR-bal（各自的 γ*；W 依該序累加到階段 t）：每序、每階段 t = 1…4 的 ACC、Masked ACC、每任務 WP、Forgetting、BWT
  （nc5_report.cil_full，Table 1 同一個函式）。主系統與 M3 照 moe1_s2.py 的算法重算，與 moe1/s2.json 比對。

    NAVCIL_MACHINE=mac python scripts/moe2_e7.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e7.json、e7.done
"""
from __future__ import annotations

import torch

import moe2_common as E
import nc2_report as N2
from moe2_common import M, N5

KEEP = ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t", "R", "Rm")
CMP = ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t")


def body(run: E.Run) -> dict:
    st = run.st
    run.total(len(run.folds) * 2 + 1)
    k1 = M.k1(run)
    rd = {"LR": E.Reader(run, "LR(42)", "s42"), "LRG": E.Reader(run, "LRG(42)", "s42", use_mv=True),
          "LR-bal": E.Reader(run, "LR-bal(42)", "s42", bal=True, grid=E.BAL_GAMMAS)}
    gam = {k: E.select_gamma(run, r) for k, r in rd.items()}
    run.tick()
    s2 = E.m1_json(run, "s2")
    names = list(rd) + ["main", "M3"]
    out = {o: {k: [] for k in names} | {"wp_task_t": {k: [] for k in rd}} for o in E.ORDER_NAMES}
    s2diff = {"main": 0.0, "M3": 0.0}
    for f in run.folds:
        S, V, D = st.split("test", f), st.split("val", f), E.vecs(run, "test", f)
        task, label, I6, mv = S["task"], S["labels"], S["I6_cos8"], S["mean_vec"]
        da, sa = M.d_heads(I6), M.sig(M.d_heads(V["I6_cos8"]), V["task"])
        for o in E.ORDER_NAMES:
            pos = run.pos(o)
            ar = [M.ar_stage(st, f, o, t, mv) for t in range(1, 5)]
            db = [M.d_cols(M.lin8_stage(st, f, o, t, mv, M.G_LIN8)) for t in range(1, 5)]
            sfix = M.sigma_b_table(run, f, o, V, M.G_LIN8)[1]

            def stage_fn(kind):
                def fn(seen):
                    t = len(seen)
                    m = torch.isin(task, torch.tensor(seen))
                    tk, th_all = task[m], ar[t - 1].argmax(-1)
                    if kind == "main":
                        pred, mk = N2.hard(I6[m], th_all[m], tk)
                    elif kind == "M3":
                        comps = [(da[m], sa, 1.0), (db[t - 1][m], sfix, 1.0)]
                        pred, mk = M.fused_pred(comps, th_all[m]), M.fused_pred(comps, tk)
                    else:
                        r, g = rd[kind], gam[kind]["star"]
                        pred = E.pred_of(r.diff(f, o, t, g, D, th_all), th_all)[m]
                        mk = E.pred_of(r.diff(f, o, t, g, D, task), task)[m]
                    return pred, mk, {"labels": label[m], "task": tk}
                return fn

            for kind in names:
                r = N5.cil_full(stage_fn(kind), o, st.tasks)
                out[o][kind].append({k: r[k] for k in KEEP})
                if kind in rd:
                    wt = [[None] * 4 for _ in range(4)]
                    for t in range(4):
                        for j in range(t + 1):
                            wt[t][pos[j]] = r["Rm"][t][j]
                    out[o]["wp_task_t"][kind].append(wt)
                else:
                    ref = s2["orders"][o][kind][E.fold_idx(s2, f)]
                    for k in CMP:
                        a, b = r[k], ref[k]
                        s2diff[kind] = max(s2diff[kind], *([abs(x - y) for x, y in zip(a, b)] if isinstance(a, list) else [abs(a - b)]))
            run.tick()
    return {"K1": k1, "gamma": gam, "orders": out, "s2_max_abs_diff": s2diff}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e7", body))
