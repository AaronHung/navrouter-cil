#!/usr/bin/env python3
"""MOE-3 f2：逐階段結果與門檻（PREREG-19 F1–F3、門檻；細則 4、6–8、16–22）。

  九個系統 × 兩序 × t = 1…4：ACC、告訴任務 WP、每任務 WP、Forgetting、BWT、Ā（nc5_report.cil_full）
  S-main、S-M3 取自 moe1/s2.json，S-LR 取自 moe2/e7.json；本批重算並報最大絕對差
  t = 4：相對 S-main 的逐折相減；兩序是否逐張相同
  門檻：G-FINAL、G-COLD、G-ANCHOR（S-A0）；同一組數字對 S-R0 再算一次（只報告）

    NAVCIL_MACHINE=mac python scripts/moe3_f2.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/f2.json、f2.done
"""
from __future__ import annotations

import json

import torch

import moe3_common as T
import nc2_report as N2
from moe3_common import C, E, M, N5


def body(run: T.Run) -> dict:
    st = run.st
    run.total(len(run.folds) * 2 + 2)
    k1 = M.k1(run)
    hp = T.load_hp(run)
    sysm = T.systems(hp)
    s2 = json.loads((run.m1 / "s2.json").read_text())
    e7 = json.loads((run.m2 / "e7.json").read_text())
    e1 = json.loads((run.m2 / "e1.json").read_text())
    run.tick()
    rows = {o: {k: [] for k in T.SYS_ORDER} for o in T.ORDER_NAMES}
    priv = {o: {k: [] for k in sysm} for o in T.ORDER_NAMES}
    rediff = {"S-main": 0.0, "S-M3": 0.0, "S-LR": 0.0}
    for f in run.folds:
        D, S, V = T.data(run, "test", f), st.split("test", f), st.split("val", f)
        task, label, I6, mv = S["task"], S["labels"], S["I6_cos8"], S["mean_vec"]
        da, sa = M.d_heads(I6), M.sig(M.d_heads(V["I6_cos8"]), V["task"])
        i2, i7 = T.fold_i(s2, f), T.fold_i(e7, f)
        for o in T.ORDER_NAMES:
            ar = T.ar_stages(run, f, o, D)
            db = [M.d_cols(M.lin8_stage(st, f, o, t, mv, M.G_LIN8)) for t in range(1, 5)]
            sfix = M.sigma_b_table(run, f, o, V, M.G_LIN8)[1]
            wp_main = C.task_mean(C.main_preds(I6, ar[3], task)[2] == label, task)

            def base_fn(kind):                                   # 主系統與 M3：moe1_s2.py／moe2_e7.py 的算法
                def fn(seen):
                    t = len(seen)
                    m = torch.isin(task, torch.tensor(seen))
                    tk, th_all = task[m], ar[t - 1].argmax(-1)
                    if kind == "main":
                        pred, mk = N2.hard(I6[m], th_all[m], tk)
                    else:
                        comps = [(da[m], sa, 1.0), (db[t - 1][m], sfix, 1.0)]
                        pred, mk = M.fused_pred(comps, th_all[m]), M.fused_pred(comps, tk)
                    return pred, mk, {"labels": label[m], "task": tk}
                return fn

            for kind, name in (("main", "S-main"), ("M3", "S-M3")):
                mine = T.cl_row(run, o, N5.cil_full(base_fn(kind), o, st.tasks), wp_main if kind == "main" else None)
                ref = T.from_s2(run, o, s2["orders"][o][kind][i2], s2["orders"][o]["wp_task_t"][kind][i2], kind == "main")
                rediff[name] = max(rediff[name], T.row_diff(mine, ref))
                rows[o][name].append(ref)
            for name, (kind, rd) in sysm.items():
                r = T.run_cl(run, f, o, D, T.reader_fn(run, f, o, D, kind, rd), ar)
                priv[o][name].append(r)
                if name == "S-LR":
                    x = e7["orders"][o]["LR"][i7]
                    ref = {"acc_t": x["acc_t"], "wp_t": x["masked_t"], "forgetting": x["forgetting"], "bwt": x["bwt"],
                           "abar": C.mean(x["acc_t"]), "wp_task_t": e7["orders"][o]["wp_task_t"]["LR"][i7], "R": x["R"], "Rm": x["Rm"]}
                    rediff[name] = max(rediff[name], T.row_diff(r, ref))
                    rows[o][name].append(ref)
                else:
                    rows[o][name].append(M.public(r))
            run.tick()
        run.drop()

    # F3：t = 4 相對 S-main
    vs_main = {o: {k: {"wp": M.paired(T.t4(rows[o][k], "wp_t"), T.t4(rows[o]["S-main"], "wp_t")),
                       "cil": M.paired(T.t4(rows[o][k], "acc_t"), T.t4(rows[o]["S-main"], "acc_t"))}
                   for k in T.SYS_ORDER if k != "S-main"} for o in T.ORDER_NAMES}
    same = {k: T.order_same({o: priv[o][k] for o in T.ORDER_NAMES}) for k in sysm}

    # 門檻
    main5, main5_diff = {}, 0.0
    for o in T.ORDER_NAMES:
        main5[o] = []
        for k, f in enumerate(run.folds):
            i1 = T.fold_i(e1, f)
            xs = [e1["seeds"][str(s)]["orders"][o]["main"][i1]["cil"] for s in T.SEEDS]
            main5_diff = max(main5_diff, *[abs(x - E.seed_base(run, s)[o]["main"][k]["cil"]) for x, s in zip(xs, T.SEEDS)])
            main5[o].append(C.mean(xs))
    gates = {}
    for name in ("S-A0", "S-R0"):
        g = {"G-FINAL": T.gate({o: [a - b for a, b in zip(T.t4(rows[o][name], "acc_t"), main5[o])] for o in T.ORDER_NAMES}, 0.007),
             "G-COLD": T.gate({o: [a["acc_t"][0] - b["acc_t"][0] for a, b in zip(rows[o][name], rows[o]["S-main"])]
                               for o in T.ORDER_NAMES}, -0.010, wins=False)}
        if name == "S-A0":
            g["G-ANCHOR"] = T.gate({o: [a["abar"] - b["abar"] for a, b in zip(rows[o]["S-A0"], rows[o]["S-R0"])]
                                    for o in T.ORDER_NAMES}, 0.005)
        gates[name] = g
    run.tick()
    return {"K1": k1, "systems": {k: {"kind": v[0], "reader": list(v[1])} for k, v in sysm.items()}, "rows": rows,
            "vs_main": vs_main, "order_same": same, "recompute_max_abs": rediff,
            "main5_cil": main5, "main5_vs_recompute_max_abs": main5_diff, "gates": gates, "gates_valid": run.full}


if __name__ == "__main__":
    raise SystemExit(T.stage_main("f2", body))
