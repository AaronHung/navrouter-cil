#!/usr/bin/env python3
"""MOE-1 S4：多 seed 與 ensemble 對照（PREREG-17 S4、細則 18–22；門檻 G-M3c、G-ENS）。

  1. K1、K3（seed 42 重訓 fold 1 tcga_esca，逐位元比對）
  2. 訓練 I6(r = 2) seed 43、44（十折 × 4 任務 × 2）→ <out>/i6_seed{s}/
  3. 新 seed 的 validation／test 推論（四個 head、四輪、等權；同 moe0_infer）→ <out>/cache/seed{s}_{split}_fold{f}.pt
  4. 每個 seed 的主系統與 M3、ensemble 對照（LL、LG、GG、LLL）、seed 間判定不同的張數、G-M3c、G-ENS

    NAVCIL_MACHINE=mac python scripts/moe1_s4.py --device cpu [--folds 1-10] [--out moe1] [--epochs 5]
輸出：outputs/navcil/<machine>/<out>/s4.json、s4.done
"""
from __future__ import annotations

import moe1_common as M
from moe1_common import C

SEEDS = (42, 43, 44)
NEW = (43, 44)
PAIRS = ((42, 43), (43, 44), (44, 42))


def gate_m3c(rows: dict, full: bool) -> dict:
    conds = []
    for s in NEW:
        for o in M.ORDER_NAMES:
            p = rows[str(s)][o]["paired"]["cil"]
            conds.append({"seed": s, "order": o, "mean": p["mean"], "wins": p["wins"],
                          "pass": bool(p["mean"] >= 0.007 and p["wins"] >= 7)})
    return {"conditions": conds, "pass": bool(all(c["pass"] for c in conds)), "valid": full}


def body(run: M.Run) -> dict:
    k1 = M.k1(run)
    k3 = M.k3(run)
    n = len(run.folds)
    run.total(n * 4 * len(NEW) + n * 2 * len(NEW))
    for s in NEW:
        M.train_seed(run, s)
    for s in NEW:
        M.infer_seed(run, s)
    rows = M.seed_rows(run, SEEDS)

    ens = {o: {} for o in M.ORDER_NAMES}
    disagree = {"any": 0, "pairs": {f"{a}-{b}": 0 for a, b in PAIRS}, "n": 0}
    for o in M.ORDER_NAMES:
        acc = {}                                            # 名稱 → 逐折 {wp, cil}

        def add(name, r):
            acc.setdefault(name, []).append({"wp": r["wp"], "cil": r["cil"]})

        for f in run.folds:
            K = M.fold_ctx(run, f, o)
            V, vt = K["V"], K["V"]["task"]
            da, sa, wp_pred = {}, {}, {}
            for s in SEEDS:
                I6 = M.seed_cos8(run, s, "test", f)
                da[s], sa[s] = M.d_heads(I6), M.sig(M.d_heads(M.seed_cos8(run, s, "val", f)), vt)
                wp_pred[s] = M.main_eval(I6, K["ar"], K["task"], K["label"])["_wp_pred"]
            db3 = M.d_cols(M.lin8_stage(run.st, f, o, 4, K["S"]["mean_vec"], M.G_LO))
            sb3 = M.sigma_b_table(run, f, o, V, M.G_LO)[1]
            for s in SEEDS:
                add(f"LG({s})", M.eval_comps([(da[s], sa[s], 1.0), (K["db"], K["sb"], 1.0)], K))
            for s, s2 in PAIRS:
                add(f"LL({s},{s2})", M.eval_comps([(da[s], sa[s], 1.0), (da[s2], sa[s2], 1.0)], K))
            add("GG", M.eval_comps([(K["db"], K["sb"], 1.0), (db3, sb3, 1.0)], K))
            add("LLL", M.eval_comps([(da[s], sa[s], 1.0) for s in SEEDS], K))
            if o == M.ORDER_NAMES[0]:                       # 告訴任務的判定與序無關
                disagree["n"] += len(K["task"])
                disagree["any"] += int(((wp_pred[42] != wp_pred[43]) | (wp_pred[43] != wp_pred[44])).sum())
                for a, b in PAIRS:
                    disagree["pairs"][f"{a}-{b}"] += int((wp_pred[a] != wp_pred[b]).sum())
        main = {s: {k: [x[k] for x in rows[str(s)][o]["main"]] for k in ("wp", "cil")} for s in SEEDS}

        def gain(name, s):
            return {k: M.paired([x[k] for x in acc[name]], main[s][k]) for k in ("wp", "cil")}

        table = []
        for s in SEEDS:
            table.append({"name": f"LG({s})", "vs_seed": s, "abs": acc[f"LG({s})"], "gain": gain(f"LG({s})", s)})
        for s, s2 in PAIRS:
            table.append({"name": f"LL({s},{s2})", "vs_seed": s, "abs": acc[f"LL({s},{s2})"], "gain": gain(f"LL({s},{s2})", s)})
        for name in ("GG", "LLL"):
            for s in SEEDS:
                table.append({"name": name, "vs_seed": s, "abs": acc[name], "gain": gain(name, s)})
        # G-ENS：每折 D = (1/3) Σ [WP(LG(s)) − WP(LL(s, s′))]
        D = [C.mean(acc[f"LG({s})"][i]["wp"] - acc[f"LL({s},{s2})"][i]["wp"] for s, s2 in PAIRS) for i in range(n)]
        ens[o] = {"table": table, "g_ens": {"per_fold": D, "mean": C.mean(D), "wins": sum(d > 1e-12 for d in D),
                                             "pass": bool(C.mean(D) >= 0.005 and sum(d > 1e-12 for d in D) >= 7)}}
    g_ens = {"orders": {o: ens[o]["g_ens"] for o in M.ORDER_NAMES},
             "pass": bool(all(ens[o]["g_ens"]["pass"] for o in M.ORDER_NAMES)), "valid": run.full}
    return {"K1": k1, "K3": k3, "seeds": list(SEEDS), "rows": rows, "ensemble": {o: ens[o]["table"] for o in M.ORDER_NAMES},
            "disagree": disagree, "G-M3c": gate_m3c(rows, run.full), "G-ENS": g_ens}


if __name__ == "__main__":
    raise SystemExit(M.stage_main("s4", body))
