#!/usr/bin/env python3
"""MOE-2 E2：是否需要 head（PREREG-18 E2、細則 18；門檻 G-HEAD）。

  LR0 = v0（g = 0、只用 s0 的四輪等權向量）的 ridge，自己選 γ；WP、CIL ACC、每任務 WP（兩序）
  D_head（每序、每折）= 五個 seed 的 WP(LR(s)) 平均 − WP(LR0)
  LT 在 g = 0 的 test WP（v0·Fᵀ 的 2 類 argmax），與 moe0/results.json 的 B2.matrix[4] 比對

    NAVCIL_MACHINE=mac python scripts/moe2_e2.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e2.json、e2.done
"""
from __future__ import annotations

import json

import torch

import moe2_common as E
from moe2_common import C, M


def body(run: E.Run) -> dict:
    run.total(len(E.SEEDS) + 3)
    k1 = M.k1(run)
    lr = {s: E.Reader(run, f"LR({s})", E.kind_of(s)) for s in E.SEEDS}
    g_lr = E.select_gamma(run, lr[42])
    lr0 = E.Reader(run, "LR0", "g0")
    g0 = E.select_gamma(run, lr0)
    run.tick()
    ev0 = E.evaluate(run, lr0, g0["star"])
    run.tick()
    wp_lr = {}
    for s in E.SEEDS:
        ev = E.evaluate(run, lr[s], g_lr["star"])
        wp_lr[str(s)] = {o: E.col(ev[o], "wp") for o in E.ORDER_NAMES}
        run.tick()
    d_head = {o: [C.mean(wp_lr[str(s)][o][i] for s in E.SEEDS) - ev0[o][i]["wp"] for i in range(len(run.folds))]
              for o in E.ORDER_NAMES}
    g_head = E.gate(d_head, 0.005)
    g_head["valid"] = run.full

    # LT 在 g = 0（test、真實任務的 v0、2 類 argmax，同 moe0_report 的 B2）
    Ft = run.ctx.F.t()
    per_t, wp4 = [[] for _ in range(4)], []
    for f in run.folds:
        D = E.vecs(run, "test", f)
        cos = E.cos_rows(D["v"]["g0"][torch.arange(len(D["task"])), D["task"]], Ft)
        accs = []
        for j in range(4):
            m = D["task"] == j
            rows = torch.tensor([2 * j, 2 * j + 1])
            a = (rows[cos[m][:, rows].argmax(-1)] == D["label"][m]).float().mean().item()
            per_t[j].append(a); accs.append(a)
        wp4.append(C.eq4(accs))
    b2 = json.loads((run.st.root / "results.json").read_text())["B2"]["matrix"][4]
    lt_g0 = {"wp": wp4, "wp_t_mean": [C.mean(x) for x in per_t], "b2_ref": b2,
             "b2_max_abs": max(abs(C.mean(per_t[j]) - b2[j]) for j in range(4)), "b2_comparable": run.full}
    run.tick()
    return {"K1": k1, "gamma_LR": g_lr, "gamma_LR0": g0, "LR0": E.pub(ev0), "order_same_LR0": E.order_same(ev0),
            "wp_LR": wp_lr, "D_head": d_head, "G-HEAD": g_head, "LT_g0": lt_g0}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e2", body))
