#!/usr/bin/env python3
"""MOE-3 chk：一致性檢查 K1–K5（PREREG-19「一致性檢查」、細則 9–13）。任何一項不過就停下（不重試、不改期望值）。

  K1 主系統 seed 42 對 nc8/per_fold.json
  K2 RDG(γ = 1e-3)、輸入 v0：t = 4 的 test WP／CIL ACC 對 moe2/e2.json 的 LR0
  K3 ANC(γ, α = 0) 與 RDG(γ) 的 W（u_64、reverse、t = 4、γ = 1e-2、第一折）
  K4 ANC(γ = 1e10, α = 1；AMENDMENT-6，原為 1e8)、輸入 v0：test 告訴任務判定與 TXT（v0）逐張相同（t = 4、兩序、每折）
  K5 TXT（v0）每任務 WP 對 MOE-0 B2 的「g = 0」列；另每折對 moe0 快取 g0_cos8 的 2 類 argmax

    NAVCIL_MACHINE=mac python scripts/moe3_chk.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/chk.json、chk.done
"""
from __future__ import annotations

import json

import torch

import moe3_common as T
from moe3_common import C, E, M

G_K2, G_K3, G_K4 = 1e-3, 1e-2, 1e10          # K4 的 γ：AMENDMENT-6（PREREG-19 原為 1e8）


def body(run: T.Run) -> dict:
    run.total(len(run.folds) + 1)
    k1 = M.k1(run)
    run.tick()
    e2 = json.loads((run.m2 / "e2.json").read_text())
    k2 = {"max_abs_fold_diff": 0.0, "wp": {o: [] for o in T.ORDER_NAMES}, "cil": {o: [] for o in T.ORDER_NAMES}}
    k3 = None
    k4 = {"gamma": G_K4, "diff": {o: 0 for o in T.ORDER_NAMES}, "n": 0, "d_txt_min_abs": float("inf")}
    k5 = {"per_task": [[] for _ in range(4)], "cache_max_abs": 0.0}
    for f in run.folds:
        D = T.data(run, "test", f)
        task, label = D["task"], D["label"]
        d_txt = T.txt_d(run, D, "g0", task)
        txt = E.pred_of(d_txt, task)
        k4["n"] += len(task)
        k4["d_txt_min_abs"] = min(k4["d_txt_min_abs"], float(d_txt.abs().min()))
        for o in T.ORDER_NAMES:
            # K2
            r = T.run_cl(run, f, o, D, T.reader_fn(run, f, o, D, "g0", ("RDG", G_K2)))
            ref = e2["LR0"][o][T.fold_i(e2, f)]
            k2["max_abs_fold_diff"] = max(k2["max_abs_fold_diff"], abs(r["wp_t"][3] - ref["wp"]), abs(r["acc_t"][3] - ref["cil"]))
            k2["wp"][o].append(r["wp_t"][3]); k2["cil"][o].append(r["acc_t"][3])
            # K4
            anc = E.pred_of(T.reader_fn(run, f, o, D, "g0", ("ANC", G_K4, 1.0))(4, task), task)
            k4["diff"][o] += int((anc != txt).sum())
        # K3（第一折）
        if k3 is None:
            st = T.stats(run, T.U_MAIN)
            Wr, _ = st.rdg(f, "reverse", 4, G_K3)
            Wa, _ = st.anc(f, "reverse", 4, G_K3, 0.0)
            k3 = {"fold": f, "gamma": G_K3, "kind": T.U_MAIN, "order": "reverse", "max_abs": float((Wr - Wa).abs().max())}
        # K5
        acc = C.task_mean(txt == label, task)
        g0 = run.st.split("test", f)["g0_cos8"]
        for q in range(4):
            k5["per_task"][q].append(acc[q])
            m = task == q
            rows = torch.tensor([2 * q, 2 * q + 1])
            ref_acc = (rows[g0[m][:, rows].argmax(-1)] == label[m]).float().mean().item()
            k5["cache_max_abs"] = max(k5["cache_max_abs"], abs(acc[q] - ref_acc))
        run.drop()
        run.tick()

    k2["wp_mean"] = {o: C.mean(v) for o, v in k2["wp"].items()}
    k2["cil_mean"] = {o: C.mean(v) for o, v in k2["cil"].items()}
    ok2 = k2["max_abs_fold_diff"] <= 1e-9
    if run.full:
        ok2 &= all(round(v, 4) == 0.9435 for v in k2["wp_mean"].values()) and all(round(v, 4) == 0.9214 for v in k2["cil_mean"].values())
    k2.update({"pass": bool(ok2), "rounding_checked": run.full})
    k3["pass"] = bool(k3["max_abs"] <= 1e-10)
    k4["pass"] = bool(all(v == 0 for v in k4["diff"].values()))
    b2 = json.loads((run.st.root / "results.json").read_text())["B2"]["matrix"][4]
    k5["wp_t_mean"] = [C.mean(x) for x in k5["per_task"]]
    k5.update({"b2_ref": b2, "b2_comparable": run.full,
               "b2_max_abs": max(abs(k5["wp_t_mean"][q] - b2[q]) for q in range(4)) if run.full else None})
    k5["pass"] = bool(k5["cache_max_abs"] <= 1e-9 and (not run.full or k5["b2_max_abs"] <= 1e-9))
    res = {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5}
    bad = [k for k in ("K2", "K3", "K4", "K5") if not res[k]["pass"]]
    T.log("K2–K5：" + json.dumps({k: {x: y for x, y in res[k].items() if x not in ("wp", "cil", "per_task")} for k in ("K2", "K3", "K4", "K5")},
                                ensure_ascii=False))
    if bad:
        raise T.CheckFailed(f"{'、'.join(bad)} 不符：" + json.dumps({k: res[k] for k in bad}, ensure_ascii=False))
    return res


if __name__ == "__main__":
    raise SystemExit(T.stage_main("chk", body))
