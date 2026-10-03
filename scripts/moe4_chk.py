#!/usr/bin/env python3
"""MOE-4 chk：一致性檢查 K1–K5（PREREG-20「一致性檢查」、細則 10–14）。任何一項不過就停下（不重試、不改期望值）。

  K1 P-main(42)：WP、CIL ACC 對 nc8/per_fold.json（逐折、逐任務）；逐階段 ACC 對 moe1/s2.json
  K2 P-4(42)：t = 4 的 WP、CIL ACC 對 moe1/s5.json 的 LR（γ = 1e-3）
  K3 P-0：逐階段 ACC、WP、Forgetting、BWT 對 moe3/f2.json 的 S-R0
  K4 P-main(s)，s = 43…46：t = 4 的 CIL ACC 對 moe2/e1.json
  K5 w(42)·Fᵀ 對 moe1 s42cells「一次取 64／等權」格（val 與 test）

    NAVCIL_MACHINE=mac python scripts/moe4_chk.py --device cpu [--folds 1-10] [--out moe4]
輸出：outputs/navcil/<machine>/<out>/chk.json、chk.done
"""
from __future__ import annotations

import json

import torch

import moe4_common as X
from moe4_common import E, M, T

TOL = 1e-9


def body(run: T.Run) -> dict:
    run.total(len(run.folds) + 1)
    specs = X.system_specs()
    nc8 = M.nc8_main(run)
    s2 = json.loads((run.m1 / "s2.json").read_text())
    s5 = json.loads((run.m1 / "s5.json").read_text())
    e1 = json.loads((run.m2 / "e1.json").read_text())
    f2m3 = json.loads((run.base / "moe3" / "f2.json").read_text())
    ctx = run.ctx
    K1 = {"max_abs_fold_diff": 0.0, "stage_acc_max_abs": 0.0, "wp": {o: [] for o in X.ORDERS}, "cil": {o: [] for o in X.ORDERS}}
    K2 = {"max_abs_fold_diff": 0.0, "wp": {o: [] for o in X.ORDERS}, "cil": {o: [] for o in X.ORDERS}}
    K3 = {"max_abs_fold_diff": 0.0}
    K4 = {"max_abs_fold_diff": 0.0, "per_seed": {}}
    K5 = {"max_abs_cos": 0.0, "argmax_equal": True}
    for f in run.folds:
        D = X.data4(run, "test", f)
        res = X.eval_fold(run, f, D, specs)
        i2, i5, i3, i1 = T.fold_i(s2, f), T.fold_i(s5, f), T.fold_i(f2m3, f), T.fold_i(e1, f)
        for o in X.ORDERS:
            # K1
            r = res[("P-main(42)", o)]
            j = nc8[o][f - 1]
            K1["max_abs_fold_diff"] = max(K1["max_abs_fold_diff"], abs(r["acc_t"][3] - j["acc"]),
                                          *[abs(r["wp_task_t"][3][q] - j["wp_task"][q]) for q in range(4)])
            ref = s2["orders"][o]["main"][i2]["acc_t"]
            K1["stage_acc_max_abs"] = max(K1["stage_acc_max_abs"], *[abs(a - b) for a, b in zip(r["acc_t"], ref)])
            K1["wp"][o].append(r["wp_t"][3]); K1["cil"][o].append(r["acc_t"][3])
            # K2（P-4(42) = v(42)／RDG(1e-3)）
            r = res[("P-4(42)", o)]
            w_ref, c_ref = s5["splits"]["test"]["alone"]["LR"]["wp"][i5], s5["lr_cil"][o]["cil"][i5]
            K2["max_abs_fold_diff"] = max(K2["max_abs_fold_diff"], abs(r["wp_t"][3] - w_ref), abs(r["acc_t"][3] - c_ref))
            K2["wp"][o].append(r["wp_t"][3]); K2["cil"][o].append(r["acc_t"][3])
            # K3（P-0 = u_64／RDG(1e-3) 對 MOE-3 的 S-R0）
            K3["max_abs_fold_diff"] = max(K3["max_abs_fold_diff"],
                                          T.row_diff(res[("P-0", o)], f2m3["rows"][o]["S-R0"][i3]))
            # K4（P-main(s)，s = 43…46）
            for s in X.NEW:
                ref = e1["seeds"][str(s)]["orders"][o]["main"][i1]["cil"]
                d = abs(res[(f"P-main({s})", o)]["acc_t"][3] - ref)
                K4["max_abs_fold_diff"] = max(K4["max_abs_fold_diff"], d)
                K4["per_seed"].setdefault(str(s), {}).setdefault(o, 0.0)
                K4["per_seed"][str(s)][o] = max(K4["per_seed"][str(s)][o], d)
        # K5（w(42) 的 cosine 對 s42cells 的「一次取 64／等權」格）
        for split in ("val", "test"):
            Ds = X.data4(run, split, f)
            cos = E.cos_rows(Ds["v"]["w42"], ctx.F.t())
            cells = M.load_cat(run, run.m1 / "cache" / f"s42cells_{split}_fold{f}.pt", split, f, ["cells_cos8"])["cells_cos8"][:, :, 0]
            K5["max_abs_cos"] = max(K5["max_abs_cos"], float((cos - cells).abs().max()))
            for q in range(4):
                rows = [2 * q, 2 * q + 1]
                K5["argmax_equal"] &= bool(torch.equal(cos[:, q][:, rows].argmax(-1), cells[:, q][:, rows].argmax(-1)))
        run.drop()
        run.tick()

    K1["wp_mean"] = {o: X.C.mean(v) for o, v in K1["wp"].items()}
    K1["cil_mean"] = {o: X.C.mean(v) for o, v in K1["cil"].items()}
    K2["wp_mean"] = {o: X.C.mean(v) for o, v in K2["wp"].items()}
    K2["cil_mean"] = {o: X.C.mean(v) for o, v in K2["cil"].items()}
    ok1 = K1["max_abs_fold_diff"] <= TOL and K1["stage_acc_max_abs"] <= TOL
    ok2 = K2["max_abs_fold_diff"] <= TOL
    if run.full:
        ok1 &= all(round(v, 4) == 0.9340 for v in K1["wp_mean"].values()) and all(round(v, 4) == 0.9128 for v in K1["cil_mean"].values())
        ok2 &= all(round(v, 4) == 0.9457 for v in K2["wp_mean"].values()) and all(round(v, 4) == 0.9252 for v in K2["cil_mean"].values())
    K1["pass"], K1["rounding_checked"] = bool(ok1), run.full
    K2["pass"], K2["rounding_checked"] = bool(ok2), run.full
    K3["pass"] = bool(K3["max_abs_fold_diff"] <= TOL)
    K4["pass"] = bool(K4["max_abs_fold_diff"] <= TOL)
    K5["pass"] = bool(K5["max_abs_cos"] <= 1e-6 and K5["argmax_equal"])
    res = {"folds": run.folds, "full": run.full, "K1": K1, "K2": K2, "K3": K3, "K4": K4, "K5": K5}
    bad = [k for k in ("K1", "K2", "K3", "K4", "K5") if not res[k]["pass"]]
    T.log("chk：" + json.dumps({k: {x: y for x, y in res[k].items() if x not in ("wp", "cil")} for k in ("K1", "K2", "K3", "K4", "K5")},
                             ensure_ascii=False, default=str))
    res["pass"] = not bad
    if bad:
        raise T.CheckFailed(f"{'、'.join(bad)} 不符：" + json.dumps({k: res[k] for k in bad}, ensure_ascii=False, default=str))
    return X.write_json_ok(res)


if __name__ == "__main__":
    raise SystemExit(X.stage_main("chk", body))
