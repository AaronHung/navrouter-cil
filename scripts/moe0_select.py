#!/usr/bin/env python3
"""MOE-0：只讀 validation 的選擇（PREREG-16 操作定義 15、20、24）與 validation 一致性檢查（5e、5f）。

    NAVCIL_MACHINE=mac python scripts/moe0_select.py
輸出：outputs/navcil/<machine>/moe0/selection.json（commit 之後才可執行 scripts/moe0_report.py）
不讀任何 test 快取。
"""
from __future__ import annotations

import json
import sys

import torch

import moe0_common as C
from selector.text_encoder import build_f_txt


def main() -> int:
    st = C.Store()
    ls = float(build_f_txt(st.tasks[0]).logit_scale)
    checks = {}
    # ── 5e：validation 自家任務 expert 與 nc7 的 i6/eval_fold{f}.pt 比對 ──
    maxd, argmax_ok, wp_fold = 0.0, True, []
    for f in C.FOLDS:
        S = st.split("val", f)
        e = torch.load(st.b.out / "i6" / f"eval_fold{f}.pt", map_location="cpu")["val"]
        accs = []
        for p, t in enumerate(st.tasks):
            mine = S["per_task"][t]["I6_cos8"][:, p]
            old = e[t]["r2"]["cos8"][:, 0]
            assert torch.equal(e[t]["labels"], S["per_task"][t]["labels"]), (f, t, "labels")
            maxd = max(maxd, (mine - old).abs().max().item())
            rows = torch.tensor([2 * p, 2 * p + 1])
            argmax_ok &= bool(torch.equal(mine[:, rows].argmax(-1), old[:, rows].argmax(-1)))
            accs.append((rows[mine[:, rows].argmax(-1)] == S["per_task"][t]["labels"]).float().mean().item())
        wp_fold.append(C.eq4(accs))
    wp_val = C.mean(wp_fold)
    checks["5e"] = {"max_abs_cos_diff": maxd, "argmax_all_equal": argmax_ok}
    checks["5f"] = {"val_wp_main": wp_val, "rounded": round(wp_val, 4)}
    if maxd > 1e-6 or not argmax_ok:
        print("一致性檢查 5e 不符", checks["5e"]); return 3
    if round(wp_val, 4) != 0.9194:
        print("一致性檢查 5f 不符", checks["5f"]); return 3

    # ── B4：選 λ（validation 告訴任務 Masked ACC；z 與 z′ 兩個版本）──
    out = {"checks": checks, "logit_scale": ls, "B4": {}, "B5": {}}
    for variant, center in (("z", True), ("zprime", False)):
        grid = {lam: [] for lam in C.LAMBDAS}
        for f in C.FOLDS:
            S = st.split("val", f)
            lin = st.lin8(f, "reverse", S["mean_vec"])
            da_all, db_all = C.all_task_diffs(S["I6_cos8"], lin)
            task, label = S["task"], S["labels"]
            da, db = da_all.gather(1, task[:, None]).squeeze(1), db_all.gather(1, task[:, None]).squeeze(1)
            pa, pb = C.zparams(da, task), C.zparams(db, task)
            for lam in C.LAMBDAS:
                ok = C.fused_masked_ok(da, db, task, label, pa, pb, lam, center)
                grid[lam].append(C.eq4(C.task_mean(ok, task)))
        means = {lam: C.mean(v) for lam, v in grid.items()}
        best = max(means.values())
        star = min(l for l, v in means.items() if v == best)
        out["B4"][variant] = {"val_masked_by_lambda": {str(l): v for l, v in means.items()},
                              "val_masked_per_fold": {str(l): v for l, v in grid.items()}, "lambda_star": star}
    # ── B5：選 T（validation CIL ACC，兩序各自）──
    for order in C.ORDER_NAMES:
        grid = {T: [] for T in C.TEMPS}
        for f in C.FOLDS:
            S = st.split("val", f)
            ar = st.ar(f, order, S["mean_vec"])
            for T in C.TEMPS:
                pred = C.soft_gate_pred(S["I6_cos8"], ar, T, ls)
                grid[T].append(C.eq4(C.task_mean(pred == S["labels"], S["task"])))
        means = {T: C.mean(v) for T, v in grid.items()}
        best = max(means.values())
        star = min(T for T, v in means.items() if v == best)
        out["B5"][order] = {"val_cil_by_T": {str(T): v for T, v in means.items()},
                            "val_cil_per_fold": {str(T): v for T, v in grid.items()}, "T_star": star}
    path = st.root / "selection.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({"checks": checks, "B4": {k: {"val": v["val_masked_by_lambda"], "lambda_star": v["lambda_star"]}
                                               for k, v in out["B4"].items()},
                      "B5": {o: {"val": v["val_cil_by_T"], "T_star": v["T_star"]} for o, v in out["B5"].items()}},
                     indent=1))
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
