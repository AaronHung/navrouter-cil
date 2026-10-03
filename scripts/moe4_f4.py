#!/usr/bin/env python3
"""MOE-4 f4：H4 γ 敏感度（PREREG-20 細則 27；只作描述，不設門檻、不改定案 γ = 1e-3）。

  P-F 在 γ ∈ {1e-4, 1e-3, 1e-2}：每折先對 seed 42–46 平均 CIL ACC_4 與 Ā，兩序分別列；再算十折 mean ± sd

    NAVCIL_MACHINE=mac python scripts/moe4_f4.py --device cpu [--folds 1-10] [--out moe4]
輸出：outputs/navcil/<machine>/<out>/f4.json、f4.done
"""
from __future__ import annotations

import moe4_common as X
from moe4_common import T


def body(run: T.Run) -> dict:
    X.require_chk(run)
    specs = [sp for sp in X.system_specs(gammas_sens=True) if sp[1] in ("P-F", "P-F-sens")]
    run.total(len(run.folds) + 1)
    per = {}
    for f in run.folds:
        D = X.data4(run, "test", f)
        res = X.eval_fold(run, f, D, specs)
        for o in X.ORDERS:
            for g in X.GAMMA_SENS:
                tag = f"{o}|{g:g}"
                rs = [res[(f"P-F({s})" if g == X.G else f"P-F({s})γ{g:g}", o)] for s in X.SEEDS]
                per.setdefault(tag, {"cil4": [], "abar": []})
                per[tag]["cil4"].append(sum(r["acc_t"][3] for r in rs) / len(rs))
                per[tag]["abar"].append(sum(sum(r["acc_t"]) / 4 for r in rs) / len(rs))
        run.drop()
        run.tick()
    out = {tag: {"per_fold": v, "cil4": X.mean_sd(v["cil4"]), "abar": X.mean_sd(v["abar"])} for tag, v in per.items()}
    run.tick()
    return {"folds": run.folds, "gammas": list(X.GAMMA_SENS), "gamma_star": X.G, "table": out,
            "note": "只作描述；定案 γ = 1e-3（PREREG-20 細則 6）"}


if __name__ == "__main__":
    raise SystemExit(X.stage_main("f4", body))
