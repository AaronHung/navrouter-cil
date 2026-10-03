#!/usr/bin/env python3
"""MOE-3 f5：K 的影響（PREREG-19 F5、細則 24；不設門檻）。

  S-T0、S-R0、S-A0 的判讀器套用在 u_16、u_32、u_64、u_128、u_256 與 mean_vec（全部 patch）：
  t = 4 的 WP、CIL ACC 與 Ā（兩序）。RDG、ANC 一律用 u_64 選定的超參數，不重選。

    NAVCIL_MACHINE=mac python scripts/moe3_f5.py --device cpu [--folds 1-10] [--out moe3]
輸出：outputs/navcil/<machine>/<out>/f5.json、f5.done
"""
from __future__ import annotations

import moe3_common as T
from moe3_common import M

KINDS = [T.ukind(k) for k in T.KS] + ["mv"]


def body(run: T.Run) -> dict:
    run.total(len(run.folds) + 1)
    k1 = M.k1(run)
    hp = T.load_hp(run)
    rds = {"S-T0": T.TXT, "S-R0": T.star_rd(hp, "RDG", T.U_MAIN), "S-A0": T.star_rd(hp, "ANC", T.U_MAIN)}
    out = {o: {n: {k: [] for k in KINDS} for n in rds} for o in T.ORDER_NAMES}
    run.tick()
    for f in run.folds:
        D = T.data(run, "test", f)
        for o in T.ORDER_NAMES:
            ar = T.ar_stages(run, f, o, D)
            for n, rd in rds.items():
                for k in KINDS:
                    r = T.run_cl(run, f, o, D, T.reader_fn(run, f, o, D, k, rd), ar)
                    out[o][n][k].append({"wp": r["wp_t"][3], "cil": r["acc_t"][3], "abar": r["abar"]})
        run.drop()
        run.tick()
    return {"K1": k1, "kinds": KINDS, "readers": {n: list(rd) for n, rd in rds.items()}, "orders": out}


if __name__ == "__main__":
    raise SystemExit(T.stage_main("f5", body))
