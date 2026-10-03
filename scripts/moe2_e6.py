#!/usr/bin/env python3
"""MOE-2 E6：每類別（PREREG-18 E6、細則 22；seed 42、test、告訴任務、t = 4、reverse；不設門檻）。

  主系統、M3、LR、LR0、LRG、LR-bal、M3-bal（seed 42；LR 另報 seed 42–46 合計）：8 類各自的張數、正確張數、正確率；
  每任務 balanced accuracy 與四任務平均；LR 相對主系統的修好／弄壞／都錯／都對（每類別、每任務）

    NAVCIL_MACHINE=mac python scripts/moe2_e6.py --device cpu [--folds 1-10] [--out moe2]
輸出：outputs/navcil/<machine>/<out>/e6.json、e6.done
"""
from __future__ import annotations

import torch

import moe2_common as E
import moe2_e5 as E5
from moe2_common import C, M

O = "reverse"
GROUPS = ["修好（主系統錯→LR 對）", "弄壞（主系統對→LR 錯）", "都錯", "都對"]


def tell_preds(run: E.Run, rd: E.Reader, g: float) -> list[torch.Tensor]:
    out = []
    for f in run.folds:
        D = E.vecs(run, "test", f)
        out.append(E.pred_of(rd.diff(f, O, 4, g, D, D["task"]), D["task"]))
    return out


def per_class(preds: list[torch.Tensor], labels: list[torch.Tensor]) -> dict:
    n, ok = [0] * 8, [0] * 8
    for p, y in zip(preds, labels):
        for c in range(8):
            m = y == c
            n[c] += int(m.sum()); ok[c] += int((p[m] == c).sum())
    acc = [ok[c] / n[c] if n[c] else None for c in range(8)]
    bal = [(acc[2 * q] + acc[2 * q + 1]) / 2 for q in range(4)]
    return {"n": n, "correct": ok, "acc": acc, "bal_acc": bal, "bal_acc_mean": C.eq4(bal),
            "acc_all": sum(ok) / sum(n)}


def body(run: E.Run) -> dict:
    run.total(4)
    k1 = M.k1(run)
    labels = [E.vecs(run, "test", f)["label"] for f in run.folds]
    tasks = [E.vecs(run, "test", f)["task"] for f in run.folds]
    rd = {"LR": E.Reader(run, "LR(42)", "s42"), "LR0": E.Reader(run, "LR0", "g0"),
          "LRG": E.Reader(run, "LRG(42)", "s42", use_mv=True),
          "LR-bal": E.Reader(run, "LR-bal(42)", "s42", bal=True, grid=E.BAL_GAMMAS)}
    grb = E.Reader(run, "GR-bal", None, use_v=False, use_mv=True, bal=True, grid=E.BAL_GAMMAS)
    gam = {k: E.select_gamma(run, r) for k, r in rd.items()}
    gam["GR-bal"] = E.select_gamma(run, grb)
    run.tick()
    base = E.seed_base(run, 42)[O]
    preds = {"主系統": [x["_wp_pred"] for x in base["main"]], "M3": [x["_wp_pred"] for x in base["M3"]]}
    for k, r in rd.items():
        preds[k] = tell_preds(run, r, gam[k]["star"])
    preds["M3-bal"] = [x["_wp_pred"] for x in E5.m3bal_eval(run, grb, gam["GR-bal"]["star"], 42)[O]]
    run.tick()
    systems = ["主系統", "M3", "LR", "LR0", "LRG", "LR-bal", "M3-bal"]
    table = {k: per_class(preds[k], labels) for k in systems}
    # LR 五個 seed 合計
    lr5_p, lr5_y = [], []
    for s in E.SEEDS:
        lr5_p += tell_preds(run, E.Reader(run, f"LR({s})", E.kind_of(s)), gam["LR"]["star"])
        lr5_y += labels
    table["LR（seed 42–46 合計）"] = per_class(lr5_p, lr5_y)
    run.tick()
    # LR 相對主系統（seed 42）
    cnt, cnt_t = [[0] * 4 for _ in range(8)], [[0] * 4 for _ in range(4)]
    for pm, pl, y, tk in zip(preds["主系統"], preds["LR"], labels, tasks):
        a, b = pm == y, pl == y
        grp = torch.where(~a & b, 0, torch.where(a & ~b, 1, torch.where(~a & ~b, 2, 3)))
        for i in range(len(y)):
            cnt[int(y[i])][int(grp[i])] += 1; cnt_t[int(tk[i])][int(grp[i])] += 1
    run.tick()
    return {"K1": k1, "order": O, "gamma": gam, "systems": systems + ["LR（seed 42–46 合計）"], "table": table,
            "class_names": C.LABEL, "groups": GROUPS, "fix_break_by_class": cnt, "fix_break_by_task": cnt_t}


if __name__ == "__main__":
    raise SystemExit(E.stage_main("e6", body))
