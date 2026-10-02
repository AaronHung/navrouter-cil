#!/usr/bin/env python3
"""MOE-1 S1：修好與弄壞（PREREG-17 S1、細則 11–12）。不重跑推論。

主系統與 M3（σ 重算版：t = 4、reverse 序的 LIN8，σ 由 moe0 的 validation 快取算）的告訴任務判定，
d_a、d_b 取自 moe0/b3_per_slide.csv；另以快取的未捨入值重算同一判定，報不同的張數。

    NAVCIL_MACHINE=mac python scripts/moe1_s1.py --device cpu [--folds 1-10] [--out moe1]
輸出：outputs/navcil/<machine>/<out>/s1.json、s1.done
"""
from __future__ import annotations

import csv

import torch

import moe1_common as M
from moe1_common import C

GROUPS = ["修好（主系統錯→M3 對）", "弄壞（主系統對→M3 錯）", "都錯", "都對"]


def med(x: torch.Tensor):
    return float(x.double().median()) if len(x) else None


def body(run: M.Run) -> dict:
    st, o = run.st, "reverse"
    k1 = M.k1(run)
    rows = {}
    with open(st.root / "b3_per_slide.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(int(r["fold"]), []).append(r)
    cnt = [[0] * 4 for _ in range(8)]                      # [類別][組]
    cnt_task = [[0] * 4 for _ in range(4)]
    cols = {k: [] for k in ("grp", "n_patch", "za", "zb", "margin", "tp_ok")}
    zero_da = mism_main = mism_m3 = n_all = 0
    run.total(len(run.folds))
    for f in run.folds:
        S, V = st.split("test", f), st.split("val", f)
        task, label = S["task"], S["labels"]
        rf = rows[f]
        if [r["slide_id"] for r in rf] != S["sids"] or [C.LABEL.index(r["true_class"]) for r in rf] != label.tolist():
            raise M.CheckFailed(f"fold {f}: b3_per_slide.csv 與 moe0 test 快取的 slide 順序不一致")
        n_patch = []
        for t in run.tasks:
            c = torch.load(st.b.cache / f"fold{f}_test_{t}.pt", map_location="cpu")
            if c["sids"] != S["per_task"][t]["sids"]:
                raise M.CheckFailed(f"fold {f} {t}: n_patch 快取的 slide id 未對齊")
            n_patch.append(c["n_patch"])
        n_patch = torch.cat(n_patch)
        # σ：t = 4、reverse 序，validation 重算
        sa = M.sig(M.d_heads(V["I6_cos8"]), V["task"])
        sb = M.sig(M.d_cols(M.lin8_stage(st, f, o, 4, V["mean_vec"], M.G_LIN8)), V["task"])
        first = (label - 2 * task) == 0
        # CSV 的值（8 位小數）
        da = torch.tensor([float(r["d_a"]) for r in rf], dtype=M.D64)
        db = torch.tensor([float(r["d_b"]) for r in rf], dtype=M.D64)
        za, zb = M.zs(da, task, sa), M.zs(db, task, sb)
        fz = za + zb
        main_first = da >= 0
        m3_first = (fz > 0) | ((fz == 0) & main_first)
        zero_da += int((da == 0).sum())
        # 快取的未捨入值
        dax = M.pick(M.d_heads(S["I6_cos8"]), task)
        dbx = M.d_cols(M.lin8_stage(st, f, o, 4, S["mean_vec"], M.G_LIN8))
        _, m3x = M.fused([(M.d_heads(S["I6_cos8"]), sa, 1.0), (dbx, sb, 1.0)], task)
        mism_main += int((main_first != (dax >= 0)).sum()); mism_m3 += int((m3_first != m3x).sum())
        a_ok, m_ok = main_first == first, m3_first == first
        grp = torch.where(~a_ok & m_ok, 0, torch.where(a_ok & ~m_ok, 1, torch.where(~a_ok & ~m_ok, 2, 3)))
        ar = M.ar_stage(st, f, o, 4, S["mean_vec"])
        top = ar.topk(2, dim=-1).values
        for i in range(len(task)):
            cnt[int(label[i])][int(grp[i])] += 1; cnt_task[int(task[i])][int(grp[i])] += 1
        cols["grp"].append(grp); cols["n_patch"].append(n_patch); cols["za"].append(za.abs()); cols["zb"].append(zb.abs())
        cols["margin"].append(top[:, 0] - top[:, 1]); cols["tp_ok"].append(ar.argmax(-1) == task)
        n_all += len(task)
        run.tick()
    cat = {k: torch.cat(v) for k, v in cols.items()}
    stats = []
    for g in range(4):
        m = cat["grp"] == g
        stats.append({"group": GROUPS[g], "n": int(m.sum()), "median_n_patch": med(cat["n_patch"][m]),
                      "median_abs_za": med(cat["za"][m]), "median_abs_zb": med(cat["zb"][m]),
                      "median_ar_margin": med(cat["margin"][m]),
                      "tp_correct": float(cat["tp_ok"][m].float().mean()) if int(m.sum()) else None})
    return {"K1": k1, "groups": GROUPS, "class_names": C.LABEL, "counts_by_class": cnt, "counts_by_task": cnt_task,
            "group_stats": stats, "n_slides": n_all, "csv_zero_da_rows": zero_da,
            "main_wrong_total": sum(c[0] + c[2] for c in cnt), "m3_wrong_total": sum(c[1] + c[2] for c in cnt),
            "mismatch_vs_cache": {"main": mism_main, "M3": mism_m3}}


if __name__ == "__main__":
    raise SystemExit(M.stage_main("s1", body))
