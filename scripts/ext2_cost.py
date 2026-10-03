#!/usr/bin/env python3
"""EXT-2 cost（PREREG-22 細則 10、16、21）：K11 與成本。fold 1、CPU；由特徵檔重算，不訓練、不寫任何既有產物。

K11(a) train 全部 slide：不載入 head，依 FINAL-B 定義（s0 → 四輪 → 等權平均、L2）重算 v，與快取的 v0（NOHEAD 列所用）比對。
K11(b) test 全部 slide：FINAL-B 的完整推論（mean_vec → AR → τ̂ 的 s0 → 四輪 → ridge），逐張判定與 A 階段（由快取算）比對。
推論秒數：FINAL-B 與 FINAL-A(42) 各跑一遍 test（各自讀檔、各自計時）；AR 與 ridge 的 W（t = 4、reverse）在計時前解好。
訓練秒數：每張 train slide 的讀檔、mean_vec、FINAL-B 的向量、FINAL-A(42) 的向量各自計時；closed-form 的累加與求解另計。

    NAVCIL_MACHINE=mac python scripts/ext2_cost.py --device cpu
輸出：outputs/navcil/<machine>/ext2/cost.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_c as CC                                                       # noqa: E402
import ext2_a as EA                                                       # noqa: E402
from ext1_c import D64, M, N5, T, X                                       # noqa: E402
from selector.cil_ops import four_round, mean_norm                        # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import text_nav_feats                         # noqa: E402
from selector.i6_expert import zscore                                     # noqa: E402

FOLD, ORDER, SEED = 1, "reverse", 42


def stat(xs) -> dict:
    xs = [float(x) for x in xs]
    return {"mean": statistics.fmean(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs), "sum": sum(xs), "n": len(xs)}


def s0_of(Z: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
    """FINAL-B 的分數：對該任務兩類文字的最大 cosine，在這張 slide 的全部 patch 內 z-score（不經任何 head）。"""
    return zscore(text_nav_feats(Z, ft)[:, 0])


def run(cx: EA.Ctx) -> dict:
    r3, r4 = cx.r3, cx.r4
    ctx = r3.ctx
    lam = ctx.lam()
    gB = json.loads((cx.base / "ext2" / "a.json").read_text())["gamma_B"]
    heads = [M.load_head(X.head_path(r4, SEED, FOLD, t)) for t in cx.tasks]
    W_ar, pos = cx.st.b.W(FOLD, ORDER, 4, M.G_AR)
    col = [pos.index(p) for p in range(4)]
    W_B, cls_B = T.stats(r3, EA.KIND_B).W(FOLD, ORDER, 4, ("RDG", gB))
    W_A, cls_A = X.acc_of(r4, f"s{SEED}").W(FOLD, ORDER, 4, CC.G)
    assert cls_B == cls_A == list(range(8))
    one = torch.ones(1, dtype=D64)
    load0 = list(os.getloadavg())

    def infer(which: str) -> dict:
        n_patch, n_sel, t_read, t_comp, preds, labels = [], [], [], [], [], []
        W = W_B if which == "B" else W_A
        with torch.no_grad():
            for task in cx.tasks:
                ds, shift = ctx.ds(FOLD, task, "test")
                for i in range(len(ds)):
                    t0 = time.perf_counter()
                    rec = read_slide(ds, shift, i)
                    t1 = time.perf_counter()
                    Z = rec.Z
                    th = int((torch.cat([mean_norm(Z).to(D64), one]) @ W_ar)[col].argmax())
                    ft = ctx.f_task(th)
                    score = s0_of(Z, ft) if which == "B" else heads[th](Z, ft)
                    idx = four_round(Z, score, lam)
                    s = torch.cat([mean_norm(Z, idx).to(D64), one]) @ W
                    pred = 2 * th + int(not bool(s[2 * th] - s[2 * th + 1] >= 0))
                    t2 = time.perf_counter()
                    n_patch.append(Z.shape[0]); n_sel.append(len(idx)); t_read.append(t1 - t0); t_comp.append(t2 - t1)
                    preds.append(pred); labels.append(rec.label)
        ok = [int(p == y) for p, y in zip(preds, labels)]
        ref_row = "FINALB" if which == "B" else "FINAL"
        ref = json.loads(cx.path(ref_row, ORDER, FOLD).read_text())
        n_diff = sum(x != y for x, y in zip(ok, ref["ok4"])) + abs(len(ok) - len(ref["ok4"]))
        pred_diff = sum(x != y for x, y in zip(preds, ref["cil4"])) if which == "B" else None
        return {"n_slides": len(ok), "acc_slide_level": sum(ok) / len(ok), "ok_diff_vs_A_stage": n_diff, "pred_diff_vs_A_stage": pred_diff,
                "n_patch": stat(n_patch), "n_selected": stat(n_sel), "t_read_s": stat(t_read), "t_compute_s": stat(t_comp),
                "t_total_s": stat([x + y for x, y in zip(t_read, t_comp)])}

    inf = {"FINALB": infer("B"), "FINALA": infer("A")}
    inf["FINALB_second_pass"] = {k: v for k, v in infer("B").items() if k.startswith("t_")}   # 讀檔已在系統快取內的第二遍，供對照

    # ── train：K11(a) 與每任務的取向量秒數 ──
    trn3, trn4 = T.data(r3, "train", FOLD), X.data4(r4, "train", FOLD)
    per_task, k11a, a_chk, mv_chk = {}, 0.0, 0.0, 0.0
    with torch.no_grad():
        for p, task in enumerate(cx.tasks):
            ds, shift = ctx.ds(FOLD, task, "train")
            ft = ctx.f_task(p)
            tr, tm, tb, ta, vB, vA, mvs, sids = [], [], [], [], [], [], [], []
            for i in range(len(ds)):
                t0 = time.perf_counter()
                rec = read_slide(ds, shift, i)
                t1 = time.perf_counter()
                Z = rec.Z
                mv = mean_norm(Z)
                t2 = time.perf_counter()
                b = mean_norm(Z, four_round(Z, s0_of(Z, ft), lam))
                t3 = time.perf_counter()
                a = mean_norm(Z, four_round(Z, heads[p](Z, ft), lam))
                t4 = time.perf_counter()
                tr.append(t1 - t0); tm.append(t2 - t1); tb.append(t3 - t2); ta.append(t4 - t3)
                vB.append(b); vA.append(a); mvs.append(mv); sids.append(rec.sid)
            if sids != trn3["raw"][task]["sids"]:
                raise M.CheckFailed(f"train {task}: slide 順序與快取不一致")
            k11a = max(k11a, float((torch.stack(vB) - trn3["per"][p]["v"][EA.KIND_B]).abs().max()))
            a_chk = max(a_chk, float((torch.stack(vA) - trn4["per"][p]["v"][f"s{SEED}"]).abs().max()))
            mv_chk = max(mv_chk, float((torch.stack(mvs) - trn4["per"][p]["mv"]).abs().max()))
            per_task[task] = {"n_train": len(sids), "read_s": sum(tr), "mean_vec_s": sum(tm), "vec_B_s": sum(tb), "vec_A_s": sum(ta)}

    # ── closed-form：每任務的累加與求解（兩序；fold 1）──
    cf = []
    eye = torch.eye(513, dtype=D64)
    for o in CC.ORDERS:
        A = {k: torch.zeros(513, 513, dtype=D64) for k in ("B", "A", "ar", "lin8")}
        cols = {k: [] for k in A}
        for p in r4.pos(o):
            d3, d4 = trn3["per"][p], trn4["per"][p]
            y = d4["label"]
            row = {"order": o, "task": cx.tasks[p], "n_train": len(y)}
            for key, v, g, two in (("B", d3["v"][EA.KIND_B], gB, True), ("A", d4["v"][f"s{SEED}"], CC.G, True),
                                   ("ar", d4["mv"], M.G_AR, False), ("lin8", d4["mv"], M.G_LIN8, True)):
                t0 = time.perf_counter()
                Xv = N5.aug(v)
                A[key] += Xv.t() @ Xv
                cols[key] += [Xv[y == c].sum(0) for c in (2 * p, 2 * p + 1)] if two else [Xv.sum(0)]
                t1 = time.perf_counter()
                torch.linalg.solve(A[key] + g * eye, torch.stack(cols[key], 1))
                row[f"{key}_accumulate_s"], row[f"{key}_solve_s"] = t1 - t0, time.perf_counter() - t1
            cf.append(row)

    # 每任務訓練秒數（fold 1；closed-form 取 reverse 序該任務那一步）：本批量到的部分，不含 head 訓練
    cfr = {r["task"]: r for r in cf if r["order"] == ORDER}
    train = {}
    for task, v in per_task.items():
        c = cfr[task]
        ar = c["ar_accumulate_s"] + c["ar_solve_s"]
        base = v["read_s"] + v["mean_vec_s"]
        train[task] = {"ZS8": 0.0, "LIN8": base + c["lin8_accumulate_s"] + c["lin8_solve_s"], "MAIN": base + ar,
                       "FINALB": base + ar + v["vec_B_s"] + c["B_accumulate_s"] + c["B_solve_s"],
                       "FINALA": base + ar + v["vec_A_s"] + c["A_accumulate_s"] + c["A_solve_s"]}
    head = {}
    for s in (43, 44, 45, 46):
        head[str(s)] = {t: statistics.fmean(json.loads((cx.base / "moe1" / f"i6_seed{s}" / f"fold{f}_{t}_train.json").read_text())["wall_s"]
                                             for f in range(1, 11)) for t in cx.tasks}
    head_avg = {t: statistics.fmean(head[s][t] for s in head) for t in cx.tasks}

    k11 = {"a_train_v_max_abs": k11a, "a_tol": 1e-6, "b_test_ok_diff": inf["FINALB"]["ok_diff_vs_A_stage"],
           "b_test_pred_diff": inf["FINALB"]["pred_diff_vs_A_stage"],
           "pass": bool(k11a <= 1e-6 and inf["FINALB"]["ok_diff_vs_A_stage"] == 0 and inf["FINALB"]["pred_diff_vs_A_stage"] == 0),
           "also": {"FINALA_train_v_max_abs_vs_cache": a_chk, "mean_vec_max_abs_vs_cache": mv_chk,
                    "FINALA_test_ok_diff": inf["FINALA"]["ok_diff_vs_A_stage"]}}
    return {"stage": "cost", "machine": "mac", "device": "cpu", "threads": torch.get_num_threads(), "fold": FOLD, "order": ORDER,
            "seed_A": SEED, "gamma_B": gB, "lambda": lam, "loadavg_start": load0, "loadavg_end": list(os.getloadavg()),
            "finished": M.now(), "K11": k11, "inference": inf, "train_per_task": per_task, "closed_form": cf, "train_seconds": train,
            "head_train_wall_s_recorded": {"per_seed": head, "four_seed_mean": head_avg,
                                           "note": "MOE-1 批的既有紀錄（moe1/i6_seed{43…46}/fold*_train.json 的 wall_s，十折平均）；本批沒有訓練 head"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.device != "cpu":
        raise SystemExit("EXT-2 cost 規定 --device cpu")
    sys.argv = sys.argv[:1]
    cx = EA.Ctx()
    root = cx.base / "ext2"
    try:
        res = run(cx)
    except Exception:
        (root / "FAILED_cost.txt").write_text(f"[{M.now()}] 階段 cost 失敗\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3
    (root / "cost.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._default))
    print(json.dumps({k: res[k] for k in ("K11", "loadavg_start", "loadavg_end", "train_seconds", "train_per_task")}, indent=1, ensure_ascii=False))
    for k, v in res["inference"].items():
        print(k, {m: round(v[m]["mean"], 5) for m in v if m.startswith("t_")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
