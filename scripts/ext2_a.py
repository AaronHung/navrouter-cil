#!/usr/bin/env python3
"""EXT-2 A（PREREG-22 細則 5、8、9、11–15、17）：FINAL-B（不用 head：s = s0、四輪各 16、[v; 1] 累加式 ridge）的全套逐折值。
只讀既有快取，不訓練、不載入任何 head 權重來算 FINAL-B（對照列 MAIN、FINAL-A 沿用既有的 v(s) 快取）。

  γ 重選   moe3_common.val_objective／select（PREREG-19 細則 5 的逐階段 validation 目標；kind = g0）；K10 對 moe3/hp.json
  對照列   ext1_c.do_fold（ZS8、LIN8、MAIN、FINAL-A 的五個 seed、NOHEAD、K32…K256）重算到 ext2/a/；K9 對 ext1/c 的逐折值
  FINAL-B  v0（g = 0 四輪；moe2 vec 快取的 g0）／RDG(γ_B)；K 列為 u_K／RDG(γ_B)；另算 TP 正確率、逐任務 WP、逐類、混淆矩陣、γ 敏感度

    NAVCIL_MACHINE=mac python scripts/ext2_a.py [--folds 1-10]
輸出：outputs/navcil/<machine>/ext2/a/<row>/<order>_fold<f>.json（＋ .done）、ext2/a.json、FINALB_perfold.csv、FINALB_perfold_extra.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_c as CC                                                       # noqa: E402
from ext1_c import C, D64, M, P, T, log                                   # noqa: E402

ORDERS, SEEDS = CC.ORDERS, CC.SEEDS
KIND_B = "g0"                                   # v0：s = s0 的四輪向量（moe2_common.build_vec）
KS = (32, 64, 128, 256)
G_REF = 1e-3
REF_ROWS = (["ZS8", "LIN8", "MAIN", "FINAL", "NOHEAD"] + [f"K{k}" for k in KS]
            + [f"{n}_s{s}" for n in ("FINAL", "MAIN") for s in SEEDS[1:]])
K9_KEYS = ("acc_t", "wp_t", "mk1_t", "forgetting", "bwt")
FP32 = 4


class Ctx(CC.Ctx):
    """ext1_c.Ctx 的唯讀來源不變；輸出改寫到 ext2/a/（不寫入 ext1/）。"""

    def __init__(self):
        super().__init__()
        self.ext1_root = self.root
        self.root = self.base / "ext2" / "a"

    def done(self, row: str, o: str, f: int) -> bool:
        return self.path(row, o, f).with_suffix(".done").exists()


def gamma_tag(g: float) -> str:
    return f"FINALB_g{g:g}"


def reselect(cx: Ctx, folds: list[int]) -> dict:
    """細則 5：FINAL-B 的 γ 用逐階段 validation 目標在 RDG_G 重選；K10 對 moe3/hp.json 的 RDG:g0。"""
    per = {T.ckey(c): [] for c in T.cands("RDG")}
    for f in folds:
        for k, v in T.val_objective(cx.r3, f, KIND_B, "RDG").items():
            per[k].append(v)
        cx.r3.drop()
    obj = {k: C.mean(v) for k, v in per.items()}
    star = T.select("RDG", obj)
    ref = cx.hp["combos"][T.hp_key("RDG", KIND_B)]
    k10 = max(abs(obj[k] - ref["obj"][k]) for k in obj) if folds == C.FOLDS else None
    return {"grid": list(T.RDG_G), "obj": obj, "obj_per_fold": per, "star": star, "moe3_star": ref["star"],
            "K10": {"max_abs": k10, "tol": 1e-9, "pass": None if k10 is None else bool(k10 <= 1e-9), "moe3_obj": ref["obj"]}}


def flat(x):
    return [v for row in x for v in (flat(row) if isinstance(row, list) else [row])] if isinstance(x, list) else [x]


def k9(cx: Ctx, folds: list[int]) -> dict:
    """細則 8：本批重算的對照列對 ext1/c 的逐折值。"""
    per, bad = {}, []
    for row in REF_ROWS:
        m = 0.0
        for o in ORDERS:
            for f in folds:
                a = json.loads(cx.path(row, o, f).read_text())
                b = json.loads((cx.ext1_root / row / f"{o}_fold{f}.json").read_text())
                for k in K9_KEYS:
                    for x, y in zip(flat(a[k]), flat(b[k])):
                        d = abs(x - y)
                        m = max(m, d)
                        if d > 1e-9:
                            bad.append({"row": row, "order": o, "fold": f, "key": k, "new": x, "old": y})
        per[row] = m
    return {"per_row_max_abs": per, "max_abs": max(per.values()), "tol": 1e-9, "n_bad": len(bad), "bad": bad[:50],
            "cells": sum(len(K9_KEYS) for _ in REF_ROWS) * len(ORDERS) * len(folds), "pass": not bad}


def first_of(cx: Ctx, f: int, o: str, D: dict, kind: str, g: float):
    st = T.stats(cx.r3, kind)
    return CC.ridge_first(CC.ridge_scores(lambda t: st.W(f, o, t, ("RDG", g)), lambda pv: T.vec_of(D, kind, pv)))


def finalb_fold(cx: Ctx, f: int, o: str, gB: float, sens: dict, agg: dict) -> dict:
    """一折一序：FINAL-B 與 K 列的逐階段值；FINAL-B 另算 TP、逐任務、逐類、γ 敏感度。回傳 {row: 結果}。"""
    r3 = cx.r3
    D = T.data(r3, "test", f)
    task, label = D["task"], D["label"]
    pos = r3.pos(o)
    ar = T.ar_stages(r3, f, o, D)
    out = {}
    first = first_of(cx, f, o, D, KIND_B, gB)
    res = CC.cl_eval(first, ar, task, label, o, cx.tasks)

    tp_t, tp_task, wp_task, cil_task = [], [], [], []
    for t in range(1, 5):
        th = ar[t - 1].argmax(-1)
        tt = [None] * 4
        for q in pos[:t]:
            tt[q] = (th[task == q] == q).float().mean().item()
        tp_task.append(tt)
        tp_t.append(C.mean(tt[q] for q in pos[:t]))
        wt, ct = [None] * 4, [None] * 4
        for j in range(t):
            wt[pos[j]], ct[pos[j]] = res["Rm_orc"][t - 1][j], res["R"][t - 1][j]
        wp_task.append(wt); cil_task.append(ct)
    th4 = ar[3].argmax(-1)
    tell = 2 * task + (~first(4, task, task)).long()
    cil = 2 * th4 + (~first(4, th4, th4)).long()
    res.update({"tp_t": tp_t, "tp_task_t": tp_task, "wp_task_t": wp_task, "cil_task_t": cil_task, "gamma": gB,
                "th4": th4.tolist(), "tell4": tell.tolist(), "cil4": cil.tolist(), "label": label.tolist(), "task": task.tolist()})
    out["FINALB"] = res

    for a in range(4):
        for b in range(4):
            agg["conf"][o][a, b] += int(((task == a) & (th4 == b)).sum())
    if o == ORDERS[0]:
        for c in range(8):
            m = label == c
            agg["cls_tell"][c] += torch.tensor([int((tell[m] == c).sum()), int(m.sum())])
            agg["cls_cil"][c] += torch.tensor([int((cil[m] == c).sum()), int(m.sum())])
    agg["tell"][o].append(tell); agg["cil"][o].append(cil)

    if gB != G_REF:                                        # 細則 5：γ_B ≠ 1e-3 時另列 1e-3 的同一套
        out[gamma_tag(G_REF)] = CC.cl_eval(first_of(cx, f, o, D, KIND_B, G_REF), ar, task, label, o, cx.tasks)
    for K in KS:
        out[f"FINALB_K{K}"] = CC.cl_eval(first_of(cx, f, o, D, T.ukind(K), gB), ar, task, label, o, cx.tasks)
    for g in T.RDG_G:                                      # 細則 17：事後描述
        r = res if g == gB else CC.cl_eval(first_of(cx, f, o, D, KIND_B, g), ar, task, label, o, cx.tasks)
        sens[o][T.ckey(("RDG", g))]["cil4"].append(r["acc_t"][3])
        sens[o][T.ckey(("RDG", g))]["abar"].append(C.mean(r["acc_t"]))
        sens[o][T.ckey(("RDG", g))]["wp4"].append(r["wp_t"][3])
    return out


def storage(cx: Ctx) -> dict:
    """細則 15：由實際張量形狀計算（fp32）。"""
    W, cls = T.stats(cx.r3, KIND_B).W(1, ORDERS[0], 4, ("RDG", G_REF))
    A, B, _ = T.stats(cx.r3, KIND_B).AB(1, ORDERS[0], 4)
    W_ar, pos = cx.st.b.W(1, ORDERS[0], 4, M.G_AR)
    Ft = cx.r3.ctx.F
    x = int(A.shape[0])
    n_head = M.load_head(CC.X.head_path(cx.r4, 42, 1, cx.tasks[0])).n_params()
    item = {"A": ("FINAL-B／FINAL-A 的 A（[v; 1] 的 Σ x xᵀ）", int(A.numel())), "A_mv": ("AR 的 A（[mean_vec; 1] 的 Σ x xᵀ）", x * x),
            "text": ("兩類文字特徵（2 × 512）", int(Ft[0:2].numel())), "B": ("判讀器的 B：該任務兩欄", int(B.shape[0]) * 2),
            "B_ar": ("AR 的 B：該任務一欄", int(W_ar.shape[0])), "head": ("I6(r = 2) head 參數", n_head),
            "B_lin8": ("LIN8 的 B：該任務兩欄（[mean_vec; 1]）", 2 * x)}
    systems = {"ZS8": {"shared": [], "per_task": [], "params": []},
               "LIN8": {"shared": ["A_mv"], "per_task": ["B_lin8"], "params": ["B_lin8"]},
               "MAIN": {"shared": ["A_mv"], "per_task": ["head", "text", "B_ar"], "params": ["head", "B_ar"]},
               "FINALB": {"shared": ["A", "A_mv"], "per_task": ["text", "B", "B_ar"], "params": ["B", "B_ar"]},
               "FINALA": {"shared": ["A", "A_mv"], "per_task": ["head", "text", "B", "B_ar"], "params": ["head", "B", "B_ar"]}}
    out = {}
    for name, s in systems.items():
        S, p = sum(item[k][1] for k in s["shared"]) * FP32, sum(item[k][1] for k in s["per_task"]) * FP32
        out[name] = {**s, "shared_bytes": S, "per_task_bytes": p, "T4_bytes": S + 4 * p,
                     "per_task_params": sum(item[k][1] for k in s["params"])}
    return {"shapes": {"A": list(A.shape), "B_t4": list(B.shape), "W_t4": list(W.shape), "W_ar_t4": list(W_ar.shape), "F": list(Ft.shape),
                       "classes_t4": cls},
            "items": {k: {"what": v[0], "n_float": v[1], "fp32_bytes": v[1] * FP32} for k, v in item.items()}, "systems": out,
            "not_counted": "CONCH 骨幹（影像與文字編碼器）、patch 特徵、超參數純量（γ、λ、K）：不計入"}


def assemble(cx: Ctx, folds: list[int], gB: float, st: dict) -> None:
    cols = ["row", "order", "fold", "t", "CIL_ACC", "WP", "MaskedACC_table1", "MaskedACC_oracle", "Forgetting", "BWT", "storage_bytes"]
    sb = {k: (v["shared_bytes"], v["per_task_bytes"]) for k, v in st["systems"].items()}
    rows = [("FINALB", "FINALB", sb["FINALB"])]
    if gB != G_REF:
        rows.append((gamma_tag(G_REF), gamma_tag(G_REF), sb["FINALB"]))
    rows += [(f"FINALB_K{K}", f"FINALB_K{K}", sb["FINALB"]) for K in KS]
    rows += [("ZS8", "ZS8", sb["ZS8"]), ("LIN8", "LIN8", sb["LIN8"]), ("MAIN", "MAIN", sb["MAIN"]), ("FINALA", "FINAL", sb["FINALA"])]
    rows += [(f"MAIN_s{s}", f"MAIN_s{s}", sb["MAIN"]) for s in SEEDS[1:]] + [(f"FINALA_s{s}", f"FINAL_s{s}", sb["FINALA"]) for s in SEEDS[1:]]

    def fm(x):
        return "" if x is None else f"{x:.10f}"

    def emit(w, name, o, f, r, sp):
        for t in range(4):
            w.writerow([name, o, f, t + 1, fm(r["acc_t"][t]), fm(r["wp_t"][t]), fm(r["mk1_t"][t]), fm(r["wp_t"][t]),
                        fm(r["forgetting"]) if t == 3 else "", fm(r["bwt"]) if t == 3 else "", sp[0] + (t + 1) * sp[1]])

    def avg(rs, k):
        v = [r[k] for r in rs]
        return [sum(x[i] for x in v) / len(v) for i in range(4)] if isinstance(v[0], list) else sum(v) / len(v)

    with (cx.base / "FINALB_perfold.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for name, src, sp in rows:
            for o in ORDERS:
                for f in folds:
                    emit(w, name, o, f, json.loads(cx.path(src, o, f).read_text()), sp)
        for name, src in (("MAIN_5seed", "MAIN"), ("FINALA_5seed", "FINAL")):
            for o in ORDERS:
                for f in folds:
                    rs = [json.loads(cx.path(src if s == 42 else f"{src}_s{s}", o, f).read_text()) for s in SEEDS]
                    emit(w, name, o, f, {k: avg(rs, k) for k in K9_KEYS}, sb["MAIN" if src == "MAIN" else "FINALA"])
    with (cx.base / "FINALB_perfold_extra.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        short = CC.TASK_SHORT
        w.writerow(["row", "order", "fold", "t", "TP_ACC"] + [f"TP_{s}" for s in short] + [f"WP_{s}" for s in short] + [f"CIL_{s}" for s in short])
        for o in ORDERS:
            for f in folds:
                r = json.loads(cx.path("FINALB", o, f).read_text())
                for t in range(4):
                    w.writerow(["FINALB", o, f, t + 1, fm(r["tp_t"][t])] + [fm(x) for x in r["tp_task_t"][t]]
                               + [fm(x) for x in r["wp_task_t"][t]] + [fm(x) for x in r["cil_task_t"][t]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1-10")
    a, _ = ap.parse_known_args()
    folds = P.parse_folds(a.folds)
    sys.argv = sys.argv[:1]
    cx = Ctx()
    root = cx.base / "ext2"
    cx.root.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    try:
        sel = reselect(cx, folds)
        gB = sel["star"]["gamma"]
        log(f"A：γ_B = {gB:g}（validation 目標 {sel['obj']}）；K10 max_abs = {sel['K10']['max_abs']}")

        b_rows = ["FINALB"] + ([gamma_tag(G_REF)] if gB != G_REF else []) + [f"FINALB_K{K}" for K in KS]
        sens = {o: {T.ckey(("RDG", g)): {"cil4": [], "abar": [], "wp4": []} for g in T.RDG_G} for o in ORDERS}
        agg = {"conf": {o: torch.zeros(4, 4, dtype=torch.long) for o in ORDERS}, "cls_tell": torch.zeros(8, 2, dtype=torch.long),
               "cls_cil": torch.zeros(8, 2, dtype=torch.long), "tell": {o: [] for o in ORDERS}, "cil": {o: [] for o in ORDERS}}
        for f in folds:
            for o in ORDERS:
                t0 = time.perf_counter()
                todo = [r for r in REF_ROWS if not cx.done(r, o, f)]
                if todo:
                    CC.do_fold(cx, f, o, todo)
                for row, res in finalb_fold(cx, f, o, gB, sens, agg).items():   # FINAL-B 每次都重算（agg、sens 需要全部折）
                    cx.save(row, o, f, res)
                log(f"A fold {f} {o}: 對照 {len(todo)} 列 ＋ FINAL-B {len(b_rows)} 列 {time.perf_counter() - t0:.1f}s")
            cx.r3.drop(); cx.r4.drop(); cx.r3.v2._vec.clear(); cx.lrg._W.clear()

        st = storage(cx)
        cx.r3.drop()
        same = {"tell_diff": sum(int((x != y).sum()) for x, y in zip(agg["tell"]["reverse"], agg["tell"]["paper"])),
                "cil_diff": sum(int((x != y).sum()) for x, y in zip(agg["cil"]["reverse"], agg["cil"]["paper"])),
                "n": sum(len(x) for x in agg["tell"]["reverse"])}
        res = {"stage": "a", "folds": folds, "wall_s": round(time.perf_counter() - t_all, 1), "finished": M.now(),
               "gamma": sel, "gamma_B": gB, "K9": k9(cx, folds), "rows_B": b_rows, "rows_ref": REF_ROWS,
               "sens": sens, "tp_confusion": {o: agg["conf"][o].tolist() for o in ORDERS},
               "per_class_tell": {C.LABEL[c]: agg["cls_tell"][c].tolist() for c in range(8)},
               "per_class_cil": {C.LABEL[c]: agg["cls_cil"][c].tolist() for c in range(8)},
               "order_same_t4": same, "storage": st}
        assemble(cx, folds, gB, st)
        (root / "a.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._default))
        log(f"A：完成（{res['wall_s']:.0f}s）；K9 max_abs = {res['K9']['max_abs']:.2e}（{'通過' if res['K9']['pass'] else '不過'}）")
        return 0
    except Exception:
        (root / "FAILED_a.txt").write_text(f"[{M.now()}] 階段 a 失敗\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
