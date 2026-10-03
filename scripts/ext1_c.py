#!/usr/bin/env python3
"""EXT-1 C1（PREREG-21 細則 17–20、22）：定稿表格各列的全量重算（十折、兩序、t = 1…4；只讀既有快取與權重，不訓練）。

每列每 (order, fold) 寫 outputs/navcil/<machine>/ext1/c/<row>/<order>_fold<f>.json 與 .done，重跑自動跳過；
全部完成後組出 outputs/navcil/<machine>/ABLATION_full.csv。

判讀器、累加統計量、AR、融合、指標一律呼叫既有函式（moe0…moe4_common、nc5_report.cil_full、nc8_report.stage_row、
moe1_s7.RF），超參數一律讀既有選定值（moe3/hp.json、moe0/selection.json、moe2/e4.json、moe1/s6.json、moe1/s7_rf.json）。

每個系統給出 first(t, pv, pc) → [N] bool：證據／向量取自任務 pv 的版本，在任務 pc 的兩類內是否判第一類。
  CIL            pv = pc = τ̂（階段 t 的 AR 在已學任務中 argmax）
  Masked（Table 1）pv = τ̂、pc = 真實任務
  Masked（oracle）pv = pc = 真實任務（= 告訴任務的 WP）

    NAVCIL_MACHINE=mac python scripts/ext1_c.py [--folds 1-10]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import moe1_s7 as S7                                                      # noqa: E402
import moe3_common as T                                                   # noqa: E402
import moe4_common as X                                                   # noqa: E402
import nc8_report as N8                                                   # noqa: E402
from moe3_common import C, D64, E, M, N5, P, log                          # noqa: E402

ORDERS = M.ORDER_NAMES
G = 1e-3
SEEDS = (42, 43, 44, 45, 46)
TASK_SHORT = ["esca", "rcc", "brca", "lung"]
# fp32 儲存（共用 S、每任務 p）：moe2 E8、moe3 F7、moe4 H5、S7／S10 的既有公式；沒有既有公式的列為 None
STORAGE = {"ZS8": (0, 0), "LIN8": (1_052_676, 4_104), "MAIN": (1_052_676, 10_280), "FINAL": (2_105_352, 14_384),
           "ONE64": (2_105_352, 14_384), "NOHEAD": (2_105_352, 10_252), "K32": (2_105_352, 10_252),
           "K64": (2_105_352, 10_252), "K128": (2_105_352, 10_252), "K256": (2_105_352, 10_252),
           "ANC": (2_105_352, 10_252), "ANC_v0": (2_105_352, 10_252), "ANC_v42": (2_105_352, 14_384),
           "M3": (1_052_676, 12_340), "CONCAT": (5_255_176, 18_480)}
ROW_ORDER = (["ZS8", "LIN8", "MAIN", "FINAL", "NOHEAD", "ONE64", "K32", "K64", "K128", "K256", "M1"]
             + [f"M2_head_{t}" for t in TASK_SHORT] + ["M2_g0", "M3", "G1", "G2", "ANC", "ANC_v0", "ANC_v42", "CONCAT", "RF"]
             + [f"{n}_s{s}" for n in ("FINAL", "MAIN", "ONE64") for s in SEEDS[1:]])


def mkrun(out: str) -> T.Run:
    """既有批次的 Run（唯讀使用其快取目錄；不呼叫 total／tick，不寫入該目錄）。"""
    old = sys.argv
    sys.argv = [old[0], "--out", out]
    try:
        return T.Run("ext1c")
    finally:
        sys.argv = old


class Ctx:
    def __init__(self):
        self.r3, self.r4 = mkrun("moe3"), mkrun("moe4")
        self.r3.v2.pos = self.r3.pos                       # moe2_common.Reader 需要 pos
        self.st, self.tasks, self.base = self.r3.st, self.r3.tasks, self.r3.base
        self.root = self.base / "ext1" / "c"
        self.hp = json.loads((self.base / "moe3" / "hp.json").read_text())
        sel = json.loads((self.base / "moe0" / "selection.json").read_text())
        self.ls, self.T_star = sel["logit_scale"], {o: sel["B5"][o]["T_star"] for o in ORDERS}
        self.g_lrg = json.loads((self.base / "moe2" / "e4.json").read_text())["gamma_LRG"]["star"]
        self.s6 = json.loads((self.base / "moe1" / "s6.json").read_text())
        self.g_rf = json.loads((self.base / "moe1" / "s7_rf.json").read_text())["gamma"]["star"]
        self.R = torch.randn(512, S7.DIM, generator=torch.Generator().manual_seed(42), dtype=D64)
        self.lrg = E.Reader(self.r3.v2, "LRG(42)", "s42", use_mv=True)

    def path(self, row: str, o: str, f: int) -> Path:
        return self.root / row / f"{o}_fold{f}.json"

    def missing(self, o: str, f: int) -> list[str]:
        return [r for r in ROW_ORDER if not self.path(r, o, f).with_suffix(".done").exists()]

    def save(self, row: str, o: str, f: int, res: dict) -> None:
        p = self.path(row, o, f)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, default=M._default))
        p.with_suffix(".done").write_text(time.strftime("%F %T") + "\n")


def cl_eval(first, ar, task, label, o, tasks) -> dict:
    """一個系統、一折、一序：四個階段的 CIL、兩種 Masked；Forgetting／BWT 用 nc5_report.cil_full（Table 1 同一個函式）。"""
    last = {}

    def stage(which):
        def fn(seen):
            t = len(seen)
            m = torch.isin(task, torch.tensor(seen))
            th = ar[t - 1].argmax(-1)
            cil = 2 * th + (~first(t, th, th)).long()
            pv = th if which == "t1" else task
            alt = 2 * task + (~first(t, pv, task)).long()
            if t == 4:
                last["ok4"] = (cil == label).long().tolist()
            return cil[m], alt[m], {"labels": label[m], "task": task[m]}
        return fn

    r1, r2 = N5.cil_full(stage("t1"), o, tasks), N5.cil_full(stage("orc"), o, tasks)
    return {"acc_t": r1["acc_t"], "mk1_t": r1["masked_t"], "wp_t": r2["masked_t"], "forgetting": r1["forgetting"],
            "bwt": r1["bwt"], "R": r1["R"], "Rm_t1": r1["Rm"], "Rm_orc": r2["Rm"], "ok4": last["ok4"]}


def ridge_first(scores8):
    """scores8(t, pv) → [N, 8]（未學類別 NaN）；d ≥ 0 判第一類（moe2_common.pred_of 的規則）。"""
    def first(t, pv, pc):
        return M.pick(M.d_cols(scores8(t, pv)), pc) >= 0
    return first


def ridge_scores(Wfn, vec):
    cache = {}

    def scores(t, pv):
        if t not in cache:
            cache[t] = Wfn(t)
        W, cls = cache[t]
        out = torch.full((len(pv), 8), float("nan"), dtype=D64)
        out[:, cls] = N5.aug(vec(pv)) @ W
        return out
    return scores


def do_fold(cx: Ctx, f: int, o: str, todo: list[str]) -> None:
    r3, r4, st, tasks = cx.r3, cx.r4, cx.st, cx.tasks
    D3, D4 = T.data(r3, "test", f), X.data4(r4, "test", f)
    S, V = st.split("test", f), st.split("val", f)
    task, label, mv, I6 = D4["task"], D4["label"], D4["mv"], S["I6_cos8"]
    if not (torch.equal(task, S["task"]) and torch.equal(label, S["labels"]) and torch.equal(task, D3["task"])):
        raise M.CheckFailed(f"fold {f}: 各快取的 test slide 順序不一致")
    n = torch.arange(len(task))
    pos = r3.pos(o)
    ar = T.ar_stages(r3, f, o, D4)
    out = {}

    def run_first(row, first):
        if row in todo:
            out[row] = cl_eval(first, ar, task, label, o, tasks)

    # ── 無 expert 的兩列：nc8_report.stage_row（Table 1 同一份程式）；兩種 Masked 相同 ──
    for row, k in (("ZS8", 1), ("LIN8", 2)):
        if row in todo:
            stage, info = N8.stage_row(st.b, k, f, o)
            r = N5.cil_full(stage, o, tasks)
            g = info["g"]
            pred4 = stage(pos)[0]
            order4 = torch.argsort(g["task"], stable=True)             # 轉回 config 任務序（與其他列的 ok4 對齊）
            if not torch.equal(g["labels"][order4], label):
                raise M.CheckFailed(f"fold {f} {row}: NC-8 快取與 moe0 快取的 test 順序不一致")
            out[row] = {"acc_t": r["acc_t"], "mk1_t": r["masked_t"], "wp_t": r["masked_t"], "forgetting": r["forgetting"],
                        "bwt": r["bwt"], "R": r["R"], "Rm_t1": r["Rm"], "Rm_orc": r["Rm"],
                        "ok4": (pred4 == g["labels"])[order4].long().tolist()}

    # ── TXT 與 ridge 判讀器 ──
    def txt_first(run, D, kind):
        def first(t, pv, pc):
            return C.pair_diff(T.cos8(run, D, kind)[n, pv], pc) >= 0
        return first

    def rdg4(kind):                                                    # moe4 的向量（s42…46、w42…46）
        return ridge_first(ridge_scores(lambda t: X.acc_of(r4, kind).W(f, o, t, G), lambda pv: T.vec_of(D4, kind, pv)))

    def rd3(kind, rd):                                                 # moe3 的向量（g0、u_K、s42）
        return ridge_first(ridge_scores(lambda t: T.stats(r3, kind).W(f, o, t, rd), lambda pv: T.vec_of(D3, kind, pv)))

    run_first("MAIN", txt_first(r4, D4, "s42"))
    run_first("FINAL", rdg4("s42"))
    run_first("ONE64", rdg4("w42"))
    for s in SEEDS[1:]:
        run_first(f"MAIN_s{s}", txt_first(r4, D4, f"s{s}"))
        run_first(f"FINAL_s{s}", rdg4(f"s{s}"))
        run_first(f"ONE64_s{s}", rdg4(f"w{s}"))
    run_first("NOHEAD", rd3("g0", ("RDG", G)))
    for K in (32, 64, 128, 256):
        run_first(f"K{K}", rd3(T.ukind(K), ("RDG", G)))
    run_first("ANC", rd3(T.U_MAIN, T.star_rd(cx.hp, "ANC", T.U_MAIN)))
    run_first("ANC_v0", rd3("g0", T.star_rd(cx.hp, "ANC", "g0")))
    run_first("ANC_v42", rd3("s42", T.star_rd(cx.hp, "ANC", "s42")))
    if "CONCAT" in todo:
        Dv = E.vecs(r3.v2, "test", f)
        cache = {}

        def lrg_first(t, pv, pc):
            key = (t, id(pv))
            if key not in cache:
                cache[key] = cx.lrg.scores(f, o, t, cx.g_lrg, Dv, pv)
            return M.pick(M.d_cols(cache[key]), pc) >= 0
        run_first("CONCAT", lrg_first)

    # ── 融合系統（M3、G1、G2、RF）：z_a = d_a/σ_a（head 與 σ_a 取 pv）、z_b = d_b/σ_b（欄位、σ_b、gate 參數取 pc）──
    sa = M.sig(M.d_heads(V["I6_cos8"]), V["task"])
    fused_rows = [r for r in ("M3", "G1", "G2", "RF") if r in todo]
    if fused_rows:
        db = [M.d_cols(M.lin8_stage(st, f, o, t, mv, M.G_LIN8)) for t in range(1, 5)]
        sb = M.sigma_b_table(r3, f, o, V, M.G_LIN8)[1]
        fi = cx.s6["folds"].index(f)
        bt = torch.tensor(cx.s6["orders"][o]["beta"][fi], dtype=D64)
        ct = torch.tensor(cx.s6["orders"][o]["c"][fi], dtype=D64)

        def fused(kind, db_t, sb_q):
            def first(t, pv, pc):
                da = I6[n, pv, 2 * pc] - I6[n, pv, 2 * pc + 1]
                za, zb = M.zs(da, pv, sa), M.zs(M.pick(db_t[t - 1], pc), pc, sb_q)
                if kind == "sum":                              # M3、RF：同 moe1_common.fused（β = 1）
                    fz = za + zb
                elif kind == "G1":
                    fz = za.to(D64) + bt[pc] * zb.to(D64)
                else:                                          # G2
                    za, zb = za.to(D64), zb.to(D64)
                    rho = torch.sigmoid(ct[pc, 0] + ct[pc, 1] * za.abs() + ct[pc, 2] * zb.abs())
                    fz = (1 - rho) * za + rho * zb
                return (fz > 0) | ((fz == 0) & (da >= 0))
            return first

        run_first("M3", fused("sum", db, sb))
        run_first("G1", fused("G1", db, sb))
        run_first("G2", fused("G2", db, sb))
        if "RF" in todo:
            rf = S7.RF(r3, f, cx.R)
            dbr = [M.d_cols(rf.scores(o, t, cx.g_rf, mv)) for t in range(1, 5)]
            re = [M.sig(M.d_cols(rf.scores(o, t, cx.g_rf, V["mean_vec"])), V["task"]) for t in range(1, 5)]
            run_first("RF", fused("sum", dbr, [re[pos.index(q)][q] for q in range(4)]))

    # ── M1：任務間 soft gating（MOE-0 B5；T* 讀 selection.json）。在真實任務兩類內 π 相消，兩種 Masked = 主系統的告訴任務判定 ──
    if "M1" in todo:
        main_first = txt_first(r4, D4, "s42")
        last = {}

        def stage(seen):
            t = len(seen)
            m = torch.isin(task, torch.tensor(seen))
            cil = C.soft_gate_pred(I6, ar[t - 1], cx.T_star[o], cx.ls)
            tell = 2 * task + (~main_first(t, task, task)).long()
            if t == 4:
                last["ok4"] = (cil == label).long().tolist()
            return cil[m], tell[m], {"labels": label[m], "task": task[m]}
        r = N5.cil_full(stage, o, tasks)
        out["M1"] = {"acc_t": r["acc_t"], "mk1_t": r["masked_t"], "wp_t": r["masked_t"], "forgetting": r["forgetting"],
                     "bwt": r["bwt"], "R": r["R"], "Rm_t1": r["Rm"], "Rm_orc": r["Rm"], "ok4": last["ok4"]}

    # ── M2：跨器官共用 head（MOE-0 B2；告訴真實任務、t = 4）。快取只有真實任務文字的版本，沒有 CIL 與 t < 4 ──
    for k, name in list(enumerate(TASK_SHORT)) + [(4, "g0")]:
        row = "M2_g0" if name == "g0" else f"M2_head_{name}"
        if row in todo:
            c8 = S["g0_cos8"] if name == "g0" else S["cross_cos8"][:, k]
            pred = 2 * task + (C.pair_diff(c8, task) < 0).long()
            wt = C.task_mean(pred == label, task)
            out[row] = {"acc_t": [None] * 4, "mk1_t": [None] * 4, "wp_t": [None, None, None, C.eq4(wt)], "forgetting": None,
                        "bwt": None, "wp_task4": wt, "ok4": None}

    for row, res in out.items():
        cx.save(row, o, f, res)


def assemble(cx: Ctx, folds: list[int]) -> Path:
    cols = ["row", "order", "fold", "t", "CIL_ACC", "WP", "MaskedACC_table1", "MaskedACC_oracle", "Forgetting", "BWT", "storage_bytes"]
    path = cx.base / "ABLATION_full.csv"

    def fm(x):
        return "" if x is None else f"{x:.10f}"

    def emit(w, row, o, f, r, sp):
        for t in range(4):
            w.writerow([row, o, f, t + 1, fm(r["acc_t"][t]), fm(r["wp_t"][t]), fm(r["mk1_t"][t]), fm(r["wp_t"][t]),
                        fm(r["forgetting"]) if t == 3 else "", fm(r["bwt"]) if t == 3 else "",
                        "" if sp is None else sp[0] + (t + 1) * sp[1]])

    def avg(rs, k):
        v = [r[k] for r in rs]
        return [sum(x[i] for x in v) / len(v) for i in range(4)] if isinstance(v[0], list) else sum(v) / len(v)

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row in ROW_ORDER:
            for o in ORDERS:
                for f in folds:
                    emit(w, row, o, f, json.loads(cx.path(row, o, f).read_text()), STORAGE.get(row.split("_s4")[0]))
        for name in ("FINAL", "MAIN", "ONE64"):                          # 五 seed 平均（每折先對 seed 平均；細則 22）
            for o in ORDERS:
                for f in folds:
                    rs = [json.loads(cx.path(name if s == 42 else f"{name}_s{s}", o, f).read_text()) for s in SEEDS]
                    emit(w, f"{name}_5seed", o, f, {k: avg(rs, k) for k in ("acc_t", "wp_t", "mk1_t", "forgetting", "bwt")},
                         STORAGE.get(name))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1-10")
    a, _ = ap.parse_known_args()
    folds = P.parse_folds(a.folds)
    sys.argv = sys.argv[:1]
    cx = Ctx()
    prog = cx.base / "ext1" / "progress_c.txt"
    total, done, failed = len(folds) * len(ORDERS), 0, 0
    for f in folds:
        for o in ORDERS:
            todo = cx.missing(o, f)
            if todo:
                t0 = time.perf_counter()
                try:
                    do_fold(cx, f, o, todo)
                    log(f"C fold {f} {o}: {len(todo)} 列 {time.perf_counter() - t0:.1f}s")
                except Exception:                                      # 記錄後繼續其他 (order, fold)；不重試
                    import traceback
                    failed += 1
                    (cx.base / "ext1" / f"FAILED_c_{o}_fold{f}.txt").write_text(traceback.format_exc())
                    traceback.print_exc()
            done += 1
            prog.write_text(f"{done} {total} failed={failed}\n")
        cx.r3.drop(); cx.r4.drop(); cx.r3.v2._vec.clear(); cx.lrg._W.clear()
    if failed:
        log(f"C：{failed} 個 (order, fold) 失敗，不組 CSV")
        return 3
    log(f"C：完成，寫出 {assemble(cx, folds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
