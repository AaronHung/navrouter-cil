#!/usr/bin/env python3
"""NC-11（PREREG-11）：3.1 逐折配對檢定、3.2 top-K ablation（每個 K 重新訓練 I6）、3.3 選取方式 ablation。

    NAVCIL_MACHINE=mac python scripts/nc11_ablation.py paired --cache-dir <NC-8 快取>
    NAVCIL_MACHINE=mac python scripts/nc11_ablation.py select --cache-dir <NC-8 快取> --i6-ref-dir <i6/r2>
    NAVCIL_MACHINE=mac python scripts/nc11_ablation.py topk   --cache-dir <NC-8 快取> --i6-ref-dir <i6/r2>
輸出：outputs/navcil/<machine>/nc11/{paired,select,topk}/（result.json、facts.json；topk 另有權重與 done 標記）
任何失敗（例外、非有限值、重現檢查不符）：結束碼 1，不重試。每 15 分鐘在 log 寫一行進度。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc1_pipeline as P                                                  # noqa: E402
import nc2_report as N2                                                   # noqa: E402
import nc5_report as N5                                                   # noqa: E402
import nc8_report as N8                                                   # noqa: E402
import nc10_dispatch_compare as N10                                       # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS, four_round, mean_norm, one_shot, train_selector  # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.i6_expert import I6Expert                                   # noqa: E402
from selector.incremental_ridge import stream                             # noqa: E402

FOLDS = N8.FOLDS
KS = (64, 16, 32, 128, 0)                  # 訓練順序：先 K = 64（重現檢查 a），0 = 全部
ARMS_TOPK = ("16", "32", "64", "128", "all", "all_softmax")
ARMS_SEL = ("four16", "oneshot64")
AR_GAMMA = 1e-3
log = P.log
fmt = N2.fmt


class Progress:
    """每 15 分鐘在 log 寫一行：子實驗、完成數／總數、失敗數。"""

    def __init__(self, sub: str, total: int, every: float = 900.0):
        self.sub, self.total, self.done, self.fail = sub, total, 0, 0
        self._stop = threading.Event()
        threading.Thread(target=self._loop, args=(every,), daemon=True).start()
        self.report("開始")

    def report(self, tag="進度"):
        log(f"{tag} {self.sub}：完成 {self.done}/{self.total}，失敗 {self.fail}")

    def _loop(self, every):
        while not self._stop.wait(every):
            self.report()

    def step(self, n=1):
        self.done += n

    def close(self, ok: bool):
        if not ok:
            self.fail += 1
        self._stop.set()
        self.report("結束")


class B11(N8.B8):
    """NC-8 評估容器：AR 由累加統計量建立；test 快取的 I6_cos8 可換成某個 arm 的證據。"""

    def __init__(self, cache_dir):
        super().__init__()
        self.cache = Path(cache_dir)
        self.inc, self.cos8, self.reads = {}, None, []
        for f in FOLDS:
            for o in ORDERS:
                pos = [self.tasks.index(x) for x in ORDERS[o]]

                def load(p, f=f, o=o):
                    self.reads.append((f, o, p))
                    return torch.load(self.cache / f"nc8_fold{f}_train_{self.tasks[p]}.pt", map_location="cpu")["mean_vec"]
                for t, m in enumerate(stream(pos, load, AR_GAMMA), 1):
                    self.inc[(f, o, t)] = (m.solve(), list(m.task_ids))

    def W(self, f, o, t, gamma, bal=False, lin8=False):
        assert gamma == AR_GAMMA and not bal and not lin8
        return self.inc[(f, o, t)]

    def c(self, f, split, t):
        d = super().c(f, split, t)
        if split == "test" and self.cos8 is not None:
            return {**d, "I6_cos8": self.cos8[(f, t)]}
        return d


def eval_arm(b: B11, cos8) -> dict:
    """task-known（oracle ＋ I6，第 8 列）與 task-inferred（AR ＋ I6，第 6 列），十折 × 兩序。"""
    b.cos8 = cos8
    out = {}
    for name, row in (("known", 8), ("inferred", 6)):
        out[name] = {}
        for o in ORDERS:
            recs = []
            for f in FOLDS:
                stage, _ = N8.stage_row(b, row, f, o)
                r = N5.cil_full(stage, o, b.tasks)
                recs.append({k: r[k] for k in ("acc", "masked", "forgetting", "bwt", "acc_t", "masked_t")})
            out[name][o] = recs
    b.cos8 = None
    return out


def repro(b: B11, res: dict) -> dict:
    """與 REPORT_stage10.md:16／:18／:29／:31 與 nc8/per_fold.json 第 6、8 列比對。"""
    old = json.loads((b.out / "nc8" / "per_fold.json").read_text())["rows"]
    r10 = (b.out / "REPORT_stage10.md").read_text().splitlines()
    lines = {("known", "reverse"): 18, ("known", "paper"): 31, ("inferred", "reverse"): 16, ("inferred", "paper"): 29}
    out = {}
    for (name, o), ln in lines.items():
        new = [x["acc"] for x in res[name][o]]
        prev = [x["acc"] for x in old[N8.ROWS[(8 if name == "known" else 6) - 1]][o]]
        cell = N10.cells(r10[ln - 1])[1]
        out[f"{name}|{o}"] = {"report10_line": ln, "report10": cell, "now": fmt(new),
                              "folds_r4_equal": sum(round(a, 4) == round(c, 4) for a, c in zip(new, prev)),
                              "max_fold_absdiff": max(abs(a - c) for a, c in zip(new, prev)),
                              "ok": cell == fmt(new) and all(round(a, 4) == round(c, 4) for a, c in zip(new, prev))}
    return out


def load_i6(path) -> I6Expert:
    m = I6Expert(2)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def jaccard(a, b) -> float:
    sa, sb = set(a.tolist()), set(b.tolist())
    return len(sa & sb) / len(sa | sb)


def write_facts(d: Path, facts: dict):
    (d / "facts.json").write_text(json.dumps(facts, indent=1, ensure_ascii=False))


def summarize(prefix: str, arm: str, res: dict, facts: dict):
    for name in ("known", "inferred"):
        for o in ORDERS:
            m, s = mean_sd([x["acc"] for x in res[name][o]])
            facts[f"{prefix}.{arm}.{name}.{o}.acc"] = m
            facts[f"{prefix}.{arm}.{name}.{o}.acc_sd"] = s


# ── 3.1 ─────────────────────────────────────────────────────────────────────
def cmd_paired(args) -> int:
    from scipy.stats import ttest_rel, wilcoxon
    prog = Progress("3.1 paired", 1)
    b = N10.B10(args.cache_dir)
    for f in FOLDS:
        for o in ORDERS:
            b.build_ar(f, o)
    pf = json.loads((b.out / "nc10" / "per_fold.json").read_text())["dispatch"]
    old = json.loads((b.out / "nc8" / "per_fold.json").read_text())["rows"][N8.ROWS[5]]
    ar_check = all(x["acc"] == y["acc"] for o in ORDERS for x, y in zip(pf["AR"][o], old[o]))
    tests, facts, worse = {}, {}, {}
    for comp in ("R3", "R0"):
        for o in ORDERS:
            for m in ("tp_micro", "tp_macro", "acc"):
                a = [x[m] for x in pf["AR"][o]]
                c = [x[m] for x in pf[comp][o]]
                d = [x - y for x, y in zip(a, c)]
                w = wilcoxon(a, c, alternative="two-sided")
                t = ttest_rel(a, c)
                key = f"{comp}|{o}|{m}"
                tests[key] = {"AR": a, comp: c, "diff": d, "mean_diff": mean_sd(d)[0], "sd_diff": mean_sd(d)[1],
                              "wins": sum(v > 0 for v in d), "ties": sum(v == 0 for v in d), "losses": sum(v < 0 for v in d),
                              "p_wilcoxon": float(w.pvalue), "p_ttest": float(t.pvalue), "t": float(t.statistic)}
                fid = f"nc11.paired.{comp.lower()}.{o}.{m}"
                for k in ("mean_diff", "wins", "ties", "p_wilcoxon", "p_ttest"):
                    facts[f"{fid}.{k}"] = tests[key][k]
                for k, v in enumerate(d, 1):
                    if v < 0:
                        worse.setdefault((comp, k), []).append((o, m))
    # 每折混淆（t = 4）與 CIL 差的拆解；十折合計須與 nc10/confusion.json 相同
    conf_all = json.loads((b.out / "nc10" / "confusion.json").read_text())
    per_fold_conf, decomp = {}, {}
    sums = {(n, o): torch.zeros(4, 4, dtype=torch.long) for n in ("AR", "R3", "R0") for o in ORDERS}
    for o in ORDERS:
        seen = [b.tasks.index(x) for x in ORDERS[o]]
        for f in FOLDS:
            g = b.gather(f, seen)
            head_ok = N2.hard(g["I6"], g["task"], g["task"])[0] == g["labels"]
            th = {n: N10.dispatch(b, n, f, o, g, seen) for n in ("AR", "R3", "R0")}
            for n in th:
                cm = torch.zeros(4, 4, dtype=torch.long)
                for a_, c_ in zip(g["task"].tolist(), th[n].tolist()):
                    cm[a_, c_] += 1
                sums[(n, o)] += cm
                per_fold_conf[f"{n}|{o}|{f}"] = cm.tolist()
            for comp in ("R3", "R0"):
                ok_a, ok_c = th["AR"] == g["task"], th[comp] == g["task"]
                rows = []
                for p in range(4):
                    mk = g["task"] == p
                    rows.append({"task": b.tasks[p], "n": int(mk.sum()),
                                 "ar_only": int((mk & ok_a & ~ok_c).sum()), "ar_only_head_ok": int((mk & ok_a & ~ok_c & head_ok).sum()),
                                 "comp_only": int((mk & ok_c & ~ok_a).sum()), "comp_only_head_ok": int((mk & ok_c & ~ok_a & head_ok).sum())})
                decomp[f"{comp}|{o}|{f}"] = rows
    conf_ok = all(sums[(n, o)].tolist() == conf_all[n][o] for n in ("AR", "R3", "R0") for o in ORDERS)
    res = {"ar_equals_nc8_row6": ar_check, "confusion_sum_equals_nc10": conf_ok, "n_tests": len(tests), "tests": tests,
           "worse": {f"{c}|{k}": v for (c, k), v in worse.items()}, "per_fold_confusion": per_fold_conf, "decomp": decomp}
    d = b.out / "nc11" / "paired"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    write_facts(d, facts)
    ok = ar_check and conf_ok
    prog.step()
    prog.close(ok)
    log(f"3.1：AR 與 nc8 第 6 列相同 {ar_check}；混淆十折合計與 nc10 相同 {conf_ok}；檢定 {len(tests)} 次")
    return 0 if ok else 1


# ── 3.3 ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def cmd_select(args) -> int:
    ctx = P.Ctx(torch.device("cpu"))
    d = ctx.out / "nc11" / "select"
    d.mkdir(parents=True, exist_ok=True)
    ctx.timing_path = d / "timing.json"
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    lam = ctx.lam()
    b = B11(args.cache_dir)
    prog = Progress("3.3 select", len(FOLDS))
    log(f"3.3 threads={torch.get_num_threads()} λ*={lam}")
    ok = False
    try:
        for f in FOLDS:
            part, done = d / f"fold{f}.pt", d / f"fold{f}.done"
            if not done.exists():
                ex = [load_i6(Path(args.i6_ref_dir) / f"fold{f}_{t}.pt") for t in ctx.tasks]
                t0 = time.perf_counter()
                out = {}
                for p, task in enumerate(ctx.tasks):
                    ds, shift = ctx.ds(f, task, "test")
                    ref = b.c(f, "test", task)
                    cos = {a: [] for a in ARMS_SEL}
                    jac, t_read, t_comp = [], [], []
                    for i in range(len(ds)):
                        ta = time.perf_counter()
                        rec = read_slide(ds, shift, i)
                        tb = time.perf_counter()
                        assert rec.sid == ref["sids"][i]
                        Z = rec.Z
                        c4, c1 = [], []
                        for q, m in enumerate(ex):
                            s = m(Z, ctx.f_task(q))
                            j4, j1 = four_round(Z, s, lam), one_shot(s)
                            c4.append(mean_norm(Z, j4) @ ctx.F.t())
                            c1.append(mean_norm(Z, j1) @ ctx.F.t())
                            if q == p:
                                jac.append(jaccard(j4, j1))
                        cos["four16"].append(torch.stack(c4)); cos["oneshot64"].append(torch.stack(c1))
                        t_comp.append(time.perf_counter() - tb); t_read.append(tb - ta)
                    out[task] = {**{a: torch.stack(v) for a, v in cos.items()}, "jaccard_own": torch.tensor(jac),
                                 "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp), "sids": ref["sids"]}
                torch.save(out, part)
                done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
                ctx.record_time(f"fold{f}", time.perf_counter() - t0)
            prog.step()
            log(f"3.3 fold {f} 完成")
        ev = {f: torch.load(d / f"fold{f}.pt", map_location="cpu") for f in FOLDS}
        diff4 = max((ev[f][t]["four16"] - b.c(f, "test", t)["I6_cos8"]).abs().max().item() for f in FOLDS for t in b.tasks)
        res, facts = {"arms": {}, "cos8_four16_vs_nc8_maxabs": diff4}, {}
        for a in ARMS_SEL:
            r = eval_arm(b, {(f, t): ev[f][t][a] for f in FOLDS for t in b.tasks})
            res["arms"][a] = r
            summarize("nc11.select", a, r, facts)
        res["repro"] = repro(b, res["arms"]["four16"])
        jac = torch.cat([ev[f][t]["jaccard_own"] for f in FOLDS for t in b.tasks])
        res["jaccard_own"] = {"mean": jac.mean().item(), "median": jac.median().item(), "min": jac.min().item(), "n": len(jac)}
        facts["nc11.select.jaccard_own.mean"] = res["jaccard_own"]["mean"]
        facts["nc11.select.cos8_four16_vs_nc8_maxabs"] = diff4
        res["t_read_s"] = sum(ev[f][t]["t_read_s"].sum().item() for f in FOLDS for t in b.tasks)
        res["t_compute_s"] = sum(ev[f][t]["t_compute_s"].sum().item() for f in FOLDS for t in b.tasks)
        (d / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        write_facts(d, facts)
        ok = all(v["ok"] for v in res["repro"].values())
        for k, v in res["repro"].items():
            log(f"3.3 重現 {k}: {v['report10']} vs {v['now']} → {'符合' if v['ok'] else '不符'}")
        log(f"3.3 four16 對 NC-8 I6_cos8 最大差 {diff4:.2e}")
    finally:
        prog.close(ok)
    return 0 if ok else 1


# ── 3.2 ─────────────────────────────────────────────────────────────────────
def train_k(ctx, root: Path, K: int, fold: int, task: str):
    """與 scripts/nc7_i6.py:42-65 train_one 相同，只把 budget 設為 K（0 = 全部）。"""
    part, done = root / f"K{K}" / f"fold{fold}_{task}.pt", root / f"K{K}" / f"fold{fold}_{task}.done"
    if done.exists():
        return
    part.parent.mkdir(parents=True, exist_ok=True)
    p = ctx.tasks.index(task)
    ds, shift = ctx.ds(fold, task, "train")
    torch.manual_seed(P.SEED)
    model = I6Expert(2)
    log(f"I6 K={K} fold {fold} {task}: train n={len(ds)} params={model.n_params()}")

    def slides(g):
        for i in torch.randperm(len(ds), generator=g).tolist():
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            yield time.perf_counter() - t0, rec.sid, rec.Z, rec.label - shift

    t0 = time.perf_counter()
    model, hist = train_selector(slides, ctx.f_task(p), ctx.ls, epochs=P.EPOCHS, lr=P.LR,
                                 weight_decay=P.WD, seed=P.SEED, log=log, model=model, budget=K)
    ctx.record_time(f"train/K{K}/fold{fold}/{task}", time.perf_counter() - t0)
    torch.save(model.state_dict(), part)
    (part.parent / f"fold{fold}_{task}_train.json").write_text(json.dumps(hist))
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")


@torch.no_grad()
def eval_topk_fold(ctx, root: Path, b: B11, lam: float, fold: int):
    part, done = root / f"eval_fold{fold}.pt", root / f"eval_fold{fold}.done"
    if done.exists():
        return
    ex = {K: [load_i6(root / f"K{K}" / f"fold{fold}_{t}.pt") for t in ctx.tasks] for K in KS}
    t0 = time.perf_counter()
    out = {}
    for p, task in enumerate(ctx.tasks):
        ds, shift = ctx.ds(fold, task, "test")
        ref = b.c(fold, "test", task)
        cos = {a: [] for a in ARMS_TOPK}
        t_read, t_comp = [], []
        for i in range(len(ds)):
            ta = time.perf_counter()
            rec = read_slide(ds, shift, i)
            tb = time.perf_counter()
            assert rec.sid == ref["sids"][i]
            Z = rec.Z
            full = mean_norm(Z) @ ctx.F.t()
            rows = {a: [] for a in ARMS_TOPK}
            for q in range(len(ctx.tasks)):
                for K in (16, 32, 64, 128):
                    s = ex[K][q](Z, ctx.f_task(q))
                    rows[str(K)].append(mean_norm(Z, four_round(Z, s, lam, budget=K)) @ ctx.F.t())
                s0 = ex[0][q](Z, ctx.f_task(q))
                rows["all"].append(full)
                rows["all_softmax"].append(mean_norm(Z, w=F.softmax(s0, dim=0)) @ ctx.F.t())
            for a in ARMS_TOPK:
                cos[a].append(torch.stack(rows[a]))
            t_comp.append(time.perf_counter() - tb); t_read.append(tb - ta)
        out[task] = {**{a: torch.stack(v) for a, v in cos.items()}, "t_read_s": torch.tensor(t_read),
                     "t_compute_s": torch.tensor(t_comp), "sids": ref["sids"]}
    torch.save(out, part)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    ctx.record_time(f"eval/fold{fold}", time.perf_counter() - t0)


def cmd_topk(args) -> int:
    ctx = P.Ctx(torch.device("cpu"))
    root = ctx.out / "nc11" / "topk"
    root.mkdir(parents=True, exist_ok=True)
    ctx.timing_path = root / "timing.json"
    ctx.timing = json.loads(ctx.timing_path.read_text()) if ctx.timing_path.exists() else {}
    lam = ctx.lam()
    b = B11(args.cache_dir)
    prog = Progress("3.2 topk", len(KS) * len(FOLDS) * len(ctx.tasks) + len(FOLDS))
    log(f"3.2 threads={torch.get_num_threads()} λ*={lam} K 順序={KS}")
    ok, wcheck = False, {}
    t_run = time.perf_counter()
    try:
        for K in KS:
            tk = time.perf_counter()
            for fold in FOLDS:
                for task in ctx.tasks:
                    train_k(ctx, root, K, fold, task)
                    prog.step()
            ctx.record_time(f"train_total/K{K}", time.perf_counter() - tk)
            log(f"3.2 K={K} 訓練完成（{time.perf_counter() - tk:.0f}s）")
            if K == 64:          # 重現檢查 a：與既有 i6/r2 權重逐位元比較（不符不停，看 b）
                for fold in FOLDS:
                    for task in ctx.tasks:
                        new = torch.load(root / "K64" / f"fold{fold}_{task}.pt", map_location="cpu")
                        ref = torch.load(Path(args.i6_ref_dir) / f"fold{fold}_{task}.pt", map_location="cpu")
                        wcheck[f"{fold}|{task}"] = {"equal": all(torch.equal(new[k], ref[k]) for k in ref),
                                                    "maxabs": max((new[k] - ref[k]).abs().max().item() for k in ref)}
                log(f"3.2 重現 a：K=64 權重逐位元相同 {sum(v['equal'] for v in wcheck.values())}/{len(wcheck)}")
        for fold in FOLDS:
            eval_topk_fold(ctx, root, b, lam, fold)
            prog.step()
            log(f"3.2 fold {fold} 評估完成")
        ev = {f: torch.load(root / f"eval_fold{f}.pt", map_location="cpu") for f in FOLDS}
        diff64 = max((ev[f][t]["64"] - b.c(f, "test", t)["I6_cos8"]).abs().max().item() for f in FOLDS for t in b.tasks)
        res, facts = {"arms": {}, "weights_K64_vs_ref": wcheck, "cos8_K64_vs_nc8_maxabs": diff64}, {}
        for a in ARMS_TOPK:
            r = eval_arm(b, {(f, t): ev[f][t][a] for f in FOLDS for t in b.tasks})
            res["arms"][a] = r
            summarize("nc11.topk", a, r, facts)
        res["repro"] = repro(b, res["arms"]["64"])
        facts["nc11.topk.cos8_K64_vs_nc8_maxabs"] = diff64
        facts["nc11.topk.weights_K64_equal"] = sum(v["equal"] for v in wcheck.values())
        res["timing"] = ctx.timing
        res["t_read_s_eval"] = sum(ev[f][t]["t_read_s"].sum().item() for f in FOLDS for t in b.tasks)
        res["t_compute_s_eval"] = sum(ev[f][t]["t_compute_s"].sum().item() for f in FOLDS for t in b.tasks)
        res["seconds_total"] = time.perf_counter() - t_run
        (root / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        write_facts(root, facts)
        ok = all(v["ok"] for v in res["repro"].values())
        for k, v in res["repro"].items():
            log(f"3.2 重現 b {k}: {v['report10']} vs {v['now']} → {'符合' if v['ok'] else '不符'}")
    finally:
        prog.close(ok)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("paired", "select", "topk"))
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--i6-ref-dir")
    args = ap.parse_args()
    args.cache_dir = os.path.expanduser(args.cache_dir)
    if args.i6_ref_dir:
        args.i6_ref_dir = os.path.expanduser(args.i6_ref_dir)
    elif args.cmd != "paired":
        ap.error("--i6-ref-dir 必填")
    return {"paired": cmd_paired, "select": cmd_select, "topk": cmd_topk}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
