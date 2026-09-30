#!/usr/bin/env python3
"""NC-15 步驟 4：r* 選定後才讀 test（PREREG-15 操作定義 5、7–10）。

啟動時檢查 nc15/selection.json 已 commit 且未修改，否則停止。之後以同一程式重算：
  LoRA 對照版 L1(r)，r ∈ {2, r*}：task-known test WP（oracle 分派）、task-inferred（AR γ = 1e-3）t = 4 ACC
  修正頭 I6(r = 2)：task-known、task-inferred（對照值，同一次執行重算）
一致性檢查（操作定義 9）不符即停。不訓練。

輸入（只讀）：--src-out 下的 cache/、lora_v2/r{1,2}/、i6/eval_fold*.pt、nc8/per_fold.json；
            r = 3：本 worktree 的 nc15/lora_r3/。
輸出：outputs/navcil/<machine>/nc15/{result.json, per_fold.json, facts.json}

    NAVCIL_MACHINE=mac python scripts/nc15_test.py --src-out <main>/outputs/navcil/mac
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
import nc5_report as N5                                                   # noqa: E402
import nc6_report as N6                                                   # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402

FOLDS = N2.FOLDS
GAMMA = 1e-3
TOL = 0.005
# 操作定義 9 的基準（四捨五入到小數第 4 位比對）
EXPECT = {
    "lora.r2.inferred.reverse": (0.9176, "REPORT_stage10.md:15"),
    "lora.r2.inferred.paper": (0.9132, "REPORT_stage10.md:28"),
    "lora.r2.known.reverse": (0.9401, "REPORT_stage5.md:43"),
    "lora.r2.known.paper": (0.9339, "REPORT_stage5.md:44"),
    "head.known.reverse": (0.9340, "REPORT_stage10.md:18"),
    "head.known.paper": (0.9340, "REPORT_stage10.md:31"),
    "head.inferred.reverse": (0.9128, "REPORT_stage10.md:16"),
    "head.inferred.paper": (0.9128, "REPORT_stage10.md:29"),
}
NC8_ROW = {"lora.r2.inferred": "5 D1：AR＋L1(r=2) v2", "head.inferred": "6 D3：AR＋I6(r=2)（主系統）",
           "head.known": "8 oracle＋I6(r=2)（上限）"}


def masked(c8, labels, p) -> float:
    rr = torch.tensor(task_rows(p))
    return (rr[c8[:, rr].argmax(-1)] == labels).float().mean().item()


def git(*a) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True)


def require_committed(path: Path) -> str:
    rel = str(path.relative_to(REPO_ROOT))
    if git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise SystemExit(f"{rel} 尚未 commit（PREREG-15 操作定義 5），停止")
    if git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} 與 HEAD 不同（PREREG-15 操作定義 5），停止")
    return git("log", "-1", "--format=%h", "--", rel).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-out", required=True, help="既有產物目錄，只讀")
    args = ap.parse_args()
    t0 = time.perf_counter()
    src = Path(args.src_out).resolve()
    d = N2.D2()
    own = d.out / "nc15"
    sel_commit = require_committed(own / "selection.json")
    sel = json.loads((own / "selection.json").read_text())
    r_star = int(sel["r_star"])
    d.out, d.cache = src, src / "cache"                  # 既有快取只讀
    C5 = N5.Ctx5(d)
    fn = N6.ARX(d, C5).fn(GAMMA)

    def lora_dir(r, o):
        return (own / f"lora_r{r}" / o) if r == 3 else (src / "lora_v2" / f"r{r}" / o)

    vals, per_fold = {}, {}

    def put(key, xs):
        per_fold[key] = xs
        vals[key] = mean_sd(xs)

    # LoRA 對照版：r = 2（一致性檢查）與 r*
    for r in sorted({2, r_star}):
        evs = {o: {f: torch.load(lora_dir(r, o) / f"fold{f}_eval.pt", map_location="cpu") for f in FOLDS}
               for o in ORDERS}
        for o in ORDERS:
            put(f"lora.r{r}.known.{o}",
                [sum(masked(evs[o][f]["tasks"][t]["l1_cos8"][:, p], evs[o][f]["tasks"][t]["labels"], p)
                     for p, t in enumerate(d.tasks)) / 4 for f in FOLDS])
            per_fold[f"lora.r{r}.known_task.{o}"] = (
                [[masked(evs[o][f]["tasks"][t]["l1_cos8"][:, p], evs[o][f]["tasks"][t]["labels"], p)
                  for p, t in enumerate(d.tasks)] for f in FOLDS])
        four = lambda f, o_, seen, e=evs: torch.cat([e[o_][f]["tasks"][d.tasks[p]]["l1_cos8"] for p in seen])  # noqa: E731
        res = N5.evaluate(d, C5, fn, "AR*", four)
        for o in ORDERS:
            put(f"lora.r{r}.inferred.{o}", [x["acc"] for x in res[o]["cil"]])

    # 修正頭 I6(r = 2)：同一程式重算
    ev6 = {f: torch.load(src / "i6" / f"eval_fold{f}.pt", map_location="cpu") for f in FOLDS}
    known6 = [sum(masked(ev6[f]["test"][t]["r2"]["cos8"][:, p], ev6[f]["test"][t]["labels"], p)
                  for p, t in enumerate(d.tasks)) / 4 for f in FOLDS]
    four6 = lambda f, o_, seen: torch.cat([ev6[f]["test"][d.tasks[p]]["r2"]["cos8"] for p in seen])  # noqa: E731
    res6 = N5.evaluate(d, C5, fn, "AR*", four6)
    for o in ORDERS:
        put(f"head.known.{o}", known6)
        put(f"head.inferred.{o}", [x["acc"] for x in res6[o]["cil"]])

    # 操作定義 9：一致性檢查
    nc8 = json.loads((src / "nc8" / "per_fold.json").read_text())["rows"]
    checks, failed = [], []
    for k, (want, where) in EXPECT.items():
        got = vals[k][0]
        ok = round(got, 4) == want
        row = {"key": k, "expected": want, "source": where, "got": got, "ok": ok}
        stem = k.rsplit(".", 1)[0]
        if stem in NC8_ROW:
            ref = [x["acc"] for x in nc8[NC8_ROW[stem]][k.rsplit(".", 1)[1]]]
            row["nc8_per_fold_maxabs"] = max(abs(a - b) for a, b in zip(per_fold[k], ref))
            if stem == "lora.r2.inferred" and row["nc8_per_fold_maxabs"] > 1e-6:
                ok = row["ok"] = False
        checks.append(row)
        if not ok:
            failed.append(k)

    # 表二：r* 與修正頭的差距（修正頭 − LoRA 對照版）
    gap = {}
    for kind in ("known", "inferred"):
        for o in ORDERS:
            h, l_ = per_fold[f"head.{kind}.{o}"], per_fold[f"lora.r{r_star}.{kind}.{o}"]
            diffs = [a - b for a, b in zip(h, l_)]
            m = mean_sd(diffs)[0]
            gap[f"{kind}.{o}"] = {"head": vals[f"head.{kind}.{o}"][0], "lora": vals[f"lora.r{r_star}.{kind}.{o}"][0],
                                  "gap": m, "per_fold": diffs, "head_higher": sum(x > 0 for x in diffs),
                                  "ties": sum(x == 0 for x in diffs), "within_tol": m >= -TOL}

    result = {"r_star": r_star, "r_star_meets": sel["r_star_meets"], "selection_commit": sel_commit,
              "values": {k: {"mean": v[0], "sd": v[1]} for k, v in vals.items()},
              "known_task": {k: v for k, v in per_fold.items() if "known_task" in k},
              "checks": checks, "checks_failed": failed, "gap": gap, "tolerance": TOL,
              "seconds": round(time.perf_counter() - t0, 1)}
    (own / "result.json").write_text(json.dumps(result, indent=1))
    (own / "per_fold.json").write_text(json.dumps({"folds": FOLDS, **per_fold}, indent=1))

    # fact-id：一行一個
    facts = [{"id": "nc15.val.l0.wp", "value": sel["l0_val_mean"], "sd": sel["l0_val_sd"],
              "definition": "L0 validation WP（四輪 Masked ACC，四任務平均，十折平均；NC-2 validation 快取）"},
             {"id": "nc15.val.threshold", "value": sel["threshold"],
              "definition": "門檻 = L0 validation WP − 0.01（未四捨五入）"}]
    for r in (1, 2, 3):
        for o in ORDERS:
            w = sel["wp_val"][f"r{r}.{o}"]
            facts.append({"id": f"nc15.val.r{r}.{o}.wp", "value": w["mean"], "sd": w["sd"],
                          "definition": f"LoRA 對照版 L1(r = {r})；{o}；validation WP 十折平均；過門檻 = {w['pass']}"})
    facts.append({"id": "nc15.rstar", "value": r_star,
                  "definition": f"validation 選出的 r（兩序都過門檻的最小 r；符合 = {sel['r_star_meets']}）"})
    for kind, name in (("known", "task-known（oracle 分派）test WP"), ("inferred", "task-inferred（AR 分派）t = 4 ACC")):
        for o in ORDERS:
            for r in sorted({2, r_star}):
                v = vals[f"lora.r{r}.{kind}.{o}"]
                facts.append({"id": f"nc15.lora.r{r}.{kind}.{o}.acc", "value": v[0], "sd": v[1],
                              "definition": f"LoRA 對照版 L1(r = {r})；{name}；{o}；十折平均"})
            v = vals[f"head.{kind}.{o}"]
            facts.append({"id": f"nc15.head.{kind}.{o}.acc", "value": v[0], "sd": v[1],
                          "definition": f"修正頭 I6(r = 2)；{name}；{o}；十折平均（本輪重算）"})
            g = gap[f"{kind}.{o}"]
            facts.append({"id": f"nc15.gap.{kind}.{o}", "value": g["gap"],
                          "definition": f"修正頭 − LoRA 對照版 L1(r* = {r_star})；{name}；{o}；十折平均差；"
                                        f"≥ −{TOL} = {g['within_tol']}；修正頭較高 {g['head_higher']}/10"})
    lines = ",\n".join(json.dumps(x, ensure_ascii=False) for x in facts)
    (own / "facts.json").write_text("[\n" + lines + "\n]\n")

    for c in checks:
        extra = f"；nc8 逐折最大差 {c['nc8_per_fold_maxabs']:.1e}" if "nc8_per_fold_maxabs" in c else ""
        print(f"check {c['key']}: {c['got']:.4f} vs {c['expected']:.4f}（{c['source']}）{'符合' if c['ok'] else '不符'}{extra}")
    if failed:
        print(f"一致性檢查不符：{failed}，停止")
        return 1
    for k, g in gap.items():
        print(f"{k}: 修正頭 {g['head']:.4f}  LoRA r*={r_star} {g['lora']:.4f}  差 {g['gap']:+.4f}  "
              f"{'在' if g['within_tol'] else '不在'} 0.005 內")
    print(f"→ {own / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
