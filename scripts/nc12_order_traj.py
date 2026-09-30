#!/usr/bin/env python3
"""NC-12（PREREG-12）：六種學習順序下 D3（AR 累加統計量 ＋ I6 r=2）的逐階段軌跡，與 t = 4 順序無關的驗證。

    NAVCIL_MACHINE=mac python scripts/nc12_order_traj.py --cache-dir <NC-8 快取目錄>
輸出：outputs/navcil/<machine>/nc12/{per_fold,result,facts}.json、REPORT_stage14.md
判準任一不符：仍寫出全部輸出，結束碼 1。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
import nc8_report as N8                                                   # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS                                       # noqa: E402
from selector.incremental_ridge import aug, stream                        # noqa: E402

FOLDS = N8.FOLDS
SEED = 20260930
AR_GAMMA = 1e-3
W_TOL = 1e-9
SHORT = {"tcga_esca": "esca", "tcga_rcc": "rcc", "tcga_brca": "brca", "tcga_lung": "lung"}
fmt = N2.fmt


def orders6(tasks: list[str]) -> dict[str, list[str]]:
    """reverse、paper ＋ 以 random.Random(SEED) 產生、與前者皆不重複的 4 種排列（PREREG-12）。"""
    out = {"reverse": list(ORDERS["reverse"]), "paper": list(ORDERS["paper"])}
    rng = random.Random(SEED)
    while len(out) < 6:
        p = rng.sample(tasks, 4)
        if p not in out.values():
            out[f"perm{len(out) - 1}"] = p
    return out


def gather(b, f, seen):
    """已見任務 test slides 依 seen 次序串接；只取 labels、mean_vec、I6_cos8 與任務標記。

    不用 nc8_report.B8.gather：它會依已見子集取 zs8_cos8 的欄，而 NC-8 快取只存 reverse、paper 的前綴子集
    （cil_eval.all_subsets），新順序會找不到（第一次執行失敗，logs/nc12_order.log）。
    """
    parts = [b.c(f, "test", b.tasks[p]) for p in seen]
    return {"labels": torch.cat([c["labels"] for c in parts]), "mean_vec": torch.cat([c["mean_vec"] for c in parts]),
            "I6": torch.cat([c["I6_cos8"] for c in parts]),
            "task": torch.cat([torch.full((len(c["labels"]),), p) for p, c in zip(seen, parts)])}


def run(b, name, names, f):
    pos = [b.tasks.index(x) for x in names]
    reads = []

    def load(p):
        reads.append(p)
        return torch.load(b.cache / f"nc8_fold{f}_train_{b.tasks[p]}.pt", map_location="cpu")["mean_vec"]
    stages = []
    for t, m in enumerate(stream(pos, load, AR_GAMMA), 1):
        assert reads == pos[:t]
        seen = pos[:t]
        W = m.solve()
        g = gather(b, f, seen)
        th = torch.tensor(seen)[(aug(g["mean_vec"]) @ W).argmax(-1)]
        pred, _ = N2.hard(g["I6"], th, g["task"])
        ok, okr = pred == g["labels"], th == g["task"]
        acc_task = [ok[g["task"] == p].float().mean().item() for p in seen]      # 依該序的任務次序（同 cil_full）
        tp_task = [okr[g["task"] == p].float().mean().item() for p in seen]
        cm = torch.zeros(4, 4, dtype=torch.long)
        for a, c in zip(g["task"].tolist(), th.tolist()):
            cm[a, c] += 1
        rec = {"acc": sum(acc_task) / t, "tp_micro": okr.float().mean().item(), "tp_macro": sum(tp_task) / t,
               "acc_task": dict(zip(map(int, seen), acc_task)), "conf": cm.tolist()}
        if t == 4:
            n = [int((g["task"] == p).sum()) for p in seen]
            parts = th.split(n)
            rec["_th_canon"] = torch.cat([parts[seen.index(p)] for p in range(4)])
            rec["_W_canon"] = W[:, [seen.index(p) for p in range(4)]]
        stages.append(rec)
    return stages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()
    t0 = time.perf_counter()
    b = N8.B8()
    b.cache = Path(os.path.expanduser(args.cache_dir))
    O = orders6(b.tasks)
    res = {o: [run(b, o, names, f) for f in FOLDS] for o, names in O.items()}

    # t = 4 一致性
    old = json.loads((b.out / "nc8" / "per_fold.json").read_text())["rows"][N8.ROWS[5]]
    check = {"tau_mismatch": {}, "acc_bitwise_all_orders": {}, "acc_equals_nc8": {}, "w_maxabs_vs_reverse": {}}
    for k, f in enumerate(FOLDS):
        ref = res["reverse"][k][3]
        check["tau_mismatch"][f] = {o: int((res[o][k][3]["_th_canon"] != ref["_th_canon"]).sum()) for o in O}
        accs = {o: res[o][k][3]["acc"] for o in O}
        check["acc_bitwise_all_orders"][f] = len({v.hex() for v in accs.values()}) == 1
        check["acc_equals_nc8"][f] = {"nc8": old["reverse"][k]["acc"], "all_equal": all(v == old["reverse"][k]["acc"] for v in accs.values())}
        check["w_maxabs_vs_reverse"][f] = {o: (res[o][k][3]["_W_canon"] - ref["_W_canon"]).abs().max().item() for o in O}
    traj_check = {o: all(res[o][k][t]["acc"] == old[o][k]["acc_t"][t] for k in range(10) for t in range(4)) for o in ("reverse", "paper")}
    crit = {"1_tau": all(v == 0 for d in check["tau_mismatch"].values() for v in d.values()),
            "2_acc": all(check["acc_bitwise_all_orders"].values()) and all(v["all_equal"] for v in check["acc_equals_nc8"].values()),
            "3_W": max(v for d in check["w_maxabs_vs_reverse"].values() for v in d.values()) <= W_TOL,
            "4_traj_nc8": all(traj_check.values())}

    # 彙總、facts
    facts, summary = {}, {}
    for o in O:
        for t in range(4):
            for m in ("acc", "tp_micro", "tp_macro"):
                xs = [res[o][k][t][m] for k in range(10)]
                mu, sd = mean_sd(xs)
                summary[(o, t + 1, m)] = (mu, sd)
                facts[f"nc12.traj.{o}.t{t + 1}.{m}"] = mu
                facts[f"nc12.traj.{o}.t{t + 1}.{m}_sd"] = sd
    mid = min(((o, t) for o in O for t in (2, 3)), key=lambda x: summary[(x[0], x[1], "tp_micro")][0])
    conf_mid = torch.zeros(4, 4, dtype=torch.long)
    for k in range(10):
        conf_mid += torch.tensor(res[mid[0]][k][mid[1] - 1]["conf"])
    facts["nc12.check.tau_mismatch_total"] = sum(v for d in check["tau_mismatch"].values() for v in d.values())
    facts["nc12.check.w_maxabs"] = max(v for d in check["w_maxabs_vs_reverse"].values() for v in d.values())
    for k, f in enumerate(FOLDS):
        facts[f"nc12.t4.fold{f}.acc"] = res["reverse"][k][3]["acc"]

    d = b.out / "nc12"
    d.mkdir(parents=True, exist_ok=True)
    clean = {o: [[{k: v for k, v in s.items() if not k.startswith("_")} for s in st] for st in v] for o, v in res.items()}
    (d / "per_fold.json").write_text(json.dumps({"orders": O, "seed": SEED, "folds": FOLDS, "per_fold": clean}, indent=1))
    out = {"orders": O, "seed": SEED, "criteria": crit, "check": check, "traj_equals_nc8": traj_check,
           "lowest_mid": {"order": mid[0], "t": mid[1], "tp_micro": summary[(mid[0], mid[1], "tp_micro")],
                          "tp_macro": summary[(mid[0], mid[1], "tp_macro")], "confusion": conf_mid.tolist()},
           "seconds": time.perf_counter() - t0}
    (d / "result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (d / "facts.json").write_text(json.dumps(facts, indent=1))
    write(b, O, res, summary, check, crit, traj_check, mid, conf_mid, d, out)
    print(f"→ {b.out / 'REPORT_stage14.md'}")
    for k, v in crit.items():
        print(f"  判準 {k}: {'通過' if v else '不通過'}")
    return 0 if all(crit.values()) else 1


def write(b, O, res, summary, check, crit, traj_check, mid, conf_mid, d, out_json):
    L = (d / "facts.json").read_text().splitlines()

    def ref(fid):
        n = next(i for i, x in enumerate(L, 1) if x.lstrip().startswith(f'"{fid}":'))
        return f"`{fid}`（nc12/facts.json:{n}）"
    ok = lambda v: "通過" if v else "**不通過**"                               # noqa: E731
    seq = lambda names: " → ".join(SHORT[x] for x in names)                   # noqa: E731
    out = ["# REPORT — NC-12：六種學習順序下的中途軌跡與 t = 4 順序無關驗證（Mac CPU，十折）", "",
           "機器：mac（Apple M1 Pro）、CPU、torch 2.11.0；所有數字來自同一台、同一次執行（`scripts/nc12_order_traj.py`，log：`logs/nc12_order_run2.log`；第一次執行見文末「執行紀錄」）。"
           "判準與定義見 `PREREG-12.md`（commit bf6cbf7）。主方法 D3 = AR 分派器（γ = 1e-3，`selector/incremental_ridge.py` 累加統計量，"
           "第 t 階段只讀該順序第 t 個任務的 train mean_vec）＋ I6(r = 2) 任務分類頭；證據與 mean_vec 為 NC-8 同一批快取。"
           "CIL 正確率只計已見任務、各任務等權平均；分派正確率 micro = 已見任務 test slides 中分派正確的比例，macro = 各已見任務比例的等權平均。"
           "數字附 fact-id 與 `outputs/navcil/mac/nc12/facts.json` 的行號。", "",
           "## 順序", "", "| 代號 | 順序 | 來源 |", "|---|---|---|"]
    for o, names in O.items():
        out.append(f"| {o} | {seq(names)} | {'既有' if o in ('reverse', 'paper') else f'random.Random({out_json["seed"]}).sample，排除重複'} |")

    out += ["", "## T1 軌跡表：CIL 正確率（十折 mean ± sd）", "", "| 順序 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|"]
    for o in O:
        out.append(f"| {o}（{seq(O[o])}） | " + " | ".join(fmt([res[o][k][t]["acc"] for k in range(10)]) for t in range(4)) + " |")
    out += ["", "fact-id：`nc12.traj.<順序>.t<階段>.acc`（例：" + ref("nc12.traj.reverse.t4.acc") + "）。", "",
            "## T2 軌跡表：分派正確率 micro（十折 mean ± sd；t = 1 只有一個任務，恆為 1）", "",
            "| 順序 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|"]
    for o in O:
        out.append(f"| {o} | " + " | ".join(fmt([res[o][k][t]["tp_micro"] for k in range(10)]) for t in range(4)) + " |")
    out += ["", "分派正確率 macro：", "", "| 順序 | t = 1 | t = 2 | t = 3 | t = 4 |", "|---|---|---|---|---|"]
    for o in O:
        out.append(f"| {o} | " + " | ".join(fmt([res[o][k][t]["tp_macro"] for k in range(10)]) for t in range(4)) + " |")
    out += ["", "fact-id：`nc12.traj.<順序>.t<階段>.tp_micro`／`.tp_macro`。", ""]

    out += ["## T3 t = 4 一致性檢查（逐折）", "", "| 判準 | 內容 | 結果 |", "|---|---|---|",
            f"| 1 | 每張 test slide 的 τ̂ 在 6 種順序下逐張一致（依任務對齊） | {ok(crit['1_tau'])}（不一致合計 {sum(v for x in check['tau_mismatch'].values() for v in x.values())} 張；{ref('nc12.check.tau_mismatch_total')}） |",
            f"| 2 | t = 4 ACC 在 6 種順序下位元相同，且等於 nc8/per_fold.json 第 6 列 | {ok(crit['2_acc'])} |",
            f"| 3 | t = 4 的 W 與 reverse 最大絕對差 ≤ 1e-9 | {ok(crit['3_W'])}（最大 {max(v for x in check['w_maxabs_vs_reverse'].values() for v in x.values()):.2e}；{ref('nc12.check.w_maxabs')}） |",
            f"| 另核對 | reverse、paper 的逐階段 CIL 與 nc8/per_fold.json 第 6 列 acc_t 位元相同 | {ok(crit['4_traj_nc8'])} |", "",
            "| 折 | test 張數 | τ̂ 不一致（6 序合計） | t = 4 ACC（6 序） | 6 序位元相同 | 等於 NC-8 D3 | W 最大差（對 reverse） | fact-id |", "|---|---|---|---|---|---|---|---|"]
    for k, f in enumerate(FOLDS):
        n = sum(sum(r) for r in res["reverse"][k][3]["conf"])
        out.append(f"| {f} | {n} | {sum(check['tau_mismatch'][f].values())} | {res['reverse'][k][3]['acc']:.4f} | "
                   f"{'是' if check['acc_bitwise_all_orders'][f] else '否'} | {'是' if check['acc_equals_nc8'][f]['all_equal'] else '否'} | "
                   f"{max(check['w_maxabs_vs_reverse'][f].values()):.2e} | {ref(f'nc12.t4.fold{f}.acc')} |")
    out += ["", f"十折平均 ± sd：{fmt([res['reverse'][k][3]['acc'] for k in range(10)])}（REPORT_stage10.md:16，`nc8.t1.d3.reverse.acc` 為 0.9128 ± 0.0258）。", ""]

    o, t = mid
    mi, ma = summary[(o, t, "tp_micro")], summary[(o, t, "tp_macro")]
    seen = [SHORT[x] for x in O[o][:t]]
    cm = conf_mid.tolist()
    names = list(SHORT.values())
    offd = sorted(((cm[i][j], names[i], names[j]) for i in range(4) for j in range(4) if i != j and cm[i][j] > 0), reverse=True)
    out += ["## T4 觀察：中途分派正確率最低的階段", "",
            f"t = 2、3 之中，十折平均分派正確率 micro 最低的是 **{o}（{seq(O[o])}）的 t = {t}**：micro {mi[0]:.4f} ± {mi[1]:.4f}、"
            f"macro {ma[0]:.4f} ± {ma[1]:.4f}（{ref(f'nc12.traj.{o}.t{t}.tp_micro')}）；已見任務為 {'、'.join(seen)}。"
            "主要錯分：" + "、".join(f"{a}→{c} {v} 張" for v, a, c in offd[:3]) + "。", "",
            f"十折合計混淆矩陣（{o}、t = {t}；列 = 真實、欄 = 分派，只含已見任務）：", "",
            "| 真實 \\ 分派 | " + " | ".join(names) + " |", "|---|" + "---|" * 4]
    for i in range(4):
        if names[i] in seen:
            out.append(f"| {names[i]} | " + " | ".join(str(cm[i][j]) if names[j] in seen else "—" for j in range(4)) + " |")
    out += ["", f"耗時：{out_json['seconds']:.0f} 秒。", "",
            "## 執行紀錄", "",
            "1. 第一次執行（2026-09-30 09:28:48，log：`outputs/navcil/mac/logs/nc12_order.log`，exit = 1）因腳本 bug 失敗："
            "`ValueError: '1,3' is not in list`，發生在 `scripts/nc8_report.py:58`（`B8.gather`）。",
            "2. 原因：當時的腳本呼叫 NC-8 的 `B8.gather` 組合已見任務的 test 資料；它會依已見子集取 zero-shot 證據 `zs8_cos8` 的欄，"
            "而 NC-8 快取的子集清單只含 reverse、paper 的 7 個前綴（`selector/cil_eval.py:83-90`、`scripts/nc8_batch.py:65-67`）。"
            "perm1 在 t = 2 的已見子集 {rcc, lung}（`'1,3'`）不在清單中。D3 不使用 `zs8_cos8`。",
            "3. 修正（commit 訊息 \"fix: own gather without zs8 (first run failed, see log)\"）：改用本檔自寫的 `gather`，只取 labels、mean_vec、"
            "`I6_cos8` 與任務標記；既有 `nc8_report.py` 未改。重跑前已確認本檔用到的資料來源中，依子集或順序預先算好的只有 "
            "`zs8_cos8`（依子集）、`L1_cos8`（依順序）與中繼清單 `subsets`／`orders`，本檔都不使用；`I6_cos8`、`mean_vec`、`labels` 與順序無關。",
            "4. 失敗發生在第一個新順序（perm1）第 1 折的 t = 2，在寫出任何結果檔之前；不涉及 PREREG-12 的定義或判準變更。"
            "本報告所有數字來自修正後的整批重跑（`logs/nc12_order_run2.log`）。", ""]
    (b.out / "REPORT_stage14.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
