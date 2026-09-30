#!/usr/bin/env python3
"""NC-11 報告：讀 nc11/{paired,select,topk}/result.json 與 facts.json，寫 REPORT_stage13.md（PREREG-11）。

    NAVCIL_MACHINE=mac python scripts/nc11_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS                                       # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

FOLDS = list(range(1, 11))
SHORT = ["esca", "rcc", "brca", "lung"]
MNAME = {"tp_micro": "分派正確率 micro", "tp_macro": "分派正確率 macro", "acc": "CIL 全程"}
fmt = N2.fmt


class Facts:
    def __init__(self, path: Path, rel: str):
        self.rel = rel
        self.lines = path.read_text().splitlines()
        self.v = json.loads(path.read_text())

    def ref(self, fid: str) -> str:
        n = next(i for i, x in enumerate(self.lines, 1) if x.lstrip().startswith(f'"{fid}":'))
        return f"`{fid}`（{self.rel}:{n}）"


def mech() -> list[str]:
    return [
        "## 0 機制說明（唯讀，寫在最前面）", "",
        "**去重懲罰在做什麼（白話）**：任務分類頭先替每個 patch 打一個「像不像該任務亞型」的分數。一次挑 64 個分數最高的 patch 時，"
        "常常會挑到一大堆彼此幾乎一模一樣的相鄰 patch（同一塊組織），證據重複。現行做法改成分 4 輪、每輪挑 16 個：從第 2 輪起，"
        "每個還沒被挑的 patch 的分數，會扣掉「λ = 1.5 × 它跟已挑 patch 中最像那一個的 cosine」。跟已挑的很像的 patch 被扣得多，"
        "於是後面幾輪會轉去挑分數也高、但長得不一樣的區域，最後 64 個 patch 涵蓋的組織比較多樣。", "",
        "**每輪之間更新了什麼（附行號）**：呼叫鏈 `scripts/nc8_batch.py:63` → `selector/cil_ops.py:39-45` `four_round`"
        "（budget = K、step = 16、redundancy_weight = λ\\* = 1.5、normalize_base = True、redundancy_mode = \"maxsim\"）→ "
        "`selector/multiround.py:134-203` `SequentialBudgetedObserver.observe`。",
        "1. 迴圈前：任務分類頭的分數 s 只算一次（`nc8_batch.py:72`），做一次 z-score（`multiround.py:146-150`）；patch 特徵 L2 正規化（`:139`）；"
        "「對已選集合的最大 cosine」max_sim_seen 初始為 0（`:152`）。",
        "2. 每輪：調整後分數 = z(s) − λ · max_sim_seen（`:156-163`，第 1 輪 seen 為空，不扣）；已選 patch 設為 −∞，不重選（`:164`）；"
        "取最高的 16 個（`:166-169`）加入已選集合（`:173`）。",
        "3. 輪與輪之間：只更新「已選集合」與 max_sim_seen = max(max_sim_seen, 各 patch 對本輪新選 16 個的最大 cosine)（`:178-180`）。"
        "分數 s 不重算、任務分類頭不再呼叫、沒有查詢向量或文字向量更新；迴圈內不做分類（`:184`，confidence_threshold = None）。",
        "4. 選完後所選 patch 等權平均並 L2 正規化（`cil_ops.py:31-36`），對 8 類文字取 cosine（`nc8_batch.py:63`）。", "",
        "**I6 是否以 top-K 的輸出訓練（3.2 的前提）**：是。`selector/cil_ops.py:87-90`：每步以分數取 one-shot top-K（`top_k_select(s.detach(), budget)`），"
        "所選 patch 以 softmax(分數) 加權後分類、算 CE；`budget` 預設 64（`cil_ops.py:22`、`:66-67`），`scripts/nc7_i6.py:60-61` 未傳 budget。"
        "訓練時是一輪、不去重；推論時才用上面的 4 輪 × 16。因此 3.2 每個 K 都以 one-shot top-K 重新訓練（「全部」= budget 0，全部 patch 參與 softmax 加權），"
        "推論每輪 16 個、輪數 = K／16。3.3 不重新訓練，只換推論的挑法。", ""]


def repro_rows(res: dict, label: str) -> list[str]:
    out = []
    for k, v in res["repro"].items():
        name, o = k.split("|")
        ref = {"known": "oracle ＋ I6（task-known）", "inferred": "AR ＋ I6（task-inferred）"}[name]
        out.append(f"| {label} | {ref}，{o} | REPORT_stage10.md:{v['report10_line']} | {v['report10']} | {v['now']}；逐折四位相同 "
                   f"{v['folds_r4_equal']}/10（最大差 {v['max_fold_absdiff']:.1e}） | {'符合' if v['ok'] else '**不符**'} |")
    return out


def arm_table(res: dict, fx: Facts, prefix: str, arms: list[tuple[str, str]], ref_arm: str) -> list[str]:
    out = ["| 設定 | task-known（oracle ＋ I6） | task-inferred（AR ＋ I6） | 每折標準差（known／inferred） | 與參照差（pp，known／inferred） | 高於／低於參照的折數（known；inferred） | fact-id（reverse） |",
           "|---|---|---|---|---|---|---|"]
    for a, label in arms:
        r = res["arms"][a]
        same = all(r[n]["reverse"][f]["acc"] == r[n]["paper"][f]["acc"] for n in ("known", "inferred") for f in range(10))
        k, i = [x["acc"] for x in r["known"]["reverse"]], [x["acc"] for x in r["inferred"]["reverse"]]
        rk = [x["acc"] for x in res["arms"][ref_arm]["known"]["reverse"]]
        ri = [x["acc"] for x in res["arms"][ref_arm]["inferred"]["reverse"]]
        dk, di = 100 * (mean_sd(k)[0] - mean_sd(rk)[0]), 100 * (mean_sd(i)[0] - mean_sd(ri)[0])
        wl = lambda x, y: f"{sum(a > b for a, b in zip(x, y))}／{sum(a < b for a, b in zip(x, y))}"  # noqa: E731
        out.append(f"| {label}{'' if same else '（兩序不同，見下）'} | {mean_sd(k)[0]:.4f} | {mean_sd(i)[0]:.4f} | {mean_sd(k)[1]:.4f}／{mean_sd(i)[1]:.4f} | "
                   f"{dk:+.2f}／{di:+.2f} | {wl(k, rk)}；{wl(i, ri)} | {fx.ref(f'{prefix}.{a}.known.reverse.acc')}；{fx.ref(f'{prefix}.{a}.inferred.reverse.acc')} |")
    out += ["", "兩序分列（t = 4 ACC，十折 mean ± sd）：", "", "| 設定 | known reverse | known paper | inferred reverse | inferred paper |", "|---|---|---|---|---|"]
    for a, label in arms:
        r = res["arms"][a]
        out.append(f"| {label} | " + " | ".join(fmt([x["acc"] for x in r[n][o]]) for n in ("known", "inferred") for o in ORDERS) + " |")
    out += ["", "Forgetting（t = 4，十折平均；task-inferred，reverse／paper）：", "", "| 設定 | reverse | paper |", "|---|---|---|"]
    for a, label in arms:
        r = res["arms"][a]["inferred"]
        out.append(f"| {label} | {mean_sd([x['forgetting'] for x in r['reverse']])[0]:.4f} | {mean_sd([x['forgetting'] for x in r['paper']])[0]:.4f} |")
    return out


def diff_pp(res, a, ref, n):
    x = mean_sd([v["acc"] for v in res["arms"][a][n]["reverse"]])[0]
    y = mean_sd([v["acc"] for v in res["arms"][ref][n]["reverse"]])[0]
    return 100 * (x - y)


def main() -> int:
    cfg = load_config()
    tasks = list(cfg["tasks"])
    out_dir = REPO_ROOT / "outputs" / "navcil" / cfg["machine"]
    d = out_dir / "nc11"
    P = json.loads((d / "paired" / "result.json").read_text())
    S = json.loads((d / "select" / "result.json").read_text())
    T = json.loads((d / "topk" / "result.json").read_text())
    fp = Facts(d / "paired" / "facts.json", "nc11/paired/facts.json")
    fs = Facts(d / "select" / "facts.json", "nc11/select/facts.json")
    ft = Facts(d / "topk" / "facts.json", "nc11/topk/facts.json")

    out = ["# REPORT — NC-11：逐折配對檢定、top-K ablation、選取方式 ablation（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、CPU、`torch.set_num_threads(8)`、torch 2.11.0；每個子實驗的所有 arm 在同一次執行中完成"
           "（log：`logs/nc11_paired.log`、`logs/nc11_select.log`、`logs/nc11_topk.log`）。判準與定義見 `PREREG-11.md`（commit 16275bc）。"
           "主方法 D3 = AR 分派器（γ = 1e-3，累加統計量）＋ I6(r = 2) 任務分類頭。task-known = oracle ＋ I6、task-inferred = AR ＋ I6，"
           "指標為 t = 4 CIL ACC（四任務等權平均），十折。數字附 fact-id 與其所在的 facts.json 行號（路徑相對於 `outputs/navcil/mac/`）。", ""]
    out += mech()

    # 重現檢查
    wk = T["weights_K64_vs_ref"]
    out += ["## 1 重現檢查（驗收 3）", "", "| 子實驗 | 項目 | 基準 | 基準值 | 本輪 | 結果 |", "|---|---|---|---|---|---|"]
    out += repro_rows(S, "3.3 arm (i) 4 輪 × 16（既有 I6 權重）")
    out += repro_rows(T, "3.2 K = 64（重新訓練）")
    out += ["", f"- 3.3 arm (i) 的 8 類 cosine 與 NC-8 快取 `I6_cos8` 逐張最大絕對差：{S['cos8_four16_vs_nc8_maxabs']:.2e}"
            f"（{fs.ref('nc11.select.cos8_four16_vs_nc8_maxabs')}）。",
            f"- 3.2 K = 64 重新訓練的權重與既有 `i6/r2/` 逐位元相同：{sum(v['equal'] for v in wk.values())}/{len(wk)}"
            f"（最大差 {max(v['maxabs'] for v in wk.values()):.2e}；{ft.ref('nc11.topk.weights_K64_equal')}）；"
            f"K = 64 的 8 類 cosine 與 `I6_cos8` 最大差 {T['cos8_K64_vs_nc8_maxabs']:.2e}（{ft.ref('nc11.topk.cos8_K64_vs_nc8_maxabs')}）。",
            f"- 3.1：`nc10/per_fold.json` 的 AR 與 `nc8/per_fold.json` 第 6 列逐折相同：{P['ar_equals_nc8_row6']}；重算的每折混淆十折合計與 "
            f"`nc10/confusion.json` 相同：{P['confusion_sum_equals_nc10']}。", ""]

    # 3.1
    out += ["## 2 逐折配對檢定（3.1；驗收 1）", "",
            "資料：`nc10/per_fold.json`（REPORT_stage12，同一批 I6 任務分類頭下的 AR、R3(k=8)、R0）。t = 4。Wilcoxon signed-rank 雙尾"
            "（`scipy.stats.wilcoxon`，n = 10、無平手時為精確檢定，最小可能 p = 2／1024 ≈ 0.0020）、paired t 雙尾。"
            f"共做 {P['n_tests']} 次檢定，未做多重比較校正。t = 4 時兩序的分派結果與 CIL 逐折相同（見下表），兩序各列一次。", ""]
    for comp in ("R3", "R0"):
        for o in ORDERS:
            t = {m: P["tests"][f"{comp}|{o}|{m}"] for m in MNAME}
            out += [f"### AR vs {comp}，{o}", "",
                    "| 折 | " + " | ".join(f"{comp} {MNAME[m]} | AR | 差" for m in MNAME) + " |", "|---|" + "---|---|---|" * 3]
            for k in range(10):
                out.append(f"| {k + 1} | " + " | ".join(f"{t[m][comp][k]:.4f} | {t[m]['AR'][k]:.4f} | {t[m]['diff'][k]:+.4f}" for m in MNAME) + " |")
            out += ["", "| 指標 | 平均差 | AR 勝／平／負 | Wilcoxon p | paired t p | fact-id |", "|---|---|---|---|---|---|"]
            for m in MNAME:
                x = t[m]
                fid = f"nc11.paired.{comp.lower()}.{o}.{m}"
                out.append(f"| {MNAME[m]} | {x['mean_diff']:+.4f} | {x['wins']}／{x['ties']}／{x['losses']} | {x['p_wilcoxon']:.4f} | "
                           f"{x['p_ttest']:.2e} | {fp.ref(fid + '.p_wilcoxon')}；{fp.ref(fid + '.p_ttest')} |")
            out.append("")
    worse = P["worse"]
    out += ["### AR 變差的折與原因初判", ""]
    if not worse:
        out += ["沒有任何折 AR 變差。", ""]
    for key, items in sorted(worse.items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[1]))):
        comp, k = key.split("|")
        k = int(k)
        ms = sorted({m for _, m in items}, key=list(MNAME).index)
        out += [f"**AR vs {comp}，第 {k} 折**：變差的指標 = " + "、".join(MNAME[m] for m in ms) + "（兩序皆然）" if len({o for o, _ in items}) == 2
                else f"**AR vs {comp}，第 {k} 折**：變差 = " + "、".join(f"{o} {MNAME[m]}" for o, m in items), "",
                "| 真實 \\ 分派 | " + " | ".join(f"{s}（{comp}／AR）" for s in SHORT) + " |", "|---|" + "---|" * 4]
        ca, cc = P["per_fold_confusion"][f"AR|reverse|{k}"], P["per_fold_confusion"][f"{comp}|reverse|{k}"]
        for i in range(4):
            out.append(f"| {SHORT[i]} | " + " | ".join(f"{cc[i][j]}／{ca[i][j]}" for j in range(4)) + " |")
        dec = P["decomp"][f"{comp}|reverse|{k}"]
        out += ["", f"| 任務 | test 張數 | TP {comp} | TP AR | 只有 AR 分派對（其中第二站判對） | 只有 {comp} 分派對（其中第二站判對） | 對 CIL 的貢獻（pp） |", "|---|---|---|---|---|---|---|"]
        worst = None
        for p, r in enumerate(dec):
            n = r["n"]
            tpc, tpa = cc[p][p] / n, ca[p][p] / n
            contrib = 100 * (r["ar_only_head_ok"] - r["comp_only_head_ok"]) / n / 4
            out.append(f"| {r['task']} | {n} | {tpc:.4f} | {tpa:.4f} | {r['ar_only']}（{r['ar_only_head_ok']}） | "
                       f"{r['comp_only']}（{r['comp_only_head_ok']}） | {contrib:+.2f} |")
            if worst is None or contrib < worst[1]:
                worst = (p, contrib, r, n)
        p, c, r, n = worst
        wrong_to = [(SHORT[j], ca[p][j]) for j in range(4) if j != p and ca[p][j] > cc[p][j]]
        out += ["", f"初判：CIL 差 = Σ_任務（只有 AR 分派對且第二站判對 − 只有 {comp} 分派對且第二站判對）／張數／4；兩者都分派對的片第二站結果相同，不影響差值。"
                f"本折負貢獻最大的是 {SHORT[p]}（{c:+.2f} pp；{n} 張中 AR 多分派錯 {r['comp_only']} 張"
                + (f"，AR 多出的錯誤去向：" + "、".join(f"{t} {v}" for t, v in wrong_to) if wrong_to else "") + "）。"
                + ("esca 每折只有約 15 張 test，一張分派錯就讓該任務正確率變動約 6.7 pp、四任務等權的 CIL 變動約 1.7 pp。" if SHORT[p] == "esca" else ""), ""]

    # 3.2
    arms_t = [("16", "K = 16（1 輪 × 16）"), ("32", "K = 32（2 輪 × 16）"), ("64", "K = 64（4 輪 × 16，現行）"),
              ("128", "K = 128（8 輪 × 16）"), ("all", "全部（訓練與推論都不選取；推論 = 全部 patch 等權平均）"),
              ("all_softmax", "另報：全部（推論以 softmax(分數) 加權）")]
    small = [a for a in ("16", "32", "128")]
    mk = max(abs(diff_pp(T, a, "64", "known")) for a in small)
    mi = max(abs(diff_pp(T, a, "64", "inferred")) for a in small)
    ak, ai = diff_pp(T, "all", "64", "known"), diff_pp(T, "all", "64", "inferred")
    out += ["## 3 top-K ablation（3.2；驗收 2）", "",
            "每個 K 都以 one-shot top-K 重新訓練 I6（其餘設定同 NC-7：r = 2、seed 42、5 epochs、lr 5e-4、wd 1e-4），推論每輪 16 個、輪數 = K／16、λ\\* = 1.5。"
            "分派器（AR）與 K 無關。參照 = K = 64。", ""]
    out += arm_table(T, ft, "nc11.topk", arms_t, "64")
    out += ["", f"**結論**：K ∈ {{16, 32, 64, 128}} 內，task-known 與 K = 64 的差距不超過 {mk:.2f} pp、task-inferred 不超過 {mi:.2f} pp；"
            f"不做選取（全部 patch 等權平均）時 task-known {ak:+.2f} pp、task-inferred {ai:+.2f} pp。", "",
            "註：「全部」的推論是全部 patch 等權平均（= 整片 mean_vec），任務分類頭不參與，所以這一列反映的是「不用任務分類頭挑證據」；"
            "它的任務分類頭雖以全部 patch 訓練，只在「另報：softmax 加權」一列被使用。", ""]
    tm = T.get("timing", {})
    out += ["訓練耗時（秒，十折四任務合計）：" + "；".join(f"K = {k if k != '0' else '全部'} {tm.get(f'train_total/K{k}', float('nan')):,.0f}"
                                                  for k in ("16", "32", "64", "128", "0"))
            + f"；評估 {sum(v for k, v in tm.items() if k.startswith('eval/')):,.0f}；全部 {T.get('seconds_total', float('nan')):,.0f}。", ""]

    # 3.3
    arms_s = [("four16", "(i) 4 輪 × 16，去重懲罰 λ = 1.5（現行）"), ("oneshot64", "(ii) 一次取 top-64，不去重")]
    sk, si = diff_pp(S, "oneshot64", "four16", "known"), diff_pp(S, "oneshot64", "four16", "inferred")
    out += ["## 4 選取方式 ablation（3.3；驗收 2）", "",
            "訓練不變（既有 I6 r = 2，one-shot top-64 訓練）。只換推論時的挑法；參照 = (i)。", ""]
    out += arm_table(S, fs, "nc11.select", arms_s, "four16")
    j = S["jaccard_own"]
    out += ["", f"兩種挑法所選 64 個 patch 的 Jaccard（自家任務的頭，{j['n']:,} 張 test slide）：平均 {j['mean']:.4f}、中位數 {j['median']:.4f}、最小 {j['min']:.4f}"
            f"（{fs.ref('nc11.select.jaccard_own.mean')}）。", "",
            f"**結論**：一次取 top-64 不去重與 4 輪 × 16 去重相比，task-known 差 {sk:+.2f} pp、task-inferred 差 {si:+.2f} pp。", ""]
    (out_dir / "REPORT_stage13.md").write_text("\n".join(out))
    print(f"→ {out_dir / 'REPORT_stage13.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
