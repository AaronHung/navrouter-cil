#!/usr/bin/env python3
"""NC-14（PREREG-14）：依外部參考方法論文的定義重算 D3 的 ACC、Masked ACC、Forgetting。

只讀已 commit 的 JSON，不訓練、不推論、不讀特徵檔：
  nc8/per_fold.json（主來源，D3 的 R、Rm）、nc9/per_fold.json 與 nc12/per_fold.json（交叉核對 R／Rm）、
  reference/external_baselines.json（外部方法名稱與已發表平均值）、nc14/external_ref.json（已發表 sd）。
一致性檢查（PREREG-14）任何一項不符即停，不寫報告。

    NAVCIL_MACHINE=mac python scripts/nc14_ext_metrics.py
輸出：outputs/navcil/<machine>/REPORT_stage16.md、nc14/result.json、nc14/per_fold.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS                                       # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

ROW = "6 D3：AR＋I6(r=2)（主系統）"
OS = ("reverse", "paper")
TARGET = {"acc": {"reverse": "0.9128", "paper": "0.9128"},
          "masked": {"reverse": "0.9312", "paper": "0.9312"},
          "forg_o": {"reverse": "0.0224", "paper": "0.0041"}}
SHORT = {"tcga_esca": "esca", "tcga_rcc": "rcc", "tcga_brca": "brca", "tcga_lung": "lung"}


def acc_q(R):                       # Eq. 11
    return sum(R[3]) / 4


def forg_terms_q(R):                # Eq. 13：max 含 i = T（只取已定義的 i ≥ t）
    return [max(R[i][t] for i in range(t, 4)) - R[3][t] for t in range(3)]


def forg_terms_o(R):                # PREREG.md:36；同 scripts/nc5_report.py:55 的算式
    return [max(R[i][t] for i in range(t, 3)) - R[3][t] for t in range(3)]


def forg_q(R):
    return sum(max(R[i][t] for i in range(t, 4)) - R[3][t] for t in range(3)) / 3


def forg_o(R):
    return sum(max(R[t][j] for t in range(j, 3)) - R[3][j] for j in range(3)) / 3


def f4(x):
    return f"{x:.4f}"


def ms(xs):
    m, s = mean_sd(xs)
    return f"{m:.4f} ± {s:.4f}"


def main() -> int:
    t0 = time.time()
    cfg = load_config()
    out = REPO_ROOT / "outputs" / "navcil" / cfg["machine"]
    nc8 = json.loads((out / "nc8" / "per_fold.json").read_text())
    nc9 = json.loads((out / "nc9" / "per_fold.json").read_text())
    nc12 = json.loads((out / "nc12" / "per_fold.json").read_text())
    ext = json.loads((REPO_ROOT / "reference" / "external_baselines.json").read_text())
    ext_sd = json.loads((out / "nc14" / "external_ref.json").read_text())
    folds = nc8["folds"]
    assert folds == nc9["folds"] == nc12["folds"], "三個來源的折序不同"

    per, checks = {}, {}
    for o in OS:
        recs8, recs9 = nc8["rows"][ROW][o], nc9["rows"][ROW][o]
        rows = []
        for k, f in enumerate(folds):
            r8, r9 = recs8[k], recs9[k]
            R, Rm = r8["R"], r8["Rm"]
            # AMENDMENT-3：NC-12 的 acc_task 以任務編號為鍵（scripts/nc12_order_traj.py:83），依該序換算成位置
            tid = [cfg["tasks"].index(x) for x in ORDERS[o]]
            R12 = [[nc12["per_fold"][o][k][t]["acc_task"].get(str(tid[j])) for j in range(4)] for t in range(4)]
            rows.append({
                "fold": f, "acc_q": acc_q(R), "masked_q": acc_q(Rm), "forg_q": forg_q(R), "forg_o": forg_o(R),
                "forg_terms_q": forg_terms_q(R), "forg_terms_o": forg_terms_o(R),
                "stored": {"acc": r8["acc"], "masked": r8["masked"], "forgetting": r8["forgetting"]},
                "same_R_nc9": R == r9["R"], "same_Rm_nc9": Rm == r9["Rm"], "same_R_nc12": R == R12,
            })
            x = rows[-1]
            x["fold_checks"] = {"1_acc_bitwise": x["acc_q"] == r8["acc"], "2_masked_bitwise": x["masked_q"] == r8["masked"],
                                "3_forg_o_bitwise": x["forg_o"] == r8["forgetting"], "4_R_nc8_nc9": x["same_R_nc9"],
                                "4_Rm_nc8_nc9": x["same_Rm_nc9"], "4_R_nc8_nc12": x["same_R_nc12"]}
        per[o] = rows
        m = {key: mean_sd([x[key] for x in rows]) for key in ("acc_q", "masked_q", "forg_q", "forg_o")}
        checks[o] = {
            "1_acc_bitwise": all(x["acc_q"] == x["stored"]["acc"] for x in rows),
            "1_acc_mean4": f4(m["acc_q"][0]) == TARGET["acc"][o],
            "2_masked_bitwise": all(x["masked_q"] == x["stored"]["masked"] for x in rows),
            "2_masked_mean4": f4(m["masked_q"][0]) == TARGET["masked"][o],
            "3_forg_o_bitwise": all(x["forg_o"] == x["stored"]["forgetting"] for x in rows),
            "3_forg_o_mean4": f4(m["forg_o"][0]) == TARGET["forg_o"][o],
            "4_R_nc8_nc9": all(x["same_R_nc9"] for x in rows),
            "4_Rm_nc8_nc9": all(x["same_Rm_nc9"] for x in rows),
            "4_R_nc8_nc12": all(x["same_R_nc12"] for x in rows),
            "means": {key: list(v) for key, v in m.items()},
        }

    ok = all(v for o in OS for key, v in checks[o].items() if key != "means")
    (out / "nc14").mkdir(parents=True, exist_ok=True)
    (out / "nc14" / "per_fold.json").write_text(json.dumps(per, ensure_ascii=False, indent=1))
    (out / "nc14" / "result.json").write_text(json.dumps(
        {"pass": ok, "checks": checks, "seconds": round(time.time() - t0, 3)}, ensure_ascii=False, indent=1))
    if not ok:
        bad = [f"{o}:{key}" for o in OS for key, v in checks[o].items() if key != "means" and not v]
        print("一致性檢查不符，停止（不寫報告）：", bad)
        return 1

    name = ext["method"]
    L = ["# REPORT — NC-14：依 " + name + " 的定義重算 D3 的 ACC、Masked ACC、Forgetting（Mac，只讀既有結果）", "",
         f"判準與定義見 `PREREG-14.md`。來源：`nc8/per_fold.json` 列「{ROW}」的 R、Rm（NC-8）；"
         "交叉核對 `nc9/per_fold.json`（NC-9）與 `nc12/per_fold.json`（NC-12）。不訓練、不推論、不讀特徵檔。"
         f"{name} 的值為已發表值（`reference/external_baselines.json`；sd 見 `nc14/external_ref.json`），非 paired。", "",
         f"## T1 主表（依 {name} 定義：ACC = Eq. 11、Masked ACC、Forgetting = Eq. 13；test、t = 4、十折 mean ± sd）", "",
         "| 序 | 方法 | ACC | Masked ACC | Forgetting | 出處 |", "|---|---|---|---|---|---|"]
    for o in OS:
        rs = per[o]
        L.append(f"| {o} | 本方法（D3：AR＋I6(r=2)） | {ms([x['acc_q'] for x in rs])} | {ms([x['masked_q'] for x in rs])} | "
                 f"{ms([x['forg_q'] for x in rs])} | 本報告 T3 |")
        sd = ext_sd["sd"][o]
        L.append(f"| {o} | {name}（已發表值，非 paired） | {ext['acc'][o]:.3f} ± {sd['acc']:.3f} | "
                 f"{ext['masked_acc'][o]:.3f} ± {sd['masked_acc']:.3f} | {ext['forgetting'][o]:.3f} ± {sd['forgetting']:.3f} | "
                 f"{ext_sd['table'][o]} |")
    L += ["", f"paper 序 = {name} 的 forward（lung→brca→rcc→esca，Tab. 1）；reverse = esca→rcc→brca→lung（Tab. 2）。"
          f"{name} 論文只給三位小數。", ""]

    L += ["## T2 一致性檢查（PREREG-14）", "", "| 檢查 | reverse | paper |", "|---|---|---|"]
    desc = {"1_acc_bitwise": "1 ACC（Eq. 11）每折與 `acc` 位元相同",
            "1_acc_mean4": "1 ACC 十折平均四位 = 0.9128",
            "2_masked_bitwise": "2 Masked ACC 每折與 `masked` 位元相同",
            "2_masked_mean4": "2 Masked ACC 十折平均四位 = 0.9312",
            "3_forg_o_bitwise": "3 原定義 Forgetting 重算，每折與 `forgetting` 位元相同",
            "3_forg_o_mean4": "3 原定義 Forgetting 十折平均四位 = 0.0224（reverse）／0.0041（paper）",
            "4_R_nc8_nc9": "4 R：NC-8 = NC-9（逐折逐格位元相同）",
            "4_Rm_nc8_nc9": "4 Rm：NC-8 = NC-9",
            "4_R_nc8_nc12": "4 R：NC-8 = NC-12 `acc_task`"}
    pf = lambda v: "通過" if v else "不符"                                        # noqa: E731
    for key, d in desc.items():
        L.append(f"| {d} | {pf(checks['reverse'][key])} | {pf(checks['paper'][key])} |")
    L += ["", "逐折結果（六項逐折檢查；三項平均檢查只有十折一個值，見上表）：", "",
          "| 序 | 檢查 | " + " | ".join(str(f) for f in folds) + " | 通過折數 |", "|---|---|" + "---|" * (len(folds) + 1)]
    for o in OS:
        for key in per[o][0]["fold_checks"]:
            v = [x["fold_checks"][key] for x in per[o]]
            L.append(f"| {o} | {desc[key].split(' ', 1)[1]} | " + " | ".join("✓" if b else "✗" for b in v)
                     + f" | {sum(v)}/{len(v)} |")
    L += ["", "十折平均（全精度）：" + "；".join(
        f"{o} ACC {checks[o]['means']['acc_q'][0]:.6f}、Masked {checks[o]['means']['masked_q'][0]:.6f}、"
        f"Forgetting 原定義 {checks[o]['means']['forg_o'][0]:.6f}" for o in OS), ""]

    L += ["## T3 每折值", ""]
    for o in OS:
        L += [f"**{o}**", "", "| 折 | ACC | Masked ACC | Forgetting（依 Eq. 13） | Forgetting（原定義） |", "|---|---|---|---|---|"]
        for x in per[o]:
            L.append(f"| {x['fold']} | {f4(x['acc_q'])} | {f4(x['masked_q'])} | {f4(x['forg_q'])} | {f4(x['forg_o'])} |")
        L.append(f"| mean ± sd | {ms([x['acc_q'] for x in per[o]])} | {ms([x['masked_q'] for x in per[o]])} | "
                 f"{ms([x['forg_q'] for x in per[o]])} | {ms([x['forg_o'] for x in per[o]])} |")
        L.append("")

    L += ["## T4 兩種 Forgetting 的差異", "",
          "| 序 | 原定義（REPORT_stage10） | 依 Eq. 13 | 差（Eq. 13 − 原定義） | 被截為 0 的項數（共 30） |", "|---|---|---|---|---|"]
    for o in OS:
        a, b = [x["forg_o"] for x in per[o]], [x["forg_q"] for x in per[o]]
        n = sum(1 for x in per[o] for v in x["forg_terms_o"] if v < 0)
        L.append(f"| {o} | {ms(a)} | {ms(b)} | {mean_sd([q - p for p, q in zip(a, b)])[0]:+.4f} | {n} |")
    L += ["", "被截為 0 的項（原定義該項 < 0：該任務在 t = 4 的正確率高於它在 t < 4 各階段的最高值）：", "",
          "| 序 | 折 | 任務（該序第 t 個） | R_{i,t}，i = t…4 | 原定義該項 | Eq. 13 該項 |", "|---|---|---|---|---|---|"]
    for o in OS:
        names = [SHORT[t] for t in ORDERS[o]]
        for x, r8 in zip(per[o], nc8["rows"][ROW][o]):
            for t, (vo, vq) in enumerate(zip(x["forg_terms_o"], x["forg_terms_q"])):
                if vo < 0:
                    col = "／".join(f4(r8["R"][i][t]) for i in range(t, 4))
                    L.append(f"| {o} | {x['fold']} | {names[t]}（t = {t + 1}） | {col} | {vo:+.4f} | {vq:+.4f} |")
    L += ["", "定義差異（公式見 PREREG-14）：",
          "- 原定義：每項 = max_{t ≤ i ≤ 3} R_{i,t} − R_{4,t}，max 不含最後一階段，單項可為負；十折平均是正負相抵後的值。",
          f"- Eq. 13（{name}）：max 的範圍寫成 i ∈ {{1,…,T}}，含 i = T，所以每項 = max(原定義該項, 0) ≥ 0；"
          "i < t 的格在其 Tab. 6 為「-」，只取 i ≥ t。",
          "- 因此依 Eq. 13 的值 ≥ 原定義，兩者只在「有舊任務在 t = 4 的正確率回升到先前最高值以上」時不同，差額就是上表負項的絕對值平均。",
          "- ACC（Eq. 11）與 Masked ACC 兩種定義相同（T2 檢查 1、2），不需重算。", "",
          "## 執行紀錄", "",
          "1. PREREG-14 於 commit 04712d8 登記後，第一次執行本程式：九項檢查中只有「4 R：NC-8 = NC-12」在 paper 序不符，"
          "其餘八項兩序皆通過；依規定停止，未寫報告。",
          "2. 原因：NC-12 的 `acc_task` 以任務編號為鍵（0 = esca、1 = rcc、2 = brca、3 = lung，`scripts/nc12_order_traj.py:83`），"
          "核對程式誤當成該序中的位置。reverse 序兩者相同故通過，paper 序（lung→brca→rcc→esca）錯位。不是數據不一致。",
          "3. AMENDMENT-3（commit 4e00ae3）寫明檢查 4 的比對方式：依該序換算任務編號後再比對；不改任何判準、公式或期望值。",
          "4. 修正核對程式後整批重跑（本報告）：十折、兩序逐一檢查，九項全部通過（T2）。", "",
          f"耗時：{time.time() - t0:.2f} 秒（只讀 JSON）。"]
    (out / "REPORT_stage16.md").write_text("\n".join(L) + "\n")
    print("ok", {o: {k: round(v[0], 4) for k, v in checks[o]["means"].items()} for o in OS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
