#!/usr/bin/env python3
"""EXT-2 C（PREREG-22 細則 22）：FINAL-B 在 FINAL_EXAMPLE.md 的同兩張 slide 上逐步算出中間量。
由特徵檔重算；只推論、不訓練、不寫任何既有產物。FINAL-A(42) 的 head 只用來算「兩套選出的 64 個的交集」與並列的判定。

    NAVCIL_MACHINE=mac python scripts/ext2_example.py
輸出：outputs/navcil/<machine>/ext2/example.json、outputs/navcil/<machine>/FINALB_EXAMPLE.md
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ext1_c as CC                                                       # noqa: E402
import ext2_a as EA                                                       # noqa: E402
from ext1_c import C, D64, M, N5, T, X                                    # noqa: E402
from selector.cil_ops import STEP, four_round, mean_norm                  # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import text_nav_feats                         # noqa: E402
from selector.i6_expert import zscore                                     # noqa: E402

FOLD, ORDER, SEED = 1, "reverse", 42
SLIDES = [("tcga_lung", 0), ("tcga_brca", 26)]
LABEL = C.LABEL


def q3(x) -> list[float]:
    return [float(x.min()), float(x.median()), float(x.max())]


def one(cx, task: str, index: int, env: dict) -> dict:
    r3, ctx = cx.r3, cx.r3.ctx
    p_true = cx.tasks.index(task)
    ds, shift = ctx.ds(FOLD, task, "test")
    rec = read_slide(ds, shift, index)
    Z = rec.Z
    n = Z.shape[0]
    with torch.no_grad():
        mv = mean_norm(Z)
        tp = (N5.aug(mv.unsqueeze(0)) @ env["W_ar"])[0][env["col"]]
        th = int(tp.argmax())
        out = {"fold": FOLD, "task": task, "index": index, "sid": rec.sid, "label": LABEL[rec.label], "n_patch": n,
               "Z_shape": list(Z.shape), "Z_norm_q3": q3(Z.norm(dim=-1)), "tp_scores": [float(x) for x in tp], "tau_hat": cx.tasks[th],
               "mean_vec_norm": float(mv.norm())}

        def walk(p: int) -> dict:
            """用任務 p 的兩類文字走完 FINAL-B；並列 FINAL-A(42) 同一個 p 的選片與判定。"""
            ft = ctx.f_task(p)
            tf = text_nav_feats(Z, ft)
            s0 = zscore(tf[:, 0])
            idx = four_round(Z, s0, env["lam"])
            jl = idx.tolist()
            rank0 = s0.argsort(descending=True).argsort() + 1
            s0h, g = env["heads"][p].parts(Z, ft)
            sA = s0h + g
            idxA = four_round(Z, sA, env["lam"])
            ja = idxA.tolist()
            rounds = [jl[STEP * r:STEP * (r + 1)] for r in range(len(jl) // STEP)]
            roundsA = [ja[STEP * r:STEP * (r + 1)] for r in range(len(ja) // STEP)]
            v, vA = mean_norm(Z, idx), mean_norm(Z, idxA)
            s8 = (N5.aug(v.unsqueeze(0)) @ env["W_B"])[0]
            s8A = (N5.aug(vA.unsqueeze(0)) @ env["W_A"])[0]
            d, dA = float(s8[2 * p] - s8[2 * p + 1]), float(s8A[2 * p] - s8A[2 * p + 1])
            cos8 = v @ ctx.F.t()
            top = s0.sort(descending=True).values
            return {"text_task": cx.tasks[p], "max_cos_q3": q3(tf[:, 0]), "max_cos_mean_std": [float(tf[:, 0].mean()), float(tf[:, 0].std())],
                    "s0_q3": q3(s0), "s0_rank64_value": float(top[min(63, n - 1)]), "s0_max_abs_diff_vs_head_parts": float((s0 - s0h).abs().max()),
                    "n_selected": len(jl), "rounds_first3": [r[:3] for r in rounds], "selected": jl,
                    "sel_rank_in_s0": [int(rank0[idx].min()), float(rank0[idx].float().median()), int(rank0[idx].max())],
                    "n_sel_rank_gt64": int((rank0[idx] > 64).sum()),
                    "s0_of_selected_q3": q3(s0[idx]), "s0_of_selected_per_round_min": [float(s0[torch.tensor(r)].min()) for r in rounds],
                    "A_g_q3": q3(g), "A_s_q3": q3(sA), "A_selected": ja,
                    "inter_total": len(set(jl) & set(ja)),
                    "inter_B_round_in_A_all": [len(set(r) & set(ja)) for r in rounds],
                    "inter_same_round": [len(set(r) & set(ra)) for r, ra in zip(rounds, roundsA)],
                    "v_norm": float(v.to(D64).norm()), "cos_v_vA": float(v @ vA), "cos8": [float(x) for x in cos8],
                    "ridge8": [float(x) for x in s8], "d": d, "pred": LABEL[2 * p + (0 if d >= 0 else 1)],
                    "A_ridge8": [float(x) for x in s8A], "A_d": dA, "A_pred": LABEL[2 * p + (0 if dA >= 0 else 1)], "_v": v}

        cil = walk(th)
        tell = cil if th == p_true else walk(p_true)
    # 檢查：重算的 v 對快取的 v0（test；第 2 軸為四個任務的文字）
    D = T.data(r3, "test", FOLD)
    gi = sum(len(D["raw"][t]["sids"]) for t in cx.tasks[:p_true]) + index
    if D["raw"][task]["sids"][index] != rec.sid:
        raise M.CheckFailed(f"{task} #{index}: slide id 與快取不一致")
    out["check"] = {"v_max_abs_vs_cache_tau": float((cil["_v"] - D["v"][EA.KIND_B][gi, th]).abs().max()),
                    "v_max_abs_vs_cache_true": float((tell["_v"] - D["v"][EA.KIND_B][gi, p_true]).abs().max()),
                    "tp_max_abs_vs_moe5": max(abs(a - b) for a, b in zip(out["tp_scores"], env["tp_ref"][task]))}
    ref = json.loads(cx.path("FINALB", ORDER, FOLD).read_text())
    out["check"]["pred_equals_A_stage"] = bool(LABEL[ref["cil4"][gi]] == cil["pred"] and LABEL[ref["tell4"][gi]] == tell["pred"])
    for w in (cil, tell):
        w.pop("_v", None)
    out["cil"], out["tell"], out["tell_same_as_cil"] = cil, tell, th == p_true
    out["correct"] = {"FINALB_cil": cil["pred"] == out["label"], "FINALB_tell": tell["pred"] == out["label"],
                      "FINALA_cil": cil["A_pred"] == out["label"], "FINALA_tell": tell["A_pred"] == out["label"]}
    return out


def run(cx) -> dict:
    r3, r4 = cx.r3, cx.r4
    gB = json.loads((cx.base / "ext2" / "a.json").read_text())["gamma_B"]
    W_ar, pos = cx.st.b.W(FOLD, ORDER, 4, M.G_AR)
    stB = T.stats(r3, EA.KIND_B)
    A, B, cls = stB.AB(FOLD, ORDER, 4)
    W_B, _ = stB.W(FOLD, ORDER, 4, ("RDG", gB))
    W_A, _ = X.acc_of(r4, f"s{SEED}").W(FOLD, ORDER, 4, CC.G)
    m5 = json.loads((cx.base / "moe5_example" / "example.json").read_text())
    env = {"W_ar": W_ar, "col": [pos.index(p) for p in range(4)], "W_B": W_B, "W_A": W_A, "lam": r3.ctx.lam(),
           "heads": [M.load_head(X.head_path(r4, SEED, FOLD, t)) for t in cx.tasks],
           "tp_ref": {"tcga_lung": m5["tp_scores"], "tcga_brca": m5["small_cell_example"]["tp_scores"]}}
    n_train = {t: len(T.data(r3, "train", FOLD)["per"][p]["label"]) for p, t in enumerate(cx.tasks)}
    return {"stage": "example", "finished": M.now(), "gamma_B": gB, "lambda": env["lam"], "n_train": n_train,
            "shapes": {"A": list(A.shape), "B": list(B.shape), "W": list(W_B.shape), "W_ar": list(W_ar.shape)},
            "A_trace": float(torch.trace(A)), "A_last": float(A[-1, -1]), "slides": [one(cx, t, i, env) for t, i in SLIDES]}


def f6(x: float) -> str:
    return f"{x:.6f}"


def lab8(xs) -> str:
    return "、".join(f"{LABEL[k]} {xs[k]:.6f}" for k in range(8))


def md(res: dict) -> str:
    L = []
    w = L.append
    nt = res["n_train"]
    w(f"# FINAL-B 逐步實例（FINAL-B ＝ 不用 head：s = s0，四輪各 16，[v; 1] 累加式 ridge γ = {res['gamma_B']:g}；fold 1；沒有 seed）")
    w("")
    w("## 數字來源")
    w("")
    w("- 全部由 `scripts/ext2_example.py` 從特徵檔重算（只推論、不訓練、不寫既有產物），輸出 `ext2/example.json`。FINAL-B 的每一步都沒有載入 head。")
    w("- FINAL-A(42) 的 head（`i6/r2/fold1_<task>.pt`）只用在兩處：算「兩套各自選出的 64 個的交集」，以及並列 FINAL-A 的判定。")
    w("- 兩張 slide 與 `FINAL_EXAMPLE.md` 相同。A、B、W 是 fold 1、reverse 序、t = 4（四個任務都學完）的累加結果。")
    w("")
    w("檢查：")
    w("")
    w("| slide | 重算的 v 對快取的 v0（最大絕對差） | TP 分數對 `moe5_example/example.json`（最大絕對差） | 判定與 A 階段（由快取算）相同 |")
    w("|---|---|---|---|")
    for s in res["slides"]:
        c = s["check"]
        w(f"| {s['task']} test 第 {s['index']} 張 | {c['v_max_abs_vs_cache_tau']:.1e} | {c['tp_max_abs_vs_moe5']:.1e} | {'是' if c['pred_equals_A_stage'] else '否'} |")
    w("")
    w("---")
    w("")
    titles = ["例一：fold 1，tcga_lung，test 第 0 張", "例二：fold 1，tcga_brca，test 第 26 張（FINAL_EXAMPLE 的同一張小葉癌）"]
    for title, s in zip(titles, res["slides"]):
        c = s["cil"]
        p = res_tasks.index(c["text_task"])
        a, b = LABEL[2 * p], LABEL[2 * p + 1]
        w(f"## {title}")
        w("")
        w(f"slide：`{s['sid']}`，標籤 {s['label']}，patch 數 N = {s['n_patch']}。")
        w("")
        w("| 步驟 | shape | 數值 |")
        w("|---|---|---|")
        zq = s["Z_norm_q3"]
        w(f"| 讀特徵 Z | ({s['n_patch']}, 512) | ‖z‖ min {zq[0]:.2f}／median {zq[1]:.2f}／max {zq[2]:.2f} |")
        tp = s["tp_scores"]
        w(f"| TP：mean_vec ＝ 全部 patch 平均後 L2 正規化，補 1；AR 分數 [esca, rcc, brca, lung] | (4,) | "
          f"{tp[0]:.4f}、{tp[1]:.4f}、{tp[2]:.4f}、{tp[3]:.4f} → τ̂ = **{s['tau_hat']}** |")
        mq = c["max_cos_q3"]
        w(f"| 每個 patch 對 τ̂ 兩類文字（{a}、{b}）的最大 cosine | ({s['n_patch']},) | min {f6(mq[0])}／median {f6(mq[1])}／max {f6(mq[2])}；"
          f"平均 {f6(c['max_cos_mean_std'][0])}、標準差 {f6(c['max_cos_mean_std'][1])} |")
        sq = c["s0_q3"]
        w(f"| s0 ＝ 上一列在這張 slide 內 z-score；s = s0（沒有 g） | ({s['n_patch']},) | min {sq[0]:.4f}／median {sq[1]:.4f}／max {sq[2]:.4f} |")
        w(f"| s0 純排序第 64 名的分數（一次取 64 的門檻） | — | {c['s0_rank64_value']:.4f} |")
        rk = c["sel_rank_in_s0"]
        w(f"| 四輪選出的 64 個（λ = {res['lambda']:g}；每輪 16） | ({c['n_selected']},) | 選出者在 s0 純排序的名次 min {rk[0]}／median {rk[1]:.0f}／max {rk[2]}；"
          f"其中 {c['n_sel_rank_gt64']} 個名次 > 64（因扣冗餘而進入） |")
        w(f"| 各輪前三個 patch index | 4 × 16 | " + "；".join(f"輪 {i + 1}：{r}…" for i, r in enumerate(c["rounds_first3"])) + " |")
        w(f"| 各輪所選 patch 的 s0 最小值 | (4,) | " + "、".join(f"{x:.4f}" for x in c["s0_of_selected_per_round_min"]) + " |")
        w(f"| 與 FINAL-A(42) 同一張所選 64 個的交集 | — | **{c['inter_total']}／64**；FINAL-B 各輪的 16 個中落在 FINAL-A 的 64 個內："
          + "、".join(str(x) for x in c["inter_B_round_in_A_all"]) + "；同一輪對同一輪的交集：" + "、".join(str(x) for x in c["inter_same_round"]) + " |")
        w(f"| v ＝ 64 個原始 Z 等權平均後 L2 正規化 | (512,) | ‖v‖ = {c['v_norm']:.6f}；與 FINAL-A 的 v 的 cosine = {c['cos_v_vA']:.6f} |")
        w(f"| 8 類 cosine（v · Fᵀ；FINAL-B 不用它判定，只供對照） | (8,) | {lab8(c['cos8'])} |")
        w("")
        w("**ridge（FINAL-B 的判讀器）**")
        w("")
        w("| 量 | shape／數值 |")
        w("|---|---|")
        w(f"| 累加的 train slide 張數 | {sum(nt.values())}（esca {nt['tcga_esca']}、rcc {nt['tcga_rcc']}、brca {nt['tcga_brca']}、lung {nt['tcga_lung']}） |")
        w(f"| A = Σ [v; 1]ᵀ[v; 1]（四個任務累加在同一個 A；float64） | ({res['shapes']['A'][0]}, {res['shapes']['A'][1]})；A 的右下角 = {res['A_last']:.0f}（＝張數） |")
        w(f"| B（8 欄，類別 c 欄 = 該類 train slide 的 x 之和） | ({res['shapes']['B'][0]}, {res['shapes']['B'][1]}) |")
        w(f"| W = solve(A + {res['gamma_B']:g} · I, B) | ({res['shapes']['W'][0]}, {res['shapes']['W'][1]}) |")
        w(f"| 8 類 ridge 分數 [v; 1] · W（v 取 τ̂ 的文字） | {lab8(c['ridge8'])} |")
        w("")
        w("**判定**")
        w("")
        w("| 量 | 數值 | 判定 |")
        w("|---|---|---|")
        w(f"| τ̂ 兩類的分數差 d = s({a}) − s({b}) | {c['d']:+.6f} | d {'≥' if c['d'] >= 0 else '<'} 0 → {c['pred']} |")
        w(f"| FINAL-B 的 CIL 最終判定 | {c['pred']} | {'正確' if s['correct']['FINALB_cil'] else '**錯**'}（標籤 {s['label']}） |")
        t = s["tell"]
        same = "；τ̂ 與真實任務相同，與 CIL 同一個向量" if s["tell_same_as_cil"] else ""
        w(f"| FINAL-B 告訴任務的判定（向量取真實任務的文字{same}） | d = {t['d']:+.6f} → {t['pred']} | {'正確' if s['correct']['FINALB_tell'] else '**錯**'} |")
        w(f"| FINAL-A(42) 同一張：head 選的 64 個、同一種 ridge（γ = 1e-3） | d = {c['A_d']:+.6f} → {c['A_pred']} | {'正確' if s['correct']['FINALA_cil'] else '**錯**'} |")
        agree = "相同" if c["pred"] == c["A_pred"] else "不同"
        w(f"| 兩套在這一張的判定 | FINAL-B {c['pred']}；FINAL-A {c['A_pred']} | {agree} |")
        w("")
        gq, aq = c["A_g_q3"], c["A_s_q3"]
        w(f"並列（FINAL-A(42) 在同一張、同一個 τ̂ 的分數）：g min {gq[0]:.4f}／median {gq[1]:.4f}／max {gq[2]:.4f}；"
          f"s = s0 + g min {aq[0]:.4f}／median {aq[1]:.4f}／max {aq[2]:.4f}。FINAL-B 的 s0 與 head 內部算出的 s0 最大絕對差 {c['s0_max_abs_diff_vs_head_parts']:.1e}。")
        w("")
        w("---")
        w("")
    w("## 驗證與限制")
    w("")
    w("- 兩張 slide 沿用 `FINAL_EXAMPLE.md` 的選法（例一與 AUDIT A9 同一張；例二是 fold 1 的 tcga_brca test 中第一張「FINAL-A 告訴任務判錯、主系統判對」的 ILC），不是為 FINAL-B 另外挑的。")
    w("- A、W 的數值沒有存在既有產物中，由示例腳本從快取的 train 向量即時算出；train 向量本身已在 K11(a) 與特徵檔重算值比對（最大絕對差見 `REPORT_ext2.md`）。")
    w("- ridge 分數差與 cosine 差尺度不同，不可直接比較大小。")
    w("")
    w("來源檔：`scripts/ext2_example.py`；輸出 `ext2/example.json`。")
    return "\n".join(L) + "\n"


res_tasks: list[str] = []


def main() -> int:
    sys.argv = sys.argv[:1]
    cx = EA.Ctx()
    root = cx.base / "ext2"
    res_tasks.extend(cx.tasks)
    try:
        res = run(cx)
        (root / "example.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=M._default))
        (cx.base / "FINALB_EXAMPLE.md").write_text(md(res))
    except Exception:
        (root / "FAILED_example.txt").write_text(f"[{M.now()}] 階段 c 失敗\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3
    print("C：完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
