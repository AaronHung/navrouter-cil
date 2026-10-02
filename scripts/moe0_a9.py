#!/usr/bin/env python3
"""MOE-0 A9：fold 1 一張 LUNG test slide 的逐步實例（只推論；不寫任何快取）。

    NAVCIL_MACHINE=mac python scripts/moe0_a9.py [--index 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc1_pipeline as P                                                  # noqa: E402
import nc5_report as N5                                                   # noqa: E402
from nc8_report import B8, router                                         # noqa: E402
from selector.cil_ops import four_round, mean_norm, task_rows             # noqa: E402
from selector.evaluate import read_slide                                  # noqa: E402
from selector.flat_selector import text_nav_feats                         # noqa: E402
from selector.i6_expert import I6Expert                                   # noqa: E402

LABEL = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]


def q3(x):
    return f"min {x.min().item():.6f}｜median {x.median().item():.6f}｜max {x.max().item():.6f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--task", default="tcga_lung")
    a = ap.parse_args()
    ctx = P.Ctx(torch.device("cpu"))
    lam, p, fold = ctx.lam(), ctx.tasks.index(a.task), a.fold
    ds, shift = ctx.ds(fold, a.task, "test")
    rec = read_slide(ds, shift, a.index)
    Z = rec.Z
    m = I6Expert(2)
    m.load_state_dict(torch.load(ctx.out / "i6" / "r2" / f"fold{fold}_{a.task}.pt", map_location="cpu"))
    m.eval()
    with torch.no_grad():
        ft = ctx.f_task(p)
        tf = text_nav_feats(Z, ft)
        s0, g = m.parts(Z, ft)
        s = s0 + g
        j = four_round(Z, s, lam)
        v = mean_norm(Z, j)
        c8 = v @ ctx.F.t()
    n = Z.shape[0]
    print(f"slide id: {rec.sid}（fold {fold}、{a.task} test 第 {a.index} 張）；label(全域) {rec.label} = {LABEL[rec.label]}")
    print(f"Z shape {tuple(Z.shape)}；patch 數 N = {n}")
    print(f"text_nav_feats shape {tuple(tf.shape)}：max cos 欄 {q3(tf[:,0])}；entropy 欄 {q3(tf[:,1])}")
    print(f"u shape {(n, 514)}；A {tuple(m.A.shape)}")
    print(f"s0（slide 內 z-score）{q3(s0)}")
    print(f"g {q3(g)}；std(g)/std(s0) = {float(g.std() / s0.std()):.4f}")
    print(f"s = s0+g {q3(s)}")
    rs0 = s0.argsort(descending=True).argsort() + 1          # 名次（1 起）
    rs = s.argsort(descending=True).argsort() + 1
    top = s.sort(descending=True).values
    print(f"s=s0+g 的第 64 名分數（純排序，未扣冗餘）= {top[63].item():.6f}；s0 的第 64 名分數 = {s0.sort(descending=True).values[63].item():.6f}")
    jl = j.tolist()
    print(f"四輪 λ={lam} 選出 {len(jl)} 個；各輪 index 範圍：" + "；".join(f"輪{r+1}={jl[16*r:16*r+16][:3]}…" for r in range(4)))
    print(f"選出者在 s 的純排序名次：min {rs[j].min().item()}、median {rs[j].float().median().item():.0f}、max {rs[j].max().item()}")
    print(f"選出者中 s0 純排序名次 > 64 的個數 = {(rs0[j] > 64).sum().item()}；s 純排序名次 > 64（因扣冗餘而進入）的個數 = {(rs[j] > 64).sum().item()}")
    top64_s0 = set((rs0 <= 64).nonzero().flatten().tolist())
    entered = [i for i in jl if i not in top64_s0]
    print(f"不在 s0 純排序前 64、卻在最終 64 的 patch 共 {len(entered)} 個")
    for i in entered[:3]:
        rd = jl.index(i) // 16 + 1
        print(f"  patch {i}：s0={s0[i].item():.4f}、g={g[i].item():.4f}、s={s[i].item():.4f}；"
              f"加 g 前名次(s0) = {rs0[i].item()}、加 g 後名次(s0+g) = {rs[i].item()}；於四輪的第 {rd} 輪被選")
    print(f"slide 向量 v = L2 normalize(mean of 64) shape {tuple(v.shape)}、norm {v.norm().item():.6f}")
    rows = task_rows(p)
    print("cosine（8 類）：" + "｜".join(f"{LABEL[k]} {c8[k].item():.6f}" for k in range(8)))
    print(f"兩個亞型文字 cosine：{LABEL[rows[0]]} = {c8[rows[0]].item():.6f}、{LABEL[rows[1]]} = {c8[rows[1]].item():.6f}；差 = {(c8[rows[0]]-c8[rows[1]]).item():+.6f}")
    pred_wp = rows[int(c8[rows].argmax())]
    print(f"WP（告訴任務）判定 = {LABEL[pred_wp]}；正確答案 = {LABEL[rec.label]}；{'正確' if pred_wp == rec.label else '錯誤'}")
    # TP
    b = B8()
    mv = mean_norm(Z)
    cache = b.c(fold, "test", a.task)
    assert cache["sids"][a.index] == rec.sid
    seen = [0, 1, 2, 3]
    W, pos = b.W(fold, "reverse", 4, 1e-3)
    sc = N5.aug(mv.unsqueeze(0)) @ W
    print(f"TP（AR）分數 [esca,rcc,brca,lung] = {[round(x, 6) for x in sc[0].tolist()]}；分派 = {ctx.tasks[int(sc.argmax())]}")
    print(f"cache I6_cos8[{p}] 與本次重算最大絕對差 = {(cache['I6_cos8'][a.index, p] - c8).abs().max().item():.2e}；mean_vec 差 = {(cache['mean_vec'][a.index]-mv).abs().max().item():.2e}")
    th = int(sc.argmax())
    z = cache["I6_cos8"][a.index, th]
    rr = task_rows(th)
    pred = rr[int(z[rr].argmax())]
    print(f"CIL 最終判定（分派任務 {ctx.tasks[th]} 的 expert、其 2 類內 argmax）= {LABEL[pred]}；正確答案 {LABEL[rec.label]}；{'正確' if pred == rec.label else '錯誤'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
