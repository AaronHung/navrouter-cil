#!/usr/bin/env python3
"""NC-13 修正頭的貢獻（PREREG-13）：只重新推論、不訓練。

每折 test slides 讀一次；每張 slide 以四個任務的修正頭（I6 r=2，既有權重）各算 s0、g，四個 arm 各選一次 patch：
  (a) s0 + g，4 輪 × 16、λ* 去重（four_round）   (b) s0，4 輪 × 16、λ* 去重
  (c) s0 + g，一次 top-64（one_shot）           (d) s0，一次 top-64
所選 patch 等權平均、L2 正規化後對 8 類文字取 cosine（同 NC-8）。另存自家修正頭四個 arm 兩兩的選取重疊數、
(a) 與 NC-8 快取 I6_cos8 的逐張最大差；fold 1 另收集自家修正頭在全部 patch 上的 h = A u + b1（瓶頸分析）。

    NAVCIL_MACHINE=mac python scripts/nc13_head_contrib.py --cache-dir <主工作樹>/outputs/navcil/mac/cache \\
        --i6-dir <主工作樹>/outputs/navcil/mac/i6/r2 [--folds 1-10]
輸出：outputs/navcil/<machine>/nc13/fold{f}.pt（+ .done；不進版控）、bottleneck_fold1.json、timing.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.cil_ops import four_round, mean_norm, one_shot, task_rows   # noqa: E402
from selector.evaluate import read_slide, slide_dataset                   # noqa: E402
from selector.flat_selector import text_nav_feats                         # noqa: E402
from selector.i6_expert import I6Expert                                   # noqa: E402
from selector.text_encoder import build_f_txt, load_config                # noqa: E402

ARMS = ("a", "b", "c", "d")
PAIRS = (("a", "b"), ("c", "d"), ("a", "c"), ("b", "d"))
LINEAR = 3.0                                     # GELU 線性區門檻（PREREG-13 附加分析 2）


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_folds(s: str) -> list[int]:
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def select(Z, s0, g, lam):
    s = s0 + g
    return {"a": four_round(Z, s, lam), "b": four_round(Z, s0, lam), "c": one_shot(s), "d": one_shot(s0)}


@torch.no_grad()
def run_fold(cfg, tasks, F8, lam, fold, cache_dir, i6_dir, out, timing) -> None:
    done = out / f"fold{fold}.done"
    if done.exists():
        log(f"fold {fold}: done, skip")
        return
    heads = []
    for t in tasks:
        m = I6Expert(2)
        m.load_state_dict(torch.load(i6_dir / f"fold{fold}_{t}.pt", map_location="cpu"))
        heads.append(m.eval())
    rec, t_fold, h_collect = {"tasks": {}}, time.perf_counter(), {}
    for p, task in enumerate(tasks):
        ds, shift = slide_dataset(dict(cfg, fold=fold), task, p, "test")
        cache = torch.load(cache_dir / f"nc8_fold{fold}_test_{task}.pt", map_location="cpu")
        cos = {a: [] for a in ARMS}
        ov = {f"{x}{y}": [] for x, y in PAIRS}
        t_read, t_comp, labels, maxdiff, hs = [], [], [], 0.0, []
        for i in range(len(ds)):
            t0 = time.perf_counter()
            r = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            if r.sid != cache["sids"][i]:
                raise SystemExit(f"fold {fold} {task} #{i}：sid 與 NC-8 快取不一致，停止")
            Z = r.Z
            per = {a: [] for a in ARMS}
            for q, head in enumerate(heads):
                f_q = F8[task_rows(q)]
                s0, g = head.parts(Z, f_q)
                idx = select(Z, s0, g, lam)
                for a in ARMS:
                    per[a].append(mean_norm(Z, idx[a]) @ F8.t())
                if q == p:
                    sets = {a: set(idx[a].tolist()) for a in ARMS}
                    for x, y in PAIRS:
                        ov[f"{x}{y}"].append(len(sets[x] & sets[y]))
                    if fold == 1:
                        u = torch.cat([Z, text_nav_feats(Z, f_q)], dim=-1)
                        hs.append(u @ head.A.t() + head.b1)
            for a in ARMS:
                cos[a].append(torch.stack(per[a]))
            maxdiff = max(maxdiff, float((cos["a"][-1] - cache["I6_cos8"][i]).abs().max()))
            t_comp.append(time.perf_counter() - t1)
            t_read.append(t1 - t0)
            labels.append(r.label)
        if maxdiff > 1e-5:
            raise SystemExit(f"fold {fold} {task}：(a) 的 8 類 cosine 與 NC-8 快取 I6_cos8 不符（最大差 {maxdiff:.2e}），停止")
        rec["tasks"][task] = {"labels": torch.tensor(labels), "cos8": {a: torch.stack(cos[a]) for a in ARMS},
                              "overlap": {k: torch.tensor(v) for k, v in ov.items()},
                              "cos8_a_vs_nc8_maxabs": maxdiff,
                              "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp)}
        if fold == 1:
            h_collect[task] = torch.cat(hs)
        log(f"fold {fold} {task}: n={len(ds)} (a)-vs-NC8 maxabs={maxdiff:.1e}")
    torch.save(rec, out / f"fold{fold}.pt")
    if fold == 1:
        bottleneck(tasks, heads, h_collect, out)
    done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
    timing[f"fold{fold}"] = round(time.perf_counter() - t_fold, 2)
    (out / "timing.json").write_text(json.dumps(timing, indent=1))


def bottleneck(tasks, heads, h_collect, out) -> None:
    """fold 1：自家修正頭在該任務全部 test slides 的所有 patch 上，h1 與 h2 的 Pearson 相關與 GELU 線性區比例。"""
    res = {}
    for p, task in enumerate(tasks):
        h = h_collect[task].double()
        h1, h2 = h[:, 0], h[:, 1]
        c1, c2 = h1 - h1.mean(), h2 - h2.mean()
        r = float((c1 * c2).sum() / (c1.pow(2).sum().sqrt() * c2.pow(2).sum().sqrt()))
        res[task] = {"n_patches": int(h.shape[0]), "pearson_h1_h2": r,
                     "frac_h1_gt3": float((h1 > LINEAR).double().mean()), "frac_h2_gt3": float((h2 > LINEAR).double().mean()),
                     "frac_both_gt3": float(((h1 > LINEAR) & (h2 > LINEAR)).double().mean()),
                     "mean_h1": float(h1.mean()), "mean_h2": float(h2.mean()), "sd_h1": float(h1.std()), "sd_h2": float(h2.std()),
                     "w2": [float(x) for x in heads[p].w2], "b1": [float(x) for x in heads[p].b1]}
        log(f"bottleneck {task}: r={r:.4f} h1>3 {res[task]['frac_h1_gt3']:.3f} h2>3 {res[task]['frac_h2_gt3']:.3f}")
    (out / "bottleneck_fold1.json").write_text(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--i6-dir", required=True, type=Path)
    ap.add_argument("--folds", default="1-10")
    args = ap.parse_args()
    cfg = load_config()
    torch.set_num_threads(int(cfg.get("threads", 8)))
    tasks = list(cfg["tasks"])
    F8 = torch.cat([build_f_txt(t, cfg).f_txt for t in tasks])
    out = REPO_ROOT / "outputs" / "navcil" / cfg["machine"]
    lam = json.loads((out / "nc1" / "lambda.json").read_text())["lambda_star"]
    out = out / "nc13"
    out.mkdir(parents=True, exist_ok=True)
    tp = out / "timing.json"
    timing = json.loads(tp.read_text()) if tp.exists() else {}
    log(f"NC-13 threads={torch.get_num_threads()} λ*={lam} cache={args.cache_dir} i6={args.i6_dir}")
    t_run = time.perf_counter()
    for fold in parse_folds(args.folds):
        run_fold(cfg, tasks, F8, lam, fold, args.cache_dir, args.i6_dir, out, timing)
    timing[f"run/{args.folds}"] = round(time.perf_counter() - t_run, 2)
    tp.write_text(json.dumps(timing, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
