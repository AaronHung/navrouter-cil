#!/usr/bin/env python3
"""MOE-1 S7（S1–S6 都結束後才跑；PREREG-17 S7、細則 33–34；不設門檻）。

  A. 再訓練 seed 45、46，重做 S4 的「每個 seed 的主系統與 M3」兩列（只在 S4 已成功時跑；先跑 K3）。
  B. 隨機特徵：φ = [ReLU(mean_vec · R); 1]，R 為 512 × 2048 的 N(0,1)（generator seed 42、float64）；
     與 LIN8 相同的累加式 ridge（每類一欄；γ ∈ {1e-2, 1e-1, 1, 10}，validation 選、同分取小）。
     報：單獨的 WP；取代 (b) 之後的 M3（σ 固定版）；兩類分數和當 TP 的 TP 正確率與 CIL ACC；統計矩陣 bytes。
先跑 B（幾分鐘）再跑 A（訓練），所以 A 沒跑完時 B 的結果已在 s7_rf.json。

    NAVCIL_MACHINE=mac python scripts/moe1_s7.py --device cpu [--folds 1-10] [--out moe1] [--epochs 5]
輸出：outputs/navcil/<machine>/<out>/s7.json、s7.done、s7_rf.json
"""
from __future__ import annotations

import json

import torch

import moe1_common as M
import nc2_report as N2
from moe1_common import C

SEEDS = (45, 46)
RF_GAMMAS = (1e-2, 1e-1, 1.0, 10.0)
DIM = 2048


def phi(mv: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    X = torch.relu(mv.to(M.D64) @ R)
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=M.D64)], 1)


class RF:
    """一折的隨機特徵 ridge：各序各階段的統計量只含已學任務（依該序累加）。"""

    def __init__(self, run: M.Run, f: int, R: torch.Tensor):
        self.run, self.f, self.R, self._w = run, f, R, {}
        self.tr = [(phi(c["mean_vec"], R), c["labels"]) for c in (run.st.b.c(f, "train", t) for t in run.tasks)]

    def W(self, o: str, t: int, gamma: float):
        k = (o, t, gamma)
        if k not in self._w:
            pos = self.run.pos(o)[:t]
            A, cols = torch.zeros(DIM + 1, DIM + 1, dtype=M.D64), []
            for p in pos:
                X, lab = self.tr[p]
                A += X.t() @ X
                y = lab - 2 * p
                cols += [X[y == 0].sum(0), X[y == 1].sum(0)]
            self._w[k] = (torch.linalg.solve(A + gamma * torch.eye(DIM + 1, dtype=M.D64), torch.stack(cols, 1)), pos)
        return self._w[k]

    def scores(self, o: str, t: int, gamma: float, mv: torch.Tensor) -> torch.Tensor:
        """[N, 8]：欄 = 固定 8 類序；未學任務為 NaN。"""
        W, pos = self.W(o, t, gamma)
        out = torch.full((len(mv), 8), float("nan"), dtype=M.D64)
        out[:, [2 * p + c for p in pos for c in (0, 1)]] = phi(mv, self.R) @ W
        return out


def part_rf(run: M.Run) -> dict:
    st = run.st
    R = torch.randn(512, DIM, generator=torch.Generator().manual_seed(42), dtype=M.D64)
    rf = {f: RF(run, f, R) for f in run.folds}

    def alone(f, o, g, split):
        S = st.split(split, f)
        d = M.pick(M.d_cols(rf[f].scores(o, 4, g, S["mean_vec"])), S["task"])
        return C.task_mean((d >= 0) == ((S["labels"] - 2 * S["task"]) == 0), S["task"])

    val = {g: [C.eq4(alone(f, "reverse", g, "val")) for f in run.folds] for g in RF_GAMMAS}
    means = {g: C.mean(v) for g, v in val.items()}
    g_star = min(g for g, v in means.items() if v == max(means.values()))
    M.log(f"S7 隨機特徵 γ* = {g_star}（validation WP {means}）")
    run.tick(len(run.folds))
    out = {}
    for o in M.ORDER_NAMES:
        rec = {k: [] for k in ("alone", "alone_t", "M3rf", "M3", "main", "tp_rf", "tp_ar", "cil_main_rftp", "cil_M3rf_rftp")}
        for f in run.folds:
            K = M.fold_ctx(run, f, o)
            S, V, task, label = K["S"], K["V"], K["task"], K["label"]
            pos = run.pos(o)
            sc = rf[f].scores(o, 4, g_star, S["mean_vec"])
            db = M.d_cols(sc)
            re = [M.sig(M.d_cols(rf[f].scores(o, t, g_star, V["mean_vec"])), V["task"]) for t in range(1, 5)]
            sb = [re[pos.index(q)][q] for q in range(4)]                   # σ 固定值
            I6 = S["I6_cos8"]
            comps = [(M.d_heads(I6), M.sig(M.d_heads(V["I6_cos8"]), V["task"]), 1.0), (db, sb, 1.0)]
            at = alone(f, o, g_star, "test")
            rec["alone"].append(C.eq4(at)); rec["alone_t"].append(at)
            rec["M3rf"].append(M.public(M.eval_comps(comps, K)))
            rec["M3"].append(M.public(M.eval_comps([comps[0], (K["db"], K["sb"], 1.0)], K)))
            rec["main"].append(M.public(M.main_eval(I6, K["ar"], task, label)))
            th = torch.stack([sc[:, 2 * q] + sc[:, 2 * q + 1] for q in range(4)], 1).argmax(-1)
            rec["tp_rf"].append(C.eq4(C.task_mean(th == task, task))); rec["tp_ar"].append(C.eq4(C.task_mean(K["th"] == task, task)))
            rec["cil_main_rftp"].append(M.acc4(N2.hard(I6, th, task)[0], label, task)[0])
            rec["cil_M3rf_rftp"].append(M.acc4(M.fused_pred(comps, th), label, task)[0])
            run.tick()
        rec["M3rf_vs_main"] = {k: M.paired([x[k] for x in rec["M3rf"]], [x[k] for x in rec["main"]]) for k in ("wp", "mk", "cil")}
        rec["M3rf_vs_M3"] = {k: M.paired([x[k] for x in rec["M3rf"]], [x[k] for x in rec["M3"]]) for k in ("wp", "cil")}
        out[o] = rec
    d = DIM + 1
    return {"gamma": {"grid": list(RF_GAMMAS), "val_wp": {str(g): v for g, v in means.items()},
                      "val_wp_per_fold": {str(g): v for g, v in val.items()}, "star": g_star},
            "orders": out,
            "bytes": {"A_float64": d * d * 8, "B_float64": d * 8 * 8, "A_fp32": d * d * 4, "B_fp32": d * 8 * 4,
                      "R_float64": 512 * DIM * 8, "dim": d}}


def body(run: M.Run) -> dict:
    n = len(run.folds)
    run.total(n * 3 + n * 4 * len(SEEDS) + n * 2 * len(SEEDS))
    k1 = M.k1(run)
    rf_path = run.root / "s7_rf.json"
    if rf_path.exists():
        rf = json.loads(rf_path.read_text())
        run.tick(n * 3)
    else:
        rf = part_rf(run)
        rf_path.write_text(json.dumps(rf, indent=1, default=M._default, ensure_ascii=False))
    res = {"K1": k1, "rf": rf, "seeds": list(SEEDS)}
    if not (run.root / "s4.done").exists():
        res["seed_part"] = {"ran": False, "reason": "S4 未成功完成（s4.done 不存在），seed 45／46 不跑（PREREG-17 細則 9）"}
        return res
    res["K3"] = M.k3(run)
    for s in SEEDS:
        M.train_seed(run, s)
    for s in SEEDS:
        M.infer_seed(run, s)
    res["seed_part"] = {"ran": True}
    res["rows"] = M.seed_rows(run, SEEDS)
    return res


if __name__ == "__main__":
    raise SystemExit(M.stage_main("s7", body))
