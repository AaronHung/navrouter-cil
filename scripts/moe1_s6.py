#!/usr/bin/env python3
"""MOE-1 S6：gate（PREREG-17 S6、細則 27–32；seed 42；門檻 G-GATE）。σ 固定值沿用 S2 的算法（moe1_common）。

gate 的訓練資料只用當前任務 j 的 train slides：
  d_a  分層切 3 份（random_state = 42）；每次用 2 份訓練一個 head（nc7_i6 相同設定、seed 42），對剩下 1 份四輪推論
  d_b  學任務 j 那個階段的 LIN8 的 closed-form leave-one-out：h = xᵀ(A + γI)⁻¹x，ŷ_loo = (ŷ − h·y)/(1 − h)
  z_a = d_a/σ_a、z_b = d_b/σ_b（σ = 固定值）
G1：每任務一個 β ∈ {0, 0.25, 0.5, 1, 2, 4}。G2：ρ = sigmoid(c0 + c1|z_a| + c2|z_b|)，f = (1 − ρ)z_a + ρ·z_b，L-BFGS。

    NAVCIL_MACHINE=mac python scripts/moe1_s6.py --device cpu [--folds 1-10] [--out moe1] [--epochs 5]
輸出：outputs/navcil/<machine>/<out>/s6.json、s6.done；gate/xfit/（3 份的 head）、gate/da_fold{f}_{task}.pt
"""
from __future__ import annotations

import json
import time

import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

import moe1_common as M
import nc5_report as N5
from moe1_common import C
from selector.cil_ops import NonFiniteError, four_round, mean_norm
from selector.evaluate import read_slide

BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
L2 = 0.01


def xfit_da(run: M.Run, fold: int, task: str) -> dict:
    """該折該任務 train slides 的 d_a：每張由「沒看過它的 head」算（3 份交叉）。"""
    root = run.root / "gate"
    path, done = root / f"da_fold{fold}_{task}.pt", root / f"da_fold{fold}_{task}.done"
    if done.exists():
        run.tick(3)
        return torch.load(path, map_location="cpu")
    (root / "xfit").mkdir(parents=True, exist_ok=True)
    ctx, lam, p = run.ctx, run.ctx.lam(), run.tasks.index(task)
    c = run.st.b.c(fold, "train", task)
    y = (c["labels"] - 2 * p).tolist()
    n = len(y)
    d_a, part = torch.full((n,), float("nan")), torch.full((n,), -1, dtype=torch.long)
    t_read, t_comp = torch.zeros(n), torch.zeros(n)
    ds, shift = ctx.ds(fold, task, "train")
    if [str(s) for s in ds.sids] != c["sids"]:
        raise M.CheckFailed(f"fold {fold} {task}: train split 的 slide 順序與 NC-8 快取不一致")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for k, (tr, ho) in enumerate(skf.split([[0]] * n, y)):
        w = root / "xfit" / f"fold{fold}_{task}_k{k}.pt"
        if not w.with_suffix(".done").exists():
            t0 = time.perf_counter()
            model, hist = M.train_head(run, fold, task, M.SEED0, run.epochs, idx=sorted(tr.tolist()))
            torch.save(model.state_dict(), w)
            w.with_name(w.stem + "_train.json").write_text(json.dumps({"wall_s": time.perf_counter() - t0, "epochs": hist}))
            w.with_suffix(".done").write_text(M.now() + "\n")
        head = M.load_head(w)
        with torch.no_grad():
            for i in ho.tolist():
                t0 = time.perf_counter()
                rec = read_slide(ds, shift, i)
                t1 = time.perf_counter()
                c8 = mean_norm(rec.Z, four_round(rec.Z, head(rec.Z, ctx.f_task(p)), lam)) @ ctx.F.t()
                d_a[i], part[i] = c8[2 * p] - c8[2 * p + 1], k
                t_comp[i], t_read[i] = time.perf_counter() - t1, t1 - t0
        run.tick()
    if not bool(torch.isfinite(d_a).all()) or int((part < 0).sum()):
        raise M.CheckFailed(f"fold {fold} {task}: 有 train slide 沒有被保留份涵蓋")
    out = {"sids": c["sids"], "labels": c["labels"], "d_a": d_a, "part": part, "t_read_s": t_read, "t_compute_s": t_comp}
    torch.save(out, path)
    done.write_text(M.now() + "\n")
    return out


def loo_db(run: M.Run, f: int, o: str, q: int):
    """任務 q 的 train slides 在「學 q 那個階段」的 LIN8（γ = 0.01）上的 leave-one-out d_b。回傳 (d_b, 與 B8.W 的最大差)。"""
    pos = run.pos(o)
    t = pos.index(q) + 1
    A, cols = torch.zeros(513, 513, dtype=M.D64), []
    for p in pos[:t]:
        c = run.st.b.c(f, "train", run.tasks[p])
        X = N5.aug(c["mean_vec"])
        A += X.t() @ X
        y = c["labels"] - 2 * p
        cols += [X[y == 0].sum(0), X[y == 1].sum(0)]
    G = A + M.G_LIN8 * torch.eye(513, dtype=M.D64)
    W = torch.linalg.solve(G, torch.stack(cols, 1))
    Wref, _ = run.st.b.W(f, o, t, M.G_LIN8, lin8=True)
    dev = float((W - Wref).abs().max())
    c = run.st.b.c(f, "train", run.tasks[q])
    X = N5.aug(c["mean_vec"])
    h = (X * torch.linalg.solve(G, X.t()).t()).sum(1)                      # [n]
    j = pos.index(q)
    yhat = X @ Wref[:, [2 * j, 2 * j + 1]]                                  # [n, 2]
    Y = F.one_hot(c["labels"] - 2 * q, 2).to(M.D64)
    loo = (yhat - h[:, None] * Y) / (1 - h[:, None])
    return loo[:, 0] - loo[:, 1], dev, float(h.max())


def fit_g1(za, zb, da, first):
    best = None
    accs = {}
    for b in BETAS:
        f = za + b * zb
        acc = (((f > 0) | ((f == 0) & (da >= 0))) == first).double().mean().item()
        accs[b] = acc
        key = (-acc, abs(b - 1.0), b)                       # 正確率最高；同分取最接近 1；仍同分取較小
        if best is None or key < best[0]:
            best = (key, b)
    return best[1], accs


def g2_f(c, za, zb):
    rho = torch.sigmoid(c[0] + c[1] * za.abs() + c[2] * zb.abs())
    return (1 - rho) * za + rho * zb, rho


def fit_g2(za, zb, first, where: str):
    y = torch.where(first, 1.0, -1.0).to(M.D64)
    c = torch.zeros(3, dtype=M.D64, requires_grad=True)
    opt = torch.optim.LBFGS([c], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
    n_eval = [0]

    def closure():
        opt.zero_grad()
        f, _ = g2_f(c, za, zb)
        loss = F.softplus(-y * f).mean() + L2 * (c[1] ** 2 + c[2] ** 2)
        loss.backward()
        n_eval[0] += 1
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(c.grad).all()):
            raise NonFiniteError(f"G2 {where}: loss／梯度非有限（第 {n_eval[0]} 次評估）")
        return loss

    opt.step(closure)
    if not bool(torch.isfinite(c).all()):
        raise NonFiniteError(f"G2 {where}: 參數非有限")
    with torch.no_grad():
        f, _ = g2_f(c, za, zb)
        loss = float(F.softplus(-y * f).mean() + L2 * (c[1] ** 2 + c[2] ** 2))
    return c.detach(), loss, n_eval[0]


def g_first(f, da):
    return (f > 0) | ((f == 0) & (da >= 0))


def body(run: M.Run) -> dict:
    st = run.st
    k1 = M.k1(run)
    k3 = M.k3(run)
    n = len(run.folds)
    run.total(n * 4 * 3)
    DA = {(f, q): xfit_da(run, f, t) for f in run.folds for q, t in enumerate(run.tasks)}
    out = {}
    for o in M.ORDER_NAMES:
        sysm = {k: [] for k in ("main", "M3", "G1", "G2")}
        train_acc = {"G1": [], "G2": [], "M3": [], "main": []}
        betas, cs, g1_grid, rho_all = [], [], [], [[] for _ in range(4)]
        fitinfo, w_dev, h_max = [], 0.0, 0.0
        union_tot = union_n = 0
        union_fold = []
        for f in run.folds:
            K = M.fold_ctx(run, f, o)
            S, V, task, label, th = K["S"], K["V"], K["task"], K["label"], K["th"]
            I6 = S["I6_cos8"]
            da_all, sa, db_all, sb = M.d_heads(I6), M.sig(M.d_heads(V["I6_cos8"]), V["task"]), K["db"], K["sb"]
            beta, cc, tacc = [], [], {k: [] for k in train_acc}
            for q in range(4):
                g = DA[(f, q)]
                first = (g["labels"] - 2 * q) == 0
                d_b, dev, hm = loo_db(run, f, o, q)
                w_dev, h_max = max(w_dev, dev), max(h_max, hm)
                pq = torch.full((len(first),), q)
                za, zb = M.zs(g["d_a"], pq, sa).to(M.D64), M.zs(d_b, pq, sb).to(M.D64)
                b, accs = fit_g1(za, zb, g["d_a"], first)
                c, loss, nev = fit_g2(za, zb, first, f"fold {f} {run.tasks[q]} {o}")
                with torch.no_grad():
                    a2 = (g_first(g2_f(c, za, zb)[0], g["d_a"]) == first).double().mean().item()
                beta.append(b); cc.append(c.tolist()); g1_grid.append(accs)
                tacc["G1"].append(accs[b]); tacc["G2"].append(a2); tacc["M3"].append(accs[1.0]); tacc["main"].append(accs[0.0])
                fitinfo.append({"fold": f, "task": q, "n": len(first), "beta": b, "c": c.tolist(), "g2_loss": loss, "g2_evals": nev,
                                "g1_grid_acc": {str(k): v for k, v in accs.items()}, "g2_train_acc": a2})
            betas.append(beta); cs.append(cc)
            for k in train_acc:
                train_acc[k].append(C.eq4(tacc[k]))
            bt, ct = torch.tensor(beta, dtype=M.D64), torch.tensor(cc, dtype=M.D64)

            def gate_pred(kind, p):
                da = M.pick(da_all, p)
                za, zb = M.zs(da, p, sa).to(M.D64), M.zs(M.pick(db_all, p), p, sb).to(M.D64)
                if kind == "G1":
                    fz, rho = za + bt[p] * zb, None
                else:
                    rho = torch.sigmoid(ct[p, 0] + ct[p, 1] * za.abs() + ct[p, 2] * zb.abs())
                    fz = (1 - rho) * za + rho * zb
                return 2 * p + (~g_first(fz, da)).long(), rho

            sysm["main"].append(M.public(M.main_eval(I6, K["ar"], task, label)))
            m3 = M.eval_comps([(da_all, sa, 1.0), (db_all, sb, 1.0)], K)
            sysm["M3"].append(M.public(m3))
            for kind in ("G1", "G2"):
                tell, rho = gate_pred(kind, task)
                cil, _ = gate_pred(kind, th)
                w, w_t = M.acc4(tell, label, task)
                cv, c_t = M.acc4(cil, label, task)
                sysm[kind].append({"wp": w, "wp_t": w_t, "mk": w, "cil": cv, "cil_t": c_t})
                if kind == "G2":
                    for q in range(4):
                        rho_all[q].append(rho[task == q])
            # 參考上限：LT 與 GR 至少一個對（告訴任務；= S5 的 LT＋GR）
            first = (label - 2 * task) == 0
            u = ((M.pick(da_all, task) >= 0) == first) | ((M.pick(db_all, task) >= 0) == first)
            union_tot += int(u.sum()); union_n += len(u); union_fold.append(C.eq4(C.task_mean(u, task)))
        ta = {k: C.mean(v) for k, v in train_acc.items()}
        rep = "G2" if ta["G2"] > ta["G1"] else "G1"
        cmpm = {k: {m: M.paired([x[m] for x in sysm[k]], [x[m] for x in sysm["M3"]]) for m in ("wp", "mk", "cil")} for k in ("G1", "G2")}
        cmpm["M3_vs_main"] = {m: M.paired([x[m] for x in sysm["M3"]], [x[m] for x in sysm["main"]]) for m in ("wp", "mk", "cil")}
        gate = {"representative": rep, "train_acc": ta, "mean": cmpm[rep]["cil"]["mean"], "wins": cmpm[rep]["cil"]["wins"],
                "pass": bool(cmpm[rep]["cil"]["mean"] >= 0.005 and cmpm[rep]["cil"]["wins"] >= 7)}
        rho_q = []
        for q in range(4):
            r = torch.cat(rho_all[q])
            rho_q.append({"q25": float(r.quantile(0.25)), "median": float(r.quantile(0.5)), "q75": float(r.quantile(0.75)), "n": len(r)})
        out[o] = {"systems": sysm, "train_acc_per_fold": train_acc, "train_acc": ta, "vs_M3": cmpm, "g_gate": gate,
                  "beta": betas, "beta_counts": [{str(b): sum(bb[q] == b for bb in betas) for b in BETAS} for q in range(4)],
                  "c": cs, "c_mean": [[C.mean(cc[q][i] for cc in cs) for i in range(3)] for q in range(4)],
                  "rho_test": rho_q, "fit": fitinfo, "lin8_W_max_dev_vs_B8": w_dev, "loo_h_max": h_max,
                  "upper_LT_GR_union": {"total": union_tot / union_n, "n": union_n, "fold_task_mean": C.mean(union_fold)}}
    g_gate = {"orders": {o: out[o]["g_gate"] for o in M.ORDER_NAMES},
              "pass": bool(all(out[o]["g_gate"]["pass"] for o in M.ORDER_NAMES)), "valid": run.full}
    t_read = sum(float(v["t_read_s"].sum()) for v in DA.values())
    t_comp = sum(float(v["t_compute_s"].sum()) for v in DA.values())
    return {"K1": k1, "K3": k3, "betas": list(BETAS), "orders": out, "G-GATE": g_gate,
            "xfit_infer_t_read_s": t_read, "xfit_infer_t_compute_s": t_comp, "n_gate_train_slides": sum(len(v["d_a"]) for v in DA.values())}


if __name__ == "__main__":
    raise SystemExit(M.stage_main("s6", body))
