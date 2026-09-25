#!/usr/bin/env python3
"""NC-4 報告：A（I7 容量依需求）、B（router I5／I4 與上限參考）、C（router 總表）。

只讀快取、lora_v2 評估檔與既有 metrics，不碰特徵檔。
    NAVCIL_MACHINE=mac python scripts/nc4_report.py
輸出：outputs/navcil/<machine>/REPORT_stage6.md、nc4/metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc2_report as N2                                                   # noqa: E402
from selector.cil_eval import mean_sd                                     # noqa: E402
from selector.cil_ops import ORDERS, task_rows                            # noqa: E402

FOLDS = N2.FOLDS
R_LIST = (1, 2, 4, 8)
CANDS = ("0z", "0b", "r1", "r2", "r4", "r8")
fmt = N2.fmt


def params_of(c: str) -> int:
    return 0 if c in ("0z", "0b") else N2.lora_params(int(c[1:]))


def masked_acc(c8: torch.Tensor, labels: torch.Tensor, p: int) -> float:
    rr = torch.tensor(task_rows(p))
    return (rr[c8[:, rr].argmax(-1)] == labels).float().mean().item()


# ── A：I7 ──────────────────────────────────────────────────────────────────
class I7:
    def __init__(self, d):
        self.d = d
        self._c, self._ev = {}, {}

    def c(self, fold, split, task):
        k = (fold, split, task)
        if k not in self._c:
            self._c[k] = torch.load(self.d.cache / f"nc4_fold{fold}_{split}_{task}.pt", map_location="cpu")
        return self._c[k]

    def ev(self, r, order, fold):
        k = (r, order, fold)
        if k not in self._ev:
            self._ev[k] = torch.load(self.d.out / "lora_v2" / f"r{r}" / order / f"fold{fold}_eval.pt",
                                     map_location="cpu")
        return self._ev[k]

    def choose(self, order, fold, p) -> tuple[str, dict]:
        v = self.c(fold, "val", self.d.tasks[p])
        wp = {c: masked_acc(v[f"{c}|{order}"] if c != "0z" else v["0z"], v["labels"], p) for c in CANDS}
        ok = [c for c in CANDS if wp[c] >= wp["r8"] - 0.01]
        zero = [c for c in ("0z", "0b") if c in ok]
        if zero:
            return max(zero, key=lambda c: (wp[c], c == "0z")), wp
        return min(ok, key=params_of), wp

    def test_c8(self, order, fold, q, cand, slide_task) -> torch.Tensor:
        """任務 slide_task 的 test slides 上，任務 q 的 expert（候選 cand）的 8 類 cosine。"""
        t = self.d.tasks[slide_task]
        if cand == "L0":
            return self.d.c(fold, "test", t)["four_cos8_uni"][:, q]
        if cand == "0z":
            return self.c(fold, "test", t)["0z"][:, q]
        if cand == "0b":
            return self.c(fold, "test", t)[f"0b|{order}"][:, q]
        return self.ev(int(cand[1:]), order, fold)["tasks"][t]["l1_cos8"][:, q]


def run_A(d, Lv2):
    A = I7(d)
    res = {}
    for o, names in ORDERS.items():
        first = d.tasks.index(names[0])
        rows = {"choice": {}, "val_wp": {}, "wp": [], "wp_task": {p: [] for p in range(4)}, "params": []}
        for f in FOLDS:
            choice = {first: "L0"}
            for t in names[1:]:
                p = d.tasks.index(t)
                c, wp = A.choose(o, f, p)
                choice[p] = c
                rows["val_wp"][f"{f}|{t}"] = wp
            rows["choice"][f] = choice
            accs = []
            for p in range(4):
                a = masked_acc(A.test_c8(o, f, p, choice[p], p), d.c(f, "test", d.tasks[p])["labels"], p)
                accs.append(a); rows["wp_task"][p].append(a)
            rows["wp"].append(sum(accs) / 4)
            rows["params"].append(sum(params_of(choice[p]) for p in range(4) if p != first))

            def stage(seen, f=f, choice=choice, o=o):
                g = N2.gather_r(d, f, "test", seen)
                th = torch.tensor(seen)[N2.scores_for(d, f, g, seen, "R3", 8).argmax(-1)]
                four = torch.cat([torch.stack([A.test_c8(o, f, q, choice[q], sp) for q in range(4)], 1)
                                  for sp in seen])
                pred, mk = N2.hard(four, th, g["task"])
                return pred, mk, g
            rows.setdefault("cil", []).append(N2.cil_from(stage, o, d.tasks))
        r2 = Lv2[(2, o)]
        rows["r2_wp"] = r2["wp"]
        rows["pass_wp"] = mean_sd(rows["wp"])[0] >= mean_sd(r2["wp"])[0] - 0.005
        rows["pass_params"] = mean_sd(rows["params"])[0] <= 0.5 * 3 * N2.lora_params(2)
        res[o] = rows
    passed = all(res[o]["pass_wp"] and res[o]["pass_params"] for o in ORDERS)
    return res, passed, A


# ── B：router ───────────────────────────────────────────────────────────────
def kmeans_keys(X: torch.Tensor, k: int) -> torch.Tensor:
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X.numpy())
    return F.normalize(torch.from_numpy(km.cluster_centers_).float(), dim=-1)


def train_linear(X: torch.Tensor, y: torch.Tensor, n_cls: int, lin: nn.Linear | None = None) -> nn.Linear:
    """全批次 CE、Adam(lr 1e-3、wd 1e-4)、200 步。lin 未給時在此建立（呼叫端先設好 seed）。"""
    lin = lin if lin is not None else nn.Linear(X.shape[1], n_cls)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    with torch.enable_grad():                      # router 分數在 no_grad 中呼叫，訓練需開梯度
        for _ in range(200):
            loss = F.cross_entropy(lin(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return lin.eval()


class Routers:
    def __init__(self, d):
        self.d = d
        self.base, self.b1, self.b2, self.up = {}, {}, {}, {}

    def train_mv(self, f, p):
        return self.d.c(f, "train", self.d.tasks[p])["mean_vec"]

    def base_keys(self, f):
        if f not in self.base:
            self.base[f] = [kmeans_keys(self.train_mv(f, p), 4) for p in range(4)]
        return self.base[f]

    def B1(self, f, o):
        """{task_pos: keys [4 + k_e, D]}，依序建立。"""
        k = (f, o)
        if k not in self.b1:
            base = self.base_keys(f)
            pos = [self.d.tasks.index(t) for t in ORDERS[o]]
            keys, extra = {}, {}
            for i, p in enumerate(pos):
                seen = pos[:i + 1]
                K = torch.cat([base[q] for q in seen])
                owner = torch.tensor([q for q in seen for _ in range(4)])
                X = self.train_mv(f, p)
                assign = owner[(X @ K.t()).argmax(-1)]
                wrong = X[assign != p]
                ke = min(4, wrong.shape[0] // 5) if wrong.shape[0] >= 5 else 0
                extra[p] = kmeans_keys(wrong, ke) if ke > 0 else torch.zeros(0, X.shape[1])
                keys[p] = torch.cat([base[p], extra[p]])
            self.b1[k] = keys
        return self.b1[k]

    def B2(self, f, o, t):
        """第 t 階段（1-based）的線性 router；類別依該序已見任務。"""
        k = (f, o, t)
        if k not in self.b2:
            base = self.base_keys(f)
            pos = [self.d.tasks.index(x) for x in ORDERS[o]][:t]
            cur = pos[-1]
            Xc = self.train_mv(f, cur)
            n = Xc.shape[0]
            torch.manual_seed(0)
            xs, ys = [], []
            for j, q in enumerate(pos[:-1]):
                var = self.train_mv(f, q).var(0, unbiased=False)
                idx = torch.randint(0, 4, (n,))
                fake = F.normalize(base[q][idx] + torch.randn(n, Xc.shape[1]) * var.sqrt(), dim=-1)
                xs.append(fake); ys.append(torch.full((n,), j))
            lin = nn.Linear(Xc.shape[1], t)
            xs.append(Xc); ys.append(torch.full((n,), t - 1))
            if t > 1:
                lin = train_linear(torch.cat(xs), torch.cat(ys), t, lin)
            self.b2[k] = (lin.eval(), pos)
        return self.b2[k]

    def upper(self, f):
        if f not in self.up:
            torch.manual_seed(0)
            X = torch.cat([self.train_mv(f, p) for p in range(4)])
            y = torch.cat([torch.full((self.train_mv(f, p).shape[0],), p) for p in range(4)])
            self.up[f] = train_linear(X, y, 4)
        return self.up[f]

    @torch.no_grad()
    def scores(self, name, f, o, g, seen) -> torch.Tensor:
        mv = g["mean_vec"]
        if name == "基準":
            return N2.scores_for(self.d, f, g, seen, "R3", 8)
        if name == "B1":
            keys = self.B1(f, o)
            return torch.stack([(mv @ keys[p].t()).amax(-1) for p in seen], -1)
        if name == "B2":
            lin, pos = self.B2(f, o, len(seen))
            assert sorted(pos) == sorted(seen)
            logits = lin(mv)                                          # 欄位依 pos 順序
            return torch.stack([logits[:, pos.index(p)] for p in seen], -1)
        if name == "上限參考":
            return self.upper(f)(mv)[:, seen]
        raise ValueError(name)


def run_B(d, RT):
    names = ("基準", "B1", "B2", "上限參考")
    seen4 = list(range(4))
    val = {n: {o: [] for o in ORDERS} for n in names}
    for f in FOLDS:
        g = N2.gather_r(d, f, "val", seen4)
        for n in names:
            for o in ORDERS:
                th = torch.tensor(seen4)[RT.scores(n, f, o, g, seen4).argmax(-1)]
                pred, _ = N2.hard(g["four"], th, g["task"])
                ok = pred == g["labels"]
                val[n][o].append(sum(ok[g["task"] == p].float().mean().item() for p in seen4) / 4)
    val_mean = {n: mean_sd(val[n]["reverse"] + val[n]["paper"])[0] for n in names}
    selected = max(names[:3], key=lambda n: (round(val_mean[n], 12), -names.index(n)))
    test = {}
    ie, il = d.tasks.index("tcga_esca"), d.tasks.index("tcga_lung")
    for n in names:
        res = {}
        for o in ORDERS:
            cil = []
            for f in FOLDS:
                def stage(seen, f=f, o=o):
                    g = N2.gather_r(d, f, "test", seen)
                    th = torch.tensor(seen)[RT.scores(n, f, o, g, seen).argmax(-1)]
                    pred, mk = N2.hard(g["four"], th, g["task"])
                    return pred, mk, g
                cil.append(N2.cil_from(stage, o, d.tasks))
            tp_task, mac, mic, e2l, l2e = {p: [] for p in seen4}, [], [], 0, 0
            for f in FOLDS:
                g = N2.gather_r(d, f, "test", seen4)
                th = torch.tensor(seen4)[RT.scores(n, f, o, g, seen4).argmax(-1)]
                ok = th == g["task"]
                accs = [ok[g["task"] == p].float().mean().item() for p in seen4]
                for p in seen4:
                    tp_task[p].append(accs[p])
                mac.append(sum(accs) / 4); mic.append(ok.float().mean().item())
                e2l += int(((g["task"] == ie) & (th == il)).sum()); l2e += int(((g["task"] == il) & (th == ie)).sum())
            res[o] = {"cil": cil, "acc_t4": [x["acc"] for x in cil], "tp_task": tp_task, "tp_macro": mac,
                      "tp_micro": mic, "esca_to_lung": e2l, "lung_to_esca": l2e}
        test[n] = res
    for n in names:
        for o in ORDERS:
            diffs = [a - b for a, b in zip(test[n][o]["acc_t4"], test["基準"][o]["acc_t4"])]
            test[n][o]["diff"] = {"per_fold": diffs, "mean": mean_sd(diffs)[0], "wins": sum(x > 0 for x in diffs)}
    b1_bytes = [(RT.B1(f, o)[p].shape[0]) * 2048 for f in FOLDS for o in ORDERS for p in range(4)]
    bytes_ = {"基準": [16384], "B1": b1_bytes, "B2": [8192 + 2048 + 513 * 4], "上限參考": [513 * 4]}
    passed = (selected != "基準" and max(bytes_[selected]) <= 16384 and
              all(test[selected][o]["diff"]["mean"] >= 0.01 and test[selected][o]["diff"]["wins"] >= 7 for o in ORDERS))
    return {"val": val, "val_mean": val_mean, "selected": selected, "test": test, "bytes": bytes_,
            "pass": passed}


def main() -> int:
    t0 = time.perf_counter()
    d = N2.D2()
    M1 = json.loads((d.out / "nc1" / "metrics.json").read_text())
    M2 = json.loads((d.out / "nc2" / "metrics.json").read_text())
    M3 = json.loads((d.out / "nc3" / "metrics.json").read_text())
    L0m = mean_sd(M1["stage1"]["fold_mean"]["e_fourround"])[0]
    Lv2, gv2 = N2.compute_L(d, L0m, "lora_v2")
    A, a_pass, I = run_A(d, Lv2)
    RT = Routers(d)
    B = run_B(d, RT)
    T0b = N2.t0b(d)

    # 新的完整系統列（PREREG-4 操作定義 13）
    full = {}
    if a_pass or B["pass"]:
        rname = B["selected"] if B["pass"] else "基準"
        for o in ORDERS:
            first = d.tasks.index(ORDERS[o][0])
            cil = []
            for i, f in enumerate(FOLDS):
                choice = A[o]["choice"][f] if a_pass else {p: ("L0" if p == first else "r2") for p in range(4)}

                def stage(seen, f=f, o=o, choice=choice):
                    g = N2.gather_r(d, f, "test", seen)
                    th = torch.tensor(seen)[RT.scores(rname, f, o, g, seen).argmax(-1)]
                    four = torch.cat([torch.stack([I.test_c8(o, f, q, choice[q], sp) for q in range(4)], 1)
                                      for sp in seen])
                    pred, mk = N2.hard(four, th, g["task"])
                    return pred, mk, g
                cil.append(N2.cil_from(stage, o, d.tasks))
            full[o] = {"router": rname, "experts": "I7" if a_pass else "L1(r=2) v2", "cil": cil}

    M = {"A": {o: {k: v for k, v in A[o].items() if k != "cil"} | {"cil": [{k: x[k] for k in ("acc", "masked", "forgetting")}
                                                                           for x in A[o]["cil"]]} for o in ORDERS},
         "A_pass": a_pass,
         "B": {"val": B["val"], "val_mean": B["val_mean"], "selected": B["selected"], "pass": B["pass"],
               "bytes": {k: max(v) for k, v in B["bytes"].items()},
               "test": {n: {o: {k: v for k, v in r.items() if k != "cil"} for o, r in x.items()} for n, x in B["test"].items()}},
         "full": {o: {"acc": [x["acc"] for x in v["cil"]], "masked": [x["masked"] for x in v["cil"]],
                      "forgetting": [x["forgetting"] for x in v["cil"]], "router": v["router"], "experts": v["experts"]}
                  for o, v in full.items()}}
    tp = d.out / "nc4" / "timing_i7.json"
    M["timing"] = json.loads(tp.read_text()) if tp.exists() else {}
    (d.out / "nc4").mkdir(exist_ok=True)
    (d.out / "nc4" / "metrics.json").write_text(json.dumps(M, indent=1, default=str))
    write(d, M, A, a_pass, Lv2, B, full, T0b, M1, M2, M3, time.perf_counter() - t0)
    print(f"→ {d.out / 'REPORT_stage6.md'}")
    return 0


def write(d, M, A, a_pass, Lv2, B, full, T0b, M1, M2, M3, t_rep):
    ok = lambda b: "**通過**" if b else "**未通過**"                     # noqa: E731
    lim = 0.5 * 3 * N2.lora_params(2)
    sel = B["selected"]
    out = ["# REPORT — NC-4：容量依需求（I7）、router I5／I4、router 總表（Mac CPU，十折兩序）", "",
           "機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；所有數字來自同一台、"
           "同一批。判準見 `PREREG-4.md`（commit 88532d6）。", "",
           "## PREREG-4 判準落點", "", "| 判準 | 數值 | 所在表格 | 結果 |", "|---|---|---|---|"]
    for o in ORDERS:
        a = A[o]
        out.append(f"| I7-pass（{o}）：test WP ≥ 固定 r=2 − 0.005 | {mean_sd(a['wp'])[0]:.4f} vs "
                   f"{mean_sd(a['r2_wp'])[0]:.4f} − 0.005 = {mean_sd(a['r2_wp'])[0] - 0.005:.4f} | T-A1 | {ok(a['pass_wp'])} |")
        out.append(f"| I7-pass（{o}）：非底座參數合計 ≤ 固定 r=2 的 50% | {mean_sd(a['params'])[0]:,.1f} vs {lim:,.1f} | "
                   f"T-A1 | {ok(a['pass_params'])} |")
    out += [f"| **I7-pass 整體**（兩序皆成立） | | | {ok(a_pass)} |",
            f"| B 選法（validation 平均 CIL） | 選出 **{sel}**（基準 {B['val_mean']['基準']:.4f}、B1 {B['val_mean']['B1']:.4f}、"
            f"B2 {B['val_mean']['B2']:.4f}） | T-B1 | — |"]
    if sel != "基準":
        for o in ORDERS:
            dd = B["test"][sel][o]["diff"]
            out.append(f"| R-機制-2（{o}）：test 差 ≥ +0.01 且贏 ≥ 7/10 | {dd['mean']:+.4f}、{dd['wins']}/10 | T-B2 | "
                       f"{ok(dd['mean'] >= 0.01 and dd['wins'] >= 7)} |")
    out += [f"| **R-機制-2 整體**（選出者 ≠ 基準、兩序皆成立、≤ 16 KB） | 選出 {sel}；最大儲存 "
            f"{max(B['bytes'][sel]):,} bytes | T-B2 | {ok(B['pass'])} |", ""]

    # A
    out += ["## T-A I7 容量依需求（expert = L 線 v2 權重，不重新訓練）", "",
            "### T-A1 WP 與參數", "",
            "| 序 | I7 test WP | 固定 r=2 test WP | 非底座參數合計（十折平均） | 固定 r=2 | "
            + " | ".join(d.tasks) + " |", "|---|---|---|---|---|" + "---|" * 4]
    for o in ORDERS:
        a = A[o]
        out.append(f"| {o} | {fmt(a['wp'])} | {fmt(a['r2_wp'])} | {mean_sd(a['params'])[0]:,.1f} | {3 * N2.lora_params(2):,} | "
                   + " | ".join(fmt(a["wp_task"][p]) for p in range(4)) + " |")
    out += ["", "### T-A2 每任務選到的容量（十折次數）", "", "| 序 | 任務 | " + " | ".join(CANDS) + " |",
            "|---|---|" + "---|" * len(CANDS)]
    for o, names in ORDERS.items():
        for t in names[1:]:
            p = d.tasks.index(t)
            cnt = {c: sum(1 for f in FOLDS if A[o]["choice"][f][p] == c) for c in CANDS}
            out.append(f"| {o} | {t} | " + " | ".join(str(cnt[c]) for c in CANDS) + " |")
    out += ["", f"底座（reverse：ESCA；paper：LUNG）維持完整 expert（{N2.FULL_PARAMS:,} 參數）。", "",
            "### T-A3 validation WP（十折平均；選法依據）", "", "| 序 | 任務 | " + " | ".join(CANDS) + " |",
            "|---|---|" + "---|" * len(CANDS)]
    for o, names in ORDERS.items():
        for t in names[1:]:
            vals = {c: [A[o]["val_wp"][f"{f}|{t}"][c] for f in FOLDS] for c in CANDS}
            out.append(f"| {o} | {t} | " + " | ".join(f"{mean_sd(vals[c])[0]:.4f}" for c in CANDS) + " |")
    out += ["", "### T-A4 I7 ＋ R3(k=8)（Hard）", "", "| 序 | ACC | Masked ACC | Forgetting |", "|---|---|---|---|"]
    for o in ORDERS:
        c = A[o]["cil"]
        out.append(f"| {o} | {fmt([x['acc'] for x in c])} | {fmt([x['masked'] for x in c])} | {fmt([x['forgetting'] for x in c])} |")

    # B
    out += ["", "## T-B router（L0 expert、Hard）", "", "### T-B1 validation 選法（十折 validation、t = 4 CIL ACC）", "",
            "| router | reverse | paper | 兩序平均 |", "|---|---|---|---|"]
    for n in ("基準", "B1", "B2", "上限參考"):
        v = B["val"][n]
        out.append(f"| {n}{'（非 CL，不參與選法）' if n == '上限參考' else ''} | {fmt(v['reverse'])} | {fmt(v['paper'])} | "
                   f"{B['val_mean'][n]:.4f}{' ★' if n == sel else ''} |")
    out += ["", "### T-B2 test", "",
            "| router | 序 | 儲存 bytes／任務（最大） | TP macro | TP micro | ESCA→Lung | Lung→ESCA | ACC | Masked ACC | Forgetting | 對基準差 | 贏折數 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("基準", "B1", "B2", "上限參考"):
        for o in ORDERS:
            r = B["test"][n][o]
            out.append(f"| {n} | {o} | {max(B['bytes'][n]):,} | {fmt(r['tp_macro'])} | {fmt(r['tp_micro'])} | {r['esca_to_lung']} | "
                       f"{r['lung_to_esca']} | {fmt([x['acc'] for x in r['cil']])} | {fmt([x['masked'] for x in r['cil']])} | "
                       f"{fmt([x['forgetting'] for x in r['cil']])} | {r['diff']['mean']:+.4f} | {r['diff']['wins']}/10 |")
    b1 = B["bytes"]["B1"]
    out += ["", f"B1 額外 key：每任務儲存 {min(b1):,}–{max(b1):,} bytes（平均 {sum(b1) / len(b1):,.0f}）。"
            "B2 = 基本 key 8,192 ＋ 變異數 2,048 ＋ 線性層一列 2,052。上限參考為非 CL，只列線性層一列。", "",
            "上限參考的 ESCA TP = 0：依 PREREG-4 規格（全批次 200 步、Adam lr 1e-3）以四任務真實 train slides 訓練，"
            "fold 1 重現時訓練 CE 仍為 0.46、logit 只在 ±2 之間，屬訓練不足；ESCA 只佔訓練資料 5%（120／2,273），"
            "連訓練集上的 ESCA 都判對 0 張。B2 每個舊任務的假特徵張數與新任務相同，類別平衡，不受此影響。"
            "未改動規格。", ""]

    # C
    tab = []
    def tpt(xs):
        return {int(k): v for k, v in xs.items()}
    for v, typ, by in (("text-class", "文字", 0), ("text-organ", "文字", 2048), ("nav", "證據層級", 0)):
        t = T0b["tables"][v]
        cm = M1["tp_confusion_t4"][f"{v}|reverse"]
        cil = mean_sd([x["acc"] for x in M1["cil"][f"{v} + Hard|reverse"]])[0]
        tab.append((v, typ, by, {p: t["per_task_t4"][p] for p in range(4)}, t["macro"][("reverse", 4)],
                    t["micro"][("reverse", 4)], cm[0][3], cm[3][0], cil))
    for nm, typ, label in (("R0", "文字", "R0（T-Hard 規則）"), ("R2", "slide 層級", "R2 proto"),
                           ("R3(k=4)", "slide 層級", "R3(k=4)"), ("R3(k=8)", "slide 層級", "R3(k=8)"),
                           ("R1", "patch 層級", "R1 patch-vote"), ("R5", "證據層級", "R5 evidence-proto")):
        r = M2["test_R"][nm]
        tab.append((label, typ, r["bytes"], tpt(r["tp_t4_task"]), r["tp_macro"], r["tp_micro"],
                    r["conf"][0][3], r["conf"][3][0], mean_sd(r["acc_t4"])[0]))
    for nm, typ, by in (("R6", "patch 層級", 16384), ("R7", "patch 層級", 4096), ("R8", "證據層級", 16384)):
        r = M3["test"][nm]
        tab.append((nm, typ, by, tpt(r["tp_task"]), r["tp_macro"], r["tp_micro"], r["esca_to_lung"],
                    r["lung_to_esca"], mean_sd(r["acc_t4"])[0]))
    for n, typ in (("B1", "slide 層級"), ("B2", "slide 層級"), ("上限參考", "slide 層級（非 CL）")):
        for o in ORDERS:
            r = B["test"][n][o]
            tab.append((f"{n}（{o}）", typ, max(B["bytes"][n]), r["tp_task"], r["tp_macro"], r["tp_micro"],
                        r["esca_to_lung"], r["lung_to_esca"], mean_sd(r["acc_t4"])[0]))
    out += ["## T-C router 總表（test、t = 4；L0 expert、Hard；十折平均，ESCA↔Lung 為十折合計）", "",
            "| router | 類型 | 儲存 bytes／任務 | " + " | ".join(f"TP {t.split('_')[1]}" for t in d.tasks)
            + " | TP macro | TP micro | ESCA→Lung | Lung→ESCA | CIL ACC |", "|---|---|---|" + "---|" * 4 + "---|---|---|---|---|"]
    for name, typ, by, tpk, mac, mic, e2l, l2e, cil in tab:
        out.append(f"| {name} | {typ} | {by:,} | " + " | ".join(f"{mean_sd(tpk[p])[0]:.4f}" for p in range(4))
                   + f" | {mean_sd(mac)[0]:.4f} | {mean_sd(mic)[0]:.4f} | {e2l} | {l2e} | {cil:.4f} |")
    out += ["", "text-class、text-organ、nav 取自第二關快取（TP）與第三關「＋ Hard」（CIL）；R0–R5 取自 NC-2；R6–R8 取自 NC-3；"
            "B1、B2、上限參考為本輪。B1、B2 與順序有關，兩序分列；其餘在 t = 4 兩序相同。", ""]

    # D
    if full:
        out += ["## T-D 新的完整系統列", "", "| 序 | router | expert | ACC | Masked ACC | Forgetting |", "|---|---|---|---|---|---|"]
        for o, v in full.items():
            c = v["cil"]
            out.append(f"| {o} | {v['router']} | {v['experts']} | {fmt([x['acc'] for x in c])} | "
                       f"{fmt([x['masked'] for x in c])} | {fmt([x['forgetting'] for x in c])} |")
        out.append("")
    else:
        out += ["## T-D 新的完整系統列", "", "A、B 皆未通過，依 PREREG-4 操作定義 13 不列。", ""]

    tm = M["timing"]
    out += ["## T-E 實際耗時（秒，wall clock；執行緒 8）", "", "| 項目 | 秒 |", "|---|---|",
            f"| I7 快取（十折 val／test，0z／0b／v2 各 r） | {tm.get('run/1-10', float('nan')):,.0f} |",
            f"| 報告（I7 選法、B1／B2／上限參考的 key 與線性 router 訓練、總表） | {t_rep:,.0f} |", ""]
    (d.out / "REPORT_stage6.md").write_text("\n".join(out))


if __name__ == "__main__":
    raise SystemExit(main())
