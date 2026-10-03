"""MOE-4 共用（PREREG-20）：w(s) 向量快取（vec 階段）、資料組裝、RDG 累加統計量、系統清單與逐折評估、彙總工具。

沿用既有程式，不重寫：
  階段外殼、Run、ar_stages、run_cl、ridge_d、txt_d、vec_of、row_diff、order_same   moe3_common
  v(s)（四輪）、one42（= w(42)）、s42cells（K5）、k2 參照、paired、AR、nc8_main       moe2_common、moe1_common
  u_64（不用 head）                                                                    moe3 cache（唯讀）
  選片與 head 前向                                                                      selector.cil_ops.mean_norm、selector.multiround.top_k_select、
                                                                                       selector.evaluate.read_slide、moe1_common.load_head
"""
from __future__ import annotations

import statistics
import time

import torch

import moe3_common as T
from moe3_common import C, CheckFailed, D64, E, M, N5, log  # noqa: F401  (re-export for stage scripts)
from selector.cil_ops import mean_norm                       # noqa: E402
from selector.evaluate import read_slide                     # noqa: E402
from selector.multiround import top_k_select                 # noqa: E402

ORDERS = M.ORDER_NAMES                  # ["reverse", "paper"]
SEEDS = (42, 43, 44, 45, 46)
NEW = (43, 44, 45, 46)
K_ONE = 64                              # 一次取 64（PREREG-20 操作定義）
G = 1e-3                                # RDG 的 γ（固定，不選擇）
GAMMA_SENS = (1e-4, 1e-3, 1e-2)         # H4（只作描述）
MOE3_DIR = "moe3"
TOL_GATE = 1e-12                        # 「> 0 的折數」：差 > 1e-12

def stage_main(stage: str, body) -> int:
    """同 moe3_common.stage_main（done 標記、FAILED_<階段>.txt、不重試），log 標籤為 MOE-4。"""
    import json
    import traceback
    from moe2_common import now
    run = T.Run(stage)
    done, fail = run.root / f"{stage}.done", run.root / f"FAILED_{stage}.txt"
    if done.exists():
        log(f"{stage}: done, skip")
        return 0
    log(f"MOE-4 {stage} folds={run.folds} out={run.root.name} threads={torch.get_num_threads()}")
    t0 = time.perf_counter()
    try:
        res = body(run)
    except Exception:
        fail.write_text(f"[{now()}] 階段 {stage} 失敗（folds={run.folds}）\n{traceback.format_exc()}")
        traceback.print_exc()
        return 3
    res = {"stage": stage, "folds": run.folds, "wall_s": round(time.perf_counter() - t0, 1), "finished": now(), **res}
    (run.root / f"{stage}.json").write_text(json.dumps(write_json_ok(res), indent=1, ensure_ascii=False, default=M._default))
    done.write_text(now() + "\n")
    log(f"{stage}: 完成（{res['wall_s']:.0f}s）")
    return 0


# ── 系統清單（PREREG-20 第一部分）────────────────────────────────────────────
def system_specs(gammas_sens: bool = False) -> list[tuple]:
    """[(key, name, seed, kind, reader)]：kind = w{s}／s{s}／u64；reader = "TXT" 或 γ。"""
    out = []
    for s in SEEDS:
        out += [(f"P-F({s})", "P-F", s, f"w{s}", G),
                (f"P-4({s})", "P-4", s, f"s{s}", G),
                (f"P-T1({s})", "P-T1", s, f"w{s}", "TXT"),
                (f"P-main({s})", "P-main", s, f"s{s}", "TXT")]
    out.append(("P-0", "P-0", None, "u64", G))
    if gammas_sens:
        for g in GAMMA_SENS:
            if g == G:
                continue
            for s in SEEDS:
                out.append((f"P-F({s})γ{g:g}", "P-F-sens", s, f"w{s}", g))
    return out


# ── 路徑 ────────────────────────────────────────────────────────────────────
def w_path(run: T.Run, split: str, f: int):
    return run.root / "cache" / f"w_{split}_fold{f}.pt"


def u_path(run: T.Run, split: str, f: int):
    return run.base / MOE3_DIR / "cache" / f"u_{split}_fold{f}.pt"


def head_path(run: T.Run, s: int, f: int, task: str):
    """seed 42 在 i6/r2/（MOE-1 之前的訓練）；seed 43–46 在 moe1/i6_seed{s}/（同 moe2_common.head_path）。"""
    root = run.base / "i6" / "r2" if s == M.SEED0 else run.m1 / f"i6_seed{s}"
    return root / f"fold{f}_{task}.pt"


def require_chk(run: T.Run) -> dict:
    """各階段開始時檢查 chk.json：不過就拒絕執行（防止手動單獨重跑）。"""
    p = run.root / "chk.json"
    if not p.exists():
        raise CheckFailed("chk.json 不存在：chk 未完成，本階段不執行")
    res = run.json("chk")
    if not res.get("pass"):
        raise CheckFailed("chk.json 的 pass 不為真：本階段不執行")
    return res


# ── w(43…46) 向量（vec 階段；細則 2–3）──────────────────────────────────────
def build_w(run: T.Run, fold: int, split: str) -> None:
    """每張 slide 讀一次。對 seed s ∈ {43…46}、任務 p：score = head_{s,p}(Z, f_task_p)（s0 + g），
    idx = top_k_select(score, 64)，w = mean_norm(Z, idx)。train 只算自己任務；validation／test 四個任務都算。"""
    path = w_path(run, split, fold)
    done = path.with_suffix(".done")
    if done.exists():
        return
    ctx, t_all, out = run.ctx, time.perf_counter(), {}
    H = {s: [M.load_head(head_path(run, s, fold, t)) for t in run.tasks] for s in NEW}
    for j, task in enumerate(ctx.tasks):
        ds, shift = ctx.ds(fold, task, split)
        rows = {s: [] for s in NEW}
        sids, labels, t_read, t_comp, n_patch = [], [], [], [], []
        ps = [j] if split == "train" else list(range(4))
        for i in range(len(ds)):
            t0 = time.perf_counter()
            rec = read_slide(ds, shift, i)
            t1 = time.perf_counter()
            Z = rec.Z
            with torch.no_grad():
                for s in NEW:
                    v = []
                    for p in ps:
                        score = H[s][p](Z, ctx.f_task(p))
                        v.append(mean_norm(Z, top_k_select(score, K_ONE)))
                    rows[s].append(v[0] if split == "train" else torch.stack(v))
            t_comp.append(time.perf_counter() - t1); t_read.append(t1 - t0)
            sids.append(rec.sid); labels.append(rec.label); n_patch.append(Z.shape[0])
        out[task] = {"sids": sids, "labels": torch.tensor(labels), "n_patch": torch.tensor(n_patch),
                     "t_read_s": torch.tensor(t_read), "t_compute_s": torch.tensor(t_comp),
                     "w": {s: torch.stack(v) for s, v in rows.items()}}
    torch.save(out, path)
    done.write_text(time.strftime("%F %T") + "\n")
    log(f"vec w fold{fold} {split}: n={sum(len(v['sids']) for v in out.values())} {time.perf_counter() - t_all:.0f}s")


def check_w_align(run: T.Run, fold: int, split: str) -> dict:
    """細則 3 對齊檢查：w 快取的 slide id、標籤與 MOE-2 vec 快取逐張相同；所有 w 有限、範數與 1 的差 ≤ 1e-5。"""
    raw = torch.load(w_path(run, split, fold), map_location="cpu")
    ref = E.vecs(run.v2, split, fold)["raw"]
    maxn, finite = 0.0, True
    for t in run.tasks:
        if raw[t]["sids"] != ref[t]["sids"] or not torch.equal(raw[t]["labels"], ref[t]["labels"]):
            raise CheckFailed(f"fold {fold} {split} {t}: w 快取的 slide id／標籤與 MOE-2 vec 快取不一致")
        for s in NEW:
            w = raw[t]["w"][s]
            finite &= bool(torch.isfinite(w).all())
            maxn = max(maxn, float((w.norm(dim=-1) - 1).abs().max()))
    if not finite or maxn > 1e-5:
        raise CheckFailed(f"fold {fold} {split}: w 向量非有限或範數偏離 1（max {maxn:.2e}）")
    return {"norm_max_abs": maxn, "finite": finite}


# ── 資料組裝（train：每任務 p 的切片；val／test：[N, 4, 512]）─────────────────
def data4(run: T.Run, split: str, f: int) -> dict:
    k = (split, f)
    if k in run._data:
        return run._data[k]
    D2 = E.vecs(run.v2, split, f)
    rw = torch.load(w_path(run, split, f), map_location="cpu")
    ru = torch.load(u_path(run, split, f), map_location="cpu")
    for t in run.tasks:
        if ru[t]["sids"] != D2["raw"][t]["sids"] or not torch.equal(ru[t]["labels"], D2["raw"][t]["labels"]):
            raise CheckFailed(f"fold {f} {split} {t}: u_64 快取與 MOE-2 快取的 slide id／標籤不一致")
    if split == "train":
        per = []
        for p, t in enumerate(run.tasks):
            d = D2["per"][p]
            v = {f"s{s}": d["v"][f"s{s}"] for s in SEEDS}
            v["w42"] = d["v"]["one42"]
            v.update({f"w{s}": rw[t]["w"][s] for s in NEW})
            v["u64"] = ru[t]["u"][K_ONE]
            per.append({"label": d["label"], "v": v, "mv": d["mv"]})
        out = {"per": per}
    else:
        v = {f"s{s}": D2["v"][f"s{s}"] for s in SEEDS}
        v["w42"] = D2["v"]["one42"]
        v.update({f"w{s}": torch.cat([rw[t]["w"][s] for t in run.tasks]) for s in NEW})
        v["u64"] = torch.cat([ru[t]["u"][K_ONE] for t in run.tasks])
        out = {"task": D2["task"], "label": D2["label"], "mv": D2["mv"], "v": v, "_cos": {}}
    run._data[k] = out
    return out


# ── RDG 累加統計量（細則 4；float64）─────────────────────────────────────────
class Acc:
    """一種輸入向量（kind）的累加式統計量；累加方式同 moe3_common.Stats，資料來源為本批的 data4。"""

    def __init__(self, run: T.Run, kind: str):
        self.run, self.kind, self._g, self._ab = run, kind, {}, {}

    def clear(self) -> None:
        self._g.clear(); self._ab.clear()

    def task_stats(self, f: int, p: int):
        k = (f, p)
        if k not in self._g:
            d = data4(self.run, "train", f)["per"][p]
            X, y = N5.aug(d["v"][self.kind]), d["label"]
            self._g[k] = (X.t() @ X, {c: X[y == c].sum(0) for c in (2 * p, 2 * p + 1)})
        return self._g[k]

    def AB(self, f: int, o: str, t: int):
        """序 o 的階段 t：依序累加前 t 個任務；B 只含已學類別（固定 8 類序）。"""
        k = (f, o, t)
        if k not in self._ab:
            A, cols = None, {}
            for p in self.run.pos(o)[:t]:
                G_, cs = self.task_stats(f, p)
                if A is None:
                    A = torch.zeros_like(G_)
                A += G_
                cols.update(cs)
            cls = sorted(cols)
            self._ab[k] = (A, torch.stack([cols[c] for c in cls], 1), cls)
        return self._ab[k]

    def W(self, f: int, o: str, t: int, gamma: float):
        A, B, cls = self.AB(f, o, t)
        return torch.linalg.solve(A + gamma * torch.eye(A.shape[0], dtype=D64), B), cls


def acc_of(run: T.Run, kind: str) -> Acc:
    if kind not in run._stats:
        run._stats[kind] = Acc(run, kind)
    return run._stats[kind]


def dfn_of(run: T.Run, f: int, o: str, D: dict, kind: str, reader):
    """回傳 dfn(t, p) → [N] 的分數差（第一類 − 第二類）。reader = "TXT" 或 γ（RDG）。"""
    if reader == "TXT":
        return lambda t, p: T.txt_d(run, D, kind, p)
    st, Ws = acc_of(run, kind), {}

    def fn(t, p):
        if t not in Ws:
            Ws[t] = st.W(f, o, t, reader)
        return T.ridge_d(*Ws[t], N5.aug(T.vec_of(D, kind, p)), p)
    return fn


def eval_fold(run: T.Run, f: int, D: dict, specs: list[tuple]) -> dict:
    """一折：每序、每個系統的 run_cl 結果（含私有欄位 _tell、_cil、_d）。回傳 {(key, 序): row}。"""
    res = {}
    for o in ORDERS:
        ar = T.ar_stages(run, f, o, D)
        for key, _name, _s, kind, reader in specs:
            res[(key, o)] = T.run_cl(run, f, o, D, dfn_of(run, f, o, D, kind, reader), ar)
    return res


# ── 彙總（mean、sd；sd 為十折的樣本標準差）────────────────────────────────────
def mean_sd(xs: list[float]) -> list[float]:
    xs = [float(x) for x in xs]
    return [statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0]


def ms_tree(xs: list):
    """xs：逐折的巢狀值（數或 list，None 保留）→ 同形狀、葉為 [mean, sd]。"""
    x0 = xs[0]
    if x0 is None:
        return None
    if isinstance(x0, list):
        return [ms_tree([x[i] for x in xs]) for i in range(len(x0))]
    return mean_sd(xs)


def avg_tree(xs: list):
    """多個 seed 同一折的巢狀值逐元素平均（先平均、後面再算十折）。"""
    x0 = xs[0]
    if x0 is None:
        return None
    if isinstance(x0, list):
        return [avg_tree([x[i] for x in xs]) for i in range(len(x0))]
    return sum(xs) / len(xs)


FIELDS = ("acc_t", "wp_t", "wp_task_t", "forgetting", "bwt", "abar")


def pub_row(row: dict) -> dict:
    d = {k: row[k] for k in FIELDS}
    d["cil4"], d["wp4"] = row["acc_t"][3], row["wp_t"][3]
    return d


def summarize(rows: list[dict]) -> dict:
    """十折（每折一列）→ 各欄位的 [mean, sd]（wp_task_t 為 4×4，未學為 None）。"""
    return {k: ms_tree([r[k] for r in rows]) for k in FIELDS}


def summarize_avg(per_seed_rows: list[list[dict]]) -> dict:
    """多個 seed 的十折列 → 先對 seed 平均（每折）、再算十折 mean ± sd。"""
    folds = list(zip(*per_seed_rows))
    avg = [{k: avg_tree([r[k] for r in fr]) for k in FIELDS} for fr in folds]
    return summarize(avg)


def pairs(a: list[float], b: list[float]) -> dict:
    """逐折相減（平均、贏折數 > 1e-12、Wilcoxon 雙尾 p、逐折值）。"""
    return M.paired(a, b)


def write_json_ok(x):
    """轉成可 JSON 序列化的純數值（不含 tensor）。"""
    if isinstance(x, dict):
        return {str(k): write_json_ok(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [write_json_ok(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, float):
        return float(x)
    return x
