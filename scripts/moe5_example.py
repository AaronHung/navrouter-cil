"""MOE-5 示例腳本：FINAL（P-4）在 fold 1 的一張 slide 上逐步算出中間量。

只推論、不訓練、不寫入任何既有產物。讀取：
- outputs/navcil/mac/cache/nc8_fold1_train_*.pt（AR 的 train mean_vec）
- outputs/navcil/mac/cache/fold1_test_*.pt（test mean_vec、sids）
- outputs/navcil/mac/moe1/cache/s42v_train_fold1.pt（FINAL 的 train v，每任務自己的 head）
- outputs/navcil/mac/moe1/cache/s42cells_test_fold1.pt（test 的 v_four，四個 head 的四輪 v）
- cache/text/f_txt_*.pt（文字特徵）
- outputs/navcil/mac/moe0/b3_per_slide.csv（主系統 WP 的 d 值，只用來找示例 slide）

輸出：outputs/navcil/mac/moe5_example/example.json，以及 stdout。
用法：PYTHONNOUSERSITE=1 NAVCIL_MACHINE=mac python scripts/moe5_example.py
"""
import csv
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
MAC = ROOT / "outputs" / "navcil" / "mac"
OUT = MAC / "moe5_example"
TASKS = ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]
LABEL = ["ESAD", "ESCC", "CCRCC", "PRCC", "IDC", "ILC", "LUAD", "LUSC"]
FOLD = 1
GAMMA = 1e-3
D64 = torch.float64
EXAMPLE_INDEX = 0  # fold 1、tcga_lung、test 第 0 張（與 AUDIT_wp.md A9 相同）


def aug(X):
    X = X.to(D64)
    return torch.cat([X, torch.ones(X.shape[0], 1, dtype=D64)], 1)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    F = torch.cat([load(ROOT / f"cache/text/f_txt_{t}.pt")["f_txt"] for t in TASKS], 0).float()  # [8,512]

    # ---- AR（TP）：A = Σ XᵀX（四任務 train mean_vec），B 第 p 欄 = Σ X_p，γ = 1e-3 ----
    A_ar = torch.zeros(513, 513, dtype=D64)
    B_ar = []
    n_train = {}
    for t in TASKS:
        X = aug(load(MAC / f"cache/nc8_fold{FOLD}_train_{t}.pt")["mean_vec"])
        n_train[t] = int(X.shape[0])
        A_ar += X.t() @ X
        B_ar.append(X.sum(0))
    B_ar = torch.stack(B_ar, 1)  # [513, 4]
    W_ar = torch.linalg.solve(A_ar + GAMMA * torch.eye(513, dtype=D64), B_ar)

    # ---- FINAL（P-4）ridge：x = [v; 1]，v 為每張 train slide 自己任務的 head 四輪向量 ----
    A_v = torch.zeros(513, 513, dtype=D64)
    B_v = torch.zeros(513, 8, dtype=D64)
    for t in TASKS:
        d = load(MAC / "moe1/cache/s42v_train_fold1.pt")[t]
        X = aug(d["v"])
        y = d["labels"]
        A_v += X.t() @ X
        for c in range(8):
            m = y == c
            if m.any():
                B_v[:, c] += X[m].sum(0)
    W_v = torch.linalg.solve(A_v + GAMMA * torch.eye(513, dtype=D64), B_v)

    # ---- 示例 slide：fold 1、tcga_lung、test 第 0 張 ----
    cells = {t: load(MAC / "moe1/cache/s42cells_test_fold1.pt")[t] for t in TASKS}
    test_mv = {t: load(MAC / f"cache/fold{FOLD}_test_{t}.pt") for t in TASKS}
    lung = TASKS.index("tcga_lung")
    i = EXAMPLE_INDEX
    sid = cells["tcga_lung"]["sids"][i]
    assert sid == test_mv["tcga_lung"]["sids"][i], "sid 對齊失敗"
    mv = test_mv["tcga_lung"]["mean_vec"][i].to(D64)

    # TP：AR 分數
    tp_scores = (aug(mv.unsqueeze(0)) @ W_ar)[0]
    tau = int(tp_scores.argmax())

    # 主系統：τ̂ head 的四輪 v 與 8 類文字 cosine；在 τ̂ 的兩類內 argmax
    v_tau = cells["tcga_lung"]["v_four"][i, tau].to(torch.float32)  # v_four 的第 2 軸為四個 head
    cos8_main = v_tau @ F.t()
    pair = slice(2 * tau, 2 * tau + 2)
    main_cil = 2 * tau + int(cos8_main[pair].argmax())
    v_true = cells["tcga_lung"]["v_four"][i, lung].to(torch.float32)
    cos_true = v_true @ F[6:8].t()

    # FINAL：8 類 ridge 分數；CIL 向量取自 τ̂；WP 向量取自真實任務 lung
    s_cil = aug(v_tau.to(D64).unsqueeze(0)) @ W_v
    s_cil = s_cil[0]
    d_cil = float(s_cil[2 * tau] - s_cil[2 * tau + 1])
    final_cil = 2 * tau + (0 if d_cil >= 0 else 1)
    s_wp = (aug(v_true.to(D64).unsqueeze(0)) @ W_v)[0]
    d_wp = float(s_wp[6] - s_wp[7])
    final_wp = 6 if d_wp >= 0 else 7

    rec = {
        "fold": FOLD, "task": "tcga_lung", "index": i, "sid": sid,
        "label": LABEL[7],
        "n_train_per_task": n_train,
        "A_ar_shape": list(A_ar.shape), "B_ar_shape": list(B_ar.shape),
        "A_ridge_shape": list(A_v.shape), "B_ridge_shape": list(B_v.shape), "W_ridge_shape": list(W_v.shape),
        "n_train_total": sum(n_train.values()),
        "tp_scores": [float(x) for x in tp_scores],
        "tau_hat": TASKS[tau],
        "v_norm": float(v_tau.to(D64).norm()),
        "cos8_main_tau": [float(x) for x in cos8_main],
        "cos_true_LUAD_LUSC": [float(x) for x in cos_true],
        "main_cil_pred": LABEL[main_cil],
        "final_ridge_scores_8": [float(x) for x in s_cil],
        "final_d_in_tau_pair": d_cil,
        "final_cil_pred": LABEL[final_cil],
        "final_wp_scores_LUAD_LUSC": [float(s_wp[6]), float(s_wp[7])],
        "final_wp_d": d_wp,
        "final_wp_pred": LABEL[final_wp],
    }
    rec["check_A9"] = {
        "tp_expected": [0.0280, 0.0230, -0.1231, 1.0721],
        "cos_expected_LUAD_LUSC": [0.577495, 0.667105],
        "tp_max_abs_diff_vs_A9": max(abs(a - b) for a, b in zip(rec["tp_scores"], [0.0280, 0.0230, -0.1231, 1.0721])),
        "cos_max_abs_diff_vs_A9": max(abs(a - b) for a, b in zip(rec["cos_true_LUAD_LUSC"], [0.577495, 0.667105])),
    }

    # ---- 找示例小葉癌（ILC）test slide：FINAL 的告訴任務判錯、主系統判對的第一張 ----
    b3 = {}
    with open(MAC / "moe0/b3_per_slide.csv") as fh:
        for row in csv.DictReader(fh):
            if int(row["fold"]) == FOLD and row["task"] == "tcga_brca":
                b3[row["slide_id"]] = row
    brca = test_mv["tcga_brca"]
    cb = cells["tcga_brca"]
    assert cb["sids"] == brca["sids"], "brca sid 對齊失敗"
    found = None
    for j, s in enumerate(brca["sids"]):
        if int(brca["labels"][j]) != 5:  # ILC
            continue
        v_b = cb["v_four"][j, 2].to(torch.float32)  # brca head
        s_b = (aug(v_b.to(D64).unsqueeze(0)) @ W_v)[0]
        fin_pred = 4 if float(s_b[4] - s_b[5]) >= 0 else 5
        fin_wrong = fin_pred != 5
        c_b = v_b @ F[4:6].t()
        main_right = int(c_b.argmax()) == 1
        if fin_wrong and main_right:
            found = (j, s, float(s_b[4] - s_b[5]), c_b.tolist(), [float(x) for x in s_b])
            break
    rec["small_cell_example"] = None
    if found is not None:
        j, s, d_fin, c_b, s8 = found
        mv_b = brca["mean_vec"][j].to(D64)
        tp_b = (aug(mv_b.unsqueeze(0)) @ W_ar)[0]
        tau_b = int(tp_b.argmax())
        rec["small_cell_example"] = {
            "index_in_brca_test": j, "sid": s, "label": "ILC",
            "told_task_final_d_IDC_minus_ILC": d_fin,
            "told_task_main_cos_IDC_ILC": c_b,
            "final_ridge_scores_8": s8,
            "tp_scores": [float(x) for x in tp_b],
            "tau_hat": TASKS[tau_b],
        }
    (OUT / "example.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
