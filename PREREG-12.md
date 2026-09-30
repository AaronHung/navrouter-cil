# PREREG-12 — NC-12 預先註冊（六種學習順序的中途軌跡；t = 4 與順序無關的驗證）

登記時間：2026-09-30，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），CPU；本輪所有數字來自同一台、同一次執行（`scripts/nc12_order_traj.py`）。
分支：`ws3-order`（worktree `~/research/01_navrouter-cil-ws34`，自 main @ ee2be2c）。全部只做推論，不訓練任何東西。

前提（已確認）：`ws1-closed-form` 已合併進 main（d09eab4），REPORT_stage11 判準 1–7 全部通過。

## 目的

主方法 D3 = AR 分派器（γ = 1e-3，`selector/incremental_ridge.py` 的累加統計量）＋ I6(r = 2) 任務分類頭。
閉式解的累加與順序無關（只差浮點捨入），各任務的任務分類頭各自獨立訓練、與順序無關（REPORT_stage11 R1），
所以學完四個任務後（t = 4）的結果應與順序無關；中途各階段（t = 1–3）的已見任務不同，結果會不同。

## 順序（6 種）

| 代號 | 順序 | 來源 |
|---|---|---|
| reverse | esca → rcc → brca → lung | 既有（`selector/cil_ops.py:18-21`） |
| paper | lung → brca → rcc → esca | 既有 |
| perm1 | lung → rcc → esca → brca | 隨機 |
| perm2 | lung → esca → brca → rcc | 隨機 |
| perm3 | esca → rcc → lung → brca | 隨機 |
| perm4 | brca → esca → rcc → lung | 隨機 |

隨機排列的產生：`random.Random(20260930)`，重複呼叫 `rng.sample(["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"], 4)`，
依序保留與 reverse、paper 及已保留者都不重複的前 4 個。seed = 20260930。

## 指標（每順序 × 每折 × 每階段 t = 1–4）

1. CIL 正確率：第 t 階段只計已見的 t 個任務，各任務正確率等權平均（與 `scripts/nc5_report.py:43-59` `cil_full` 的 acc_t 相同）；
   第二站 Hard（`scripts/nc2_report.py:89-96`），證據為 NC-8 快取的 `I6_cos8`。
2. 分派正確率：已見任務 test slides 中 τ̂ = 真實任務的比例（micro）與已見任務各自比例的等權平均（macro）。
3. 4 × 4 分派混淆（每階段、十折合計）。
資料：NC-8 同一批快取（test 的 mean_vec、`I6_cos8`；train 的 mean_vec），以 `--cache-dir` 唯讀指向主工作樹 `outputs/navcil/mac/cache/`。
AR 在每個順序、每折以 `stream` 逐階段累加，第 t 階段只讀該順序第 t 個任務的 train mean_vec。

## 預期與判準（t = 4 一致性；任一不符即停，不調參、不改判準）

預期：「t = 4 所有順序的最終值完全一致」。操作判準：
1. 每折、每張 test slide 的 τ̂ 在 6 種順序下逐張一致（依 canonical 任務順序對齊後比對；不一致張數 = 0）。
2. 每折 t = 4 的 CIL ACC 在 6 種順序下位元相同，且等於 `nc8/per_fold.json` 第 6 列（D3）該折的 ACC（位元相同；十折平均 ± sd 即
   REPORT_stage10.md:16 的 0.9128 ± 0.0258）。
3. 每折 t = 4 的 W（欄依任務對齊）與 reverse 的最大絕對差 ≤ 1e-9（不同順序加總 A 的浮點誤差；REPORT_stage11 記錄為 1.12e-10）。

另核對（不符亦停）：reverse、paper 的逐階段 CIL（acc_t）逐折與 `nc8/per_fold.json` 第 6 列位元相同。

## 另報（不設判準）

- 軌跡表：順序 | t = 1 | t = 2 | t = 3 | t = 4（CIL 與分派正確率，十折 mean ± sd）。
- 觀察：t = 2、3 中，十折平均分派正確率 micro 最低的（順序、階段），附其十折合計的混淆矩陣。

## 執行時間估算

只讀快取、不讀特徵檔。6 順序 × 十折 × 4 階段的累加與評估；NC-9（2 順序 × 2 分派器）實測 14–15 秒，估本輪 < 1 分鐘。
以 `scripts/run_stage.sh` 在 tmux 中執行。

## 輸出

`outputs/navcil/mac/nc12/`（`per_fold.json`、`result.json`、`facts.json`）與 `outputs/navcil/mac/REPORT_stage14.md`。不改既有腳本與既有 outputs。
