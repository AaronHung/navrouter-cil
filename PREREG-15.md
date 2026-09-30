# PREREG-15 — NC-15 預先註冊（LoRA 對照版 L1 v2 的 r 改以 validation 選定）

登記時間：2026-09-30，任何訓練、評估開始前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；本輪所有數字來自同一台。
分支：`lora-val-select`（worktree `01_navrouter-cil-lora-val`）。長時間工作在 tmux 裡以 `caffeinate -dimsu` 執行，每 15 分鐘在 log 記一行進度。任何 run 失敗即停，不重試、不跳過。

## 背景

LoRA 對照版（L1 v2）目前的 r = 2 是依 test WP 選的（PREREG-2.md:15、:46-47；REPORT_stage5.md:37-44）。
本輪改用與修正頭（I6）相同的規則（PREREG-7.md:17、操作定義 4），在 validation 上重選 r。

## 判準與設定（原文）

- 候選：r ∈ {1, 2, 3}。
- 規則：取「十折四任務 validation WP ≥ L0 validation WP − 0.01（= 0.9190）」的最小 r，記為 r\*。
- 同分規則：多個 r 符合時取較小的 r。
- 選定 r\* 時不看 test：選 r 只讀 validation 的評估結果；test 的任何輸出（含 r = 3 訓練附帶產生的 test 評估）在 r\* 寫入並 commit 之前不讀取。
- 選定後才讀 test：
  - r\* = 2：確認既有數字不變，並註明「r = 2 已由 validation 選定」。
  - r\* ≠ 2：報 r\* 的 task-known test WP（兩序），並做 AR ＋ L1(r\*) 的離線推論，報 task-inferred 的 t = 4 ACC（兩序、十折）；不重訓。
- 比較：與修正頭（task-known 0.9340、task-inferred 0.9128）的差距，以及是否在 0.005 容許範圍內（不設為通過／未通過的判準，只報）。

## 操作定義（與判準同時登記）

1. **L1(r) 的訓練**：沿用 `scripts/nc2_lora.py --tag lora_v2` 的設定（PREREG-2 操作定義 8；AMENDMENT-1 後的 v2）：
   底座 = 同一折第一關 bank 中該序第一個任務的 expert，凍結、直接使用；其餘任務各自從底座訓練
   rank-r 增量（seed 42、5 epochs、Adam lr 5e-4、wd 1e-4、每步有限性檢查、執行緒 8）。
   r = 1、2 沿用既有權重 `outputs/navcil/mac/lora_v2/r{1,2}/{reverse,paper}/`（不重訓、不覆寫）。
   r = 3 為新訓練，兩序、十折，程式呼叫 `nc2_lora.py` 的 `train_order_fold`、`eval_order_fold`
   （新檔 `scripts/nc15_lora_train.py`，不改既有腳本），輸出到新目錄 `outputs/navcil/mac/nc15/lora_r3/{reverse,paper}/`。
   `eval_order_fold` 附帶的 test 評估照常產生，但在操作定義 5 的 commit 之前不讀取。
2. **validation WP**（新檔 `scripts/nc15_lora_val.py`）：對每折、每個任務 τ 的 validation slides，取該序中
   τ 的 expert（τ 為該序第一個任務時是底座 L0 expert，否則為 L1(r) expert τ），做與 test 相同的四輪選片
   （K = 64、每輪 16、λ\* = 1.5），所選 patch 等權平均後 L2 正規化，對 8 類文字取 cosine，在 τ 的 2 類內取
   argmax，計正確率（Masked ACC）。四任務等權平均 = 該折 validation WP；再取十折平均。L1 依序而異
   （底座不同），所以 reverse、paper 各算一次。
3. **L0 validation WP**：第一關 bank 的 L0 expert 在 validation 上的四輪 Masked ACC（NC-2 validation 快取
   `cache/nc2_fold{f}_val_{task}.pt` 的 `four_cos8_uni`），四任務平均、十折平均，算法與 `scripts/nc7_report.py:51-55`
   相同；期望值 0.9290（REPORT_stage9.md:19）。重算值四捨五入到小數第 4 位不等於 0.9290 即停。
   門檻 = 重算的 L0 validation WP − 0.01（未四捨五入，與 nc7_report.py:67 相同的比較方式）；四捨五入為 0.9190。
   若任何候選的過門檻判定在「未四捨五入門檻」與「字面 0.9190」之間不同，停下來回報。
4. **選 r**：r 符合 ⇔ reverse 與 paper 兩序各自的十折平均 validation WP 都 ≥ 門檻（兩序都要，同 PREREG-2
   操作定義 10）。r\* = 符合者中最小的 r。三個都不符合時，取「兩序中較低者的十折平均 validation WP」最高的 r，
   並標記「未符合」（同分取較小的 r）。
5. **不看 test 的保證**：選 r 的程式（`scripts/nc15_select.py`）只讀 `nc15/val/` 與 NC-2 validation 快取，
   輸出 `nc15/selection.json`；此檔 commit 之後才執行讀 test 的程式（`scripts/nc15_test.py`），後者啟動時
   檢查 `selection.json` 已 commit 且未修改，否則停止。
6. **validation 評估的一致性檢查**：每序第一個任務（底座 = L0 expert）的 validation 8 類 cosine，必須與 NC-2
   validation 快取 `four_cos8_uni[:, τ]` 逐張逐位相同（`torch.equal`）；slide 順序與 slide id 必須與快取一致。
   不符即停。
7. **task-known test WP**：test 上 oracle 分派（真實任務的 expert），t = 4 四任務等權平均的 Masked ACC，
   由各折 `fold{f}_eval.pt` 的 `l1_cos8[:, τ]` 在 τ 的 2 類內取 argmax 計算（與 `scripts/nc2_report.py:174-177`
   相同）；兩序各報十折平均。
8. **task-inferred**：router = AR（mean_vec、γ = 0.001、float64，`nc6_report.ARX`），Hard，分派到 τ̂ 時用
   L1(r\*) expert τ̂ 的 test 8 類 cosine（`l1_cos8`）；t = 4 CIL ACC（四任務等權平均），兩序、十折。算法與
   `scripts/nc7_report.py:75-83` 的 D1 相同，只把 r = 2 的權重目錄換成 r\* 的目錄。
9. **一致性檢查（test，讀 test 之後）**：本輪以同一程式重算下列值，四捨五入到小數第 4 位須與基準相同，
   不符即停：
   - AR ＋ L1(r = 2)（task-inferred）：reverse 0.9176、paper 0.9132（REPORT_stage10.md:15、:28）；另與
     `nc8/per_fold.json` 的逐折值比對（逐折最大絕對差 ≤ 1e-6）。不論 r\* 是否為 2 都做。
   - L1(r = 2) task-known WP：reverse 0.9401、paper 0.9339（REPORT_stage5.md:43、:44）。
   - 修正頭 I6(r = 2)：task-known 0.9340（REPORT_stage10.md:18）、task-inferred 0.9128（REPORT_stage10.md:16），
     由 `i6/eval_fold{f}.pt` 的 test cosine 以同一程式重算。
10. **與修正頭的差距**：差距 = 修正頭 − LoRA 對照版（兩序、task-known 與 task-inferred 各自以十折平均相減）。
    「在 0.005 容許範圍內」⇔ 差距 ≥ −0.005（即修正頭不低於 LoRA 對照版 − 0.005，同 PREREG-7 的 D3-次要判準
    方向）。另報逐折差與修正頭較高的折數。
11. **輸出**：一律在 `outputs/navcil/mac/nc15/`（worktree 內）；既有 `lora_v2/`、`i6/`、`cache/`、`bank/` 只讀
    （路徑以命令列參數傳入，程式內不寫絕對路徑）。數字附 fact-id（`nc15.*`，`nc15/facts.json` 一行一個）。
12. **術語**：報告正文稱「修正頭」（I6）與「LoRA 對照版」（L1 v2）。
