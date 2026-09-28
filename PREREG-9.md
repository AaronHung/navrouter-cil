# PREREG-9 — NC-9 預先註冊（AR 分派器改為只存累加統計量；證明十折數字不變）

登記時間：2026-09-28，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`；本輪所有數字來自同一台、同一批。
分支：`ws1-closed-form`。全部只做推論，不訓練任何東西；任務分類頭權重全部沿用已存的（I6 r=2），
證據沿用 NC-8 同一批快取（`outputs/navcil/mac/cache/nc8_fold*_{train,test}_*.pt`）。

## 目的

NC-8 的 AR 分派器（`scripts/nc8_report.py:71-88`）在第 t 階段重新載入所有已見任務的 train mean_vec，
從零加總 A = Σ_j X_jᵀX_j 與 b_j，等於保留了全部舊訓練片的向量。本輪把分派器改寫成只保存累加統計量
（A 與 b_1…b_t）的形式：第 t 階段只讀任務 t 的 train mean_vec，更新統計量後即丟棄該批向量。
目標是證明改寫後主結果 D3（AR γ = 1e-3 ＋ I6 r=2）與 AR-bal（γ = 1e-4，w_j = 1/n_j）＋ I6 r=2 的十折數字
與 NC-8 完全一致。

## 改動範圍

1. 新增 `selector/incremental_ridge.py`：`IncrementalRidge`，狀態只有 A（513 × 513，float64）、
   b_1…b_t（各 513 維，float64）與任務 id 清單；`add_task(task_id, X_j)` 只接收當前任務的 train mean_vec，
   函式結束後不保留 X_j 的任何引用；`solve()` 回傳 W = solve(A + γI, B)；`save`／`load` 存取統計量。
   AR：A += X_jᵀX_j、b_j = X_jᵀ1；AR-bal：A += w_j X_jᵀX_j、b_j = w_j X_jᵀ1，w_j = 1/n_j。
   X = mean_vec 轉 float64 後接常數 1（與 `scripts/nc5_report.py:151-154` 相同）。
2. 新增 `tests/test_incremental_ridge.py`（三項，見判準 6）。
3. 新增 `scripts/nc9_incremental.py`：流程與 `scripts/nc8_report.py` 相同（以唯讀 import 沿用其評估程式），
   只把 AR／AR-bal 分派器換成 `IncrementalRidge`。統計量在階段之間經由檔案傳遞（載入 → add_task → 存檔），
   不在記憶體保留舊任務的向量。十折、reverse 與 paper 兩序全跑。
4. 輸出：`outputs/navcil/mac/nc9/`（統計量檔、比對結果 json）與 `outputs/navcil/mac/REPORT_stage11.md`。
5. 不改動任何既有腳本（含 `nc8_batch.py`、`nc8_report.py`、`nc7_*.py`）與既有 outputs。

## 判準（全部通過才算成立；任一不通過即停下回報，不調參、不改判準）

比對基準一律是 NC-8 的既有產物（`outputs/navcil/mac/nc8/per_fold.json`、`REPORT_stage10.md`、
`nc8/fig/confusion_{AR,AR-bal}.csv`），以及以唯讀方式呼叫 `nc8_report.B8.W`（從頭加總）重算的參考值。

1. **D3 數字**：兩序各自的十折 ACC 平均 ± 標準差（小數點後四位）與 `REPORT_stage10.md:16`、`:29` 相同；
   每折 ACC 與 `per_fold.json` 第 6 列在小數點後四位相同。Masked ACC、Forgetting、BWT 同樣逐折比對。
2. **逐張 τ̂**：D3 在十折 × 兩序 × 階段 t = 1–4 的每張 test slide，分派任務 τ̂ 與從頭加總版本完全相同
   （不一致張數 = 0）。
3. **分派混淆**：AR 在 reverse、t = 4、十折合計的 4 × 4 混淆矩陣 16 格與 `REPORT_stage10.md:103-106` 完全相同。
4. **AR-bal**：第 7 列的十折數字（同判準 1，對 `REPORT_stage10.md:17`、`:30` 與 `per_fold.json` 第 7 列）、
   逐張 τ̂（同判準 2）、混淆矩陣（對 `REPORT_stage10.md:112-115`）全部相同。
5. **W**：十折 × 兩序 × 兩個分派器 × 階段 t = 1–4，累加版 W 與從頭加總版 W 的最大絕對差 ≤ 1e-10。
6. **測試**：`tests/test_incremental_ridge.py` 三項全部通過，且全體 `pytest`（含 `test_no_banned_deps.py`）通過：
   a. 各階段 W 與從頭加總實作的最大絕對差 ≤ 1e-10；
   b. 以假的 mean_vec 載入器記錄讀取，第 t 階段只讀任務 t 的 train mean_vec；
   c. 同一組任務以不同順序加入，最終 W 相同（最大絕對差 ≤ 1e-10，W 的欄依任務 id 對齊）。
7. **只讀當前任務**：`nc9_incremental.py` 在第 t 階段的讀取紀錄只有任務 t 的 train 快取。

## 另報（不設判準）

- 統計量檔案大小（bytes，實際檔案）與「保留全部舊訓練片 mean_vec」的大小對照（每折 n_train × 512 × 4 bytes，
  另列 NC-8 train 快取檔的實際檔案大小）。
- 唯讀確認：(a) I6 任務分類頭的訓練在第 t 階段是否只讀任務 t 的資料（呼叫鏈與行號）；(b) I6 每層形狀與
  1,033 參數的計算，判斷它是獨立的小頭還是共享權重上的低秩增量；(c) top-64 的 4 輪 × 16 每輪之間更新了什麼。
- 唯讀確認：本 repo 十折 split 檔與 pathselect repo 所用 split 檔的 sha256 與逐折病人名單比對。
