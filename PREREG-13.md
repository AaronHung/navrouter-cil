# PREREG-13 — NC-13 預先註冊（修正頭的貢獻：修正量設為 0，只重新推論）

登記時間：2026-09-29，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro，10 核、16 GB），CPU、`torch.set_num_threads(8)`；四個 arm 與附加分析在同一次執行中完成。
另一個 session 可能同時在本機執行 3.4，耗時僅供參考。
分支：`ws3-head-contrib`（worktree `~/research/01_navrouter-cil-ws35`，自 main @ ee2be2c）。報告編號：REPORT_stage15.md。

## 問題

第二站每個 patch 的分數 = s0（zero-shot 分數）＋ 修正量 g（修正頭，程式代號 I6，r = 2，每任務 1,033 參數）。
現行 task-known 0.9340 ± 0.0202、task-inferred 0.9128 ± 0.0258（REPORT_stage10.md:18、:16）。
報告中沒有「g = 0」在同一設定下的十折結果（NC-1 的 zero-shot top-64 是一次挑 64 個、以未 z-score 的最大 cosine 排序，
task-known 0.9005，REPORT_stage1-3.md:38，挑法不同）。本實驗量測修正量在同一推論流程中的貢獻。

## 設定（不訓練、不改既有腳本與 outputs）

- 修正頭：既有 I6(r = 2) 權重 `outputs/navcil/mac/i6/r2/fold{f}_{task}.pt`（主工作樹，唯讀，以參數指向）。
- 快取：NC-8 同一批 `outputs/navcil/mac/cache/nc8_fold{f}_{train,test}_{task}.pt`（主工作樹，唯讀）：test 的 sid 對齊、
  test mean_vec 供分派器、train mean_vec 供 AR 累加。特徵檔：`can_dataset/<task>/feats-l1-s256_CONCH/`（每折 test 讀一次）。
- 分派器：AR（γ = 1e-3），一律用 `selector/incremental_ridge.py`（與 NC-8 逐位元相同，REPORT_stage11），第 t 階段只讀任務 t 的 train mean_vec。
- CIL 定義同 `scripts/nc8_report.py`（`nc5_report.cil_full`、`nc2_report.hard`）：task-known = oracle 分派、task-inferred = AR 分派，
  主指標為 t = 4 的 ACC（四任務等權平均），十折 mean ± sd（`selector/cil_eval.mean_sd`，樣本標準差），reverse 與 paper 兩序分列。
- 新程式寫在新檔：`scripts/nc13_head_contrib.py`（推論與附加分析）、`scripts/nc13_report.py`（報告）。輸出：`outputs/navcil/mac/nc13/`。
- 長時間執行以 `scripts/run_stage.sh` 在 tmux 中進行（每 15 分鐘心跳）；每折完成寫 `.done`；逐張記錄 `t_read_s`、`t_compute_s`。

## 四個 arm（只改推論）

每個 arm 對每張 test slide 用四個任務的修正頭各算一次證據，選出的 patch 等權平均並 L2 正規化後對 8 類文字取 cosine（同 NC-8）。

| arm | 分數 | 挑法 |
|---|---|---|
| (a) 現行 | s0 + g | 4 輪 × 16，去重懲罰 λ* = 1.5（`selector/cil_ops.py:39-45` `four_round`） |
| (b) g = 0 | s0 | 4 輪 × 16，去重懲罰 λ* = 1.5 |
| (c) | s0 + g | 一次挑 top-64（`selector/cil_ops.py:48-49` `one_shot`） |
| (d) g = 0 | s0 | 一次挑 top-64 |

s0 與 g 取自 `I6Expert.parts`（`selector/i6_expert.py:35-45`）；g = 0 即只用 s0（等同 w2 = b2 = 0）。

## 輸出

- 每個 arm：task-known、task-inferred 的十折平均、標準差、每折值（兩序）；另存逐階段 ACC、Masked ACC、Forgetting、BWT。
- 主表：arm｜task-known｜task-inferred｜與 (a) 的差｜高於／低於 (a) 的折數。
- 配對：(a) vs (b)、(c) vs (d)，task-known 與 task-inferred、兩序：Wilcoxon signed-rank 雙尾（`scipy.stats.wilcoxon`，預設
  zero_method = "wilcox"）與 paired t 雙尾（`scipy.stats.ttest_rel`）的 p 值，以及勝／平／負折數。不做多重比較校正，另列檢定次數。

## 方向性預測（事先登記；無論結果如何都照實報告）

「加修正量的 task-known 高於 g = 0，十折中至少 8 折」：即 (a) 的 task-known 逐折高於 (b)，十折中至少 8 折（reverse 序；
t = 4 時兩序的 task-known 逐折相同，paper 序一併列出）。(c) 對 (d) 同樣報告勝負折數，但不列為預測。

## 附加分析（同一次執行）

1. 修正量的影響：以自家任務的修正頭，逐張計算 (a) 與 (b)、(c) 與 (d) 所選 64 個 patch 的重疊數（另列 (a)–(c)、(b)–(d)），
   十折平均（每折先對該折全部 test slides 平均）。
2. 瓶頸是否退化：fold 1、每個任務的修正頭，在該任務全部 test slides 的所有 patch 上計算 h = A u + b1 的兩個維度 h1、h2：
   Pearson 相關係數，以及 h1、h2 各自落在 GELU 線性區（值 > 3）的比例（另報兩者同時 > 3 的比例）。

## 一致性檢查（不符即停，不繼續、不改判準）

1. (a) 的 8 類 cosine 與 NC-8 快取 `I6_cos8` 逐張比對（報最大絕對差，須 ≤ 1e-5）。
2. (a) 的 task-known 兩序 = 0.9340 ± 0.0202（REPORT_stage10.md:18、:31），task-inferred 兩序 = 0.9128 ± 0.0258
   （REPORT_stage10.md:16、:29）；逐折與 `nc8/per_fold.json` 第 8、6 列在小數點後四位相同。
3. (c) 的 task-known 兩序 = 0.9372、task-inferred 兩序 = 0.9160（REPORT_stage13.md:215，分支 `ws3-ablation` @ bf488ff，尚未合併；
   以 `git show` 唯讀讀取）；逐折與同一 commit 的 `nc11/select/result.json`（`oneshot64`）在小數點後四位相同。

## 執行時間估算

每折 test slides 讀一次（十折共 2,835 張），每張 4 個修正頭 × 4 個 arm 的選取與 cosine；參考 REPORT_stage13 選取方式 ablation
（2 個 arm）的耗時，估 15–30 分鐘；與 3.4 同時執行時估上限 1 小時。報告 < 1 分鐘。
