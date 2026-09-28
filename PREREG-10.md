# PREREG-10 — NC-10 預先註冊（分派器同折、同任務分類頭比較：R0、text-organ、R3、AR、oracle ＋ I6）

登記時間：2026-09-28，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），CPU；本輪所有數字來自同一台、同一次執行（`scripts/nc10_dispatch_compare.py`）。
分支：`ws2-dispatch-compare`（worktree `~/research/01_navrouter-cil-ws2`）。全部只做推論，不訓練任何東西。

## 目的

兩站架構：第一站由分派器以整片 mean_vec 判斷任務，第二站由該任務的 I6(r=2) 任務分類頭以四輪 top-64 證據判亞型（Hard）。
現有結果中 R0（T-Hard 規則）只搭配過全參數 L0 頭（REPORT_stage6.md:94），無法與 AR ＋ I6（CIL 0.9128，REPORT_stage10.md:16）
公平比較。本輪讓所有分派器接同一批 I6 頭、同一組十折、兩種順序，在同一次執行中比較分派正確率與全程 CIL。

## 比較對象（全部接 I6(r=2)，Hard）

| 名稱 | 任務 key／統計量 | 定義位置 |
|---|---|---|
| oracle | 直接給真實任務（上限） | `selector/router.py:20-21`；`scripts/nc8_report.py:125`（row 8） |
| R0（T-Hard 規則） | 每任務 = 該任務兩個亞型文字 embedding 的平均再 L2 正規化；分數 = mean_vec · key，取 argmax；零訓練 | `selector/router.py:56-58`、`:87-88` |
| text-organ | 每任務 = 器官名稱 3 個模板的文字 embedding（`cache/text/organ_keys.pt`）；分數 = mean_vec · key；零訓練 | `selector/router.py:24-25`；`scripts/nc1_organ_keys.py:23-25` |
| R3(k=8) | 每任務 train mean_vec 的 KMeans(k=8, n_init=10, random_state=0) 中心；分數 = 對各中心的最大 cosine | `scripts/nc4_report.py:121-124`；`scripts/nc8_report.py:66-69`、`:92-94` |
| AR（γ = 1e-3） | 累加統計量 A、b_j（`selector/incremental_ridge.py`，與 NC-8 逐位元相同，見 REPORT_stage11） | `scripts/nc8_report.py:71-97`；`selector/incremental_ridge.py` |

所有分派器在第 t 階段只比較前 t 個已見任務的 key（R0、text-organ 取 key 的已見列；R3 只取已見任務的中心；AR 的 W 只含已見任務的欄）。
資料：NC-8 同一批快取（`cache/nc8_fold{f}_{train,test}_{task}.pt`：test 的 mean_vec 與 `I6_cos8`、`L0_cos8`，train 的 mean_vec），
以參數 `--cache-dir` 指向主工作樹的 `outputs/navcil/mac/cache/`（gitignored，唯讀）。文字特徵：`cache/text/f_txt_<task>.pt`、`organ_keys.pt`。

## 指標（每分派器 × 每折 × 每序 × 每階段 t = 1–4）

1. 分派正確率 micro = 已見任務全部 test slides 中 τ̂ = 真實任務的比例；macro = 已見任務各自比例的等權平均。
2. 全程 CIL（task-inferred）：`scripts/nc5_report.py:43-59` `cil_full` 的 acc_t（第 t 階段已見任務各自正確率的等權平均），
   第二站用 `scripts/nc2_report.py:89-96` `hard`，與 `nc8_report.py` 相同。主表取 t = 4（ACC）；另存 Masked ACC、Forgetting、BWT。
3. 4 × 4 分派混淆矩陣：t = 4、十折合計、每序各一。肺→食道 = 真實 lung 分派到 esca 的張數；食道→肺反之。
4. 主表：t = 4、十折 mean ± sd，兩序分列。

## 方向性預測（事前登記，無論結果如何都照實報告）

**AR 的分派正確率高於 R0，十折中至少 8 折。** 操作定義：t = 4 的分派正確率 micro，逐折比較 AR − R0 > 0（相等不算勝）；
兩序各自計算，兩序都 ≥ 8/10 才算預測成立。另報 macro 的逐折勝數（不作為判定）。

## 乘法框架核對（另報，不設判準）

每分派器、每折、每序（t = 4）：(a) 分派 macro × oracle CIL；(b) 分派 micro × oracle CIL；(c) Σ_p TP_p × WP_p ／ 4，
WP_p = oracle ＋ I6 在任務 p 的正確率。十折平均後與實測 CIL 並列，並列出差值。
Hard 之下分派錯的片一定判錯，所以 (c) 在「分派正確與第二站正確在同一任務內獨立」時等於實測；差值反映兩者的相關。

## 一致性檢查（任一不符即停下回報，不繼續、不改判準）

1. AR ＋ I6：兩序 t = 4 ACC 十折 mean ± sd 與 `REPORT_stage10.md:16`、`:29`（0.9128 ± 0.0258）相同；每折與 `nc8/per_fold.json` 第 6 列
   在小數點後四位相同。
2. oracle ＋ I6：兩序與 `REPORT_stage10.md:18`、`:31`（0.9340 ± 0.0202）相同；每折與 `nc8/per_fold.json` 第 8 列在小數點後四位相同。
3. R3：reverse、t = 4、十折合計的 4 × 4 混淆與 `REPORT_stage10.md:94-97` 16 格相同（該表搭配 L1 v2，但分派結果與第二站無關）。
4. R0：t = 4 分派正確率十折平均 micro 0.9288、macro 0.9154，與 `REPORT_stage6.md:94` 在小數點後四位相同（兩序皆檢查）。

另報（不作為停止條件）：R0 每任務 TP 與 ESCA↔Lung 張數對 `REPORT_stage6.md:94`；text-organ 的 TP micro／macro 對 `REPORT_stage6.md:92`；
R0 ＋ L0 的 reverse CIL 對 `REPORT_stage6.md:94` 的 0.8617。

## 執行時間估算

NC-8 報告（8 列 × 十折 × 兩序）實測 8 秒（REPORT_stage10.md:136），NC-9 含 pytest 實測 14–15 秒（`logs/nc9_incremental.log`）。
本輪 5 個分派器 × 十折 × 兩序 × 4 階段，外加 R3 的 40 次 KMeans。估計單獨執行約 30 秒；另一個 session 正同時在 Mac 上跑 WS3
（開跑前 load average 約 7），估計上限 2 分鐘。只讀快取、不讀特徵檔。以 `scripts/run_stage.sh` 在 tmux 中執行，實際耗時寫進報告。

## 輸出

`outputs/navcil/mac/nc10/`（`per_fold.json`、`confusion.json`、`result.json`）與 `outputs/navcil/mac/REPORT_stage12.md`。
不改既有腳本與既有 outputs。
