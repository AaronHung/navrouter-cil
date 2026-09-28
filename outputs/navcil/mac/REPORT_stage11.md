# REPORT — NC-9：AR 分派器改為只存累加統計量（Mac CPU，十折兩序）

機器：mac（Apple M1 Pro）、CPU、`torch.set_num_threads(8)`、torch 2.11.0。所有數字來自同一台、同一批。判準與定義見 `PREREG-9.md`（commit 33f76a4）。分派器 = `selector/incremental_ridge.py` 的 `IncrementalRidge`；第 t 階段只讀任務 t 的 train mean_vec，統計量（A、b_1…b_t）經由檔案在階段之間傳遞。證據與任務分類頭沿用 NC-8 同一批快取（`cache/nc8_fold*_*.pt`）與 I6(r=2) 權重；比對基準為 `REPORT_stage10.md`、`nc8/per_fold.json` 與以唯讀方式呼叫 `nc8_report.B8.W`（從頭加總）重算的參考值。

## 判準總表

| 判準 | 內容 | 結果 |
|---|---|---|
| 1 | D3 十折 ACC／Masked／Forgetting／BWT（兩序）與 NC-8 小數點後四位相同 | 通過 |
| 2 | D3 逐張 τ̂（十折 × 兩序 × t = 1–4）與從頭加總相同 | 通過 |
| 3 | AR 分派混淆（reverse、t = 4、十折合計）與 REPORT_stage10.md:103-106 相同 | 通過 |
| 4 | AR-bal＋I6：十折數字、逐張 τ̂、混淆（REPORT_stage10.md:112-115）相同 | 通過 |
| 5 | W 與從頭加總的最大絕對差 ≤ 1e-10（全部 160 組） | 通過（最大 0.00e+00；逐位元相同 160/160） |
| 6 | `tests/test_incremental_ridge.py` 與全體 pytest | 通過（79 passed in 7.46s） |
| 7 | 第 t 階段只讀任務 t 的 train 快取（`nc9/reads.json`，160 筆） | 通過 |

## T1 十折平均 ± 標準差（test、t = 4）：REPORT_stage10 對 NC-9

| 列 | 序 | 來源 | ACC | Masked ACC | Forgetting | BWT | TP esca／rcc／brca／lung | WP esca／rcc／brca／lung | 相同 |
|---|---|---|---|---|---|---|---|---|---|
| 6 D3：AR＋I6(r=2)（主系統） | reverse | REPORT_stage10.md:16（`nc8.t1.d3.reverse.acc`） | 0.9128 ± 0.0258 | 0.9312 ± 0.0215 | 0.0224 ± 0.0172 | -0.0220 ± 0.0170 | 0.9340／0.9908／0.9874／0.9871 | 0.9608／0.9611／0.9181／0.8958 | |
| | | NC-9（`nc9.t1.d3.reverse.acc`） | 0.9128 ± 0.0258 | 0.9312 ± 0.0215 | 0.0224 ± 0.0172 | -0.0220 ± 0.0170 | 0.9340／0.9908／0.9874／0.9871 | 0.9608／0.9611／0.9181／0.8958 | 通過 |
| 6 D3：AR＋I6(r=2)（主系統） | paper | REPORT_stage10.md:29（`nc8.t1.d3.paper.acc`） | 0.9128 ± 0.0258 | 0.9312 ± 0.0215 | 0.0041 ± 0.0055 | -0.0034 ± 0.0049 | 0.9340／0.9908／0.9874／0.9871 | 0.9608／0.9611／0.9181／0.8958 | |
| | | NC-9（`nc9.t1.d3.paper.acc`） | 0.9128 ± 0.0258 | 0.9312 ± 0.0215 | 0.0041 ± 0.0055 | -0.0034 ± 0.0049 | 0.9340／0.9908／0.9874／0.9871 | 0.9608／0.9611／0.9181／0.8958 | 通過 |
| 7 AR-bal（γ = 1e-4）＋I6(r=2) | reverse | REPORT_stage10.md:17（`nc8.t1.arbal_i6.reverse.acc`） | 0.9150 ± 0.0212 | 0.9327 ± 0.0196 | 0.0064 ± 0.0059 | -0.0051 ± 0.0064 | 0.9875／0.9868／0.9768／0.9603 | 0.9608／0.9611／0.9181／0.8958 | |
| | | NC-9（`nc9.t1.arbal_i6.reverse.acc`） | 0.9150 ± 0.0212 | 0.9327 ± 0.0196 | 0.0064 ± 0.0059 | -0.0051 ± 0.0064 | 0.9875／0.9868／0.9768／0.9603 | 0.9608／0.9611／0.9181／0.8958 | 通過 |
| 7 AR-bal（γ = 1e-4）＋I6(r=2) | paper | REPORT_stage10.md:30（`nc8.t1.arbal_i6.paper.acc`） | 0.9150 ± 0.0212 | 0.9327 ± 0.0196 | 0.0121 ± 0.0066 | -0.0118 ± 0.0065 | 0.9875／0.9868／0.9768／0.9603 | 0.9608／0.9611／0.9181／0.8958 | |
| | | NC-9（`nc9.t1.arbal_i6.paper.acc`） | 0.9150 ± 0.0212 | 0.9327 ± 0.0196 | 0.0121 ± 0.0066 | -0.0118 ± 0.0065 | 0.9875／0.9868／0.9768／0.9603 | 0.9608／0.9611／0.9181／0.8958 | 通過 |

## T2 每折 ACC（test、t = 4）

行號：NC-8 值在 `outputs/navcil/mac/nc8/per_fold.json`，NC-9 值在 `outputs/navcil/mac/nc9/per_fold.json`。「四位」= 四捨五入到小數點後四位相同；「位元」= 兩個 float 完全相等。Masked ACC、Forgetting、BWT 與逐階段 ACC、每任務 TP／WP 的逐折比對見 `nc9/result.json` 的 `compare_rows`，結果列在表下。

**6 D3：AR＋I6(r=2)（主系統），reverse**

| 折 | NC-8 ACC | per_fold.json 行 | NC-9 ACC | nc9/per_fold.json 行 | 四位 | 位元 |
|---|---|---|---|---|---|---|
| 1 | 0.9117 | 8036 | 0.9117 | 6 | 通過 | 是 |
| 2 | 0.9372 | 8118 | 0.9372 | 88 | 通過 | 是 |
| 3 | 0.9414 | 8200 | 0.9414 | 170 | 通過 | 是 |
| 4 | 0.8914 | 8282 | 0.8914 | 252 | 通過 | 是 |
| 5 | 0.9583 | 8364 | 0.9583 | 334 | 通過 | 是 |
| 6 | 0.9164 | 8446 | 0.9164 | 416 | 通過 | 是 |
| 7 | 0.8901 | 8528 | 0.8901 | 498 | 通過 | 是 |
| 8 | 0.8772 | 8610 | 0.8772 | 580 | 通過 | 是 |
| 9 | 0.8987 | 8692 | 0.8987 | 662 | 通過 | 是 |
| 10 | 0.9056 | 8774 | 0.9056 | 744 | 通過 | 是 |

四位相同／位元相同的折數（共 10）：acc 10/10；masked 10/10；forgetting 10/10；bwt 10/10。逐項位元相同的折數：acc_t 10；masked_t 10；tp_task 10；wp_task 10。

**6 D3：AR＋I6(r=2)（主系統），paper**

| 折 | NC-8 ACC | per_fold.json 行 | NC-9 ACC | nc9/per_fold.json 行 | 四位 | 位元 |
|---|---|---|---|---|---|---|
| 1 | 0.9117 | 8858 | 0.9117 | 828 | 通過 | 是 |
| 2 | 0.9372 | 8940 | 0.9372 | 910 | 通過 | 是 |
| 3 | 0.9414 | 9022 | 0.9414 | 992 | 通過 | 是 |
| 4 | 0.8914 | 9104 | 0.8914 | 1074 | 通過 | 是 |
| 5 | 0.9583 | 9186 | 0.9583 | 1156 | 通過 | 是 |
| 6 | 0.9164 | 9268 | 0.9164 | 1238 | 通過 | 是 |
| 7 | 0.8901 | 9350 | 0.8901 | 1320 | 通過 | 是 |
| 8 | 0.8772 | 9432 | 0.8772 | 1402 | 通過 | 是 |
| 9 | 0.8987 | 9514 | 0.8987 | 1484 | 通過 | 是 |
| 10 | 0.9056 | 9596 | 0.9056 | 1566 | 通過 | 是 |

四位相同／位元相同的折數（共 10）：acc 10/10；masked 10/10；forgetting 10/10；bwt 10/10。逐項位元相同的折數：acc_t 10；masked_t 10；tp_task 10；wp_task 10。

**7 AR-bal（γ = 1e-4）＋I6(r=2)，reverse**

| 折 | NC-8 ACC | per_fold.json 行 | NC-9 ACC | nc9/per_fold.json 行 | 四位 | 位元 |
|---|---|---|---|---|---|---|
| 1 | 0.9097 | 9682 | 0.9097 | 1652 | 通過 | 是 |
| 2 | 0.9434 | 9764 | 0.9434 | 1734 | 通過 | 是 |
| 3 | 0.9269 | 9846 | 0.9269 | 1816 | 通過 | 是 |
| 4 | 0.9048 | 9928 | 0.9048 | 1898 | 通過 | 是 |
| 5 | 0.9437 | 10010 | 0.9437 | 1980 | 通過 | 是 |
| 6 | 0.9107 | 10092 | 0.9107 | 2062 | 通過 | 是 |
| 7 | 0.9131 | 10174 | 0.9131 | 2144 | 通過 | 是 |
| 8 | 0.8692 | 10256 | 0.8692 | 2226 | 通過 | 是 |
| 9 | 0.9112 | 10338 | 0.9112 | 2308 | 通過 | 是 |
| 10 | 0.9173 | 10420 | 0.9173 | 2390 | 通過 | 是 |

四位相同／位元相同的折數（共 10）：acc 10/10；masked 10/10；forgetting 10/10；bwt 10/10。逐項位元相同的折數：acc_t 10；masked_t 10；tp_task 10；wp_task 10。

**7 AR-bal（γ = 1e-4）＋I6(r=2)，paper**

| 折 | NC-8 ACC | per_fold.json 行 | NC-9 ACC | nc9/per_fold.json 行 | 四位 | 位元 |
|---|---|---|---|---|---|---|
| 1 | 0.9097 | 10504 | 0.9097 | 2474 | 通過 | 是 |
| 2 | 0.9434 | 10586 | 0.9434 | 2556 | 通過 | 是 |
| 3 | 0.9269 | 10668 | 0.9269 | 2638 | 通過 | 是 |
| 4 | 0.9048 | 10750 | 0.9048 | 2720 | 通過 | 是 |
| 5 | 0.9437 | 10832 | 0.9437 | 2802 | 通過 | 是 |
| 6 | 0.9107 | 10914 | 0.9107 | 2884 | 通過 | 是 |
| 7 | 0.9131 | 10996 | 0.9131 | 2966 | 通過 | 是 |
| 8 | 0.8692 | 11078 | 0.8692 | 3048 | 通過 | 是 |
| 9 | 0.9112 | 11160 | 0.9112 | 3130 | 通過 | 是 |
| 10 | 0.9173 | 11242 | 0.9173 | 3212 | 通過 | 是 |

四位相同／位元相同的折數（共 10）：acc 10/10；masked 10/10；forgetting 10/10；bwt 10/10。逐項位元相同的折數：acc_t 10；masked_t 10；tp_task 10；wp_task 10。

## T3 逐張分派結果 τ̂（累加版對從頭加總版）

| 分派器 | 序 | t = 1 | t = 2 | t = 3 | t = 4 | 比對張數 | 不一致張數 |
|---|---|---|---|---|---|---|---|
| AR | reverse | 0 | 0 | 0 | 0 | 5,773 | 0 |
| AR | paper | 0 | 0 | 0 | 0 | 8,402 | 0 |
| AR-bal | reverse | 0 | 0 | 0 | 0 | 5,773 | 0 |
| AR-bal | paper | 0 | 0 | 0 | 0 | 8,402 | 0 |

比對張數 = 十折 × 階段 t = 1–4 的已見任務 test slides 累計（兩序在 t < 4 的已見任務不同，故張數不同）。每張 test slide 的 τ̂ 存於 `nc9/tau_hat.pt`（鍵 `折|序|分派器|t`，slides 依該序的任務順序串接）。

## T4 分派混淆矩陣（test、t = 4、十折合計；列 = 真實、欄 = 分派）

**AR**：REPORT_stage10.md:103-106（`router.nc8.confusion.ar.*`）對 NC-9（reverse；`nc9.conf.ar.*`）——通過

| 真實 \ 分派 | esca（NC-8／NC-9） | rcc（NC-8／NC-9） | brca（NC-8／NC-9） | lung（NC-8／NC-9） |
|---|---|---|---|---|
| esca | 140／140 | 1／1 | 3／3 | 6／6 |
| rcc | 1／1 | 761／761 | 4／4 | 2／2 |
| brca | 1／1 | 1／1 | 940／940 | 10／10 |
| lung | 2／2 | 2／2 | 8／8 | 953／953 |

**AR-bal**：REPORT_stage10.md:112-115（`router.nc8.confusion.arbal.*`）對 NC-9（reverse；`nc9.conf.arbal.*`）——通過

| 真實 \ 分派 | esca（NC-8／NC-9） | rcc（NC-8／NC-9） | brca（NC-8／NC-9） | lung（NC-8／NC-9） |
|---|---|---|---|---|
| esca | 148／148 | 1／1 | 0／0 | 1／1 |
| rcc | 4／4 | 758／758 | 4／4 | 2／2 |
| brca | 10／10 | 3／3 | 930／930 | 9／9 |
| lung | 26／26 | 5／5 | 7／7 | 927／927 |

AR 的主要錯分：肺→食道 2、食道→肺 6、乳→肺 10、肺→乳 8、食道→乳 3。paper 序的 t = 4 混淆矩陣（另報）：AR [[140, 1, 3, 6], [1, 761, 4, 2], [1, 1, 940, 10], [2, 2, 8, 953]]；AR-bal [[148, 1, 0, 1], [4, 758, 4, 2], [10, 3, 930, 9], [26, 5, 7, 927]]。

## T5 統計量檔案大小與「保留全部舊訓練片 mean_vec」的對照（bytes）

統計量檔 = `nc9/state/fold{f}_{序}_{分派器}.pt`（`torch.save`：A 513 × 513 float64 ＋ b 4 × 513 float64 ＋ 任務 id、γ 等中繼資料）。mean_vec 原始大小 = 已見 train slides 數 × 512 × 4（float32，與快取相同精度）；另列 NC-8 四個 train 快取檔的實際大小（含 sid、label、計時欄位）。

| 折 | train slides（esca／rcc／brca／lung，合計） | 統計量檔 t = 1 | t = 4 | mean_vec 原始 t = 4 | 倍數（mean_vec ÷ 統計量） | NC-8 train 快取檔合計 |
|---|---|---|---|---|---|---|
| 1 | 120／616／763／774（2,273） | 2,111,469 | 2,123,757 | 4,655,104 | 2.19 | 4,859,386 |
| 2 | 120／613／764／761（2,258） | 2,111,469 | 2,123,757 | 4,624,384 | 2.18 | 4,827,450 |
| 3 | 120／617／763／765（2,265） | 2,111,469 | 2,123,757 | 4,638,720 | 2.18 | 4,842,362 |
| 4 | 119／613／762／756（2,250） | 2,111,469 | 2,123,757 | 4,608,000 | 2.17 | 4,810,362 |
| 5 | 120／619／763／764（2,266） | 2,111,469 | 2,123,757 | 4,640,768 | 2.19 | 4,844,474 |
| 6 | 119／616／758／759（2,252） | 2,111,469 | 2,123,757 | 4,612,096 | 2.17 | 4,814,714 |
| 7 | 120／615／760／774（2,269） | 2,111,469 | 2,123,757 | 4,646,912 | 2.19 | 4,850,938 |
| 8 | 120／617／762／777（2,276） | 2,111,469 | 2,123,757 | 4,661,248 | 2.19 | 4,865,850 |
| 9 | 121／612／760／762（2,255） | 2,111,469 | 2,123,757 | 4,618,240 | 2.17 | 4,820,986 |
| 10 | 121／618／763／771（2,273） | 2,111,477 | 2,123,765 | 4,655,104 | 2.19 | 4,859,426 |

全部 160 個（折 × 序 × 分派器 × 階段）統計量檔大小的範圍：2,111,453–2,123,797 bytes；只隨已見任務數 t 增加 513 × 8 = 4,104 bytes／任務，與 slide 數無關。mean_vec ÷ 統計量（t = 4）十折範圍 2.17–2.19。統計量以 float64 存（AMENDMENT-2）；PREREG-8 操作定義 9 的 fp32 計法為 A 1,052,676 ＋ 4 × 2,052 = 1,060,884 bytes。

## T6 其他數字

| 項目 | 值 |
|---|---|
| W 最大絕對差（累加 − 從頭加總；AR） | 0.00e+00 |
| W 最大絕對差（累加 − 從頭加總；AR-bal） | 0.00e+00 |
| 真實資料 t = 4：reverse 與 paper 的 W 最大絕對差（欄依任務對齊；AR／AR-bal） | 1.12e-10／9.55e-13 |
| 真實資料 t = 4：reverse 與 paper 的 τ̂ 不一致張數（依任務對齊後逐張；十折合計；AR／AR-bal） | 0／0 |
| 每階段讀檔秒數（十折兩序兩分派器平均） | 0.0026 |
| 每階段 add_task＋solve 秒數（平均） | 0.0052 |
| 本腳本總耗時（秒） | 12 |

兩序的 W 差不是判準（判準 6c 用合成資料）；它來自 A 以不同任務順序加總的浮點捨入。AR 的最大差超過 1e-10，照實列出。

測試紀錄（判準 6；本腳本以子行程執行 `python -m pytest -v`，結束碼 0）：

```
tests/test_incremental_ridge.py::test_a_matches_from_scratch_every_stage[AR] PASSED [  1%]
tests/test_incremental_ridge.py::test_a_matches_from_scratch_every_stage[AR-bal] PASSED [  2%]
tests/test_incremental_ridge.py::test_b_stage_t_reads_only_task_t[AR] PASSED [  3%]
tests/test_incremental_ridge.py::test_b_stage_t_reads_only_task_t[AR-bal] PASSED [  5%]
tests/test_incremental_ridge.py::test_c_order_independent[AR] PASSED     [  6%]
tests/test_incremental_ridge.py::test_c_order_independent[AR-bal] PASSED [  7%]
tests/test_incremental_ridge.py::test_a_matches_nc8_report_fold1[AR] PASSED [  8%]
tests/test_incremental_ridge.py::test_a_matches_nc8_report_fold1[AR-bal] PASSED [ 10%]
tests/test_no_banned_deps.py::test_file_has_no_banned_token[selector/incremental_ridge.py] PASSED [ 40%]
============================== 79 passed in 7.46s ==============================
```

## R1 任務分類頭（I6）的訓練資料範圍（唯讀）

呼叫鏈（第 t 階段訓練任務 t 的任務分類頭）：
1. `scripts/nc7_i6.py:146-150`：`for r → for fold → for task: train_one(ctx, root, r, fold, task)`；每（r、折、任務）各訓練一次，與任務順序無關（`nc7_i6.py:2`）。
2. `scripts/nc7_i6.py:47-50`：`p = ctx.tasks.index(task)`；`ds, shift = ctx.ds(fold, task, "train")`；`torch.manual_seed(42)`；新建 `I6Expert(r)`（不載入任何其他任務的權重）。
3. `scripts/nc1_pipeline.py:78-80` `Ctx.ds` → `selector/evaluate.py:29-46` `slide_dataset`：只讀該任務自己的 `<task>/datasplit/fold_{f}.npz` 的 train 病人名單（`evaluate.py:40-41`），特徵在 `read_slide` 時才讀（`nc7_i6.py:53-57`）。
4. `scripts/nc7_i6.py:60-61` → `selector/cil_ops.py:65-106` `train_selector`：迭代只來自上述 slides；文字只用該任務的 2 類 `ctx.f_task(p)`（`nc1_pipeline.py:75-76`）與 CONCH 的 logit_scale；optimizer 只含該頭的參數（`cil_ops.py:78-79`）。
5. 推論：`scripts/nc8_batch.py:51-55` 逐任務載入 `i6/r2/fold{f}_{task}.pt`。

結論：每個任務分類頭的訓練只讀任務 t 自己的 train slides（外加該任務 2 類的文字特徵），不讀其他任務的 slides、mean_vec 或權重，所以任務分類頭這一側沒有 replay；搭配本輪的累加分派器，「訓練時不重讀舊任務資料」對整個系統成立。需要另外註明的兩點（不是 replay，但屬跨任務資訊）：(i) λ\*=1.5 由 fold 1 四個任務的 validation 一次選定（`nc1_pipeline.py:263-285`）；(ii) r\*=2 由十折四任務的 validation 選定（`nc7_report.py:59-71`）。兩者都是事前一次性的超參數選擇，不在逐階段流程內。另外程式實際上是一次把四個任務都訓練完（步驟 1 的迴圈），不是逐階段觸發；因為各頭之間沒有相依，結果與逐階段訓練相同。

## R2 I6 的結構與 1,033 參數（唯讀）

| 層／張量 | 形狀（r = 2） | 參數數 | 位置 |
|---|---|---|---|
| s0 = zscore(text_nav_feats(Z, f_task)[:, 0])：對該任務 2 類文字的最大 cosine，slide 內 z-score | [n] | 0（不訓練） | `selector/i6_expert.py:37-38`；`selector/flat_selector.py:19-33` |
| u = [Z ; text_nav_feats]（512 ＋ 2：最大 cosine、文字相似度分布熵） | [n, 514] | 0 | `i6_expert.py:39` |
| A | [2, 514] | 1,028 | `i6_expert.py:29-30`（kaiming_uniform a = √5） |
| b1 | [2] | 2 | `i6_expert.py:31`（初始 0） |
| w2 | [2] | 2 | `i6_expert.py:32`（初始 0） |
| b2 | 純量 | 1 | `i6_expert.py:33`（初始 0） |
| 輸出 score = s0 ＋ w2ᵀ GELU(A u ＋ b1) ＋ b2 | [n] | 合計 1,033 = 514r ＋ r ＋ r ＋ 1 = 516r ＋ 1 | `i6_expert.py:44`、`:48-49`；計數 `:51-52`、`scripts/nc7_report.py:66` |

判斷：**獨立的小頭**。每個任務各自新建一個 `I6Expert`（`nc7_i6.py:50`），沒有任何共享的可訓練或凍結權重矩陣；「底座」s0 是無參數的 zero-shot 文字相似度分數，不是權重。A 的「低秩」指隱藏寬度 r = 2 的瓶頸，不是 ΔW = BA 疊在共享 W 上。對照：`selector/lora_expert.py:21-27`、`:46-49` 的 `LowRankExpert`（D1 用的 L1 v2）才是 W1 = W1_base ＋ B·A，W1_base（256 × 514）凍結並取自該序第一個任務的 L0。初始時 w2 = b2 = 0，所以訓練開始時 score = s0（`i6_expert.py:6`）。fp32 儲存 1,033 × 4 = 4,132 bytes／任務（PREREG-8 操作定義 9）。

## R3 top-64 的 4 輪 × 16：每輪之間更新什麼（唯讀）

呼叫點：`scripts/nc8_batch.py:63`、`:72` → `selector/cil_ops.py:39-45` `four_round`（budget 64、step 16、redundancy_weight = λ\* = 1.5、normalize_base = True、redundancy_mode = "maxsim"）→ `selector/multiround.py:134-203` `SequentialBudgetedObserver.observe`。

機制：任務分類頭的分數 s = s0 ＋ g 在進入迴圈前算一次（`nc8_batch.py:72` 的 `m(Z, ctx.f_task(p))`），在迴圈前做一次 z-score（`multiround.py:146-150`），之後不再重算，任務分類頭也不再被呼叫。Zn = 各 patch 的 L2 正規化特徵（`:139`）；max_sim_seen 初始為 0（`:152`）。每一輪：(1) 調整後分數 adj = z(s) − λ · max_sim_seen（`:156-163`；第 1 輪 seen 為空，沒有懲罰）；(2) 已選 patch 設為 −∞，不重選（`:164`）；(3) 取 adj 最高的 16 個（`:166-169`）加入已選集合（`:173`）；(4) 更新 max_sim_seen = max(max_sim_seen, 每個 patch 對本輪新選 16 個的最大 cosine)（`:178-180`）。輪與輪之間只更新「已選集合（排除）」與「對已選集合的最大相似度」兩個狀態；沒有查詢向量、沒有文字向量更新、分數本身不重算。confidence_threshold 為 None，迴圈內不呼叫分類（`:184`）；迴圈後的 predict_fn 是回傳 0 的佔位函式（`cil_ops.py:45`），不影響結果。選完的 64 個 patch 等權平均後 L2 正規化（`cil_ops.py:31-36` `mean_norm`），再對 8 類文字取 cosine（`nc8_batch.py:63`）。注意：訓練時不是四輪，而是 one-shot top-64 ＋ softmax(分數) 加權（`cil_ops.py:87-90`）。

## R4 十折切分來源（唯讀）

本 repo：`configs/base.yaml + configs/machine_mac.yaml` → `/Users/aaron/research/can_dataset/{}/datasplit/fold_{}.npz`（四任務 × 十折 = 40 檔）。

| pathselect clone | HEAD | 設定 | 解析後路徑 | 同一實體檔 | sha256 相同 | 病人名單（train／val／test）逐折相同 | 不同檔數 |
|---|---|---|---|---|---|---|---|
| `/Users/aaron/research/02_pathselect` | 2435d0d | `configs/pathselect.yaml`（`/Users/aaron/research/can_dataset/{}/datasplit/fold_{}.npz`） | 同左 | 40/40 | 40/40 | 40/40 | 0 |
| `/Users/aaron/research/pathselect-paper` | 081bdcd | `configs/pathselect.yaml`（`/Users/aaron/research/can_dataset/{}/datasplit/fold_{}.npz`） | 同左 | 40/40 | 40/40 | 40/40 | 0 |
| `/Users/aaron/research/pathselect-exp5` | 1529c0b | `configs/pathselect.yaml`（`/Users/aaron/research/can_dataset/{}/datasplit/fold_{}.npz`） | 同左 | 40/40 | 40/40 | 40/40 | 0 |

各檔 sha256 與 train／val／test 人數：

| 檔 | sha256 | train／val／test |
|---|---|---|
| tcga_esca/fold_1 | `28a6b008f14841847c2947b13f27cae91e249ca7640b8c24c5affa91748b280f` | 118／15／15 |
| tcga_esca/fold_2 | `c978cc43e0483912ffaf13a4daf34dee27d9d23ea9e67547dfcd89be3b46849b` | 118／15／15 |
| tcga_esca/fold_3 | `41a60595d526981fbd2292ed69d9ebfee4a2438b5bacc55b96d2526f1268f002` | 118／15／15 |
| tcga_esca/fold_4 | `d374634a92959213c22752d603f7c9f3d689c14c82fa68854cbc0b17f4e01d0c` | 118／15／15 |
| tcga_esca/fold_5 | `5d851d23a8260d36e0ba502d1321f408990bc5adb389ed6af31639b70582ddf1` | 118／15／15 |
| tcga_esca/fold_6 | `f289f2112571c46c9f7765dc386440f40c76c5441f10ad52eed6e102a113b3e3` | 118／15／15 |
| tcga_esca/fold_7 | `4bc06bda6ec5f9819081b2f1b36d85629a2e47561a44f4ea1a1be36cffbbee5e` | 118／15／15 |
| tcga_esca/fold_8 | `41d4d47fc574b1856544fcb213675842f45b7e6ec1a56c188cc3759f5e84df9b` | 118／15／15 |
| tcga_esca/fold_9 | `1457478d09db999af00e59cc31958d43813bd814ea1ad152122d3873f6de50a0` | 119／15／14 |
| tcga_esca/fold_10 | `eb3d0b6b65fb761fb710567b7312a2b17146a951132d47e291d6523e86f29dd2` | 119／15／14 |
| tcga_rcc/fold_1 | `1d9d2392201d7dc7936c999dcec778fcda739ca1092564cf0977594c560aed3c` | 590／74／74 |
| tcga_rcc/fold_2 | `ad395a006f10a8c7977233fe62366e9c4509c232256a2b4839a219a3817cf1e5` | 590／74／74 |
| tcga_rcc/fold_3 | `65d6d201a7b5aeab69be8f3a70efd0072eda8cfddd1b3161b1e622a6a0d64da4` | 590／74／74 |
| tcga_rcc/fold_4 | `174bb1aae0e95e1efb1b83ca83df59bd6e836e80a903775079be4d26a00a6507` | 590／74／74 |
| tcga_rcc/fold_5 | `e289bb712a2a510409478676a48b5fbaf83766c96a123a46dcf7f332d0df77a9` | 590／74／74 |
| tcga_rcc/fold_6 | `bd2734771f58e36ff4673d682c36c2d27b2e03ae631b1ada3796443d989ad4aa` | 590／74／74 |
| tcga_rcc/fold_7 | `6124f4bc830c852f175b3b22833ca19402740f10a197411273e6807460cacdea` | 590／74／74 |
| tcga_rcc/fold_8 | `c4f6f4eb101bd397a399e4d9556864afd14ba442eccae751da01262e84579565` | 590／74／74 |
| tcga_rcc/fold_9 | `e5d351125d988c3d09ff83b2d19391a41081d51a9fc4132768374dc893a33c85` | 591／74／73 |
| tcga_rcc/fold_10 | `79e4ab1960382ce216ec2a6635a8e0668a0b368f47799dfe4d36e74470e1fb18` | 591／74／73 |
| tcga_brca/fold_1 | `8508790e82973e2eb76358d7e443dad6080ad161bcf190bcd1f945c83354fe23` | 712／89／90 |
| tcga_brca/fold_2 | `109d639bf3721c6bdcb60b56e2547bada9c82119cfacf3547f76e63f031c2e3e` | 712／90／89 |
| tcga_brca/fold_3 | `b237195c7b40f54b8ecda7d7b025b4d0cbb33e6526d2fcc8599a6d7a73495572` | 712／90／89 |
| tcga_brca/fold_4 | `13c15149394d5c3388bb2dbcc15d9492f626a1d1766a464aeb2c1266d6c98632` | 712／90／89 |
| tcga_brca/fold_5 | `2763ec8f11a0a37ae24d748517630deecd6c7b5b07e9ad491a50c64a848c4b46` | 712／90／89 |
| tcga_brca/fold_6 | `4b3a38b16b9724f8aa9d732eee9f35d0d62a4121ab6c4bbf3b1d9f5ad9c5f267` | 712／90／89 |
| tcga_brca/fold_7 | `49c540f792f710c0e4e8c2a41ba066ae22820813c64ab2b031a964d356e329ff` | 712／90／89 |
| tcga_brca/fold_8 | `db1a65f7e46f409f2b19eef8a3c53d82c4b53db53dc153869a960e118f8b07f4` | 712／90／89 |
| tcga_brca/fold_9 | `55293ba02e7d13501e43eb679235a5ab6ec117f3c2c84e4bb8ed67f1cf44d0dd` | 712／90／89 |
| tcga_brca/fold_10 | `2c9e342eac98327d6c910c72f79bc2077b0131e796aad9f93918e60e92209123` | 712／90／89 |
| tcga_lung/fold_1 | `2543ea6401651eeda8c37b4f28f4d0a75a8e709b428f3105fa58cdbe99a86a13` | 694／87／87 |
| tcga_lung/fold_2 | `e18329e28fad96d614fce28c60365d732e6b99e95c8d406c179d3dd180751fa5` | 694／87／87 |
| tcga_lung/fold_3 | `1a27fc71e0aed6bfb567c0a87c52e6fbb523e576a15c835397904942743e20ee` | 694／87／87 |
| tcga_lung/fold_4 | `fb61430263d385d869ca73a73224507ac3de7dced1b0955b7a0e06f64400e475` | 694／87／87 |
| tcga_lung/fold_5 | `e5e9fbadd053a0bf4d2ebc36438a5799953b4f080b88acf8d7239d38b1580abe` | 694／87／87 |
| tcga_lung/fold_6 | `6039693694c0cc05dd361df69a79e682fa98ed867cf3c44f48d559e04fe131e1` | 694／87／87 |
| tcga_lung/fold_7 | `66bc7edd94ad17bd229a1b697d5d1508ba44bf1ede4b5c8d5eb8e6cb1afef5b1` | 694／87／87 |
| tcga_lung/fold_8 | `6c0d8c311d6aa98c554c0986a74f5773540634519a3af1cb7cec510ace16494b` | 694／87／87 |
| tcga_lung/fold_9 | `915bdaa9703fa8931496901f6bc0c5535d758a422da8d50af09e6a3f87b06a99` | 695／87／86 |
| tcga_lung/fold_10 | `ae7e56d3ff80c57c6469bd691a36a26a10984dc259eb3ce175032db8ffb18a70` | 695／87／86 |

結論：一致。三個 pathselect clone 的設定都解析到與本 repo 相同的實體檔（`can_dataset/<task>/datasplit/fold_{1..10}.npz`），40 檔 sha256 全同、病人名單逐折逐鍵相同（含順序）。限制：這是「同一份檔」的確認，不是兩份獨立來源的比對；pathselect 在 RunPod 上使用的資料集副本（見其 `docs/repro/RUNPOD.md:107`、`docs/ledger/DR-048.md:444`）不在本機，未比對。切分的產生邏輯與 seed 仍未找到（RUNBOOK_navcil.md §8-1）。
