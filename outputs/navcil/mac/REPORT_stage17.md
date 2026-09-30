# REPORT — NC-15：LoRA 對照版（L1 v2）的 r 改以 validation 選定（Mac CPU，十折兩序）

機器：mac（Apple M1 Pro）、`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；所有數字來自同一台。判準見 `PREREG-15.md`（b14bc47），檢查 6 的比對方式見 `AMENDMENT-4.md`（eb327d5）。選 r 只讀 validation；`nc15/selection.json` 於 1357af9 commit 之後才讀 test。r = 1、2 的權重沿用 `lora_v2/`（NC-3 批次），r = 3 為本輪新訓練（`nc15/lora_r3/`，log：`logs/nc15_lora_r3.log`）；validation 評估 log：`logs/nc15_val.log`。task-inferred = AR 分派（γ = 0.001，float64），task-known = oracle 分派；表二的修正頭數字在本輪以同一程式重算。數字附 fact-id 與所在 facts.json 行號（路徑相對於 `outputs/navcil/mac/`）。

## 判準落點

| 項目 | 數值 | 結果 |
|---|---|---|
| L0 validation WP 重算 = 0.9290（REPORT_stage9.md:19） | 0.9290 ± 0.0126（`nc15.val.l0.wp`（nc15/facts.json:2）） | 符合 |
| 門檻 = L0 validation − 0.01 | 0.919017（四捨五入 0.9190；`nc15.val.threshold`（nc15/facts.json:3）） | — |
| 未四捨五入門檻與字面 0.9190 的判定是否一致 | 三個候選、兩序皆一致 | 符合 |
| validation 評估檢查 6（AMENDMENT-4）：L0 四輪 index 逐位相同、cosine 最大差 ≤ 1e-6、2 類內 argmax 全同 | 十折四任務皆通過；cosine 最大差 6.6e-07 | 符合 |
| 選出的 r\* | r\* = 1（`nc15.rstar`（nc15/facts.json:10）） | 符合門檻 |
| 一致性：lora.r2.inferred.reverse = 0.9176（REPORT_stage10.md:15） | 0.9176；與 nc8/per_fold.json 逐折最大差 0.0e+00 | 符合 |
| 一致性：lora.r2.inferred.paper = 0.9132（REPORT_stage10.md:28） | 0.9132；與 nc8/per_fold.json 逐折最大差 0.0e+00 | 符合 |
| 一致性：lora.r2.known.reverse = 0.9401（REPORT_stage5.md:43） | 0.9401 | 符合 |
| 一致性：lora.r2.known.paper = 0.9339（REPORT_stage5.md:44） | 0.9339 | 符合 |
| 一致性：head.known.reverse = 0.9340（REPORT_stage10.md:18） | 0.9340；與 nc8/per_fold.json 逐折最大差 0.0e+00 | 符合 |
| 一致性：head.known.paper = 0.9340（REPORT_stage10.md:31） | 0.9340；與 nc8/per_fold.json 逐折最大差 0.0e+00 | 符合 |
| 一致性：head.inferred.reverse = 0.9128（REPORT_stage10.md:16） | 0.9128；與 nc8/per_fold.json 逐折最大差 0.0e+00 | 符合 |
| 一致性：head.inferred.paper = 0.9128（REPORT_stage10.md:29） | 0.9128；與 nc8/per_fold.json 逐折最大差 0.0e+00 | 符合 |

## 表一 validation WP 與選 r（十折四任務平均，mean ± sd）

門檻 = 0.9190（L0 validation 0.9290 − 0.01）；兩序都 ≥ 門檻才算過（PREREG-15 操作定義 4）。

| r | reverse validation WP | paper validation WP | reverse ≥ 0.9190 | paper ≥ 0.9190 | 是否過門檻 | fact-id |
|---|---|---|---|---|---|---|
| 1 ★ | 0.9274 ± 0.0175 | 0.9380 ± 0.0168 | ✓ | ✓ | 過 | `nc15.val.r1.reverse.wp`（nc15/facts.json:4）；`nc15.val.r1.paper.wp`（nc15/facts.json:5） |
| 2 | 0.9296 ± 0.0155 | 0.9370 ± 0.0192 | ✓ | ✓ | 過 | `nc15.val.r2.reverse.wp`（nc15/facts.json:6）；`nc15.val.r2.paper.wp`（nc15/facts.json:7） |
| 3 | 0.9298 ± 0.0126 | 0.9339 ± 0.0172 | ✓ | ✓ | 過 | `nc15.val.r3.reverse.wp`（nc15/facts.json:8）；`nc15.val.r3.paper.wp`（nc15/facts.json:9） |

★ = r\* = 1（符合門檻者中最小的 r；同分取較小的 r）。

每任務 validation WP（十折平均）：

| r | 序 | tcga_esca | tcga_rcc | tcga_brca | tcga_lung |
|---|---|---|---|---|---|
| 1 | reverse | 0.9533 | 0.9339 | 0.9045 | 0.9177 |
| 1 | paper | 0.9800 | 0.9404 | 0.9181 | 0.9134 |
| 2 | reverse | 0.9533 | 0.9392 | 0.9087 | 0.9173 |
| 2 | paper | 0.9800 | 0.9404 | 0.9139 | 0.9134 |
| 3 | reverse | 0.9533 | 0.9406 | 0.9108 | 0.9147 |
| 3 | paper | 0.9667 | 0.9457 | 0.9097 | 0.9134 |

各序第一個任務（reverse：esca；paper：lung）使用凍結的底座（L0），該任務的數值與 r 無關。

## 表二 LoRA 對照版 L1(r\* = 1) 與修正頭（test、t = 4、十折 mean ± sd）

差距 = 修正頭 − LoRA 對照版；容許範圍：差距 ≥ −0.005（PREREG-15 操作定義 10）。

| 指標 | 序 | LoRA 對照版 L1(r\*) | 修正頭 I6(r = 2) | 差距 | 在 0.005 內 | 修正頭較高折數 | fact-id |
|---|---|---|---|---|---|---|---|
| task-known WP | reverse | 0.9403 ± 0.0172 | 0.9340 ± 0.0202 | -0.0063 | 否 | 3/10（平 0） | `nc15.lora.r1.known.reverse.acc`（nc15/facts.json:11）；`nc15.head.known.reverse.acc`（nc15/facts.json:13）；`nc15.gap.known.reverse`（nc15/facts.json:14） |
| task-known WP | paper | 0.9295 ± 0.0208 | 0.9340 ± 0.0202 | +0.0045 | 是 | 5/10（平 0） | `nc15.lora.r1.known.paper.acc`（nc15/facts.json:15）；`nc15.head.known.paper.acc`（nc15/facts.json:17）；`nc15.gap.known.paper`（nc15/facts.json:18） |
| task-inferred ACC | reverse | 0.9180 ± 0.0235 | 0.9128 ± 0.0258 | -0.0052 | 否 | 4/10（平 0） | `nc15.lora.r1.inferred.reverse.acc`（nc15/facts.json:19）；`nc15.head.inferred.reverse.acc`（nc15/facts.json:21）；`nc15.gap.inferred.reverse`（nc15/facts.json:22） |
| task-inferred ACC | paper | 0.9119 ± 0.0205 | 0.9128 ± 0.0258 | +0.0009 | 是 | 4/10（平 0） | `nc15.lora.r1.inferred.paper.acc`（nc15/facts.json:23）；`nc15.head.inferred.paper.acc`（nc15/facts.json:25）；`nc15.gap.inferred.paper`（nc15/facts.json:26） |

另報（非判準）：L1(r = 1) 的 task-known WP 與 REPORT_stage5.md:41、:42（0.9403／0.9295）四位相同。

參考（既有 r = 2，本輪重算）：

| 指標 | 序 | L1(r = 2) | fact-id |
|---|---|---|---|
| task-known WP | reverse | 0.9401 ± 0.0192 | `nc15.lora.r2.known.reverse.acc`（nc15/facts.json:12） |
| task-known WP | paper | 0.9339 ± 0.0197 | `nc15.lora.r2.known.paper.acc`（nc15/facts.json:16） |
| task-inferred ACC | reverse | 0.9176 ± 0.0252 | `nc15.lora.r2.inferred.reverse.acc`（nc15/facts.json:20） |
| task-inferred ACC | paper | 0.9132 ± 0.0210 | `nc15.lora.r2.inferred.paper.acc`（nc15/facts.json:24） |

逐折差距（修正頭 − LoRA 對照版 L1(r\*)）：

| 指標 | 序 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| task-known | reverse | -0.0133 | -0.0088 | +0.0059 | -0.0045 | +0.0078 | -0.0251 | -0.0025 | -0.0169 | +0.0049 | -0.0105 |
| task-known | paper | -0.0053 | -0.0057 | +0.0213 | -0.0088 | +0.0079 | +0.0150 | -0.0133 | -0.0056 | +0.0235 | +0.0156 |
| task-inferred | reverse | +0.0034 | -0.0088 | +0.0059 | -0.0045 | +0.0078 | -0.0251 | -0.0025 | -0.0169 | +0.0022 | -0.0139 |
| task-inferred | paper | -0.0053 | -0.0057 | +0.0213 | -0.0088 | +0.0105 | +0.0150 | -0.0133 | -0.0056 | +0.0029 | -0.0022 |

L1(r\*) 每任務 task-known test WP（十折平均）：

| 序 | tcga_esca | tcga_rcc | tcga_brca | tcga_lung |
|---|---|---|---|---|
| reverse | 0.9804 | 0.9568 | 0.9108 | 0.9130 |
| paper | 0.9403 | 0.9531 | 0.9202 | 0.9044 |

## 耗時（秒，wall clock；執行緒 8）

| 項目 | 秒 |
|---|---|
| L1(r = 3) 訓練（兩序 × 十折 × 3 任務 × 5 epochs） | 2,105 |
| L1(r = 3) 附帶 test 評估（兩序 × 十折） | 398 |
| validation 評估（十折，r = 1、2、3 × 兩序同時） | 169 |
| test 讀取與重算（nc15_test.py） | 1 |

逐張讀檔／計算秒數：`nc15/val/fold*.pt`、`nc15/lora_r3/*/fold*_eval.pt` 的 `t_read_s`、`t_compute_s`（本機，未 commit）。
