# PREREG-14 — NC-14 預先註冊（依外部參考方法 QPMIL-VL 的定義重算 D3 的 Forgetting 與 Masked ACC）

登記時間：2026-09-30，計算前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），CPU；只讀既有 JSON 重算，不訓練、不推論、不讀特徵檔。
分支：`ws4-forgetting`（worktree `~/research/01_navrouter-cil-ws4f`，自 main @ f431246）。報告編號：REPORT_stage16.md。

## 問題

REPORT_ws4_external.md T3 指出：本研究的 Forgetting（`PREREG.md:36`）與 QPMIL-VL（Gou et al., AAAI 2025）Eq. 13 不同，
主表並列 QPMIL-VL 的已發表值時，Forgetting 欄不是同一個量。本實驗依 QPMIL-VL 論文的定義，
用 D3（AR γ = 0.001 ＋ I6(r = 2)，主系統）既有的逐階段、逐任務正確率重算 ACC、Masked ACC、Forgetting。

## QPMIL-VL 論文原文（`reference/papers/2410.10573v3-qpmil.pdf`，逐字抄錄）

**正確率矩陣**（p.10，Supplementary B 與 Tab. 6）：
> "After completing the incremental training, we can obtain an Accuracy performance matrix, as illustrated in Tab. 6.
> Subsequently, the relevant metrics reported in our experiments are calculated using this performance matrix."

Tab. 6（p.10）：列 = "After Training Dataset i"，欄 = "Test on Dataset t"；R_{i,t}；i < t 的格為 "-"（未定義）。

**ACC**（p.10，Eq. 11）：
> "ACC. Average Accuracy (ACC) represents the model's average performance across all datasets, which is calculated by
> ACC = (1/T) Σ_{t=1}^{T} R_{T,t}. (11)"

**Forgetting**（p.10，Eq. 13）：
> "Forgetting. Forgetting quantifies how much knowledge the model has forgotten about previous datasets, which is defined as
> Forgetting = 1/(T − 1) Σ_{t=1}^{T−1} max_{i∈{1,··· ,T}} R_{i,t} − R_{T,t}. (13)"

**Masked ACC**（p.5，Evaluation Metrics；論文沒有公式）：
> "Besides, we provide the Masked Average Accuracy (Masked ACC) metric, which is used for reference only to evaluate
> performance in task-incremental learning scenario by masking the logits of irrelevant classes."

Tab. 1 註（p.5）："We present the Masked ACC reflecting task-incremental learning scenario for reference."
task-incremental 的定義（p.3）："in task-incremental learning scenario, the model combines sample x_i^t with its
corresponding task identity t together to predict label y_i^t."

## 操作定義（T = 4）

1. **R 的索引**：R_{i,t} = 依該序學完第 i 個任務後，在該序第 t 個任務 test slides 上的正確率（該任務內正確張數 ÷ 張數）。
   對應 `per_fold.json` 的 `R[i−1][t−1]`（`scripts/nc5_report.py:47-52` `cil_full`）。
2. **ACC_Q** = (1/4) Σ_{t=1}^{4} R_{4,t}（Eq. 11）。
3. **Forgetting_Q** = (1/3) Σ_{t=1}^{3} ( max_{i∈{t,…,4}} R_{i,t} − R_{4,t} )。
   解讀：(a) max 只作用在 R_{i,t}，減號在 max 之外；(b) Eq. 13 寫 i ∈ {1,…,T}，但 i < t 的格在 Tab. 6 為 "-"，
   所以 max 只取已定義的 i = t,…,T；**i = T 包含在內**，因此每一項 ≥ 0。
4. **Masked ACC_Q** = (1/4) Σ_{t=1}^{4} Rm_{4,t}，Rm_{4,t} = 最終模型的類別分數只留第 t 個任務的類別（其餘遮掉）後的正確率。
   平均方式沿用 Eq. 11（論文名稱為 "Masked Average Accuracy"，沒有另給公式）。
   D3 的「最終模型的類別分數」= 分派到的 expert τ̂ 的 8 類 cosine（`scripts/nc2_report.py:89-96` `hard`，
   `pred` 在 C_τ̂ 取 argmax）；遮掉無關類別 = 在真實任務的 2 類內取 argmax，即 `hard` 回傳的 `masked`。
   **判定：與本研究既有 Masked ACC（PREREG-3 操作定義 9）相同**，所以一致性檢查 2 適用。
5. **本研究原定義 Forgetting_O**（`PREREG.md:36`；`scripts/nc5_report.py:55`）=
   (1/3) Σ_{j=1}^{3} ( max_{j≤i≤3} R_{i,j} − R_{4,j} )：max 不含 i = 4，單項可為負。
   兩者關係：每一項 Forgetting_Q = max(Forgetting_O 該項, 0)，所以 Forgetting_Q ≥ Forgetting_O；
   兩者相等 ⇔ 沒有任何舊任務在 t = 4 的正確率嚴格高於它先前各階段的最高值。
6. **十折統計**：每折算一個值，十折 mean ± sd（`selector/cil_eval.mean_sd`，樣本標準差），reverse、paper 兩序分列；另列每折值。

## 資料（唯讀，皆已 commit）

- 主來源：`outputs/navcil/mac/nc8/per_fold.json`，列「6 D3：AR＋I6(r=2)（主系統）」的 `R`、`Rm`、`acc`、`masked`、`forgetting`（NC-8）。
- 交叉核對：`outputs/navcil/mac/nc9/per_fold.json` 同一列的 `R`、`Rm`（NC-9，累加版分派器）；
  `outputs/navcil/mac/nc12/per_fold.json` 的 `per_fold[reverse|paper][折][階段].acc_task`（NC-12，只有 R，沒有 Rm）。
- QPMIL-VL 已發表值：`reference/external_baselines.json`（ACC、Masked ACC、Forgetting 平均值）；sd 另由論文 Tab. 1（p.5）、Tab. 2（p.6）抄錄到
  `outputs/navcil/mac/nc14/external_ref.json`。

## 一致性檢查（任何一項不符即停下回報，不寫主表）

1. ACC_Q 每折與 `acc` 位元相同；十折平均四捨五入到小數第四位 = 0.9128（reverse、paper 皆是）。
2. Masked ACC_Q 每折與 `masked` 位元相同；十折平均四捨五入到第四位 = 0.9312（兩序皆是）。
3. 以 Forgetting_O 公式從 R 重算，每折與 `forgetting` 位元相同；十折平均 = 0.0224（reverse）、0.0041（paper）（REPORT_stage10.md:16、:29）。
4. 三個來源的 R 逐折、逐格位元相同（NC-8 = NC-9 = NC-12）；NC-8 與 NC-9 的 Rm 逐格位元相同。

## 輸出

- 程式（新檔）：`scripts/nc14_ext_metrics.py`（讀 JSON、重算、寫結果與報告）。依 `tests/test_no_banned_deps.py`，程式內不出現外部方法名稱；
  名稱與已發表值由 `reference/external_baselines.json` 讀入。
- `outputs/navcil/mac/nc14/result.json`（檢查結果、十折統計）、`outputs/navcil/mac/nc14/per_fold.json`（每折值與每項 Forgetting）。
- `outputs/navcil/mac/REPORT_stage16.md`：
  - T1 主表：本方法（D3）與 QPMIL-VL，兩序，ACC／Masked ACC／Forgetting，全部依 QPMIL-VL 定義。
  - T2 一致性檢查結果。
  - T3 每折值（ACC_Q、Masked ACC_Q、Forgetting_Q、Forgetting_O）。
  - T4 Forgetting_O 與 Forgetting_Q 並列，列出被截為 0 的項（折、序、任務、該項原值）。
