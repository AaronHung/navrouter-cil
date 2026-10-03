# PREREG-21 — EXT-1 預先註冊（外部對照改為同折逐折配對；定稿表格全量重算）

登記時間：2026-10-03，EXT-1 任何程式執行前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro，16 GB），navrouter-cil 內的計算一律 `--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0、closed-form float64；
C、D 的數字來自同一台、同一批。外部方法（B）的裝置依 B3 計時決定，寫在其 PROVENANCE.md。
分支：`ext-1`（自 main @ 1497925）。不合併；不修改既有程式、既有報告與既有 PREREG；新程式放新檔（`scripts/ext1_*.py`）。
報告：`outputs/navcil/mac/REPORT_ext1.md`；判斷紀錄 `outputs/navcil/mac/ext1/DECISIONS.md`。

本檔＝指令原文（第一部分），加上執行前同時登記的操作化細則（第二部分；不改變第一部分）。

---

## 第一部分：指令原文

### 目的與不變項

把主表的外部對照從「發表值」變成「同一組十折」的數字，並把定稿表格裡只抽查過的列全部重算。
全程不改 FINAL 的任何定義與參數（AR TP ＋ head 四輪 ＋ ridge readout γ = 1e-3），不在 test 上調任何東西。

### 比較規則

所有外部對照一律逐折配對（同 fold、同序），報 mean ± sd、FINAL 贏的折數／10、配對符號檢定 p 值（exact binomial，雙尾）。
不同折的數字不進主表，發表值只放表註。

### MergeSlide 移植規則

只改四件事：(1) 特徵維度 768 → 512；(2) 文字向量改用 navrouter-cil `cache/text/` 的 CONCH 類別文字；
(3) 任務數 6 → 4 與類別定義（ESCA、RCC、BRCA、LUNG 各 2 類）；(4) 切分改用我們的十折。
其餘超參數全用其 repo 預設；不在我們的 test 上調任何超參數；若其程式需要 validation，用我們每折的 validation。

### 外部程式不得進 navrouter-cil

MergeSlide clone 到 `~/research/03_mergeslide`（獨立資料夾）；navipath 不 clone 進 navrouter-cil。
navrouter-cil 只收數字（CSV／JSON／MD）到 `outputs/external/…`；`tests/test_no_banned_deps.py` 必須仍通過。

### 查核

- **K6 折一致性**：三方的每折 slide ID 集合（排序後 sha256）相同——navrouter-cil `data/` 的切分、navipath 當年跑 QPMIL-VL 實際載入的切分、
  MergeSlide 移植後實際載入的切分。任一方不一致就記錄差異（哪一折、差幾張、哪些 ID），該方的數字不進配對表。
- **K7 重算一致性**：C 的重算值與 FINAL_RESULTS.md／REPORT_moe* 既有數字逐格差 ≤ 1e-4。
- **K8 MergeSlide 健全性**：每任務剛學完時的 Masked ACC ≥ 該任務 zero-shot top-64 Masked ACC − 0.02。
  低於此視為移植有誤：停該階段、記錄，不調參救它。
- 若 QPMIL-VL 逐折輸出找不到或 K6 不過：不重跑它（需要 GPU），只記錄；主表維持發表值加表註。

### 階段（A–E 與補充的原文摘要；全文見 REPORT_ext1.md 的「指令」節）

- **A** QPMIL-VL 逐折數字匯入（A1 找輸出、A2 K6、A3 匯入 `outputs/external/qpmil_vl/`、A4 與 FINAL〔seed 42 一欄、五 seed 平均一欄〕及主系統逐折配對）。
- **B** MergeSlide 同折重跑（B1 clone ＋ PORT.md、B2 接 CONCH 特徵、B3 計時〔fold 1、reverse、第一個任務；cpu 與 mps 各一次完整「訓練 ＋ 合併 ＋ 推論」〕、
  B4 十折兩序、B5 配對）。估計總時數 = 單任務秒數 × 4 × 10 × 2 × 1.2，取較快的裝置；> 12 小時不跑。
- **C** FINAL_RESULTS 的消融與負結果列全量重算（seed 42，十折兩序，t = 1…4），寫 `ABLATION_full.csv`；K7；Masked ACC 兩種定義都算。
- **D** D1 逐折配對統計與 slide 層級 bootstrap；D2 推論與訓練成本；D3 γ 敏感度（事後、只放補充）；D4 TP 混淆矩陣、逐任務 WP、FINAL 逐類正確率。
- **E** REPORT_ext1.md、DECISIONS.md、每 15 分鐘心跳。任何階段失敗：記錄到 REPORT 的「失敗」節，繼續其他獨立階段；不重試超過一次。

### 補充（取代前一段補充）

1. 執行順序：A → B1–B3（只估時，不跑 B4）→ C → D → B4。各階段依序、不並行；B3 計時時機器上不得有其他 job。PORT.md 與 B3 估時一出來立刻回報。
2. B4 不自動開跑：必須等 PI 在聊天說「B4 go」才跑。C、D 先照常跑完；PI 的 go 若比 D 早到，排在 D 之後。
3. PORT.md 逐條回答 (a) backbone 與初始權重、(b) 合併對象與基準權重、(c) task-to-class 文字向量的產生與 CONCH 設定下的對應檔、
   (d) slide 向量與文字是否同一空間、(e) 指標定義與我們的差異、(f) 預設超參數清單與出處行號。
4. 停止條件：若 (a) 的答案是「backbone 必須是 TITAN 預訓練的 slide encoder，CONCH 沒有對應物」，B 停在 B1，不自行用替代 backbone，
   在回報裡列出可能做法與各自偏離原方法的地方，等 PI 決定。
5. B4 的輸出同時算兩套指標：我們的 ACC／Masked ACC（Table 1 定義）／Forgetting／BWT，以及它論文的 bACC／Masked bACC／forgetting。
6. C1 的每一列每個 (order, fold) 完成寫 done 標記，重跑自動跳過。

---

## 第二部分：操作化細則（與第一部分同時登記；不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `outputs/navcil/mac/ext1/DECISIONS.md`（做了什麼判斷、為什麼），並印在報告內。

### 0. 共通

1. **資料、類別順序、序**：同 PREREG-20 細則 1。8 類固定序 ESAD, ESCC, CCRCC, PRCC, IDC, ILC, LUAD, LUSC；任務位置 esca = 0、rcc = 1、brca = 2、lung = 3；
   reverse = esca → rcc → brca → lung，paper = lung → brca → rcc → esca。
2. **FINAL** = PREREG-20 的 P-4(s)：v(s)／RDG(γ = 1e-3)，TP = AR（γ = 1e-3）。**主系統** = P-main(s)：v(s)／TXT。seed 42–46。
   向量、head、AR 一律讀既有快取與權重（唯讀）；不訓練任何 head。
3. **指標（我們的定義）**：ACC_t、Forgetting、BWT 由 `nc5_report.cil_full`（Table 1 同一個函式）；四任務（已學任務）等權。
   - **Masked ACC（Table 1 定義）**：向量取自 τ̂（TP 判定的任務）的 expert／文字，在**真實任務**兩類內 argmax。
   - **Masked ACC（oracle 定義）**＝告訴任務的 WP：向量取自真實任務的 expert／文字，在真實任務兩類內 argmax。
   - 沒有 expert、向量與任務無關的列（zero-shot 8 類、LIN8、mean_vec 類）：兩種定義相同，兩欄填同一個數。
   - 主表之後統一用 Table 1 定義；oracle 定義放補充。
4. **逐折配對（比較規則的操作化）**：每序、每個指標：逐折差 D_f = FINAL_f − 對照_f（f = 1…10，同折、同序）。
   報：FINAL 與對照各自的十折 mean ± sd（樣本標準差）、D 的平均、FINAL 贏的折數（D > 1e-12）、輸的折數（D < −1e-12）、平手折數；
   **符號檢定**：`scipy.stats.binomtest(贏, 贏 + 輸, 0.5, alternative="two-sided").pvalue`（平手折不計入 n；贏 + 輸 = 0 時 p = 不適用）。
   配對的指標：t = 4 的 ACC、t = 4 的 Masked ACC（Table 1 定義）、Forgetting、BWT；另報 Ā（四階段 ACC 平均）。
   FINAL 兩欄：seed 42；五 seed 平均（每折先對 seed 42–46 平均，再配對）。
5. **失敗處理**：階段失敗 → 寫 `ext1/FAILED_<階段>.txt`、記入 REPORT「失敗」節、最多重試一次（只在原因是程式錯誤且已修正時；不改期望值、不改判準），
   之後繼續其他獨立階段。K 不過不算程式錯誤，不重試。
6. **不看 test 調參**：本批沒有任何由 test 決定的選擇。D3 的 γ 網格只作事後描述，不改 γ = 1e-3。

### 1. K6

7. **slide ID 集合**：每個 (fold, task, split ∈ {train, val, test})：實際載入的 slide ID（不含副檔名的字串）排序後以 `\n` 連接、UTF-8 編碼的 sha256。
   navrouter-cil 一方 = `selector.evaluate.slide_dataset`（即 `Ctx.ds`）實際列出的 slide；另兩方 = 該方程式／log／輸出中可取得的「實際載入清單」。
   若某一方只留下 patient 層級的切分檔而沒有 slide 清單：以「同一份切分檔（sha256 相同）＋同一份標籤表」為間接證據，在報告標為「間接」，
   並另比對輸出中每折每任務的 test 張數；間接證據不算 K6 通過，該方的數字是否進配對表在報告中明列為待 PI 決定，預設不進。
8. 判定：某一方的 120 個（10 折 × 4 任務 × 3 split）雜湊與 navrouter-cil 全同 → 該方 K6 通過。不同的格列出差異（哪一折、哪個任務／split、差幾張、哪些 ID）。
   配對只用到 test；若只有 train／val 不同、test 全同，照實列出，該方仍視為 K6 不過（訓練資料不同）。

### 2. A（QPMIL-VL 匯入）

9. 只讀 navipath 既有輸出，不 import、不複製其程式；匯入腳本放 `~/research/03_mergeslide` 之外的獨立位置
   （`~/research/ext/ext1_import/`，不進 navrouter-cil 的 scripts/），navrouter-cil 只收 `outputs/external/qpmil_vl/perfold.csv` 與 `PROVENANCE.md`。
10. `perfold.csv` 欄位：order, fold, t, ACC, MaskedACC, Forgetting, BWT。Forgetting、BWT 只在 t = 4 填（其餘留空）。
    若來源有 R[t][j]（階段 t 對已學任務 j 的正確率），ACC_t、Forgetting、BWT 以 Table 1 同一公式由 R 重算，並與來源自帶的彙總值並列；
    若來源只有彙總值，照錄並註明其定義。PROVENANCE.md：repo、commit、輸出路徑、機器、日期、指標定義、與發表值的差。
11. 找不到逐折輸出或 K6 不過：照第一部分，不重跑、只記錄，主表維持發表值加表註；A4 不做（或只列為「非配對、僅供參考」，不進配對表）。

### 3. B（MergeSlide）

12. clone 後記錄上游 commit；所有改動在 `~/research/03_mergeslide` 的獨立分支，以 `git diff --stat` 與逐行 diff 記錄在 PROVENANCE.md。
    loader 直接讀 `can_dataset/<task>/feats-l1-s256_CONCH/pt_files/`，逐 slide 載入；轉檔只寫 `~/research/03_mergeslide/cache/`。
13. **B3 計時**：fold 1、reverse 序、第一個任務（tcga_esca）；cpu 與 mps 各一次完整「訓練 ＋ 合併 ＋ 推論」；wall 秒數（`time.perf_counter`）與
    峰值記憶體（`/usr/bin/time -l` 的 maximum resident set size；mps 另記 `torch.mps.driver_allocated_memory` 的最大值）。計時前確認沒有其他 tmux job 在跑計算。
    估計總時數 = min(cpu, mps) 單任務秒數 × 4 × 10 × 2 × 1.2 ÷ 3600。ESCA 是最小的任務（train 約 120 張），估計值會偏低：
    另報「以各任務 train 張數等比例放大」的估計，兩個數字都列；**是否 ≤ 12 小時以指令原式判定**，等比例估計只作揭露。
14. **B4** 只在 PI 於聊天說「B4 go」之後、且 D 完成之後才跑。輸出 `outputs/external/mergeslide/`：perfold.csv（我們的指標）、
    perfold_paper_metrics.csv（其論文的 bACC／Masked bACC／forgetting）、每任務訓練秒數、儲存 bytes、PROVENANCE.md。每個 (order, fold) 一個 done 標記。
15. **K8**：每序、每任務 j：「剛學完」= 階段 t_j（任務 j 在該序的位置）的模型在任務 j test 上的 Masked ACC（Table 1 定義；其方法若沒有 expert 選擇，
    即在任務 j 兩類內 argmax 的正確率）；對照 = 同折 zero-shot top-64 在任務 j 的 Masked ACC（C 的 ZS8 列）。
    判定用十折平均：mean_f(MergeSlide_j) ≥ mean_f(ZS_j) − 0.02，8 格（4 任務 × 2 序）都滿足才通過。逐折值照列（ESCA 每折 test 約 15 張，單折不判定）。
    B3 的 fold 1 單任務結果只作早期警訊：若 fold 1 ESCA 剛學完的 Masked ACC 低於同折 zero-shot 0.10 以上，在回報中標示，仍由 PI 決定是否 B4。
16. **停止條件**（補充 4）照原文執行。

### 4. C（全量重算）

17. **列**（`row` 欄的鍵；seed 42；兩序；t = 1…4；每列每 (order, fold) 一個 done 標記）：

    | row | 定義 | 既有來源（K7 比對） |
    |---|---|---|
    | ZS8 | zero-shot 8 類 top-64（無訓練、無 TP） | nc8/per_fold.json 第 1 列；nc1/metrics.json |
    | LIN8 | mean_vec／8 類 ridge γ = 0.01（無 TP） | nc5/metrics.json `lin8.test`；nc8 |
    | MAIN | 主系統 P-main(42) | moe4/f2.json `h1["P-main(42)"]` |
    | FINAL | P-4(42) | moe4/f2.json `h1["P-4(42)"]` |
    | NOHEAD | v0（g = 0 四輪）／RDG 1e-3（= S-R0f） | moe3/f2.json `S-R0f` |
    | ONE64 | 一次取 64：P-F(42) | moe4/f2.json `h1["P-F(42)"]` |
    | K32、K64、K128、K256 | u_K／RDG 1e-3（不用 head） | moe3/f5.json；K64 另對 `S-R0` |
    | M1 | 任務間 soft gating（MOE-0 B5，T\* = 0.1） | moe0/results.json B5 |
    | M2 | 跨器官共用 head（MOE-0 B2 的 4 × 4；另列一個 head 用於全部任務的四列） | moe0/results.json B2 |
    | M3 | σ 固定版融合（MOE-1；β = 1；LIN8 γ = 0.01） | moe1/s2.json `M3` |
    | G1、G2 | M3 ＋ gate（MOE-1 S6） | moe1/s6.json |
    | ANC | u_64／ANC(γ\*, α\*)（= S-A0）；另列 ANC × v0、ANC × v(42) | moe3/f2.json `S-A0`、`S-A0f`、`S-Ah` |
    | CONCAT | LRG(42)：[v; mean_vec; 1] ridge（MOE-2 E4 的 γ\*） | moe2/e4.json、e7.json |
    | RF | 隨機特徵 ridge 取代 (b) 的 M3（MOE-1 S7 的 γ\*） | moe1/s7_rf.json |

    指令的「M1、M2」在 repo 內沒有定義；此處取 MOE-0 的對應（M1 = B5 soft gating、M2 = B2 跨 head，與門檻 G-M2 同名），在 DECISIONS 註記，
    若 PI 所指不同，該兩列作廢重算。
18. **算法**：一律呼叫既有函式（`moe0_common`…`moe4_common`、`nc5_report.cil_full`、`nc8_report.B8`），不重寫判讀器；超參數全部讀既有選定值
    （`moe3/hp.json`、`moe0/selection.json`、各 e*.json／s*.json 的 γ\*），不重選。gate（G1、G2）的參數讀既有結果，不重學；若既有產物沒有存參數，
    以既有程式與既有訓練資料快取重算（決定性），並在 DECISIONS 註明。
19. **t < 4**：原批次只在 t = 4 定義的列（M1、M2、G1、G2、RF 的部分欄位），若既有程式的定義可直接延伸到「只含已學任務」的階段就算；
    否則 t < 4 留空並在 REPORT 註明原因。不為了填滿表格新設計算法。
20. **CSV 欄位**：row, order, fold, t, CIL_ACC, WP, MaskedACC_table1, MaskedACC_oracle, Forgetting, BWT, storage_bytes。
    WP = 告訴任務（已學任務等權）＝ MaskedACC_oracle（兩欄同值，保留兩欄是指令的欄位）；無 expert 的列 WP 記為其 Masked ACC。
    Forgetting、BWT 只在 t = 4 填。storage_bytes = S + t·p（fp32；公式取自 moe2 E8／moe3 F7／moe4 H5；該批沒有儲存公式的列留空並註明）。
21. **K7**：(i) 對既有 JSON 的逐折值：最大絕對差 ≤ 1e-4（預期為 0 或浮點尾數）；(ii) 對 FINAL_RESULTS.md／REPORT 的四位小數 mean ± sd：
    |重算 − 報告印出值| ≤ 1e-4（含捨入）。差 > 1e-4 的格列表（列、序、折或「十折平均」、舊值、新值、可能原因）；不覆寫舊報告。K7 不過不擋 D。
22. FINAL_RESULTS B 表中五 seed 的列（一次取 64 的五 seed 平均、γ 敏感度 H4）另以五 seed 重算比對（只比對，不進 `ABLATION_full.csv` 的 seed 42 列；
    以 `row` = `ONE64_5seed`、`FINAL_5seed`、`MAIN_5seed` 附在 CSV 後段）。

### 5. D

23. **D1**：四組（FINAL − 主系統、FINAL − NOHEAD、FINAL − ZS8、FINAL − LIN8）× 兩序 × {seed 42；五 seed 平均（NOHEAD、ZS8、LIN8 與 seed 無關）}：
    t = 4 CIL ACC 的逐折配對（細則 4）。**bootstrap**：每序；單位 = test slide；在每個（折、任務）層內有放回重抽同樣張數；
    統計量 = 十折平均的四任務等權 CIL ACC 差（與主表同一個彙總）；1000 次；`torch.Generator().manual_seed(0)`；95% CI = 2.5 與 97.5 百分位。
    兩個系統用同一組重抽索引（配對）。五 seed 版本：每張 slide 的正確與否先對 seed 平均。
24. **D2**：推論成本以新腳本在 fold 1 全部 test slides 上量測 FINAL(42) 的完整推論（讀檔、mean_vec、AR、τ̂ 的 head 評分全部 patch、四輪選 64、ridge 判讀）：
    每張 `t_read_s`、`t_compute_s`；報平均 patch 數、head 評分的 patch 數（= 全部 patch）、選出張數、每張秒數的平均與中位數。CPU、8 執行緒、機器上無其他 job。
    訓練成本：每任務 head 訓練秒數 = `moe1/i6_seed{43…46}/fold*_train.json` 的 `wall_s`（seed 42 的 train json 若在 `i6/r2/` 則併入，成五 seed；否則四 seed 並註明）；
    ridge 累加秒數 = 新量測（每任務 A += XᵀX 與 B，含解 W），fold 1、兩序。
25. **D3**：γ ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}（只改 ridge readout 的 γ；AR 的 γ 維持 1e-3）：FINAL 兩序 t = 4 CIL ACC 的十折 mean ± sd（seed 42 與五 seed 平均）。
    事後分析、只放補充；γ 的選擇仍由 MOE-3 的逐階段 validation 決定。
26. **D4**：TP（AR，t = 4）的 4 × 4 混淆矩陣（列 = 真實任務、欄 = τ̂；兩序；十折合計張數）、逐任務 WP、FINAL 逐類正確率（seed 42 與五 seed 合計）。
    moe4 H3、moe0 B1 已有的直接引用並註明出處；本批重算只作核對。

### 6. 執行與交付

27. navrouter-cil 內的長時間指令在 tmux 內以 `caffeinate -ims` 執行；各階段依序、不並行。C 的每列每 (order, fold) 寫 done 標記。
28. 每 15 分鐘在聊天回報心跳（目前步驟、完成／總數、失敗數）；全部跑完回報完整摘要；不關機、不停既有 tmux session。
29. `REPORT_ext1.md`：A–D 全部表格、K6–K8、B3 的時間估計與決定、未完成項與原因、「失敗」節、DECISIONS 全文。只給數字與表格。
    commit 前跑 pytest；不 merge、不 push 到 main。
