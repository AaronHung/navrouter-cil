# PREREG-22 — EXT-2 預先註冊（FINAL-B：不訓練的變體做成與 FINAL 同等完整的一套；兩套並排；架構圖與質性圖的數據）

登記時間：2026-10-03，EXT-2 任何程式執行前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro，16 GB），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0、closed-form float64；本批我方的數字來自同一台、同一批
（對照列 zero-shot、LIN8、主系統、FINAL-A 也在本批重算，不沿用 EXT-1 的檔）。外部對照的數字來自其他機器，表註寫機器與日期。
分支：`ext-2`（自 main @ 1fd0c47）。不合併；不修改既有程式、既有報告與既有 PREREG；新程式放新檔（`scripts/ext2_*.py`）。
報告：`outputs/navcil/mac/REPORT_ext2.md`、`FINALB_RESULTS.md`、`FINALB_EXAMPLE.md`；判斷紀錄 `outputs/navcil/mac/ext2/DECISIONS.md`。

本檔＝指令原文（第一部分），加上執行前同時登記的操作化細則（第二部分；不改變第一部分）。

---

## 第一部分：指令原文

### 目的

把「不訓練」的變體做成和 FINAL 同等完整的一套（以下稱 **FINAL-B**；原 FINAL 改稱 **FINAL-A**，定義不動），讓兩套可以並排給 PI 與老師挑；
另外為架構圖與質性圖取數據。完成後不自己 merge。

### 0. PREREG-22（先 commit 再跑）

1. 承 EXT-1 的三個裁決，記進 `ext2/DECISIONS.md` 並反映到報告：
   a. 外部對照表註加兩句：reverse 序為 b8（論文設定）、paper 序為 b16；匯入欄為 acc@mid（argmax、不含 test 資訊），該欄在兩序都高於 test 最佳門檻的 acc，
      對外部方法有利。不重跑 paper 序。
   b. MergeSlide 做法 1 不做（成本見 DECISION.md）；做法 2–5 不做；論文只放相關工作與表註。
   c. M2 只報 WP，CIL 與 t < 4 的欄填「—」，不補算。
2. **FINAL-B 定義**：與 FINAL-A 完全相同，只拿掉 head：s_i = s0_i（z-scored max cosine 對該任務兩類文字），四輪各 16、λ = 1.5、原始 Z 等權平均後 L2 再補 1、
   [v; 1] 累加式 ridge、τ̂ 兩類內 argmax。沒有 seed。
3. 先確認 `ABLATION_full.csv` 的「拿掉 head」列是不是就是這個定義；是就直接引用，不是就寫出差異，FINAL-B 以本定義為準重算。
4. FINAL-B 的 γ 不直接沿用 1e-3：用 PREREG-20 的 trajectory 驗證規則在同一個 grid 重選一次；選到的若不是 1e-3，兩個都報，主結果用規則選到的。
5. 從程式確認並寫進 REPORT，逐條附檔名與行號：TP 的 x 是「patch 平均後 L2 再補 1」還是「L2 後平均」；TP 的 γ；讀出的 y 是 one-hot {0,1} 還是 ±1；
   B 的 shape；A 是否跨任務共用累加；s0 的 z-score 是在哪個範圍內算（單張 slide 內、該任務 train 集、或其他）。
6. 不在 test 上調任何東西。

### A. FINAL-B 全套（十折兩序，t = 1…4）

CIL ACC、WP、Masked ACC（Table 1 定義）、Forgetting、BWT、TP 正確率；逐任務 WP；逐類正確率；t = 4 的 TP 4×4 混淆矩陣；
儲存 bytes（共用／每任務／T = 4 總量）；每 slide 推論秒數（fold 1 全部 test）；K ∈ {32, 64, 128, 256}；一次 top-64 對四輪。
輸出 `outputs/navcil/mac/FINALB_RESULTS.md` 與逐折 CSV，欄位與 `ABLATION_full.csv` 相同。

### B. 兩套並排

1. FINAL-A（五 seed 平均與 seed 42）對 FINAL-B 逐折配對：mean diff、贏的折數、exact binomial p、slide 層級 bootstrap 95% CI（1000 次，seed 0）。
2. FINAL-B 對 QPMIL-VL（`outputs/external/qpmil_vl/perfold.csv`）逐折配對，格式同 REPORT_ext1 的 A 節。
3. 一張總表：列 = zero-shot 8 類、LIN8、QPMIL-VL（同折）、主系統、FINAL-B、FINAL-A；欄 = CIL ACC、WP、Masked ACC（Table 1）、Forgetting、BWT、
   每任務參數、每任務儲存 bytes、每任務訓練秒數。兩序各一張，表註照 0-1a。

### C. FINAL-B 範例

用 `FINAL_EXAMPLE.md` 的同兩張 slide 走完 FINAL-B 全程，格式相同：TP 四個分數、s0 的範圍、四輪選出的 64 個（與 FINAL-A 選出的交集張數）、v 的範數、
ridge 分數與判定。寫 `outputs/navcil/mac/FINALB_EXAMPLE.md`。

### D. 質性圖數據（可選；任一條件不成立就記錄並跳過，不要硬做）

1. 檢查 can_dataset 的 patch 特徵檔是否帶座標與倍率；寫明檔名與欄位。
2. 若有座標：fold 1 reverse 序，每個任務取一張 FINAL-A 與 FINAL-B 都判對的 test slide，輸出兩套各四輪的 64 個 patch 座標
   （CSV：slide_id, round, patch_idx, x, y, s0, g, s）。
3. 本機的 10 張 .svs 若有和 2) 重疊的 slide：用 openslide 或 tifffile 取縮圖（最低解析層），把 64 個 patch 的框畫上去，兩套各一張 PNG，四輪用四種顏色。
   沒有重疊就只交座標。
4. 不下載任何新切片。

### E. 回報

- `REPORT_ext2.md`：A 到 D 的表、PREREG-22 第 5 點的程式確認、未完成項與原因；`ext2/DECISIONS.md`。
- 每階段完成在聊天回報一次；全部完成回報總摘要；不問問題；失敗記錄後繼續其他階段。
- pytest 通過；逐折 CSV 進版控。

---

## 第二部分：操作化細則（與第一部分同時登記；不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `outputs/navcil/mac/ext2/DECISIONS.md`（做了什麼判斷、為什麼），並印在報告內。

### 0. 共通

1. **資料、類別順序、序、指標**：同 PREREG-21 細則 1、3。ACC_t、Forgetting、BWT 由 `nc5_report.cil_full`（Table 1 同一個函式）；四任務（已學任務）等權。
   Masked ACC（Table 1 定義）＝向量取自 τ̂ 的版本、在真實任務兩類內 argmax；WP ＝ Masked ACC（oracle 定義）＝向量取自真實任務的版本、在真實任務兩類內 argmax。
2. **FINAL-A** ＝ PREREG-20 的 P-4(s)（v(s)／RDG γ = 1e-3；TP ＝ AR γ = 1e-3；seed 42–46），定義不動。**主系統** ＝ P-main(s)。
3. **FINAL-B 的操作定義**（逐步；每一步與 FINAL-A 相同，唯一差別是第 (b) 步的分數沒有 g）：
   (a) TP：τ̂ ＝ 階段 t 的 AR（γ = 1e-3，輸入 [mean_vec; 1]）在已學任務中 argmax。
   (b) 分數：對任務 p 的兩類文字，s0 ＝ `zscore(text_nav_feats(Z, f_task_p)[:, 0])`（每個 patch 對兩類文字 cosine 的最大值，在該張 slide 的全部 patch 內 z-score）；s ＝ s0。
   (c) 選片：`four_round(Z, s0, λ = 1.5)`（四輪各 16，第 2 輪起扣 λ·與已選 patch 的最大 cosine）；patch 數 ≤ 64 的 slide 取全部。
   (d) v ＝ 所選 patch 的原始 Z 等權平均後 L2 正規化（`mean_norm`）；x ＝ [v; 1]（float64）。
   (e) 判讀：依序每學一個任務累加 A += XᵀX、B 的該任務兩欄 ＝ 各類 train slide 的 x 之和；每個階段解 W ＝ (A + γI)⁻¹B；分數 ＝ xW。
       train slide 用自己任務的文字算 v；test／validation 四個任務的文字都算，CIL 取 τ̂ 的版本，在 τ̂ 兩類內 d ≥ 0 判第一類。
   沒有任何訓練出來的參數進入 (b)、(c)；不載入任何 head。
4. **第一部分 0-3 的確認方式**（「拿掉 head」列 ＝ `ABLATION_full.csv` 的 `NOHEAD` ＝ MOE-3 的 S-R0f ＝ v0／RDG）：
   (i) 讀程式：v0 的產生處（`moe2_common.build_vec`）、s0 的定義（`selector/i6_expert.py`）、判讀器（`moe3_common.Stats`）與 γ 來源（`moe3/hp.json`），逐條附行號；
   (ii) 數值：**K11**（細則 9）不經任何 head、直接由特徵檔依細則 3 重算 fold 1 的 train 與 test 向量，與 `NOHEAD` 所用快取比對。
   (i)、(ii) 都相符才寫「是同一個定義，直接引用」；任何一項不符就列出差異，並以細則 3 由特徵檔十折重算（屆時另記 DECISIONS）。
5. **γ 重選**（第一部分 0-4）：PREREG-20 的 γ 固定值來自 MOE-3 的逐階段 validation 目標（PREREG-19 細則 5）；本批對 FINAL-B 的向量用同一規則、同一 grid 重算一次：
   每個候選 γ ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}、每折：兩序 × t = 1…4 共 8 個「validation 告訴任務 WP_t（已學任務等權）」取平均，再取十折平均；
   取最大者，完全相等才算同分，同分取較大的 γ；選在邊界照報、不延伸（`moe3_common.val_objective`、`select`）。只用 validation。
   選定值記為 γ\_B。γ\_B ≠ 1e-3 時：主結果用 γ\_B，另列 γ = 1e-3 的同一套 t = 4 與逐階段數字。AR 的 γ 維持 1e-3，不重選。
6. **不看 test 調參**：本批沒有任何由 test 決定的選擇。A 的 γ 敏感度表（test）只作事後描述，不改 γ\_B；K、一次取 64 的列只作消融，不改 FINAL-B 的定義。
7. **失敗處理**：階段失敗 → 寫 `ext2/FAILED_<階段>.txt`、記入 REPORT「失敗」節，繼續其他獨立階段；只在原因是本批新程式的錯誤且已修正時重試一次
   （不改期望值、不改判準）。K 不過不算程式錯誤，不重試；依各 K 的寫法處理。

### 1. 一致性檢查

8. **K9（本批重算 ＝ EXT-1 的逐折值）**：本批重算的列（ZS8、LIN8、MAIN 與 FINAL 的 seed 42–46、NOHEAD、K32、K64、K128、K256）的 `acc_t`、`wp_t`、`mk1_t`、
   Forgetting、BWT 與 `ext1/c/<row>/<order>_fold<f>.json` 逐格最大絕對差 ≤ 1e-9。不過 → 列出差異、照報本批的值，不覆寫 EXT-1 的檔。
9. **K10（γ 目標值）**：細則 5 重算的 validation 目標值（十折平均，五個 γ）與 `moe3/hp.json` 的 `combos["RDG:g0"].obj` 最大絕對差 ≤ 1e-9。
   不過 → 列出兩邊數值，γ\_B 以本批重算為準。
10. **K11（由特徵檔重算 ＝ 快取）**：fold 1：(a) train 全部 slide 依細則 3 (b)–(d) 重算的 v 與 `NOHEAD` 所用的 train 向量最大絕對差 ≤ 1e-6；
    (b) test 全部 slide 的完整推論（細則 3 (a)–(e)，t = 4、reverse 序）逐張判定與本批由快取算出的 FINAL-B（γ\_B）相同（不同張數 = 0）。
    不過 → 「拿掉 head」列不能直接引用：記錄差異，A 的數字標為「未確認」，並依細則 4 處理。

### 2. A（FINAL-B 全套）

11. **逐折值與 CSV**：`outputs/navcil/mac/FINALB_perfold.csv`，欄位與 `ABLATION_full.csv` 相同
    （row, order, fold, t, CIL_ACC, WP, MaskedACC_table1, MaskedACC_oracle, Forgetting, BWT, storage_bytes）。列：`FINALB`（γ\_B）、
    γ\_B ≠ 1e-3 時另加 `FINALB_g1e-3`；`FINALB_K32`、`FINALB_K64`、`FINALB_K128`、`FINALB_K256`；對照列 `ZS8`、`LIN8`、`MAIN`、`FINALA`（seed 42）、
    `MAIN_s43…46`、`FINALA_s43…46`、`MAIN_5seed`、`FINALA_5seed`（每折先對 seed 平均）。逐折 JSON 放 `ext2/a/<row>/`，每 (order, fold) 一個 done 標記。
    TP 正確率與逐任務 WP 不在上述欄位內，另存 `FINALB_perfold_extra.csv`（order, fold, t, TP_ACC, WP_<任務>, CIL_<任務>）。
12. **TP 正確率**：階段 t 的 τ̂ ＝ 真實任務的比例，已學任務等權（每任務各算後平均）；t = 4 的 4 × 4 混淆矩陣為十折合計張數（列 = 真實任務、欄 = τ̂）。
13. **逐任務 WP、逐類正確率**：逐任務 WP 每階段都報（未學為 —）。逐類正確率在 t = 4、reverse 序（告訴任務判定在 t = 4 與序無關，另報兩序逐張是否相同）：
    告訴任務與 CIL 各一份，格式為正確張數／張數（十折合計）；每任務 balanced accuracy 與四任務平均。
14. **K ∈ {32, 64, 128, 256} 與「一次 top-64 對四輪」**：K 列取既有定義 u_K（依 s0 一次取前 K 個、不扣冗餘；`ABLATION_full.csv` 的 K32…K256 列），
    判讀器同 FINAL-B（RDG，γ\_B）。「一次 top-64」＝ K = 64 的那一列；「四輪」＝ FINAL-B 本身。兩者逐折配對（細則 16 的格式）。
    四輪版本的 K ≠ 64 沒有既有定義（每輪幾張未定），不新設計。
15. **儲存**：由實際張量形狀計算（fp32）。共用：A（[v; 1] 的 XᵀX，513²）、A_mv（AR 的 XᵀX，513²）；每任務：兩類文字特徵（2 × 512）、B（該任務兩欄，513 × 2）、
    B_ar（該任務一欄，513）。不存 head。CONCH 骨幹、patch 特徵、超參數純量列出但不計入。
    **每任務參數**（B-3 的欄）＝ 由該任務 train 資料得到、需要保存的數值個數（head 參數、B 的該任務兩欄、B_ar 的該任務一欄）；兩類文字特徵由凍結的文字編碼器算出，
    計入「每任務儲存 bytes」但不計入「每任務參數」。共用的 A 不屬於任何單一任務，列在儲存表的共用欄。
16. **推論秒數**（fold 1 全部 test，CPU、8 執行緒）：每張 `t_read_s`、`t_compute_s`；計算 ＝ mean_vec → AR → τ̂ 的 s0（全部 patch）→ 四輪選 64 → ridge 判讀；
    AR 與 ridge 的 W（t = 4、reverse）在計時前解好。同一支程式、同一次執行另量 FINAL-A(42) 的同一流程（head 評分取代 s0）供並排。記錄當下的 load average。
17. **γ 敏感度（事後描述）**：FINAL-B 在五個 γ 的 test t = 4 CIL ACC 與 Ā（兩序；十折 mean ± sd）；另列 validation 目標值。不改 γ\_B。

### 3. B（兩套並排）

18. **逐折配對**（同 PREREG-21 細則 4）：差 ＝ FINAL-A − FINAL-B（同折、同序）；指標：t = 4 的 CIL ACC、WP、Masked ACC（Table 1）、Forgetting、BWT、Ā。
    報兩邊的十折 mean ± sd、平均差、FINAL-A 贏／輸／平手的折數（|差| ≤ 1e-12 為平手）、
    `scipy.stats.binomtest(贏, 贏 + 輸, 0.5, alternative="two-sided")`（平手不計；贏 + 輸 = 0 時 p = 不適用）。
    FINAL-A 兩欄：seed 42；五 seed 平均（每折先對 seed 42–46 平均）。Forgetting 另列「較好的折數」（差 < 0）。
19. **bootstrap**（同 PREREG-21 細則 23）：t = 4 CIL ACC；單位 = test slide；每個（折、任務）層內有放回重抽同樣張數；統計量 = 十折平均的四任務等權 CIL ACC 差；
    1000 次；`torch.Generator().manual_seed(0)`；95% CI = 2.5 與 97.5 百分位；兩個系統用同一組索引；五 seed 版本每張的正確與否先對 seed 平均。
20. **外部對照**：`outputs/external/<dir>/perfold.csv`（目錄名在執行時列出，`scripts/` 不寫外部方法的名字）；FINAL-B 對它逐折配對，
    指標與格式同 REPORT_ext1 的 A 節（ACC、Masked ACC 兩種我方定義、Forgetting、BWT、Ā）。只用 K6 通過的批次（reverse_b8、paper_b16）。
    表註：機器與日期（PI 2026-10-03 裁決）＋第一部分 0-1a 的兩句。
21. **總表**（兩序各一張，t = 4，十折 mean ± sd）：列 ＝ zero-shot 8 類、LIN8、外部對照（同折）、主系統（五 seed 平均；seed 42）、FINAL-B、
    FINAL-A（五 seed 平均；seed 42）。外部對照的 Masked ACC 是告訴任務的版本，填在 Masked 欄並加註；它的 WP、每任務參數、儲存、訓練秒數填「—」。
    **每任務訓練秒數**：本批在 fold 1 量測（CPU、8 執行緒）每個任務的「讀 train 特徵檔 ＋ 該系統需要的向量 ＋ closed-form 累加與求解」的 wall 秒數，四任務各列並給平均。
    head 的訓練秒數沒有在本批重訓（不訓練任何 head），表內的格只填本批量到的部分並標「不含 head 訓練」；head 訓練的既有紀錄
    （`moe1/i6_seed{43…46}/fold*_train.json` 的 `wall_s`，MOE-1 批）只寫在表註，不混進格內。zero-shot 為 0。

### 4. C（範例）

22. 兩張 slide 同 `FINAL_EXAMPLE.md`（fold 1：tcga_lung test 第 0 張；tcga_brca test 第 26 張）。由特徵檔重算，不寫任何既有產物：
    TP 的四個 AR 分數、s0 的 min／median／max 與第 64 名、四輪選出的 64 個（各輪 index；選出者在 s0 純排序的名次；與 FINAL-A(42) 同一張所選 64 個的交集張數，
    逐輪與合計）、‖v‖、A／B／W 的 shape 與累加張數、8 類 ridge 分數、τ̂ 兩類內的差與判定、告訴任務的判定；並列 FINAL-A(42) 在同一張的 ridge 差與判定。
    檢查：重算的 v 與快取的 v0 最大絕對差、TP 分數與 `moe5_example/example.json` 的最大絕對差。輸出 `ext2/example.json` 與 `FINALB_EXAMPLE.md`。

### 5. D（質性圖數據）

23. **D1**：列出 `can_dataset/<task>/` 下的目錄與檔案類型；每個任務讀一個特徵檔，記錄 Python 型別、shape、dtype、（若為 dict）鍵名；
    另在 `can_dataset` 與 `~/research/WSI_data` 下找 `.h5`／含 coord 字樣的檔。倍率只記錄檔名或目錄名能直接讀到的字樣，不推測。
24. **D2、D3 的條件**：特徵檔（或同目錄的對應檔）內有逐 patch 座標才做 D2；D2 做了且本機 .svs 與所選 slide 重疊、且 openslide 或 tifffile 已安裝才做 D3
    （不為此安裝套件、不下載切片）。任一條件不成立 → 在 REPORT 記錄哪個條件不成立與查到的事實，跳過。
    不論 D2 是否成立，都記錄本機 .svs 的 slide ID 與四個任務特徵檔、fold 1 test 的重疊張數（只比對檔名，不讀切片）。

### 6. 執行與交付

25. 階段：`a`（K9、K10、γ 重選、FINAL-B 全套與對照列）→ `cost`（K11、推論與訓練秒數）→ `b`（配對、bootstrap、外部對照、總表）→ `c`（範例）→ `d`（D1 與條件檢查）→ 報告。
    預估各階段都在數分鐘內；以 `caffeinate -ims` 執行；若任一階段預估超過 10 分鐘則改在 tmux 內以 `scripts/run_stage.sh` 執行並每 15 分鐘回報心跳。不關機、不停既有 tmux session。
26. `FINALB_RESULTS.md`：系統定義、γ 重選、A 的全部表；`REPORT_ext2.md`：狀態、PREREG-22 第 5 點的程式確認（附檔名與行號）、「拿掉 head」列的確認、A–D 的表、
    未完成項與原因、「失敗」節、DECISIONS 全文。只給數字與表格。
27. commit 前跑 `PYTHONNOUSERSITE=1 python -m pytest`；逐折 CSV 與 `ext2/a/` 的逐折 JSON 進版控（`.done` 不進）；不 merge、不 push 到 main。
