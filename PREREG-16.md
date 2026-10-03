# PREREG-16 — MOE-0 預先註冊（MoE 前置診斷：只推論；B 段）

登記時間：2026-10-02，B 段任何程式執行前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），CPU、`torch.set_num_threads(8)`、torch 2.11.0；本輪所有數字來自同一台、同一批。
分支：`moe-0`（自 main @ 70d20d1）。不訓練任何模型；不修改既有程式、既有報告與既有 PREREG；新程式放新檔。
報告編號：`outputs/navcil/mac/REPORT_moe0.md`（表 T1–T5 對應 B1–B5）；A 段稽核另存 `outputs/navcil/mac/AUDIT_wp.md`（不在本檔範圍）。

---

## 判準與設定（B 段原文）

「主系統」＝ ridge TP（程式代號 AR，γ 用既有選定值）＋ residual scoring head（程式代號 I6），與 9/30 Table 1 的 0.9128 是同一組設定。

B. 診斷（只推論）。先寫 PREREG（檔名沿用 repo 既有編號的下一號）並 commit，之後不得修改，再跑。PREREG 內容＝B 段全文＋文末的「判讀門檻」。
共同設定：t = 4，十折，reverse 與 forward 兩序；所有比較都在同一批、同一台機器上算；closed-form 的部分用 float64；跨任務的平均一律用四任務等權平均（與 9/30 Table 1 相同）。

B1 每任務明細：每任務的 test slide 數、TP 正確率、告訴任務時的 WP 正確率、CIL 正確率；WP 錯誤張數按真實類別列出（十折合計）。

B2 交叉 head 矩陣（4 × 4）：告訴真實任務 j；s0 與 head 的輸入都用任務 j 的兩個亞型文字計算，只把 head 的權重換成任務 k 的；報 2 類正確率（十折平均）。對角線應重現 B1 的 WP。另加一列 g = 0（只用 s0）。同時回報每個任務兩類的順序（哪一類排第一）；若 ESCA 與 LUNG 的腺癌／鱗癌順序不同，另報把順序對齊後的結果。若 head 輸入的定義讓「換 head 權重」不成立，回報原因並跳過這一項。

B3 整片統計與局部證據的錯誤重疊：告訴真實任務，比較 (a) 主系統 WP 的判定，與 (b) 8 類 ridge（程式代號 LIN8，γ 用既有選定值）只在真實任務兩類內取 argmax 的判定。每任務、十折合計，列出：兩者都對、只有 (a) 對、只有 (b) 對、兩者都錯的張數；以及「至少一個對」的比例。把每張 slide 兩邊的兩類分數差存成 per-slide 檔（fold、slide id、任務、真實類別、(a) 分數差、(b) 分數差）。

B4 簡單融合：(a)、(b) 的兩類分數差各自用該任務 validation slides 的平均與標準差做 z-score（不用 train slides，因為 head 與 ridge 都在 train 上學過，分數會偏大）；融合分數 = z_a + λ·z_b，λ ∈ {0, 0.25, 0.5, 1, 2}；以十折 validation 的平均 Masked ACC 選一個所有任務共用的 λ（同分取小）。報 test 的 Masked ACC 與 CIL ACC（TP 用 ridge），並與主系統逐折相減：平均差、贏折數、Wilcoxon p。

B5 soft gating 基準：π = softmax(ridge 任務分數 / T)，T ∈ {0.01, 0.03, 0.1, 0.3, 1}，以十折 validation 的平均 CIL ACC 選（同分取小）。四個任務的 head 都跑；類別 c 的分數 = π[c 所屬任務] × q[c]，q 為該任務兩類 logits 的 softmax；在 8 類中取最大。報 CIL ACC（兩序）、與主系統逐折相減（平均差、贏折數）、判定改變的張數（改對幾張、改錯幾張）。

## 判讀門檻（原文；只用來決定下一步做哪個設計，不是論文結果）

- G-M3：B3 的「至少一個對」比例 − 主系統告訴任務時的 WP ≥ 0.02，且 B4 選出的 λ 在 validation 上的 Masked ACC − 主系統 ≥ 0.005。
- G-M2：B2 中 LUNG 的 head 用在 ESCA、ESCA 的 head 用在 LUNG，兩格都不低於各自對角線 0.01 以上。
- B5 不設門檻。

---

## 操作定義（與判準同時登記；不改變上文）

### 0. 名稱與資料

1. **主系統**＝ NC-8 主表第 6 列 D3：AR（mean_vec 接常數 1、513 維、γ = 1e-3、float64）＋ I6(r = 2)（1,033 參數／任務）；
   TP = AR 分數 argmax；WP = 分派任務 τ̂ 的 I6 expert 四輪（K = 64、每輪 16、λ\* = 1.5）選片、等權平均後 L2 正規化、
   對 8 類文字取 cosine，在 τ̂ 的 2 類內取 argmax（`nc2_report.hard`）。Table 1 的 ACC = 0.9128（REPORT_stage10.md:359）。
2. **「forward」序 = 本 repo 的 `paper` 序**（lung → brca → rcc → esca）；「reverse」= `reverse` 序（esca → rcc → brca → lung）。
   t = 4 時兩序看過的任務集合相同，ACC、Masked ACC、WP、TP 理應相同（AR 只差累加順序的浮點尾數）；仍兩序各算各報，
   不相同時逐序報告，不合併。
3. **資料與快取**：特徵檔 `can_dataset/<task>/feats-l1-s256_CONCH/`；權重 `outputs/navcil/mac/i6/r2/fold{f}_{task}.pt`
   （唯讀）；NC-8 同批快取 `outputs/navcil/mac/cache/nc8_fold{f}_{train,test}_{task}.pt`（唯讀）：train mean_vec 供 AR／LIN8，
   test 的 sid、labels、mean_vec、`I6_cos8` 供主系統。8 類固定順序 ESAD, ESCC, CCRCC, PRCC, IDC, ILC, LUAD, LUSC；
   任務位置 esca = 0、rcc = 1、brca = 2、lung = 3，任務 p 的兩類為第 2p、2p+1 列（「第一類」= 第 2p 列）。
4. **需要重算的範圍**（既有快取不足；估計時間依 NC-8 test 計時 445 s／2,835 張／每張 20 次四輪選片）：
   (i) validation 全部 10 折（2,878 張）：全部 patch 平均 mean_vec，以及四個任務 expert 各自的四輪 8 類 cosine（既有 `i6/eval_fold{f}.pt`
   的 validation 只有自家任務 expert、且沒有 mean_vec）；每張 4 次四輪選片，估 ≤ 3 分鐘。
   (ii) test 全部 10 折（2,835 張）：B2 的跨 head（3 個非對角 head × 該任務文字）與 g = 0 共 4 次四輪選片；另重算主系統四個 expert
   各 1 次（共 8 次）以做一致性檢查；估 ≤ 3 分鐘。
   兩部分合計估 6 分鐘以內；每折完成寫 done 標記，逐張記錄 `t_read_s`、`t_compute_s`；新快取存 `outputs/navcil/mac/moe0/`（`.pt` 不進版控）。
5. **一致性檢查（任一不符即停下回報，不重試、不改期望）**（比對方式沿用 AMENDMENT-4）：
   (a) 重算的 test `I6_cos8`（四個 expert）與 NC-8 快取：四輪選中的 64 個 patch index 無法比對（NC-8 未存），改比 8 類 cosine 最大絕對差
   ≤ 1e-6 且各任務 2 類內 argmax 全同；
   (b) 重算的 test mean_vec 與快取最大絕對差 ≤ 1e-6；
   (c) 由快取重算的主系統 t = 4 ACC（兩序、逐折）與 `nc8/per_fold.json` 第 6 列逐折最大絕對差 ≤ 1e-9；平均四捨五入到小數第 4 位須為 0.9128；
   (d) B2 對角線（head j、任務 j 文字）與 `I6_cos8[:, j]` 的 2 類內 argmax 全同；
   (e) 重算的 validation 自家任務 expert 8 類 cosine 與 `i6/eval_fold{f}.pt` validation 的最大絕對差 ≤ 1e-6、argmax 全同；
   (f) 主系統 validation 四任務 WP 十折平均四捨五入到小數第 4 位須為 0.9194（REPORT_stage9.md:19）。

### 1. B1

6. 每折每任務：TP 正確率 = AR 分派 = 真實任務的比例；WP = 真實任務 expert 在 2 類內的正確率；CIL 正確率 = 主系統最終判定（分派任務 τ̂ 的 expert，
   在 τ̂ 的 2 類內取 argmax）等於真實 8 類標籤的比例。三者各取十折平均（與 Table 1 一致），並另列四任務等權平均。test slide 數、WP 錯誤張數為十折合計；
   WP 錯誤按真實類別（8 類）列出，並列出錯判成同任務另一類的方向（只有 2 類，故錯誤方向唯一）。

### 2. B2

7. **「換 head 權重」成立性**：I6 的 head 輸入 u = [Z(512); text_nav_feats(Z, f_task)(2)]（514 維，與任務無關的維度）；任務 k 的權重 (A, b1, w2, b2) 形狀與任務無關，
   可直接套用在以任務 j 文字算出的 u 上，故不跳過。s0 = text_nav_feats 第 1 維（對任務 j 兩類文字的最大 cosine）slide 內 z-score，與 head 權重無關。
8. 對每張任務 j 的 test slide 與每個 k ∈ {esca, rcc, brca, lung}：score = s0_j + g_k(u_j)；g = 0 一列：score = s0_j。其後與主系統相同（四輪選片 K = 64、每輪 16、λ\* = 1.5，
   等權平均、L2 正規化、對任務 j 兩類文字 cosine、2 類 argmax），2 類正確率每折計後取十折平均（4 × 4 矩陣、列 = 真實任務 j、欄 = head k）。
9. **類別順序**：每任務 label 0 / 1 依 `can_dataset/<task>/table` 的 `subtype`／`label` 欄與 `data/class_prompts.json` 的 classnames 順序回報。
   若 ESCA 與 LUNG 的腺癌／鱗癌位置不同則另報「順序對齊」結果：對齊 = 對 ESCA 的 head 與 LUNG 的文字（或反向）互換兩類文字列的順序後重算
   （只在順序不同時執行；順序相同則寫「順序已相同，不需對齊」）。

### 3. B3

10. (a) 判定 = 真實任務 expert 的 I6 四輪 8 類 cosine 在該任務 2 類內 argmax（= 主系統告訴任務時的 WP）。(b) 判定 = LIN8（mean_vec 接常數 1、γ = 0.01、float64、
    每類一個欄位，與 `nc8_report.B8.W(..., lin8=True)` 相同；用四個任務的 train mean_vec）的 8 個 logits 只取真實任務兩類，argmax。
11. 分數差 = 第一類（第 2p 列）分數 − 第二類（第 2p+1 列）分數：(a) 為 8 類 cosine 之差、(b) 為 LIN8 logits 之差。
    per-slide 檔：`outputs/navcil/mac/moe0/b3_per_slide.csv`（欄位 fold、slide_id、task、true_class、d_a、d_b）；兩序 t = 4 相同，只存 reverse 序一份，並在報告中註明。
12. 「至少一個對」比例：每折每任務計，取十折平均，再四任務等權平均（用於 G-M3）；表內另列十折合計張數的比例。

### 4. B4

13. z-score：每折、每任務 q，以該折該任務 **validation** slides 的 d_a、d_b 各自的平均 μ 與樣本標準差 σ（ddof = 1）：z = (d − μ)/σ。
    d_a 取任務 q expert 的 8 類 cosine 之差、d_b 取 LIN8 logits 之差（第一類 − 第二類）。此 μ、σ 同時用於「被分派到任務 q 的任何 test slide」。
14. 融合分數 f = z_a + λ·z_b；f > 0 判第一類，否則第二類。λ ∈ {0, 0.25, 0.5, 1, 2}。
    **λ = 0 是「只用 (a)、但已減去 validation 平均並除以標準差」的版本**，其決策門檻 d_a = μ_a，與主系統（門檻 d_a = 0）不同；
    主系統本身不在 λ 網格內，另列。
15. **選 λ**：每折 validation 四任務等權平均的告訴任務 Masked ACC（每個 validation slide 用其真實任務的 expert 與 LIN8），再取十折平均；取最大者，同分取小的 λ。
    只讀 validation；選定寫入 `outputs/navcil/mac/moe0/selection.json` 並 commit 之後才讀 test（見 24）。
16. **test 報告**：
    - Masked ACC = 告訴任務、融合判定的 2 類正確率，四任務等權，十折。與兩個基準逐折相減：
      **主要基準 = 主系統告訴任務的 WP（Table 1 oracle 列 0.9340，λ 網格之外的未經 z-score 版本）**；
      次要基準 = Table 1 主系統的 Masked ACC 欄（0.9312；該欄以分派 expert 的證據在真實任務兩類內取 argmax）。
    - CIL ACC：TP = AR 分派 τ̂；用 τ̂ 的 expert 與 τ̂ 的 LIN8 欄位、τ̂ 的 validation μ／σ 算 f，在 τ̂ 的 2 類中選；8 類標籤與之相等為正確；四任務等權，十折；與主系統 CIL ACC（0.9128）逐折相減。
    - 逐折相減：平均差、贏折數（差 > 0）、`scipy.stats.wilcoxon`（雙尾，`zero_method = "wilcox"`；10 折全為 0 時回報 p = 不適用）。兩序各報。
17. **附帶（不用於門檻）**：「只除以標準差、不減平均」（z′ = d/σ）的同樣流程（同一網格、同一選法、同一報告），用來分離「重新置中」與「融合」的效果。

### 5. B5

18. π = softmax(AR 任務分數 / T)：AR 分數 = AR（γ = 1e-3）在 mean_vec 上對四個任務的輸出（513 維、float64）。
19. q：任務 τ 的 expert（在該張 slide 上以 τ 的文字算 s0、g，四輪選片，等權平均 L2 正規化）對任務 τ 兩類文字的 cosine × logit_scale，再對這兩個值做 softmax；
    logit_scale = CONCH 的 logit_scale（56.3477，`cache/text/f_txt_*.pt`，與訓練所用的 `conch_classify` 相同）。
20. 類別 c 的分數 = π[task(c)] × q[c]；8 類取最大。**選 T**：validation 全部 slides（四個 expert 都跑）、每折四任務等權 CIL ACC、再取十折平均，最大者勝，同分取小的 T；
    兩序各自選。只讀 validation，寫入 `selection.json` 並 commit 後才讀 test。
21. 報告：CIL ACC（兩序，十折 mean ± sd）、與主系統逐折相減（平均差、贏折數；另附 Wilcoxon p 作參考）、判定改變張數（十折合計）：
    主系統判定與 soft gating 判定不同的張數，分「原錯→改對」「原對→改錯」「錯→另一個錯」。

### 6. 門檻的操作化

22. **G-M3**：需同時滿足：
    (i) U − WP₄ ≥ 0.02，U = 「至少一個對」比例（操作定義 12，每折每任務計、十折平均、四任務等權）、WP₄ = 主系統告訴任務時 WP 的四任務等權十折平均（0.9340）；
    (ii) B4 在操作定義 15 選出的 λ\* 的 validation Masked ACC（四任務等權、十折平均）− 主系統 validation Masked ACC（= 主系統 validation WP，0.9194）≥ 0.005。
    「主系統」在 validation 上只能是告訴任務的 WP（validation 沒有分派）。兩項都滿足才算通過；(i)、(ii) 各自的數值與是否滿足都要列出。
23. **G-M2**：以 B2 十折平均 2 類正確率矩陣（列 = 真實任務、欄 = head）：格 (ESCA, head LUNG) 與格 (LUNG, head ESCA)，
    各自須 (對角線 − 該格) < 0.01（嚴格小於；「不低於各自對角線 0.01 以上」＝ 沒有比對角線低 0.01 或更多）；對角線分別為 (ESCA, head ESCA)、(LUNG, head LUNG)。
    兩格皆滿足才算通過。若順序不同需要對齊，以「對齊後」的結果為準並同時列出未對齊結果。
24. **不看 test 的選擇**：選 λ、選 T 的程式只讀 validation 快取；`selection.json` commit 之後才執行讀 test 的程式，後者啟動時檢查 `selection.json` 已 commit 且未修改，否則停止。
25. 失敗就停：任何指令非零結束、一致性檢查不符，立刻停下回報，不重試、不跳過、不改期望值。

### 7. 交付

26. `outputs/navcil/mac/REPORT_moe0.md`：T1–T5 對應 B1–B5，逐條標明 PREREG 落點；寫明 G-M3、G-M2 是否通過，與 (i)(ii)、兩格的數值；只給數字與表格。
    另交 `outputs/navcil/mac/AUDIT_wp.md`（A 段，不在本 PREREG 判準內）。pytest 通過後 commit 並 push `moe-0`，不合併。
