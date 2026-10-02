# PREREG-17 — MOE-1 預先註冊（M3 確認、ensemble 對照、CL 流程、訓練／推論落差、視角 × 判讀 2×2、gate）

登記時間：2026-10-02，MOE-1 任何程式執行前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；本輪所有數字來自同一台、同一批。
分支：`moe-1`（自 `moe-0` @ d56cbd1）。不合併；不修改既有程式、既有報告與既有 PREREG；新程式放新檔
（`scripts/moe1_*.py`、`scripts/moe1_run_all.sh`）。前置：`AMENDMENT-5.md`（PREREG-16 的 B4 減去 validation 平均是指令的錯；G-M3 維持未通過）。
報告：`outputs/navcil/mac/REPORT_moe1.md`（`scripts/moe1_report.py` 產生）。

本檔＝指令原文的「共同設定」「一致性檢查」「操作定義」「門檻」全文（第一部分），加上執行前同時登記的操作化細則（第二部分；不改變第一部分）。

---

## 第一部分：指令原文

### 共同設定

--device cpu、8 條執行緒；closed-form 一律 float64；十折；reverse 與 paper 兩序；跨任務平均用四任務等權；所有比較在同一批、同一台機器上算。沿用 MOE-0 的快取與程式（moe0_infer.py、b3_per_slide.csv、selection.json、results.json）。
「主系統」＝ AR（γ = 1e-3）＋ I6(r = 2, seed 42)，四輪選片、等權平均。(a) = I6 路徑的兩類 cosine 差 d_a；(b) = LIN8（γ = 0.01）在該任務兩類內的分數差 d_b；差一律是第一類 − 第二類。

### 一致性檢查（不過就停下該階段並記錄）

K1 主系統 seed 42 重算：test WP 0.9340、CIL ACC 0.9128，與 nc8/per_fold.json 逐折最大絕對差 ≤ 1e-9。
K2 M3 在「σ 用 t = 4 的 validation 重算」的版本：test Masked ACC 與 CIL ACC 與 moe0/results.json 的 z′、λ = 1.0 逐折最大絕對差 ≤ 1e-9（平均 0.9448、0.9226）。
K3 seed 42 用 S4 的訓練程式重訓一個 (fold 1, tcga_esca) 的 head，權重與既有檔案逐位元相同。

### 操作定義

M3 的融合分數：f = d_a/σ_a + β·d_b/σ_b，β = 1.0（MOE-0 在 validation 上選定，本批固定，不再重選）；f > 0 判第一類，f < 0 判第二類，f = 0 時用 (a) 的判定。σ 是樣本標準差，不減平均。
「σ 固定版」（M3 的正式版本，以下稱 M3）：任務 j 的 σ_a、σ_b 在「學任務 j 的那個階段」用該折該任務的 validation slides 算一次，之後不再更新；其中 σ_b 用的是當時（只含到任務 j 為止）的 LIN8 權重。之後各階段，d_b 用當時的 LIN8 權重算，除以固定的 σ_b。
「σ 重算版」：σ 用當下階段的權重在 validation 上重算（只作對照，CL 流程下不合法）。
CIL 的 TP 一律用 AR（γ = 1e-3）；CIL 判定 = 在 τ̂ 的兩類內用 τ̂ 的 head 與 τ̂ 的 σ 算 f。

S1 修好與弄壞（不重跑，用 moe0/b3_per_slide.csv 與 σ 重算版、seed 42、test、告訴任務）
- 每個類別（8 類）：主系統錯→M3 對（修好）、主系統對→M3 錯（弄壞）、兩者都錯、兩者都對的張數（十折合計）。
- 對「修好」「弄壞」「都錯」「都對」四組，各報：patch 數 N 的中位數、|d_a/σ_a| 與 |d_b/σ_b| 的中位數、AR 第一名與第二名分數差的中位數、TP 是否正確的比例。

S2 CL 流程（seed 42，只推論）
- 每個順序、每個階段 t = 1…4：主系統、M3、σ 重算版 的 ACC（只在已學類別中判）、Masked ACC、每任務 WP；Forgetting 與 BWT 用 Table 1 同一個函式。十折 mean ± sd。
- 舊任務的整片 expert 會不會變：每個任務 j、每個 t ≥ j，報 (b) 單獨的告訴任務正確率，以及 σ_b（重算值）相對固定值的比值。
- γ 統一：LIN8 改用 γ = 1e-3 時，(i) 驗證「任務 j 的 AR 分數 = 該任務兩個類別的 LIN8 分數之和」（test 全部 slide 最大絕對差，預期 < 1e-8）；(ii) 報 (b) 單獨與 M3 在 γ = 1e-3 下的 t = 4 結果。另報用 γ = 0.01 的 LIN8 兩類分數和當 TP 的 TP 正確率與 CIL ACC。不設門檻。

S3 訓練／推論落差 2×2（seed 42，只推論，validation 與 test）
- 選片 ∈ {一次取 s 前 64（不扣冗餘）、四輪各 16（λ* = 1.5，現行）} × 彙整 ∈ {等權平均（現行）、softmax(s[idx]) 加權（與訓練相同，用未做 z-score 的 s）}。彙整後 L2 正規化，與兩類文字比 cosine。
- 四格各報：告訴任務的 WP、CIL ACC（十折 mean ± sd）；與現行格逐折相減的平均差、贏折數。現行格必須重現 K1。不設門檻、不改主系統。

S4 多 seed 與 ensemble 對照
- 用與 nc7_i6.py 完全相同的設定重訓 I6(r = 2)，只把 seed 42 換成 43、44（建模型前與訓練開始時的 manual_seed，以及每個 epoch 的順序 generator 用 seed + ep）。十折 × 4 任務 × 2 個 seed = 80 次訓練。權重存 outputs/navcil/mac/moe1/i6_seed{s}/。
- 每個 seed（42、43、44）：主系統的 WP、Masked ACC、CIL ACC；M3（σ 固定版，σ_a 用該 seed 的 head）的同三項；M3 − 同 seed 主系統的逐折差、平均差、贏折數、Wilcoxon 雙尾 p；兩序。
- ensemble 對照（告訴任務的 WP 與 CIL ACC，σ 固定版的算法）：
  LL：d_a(seed s)/σ_a(s) + d_a(seed s′)/σ_a(s′)，(s, s′) ∈ {(42,43), (43,44), (44,42)}；
  LG：d_a(seed s)/σ_a(s) + d_b/σ_b，即 M3；
  GG：d_b(γ = 0.01)/σ_b + d_b(γ = 1e-3)/σ_b′；
  LLL：三個 seed 的 d_a/σ_a 相加。
  每個都報相對「seed s 的主系統」的增益（逐折差、平均、贏折數）。
- 另報三個 seed 的主系統彼此判定不同的張數（告訴任務，十折合計），作為 seed 造成的變動量。

S5 視角 × 判讀 2×2（seed 42，告訴任務；validation 與 test）
四個單獨的判讀器，都只輸出該任務兩類的分數差：
  GT：整片平均向量（mean_vec，512，L2 正規化）與兩類文字的 cosine 差。不訓練。
  GR：= (b)，LIN8（γ = 0.01）。
  LT：= (a)，主系統。
  LR：對每張 train slide，用它自己任務的 head 四輪選出 64 個 patch，等權平均後 L2 正規化得 v（512），接常數 1；以 v 為輸入、8 類 one-hot 為目標做與 LIN8 相同的累加式 ridge（A += VᵀV、每類一欄）。γ ∈ {1e-3, 1e-2, 1e-1}，以十折 validation 平均 WP 選（同分取小）。test 時用真實任務的 head 取 v。
報：四者單獨的 WP（十折 mean ± sd，另報十折合計的正確張數）；六種兩兩組合的「都對／只有前者對／只有後者對／都錯」張數（十折合計）與「至少一個對」比例；六種兩兩組合與四者全加的融合（各自除以該任務 validation 上的 σ 後等權相加）的 validation 與 test WP，以及相對 LT 的逐折差、平均、贏折數。不設門檻。
另報 LR 在 CIL 下的結果（TP = AR，v 取自 τ̂ 的 head，在 τ̂ 的兩類內判）。

S6 gate（seed 42；依賴 S2 的 σ 固定版程式）
- gate 的訓練資料只用當前任務：任務 j 的 train slides，其 d_a 必須是「沒看過這張 slide 的 head」算的，d_b 必須是 leave-one-out 的。
  d_a：把該折該任務的 train slides 依類別分層切成 3 份（random_state = 42）；每次用 2 份、以 nc7_i6.py 的相同設定（seed 42）訓練一個 head，對剩下 1 份做四輪推論得 d_a。共 10 折 × 4 任務 × 3 = 120 次訓練。test 時仍用原本（全部 train 訓練）的 head。
  d_b：學任務 j 那個階段的 LIN8（含到任務 j 為止的統計量），對任務 j 的 train slides 用 closed-form leave-one-out：h = xᵀ(A + γI)⁻¹x，ŷ_loo = (ŷ − h·y)/(1 − h)，逐類別欄計算後取兩類的差。
  z_a = d_a/σ_a、z_b = d_b/σ_b，σ 用 S2 的固定值。
- G1（每任務一個 β）：β_j ∈ {0, 0.25, 0.5, 1, 2, 4}，在上述訓練資料上取正確率最高者，同分取最接近 1 者。
- G2（每張 slide 的權重）：ρ = sigmoid(c0 + c1·|z_a| + c2·|z_b|)，f = (1 − ρ)·z_a + ρ·z_b；y ∈ {+1（第一類）, −1}；loss = mean log(1 + exp(−y·f)) + 0.01·(c1² + c2²)；初始 c = 0；L-BFGS，最多 200 步。每 (折, 任務, 順序) 一組 c，學完固定。
- 報：G1、G2、固定 β = 1 的 M3、主系統 的 test WP、Masked ACC、CIL ACC（兩序）；G1、G2 各自相對 M3 的逐折差、平均、贏折數、Wilcoxon p；每任務選到的 β_j 分佈與 c 的平均；test 上 ρ 的分佈（每任務的中位數與四分位）。
- 參考上限：S5 的 LT＋GR「至少一個對」比例。

S7（S1–S6 都結束後才跑；可以沒跑完）
- 再訓練 seed 45、46，重做 S4 中「每個 seed 的主系統與 M3」兩列。
- 隨機特徵：mean_vec（512）乘固定的隨機矩陣 R（512 × 2048，N(0,1)，generator seed 42）後 ReLU，接常數 1，做與 LIN8 相同的累加式 ridge（γ ∈ {1e-2, 1e-1, 1, 10}，validation 選，同分取小）。報：單獨的 WP；取代 (b) 之後的 M3（σ 固定版）；用它的兩類分數和當 TP 的 TP 正確率與 CIL ACC。揭露統計矩陣大小（bytes）。

每一步檢查梯度與參數有限（沿用既有檢查）。

### 門檻

G-M3c：seed 43 與 44 各自、兩序各自：CIL ACC(M3) − CIL ACC(同 seed 主系統) 的十折平均 ≥ +0.007，且差 > 0 的折數 ≥ 7。四個條件都滿足才算通過。
G-ENS：告訴任務的 WP 上，[LG(s) 的增益 − LL(s, s′) 的增益] 對三組 (s, s′) 取平均後的十折平均 ≥ +0.005，且差 > 0 的折數 ≥ 7。
G-GATE：G2 與 G1 中，以訓練資料上平均正確率較高者為代表（同分取 G1）；其 test CIL ACC − M3（固定 β = 1）的十折平均 ≥ +0.005 且差 > 0 的折數 ≥ 7，兩序都要。
S1、S2、S3、S5、S7 不設門檻。seed 42 的 test 數字在 MOE-0 已看過，本批 seed 42 的 M3 結果只作描述，不作確認。

---

## 第二部分：操作化細則（與第一部分同時登記；不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `outputs/navcil/mac/moe1/DECISIONS.md`（做了什麼判斷、為什麼），並印在報告內。

### 0. 共通

1. **主系統、資料、類別順序**同 PREREG-16 操作定義 1–3：seed 42 權重 `outputs/navcil/mac/i6/r2/fold{f}_{task}.pt`（唯讀）；
   8 類固定順序 ESAD, ESCC, CCRCC, PRCC, IDC, ILC, LUAD, LUSC；任務位置 esca = 0、rcc = 1、brca = 2、lung = 3，任務 p 的兩類為第 2p、2p+1 列，
   「第一類」= 第 2p 列。reverse = esca → rcc → brca → lung；paper = lung → brca → rcc → esca。任務 j 的「學習階段」t_j = 它在該序中的位置（1 起）。
2. **各階段的 AR／LIN8 權重**：`nc8_report.B8.W(fold, 序, t, γ[, lin8=True])`（NC-8 同一份封閉解，float64，統計量只含該序前 t 個任務的 train mean_vec）。
3. **σ**：`torch.std`（樣本標準差，ddof = 1）；融合時不減平均，z = d/σ。算術沿用 `moe0_common.zapply(center=False)`。
   σ_a(j, seed)：該折任務 j 的 validation slides 上、任務 j 的 head（該 seed）的 d_a；head 學完即固定，故與階段、順序無關。
   σ_b 固定值(j, 序)：該折任務 j 的 validation slides 上、階段 t_j 的 LIN8 的 d_b。σ_b 重算值(j, 序, t)：同樣的 slides、階段 t 的 LIN8。
   σ 固定版與 σ 重算版只差在 σ_b。
4. **f = 0**：用 (a) 的判定 = d_a ≥ 0 判第一類（與主系統 2 類 argmax 同分取第一類的規則一致）。沒有 (a) 的融合（GG、S5 的組合、S7 的隨機特徵）：
   f = 0 時看式子裡第一項的符號，≥ 0 判第一類。單獨的判讀器：d > 0 判第一類，d = 0 判第一類。
5. **指標**：
   - WP（告訴任務）：每張 slide 用其真實任務的 head／LIN8 欄位／σ，2 類正確率；每折每任務計，四任務等權。
   - Masked ACC：主系統 = Table 1 的 Masked ACC 欄（`nc2_report.hard` 的 masked：τ̂ 的 head 的證據在真實任務兩類內 argmax）；
     融合系統（M3、σ 重算版、G1、G2、各 ensemble）= 告訴任務的融合判定的 2 類正確率（同 PREREG-16 操作定義 16，即 K2 的 0.9448 那一欄）。
     因此融合系統的四任務等權 WP 與 Masked ACC 是同一個數；主系統兩者不同（0.9340 與 0.9312）。
     「M3 − 主系統」在 Masked ACC 上 = 融合的 Masked ACC − 主系統的 Masked ACC 欄；在 WP 上 = 融合的 WP − 主系統的 WP。
   - CIL ACC：TP = 該階段的 AR（γ = 1e-3）在已學任務中 argmax 得 τ̂；在 τ̂ 的兩類內，用 τ̂ 的 head、τ̂ 的 LIN8 欄位、τ̂ 的 σ（gate 另用 τ̂ 的 β／c）算 f。
   - 階段 t < 4：只用已學任務的 test slides；ACC_t、Masked ACC_t = 已學任務等權平均。Forgetting、BWT：`nc5_report.cil_full`（Table 1 同一個函式）。
   - 逐折相減：平均差；贏折數 = 差 > 1e-12 的折數（同 `moe0_report.paired`）；Wilcoxon = `scipy.stats.wilcoxon`（雙尾、`zero_method = "wilcox"`），十折全為 0 時 p = 不適用。
6. **K1**：由 `moe0/test_fold{f}.pt` 的 `I6_cos8` 與 AR 重算主系統，兩序：逐折 CIL ACC、逐折每任務 WP 與 `nc8/per_fold.json` 第 6 列的 `acc`、`wp_task`
   最大絕對差 ≤ 1e-9；十折全跑時另檢查平均四捨五入到小數第 4 位為 WP 0.9340、CIL ACC 0.9128。凡報告 seed 42 主系統的階段都先跑 K1。
7. **K2**（在 S2）：`moe0/results.json` 沒有存逐折絕對值，逐折期望值 = `nc8/per_fold.json` 第 6 列的逐折值（四任務 WP 平均／`acc`）
   ＋ `B4.zprime.paired.masked_vs_wp.per_fold`／`B4.zprime.paired.cil[序].per_fold`；與 σ 重算版 t = 4 的逐折 Masked ACC、CIL ACC 最大絕對差 ≤ 1e-9；
   十折全跑時另檢查平均四捨五入為 0.9448、0.9226。
8. **K3**：訓練函式 = `selector.cil_ops.train_selector`，呼叫方式與 `nc7_i6.train_one` 相同（r = 2、5 epochs、lr 5e-4、wd 1e-4），seed 為參數。
   以 seed 42、fold 1、tcga_esca、5 epochs 重訓，state_dict 每個張量與 `i6/r2/fold1_tcga_esca.pt` `torch.equal`。
   S4 在任何新訓練之前先跑；S6、S7 用同一個訓練函式，各自在訓練前也先跑一次 K3，不過就停下該階段。
9. **失敗處理**（今晚無人看管）：任何階段的一致性檢查不過或程式出錯 → 該階段停止、寫 `FAILED_<階段>.txt`、不重試、不改期望值；
   `moe1_run_all.sh` 接著跑不依賴它的階段（S2 失敗則不跑 S6；S7 在 S1–S6 都結束後才跑，其中 seed 45／46 的部分只在 S4 成功時跑）。
10. **冒煙測試**：只跑 fold 1（訓練 1 個 epoch；K3 仍為 5 epochs），輸出到 `outputs/navcil/mac/moe1_smoke/`，數字不寫進報告；
    十折平均的四捨五入檢查在冒煙測試中不適用。

### S1

11. 主系統判定取自 `b3_per_slide.csv` 的 d_a（> 0 第一類；CSV 只存 8 位小數，= 0 的列數另報）；M3 = σ 重算版（t = 4、reverse 序的 LIN8；
    σ 由 `moe0/val_fold{f}.pt` 算），d_a、d_b 取自 CSV。另以快取的未捨入值重算同一判定，報兩者不同的張數（預期 0）。
12. patch 數 N：既有快取 `cache/fold{f}_test_{task}.pt` 的 `n_patch`（檢查 slide id 逐張對齊），不讀特徵檔。
    AR 第一名 − 第二名：AR（γ = 1e-3、t = 4、reverse 序）四個任務分數。TP 正確 = AR argmax = 真實任務。

### S2

13. 三個系統各自以 `nc5_report.cil_full` 計 R、ACC_t、Masked ACC_t、Forgetting、BWT；每任務 WP 另列。主系統的 t = 4 結果即 K1。
14. 「(b) 單獨」= 階段 t 的 LIN8 在任務 j 兩欄的 d_b 判定（告訴任務）。比值 = σ_b 重算值(j, t)／σ_b 固定值(j)，每折計，報十折 mean ± sd。
15. γ 統一：(i) 每序、每階段 t、已學任務的全部 test slides：|AR_j − (LIN8_{2j} + LIN8_{2j+1})| 的最大值（兩者都是 γ = 1e-3）；
    只報數值，≥ 1e-8 也不停。(ii) γ = 1e-3 的 (b) 單獨與 M3（d_b 與 σ_b 都用 γ = 1e-3 的 LIN8，σ 固定版）的 t = 4 WP／Masked ACC／CIL ACC。
    LIN8（γ = 0.01）兩類分數和當 TP：τ̂ = argmax_j (LIN8_{2j} + LIN8_{2j+1})；報 TP 正確率（四任務等權）與主系統、M3 在此 TP 下的 CIL ACC。

### S3

16. 每張 slide、每個 head：s = s0 + g（head 的輸出，不再做 z-score）；一次取 = `one_shot(s)`（top-64）；四輪 = `four_round(Z, s, λ*)`；
    等權 = `mean_norm(Z, idx)`；softmax = `mean_norm(Z, idx, softmax(s[idx]))`；再與 8 類文字取 cosine。四個 head 都算（CIL 需要 τ̂ 的 head）。
    validation 的 CIL：TP = AR 作用在 validation 的 mean_vec。
17. 現行格（四輪、等權）：與 `moe0` 快取的 `I6_cos8` 最大絕對差 ≤ 1e-6 且各任務 2 類內 argmax 全同（同 PREREG-16 的 5a），並通過 K1。

### S4

18. 新 seed 的推論與 `moe0_infer.py` 相同（每張 slide 四個 head、四輪、等權平均、8 類 cosine），validation 與 test 各一次。
19. ensemble 一律在 t = 4、σ 固定版、兩序各算。LL(s, s′) 與 LG(s) 的增益相對 seed s 的主系統；GG、LLL 沒有單一的 s，對三個 seed 的主系統各報一次。
    GG 的 σ_b′ = γ = 1e-3 的 LIN8 的固定值。CIL：TP = AR，τ̂ 的各 head／各欄位／各 σ。
20. 「彼此判定不同的張數」= 三個 seed 的主系統告訴任務判定不全相同的 test slides 數（十折合計），另列三組兩兩不同的張數。
21. **G-M3c** 取自 S4 的 seed 43、44 兩列、兩序。
22. **G-ENS**：每折 D = (1/3) Σ_{(s,s′)} [WP(LG(s)) − WP(LL(s, s′))]（兩個增益相減後，seed s 主系統的 WP 相消）。
    LG 的 σ_b 固定值與順序有關，門檻原文未指定順序：兩序各算，兩序都滿足（十折平均 ≥ +0.005 且 D > 0 的折數 ≥ 7）才算通過。

### S5

23. GT：`mean_vec @ F.t()` 的第 2p、2p+1 列之差（mean_vec 取自 moe0 快取）。GR：LIN8（γ = 0.01、t = 4、reverse 序）。LT：seed 42 的 (a)。
    LR：v 接常數 1 成 513 維 float64；A、B 依 reverse 序累加四個任務的 train slides；W = solve(A + γI, B)；
    選 γ：validation 告訴任務 WP（四任務等權、十折平均）最大者，同分取小；選定後才算 test。
24. 兩兩組合的張數：validation 與 test 各報；「至少一個對」報十折合計比例，另報每折每任務計、四任務等權的十折平均。
25. 融合：Σ_k d_k/σ_k，σ_k = 該折該任務 validation 上的樣本標準差（t = 4）；項的順序 GT、GR、LT、LR；f = 0 依細則 4。
26. LR 在 CIL 下：TP = AR；v 取自 τ̂ 的 head；在 τ̂ 的兩欄內判；兩序。

### S6

27. 分層切 3 份：`sklearn.model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`，對象 = 該折該任務 train split（資料集順序）、
    標籤 = 任務內 0／1。每份當一次保留份；訓練用另外兩份（索引遞增排列），每個 epoch 的順序 = `randperm(子集大小, generator(42 + ep))`。
    d_a 與順序無關（120 次訓練兩序共用）。
28. d_b 的 leave-one-out：A = 該序前 t_j 個任務 train mean_vec（接常數 1、float64）的 Σ x xᵀ，γ = 0.01；
    y = 該 slide 在任務 j 兩個類別欄的 one-hot；ŷ_loo = (ŷ − h·y)/(1 − h) 逐欄計算後取第一類 − 第二類。
29. G1：正確率 = gate 訓練資料上 f = z_a + β·z_b 的 2 類正確率（f = 0 依細則 4）；同分取 |β − 1| 最小者，仍同分取較小的 β。
30. G2：float64；`torch.optim.LBFGS(lr=1, max_iter=200, line_search_fn="strong_wolfe")`，呼叫一次 `step(closure)`；
    每次 closure 檢查 loss 與梯度有限、結束時檢查參數有限，不有限即停下該階段。
31. **G-GATE** 的代表：每序各自決定；訓練資料上平均正確率 = 每折四任務等權、再十折平均（G1 用選定的 β_j、G2 用學完的 c）；較高者為代表，同分取 G1。
    兩序各自的代表都滿足（test CIL ACC − M3 的十折平均 ≥ +0.005 且差 > 0 的折數 ≥ 7）才算通過。
32. 參考上限在 S6 內由同一批資料直接計算（= S5 的 LT＋GR「至少一個對」），不依賴 S5 是否完成。

### S7

33. seed 45、46：訓練、推論、表格同 S4 的每 seed 兩列（主系統與 M3）。不納入任何門檻。
34. 隨機特徵：R = `torch.randn(512, 2048, generator=torch.Generator().manual_seed(42), dtype=torch.float64)`；
    φ = [ReLU(mean_vec · R); 1]（2049 維、float64）；累加式 ridge 與 LIN8 相同（每類一欄、各序各階段只含已學任務）。
    選 γ：validation 告訴任務 WP（t = 4、reverse 序、四任務等權、十折平均）最大者，同分取小。
    「取代 (b) 的 M3」：d_b 與 σ_b 都換成隨機特徵 ridge 的值（σ 固定版，兩序）。TP = argmax_j（任務 j 兩類分數和）；
    報 TP 正確率與此 TP 下主系統、取代後 M3 的 CIL ACC。統計矩陣大小：A（2049 × 2049）與 B（2049 × 8）的 float64 bytes，另列 fp32 換算。
    2049 × 2049 的 float64 封閉解超出 AGENTS.md 可攜規則 3 的例外範圍（513 × 513），依本批指令「closed-form 一律 float64」執行。

### 交付

35. `scripts/moe1_report.py` 產生 `outputs/navcil/mac/REPORT_moe1.md`：開頭三個門檻與 K1–K3；每階段一節並標明本檔落點；
    最後列失敗階段（`FAILED_*.txt` 摘要）、`DECISIONS.md` 全文、各階段實際耗時。只給數字與表格。
36. seed 42 的 test 數字在 MOE-0 已看過：本批 seed 42 的 M3 結果只作描述，不作確認；M3 的確認只看 G-M3c（seed 43、44）。
