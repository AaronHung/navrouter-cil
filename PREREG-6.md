# PREREG-6 — NC-6 預先註冊（主 router 改 AR、AR 的兩個調整、TSP、機制檢查、新主系統）

登記時間：2026-09-26，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`；本輪所有數字來自同一台、同一批。
全部只做推論，不訓練任何 expert。AR 類的累加、求解、白化一律 float64（CPU；AGENTS.md 可攜規則 3 的
AMENDMENT-2 例外）。

## 判準與設定（原文）

0. 設計決定（依 NC-5 的 validation，不是 test）
- 主 router 由 R3(k=8) 改為 AR。依據：REPORT_stage7 T4-a，十折 validation t = 4 CIL：AR 0.9166、R3(k=8) 0.8942。
- 儲存規則由「每任務 ≤ 16 KB」改為「不存任何 slide 或 patch 特徵；統計量可存，逐項揭露 bytes（fp32 與 fp64 都列）」。

A. AR 的兩個調整（L0 expert、Hard）
- A1 γ 範圍：NC-5 選出的 γ = 0.01 在候選下緣，本輪候選改為 {1e-4, 1e-3, 1e-2, 1e-1}。
- A2 AR-bal（依任務樣本數加權）：A = Σ_j w_j Xjᵀ Xj、bj = w_j Xjᵀ 1，w_j = 1 / n_j（n_j = 任務 j 的 train 張數），其餘同 AR。依序累加仍成立（w_j 在學任務 j 時已知）。
- 候選：{AR, AR-bal} × 四個 γ，共 8 個；以十折 validation t = 4 CIL 平均選定（同分取 AR、γ 較大者）。
- 選出者做 fold 1 一致性檢查（依序累加 vs 四任務一次解，W 最大差 < 1e-4），未過即停下回報。
- 報：8 個候選的 validation／test CIL；選出者的每任務 TP、macro／micro、ESCA↔Lung 混淆張數與真實類別（LUAD／LUSC、ESAD／ESCC）。

B. TSP 文字引導的組織型方向移除（只用已有的類別文字 f_txt，不多存東西）
- 方向集合（由已見任務的類別文字計算；t = 4 時為全部 4 個任務）：
  U_diff = 每個已見任務的「類別 2 文字 − 類別 1 文字」，共 t 個向量，QR 正交化；
  U_all = 全部已見類別文字，共 2t 個向量，QR 正交化。
- 投影 P = I − U Uᵀ。router 的輸入與 key 都先投影、再 L2 正規化。
- 候選四個：
  R3＋U_diff、R3＋U_all：沿用既有 R3(k=8) key，測試時對 slide 的 mean_vec 與 key 同時投影，再算 cosine；
  AR＋U_diff、AR＋U_all：train 與 test 的 mean_vec 都先投影，再依 AR（A 段選出的變體與 γ）重算 A、bj。
- TSP-pass-1（低儲存）：R3＋U_diff 與 R3＋U_all 以 validation 選一個；兩序各自 test CIL ≥ A 段選出者的 test CIL − 0.005；每任務儲存仍為 16,384 bytes。
- TSP-pass-2（加在 AR 上）：AR＋U_diff 與 AR＋U_all 以 validation 選一個；兩序各自 test CIL − A 段選出者 ≥ +0.01，且差 > 0 的折數 ≥ 7。
- 必報：四個候選的 validation／test CIL、TP micro、ESCA↔Lung 混淆；LUSC 與 ESCC 兩類 train slide 平均向量的 cosine，投影前與兩種投影後。

C. 機制檢查（不設門檻，必報）
- C1 比法對了之後輸入還重要嗎：A 段選出的 AR 設定，改用 NC-5 快取的 e0.75 背景向量、z0.75 背景向量、e 腫瘤半向量當輸入（train 與 test 用同一種），報 TP micro、兩個混淆張數、CIL。
- C2 γ 趨勢：A 段每個 γ（AR 與 AR-bal 各自）的 test TP micro 與 Lung→ESCA 張數。
- C3 是「重新加權」還是「判別訓練」：S = A[:512, :512] / N（A 為 AR 累加矩陣去掉常數維，N = 已見 train 總張數），x̃ = (S + γI)^(−1/2) x，γ 同 A 段選出值；在 x̃ 上做 R2（每任務一個平均向量、cosine）。報 TP micro、兩個混淆張數、CIL，並與 AR、原始 R2 並列。

D. 新主系統（兩序，不設門檻，必報）
- D1 = A 段選出的 router＋L1(r=2) v2；D2 = 同一 router＋L0。
- 報：t = 4 的 ACC、Masked ACC、Forgetting、BWT；t = 1–4 逐階段 ACC 與 Masked ACC；每任務 TP 與 WP；儲存分項（expert 參數、router 統計量；fp32 與 fp64）。
- 若 TSP-pass-1 或 TSP-pass-2 通過，另列 TSP 版本的 D1。

操作細節：
- 需要的 slide 向量都用既有快取（NC-1 的 mean_vec、NC-5 的背景與腫瘤半向量），不重讀 slide。
- 其餘設定（K=64、每輪 16、λ*=1.5、Hard、Masked ACC 的算法、十折與兩序）與 PREREG-3 至 PREREG-5 相同。

## 操作定義（與判準同時登記）

1. **A 段選法**：validation t = 4 的 Hard CIL（四任務平均），取十折 × 兩序的平均。同分依序取 AR
   優先、γ 較大者。t = 4 的 AR 在兩序數學上相同，仍兩序都算。
2. **AR-bal 的一次解**：A = Σ_j (1/n_j) X_jᵀ X_j、b_j = (1/n_j) X_jᵀ 1 在四個任務一次算出；與依序累加
   比較 t = 4 的 W（欄依 canonical 任務序對齊）。
3. **U 與 P**：第 t 階段以該序前 t 個任務的 f_txt 建立（f_txt 本來就是分類頭的一部分）。U_diff 的欄為
   f_txt[2p+1] − f_txt[2p]，U_all 的欄為 f_txt 的 2t 列；`torch.linalg.qr`（reduced，float64）取 Q，
   P = I − Q Qᵀ。
4. **R3＋U**：key = NC-2 的 R3(k=8) centroid（不重算）；第 t 階段 slide 的 mean_vec 與 key 都乘 P_t 後
   正規化，分數 = 對該任務 8 個 key 的最大 cosine。儲存不變。
5. **AR＋U**：第 t 階段，所有已見任務的 train mean_vec 乘 P_t、正規化後，以 A 段選出的變體與 γ 重算
   A_t 與 b_j（不重選 γ）；test 的 mean_vec 同樣處理。**揭露**：這表示第 t 階段要以新的投影重算
   舊任務的統計量，嚴格的持續學習下需保存舊任務特徵或改用不正規化的 P A Pᵀ；本輪以 t = 4 CIL 為判定
   值，逐階段的 Forgetting 只作參考。
6. **TSP 的選法**：各組兩個候選以 validation t = 4 CIL（十折 × 兩序平均）選一個，同分取 U_diff。
7. **TSP-pass-1**：兩序各自「選出的 R3＋U」的 test t = 4 CIL 十折平均 ≥ A 段選出者同序同值 − 0.005。
   **TSP-pass-2**：兩序各自逐折差（選出的 AR＋U − A 段選出者）十折平均 ≥ +0.01 且差 > 0 的折數 ≥ 7。
8. **LUSC–ESCC cosine**：每折以 train slides 計算；投影前 = 各類 mean_vec 平均後正規化，再取 cosine；
   投影後 = 每張 mean_vec 乘 P（t = 4，U_diff 或 U_all）並正規化、各類平均後正規化，再取 cosine。報十折
   mean ± sd。
9. **C1**：NC-5 快取的向量；train 的任務 j 用該序到 j 為止的已見集合（e 來源），validation 用全部 4 個
   任務，test 第 t 階段用該序前 t 個任務；AR 的變體與 γ 同 A 段選出者，不重選。
10. **C3**：第 t 階段 S_t = A_t[:512, :512] / N_t（A_t 為 A 段選出變體的累加矩陣，AR-bal 時為加權後的
    矩陣；N_t = Σ_{j≤t} n_j），W_t = (S_t + γI)^(−1/2) 以 `torch.linalg.eigh` 計算。key_j =
    normalize(W_t μ_j)，μ_j = 任務 j train mean_vec 的平均（即 R2 的統計量，存下即可，不需舊資料）；
    測試分數 = cos(normalize(W_t x), key_j)。原始 R2 在同一批重算並列。
11. **D 的 WP**：真實任務的 expert 的 Masked ACC（四輪、等權平均正規化）：D1 為 L1(r=2) v2、D2 為 L0。
    TP 為 t = 4 每任務。BWT、Forgetting 沿用 PREREG-5 操作定義 10。
12. **儲存揭露**：expert 參數 × 4（fp32）與 × 8（fp64）bytes；AR 的 A 共用 513 × 513 × 4／8 bytes，
    每任務 b_j 513 × 4／8 bytes；AR-bal 的 w_j 已乘入 A 與 b_j，不另存。R3 key 16,384（fp32）／
    32,768（fp64）bytes。TSP 不增加儲存。
