# PREREG-4 — NC-4 預先註冊（容量依需求 I7、router I5／I4、router 總表）

登記時間：2026-09-25，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`；本輪所有數字來自同一台、同一批。

## 判準與設定（原文）

A. I7 容量依需求（expert 用 L 線 v2 的權重，不重新訓練）
- 每個順序的第一個任務維持完整 expert 當底座。之後每個任務的候選容量：0z（zero-shot 選片：分數 = 對該任務 2 個類別文字的最大 cosine，四輪 K=64、每輪 16、λ*=1.5，0 參數）、0b（直接沿用底座 expert，0 參數）、r=1、2、4、8（v2 權重）。
- 選法（在學該任務時，只用該任務自己的 validation）：在「validation WP ≥ r=8 的 validation WP − 0.01」的候選中取參數最少者；0z 與 0b 同為 0 參數時取 validation WP 較高者。
- I7-pass：兩序的 test 四任務平均 WP ≥ 固定 r=2 系統的 WP − 0.005，且非底座任務的 expert 參數合計 ≤ 固定 r=2 的 50%。
- 另報：每任務選到的容量、每任務 test WP、配上 R3(k=8) 的 CIL ACC／Masked ACC／Forgetting（兩序）。

B. router（只推論；expert 用 L0，與第四、五關 router 評估相同）
- B1 I5 撞車加 key：每任務以其全部訓練 slides 的平均向量做 KMeans(k=4, n_init=10, random_state=0) 得 4 個基本 key（8 KB）。任務 t 到來時，用「已見任務的基本 key＋任務 t 的基本 key」分派任務 t 的訓練 slides；被分到其他任務的 slide 若 ≥ 5 張，對它們做 KMeans(k_e = min(4, 張數 // 5)) 得額外 key，屬於任務 t（≤ 8 KB）。每任務合計 ≤ 16 KB。與順序有關，兩序都報。
- B2 I4 假特徵線性 router：每任務存 4 個基本 key（同 B1）與該任務 slide 平均向量的逐維變異數（2 KB）。任務 t 到來時從頭訓練一個線性分類器（512 → 已見任務數）：任務 t 用真實訓練 slide 的平均向量；每個舊任務抽與任務 t 訓練張數相同數量的假特徵 = 隨機一個基本 key＋依逐維變異數的高斯雜訊，再正規化。CE，Adam lr 1e-3，weight decay 1e-4，200 步全批次，seed 0。儲存：key＋變異數＋線性權重，每任務 ≤ 16 KB。
- 上限參考（非 CL）：用四任務全部真實訓練 slides 一起訓練的同規格線性分類器。
- 選法：{基準 R3(k=8), B1, B2} 用十折 validation 平均 CIL 選定；test 全部報告。
- R-機制-2：選出的不是基準，test CIL − 基準 ≥ +0.01 且贏 ≥ 7/10 折，每任務儲存 ≤ 16 KB。

C. router 總表（不重跑，從既有報告整理）
一張表列出所有已測 router：text-class、R0（T-Hard 規則）、text-organ、R2 proto、R3(k=4／8)、B1、B2、R1 patch-vote、R6、R7、nav、R5、R8，以及上限參考。欄位：類型（文字／slide 層級／patch 層級／證據層級）、每任務儲存 bytes、每任務 TP、macro／micro TP、ESCA↔Lung 兩個方向的混淆張數、CIL ACC。

## 操作定義（與判準同時登記）

1. **WP**：該任務 slides 上的 Masked ACC；四輪 K=64、每輪 16、λ\*=1.5、所選 patch 等權平均正規化，
   同第一關 (e)。validation WP 用該折的 validation split，test WP 用 test split。
2. **候選**：0z = 分數為 patch 對該任務 2 類文字的最大 cosine（不經任何 expert）；0b = 同一折
   第一關 bank 中該序第一個任務的 expert，以該任務自己的 2 類文字計算 text_nav_feats；
   r = 1、2、4、8 = L 線 v2（AMENDMENT-1 後）的同一折、同一序權重。
3. **I7 選法**：逐（序、折、非底座任務）決定。合格 = validation WP ≥ r=8 的 validation WP − 0.01
   （r=8 恆合格）。參數量：0z、0b 為 0；r 為 770r + 513。取參數最少者；0z 與 0b 皆合格時取
   validation WP 較高者，同分取 0z。
4. **I7-pass**：每一序各自檢查：(a) 十折平均的 test 四任務平均 WP（底座任務用完整 expert）
   ≥ 固定 r=2（v2）同序的同一值 − 0.005；(b) 十折平均的「非底座 3 個任務的 expert 參數合計」
   ≤ 0.5 × 3 × 2,053 = 3,079.5。兩序都成立才通過。
5. **I7 的 CIL**：router = R3(k=8)（與 NC-2／NC-3 相同 key），Hard；分派到任務 τ 時使用 I7 為
   τ 選的 expert。Masked ACC 沿用 PREREG-3 操作定義 9（分派後 expert 的證據）。
6. **基本 key**：NC-1 train 快取中每張 train slide 的全部 patch 平均正規化向量（mean_vec），
   KMeans(k=4, n_init=10, random_state=0) 的 centroid 正規化。
7. **B1**：依該序逐任務建立。任務 t 的 train slides 以 cosine 最大的 key（已見任務的基本 key 與
   任務 t 的基本 key，不含先前的額外 key）分派；分到其他任務的張數 m ≥ 5 時，對這些 slide 的
   mean_vec 做 KMeans(k_e = min(4, m // 5), n_init=10, random_state=0)，centroid 正規化為任務 t 的
   額外 key。測試時每個已見任務的分數 = 對其所有 key（基本＋額外）的最大 cosine，取最大者。
   儲存 = (4 + k_e) × 2,048 bytes。
8. **B2**：逐序、逐階段 t 建立。`torch.manual_seed(0)` 後依序：對每個舊任務 j（依該序）抽 N_t 個
   假特徵（N_t = 任務 t 的 train 張數；基本 key 以均勻隨機取一個、雜訊 = randn(512) × √var_j，
   相加後正規化），再建 `nn.Linear(512, t)`（預設初始化）。資料 = 任務 t 的真實 mean_vec ＋ 舊任務
   假特徵；類別依該序的已見任務順序。全批次 CE、Adam(lr 1e-3、weight decay 1e-4)、200 步。
   t = 1 只有一個候選。var_j = 任務 j 的 train mean_vec 逐維母體變異數。測試輸入 = slide 的
   mean_vec，logit 最大者為分派任務。儲存／任務 = 基本 key 8,192 ＋ 變異數 2,048 ＋ 其在最終
   線性層的一列（513 × 4 = 2,052）= 12,292 bytes。
9. **上限參考**：同規格線性分類器（seed 0、200 步、Adam lr 1e-3、wd 1e-4、全批次），以四任務全部
   真實 train mean_vec 訓練；第 t 階段只在已見任務的 logit 中取最大。非 CL，只作參考，不參與選法。
10. **validation 平均 CIL**：validation slides 在 t = 4 的 Hard ACC（四任務平均）。與順序有關的
    B1、B2 取兩序、十折的平均；基準兩序相同。同分依 基準、B1、B2 的順序取前者。
11. **R-機制-2**：以 test 的 t = 4 ACC 逐折計算（選出者 − 基準）；與順序有關者兩序各自都要
    十折平均 ≥ +0.01 且差 > 0 的折數 ≥ 7；儲存取所有折、任務中的最大值 ≤ 16,384 bytes。
12. **總表**：TP 為 test t = 4 的十折平均；ESCA↔Lung 為十折合計張數；CIL ACC 為 test t = 4
    （L0 expert、Hard；NC-1 的 TP 變體取其「＋ Hard」列）。B1、B2 兩序分列。
13. **新的完整系統列**：只在 A 或 B 通過時列出：router =（B 通過時為選出者，否則 R3(k=8)）、
    expert =（A 通過時為 I7 的選擇，否則 L1(r=2) v2），Hard，兩序。
