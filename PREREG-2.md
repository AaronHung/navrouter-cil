# PREREG-2 — NC-2 預先註冊（第四關：R 線與 L 線）

登記時間：2026-09-25，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`；本輪所有數字來自同一台、同一批。

## 判準與設定（原文）

- 共同設定：expert 用第一關 bank；四輪選片 K=64、每輪 16、λ*=1.5；Hard 組合。
- R 線變體：R0 T-Hard 原規則（slide 全部 patch 平均正規化，對已見任務的「任務文字 prototype = 該任務 2 個類別文字平均再正規化」取 cosine 最大）；R1 patch-vote；R2 proto；R3 multi-proto（每任務訓練 slides 的平均向量做 k-means，k∈{2,4,8}，KMeans(n_init=10, random_state=0)，取所有已見任務 centroid 中 cosine 最大者）；R4 proto 與 patch-vote 分數在候選任務間各自 z-normalize 後相加；R5 evidence-proto（每任務存 expert τ 在 τ 自己訓練 slides 上四輪選中 64 張的平均正規化，再取平均正規化；測試時每個候選 expert τ 各選 64 張，取平均正規化後與自己的 evidence-proto 算 cosine，取最大者）。所有 key 只用該任務自己的訓練資料，在學該任務時建立。
- 選法：主 router 與 R3 的 k 都用十折 validation 的平均 CIL 選定；test 對所有變體都報告。
- R-pass：選出的 router，儲存 ≤ 16 KB／任務、不存任何 slide 資料，test CIL ACC 十折平均 ≥ 0.890。
- R-改進：相對 R2 的每折 paired 差 ≥ +0.01，且贏 ≥ 7/10 折。
- 無儲存 router：R0、R1 中較好者比 zero-shot 8 類 top-64（0.8270）高 ≥ 0.02 且贏 ≥ 7/10 折。
- L 線：L0 完整 expert；L1(r) 以該順序第一個任務的完整 expert 為底座並凍結，之後每任務只學第一層的 rank-r 增量 BA（B 初始化為 0）、第一層 bias 與第二層，r∈{1,2,4,8}，其餘訓練設定同第一關；L2 為各任務增量相加的合併 expert、不用 router、在已見類別裡判。兩序都跑。
- L-pass：最小的 r 使四任務平均 WP ≥ L0 − 0.01（十折、兩序都要），且 r ≤ 4。
- 完整系統列：R 線選出的 router＋L1(r*)，兩序報告，不設門檻。

## 操作定義（與判準同時登記）

1. **validation 的平均 CIL**：每折 validation slides 在 t = 4（四個任務皆已見；兩序相同）以
   Hard 組合的 ACC（四任務 R[4][j] 平均），再對十折取平均。R3 的 k 先依此選定；主 router
   在 R0–R5（R3 用選定的 k）中依此選定。同分取編號小者（R0 < … < R5；k 取小）。
   key 一律用同一折的 train split 建立。
2. **test CIL ACC 與 Forgetting**：沿用 PREREG.md 第 10 條，兩序各報。
3. **R-改進**：以 test 的 t = 4 ACC 逐折計算（選出的 router − R2）；十折平均 ≥ +0.01 且
   差 > 0 的折數 ≥ 7。
4. **無儲存 router**：R0、R1 以操作定義 1 的 validation 平均 CIL 取較好者（同分取 R0）；
   與 zero-shot 8 類 top-64 的 test t = 4 ACC 逐折比較：十折平均差 ≥ 0.02 且差 > 0 的折數 ≥ 7。
5. **儲存與 slide 資料**：以 fp32 計 key 的位元組數。R0（由 f_txt 推得，分類頭本來就存）與
   R1 為 0 bytes；R2、R4、R5 為 512 × 4 = 2,048 bytes；R3 為 k × 2,048 bytes。
   「slide 資料」指任何逐張的特徵、選片索引或 slide id；對訓練 slides 取平均或 k-means
   所得的向量不算 slide 資料。
6. **patch-vote 分數**：R1 與 PREREG.md 第 9 條相同（票數，同票比 max cosine 總和）；R4 用
   票數比例（票數 / 投票 patch 數）在候選任務間 z-normalize（母體標準差，sd = 0 記 0）。
7. **margin**：t = 4 時各 slide 第一名與第二名候選分數之差（各變體自己的分數單位）；報
   平均、中位數、第 10／90 百分位數，並分判對／判錯。
8. **L1 初始化與訓練**：底座 = 同一折第一關 bank 中該序第一個任務的 expert（reverse：ESCA；
   paper：LUNG），完全凍結；第一個任務直接使用底座、不訓練。之後每個任務各自從底座出發
   （不接續前一任務）：W1 = W1_base + B·A（A ∈ R^{r×514}、B ∈ R^{256×r}，A 以 seed 42 的
   kaiming_uniform(a=√5) 初始化、B = 0，縮放係數 1），第一層 bias 與第二層（Linear 256→1）
   以底座數值初始化後可訓練。seed 42、5 epochs、Adam(lr 5e-4、weight decay 1e-4)、top-64
   後 softmax(分數) 加權聚合、只用該任務 2 類文字的 CE，與第一關相同。
9. **L2 合併 expert**：第 t 階段 = 底座 + 前 t 個任務（該序）的增量相加：
   W1 = W1_base + Σ B_τA_τ，b1 = b1_base + Σ (b1_τ − b1_base)，第二層同法。text_nav_feats 的
   2 維摘要用所有已見類別的文字。四輪選 64 張、等權平均正規化，在已見類別裡取 argmax。
10. **WP**：test 上四任務 Masked ACC 的平均（四輪 λ*、等權平均正規化，同第一關 (e)）；
    L0 = 第一關 (e) 的逐折值。L-pass 要求兩序各自的十折平均 WP ≥ L0 十折平均 − 0.01。
11. **r\* 找不到時**：若沒有 r 通過 L-pass，完整系統列用「兩序中較低者的十折平均 WP 最高」
    的 r，並標明未通過 L-pass。
12. **L 行為指標**：對每個非底座任務 τ，在 τ 的 test slides 上比較 L1(r) expert τ 與 L0
    expert τ：patch 分數的 Pearson 相關（逐張算再平均）、四輪選中 64 張的 Jaccard（逐張
    平均）；‖B_τA_τ‖_F 逐任務報十折平均。L2 的 Jaccard：t = 4 合併 expert 與各任務 L1
    expert 在該任務 test slides 上四輪選片的 Jaccard。
