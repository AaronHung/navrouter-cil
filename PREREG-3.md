# PREREG-3 — NC-3 預先註冊（第五關：router 機制）

登記時間：2026-09-25，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`（AMENDMENT-1）；
本輪所有數字來自同一台、同一批。

## 判準與設定（原文）

- 基準：R3(k=8)＋Hard，L0 expert（與第四關 R 線相同）。
- R6 patch-codebook vote：每任務從該任務每張訓練 slide 隨機取最多 200 個 patch（seed 0），KMeans(k=8, n_init=10, random_state=0) 得 8 個 codeword（正規化後存，16 KB/任務）；測試時每個 patch 找所有已見任務 codeword 中 cosine 最大者並投給該任務，只算最大 cosine 前 10% 的 patch，多數決。
- R7 patch 對角高斯：同樣取樣，存每任務 patch 特徵的平均與變異數（4 KB/任務）；每個 patch 投給對角 Mahalanobis 距離最小的任務，只算距離最小的前 10% patch，多數決。
- R8 top-2 會診：用 R3(k=8) 的任務分數，margin = 第一名 − 第二名；margin < θ 時，前兩名任務的 expert 各自四輪選 64 張，各自對自己的 2 個類別文字算 cosine，在這 4 個類別中取 cosine 最大者；margin ≥ θ 時照 Hard。θ 在十折 validation 上從「validation margin 的 0%、10%、20%、30%、50% 分位數」中選，0% 等於基準。不增加儲存。
- 所有 key 只用該任務自己的訓練資料，在學該任務時建立。
- 選法：{基準, R6, R7, R8} 用十折 validation 平均 CIL 選定；test 全部報告。
- 判準（R-機制）：選出的不是基準，test CIL − 基準 ≥ +0.01 且贏 ≥ 7/10 折，儲存 ≤ 16 KB/任務。

## 操作定義（與判準同時登記）

1. **基準**：與 NC-2 相同的 R3(k=8) key（每任務 train slides 平均向量的 KMeans(k=8, n_init=10,
   random_state=0) centroid，正規化）、L0 expert（第一關 bank）、四輪 λ\*=1.5、Hard。
2. **取樣（R6、R7 共用）**：每（fold、任務）一個 `torch.Generator().manual_seed(0)`，依 train
   split 的 slide 順序，每張取 `randperm(n)[:200]`（n ≤ 200 時全取）。用原始 CONCH patch 特徵
   （未正規化）做 KMeans 與平均／變異數。
3. **R6**：codeword = KMeans centroid 正規化後的 8 × 512 向量。測試 patch 先正規化，對所有已見
   任務的 codeword 算 cosine；該 patch 的 owner = 最大 cosine 的 codeword 所屬任務、分數 m = 該
   最大 cosine。取 m 最大的前 ⌈0.1·n⌉ 個 patch 投票；票數同分時取投票 patch 的 m 總和較大者，
   再同分取候選順序中的第一個。
4. **R7**：每任務存取樣 patch 的逐維平均 μ 與母體變異數 σ²（ddof = 0，下限 1e-6）。距離
   d_τ(x) = Σ_d (x_d − μ_τ,d)² / σ²_τ,d。owner = argmin_τ d_τ(x)；取最小距離最小的前 ⌈0.1·n⌉
   個 patch 投票；票數同分時取投票 patch 的最小距離總和較小者。
5. **R8**：margin 為 R3(k=8) 在已見候選任務間第一名 − 第二名的分數差（t = 1 只有一個候選，
   不會診）。分位數 q ∈ {0, 0.1, 0.2, 0.3, 0.5}；每折的 θ = 該折 validation slides 在 t = 4
   的 margin 的 q 分位數（`torch.quantile`，線性內插），test 的各階段都用同一個 θ。會診時，
   前兩名任務 τ1、τ2 的 expert 各自四輪選 64 張、等權平均正規化，對自己的 2 個類別文字算
   cosine，4 個值中取最大者為預測類別。q 依十折 validation 平均 CIL 選定，同分取小。
6. **validation 平均 CIL**：每折 validation slides 在 t = 4 的 ACC（四任務 R[4][j] 平均），
   再對十折取平均。{基準, R6, R7, R8(q\*)} 依此選定；同分依此順序取前者（基準優先）。
7. **R-機制**：以 test 的 t = 4 ACC 逐折計算（選出者 − 基準）：十折平均 ≥ +0.01 且差 > 0
   的折數 ≥ 7，且儲存 ≤ 16,384 bytes／任務。
8. **儲存**：fp32。基準 16,384；R6 16,384（8 × 512 × 4）；R7 4,096（2 × 512 × 4）；
   R8 與基準相同（16,384，不增加）。
9. **Masked ACC**：在真實任務的 2 個類別中取 argmax，證據用「分派後的 expert」的四輪選片
   （Hard：router 選出的 τ̂；R8 會診時：最終預測類別所屬任務的 expert）。
10. **Forgetting、各階段**：沿用 PREREG.md 第 10 條。R8 另報 test t = 4（十折合計）會診的
    slide 比例，以及相對基準「改對」（基準錯 → R8 對）與「改錯」（基準對 → R8 錯）的張數。
