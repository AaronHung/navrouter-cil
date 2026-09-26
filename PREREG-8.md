# PREREG-8 — NC-8 預先註冊（主表定稿：所有系統同一批重算、圖的資料）

登記時間：2026-09-26，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`；本輪所有數字來自同一台、同一批。
全部只做推論，不訓練任何東西；expert 權重全部沿用已存的（L0、L1 r=2 v2、I6 r=2）。

## 內容（原文）

0. 設計決定（依 NC-7 事先訂的判準，不是本輪 test）：主系統 = D3 = AR（γ = 0.001）＋ I6(r = 2)。本輪不做任何選擇，只重算與彙整。

A. 主表（test，t = 4，十折 mean ± sd，兩序）
列：
1. zero-shot 8 類 top-64（不分派、不訓練）
2. LIN8（γ = 0.01）
3. R3(k=8)＋L1(r=2) v2
4. D2：AR＋L0
5. D1：AR＋L1(r=2) v2
6. D3：AR＋I6(r=2)（主系統）
7. AR-bal（γ = 1e-4）＋I6(r=2)
8. oracle 分派＋I6(r=2)（上限）
欄：ACC、Masked ACC、Forgetting、BWT、每任務 TP、每任務 WP、儲存（expert 合計、router 統計量；fp32）。
另報：D3 與第 1–5、7 列逐折相減（兩序），十折平均差與 D3 贏的折數。

B. 圖的資料（CSV，放 outputs/navcil/mac/nc8/fig/）
- 逐階段 ACC（t = 1–4，兩序，十折 mean 與 sd）：第 3、5、6 列。
- 轉交曲線：router = AR（γ = 0.001），expert = I6(r=2)，轉交比例 {0, 1, 2, 3, 5, 7, 10}%：剩下片子的 ACC（兩序）、被轉交片子中原本分派錯的比例、轉交片子的任務與類別組成。
- 分派混淆矩陣（4 × 4，十折合計）：R3(k=8)、AR、AR-bal 三個 router。

C. 每折原始值
- 所有列、所有指標的每折數值存成 outputs/navcil/mac/nc8/per_fold.json（之後做配對比較用）。

## 操作定義（與內容同時登記）

1. **同一批**：每折讀一次 train slides（只算 slide 的全部 patch 平均正規化 mean_vec，供 R3 key、AR、
   AR-bal、LIN8 的統計量）與一次 test slides（算出所有列需要的證據）。本輪不做選擇，不讀 validation。
   所有證據由已存權重在本批重新計算；不讀先前的快取。
2. **expert 證據**：四輪 K=64、每輪 16、λ\*=1.5，所選 patch 等權平均正規化，對 8 類文字的 cosine。
   L0 = 第一關 bank；L1(r=2) v2 = lora_v2（每序的底座為該序第一個任務的 L0）；I6(r=2) = i6/r2。
3. **router**：R3(k=8) key 以本批 train mean_vec 做 KMeans(k=8, n_init=10, random_state=0)；AR（γ = 1e-3）、
   AR-bal（γ = 1e-4，w_j = 1/n_j）與 LIN8（γ = 0.01）依該序逐任務累加，float64（AMENDMENT-2）；oracle =
   真實任務。Hard：分派到 τ̂ 後在 C_τ̂ 內取 argmax。
4. **zero-shot 8 類 top-64**：第 t 階段以對已見類別文字的最大 cosine 取 top-64，等權平均正規化，在已見類別中
   取 argmax。
5. **Masked ACC**：真實任務的 2 類內取 argmax；有 router 的列用分派後 expert 的證據（PREREG-3 操作定義 9）；
   zero-shot 列用同一個 top-64 向量；LIN8 用真實任務的 2 個輸出欄；oracle 列的 ACC 等於 Masked ACC。
6. **TP**：t = 4 每任務（分派任務 = 真實任務的比例）。zero-shot 與 LIN8 以預測類別所屬任務計；oracle 為 1。
7. **WP**：真實任務的 expert 的 Masked ACC（四輪）。zero-shot 與 LIN8 沒有 expert，不報。
8. **Forgetting、BWT**：沿用 PREREG-5 操作定義 10。
9. **儲存（fp32，不含所有列共用的類別文字 f_txt）**：expert 合計 —— L0：4 × 528,388；L1(r=2) v2：
   528,388 ＋ 3 × 8,212 = 553,024；I6(r=2)：4 × 4,132 = 16,528。router —— R3(k=8)：4 × 16,384；
   AR／AR-bal：A 1,052,676（共用）＋ 4 × 2,052；LIN8：A 1,052,676 ＋ 4 × 2 × 2,052；zero-shot、oracle：0。
10. **D3 逐折比較**：兩序各自逐折計算 t = 4 ACC 之差（D3 − 該列），報十折平均與差 > 0 的折數。
11. **轉交**：每折 test slides 在 t = 4 的 AR 分數第一名 − 第二名為把握度，轉交最低的 ⌈q·N⌉ 張；剩下片子的
    ACC = 四任務各自正確率的平均。「被轉交片子中原本分派錯的比例」與組成為十折合計。I6 與順序無關、t = 4
    的 AR 分派兩序相同，ACC 仍兩序分報。
12. **與先前報告比對**：各列的十折平均 ACC（有先前值者再加 Masked ACC、Forgetting）與先前報告相比，
    差 > 0.001 者逐列列出差異與原因。第 7、8 列先前沒有報告，標為新列。
