# PREREG-7 — NC-7 預先註冊（I6：以 zero-shot 分數為底座的低秩 expert；分派不確定時轉交）

登記時間：2026-09-26，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`；本輪所有數字來自同一台、同一批。
A 段要訓練；B 段只做推論。長時間工作在 tmux 裡以 `scripts/run_stage.sh` 執行，每 15 分鐘心跳。

## 判準與設定（原文）

A. I6 expert（每任務獨立訓練，與順序無關）
- 形式：score(x) = s0(x) + g(u)。
  s0 = 該任務 2 類文字對 patch 的最大 cosine（text_nav_feats 的第 1 維），在 slide 內做 z-score；不訓練。
  u = [Z ; text_nav_feats(Z, f_task)]（514 維，與 EvidenceSelector 相同）。
  g(u) = w2ᵀ GELU(A u + b1) + b2，A ∈ R^{r×514}（kaiming_uniform a=√5）、b1 ∈ R^r、w2 ∈ R^r（初始化為 0）、b2 ∈ R（0）。w2 = 0 使訓練開始時 score = s0（與 LoRA 的 B = 0 同一精神）。
  每任務參數 = 516r + 1。r ∈ {1, 2, 3}（依序 517、1,033、1,549，都不超過 L1 r=2 的 2,053）。
- 訓練：與 L 線 v2 完全相同的 train_selector 設定（seed 42、5 epochs、lr 5e-4、wd 1e-4、同一個 loss 與 batch 流程）；每折每任務訓練一次（不分順序）。r = 1 若走到 gemv，比照 AMENDMENT-1 用廣播乘法，並做逐步 finite 檢查。
- 評估：四輪 K=64、每輪 16、λ*=1.5，所選 patch 等權平均後正規化，凍結 CONCH 頭（與第一關 (e) 相同）。
- 選 r：在十折 validation 的四任務平均 WP ≥ L0 的 validation WP − 0.01 的 r 中取最小者（r*）；都不符合則取 validation WP 最高者，並標記「未符合」。
- I6-pass：r* 符合上述條件，且 test 四任務平均 WP（十折）≥ 0.9299（L0 0.9399 − 0.01），且每任務參數 ≤ 2,053。
- D3 = AR（γ = 0.001，與 NC-6 相同）＋ I6(r*)，Hard，兩序：ACC、Masked ACC、Forgetting、BWT、t = 1–4 逐階段 ACC；與 D1（AR＋L1 r=2 v2）逐折相減，報平均差與贏折數。D3-次要判準：兩序 test ACC 皆 ≥ D1 − 0.005。
- 必報的狀態與行為：每任務學到的 g 的標準差對 s0 標準差的比值（test slides 平均）；I6 四輪 64 張與 zero-shot top-64（第一關 (b)）、與 L0 的 Jaccard；每任務 WP；四個任務 expert 合計 bytes（fp32），並與 D1 的 553,024 bytes 並列。

B. 分派不確定時轉交（不設門檻，只推論；router = AR γ = 0.001，expert = D1）
- 每張 test slide 的分派把握度 = AR 分數第一名減第二名。依把握度由低到高轉交 {0%, 1%, 2%, 5%, 10%} 的片子給病理醫師（不計入正確率）。
- 報：各轉交比例下，剩下片子的 ACC（t = 4、兩序）；被轉交的片子依真實任務、真實類別的組成（十折合計）；轉交 5% 時，原本分派錯的片子有多少比例被轉交。

## 操作定義（與判準同時登記）

1. **s0**：text_nav_feats(Z, f_task) 第 1 維，slide 內 (s − mean) / (std + 1e-6)（std 為樣本標準差，
   與第一關的 zscore 相同）；不訓練、不存參數。
2. **避開 gemv**：r = 1 的 A u 以 `(u * A[0]).sum(-1, keepdim=True)` 計算；所有 r 的 w2ᵀ GELU(·) 以
   `(GELU(·) * w2).sum(-1)` 計算（逐元素，不經 BLAS gemv）。r = 2、3 的 A u 用 `u @ Aᵀ`（gemm）。
   train_selector 每步檢查梯度與參數有限（AMENDMENT-1），出現 NaN／Inf 即停。
3. **初始化**：`torch.manual_seed(42)` 後建立模型（A 以 kaiming_uniform(a=√5)、b1 為 0、w2 為 0、
   b2 為 0），再交給 train_selector（其內再以 seed 42 與每 epoch 的 42 + epoch 洗牌，與 L 線 v2 相同）。
4. **L0 的 validation WP**：第一關 bank 的 L0 expert 在 validation slides 上四輪選片的 Masked ACC
   （NC-2 validation 快取），四任務平均，再取十折平均。r 的選法以十折平均比較：
   WP_val(r) ≥ WP_val(L0) − 0.01。
5. **I6-pass**：r\* 符合操作定義 4 的條件，且 r\* 的 test 四任務平均 WP 十折平均 ≥ 0.9299，且 516r\* + 1
   ≤ 2,053。
6. **D3 與 D1**：router = AR（mean_vec、γ = 0.001、float64，與 NC-6 相同）；Hard；分派到任務 τ 時使用
   τ 的 I6(r\*) expert（D1 用 L1(r=2) v2）。兩序各自逐折計算 t = 4 ACC 之差（D3 − D1），報十折平均與
   差 > 0 的折數。D1 在本批重算。D3-次要判準：兩序各自 D3 十折平均 ACC ≥ D1 十折平均 ACC − 0.005。
7. **g／s0 比值**：每個任務 expert 在該任務的 test slides 上逐張計算 std(g(u)) / std(s0)（樣本標準差），
   逐張平均、再十折平均。
8. **Jaccard**：該任務 test slides 上，I6(r) 四輪 64 張 vs (a) zero-shot top-64（分數 = 對該任務 2 類
   文字的最大 cosine，一次 top-64，即第一關 (b)）；vs (b) L0 四輪 64 張（第一關 test 快取的 four_idx）。
   逐張平均、再十折平均。
9. **轉交**：router 與 expert 同 D1；每折 test slides 在 t = 4 的 AR 分數第一名 − 第二名為把握度；每折
   轉交把握度最低的 ⌈q·N⌉ 張（N = 該折 test 張數；同值依 topk 決定）。剩下片子的 ACC = 四任務各自
   正確率的平均（同 CIL 的 ACC）。組成與「分派錯被轉交比例」為十折合計（分派錯 = 分派任務 ≠ 真實任務）；
   t = 4 的 AR 分派兩序相同，組成與比例只報一次，ACC 兩序分報。
10. **儲存**：I6 四個任務合計 4 × (516r + 1) × 4 bytes（fp32）；D1 = 528,388 ＋ 3 × 8,212 = 553,024
    bytes（fp32）。router 的儲存與 NC-6 相同，不重複計。
