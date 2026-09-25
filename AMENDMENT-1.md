# AMENDMENT-1 — L 線 r=1 的數值修正

日期：2026-09-25。PI 授權修正 r=1，範圍以本文件為限。**PREREG-2.md 不修改**；其設計與判準全部照舊。

## 現象（NC-2，commit 3a9e622）

- L1(r=1) 的 60 個增量（十折 × 兩序 × 3 個非底座任務）中，51 個在訓練中參數變成 NaN
  （reverse 28/30、paper 23/30）。r = 2、4、8 的 180 個增量皆無 NaN。
- 重播 fold 1 reverse RCC（r=1）：NaN 只出現在 A 梯度的第 512、513 欄（514 維輸入中
  text_nav_feats 的 2 維摘要），同一步的 loss（約 5e-5）與上游梯度（dL/da、dL/dh、dL/ds）
  皆為有限值。
- 是否出現與 CPU 執行緒數有關：8 條在第 263 步、1 條在第 215 步（另一次重播為第 65 步）、
  3 條在前 320 步未出現（原 run 用 3 條，在第 2 個 epoch 出現）。
- 把出錯那一步的 u 與 dL/da 存下來單獨重算（mv、mm、廣播、補零到 520 欄，1／3／8 條執行緒，
  以及單獨再跑一次 autograd），**全部有限**，與 float64 參考值差 < 3e-12。

## 判斷

數值運算問題，不是訓練發散。r = 1 時 `u @ A.t()` 的右運算元是 [514, 1]，PyTorch 走
matrix-vector（gemv）路徑，反向的 `uᵀ·g` 也是 gemv（輸出長度 514）。本機 PyTorch 2.11.0 的
BLAS 為 Apple Accelerate。同樣輸入單獨重算不會重現，表示 NaN 不是由輸入決定，而是執行當下的
記憶體狀態；最符合的解釋是 gemv 在 beta = 0 時仍讀取了未初始化的輸出緩衝（0 × NaN = NaN），
且只發生在 512 之後的尾端元素。此為推論，未對 Accelerate 做進一步反組譯驗證。

## 修法

1. **LoRA 分支的計算方式（僅 r = 1）**：改為廣播乘法加總，不經任何 BLAS 呼叫：

   ```
   r = 1： lora(u) = (u ⊙ A[0]).sum(-1, keepdim=True) ⊙ B[:, 0]      # [n,1] ⊙ [256] → [n,256]
   r ≥ 2： lora(u) = (u @ Aᵀ) @ Bᵀ                                  # 不變
   ```

   數學上 (u ⊙ A[0]).sum(-1) = u·A[0] = (u @ Aᵀ)[:, 0]，再乘 B[:, 0] 即外積 (u @ Aᵀ) @ Bᵀ，
   兩者等價。前向與反向（dA = Σ_n g_a ⊙ u、dB = Σ_n g_h ⊙ a）都只用逐元素乘法與加總，
   不配置、不讀取 gemv 的輸出緩衝，因此避開出問題的路徑。合併 expert 用的增量 B·A 在 r = 1
   時同樣改成廣播外積。維持 float32。r ≥ 2 走 gemm，NC-2 中 180 個增量皆無 NaN，不改動。
   選這個方法而不是「補零到 8 的倍數欄」：補零後 r = 1 仍然走 gemv，只是尾端長度改變，
   不能保證避開；廣播寫法完全不經 gemv。
2. **每一步檢查**：所有訓練（含第一關 expert 與 L 線）每一步在 backward 後檢查所有可訓練
   參數的梯度、在 optimizer.step 後檢查參數，任一非有限（NaN／Inf）立刻中止並回報
   （程式以非零結束）。
3. **固定執行緒數**：`torch.set_num_threads(8)`（本機 PyTorch 預設值，也是 NC-1 使用的數），
   寫入 `configs/machine_mac.yaml` 與 `outputs/navcil/mac/MACHINE.md`；之後所有 run 都用 8。

## 驗證（結果記在 outputs/navcil/mac/REPORT_stage5.md）

- (i) fold 1 reverse RCC、r = 1，在 1、3、8 條執行緒下各完整訓練 5 epochs，都不得出現 NaN。
- (ii) 等價性：r = 2、fold 1 reverse（RCC），新舊寫法各訓練 50 步，參數最大絕對差 ≤ 1e-5，
  超過就停下回報。另外加做 r = 1 新舊寫法前 50 步（舊寫法尚未出現 NaN 的區段）的同一比較。

## 範圍

- 只改：LoRA 分支在 r = 1 的計算方式、訓練的每步有限性檢查、固定執行緒數。
- 不改：PREREG-2 的設計、超參數、判準與 L-pass 規則；第一關 bank；R 線。
- L 線 v2：以修正後的同一套程式，依序重跑 r ∈ {1, 2, 4, 8} × 十折 × 兩序（不同時跑多個 r），
  依 PREREG-2 的 L-pass 以 v2 判定 r\*。
