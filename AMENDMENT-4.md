# AMENDMENT-4 — PREREG-15 操作定義 6（validation 評估一致性檢查）的比對方式

登記時間：2026-09-30，任何 LoRA 對照版 validation 評估執行前 commit；經使用者同意。
範圍：只改 PREREG-15 操作定義 6 的比對方式；候選、門檻、選 r 規則、不看 test 的規定都不變。

## 發現

預檢（fold 1、只算 L0 expert、只讀 validation，未寫任何輸出）：以 `nc2_lora.eval_order_fold` 的算法
（逐張 `mean_norm(Z, j) @ F.t()`，gemv）重算 L0 的 validation 8 類 cosine，與 NC-2 validation 快取
`four_cos8_uni[:, τ]` 相比：

| 任務 | 張數 | 四輪 64 個 patch index | 8 類 cosine 最大絕對差 | 2 類內 argmax |
|---|---|---|---|---|
| tcga_esca | 15 | 逐位相同 | 3.6e-07 | 全同 |
| tcga_rcc | 76 | 逐位相同 | 6.0e-07 | 全同 |
| tcga_brca | 96 | 逐位相同 | 3.6e-07 | 全同 |
| tcga_lung | 96 | 逐位相同 | 4.2e-07 | 全同 |

原因：NC-2 快取把四個 expert 的 zbar 疊成 [4, 512] 後乘 `F.t()`（gemm，`scripts/nc2_r_cache.py:48-62`），
test 評估與本輪 validation 評估逐張、逐 expert 以向量乘 `F.t()`（gemv，`scripts/nc2_lora.py:113`）。
改用 4 條執行緒（快取當初的設定）差值相同，與執行緒數無關。

## 修改

PREREG-15 操作定義 6 改為：每折、每個任務的 validation slides，L0 expert 的
(a) 四輪選中 64 個 patch index 與快取 `four_idx[:, τ]` 逐位相同（`torch.equal`）；
(b) 8 類 cosine 與快取 `four_cos8_uni[:, τ]` 的最大絕對差 ≤ 1e-6；
(c) τ 的 2 類內 argmax 與快取全同；
slide 順序、slide id、labels 與快取一致。任一不符即停。
LoRA 對照版的 validation cosine 仍用與 test 評估相同的 gemv 算法。
L0 validation WP（操作定義 3）仍由快取 `four_cos8_uni` 計算，與 `scripts/nc7_report.py:51-55` 相同。
