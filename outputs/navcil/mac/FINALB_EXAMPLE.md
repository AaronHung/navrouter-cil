# FINAL-B 逐步實例（FINAL-B ＝ 不用 head：s = s0，四輪各 16，[v; 1] 累加式 ridge γ = 0.001；fold 1；沒有 seed）

## 數字來源

- 全部由 `scripts/ext2_example.py` 從特徵檔重算（只推論、不訓練、不寫既有產物），輸出 `ext2/example.json`。FINAL-B 的每一步都沒有載入 head。
- FINAL-A(42) 的 head（`i6/r2/fold1_<task>.pt`）只用在兩處：算「兩套各自選出的 64 個的交集」，以及並列 FINAL-A 的判定。
- 兩張 slide 與 `FINAL_EXAMPLE.md` 相同。A、B、W 是 fold 1、reverse 序、t = 4（四個任務都學完）的累加結果。

檢查：

| slide | 重算的 v 對快取的 v0（最大絕對差） | TP 分數對 `moe5_example/example.json`（最大絕對差） | 判定與 A 階段（由快取算）相同 |
|---|---|---|---|
| tcga_lung test 第 0 張 | 0.0e+00 | 0.0e+00 | 是 |
| tcga_brca test 第 26 張 | 0.0e+00 | 0.0e+00 | 是 |

---

## 例一：fold 1，tcga_lung，test 第 0 張

slide：`TCGA-51-6867-01Z-00-DX1.5f3a0562-efbe-413f-8e13-9826aaefa298`，標籤 LUSC，patch 數 N = 1263。

| 步驟 | shape | 數值 |
|---|---|---|
| 讀特徵 Z | (1263, 512) | ‖z‖ min 24.55／median 25.24／max 25.80 |
| TP：mean_vec ＝ 全部 patch 平均後 L2 正規化，補 1；AR 分數 [esca, rcc, brca, lung] | (4,) | 0.0280、0.0230、-0.1231、1.0721 → τ̂ = **tcga_lung** |
| 每個 patch 對 τ̂ 兩類文字（LUAD、LUSC）的最大 cosine | (1263,) | min 0.024017／median 0.489071／max 0.746604；平均 0.473722、標準差 0.119818 |
| s0 ＝ 上一列在這張 slide 內 z-score；s = s0（沒有 g） | (1263,) | min -3.7532／median 0.1281／max 2.2774 |
| s0 純排序第 64 名的分數（一次取 64 的門檻） | — | 1.4731 |
| 四輪選出的 64 個（λ = 1.5；每輪 16） | (64,) | 選出者在 s0 純排序的名次 min 1／median 32／max 115；其中 7 個名次 > 64（因扣冗餘而進入） |
| 各輪前三個 patch index | 4 × 16 | 輪 1：[873, 317, 948]…；輪 2：[215, 787, 1204]…；輪 3：[771, 674, 1041]…；輪 4：[951, 881, 927]… |
| 各輪所選 patch 的 s0 最小值 | (4,) | 1.7266、1.3215、1.2450、1.4324 |
| 與 FINAL-A(42) 同一張所選 64 個的交集 | — | **19／64**；FINAL-B 各輪的 16 個中落在 FINAL-A 的 64 個內：8、5、3、3；同一輪對同一輪的交集：2、1、0、1 |
| v ＝ 64 個原始 Z 等權平均後 L2 正規化 | (512,) | ‖v‖ = 1.000000；與 FINAL-A 的 v 的 cosine = 0.968899 |
| 8 類 cosine（v · Fᵀ；FINAL-B 不用它判定，只供對照） | (8,) | ESAD 0.481399、ESCC 0.499857、CCRCC 0.101049、PRCC 0.158345、IDC 0.326859、ILC 0.090841、LUAD 0.546558、LUSC 0.742696 |

**ridge（FINAL-B 的判讀器）**

| 量 | shape／數值 |
|---|---|
| 累加的 train slide 張數 | 2273（esca 120、rcc 616、brca 763、lung 774） |
| A = Σ [v; 1]ᵀ[v; 1]（四個任務累加在同一個 A；float64） | (513, 513)；A 的右下角 = 2273（＝張數） |
| B（8 欄，類別 c 欄 = 該類 train slide 的 x 之和） | (513, 8) |
| W = solve(A + 0.001 · I, B) | (513, 8) |
| 8 類 ridge 分數 [v; 1] · W（v 取 τ̂ 的文字） | ESAD -0.003004、ESCC -0.011697、CCRCC -0.026064、PRCC 0.023941、IDC -0.185092、ILC 0.112011、LUAD 0.256269、LUSC 0.833629 |

**判定**

| 量 | 數值 | 判定 |
|---|---|---|
| τ̂ 兩類的分數差 d = s(LUAD) − s(LUSC) | -0.577360 | d < 0 → LUSC |
| FINAL-B 的 CIL 最終判定 | LUSC | 正確（標籤 LUSC） |
| FINAL-B 告訴任務的判定（向量取真實任務的文字；τ̂ 與真實任務相同，與 CIL 同一個向量） | d = -0.577360 → LUSC | 正確 |
| FINAL-A(42) 同一張：head 選的 64 個、同一種 ridge（γ = 1e-3） | d = -0.489033 → LUSC | 正確 |
| 兩套在這一張的判定 | FINAL-B LUSC；FINAL-A LUSC | 相同 |

並列（FINAL-A(42) 在同一張、同一個 τ̂ 的分數）：g min -3.3021／median 0.7384／max 4.6674；s = s0 + g min -5.6860／median 0.7849／max 6.1359。FINAL-B 的 s0 與 head 內部算出的 s0 最大絕對差 0.0e+00。

---

## 例二：fold 1，tcga_brca，test 第 26 張（FINAL_EXAMPLE 的同一張小葉癌）

slide：`TCGA-LL-A440-01Z-00-DX1.6E031FD6-236C-49FA-B920-4CB120C59037`，標籤 ILC，patch 數 N = 1193。

| 步驟 | shape | 數值 |
|---|---|---|
| 讀特徵 Z | (1193, 512) | ‖z‖ min 24.71／median 25.30／max 25.96 |
| TP：mean_vec ＝ 全部 patch 平均後 L2 正規化，補 1；AR 分數 [esca, rcc, brca, lung] | (4,) | 0.0776、0.0475、0.9259、-0.0510 → τ̂ = **tcga_brca** |
| 每個 patch 對 τ̂ 兩類文字（IDC、ILC）的最大 cosine | (1193,) | min 0.003940／median 0.367401／max 0.734716；平均 0.389420、標準差 0.148610 |
| s0 ＝ 上一列在這張 slide 內 z-score；s = s0（沒有 g） | (1193,) | min -2.5939／median -0.1482／max 2.3235 |
| s0 純排序第 64 名的分數（一次取 64 的門檻） | — | 1.7780 |
| 四輪選出的 64 個（λ = 1.5；每輪 16） | (64,) | 選出者在 s0 純排序的名次 min 1／median 32／max 71；其中 3 個名次 > 64（因扣冗餘而進入） |
| 各輪前三個 patch index | 4 × 16 | 輪 1：[442, 544, 524]…；輪 2：[578, 187, 818]…；輪 3：[566, 858, 967]…；輪 4：[969, 506, 636]… |
| 各輪所選 patch 的 s0 最小值 | (4,) | 2.1214、1.9358、1.7494、1.7194 |
| 與 FINAL-A(42) 同一張所選 64 個的交集 | — | **24／64**；FINAL-B 各輪的 16 個中落在 FINAL-A 的 64 個內：3、8、8、5；同一輪對同一輪的交集：0、2、3、0 |
| v ＝ 64 個原始 Z 等權平均後 L2 正規化 | (512,) | ‖v‖ = 1.000000；與 FINAL-A 的 v 的 cosine = 0.971009 |
| 8 類 cosine（v · Fᵀ；FINAL-B 不用它判定，只供對照） | (8,) | ESAD 0.114974、ESCC 0.010813、CCRCC 0.026600、PRCC -0.035784、IDC 0.657135、ILC 0.746219、LUAD 0.126221、LUSC 0.098842 |

**ridge（FINAL-B 的判讀器）**

| 量 | shape／數值 |
|---|---|
| 累加的 train slide 張數 | 2273（esca 120、rcc 616、brca 763、lung 774） |
| A = Σ [v; 1]ᵀ[v; 1]（四個任務累加在同一個 A；float64） | (513, 513)；A 的右下角 = 2273（＝張數） |
| B（8 欄，類別 c 欄 = 該類 train slide 的 x 之和） | (513, 8) |
| W = solve(A + 0.001 · I, B) | (513, 8) |
| 8 類 ridge 分數 [v; 1] · W（v 取 τ̂ 的文字） | ESAD -0.024430、ESCC 0.098959、CCRCC 0.029636、PRCC -0.032928、IDC 0.522571、ILC 0.504041、LUAD -0.029830、LUSC -0.068014 |

**判定**

| 量 | 數值 | 判定 |
|---|---|---|
| τ̂ 兩類的分數差 d = s(IDC) − s(ILC) | +0.018530 | d ≥ 0 → IDC |
| FINAL-B 的 CIL 最終判定 | IDC | **錯**（標籤 ILC） |
| FINAL-B 告訴任務的判定（向量取真實任務的文字；τ̂ 與真實任務相同，與 CIL 同一個向量） | d = +0.018530 → IDC | **錯** |
| FINAL-A(42) 同一張：head 選的 64 個、同一種 ridge（γ = 1e-3） | d = +0.035947 → IDC | **錯** |
| 兩套在這一張的判定 | FINAL-B IDC；FINAL-A IDC | 相同 |

並列（FINAL-A(42) 在同一張、同一個 τ̂ 的分數）：g min -5.9046／median -3.2844／max 0.0196；s = s0 + g min -6.3500／median -3.7924／max 1.6093。FINAL-B 的 s0 與 head 內部算出的 s0 最大絕對差 0.0e+00。

---

## 驗證與限制

- 兩張 slide 沿用 `FINAL_EXAMPLE.md` 的選法（例一與 AUDIT A9 同一張；例二是 fold 1 的 tcga_brca test 中第一張「FINAL-A 告訴任務判錯、主系統判對」的 ILC），不是為 FINAL-B 另外挑的。
- A、W 的數值沒有存在既有產物中，由示例腳本從快取的 train 向量即時算出；train 向量本身已在 K11(a) 與特徵檔重算值比對（最大絕對差見 `REPORT_ext2.md`）。
- ridge 分數差與 cosine 差尺度不同，不可直接比較大小。

來源檔：`scripts/ext2_example.py`；輸出 `ext2/example.json`。
