# FINAL 逐步實例（fold 1，tcga_lung，test 第 0 張）— 未完成，待決定

Written for: 想看 FINAL 每一步在一張 slide 上實際長什麼樣的研究者。

**狀態：部分完成。** 本檔只寫入既有產物可以轉錄的步驟。其餘步驟的數字在既有 json／報告裡不存在，依 MOE-5 規則（數字只能轉錄，不得重算產生新數字）未產生。見檔末「需要決定的事」。

格式照 `AUDIT_wp.md` A9。slide：`TCGA-51-6867-01Z-00-DX1.5f3a0562-efbe-413f-8e13-9826aaefa298`（與 A9 相同，取法：fold 1 LUNG test split 第 0 張），標籤 LUSC，patch 數 N = **1263**。

## 來源與可轉錄的範圍

| 步驟 | 狀態 | 來源 |
|---|---|---|
| 讀特徵 Z (1263, 512) | 可轉錄（A9） | `AUDIT_wp.md` A9 |
| s0、g、s = s0 + g 的範圍 | 可轉錄（A9）。FINAL 與主系統用同一顆 seed 42 head，s0 與 g 與選片無關，數值相同 | `AUDIT_wp.md` A9 |
| 一次取 64 的閾值（s 純排序第 64 名） | 可轉錄（A9）：3.9449。FINAL 的 v(42) 用的是四輪，不是一次取 64；閾值只用來說明 w 的選法，見下方「未完成」 | `AUDIT_wp.md` A9 |
| TP：AR 的 4 個分數與判定 | 可轉錄（A9）。FINAL 的 TP 就是 AR（γ = 1e-3），與主系統完全相同 | `AUDIT_wp.md` A9 |
| 四輪選出的 64 個 | 可轉錄（A9 的 min／median／max 排序名次）。逐一名單未存 | `AUDIT_wp.md` A9 |
| v 的範數 | 可轉錄（A9：‖v‖ = 1.000000，v 是 L2 正規化後） | `AUDIT_wp.md` A9 |
| v 與 8 類文字的 cosine | 可轉錄（A9）。FINAL 的 v 與主系統相同，所以這組 cosine 就是 FINAL 的輸入 | `AUDIT_wp.md` A9 |
| 主系統「和文字比」的兩個 cosine 與判定 | 可轉錄（A9）：LUAD 0.577495、LUSC 0.667105，差 −0.089611；判定 LUSC，正確 | `AUDIT_wp.md` A9 |
| ridge 的 A 累加的 train slide 張數 | 可轉錄：fold 1 train 四任務為 esca 120、rcc 616、brca 763、lung 774，合計 2273（與 `moe4/logs/vec.log` 的 n = 2273 相符） | `i6/r2/fold1_tcga_*_train.json` 的 `epochs[0].n`；`moe4/logs/vec.log` 行 3 |
| ridge 的 A、B、W 的 shape | 由設計直接給出：A (513, 513)（[v; 1] 的 XᵀX）、B (513, 8)（每類一欄，t = 4 時 8 類皆已學）、W (513, 8) | `PREREG-20.md` 操作定義 |

## 未完成（既有產物中沒有，需要新推論）

| 步驟 | 為什麼沒有 |
|---|---|
| A 與 B 的實際數值，以及 W 的數值 | 只存了 w 快取（`moe4/cache/w_*_fold1.pt`，每張 slide 的 4 個 w 向量），沒有存 ridge 的累加統計量 |
| 8 類 ridge 分數 | 需要 W，不存在 |
| 在 τ̂ 兩類內的分數差與判定 | 同上 |
| 與「和文字比」的判定差異（逐張） | 只有 test 十折的總數，沒有逐張判定 |
| FINAL 判錯、主系統判對的第一張小葉癌（ILC）test slide | 沒有逐張 FINAL 判定檔；`moe0/b3_per_slide.csv` 只存主系統與 LIN8 的 d 值，不含 FINAL |

## 需要決定的事

上面的未完成項目只能靠新寫一支「只推論、不訓練」的示例腳本（同 `scripts/moe0_a9.py` 的作法）算出。這會產生新的數字，與 MOE-5 規則「數字只能從既有 json／報告轉錄，不得重算產生新數字」相衝突。兩種做法：

1. 允許寫示例腳本，只為本檔產生上表所列的數字；新數字只進本檔，不進 FINAL_RESULTS；腳本與輸出另存。
2. 維持本檔現狀，不產生新數字。

在您決定之前，本檔保持部分完成。
