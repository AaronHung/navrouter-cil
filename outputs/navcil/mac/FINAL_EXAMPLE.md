# FINAL 逐步實例（FINAL ＝ MOE-4 的 P-4，fold 1，seed 42）

Written for: 想看 FINAL 每一步在實際 slide 上長什麼樣子的研究者。

## 數字來源（每個數字旁都有標）

- **轉錄**：來自 `AUDIT_wp.md` A9（主系統在同一張 slide 上的逐步值；FINAL 與主系統共用 seed 42 的 head，s0、g、s、TP 與 v 的算法相同，只有選片後的 ridge 不同）。
- **示例腳本新算**：`scripts/moe5_example.py`（只推論、不訓練、不寫既有產物），輸出 `moe5_example/example.json`。它先重現 A9 的 TP 與 cosine 作為檢查。
- **既有推論腳本新算**：`scripts/moe0_a9.py --task tcga_brca --index 26 --fold 1`（只推論，stdout）。

檢查：示例腳本對 A9 的 TP 分數最大差 4.6e-5，cosine 最大差 5.0e-7（A9 以四位小數顯示，皆在捨入範圍內）。

---

## 例一：fold 1，tcga_lung，test 第 0 張（與 A9 同一張）

slide：`TCGA-51-6867-01Z-00-DX1.5f3a0562-efbe-413f-8e13-9826aaefa298`，標籤 LUSC，patch 數 N = 1263。

| 步驟 | shape | 數值 | 來源 |
|---|---|---|---|
| 讀特徵 Z | (1263, 512) | ‖z‖ 24.55／25.24／25.80 | A9 |
| s0（slide 內 z-score） | (1263,) | min −3.7532／median 0.1281／max 2.2774 | A9 |
| g | (1263,) | min −3.3021／median 0.7384／max 4.6674 | A9 |
| s = s0 + g | (1263,) | min −5.6860／median 0.7849／max 6.1359 | A9 |
| 一次取 64 的閾值（s 純排序第 64 名） | — | 3.9449（s0 的第 64 名為 1.4731） | A9 |
| 四輪選出的 64 個（λ = 1.5） | (64,) | 選出者在 s 純排序名次 min 1／median 32／max 106；其中 5 個名次 > 64 | A9 |
| v = 64 個原始 Z 等權平均後 L2 正規化 | (512,) | ‖v‖ = 1.000000（示例腳本 1.0000001） | A9；示例腳本 |
| 8 類 cosine（v · Fᵀ） | (8,) | ESAD 0.5026、ESCC 0.3916、CCRCC 0.0849、PRCC 0.1634、IDC 0.3588、ILC 0.1101、LUAD 0.5775、LUSC 0.6671 | A9；示例腳本（最大差 5.0e-7） |
| TP：AR 分數 [esca, rcc, brca, lung] | (4,) | 0.0280、0.0230、−0.1231、**1.0721** → τ̂ = tcga_lung | A9；示例腳本（最大差 4.6e-5） |

**ridge（FINAL 的判讀器）**

| 量 | shape／數值 | 來源 |
|---|---|---|
| 累加的 train slide 張數 | 2273（esca 120、rcc 616、brca 763、lung 774） | `i6/r2/fold1_tcga_*_train.json` 的 `epochs[0].n`；`moe4/logs/vec.log` |
| A = Σ [v; 1]ᵀ[v; 1]（513 × 513，float64） | (513, 513) | 示例腳本（A 累加的是 2273 張 train slide 的 x） |
| B（8 欄，類別 c 欄 = 該類 train slide 的 x 之和） | (513, 8) | 示例腳本 |
| W = solve(A + 1e-3 · I, B) | (513, 8) | 示例腳本 |
| 8 類 ridge 分數 s = [v; 1] · W（v 取 τ̂ = lung head） | (8,) | 示例腳本 |

8 類 ridge 分數（示例腳本）：ESAD 0.018975、ESCC 0.003067、CCRCC 0.048595、PRCC −0.012161、IDC −0.198399、ILC 0.104107、**LUAD 0.273392**、**LUSC 0.762425**。

**判定**

| 量 | 數值 | 判定 |
|---|---|---|
| τ̂ 兩類（LUAD、LUSC）的分數差 d = s(LUAD) − s(LUSC) | −0.489033 | d < 0 → 第二類 |
| CIL 最終判定（FINAL） | LUSC | 正確（標籤 LUSC） |
| 告訴任務判定（FINAL 的 WP；同一組 W，向量取 lung head） | d = −0.489033 → LUSC | 正確 |
| 主系統（A9）「和文字比」的兩個 cosine：LUAD 0.577495、LUSC 0.667105，差 LUAD − LUSC = −0.089611 | — | 判定 LUSC，正確 |
| FINAL 與主系統在這一張上的判定 | 皆為 LUSC | 相同 |

**註**：這張 slide 的 FINAL 分數差與主系統的 cosine 差符號相同，但 FINAL 的差值（−0.4890）是 ridge 分數差，與 cosine 差（−0.0896）尺度不同，不可直接比較大小。

---

## 例二：同一折的一張小葉癌（ILC）test slide

**選取規則**：fold 1、tcga_brca 的 test slides 中，依 test 順序取第一張「FINAL 告訴任務判錯、主系統告訴任務判對」的 ILC。第 26 張（index 26）符合。它不是挑選出來的最佳例子，只是規則下的第一張。

slide：`TCGA-LL-A440-01Z-00-DX1.6E031FD6-236C-49FA-B920-4CB120C59037`，標籤 ILC，patch 數 N = 1193，TP 分派 tcga_brca。

| 步驟 | shape | 數值 | 來源 |
|---|---|---|---|
| 讀特徵 Z | (1193, 512) | — | `moe0_a9.py` |
| text_nav_feats | (1193, 2) | max cos min 0.003940／median 0.367401／max 0.734716；entropy min 0.686248／median 0.692081／max 0.693147 | `moe0_a9.py` |
| s0（slide 內 z-score） | (1193,) | min −2.593881／median −0.148165／max 2.323482 | `moe0_a9.py` |
| g | (1193,) | min −5.904598／median −3.284369／max 0.019630；std(g)/std(s0) = 1.1934 | `moe0_a9.py` |
| s = s0 + g | (1193,) | min −6.350046／median −3.792408／max 1.609252 | `moe0_a9.py` |
| 一次取 64 的閾值（s 純排序第 64 名） | — | 0.313024（s0 的第 64 名為 1.777996） | `moe0_a9.py` |
| 四輪選出的 64 個（λ = 1.5） | (64,) | 選出者在 s 純排序名次 min 1／median 32／max 75；其中 5 個名次 > 64（因扣冗餘而進入） | `moe0_a9.py` |
| 進入前 64 的例子 | — | patch 657：s0 名次 87 → s 名次 1，第 1 輪被選；patch 1000：66 → 6；patch 656：115 → 7 | `moe0_a9.py` |
| v（64 個原始 Z 等權平均後 L2 正規化） | (512,) | ‖v‖ = 1.000000 | `moe0_a9.py` |
| 8 類 cosine | (8,) | ESAD 0.131598、ESCC 0.027189、CCRCC 0.050038、PRCC −0.026479、IDC 0.668018、ILC 0.677568、LUAD 0.147952、LUSC 0.122831 | `moe0_a9.py`（與示例腳本的 IDC／ILC 相同） |
| 主系統「和文字比」兩個 cosine | — | IDC 0.668018、ILC 0.677568，差 IDC − ILC = −0.009550 | `moe0_a9.py` |
| 主系統 WP（告訴任務） | — | ILC，正確 | `moe0_a9.py` |
| TP：AR 分數 [esca, rcc, brca, lung] | (4,) | 0.077647、0.047507、0.925872、−0.051025 → tcga_brca | `moe0_a9.py`；示例腳本（相同） |
| 主系統 CIL 最終判定 | — | ILC，正確 | `moe0_a9.py` |

**FINAL 的 ridge**（同一組 A、B、W，v 取 brca head 的四輪）

| 量 | 數值 | 來源 |
|---|---|---|
| A（513 × 513） | 同例一（2273 張 train） | 示例腳本 |
| W（513 × 8） | 同例一 | 示例腳本 |
| 8 類 ridge 分數 | ESAD 0.001477、ESCC 0.052815、CCRCC 0.043695、PRCC −0.026203、**IDC 0.540624**、**ILC 0.504677**、LUAD −0.022245、LUSC −0.094844 | 示例腳本 |
| 在 τ̂ = brca 兩類內的差 d = s(IDC) − s(ILC) | +0.035947 → 第一類 | 示例腳本 |
| FINAL CIL 最終判定 | **IDC**（錯；標籤 ILC） | 示例腳本 |
| FINAL 告訴任務判定 | 同上，IDC（錯；CIL 的 τ̂ 與告訴任務相同，都是 brca head） | 示例腳本 |
| 主系統 | ILC（對） | `moe0_a9.py` |

**這一張的結論**：主系統的兩個 cosine 中 ILC 較大（0.677568 > 0.668018），判 ILC；FINAL 的 ridge 在 IDC 與 ILC 之間差 +0.035947，判 IDC。差距很小，兩個判定只差在最後一步的判讀器。

---

## 驗證與限制

- 示例腳本的 TP 與 cosine 與 A9 相符；`moe0_a9.py` 的 TP、cosine 與示例腳本相符（小數六位）。兩者互相獨立，但都讀同一批快取。
- A 與 W 的數值沒有存在既有產物中，只能由示例腳本即時算出；未寫入任何既有產物，也未進 FINAL_RESULTS。
- 示例 slide 的 τ̂ 與告訴任務相同（brca），所以 CIL 與 WP 的 FINAL 判定相同；例一的 τ̂ 與告訴任務也相同（lung）。
- 這兩個例子是在整個 fold 1 的規則下取出的，不是挑選來展示好例子。

來源檔：`scripts/moe5_example.py`；輸出 `moe5_example/example.json`、`moe5_example/stdout.txt`。
