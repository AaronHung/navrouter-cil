# PREREG-19 — MOE-3 預先註冊（定案批：不用 head 的證據向量、以文字為起點的 ridge 判讀、逐階段結果）

登記時間：2026-10-03，MOE-3 任何程式執行前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；本輪所有數字來自同一台、同一批。只算向量與 closed-form，不訓練。
分支：`moe-3`（自 `moe-2` @ a47713f）。不合併；不修改既有程式、既有報告與既有 PREREG；新程式放新檔
（`scripts/moe3_*.py`、`scripts/moe3_run_all.sh`）。
報告：`outputs/navcil/mac/REPORT_moe3.md`（`scripts/moe3_report.py` 產生）。

本檔＝指令原文的「共同設定」「操作定義」「一致性檢查」「要報的」「門檻」全文（第一部分），加上執行前同時登記的操作化細則（第二部分；不改變第一部分）。

---

## 第一部分：指令原文

### 共同設定

--device cpu、8 條執行緒；closed-form 一律 float64；十折；reverse 與 paper 兩序；四任務等權；同一批、同一台機器。沿用 MOE-2 的程式與快取（v(42)、四輪的 v0）。CIL 的 TP 一律 AR（γ = 1e-3）。

### 操作定義

向量：
- u_K = 不用 head（g = 0），依 s0 一次取前 K 個 patch（不扣冗餘），原始 Z 等權平均後 L2 正規化（512）。K ∈ {16, 32, 64, 128, 256}。train slide 用自己任務的文字算 s0；validation／test 四個任務的文字都算（CIL 取 τ̂ 的）。每張 slide 讀一次，同時算出五個 K。
- v0 = MOE-2 的四輪版本（沿用快取）；v(42) = seed 42 head 的四輪版本（沿用快取）；mean_vec = 全部 patch 的平均（沿用）。

判讀器（輸入一律 [向量; 1]，513 維；8 類；依序每學一個任務累加 A、B，每個階段重解 W；告訴任務時在真實任務兩類內判，CIL 時向量取自 τ̂、在 τ̂ 兩類內判）：
- TXT：與該任務兩類文字比 cosine（不訓練）。
- RDG(γ)：W = (A + γI)⁻¹ B。
- ANC(γ, α)：W = (A + γI)⁻¹ (B + γ·α·T)。T 為 513 × 已學類別數；第 c 欄的前 512 列 = 類別 c 的文字向量（與 TXT 用的 f_txt 相同），第 513 列 = 0。

超參數的選法（RDG、ANC 對每一種輸入向量各自選一次）：用十折 validation，目標 = 「兩序 × t = 1…4 的告訴任務 WP（已學任務等權）的平均」。RDG：γ ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}。ANC：γ ∈ {1e-3, 1e-2, 1e-1, 1, 10}、α ∈ {0.5, 1, 2, 4}。同分先取較大的 γ，再取較小的 α。選在邊界照報，不延伸。K ≠ 64 的向量沿用 K = 64 選定的超參數，不重選。

### 一致性檢查

K1 主系統 seed 42：WP 0.9340、CIL ACC 0.9128，與 nc8/per_fold.json 逐折最大絕對差 ≤ 1e-9。
K2 RDG(γ = 1e-3) 用 v0：t = 4 的 test WP 與 CIL ACC 與 moe2 的 LR0 逐折最大絕對差 ≤ 1e-9（平均 0.9435、0.9214）。
K3 ANC(γ, α = 0) 與 RDG(γ) 的 W 最大絕對差 ≤ 1e-10（任取一個 γ、fold 1、t = 4）。
K4 ANC(γ = 1e8, α = 1) 用 v0：test 告訴任務的判定與 TXT 用 v0 的判定逐張相同（t = 4，十折）。
K5 TXT 用 v0 的每任務 WP 與 MOE-0 B2 的「g = 0」列最大絕對差 ≤ 1e-9。

### 要報的

F1 系統列表（輸入向量／判讀器）：
  S-main = 主系統（seed 42；逐階段取自 moe1/s2.json）
  S-M3 = M3（取自 moe1/s2.json）
  S-LR = v(42)／RDG(γ = 1e-5)（取自 moe2 E7 並比對）
  S-T0 = u_64／TXT
  S-R0 = u_64／RDG
  S-A0 = u_64／ANC
  S-R0f = v0／RDG（四輪）
  S-A0f = v0／ANC（四輪）
  S-Ah = v(42)／ANC
F2 逐階段：每個系統、每個順序、t = 1…4：ACC（只在已學類別中判）、告訴任務的 WP（已學任務等權）、每任務 WP；Forgetting、BWT（與 Table 1 同一個函式）；四個階段 ACC 的平均（記為 Ā）。十折 mean ± sd。
F3 t = 4：每個系統的 WP、CIL ACC；相對 S-main(42) 的逐折差、平均、贏折數、Wilcoxon 雙尾 p；確認 S-R0、S-A0 兩序是否逐張相同。
F4 每類別（t = 4、告訴任務、十折合計）：8 類各自的正確張數與正確率；每任務 balanced accuracy 與四任務平均。系統：S-main、S-M3、S-LR、S-R0、S-A0。
F5 K 的影響：S-T0、S-R0、S-A0 在 K ∈ {16, 32, 64, 128, 256} 與「全部 patch」（mean_vec）下的 t = 4 WP、CIL ACC 與 Ā（兩序）。
F6 超參數：每個判讀器 × 輸入的整張 validation 網格（目標值）、選定值、是否在邊界；另列 S-A0 選定值附近（γ 上下各一格、α 上下各一格）的 test t = 4 CIL ACC 與 Ā，作為敏感度。
F7 儲存：S-R0、S-A0 的共用與每任務 bytes（fp32），格式同 moe2 E8。

### 門檻

G-FINAL：每折 D = CIL ACC(S-A0, t = 4) − [seed 42–46 五個主系統的 CIL ACC 平均]（主系統數字取自 moe2 E1 並比對）；D 的十折平均 ≥ +0.007 且 D > 0 的折數 ≥ 7。
G-COLD：兩序各自：t = 1 的 ACC(S-A0) − ACC(S-main) 的十折平均 ≥ −0.010。
G-ANCHOR：兩序各自：Ā(S-A0) − Ā(S-R0) 的十折平均 ≥ +0.005 且 > 0 的折數 ≥ 7。
三個門檻的同一組數字，也對 S-R0 算一次（把 S-A0 換成 S-R0；G-ANCHOR 那一格留空），只報告、不判定。
F4、F5、F6、F7 不設門檻。

---

## 第二部分：操作化細則（與第一部分同時登記；不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `outputs/navcil/mac/moe3/DECISIONS.md`（做了什麼判斷、為什麼），並印在報告內。

### 0. 共通

1. **資料、類別順序、各階段的 AR**：同 PREREG-18 細則 1（8 類固定序 ESAD, ESCC, CCRCC, PRCC, IDC, ILC, LUAD, LUSC；任務 p 的兩類為第 2p、2p+1 列，
   「第一類」= 第 2p 列；reverse = esca → rcc → brca → lung，paper = lung → brca → rcc → esca）。階段 t 的 AR = `moe1_common.ar_stage`（`nc8_report.B8.W`，γ = 1e-3，
   只含該序前 t 個任務；未學任務的分數為 −inf）。
2. **向量**（`vec` 階段；每張 slide 讀一次特徵檔；逐張記錄 `t_read_s`、`t_compute_s`）：
   - u_K：s0 = `zscore(text_nav_feats(Z, f_task)[:, 0])`（對該任務兩類文字的最大 cosine、slide 內 z-score；與 `I6Expert.parts` 的 s0 同一段算式，不載入任何 head 權重）；
     `top_k_select(s0, K)`（patch 數 ≤ K 的 slide 取全部 patch）；`mean_norm(Z, idx)`（等權、L2 正規化）。train slide 只算自己任務的文字；
     validation、test 四個任務的文字都算。五個 K 由同一次讀檔、同一個 s0 算出。存 `moe3/cache/u_{split}_fold{f}.pt`（每（split、折）一個 done 標記）。
     另記錄每個 K 下 patch 數 < K 的張數（描述）。
   - v0、v(42)、mean_vec：經 `moe2_common.vecs` 唯讀讀取 MOE-2／MOE-1／NC-8／moe0 的既有快取（v0 = `g0`、v(42) = `s42`、mean_vec = `mv`）；不重算。
   - 對齊檢查（不過就停下 `vec`）：u 快取的 slide id 與標籤，train 與 `moe2/cache/vec_train` 逐張相同，validation／test 與 moe0 快取逐張相同；所有 u 向量有限且範數與 1 的差 ≤ 1e-5。
3. **累加式統計量**（RDG、ANC 共用）：x = [向量; 1]，轉 float64；序 o 的階段 t：依序對前 t 個任務 A += XᵀX，B 的第 c 欄 = 類別 c 的 train slides 的 x 之和；
   B 只含已學類別、依固定 8 類序排列（累加方式與 `moe2_common.Reader.W` 相同）。
   - RDG(γ)：W = `torch.linalg.solve(A + γI, B)`（I 為 513 × 513，含常數項那一維）。
   - ANC(γ, α)：W = `torch.linalg.solve(A + γI, B + γ·α·T)`；T 的第 c 欄前 512 列 = `ctx.F` 的第 c 列（轉 float64），第 513 列 = 0。
   - 分數 = xW；任務 q 的分數差 d = 第 2q 欄 − 第 2q+1 欄；d ≥ 0 判第一類。
   - TXT：cos = 向量 · Fᵀ（float32、逐張相乘，同 moe0／MOE-2 的算法；mean_vec 先 L2 正規化）；d = 任務 q 第一類 − 第二類的 cosine；d ≥ 0 判第一類（等同 2 類 argmax、同分取第一類）。
4. **告訴任務與 CIL（每個階段 t）**：只對已學任務的 test slides 算。告訴任務 = 向量取自真實任務（u_K、v0 取真實任務文字的版本；v(42) 取真實任務的 head），在真實任務兩欄內判；
   CIL = τ̂ 為階段 t 的 AR 在已學任務中的 argmax，向量取自 τ̂，在 τ̂ 的兩欄內判（mean_vec 與任務無關）。W 依該序累加到階段 t。
   ACC_t、每任務正確率、Forgetting、BWT 由 `nc5_report.cil_full` 計（Table 1 同一個函式）；告訴任務的 WP_t = 已學任務的每任務 2 類正確率等權平均（= `masked_t`）；
   Ā = 四個階段 ACC_t 的平均（每折先算，再取十折 mean ± sd）。
5. **超參數的選法**（`hp` 階段；只用 validation；先選定、寫入 `hp.json`，之後的階段才算 test）：
   每個候選、每折：對兩序 × t = 1…4 共 8 個「validation 告訴任務 WP_t（已學任務等權）」取平均；再取十折平均為目標值。取目標值最大者；
   目標值完全相等才算同分，同分先取較大的 γ，再取較小的 α。
   - 各自選一次的組合：RDG × {u_64, v0}、ANC × {u_64, v0, v(42)}。另列 RDG × v(42) 的網格（只作描述；S-LR 依指令固定 γ = 1e-5，不用這個選擇）。
   - K ∈ {16, 32, 128, 256} 與 mean_vec（F5 的「全部 patch」）沿用 u_64 選定的超參數，不重選。
   - 邊界：γ 是否在 γ 網格的最小／最大值、α 是否在 α 網格的最小／最大值，分別標示；不延伸。
6. **系統**：S-T0 = u_64／TXT；S-R0 = u_64／RDG(γ\*)；S-A0 = u_64／ANC(γ\*, α\*)；S-R0f = v0／RDG(γ\*)；S-A0f = v0／ANC(γ\*, α\*)；S-Ah = v(42)／ANC(γ\*, α\*)（各自的選定值）；
   S-LR = v(42)／RDG(γ = 1e-5)：表內數值取自 `moe2/e7.json` 的 `LR`，本批以本檔的 RDG 重算並報最大絕對差（ACC、WP、Forgetting、BWT、逐階段值；只報，不是 K）；
   S-main、S-M3：逐階段數值取自 `moe1/s2.json`（`main`、`M3`；S-main 的告訴任務 WP_t = `wp_task_t.main` 在已學任務上的等權平均，即真實任務 head 的 2 類正確率；
   S-M3 的 WP_t = `masked_t`），本批照 `moe1_s2.py` 的算法重算並報最大絕對差（只報，不是 K）。
7. **逐折相減**：`moe1_common.paired`（平均差；贏折數 = 差 > 1e-12 的折數；Wilcoxon = `scipy.stats.wilcoxon` 雙尾，全為 0 時 p = 不適用）。mean ± sd 的 sd 為十折的樣本標準差。
8. **兩序是否逐張相同**（F3；S-R0、S-A0，另列其餘 ridge 系統）：兩序 t = 4 的 test：告訴任務判定不同的張數、CIL 判定不同的張數、告訴任務 d 的最大絕對差。

### 1. 一致性檢查（`chk` 階段；任何一項不過 → 該階段停止、寫 `FAILED_chk.txt`，不重試、不改期望值）

9. **K1**：`moe1_common.k1`（同 PREREG-17 細則 6）。
10. **K2**：RDG(γ = 1e-3)、輸入 v0、兩序各自累加、t = 4：test 告訴任務 WP 與 CIL ACC 的逐折值與 `moe2/e2.json` 的 `LR0[序][折].wp`、`.cil` 最大絕對差 ≤ 1e-9；
    十折全跑時另檢查平均四捨五入到小數第 4 位為 0.9435、0.9214。
11. **K3**：輸入 u_64、reverse、t = 4、γ = 1e-2、所跑折數中的第一折（十折時 = fold 1）：ANC(γ, α = 0) 的 W 與 RDG(γ) 的 W 最大絕對差 ≤ 1e-10。
    兩者經各自的程式路徑（ANC 路徑照式子算 B + γ·0·T 後求解）。
12. **K4**：ANC(γ = 1e8, α = 1)、輸入 v0、t = 4、兩序、所跑的每一折：test 告訴任務的判定與 TXT（輸入 v0）的判定逐張相同（不同張數 = 0）。
13. **K5**：TXT（輸入 v0）、test、告訴任務：每任務十折平均 WP 與 `moe0/results.json` 的 `B2.matrix[4]` 最大絕對差 ≤ 1e-9（十折全跑時才可比）；
    另（任何折數都做）每折每任務 WP 與由 moe0 快取 `g0_cos8` 算出的 2 類 argmax 正確率最大絕對差 ≤ 1e-9。
14. **失敗處理**：任何階段的檢查不過或程式出錯 → 該階段停止、寫 `FAILED_<階段>.txt`、不重試、不改期望值。依賴關係：`vec` 失敗 → `chk`、`hp`、`f2`、`f4`、`f5`、`f6` 不跑；
    `chk` 失敗 → `hp`、`f2`、`f4`、`f5`、`f6` 不跑（判讀器的實作未通過檢查，不產生結果）；`hp` 失敗 → `f2`、`f4`、`f5`、`f6` 不跑；`f7` 不依賴其他階段。
    `f2`、`f4`、`f5`、`f6` 彼此獨立，各自開始時再跑一次 K1。
15. **冒煙測試**：同一支 `moe3_run_all.sh`，`MOE3_OUT=moe3_smoke MOE3_FOLDS=1`，報告寫到 `moe3_smoke/REPORT_smoke.md`；數字不寫進 `REPORT_moe3.md`；
    十折平均的四捨五入檢查與 K5 的 B2 比對在冒煙測試中不適用；門檻在非十折時不作判定。

### 2. 門檻

16. **G-FINAL**（`f2`）：兩序各算；每折 D = 該序 t = 4 的 CIL ACC(S-A0) − seed 42–46 五個主系統在該序的 CIL ACC 平均。主系統逐折數值取自 `moe2/e1.json` 的
    `seeds[s].orders[序].main[折].cil`；本批以 `moe2_common.seed_base` 重算並報最大絕對差（只報）。每序：D 的十折平均 ≥ +0.007 且 D > 1e-12 的折數 ≥ 7；兩序都滿足才通過。
17. **G-COLD**（`f2`）：每序：每折 ACC_1(S-A0) − ACC_1(S-main)（S-main 取自 `moe1/s2.json` 的 `acc_t[0]`）；十折平均 ≥ −0.010；兩序都滿足才通過。
18. **G-ANCHOR**（`f2`）：每序：每折 Ā(S-A0) − Ā(S-R0)；十折平均 ≥ +0.005 且 > 1e-12 的折數 ≥ 7；兩序都滿足才通過。
19. **S-R0 的同一組數字**：G-FINAL、G-COLD 把 S-A0 換成 S-R0 算一次，列在同表，只報告、不判定（不寫「通過／未通過」）；G-ANCHOR 那一格留空。

### 3. 各節

20. **F1**：系統列表，含各系統的輸入向量、判讀器、選定的超參數與數值來源。
21. **F2**（`f2`）：九個系統 × 兩序：ACC_t、WP_t（t = 1…4）、Forgetting、BWT、Ā（mean ± sd）；每任務 WP（告訴任務；未學為 —）。
22. **F3**（`f2`）：t = 4：每個系統的 WP、CIL ACC（mean ± sd）；相對 S-main（seed 42；WP = 真實任務 head 的 2 類正確率四任務等權、CIL ACC = `acc_t[3]`）的逐折相減（兩序）；
    兩序是否逐張相同（細則 8）。
23. **F4**（`f4`）：reverse 序（S-M3 的 σ_b 固定值與序有關；與 MOE-2 E6 同序）、test、告訴任務、t = 4、十折合計：8 類各自的張數、正確張數、正確率；
    每任務 balanced accuracy = 十折合計的兩類正確率平均；四任務平均。S-main、S-M3 的判定 = `moe2_common.seed_base(42)` 的告訴任務判定（同 MOE-2 E6）。
24. **F5**（`f5`）：S-T0、S-R0、S-A0 的判讀器分別套用在 u_16、u_32、u_64、u_128、u_256、mean_vec：t = 4 的 WP、CIL ACC 與 Ā（兩序；mean ± sd）。RDG、ANC 一律用 u_64 選定的超參數。
25. **F6**（`hp`、`f6`）：細則 5 各組合的整張 validation 網格（目標值）、選定值、是否在邊界。敏感度：S-A0 的 (γ\*, α\*) 與 γ 上下各一格（α = α\*）、α 上下各一格（γ = γ\*）共至多五格的
    test t = 4 CIL ACC 與 Ā（兩序；mean ± sd）；超出網格的那一格記為 —（不延伸）。
26. **F7**（`f7`）：由實際張量形狀計算（d = 512、x = 513）。S-R0、S-A0 各自「所有任務共用」與「每任務」存了什麼、fp32 的 bytes、總量 = S + T·p 與 T = 4 的數值；
    只算推論與繼續學習必須保存的量（ridge 存累加統計量 A、B；每任務兩類的文字特徵；TP 用的 AR 統計量）。同表列主系統與 S-LR（算法同 MOE-2 E8）供對照。
    CONCH 骨幹、patch 特徵、超參數純量列出但不計入。

### 4. 執行

27. 階段：`vec` → `chk` → `hp` → `f2` → `f4` → `f5` → `f6` → `f7`。每階段完成寫 `<階段>.json` 與 `<階段>.done`，重跑自動跳過。
28. 啟動：`tmux new-session -d -s moe3 "env NAVCIL_MACHINE=mac MOE3_PY=<venv python> caffeinate -dimsu scripts/moe3_run_all.sh"`；
    `HEARTBEAT.log` 每 900 秒一行，每階段開始與結束各一行；每階段結束後重產報告（`scripts/moe3_report.py`）；失敗不重試；跑完不關機、不做 git 操作。
29. 報告：開頭三個門檻與 K1–K5；F1–F7 各一節，表格標明本檔落點；最後列失敗的階段（`FAILED_*.txt` 摘要）、`DECISIONS.md` 全文、各階段耗時。只給數字與表格。
