# PREREG-18 — MOE-2 預先註冊（證據向量的 closed-form 判讀 LR：多 seed 確認、是否需要 head、與整片向量合併、類別加權、CL 流程）

登記時間：2026-10-03，MOE-2 任何程式執行前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；本輪所有數字來自同一台、同一批。
分支：`moe-2`（自 `moe-1` @ 0979a46）。不合併；不修改既有程式、既有報告與既有 PREREG；新程式放新檔
（`scripts/moe2_*.py`、`scripts/moe2_run_all.sh`）。
報告：`outputs/navcil/mac/REPORT_moe2.md`（`scripts/moe2_report.py` 產生）。

本檔＝指令原文的「共同設定」「一致性檢查」「操作定義」「門檻」全文（第一部分），加上執行前同時登記的操作化細則（第二部分；不改變第一部分）。

---

## 第一部分：指令原文

### 共同設定

--device cpu、8 條執行緒；closed-form 一律 float64；十折；reverse 與 paper 兩序；四任務等權；同一批、同一台機器。沿用 MOE-1 的程式、快取與 head 權重：seed 42 在 outputs/navcil/mac/i6/r2/，seed 43–46 在 outputs/navcil/mac/moe1/i6_seed{s}/。不訓練任何 head；若缺少某個 seed 的權重，停下回報，不要重訓。
「主系統」「M3」「LT、GR、LR」的定義與 PREREG-17 相同。CIL 的 TP 一律 AR（γ = 1e-3）。

### 一致性檢查

K1 主系統 seed 42：WP 0.9340、CIL ACC 0.9128，與 nc8/per_fold.json 逐折最大絕對差 ≤ 1e-9。
K2 LR（seed 42、γ = 1e-3、四輪等權向量）：test WP 與 CIL ACC 與 moe1/s5.json 逐折最大絕對差 ≤ 1e-9（平均 0.9457、0.9252）。
K3 seed 43–46 的主系統 CIL ACC 與 moe1 的 s4.json、s7.json 逐折最大絕對差 ≤ 1e-9。

### 操作定義

證據向量 v(s)：用 seed s 的 head，四輪各 16（λ* = 1.5）選 64 個 patch，原始 Z 等權平均後 L2 正規化（512）。train、validation、test 都算；train slide 用它自己任務的 head。
LR(s)：輸入 [v(s); 1]（513），8 類 one-hot 目標，與 LIN8 相同的累加式 ridge（依序每個任務 A += VᵀV、每類一欄）。告訴任務時在真實任務兩類內判；CIL 時 v 取自 τ̂ 的 head，在 τ̂ 的兩類內判。
γ 的選法（LR 及以下各變體各自選一次）：候選 {1e-5, 1e-4, 1e-3, 1e-2}；用 seed 42 的十折 validation 平均 WP 選，同分取較大的 γ；選定後所有 seed 共用。若選在邊界，照報並在報告註明，不再延伸。

E1 多 seed 確認：seed 42–46 各自的 LR：WP、CIL ACC（十折 mean ± sd）、每任務 WP；LR − 同 seed 主系統、LR − 同 seed M3（σ 固定版，兩序）的逐折差、平均、贏折數、Wilcoxon 雙尾 p。確認 LR 在 t = 4 的兩序結果是否逐張相同，並回報。

E2 是否需要 head：v0 = 不用 head（g = 0，只用 s0）以同樣四輪選片得到的向量；LR0 = 以 v0 做同樣的 ridge。報 LR0 的 WP、CIL ACC、每任務 WP；每折 D_head = [五個 seed 的 WP(LR(s)) 平均] − WP(LR0)；逐折值、平均、D_head > 0 的折數。同表列出 LT 在 g = 0 時的 WP（MOE-0 的 B2 已有，重算並比對）。

E3 一次取 64：v1(42) = seed 42 的 head，一次取 s 前 64（不扣冗餘）、等權平均；LR1 = 以 v1 做 ridge。報 WP、CIL ACC 與相對 LR(42) 的逐折差。只做 seed 42。

E4 與整片向量合併：LRG(s)：輸入 [v(s); mean_vec; 1]（1025），同樣的累加式 ridge 與 γ 選法。報 seed 42–46 的 WP、CIL ACC；LRG − 同 seed LR 的逐折差、平均、贏折數。另報分數層級的相加（各自除以該任務 validation 的 σ，t = 4、只作描述）：LR＋GR、LT＋LR＋GR 的 WP（seed 42）。

E5 類別加權：ridge 的每張 train slide 乘權重 1/n_c（n_c = 該折該類別的 train slide 數），累加 A += Σ w·v vᵀ、B_c += Σ w·v。γ 候選改為 {1e-8, 1e-7, 1e-6, 1e-5, 1e-4}，選法相同。做兩個：LR-bal（輸入同 LR）與 GR-bal（輸入同 LIN8）；M3-bal = LT 與 GR-bal 的 σ 固定版相加（β = 1）。報 seed 42–46 的 WP、CIL ACC。

E6 每類別：對 主系統、M3、LR、LR0、LRG、LR-bal、M3-bal（各取 seed 42；LR 另報五個 seed 合計），告訴任務、十折合計：8 個類別各自的張數、正確張數、正確率；每任務的 balanced accuracy（兩類正確率的平均）與四任務平均。另對 LR 做 MOE-1 S1 的表：相對主系統的修好、弄壞、都錯、都對（每類別）。

E7 CL 流程（seed 42）：LR、LRG、LR-bal，每個順序、每個階段 t = 1…4：ACC（只在已學類別中判）、Masked ACC（告訴任務的判定）、每任務 WP、Forgetting、BWT（與 Table 1 同一個函式）。同表列主系統與 M3（取自 moe1/s2.json 並比對）。

E8 儲存：主系統、M3、LR、LRG 各自「所有任務共用」與「每任務」存了什麼、fp32 的 bytes、隨任務數怎麼增加。

### 門檻

G-LR：seed 43、44、45、46 各自：CIL ACC(LR) − CIL ACC(同 seed 主系統) 的十折平均 ≥ +0.007，且差 > 0 的折數 ≥ 7。四個 seed 都滿足才通過（若兩序結果不同，兩序各算、都要滿足）。
G-HEAD：D_head 的十折平均 ≥ +0.005，且 D_head > 0 的折數 ≥ 7。通過代表 ridge 判讀下仍需要 head。
G-CAT：每折取 seed 43–46 的 [WP(LRG(s)) − WP(LR(s))] 平均；十折平均 ≥ +0.005 且 > 0 的折數 ≥ 7。
E3、E5、E6、E7、E8 不設門檻。seed 42 的 LR test 數字在 MOE-1 已看過，只作描述。

---

## 第二部分：操作化細則（與第一部分同時登記；不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `outputs/navcil/mac/moe2/DECISIONS.md`（做了什麼判斷、為什麼），並印在報告內。

### 0. 共通

1. **資料、類別順序、各階段的 AR／LIN8**：同 PREREG-17 細則 1–3（8 類固定序 ESAD, ESCC, CCRCC, PRCC, IDC, ILC, LUAD, LUSC；任務 p 的兩類為第 2p、2p+1 列，
   「第一類」= 第 2p 列；reverse = esca → rcc → brca → lung，paper = lung → brca → rcc → esca；AR／LIN8 = `nc8_report.B8.W`）。
   M3 = PREREG-17 的 σ 固定版（f = d_a/σ_a + d_b/σ_b，β = 1；LIN8 γ = 0.01；σ_b 固定值 = 學任務 j 那個階段的 LIN8 在任務 j validation 上的樣本標準差），
   算術沿用 `moe1_common`（`fused`、`eval_comps`、`sigma_b_table`、`main_eval`）。
2. **head 權重**：seed 42 = `outputs/navcil/mac/i6/r2/fold{f}_{task}.pt`；seed 43–46 = `outputs/navcil/mac/moe1/i6_seed{s}/fold{f}_{task}.pt`（唯讀）。
   `vec` 階段開始時檢查 5 × 10 × 4 = 200 個檔案都在；缺任何一個就停下（寫 `FAILED_vec.txt`），不重訓。
3. **證據向量**（每張 slide 讀一次特徵檔；逐張記錄 `t_read_s`、`t_compute_s`）：
   - v(s)：s = head 的輸出（s0 + g），`four_round(Z, s, λ* = 1.5)`、`mean_norm(Z, idx)`（等權、L2 正規化）。train slide 只算自己任務的 head；
     validation、test 四個任務的 head 都算（CIL 需要 τ̂ 的 head）。
     seed 42 沿用 MOE-1 快取：train = `moe1/cache/s42v_train_fold{f}.pt`，validation／test = `moe1/cache/s42cells_{split}_fold{f}.pt` 的 `v_four`。
     seed 43–46 在本批新算，存 `moe2/cache/vec_{split}_fold{f}.pt`。
   - v0：s = s0 = head 的 `parts(Z, f_task)[0]`（對該任務兩類文字的最大 cosine、slide 內 z-score；與 head 參數無關，同 MOE-0 的 g0），同樣四輪、等權。
     train 用自己任務的文字；validation／test 四個任務都算（CIL 時取 τ̂ 的 s0）。
   - v1(42)：seed 42 的 head 的 s，`one_shot(s)`（前 64，不扣冗餘），等權平均、L2 正規化。train 只算自己任務；validation／test 四個 head 都算。
4. **快取檢查**（`vec` 階段；任何一項不過就停下該階段，之後依賴它的階段不跑）：
   (a) 本批另以 seed 42 的 head 重算 train 的 v(42)，與 `s42v_train` 最大絕對差 ≤ 1e-6；
   (b) seed 43–46 的 validation／test v(s)·Fᵀ 與 `moe1/cache/seed{s}_{split}_fold{f}.pt` 的 `I6_cos8` 最大絕對差 ≤ 1e-6，且每個 head 在自己任務兩類內的 argmax 全同；
   (c) v1(42)·Fᵀ 與 `s42cells` 的「一次取 64／等權」格（`cells_cos8[:, :, 0]`）最大絕對差 ≤ 1e-6、argmax 全同；
   (d) test 的 v0（真實任務的 s0）·Fᵀ 與 `moe0/test_fold{f}.pt` 的 `g0_cos8` 最大絕對差 ≤ 1e-6、真實任務兩類內 argmax 全同；
   (e) validation／test 的 slide id 與標籤與 moe0 快取逐張相同；train 的 slide id 與 NC-8 快取（`B8.c`）、`s42v_train` 逐張相同。
5. **mean_vec**：train = NC-8 快取 `B8.c(f, "train", task)["mean_vec"]`；validation／test = moe0 快取的 `mean_vec`。
6. **累加式 ridge**（LR、LR0、LR1、LRG、LR-bal、GR-bal 共用一個函式）：x = 輸入接常數 1，轉 float64；序 o 的階段 t：依序對前 t 個任務
   A += XᵀX（加權版 A += Xᵀ diag(w) X），B 的第 c 欄 = 類別 c 的 train slides 的 x 之和（加權版 Σ w·x）；B 只含已學類別、依固定 8 類序排列；
   W = `torch.linalg.solve(A + γI, B)`。分數 = xW；任務 q 的分數差 d = 第 2q 欄 − 第 2q+1 欄。d ≥ 0 判第一類（單獨判讀器的規則，同 PREREG-17 細則 4）。
   LR 的輸入 = v（513 維）；LR0 = v0；LR1 = v1；LRG = [v; mean_vec]（1025 維）；LR-bal = v；GR-bal = mean_vec（513 維，同 LIN8）。
   1025 × 1025 的 float64 封閉解超出 AGENTS.md 可攜規則 3 例外的 513 × 513，依本批指令「closed-form 一律 float64」執行，在報告揭露。
7. **告訴任務（WP、Masked ACC）與 CIL**：告訴任務 = v 取自真實任務的 head，在真實任務兩欄內判；CIL = TP 為該階段 AR（γ = 1e-3）在已學任務中 argmax 得 τ̂，
   v 取自 τ̂ 的 head（v0 取 τ̂ 的 s0；LRG 的 mean_vec 不變），在 τ̂ 的兩欄內判。t = 4 的結果：每個序用該序自己的累加（W 依該序累加）與該序的 AR。
   WP = 每折每任務的 2 類正確率、四任務等權；CIL ACC 同（四任務等權）。ridge 判讀器的 Masked ACC = 其告訴任務判定的正確率（= WP）。
8. **γ 的選法**：每個變體各自選一次；以 seed 42（LR0、GR-bal 沒有 seed，用自己的向量）的 validation、reverse 序、t = 4、告訴任務 WP
   （四任務等權）的十折平均取最大者；平均值完全相等才算同分，同分取較大的 γ。選定之後才算 test。LR 的 γ\* 供 seed 42–46 的 LR、
   E2 的 D_head、E3、E4 的 LRG − LR、E6、E7 共用；LRG、LR-bal 的 γ\* 同樣五個 seed 共用。選在網格邊界時照報並註明，不延伸。
9. **逐折相減**：`moe1_common.paired`（平均差；贏折數 = 差 > 1e-12 的折數；Wilcoxon = `scipy.stats.wilcoxon` 雙尾、`zero_method = "wilcox"`，
   全為 0 時 p = 不適用）。mean ± sd 的 sd 為十折的樣本標準差。
10. **兩序是否逐張相同**（E1）：每個 seed，比較兩序 t = 4 的 test 判定：告訴任務判定不同的張數、CIL 判定不同的張數、告訴任務 d 的最大絕對差。
    凡門檻與「若兩序結果不同」有關者（G-LR、G-HEAD、G-CAT），一律兩序各算、兩序都滿足才通過（兩序相同時兩者等價）。
11. **主系統、M3 的各 seed 版本**：主系統 = `main_eval`（seed s 的 `I6_cos8`：seed 42 取 moe0 快取，seed 43–46 取 `moe1/cache/seed{s}_{split}_fold{f}.pt`）；
    M3 = σ_a 用該 seed 的 head 在 validation 上的 d_a、σ_b 用 LIN8 固定值（兩序）。M3 的逐折 CIL ACC 另與 `s4.json`／`s7.json` 的 M3 比對（只報最大絕對差，不是 K）。

### 1. 一致性檢查

12. **K1**：同 PREREG-17 細則 6（`moe1_common.k1`）。E1–E7 每個階段開始時都跑；不過就停下該階段。
13. **K2**（E1）：照 MOE-1 S5 的算法（PREREG-17 細則 23、26；W 依 reverse 序累加、γ = 1e-3、seed 42 的四輪等權向量）重算：
    test 告訴任務 WP 的逐折值與 `moe1/s5.json` 的 `splits.test.alone.LR.wp`、兩序的 CIL ACC 逐折值與 `lr_cil[序].cil` 的最大絕對差 ≤ 1e-9；
    十折全跑時另檢查平均四捨五入到小數第 4 位為 0.9457、0.9252。（MOE-1 S5 的 CIL 兩序都用 reverse 累加的 W；K2 照該定義重算。本批 E1 的兩序結果用各自的累加，見細則 7。）
14. **K3**（E1）：seed 43、44 的主系統兩序逐折 CIL ACC 與 `moe1/s4.json` 的 `rows[s][序].main[折].cil`、seed 45、46 與 `moe1/s7.json` 的同欄位，最大絕對差 ≤ 1e-9。
15. **失敗處理**：任何階段的一致性檢查或快取檢查不過、或程式出錯 → 該階段停止、寫 `FAILED_<階段>.txt`、不重試、不改期望值；
    `moe2_run_all.sh` 接著跑不依賴它的階段（`vec` 失敗則 E1–E7 不跑；E8 不依賴 `vec`）。各 E 階段彼此獨立（各自重算所需的 γ 選擇，算法相同、結果決定性）。
16. **冒煙測試**：同一支 `moe2_run_all.sh`，`MOE2_OUT=moe2_smoke MOE2_FOLDS=1`，報告寫到 `moe2_smoke/REPORT_smoke.md`；數字不寫進 `REPORT_moe2.md`；
    十折平均的四捨五入檢查在冒煙測試中不適用。

### 2. 各節

17. **E1**：seed 42–46 各自：LR 的 WP、CIL ACC（mean ± sd）、每任務 WP（十折平均）；LR − 主系統、LR − M3 的 WP 與 CIL ACC 逐折相減（兩序）。
    **G-LR**：seed 43–46 × 兩序：CIL ACC(LR) − CIL ACC(主系統) 的十折平均 ≥ +0.007 且差 > 1e-12 的折數 ≥ 7；八個條件都滿足才通過。
18. **E2**：LR0 的 WP、CIL ACC、每任務 WP（兩序）。D_head（每序）：每折 = 五個 seed 的 WP(LR(s)) 平均 − WP(LR0)。
    **G-HEAD**：每序 D_head 的十折平均 ≥ +0.005 且 D_head > 1e-12 的折數 ≥ 7；兩序都滿足才通過。
    LT 在 g = 0：test、真實任務的 v0 與兩類文字的 cosine，2 類 argmax（同分取第一類）；每任務十折平均與 `moe0/results.json` 的 `B2.matrix[4]`
    逐格比對，報最大絕對差（描述性比對，不是 K；cosine 層級的檢查在細則 4(d)）；另報四任務等權 WP 的 mean ± sd。
19. **E3**：LR1 自己選 γ；WP、CIL ACC（兩序）；LR1 − LR(42) 的逐折相減（WP、CIL ACC）。
20. **E4**：LRG 自己選 γ；seed 42–46 的 WP、CIL ACC（兩序）；LRG − 同 seed LR 的逐折相減（WP、CIL ACC）。
    **G-CAT**（每序）：每折 = seed 43–46 的 [WP(LRG(s)) − WP(LR(s))] 平均；十折平均 ≥ +0.005 且 > 1e-12 的折數 ≥ 7；兩序都滿足才通過。
    分數層級的相加（seed 42、t = 4、告訴任務、validation 與 test 都報）：GR = LIN8（γ = 0.01、reverse 序、t = 4），LR = LR(42)（reverse 序、γ\*），
    LT = 主系統的 d_a；各自除以該折該任務 validation 上 d 的樣本標準差後相加；f = 0 時看式子第一項的符號（≥ 0 判第一類）；項的順序照名稱（LR＋GR、LT＋LR＋GR）。
    報 WP（mean ± sd）與相對 LT 的逐折相減。
21. **E5**：權重 w = 1/n_c，n_c = 該折 train split 中類別 c 的張數（每類權重和 = 1）。LR-bal、GR-bal 各自選 γ（候選 {1e-8, …, 1e-4}）。
    M3-bal(s) = LT(seed s) 的 d_a/σ_a ＋ GR-bal 的 d_b/σ_b，σ_b 的固定值 = 學任務 j 那個階段的 GR-bal 在任務 j validation 上的樣本標準差（σ 固定版，兩序）；
    CIL 的 TP 仍為 AR（γ = 1e-3、不加權）。報 seed 42–46 的 LR-bal、M3-bal 的 WP、CIL ACC（兩序）；GR-bal 單獨的 WP、CIL ACC；
    另報 LR-bal − LR、M3-bal − M3 的逐折相減（描述）。
22. **E6**：seed 42、test、告訴任務、t = 4；M3、M3-bal 與各 ridge 判讀器用 reverse 序（告訴任務 t = 4 的判定與 MOE-1 S5 相同的序）。
    每類別：張數、正確張數、正確率（十折合計）；每任務 balanced accuracy = 十折合計的兩類正確率平均；四任務平均。LR 另列 seed 42–46 合計（張數 × 5）。
    LR 相對主系統（主系統的告訴任務判定 = 真實任務 head 的 2 類 argmax）：每類別的修好、弄壞、都錯、都對（十折合計），另列每任務。
23. **E7**：seed 42；LR、LRG、LR-bal 用各自的 γ\*，W 依該序累加到階段 t（只含已學任務）；`nc5_report.cil_full` 計 R、ACC_t、Masked ACC_t、Forgetting、BWT；
    每任務 WP = 告訴任務判定在該任務的正確率（= `Rm`）。主系統與 M3（σ 固定版）照 `moe1_s2.py` 的算法重算，表格列 `moe1/s2.json` 的數值，
    並報重算與 `s2.json` 的最大絕對差（ACC、Masked、Forgetting、BWT、逐階段 ACC／Masked ACC）。
24. **E8**：由實際張量形狀計算（d = 512、x = 513 或 1025）。只算推論與繼續學習時必須保存的量：ridge 存累加統計量 A、B（W 可由 A、B 解出，不另存）；
    head 參數（I6 r = 2：516·2 + 1 = 1033 個）；每任務兩類的文字特徵（2 × 512）；σ（每任務 2 個純量）。CONCH 骨幹、patch 特徵、λ\*、logit scale 各系統相同，
    列出但不計入差異。報每個系統的「共用」bytes S、「每任務」bytes p，總量 = S + T·p（T = 任務數），以及 T = 4 的數值；fp32。

### 3. 執行

25. 階段：`vec`（向量快取與細則 4 的檢查）→ `e1` … `e7` → `e8`。每階段完成寫 `<階段>.json` 與 `<階段>.done`，重跑自動跳過；`vec` 另有每（split、折）的 done 標記。
26. 啟動：`tmux new-session -d -s moe2 "env NAVCIL_MACHINE=mac MOE2_PY=<venv python> caffeinate -dimsu scripts/moe2_run_all.sh"`；
    `HEARTBEAT.log` 每 900 秒一行，每階段開始與結束各一行；每階段結束後重產報告（`scripts/moe2_report.py`）；失敗不重試；跑完不關機、不做 git 操作。
27. 報告：開頭三個門檻與 K1–K3；E1–E8 各一節，表格標明本檔落點；最後列失敗的階段（`FAILED_*.txt` 摘要）、`DECISIONS.md` 全文、各階段耗時。只給數字與表格。
28. seed 42 的 LR test 數字在 MOE-1 已看過：本批 seed 42 的 LR 結果只作描述；LR 的確認只看 G-LR（seed 43–46）。
