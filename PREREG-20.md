# PREREG-20 — MOE-4 預先註冊（定案確認：head 一次取 64 ＋ ridge 判讀 γ = 1e-3，多 seed、逐階段）

登記時間：2026-10-03，MOE-4 任何程式執行前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`、torch 2.11.0；本輪所有數字來自同一台、同一批。只算向量與 closed-form，不訓練任何 head。
分支：`moe-4`（自 `moe-3` @ fb2a6d8）。不合併；不修改既有程式、既有報告與既有 PREREG；新程式放新檔
（`scripts/moe4_*.py`、`scripts/moe4_run_all.sh`）。
報告：`outputs/navcil/mac/REPORT_moe4.md`（`scripts/moe4_report.py` 產生）。

本檔＝指令原文的「操作定義」「一致性檢查」「要報的」「門檻」全文（第一部分），加上執行前同時登記的操作化細則（第二部分；不改變第一部分）。

---

## 第一部分：指令原文

### 共同設定

--device cpu、8 條執行緒；closed-form 一律 float64；十折；reverse 與 paper 兩序；四任務等權；同一批、同一台機器。沿用既有快取與 head 權重（seed 42：i6/r2/；seed 43–46：moe1/i6_seed{s}/）。不訓練任何 head；缺權重就停下回報。CIL 的 TP 一律 AR（γ = 1e-3）。

### 操作定義

向量（train、validation、test 都算；train slide 用自己任務的 head；validation／test 四個任務的 head 都算，CIL 取 τ̂ 的）：
- w(s) = seed s 的 head，s = s0 + g，一次取前 64 個 patch（不扣冗餘），原始 Z 等權平均後 L2 正規化（512）。seed 42 沿用 MOE-2 的 v1(42) 快取；seed 43–46 由特徵檔新算。
- v(s) = 同一個 head 的四輪版本（沿用 MOE-2 快取）。
- u_64 = 不用 head 的一次取 64（沿用 MOE-3 快取）。

判讀器：RDG(γ)：輸入 [向量; 1]，8 類，依序每學一個任務累加 A、B，每個階段重解 W = (A + γI)⁻¹ B；告訴任務時在真實任務兩類內判，CIL 時向量取自 τ̂、在 τ̂ 兩類內判。TXT：與該任務兩類文字比 cosine。
γ 固定為 1e-3，不做任何選擇（依據：MOE-3 F6，三種輸入向量以逐階段 validation 目標都選到 1e-3，且不在邊界）。
系統（s = 42…46）：
  P-F(s) = w(s)／RDG(1e-3)　　定案候選
  P-4(s) = v(s)／RDG(1e-3)　　四輪版本
  P-T1(s) = w(s)／TXT　　　　主系統改成一次取 64
  P-main(s) = v(s)／TXT　　　主系統
  P-0 = u_64／RDG(1e-3)　　　不用 head（= MOE-3 的 S-R0）

### 一致性檢查（不過就停，並擋住後面階段）

K1 P-main(42)：t = 4 的 WP 0.9340、CIL ACC 0.9128，與 nc8/per_fold.json 逐折最大絕對差 ≤ 1e-9；逐階段 ACC 與 moe1/s2.json 最大絕對差 ≤ 1e-9。
K2 P-4(42)：t = 4 的 WP、CIL ACC 與 moe1/s5.json 的 LR（γ = 1e-3）逐折最大絕對差 ≤ 1e-9（平均 0.9457、0.9252）。
K3 P-0：逐階段 ACC、WP 與 moe3 的 S-R0（f2.json）最大絕對差 ≤ 1e-9。
K4 P-main(s)，s = 43…46：t = 4 的 CIL ACC 與 moe2/e1.json 逐折最大絕對差 ≤ 1e-9。
K5 w(42)·Fᵀ 與 moe1 s42cells「一次取 64／等權」格的 cosine 最大絕對差 ≤ 1e-6、argmax 全同。

### 要報的

H1 逐階段：P-F、P-4、P-T1、P-main 的每個 seed、每個順序、t = 1…4：ACC（只在已學類別中判）、告訴任務的 WP（已學任務等權）、每任務 WP；Forgetting、BWT（與 Table 1 同一個函式）；四個階段 ACC 的平均 Ā。十折 mean ± sd。P-0 列一次。另列「五個 seed 的平均（先對 seed 平均再算十折 mean ± sd）」。
H2 t = 4：每個 seed 的 WP、CIL ACC；逐折相減（平均、贏折數、Wilcoxon 雙尾 p、逐折值）：P-F(s) − P-main(s)、P-F(s) − P-4(s)、P-F(s) − P-0、P-F(s) − P-T1(s）。確認 P-F 兩序在 t = 4 是否逐張相同。
H3 每類別（t = 4、告訴任務、五個 seed 合計與 seed 42 各一份）：8 類各自的正確張數與正確率；每任務 balanced accuracy 與四任務平均。系統：P-F、P-4、P-main、P-0。
H4 γ 敏感度（只作描述）：P-F 在 γ ∈ {1e-4, 1e-3, 1e-2} 的 t = 4 CIL ACC 與 Ā（五個 seed 平均，兩序）。
H5 儲存：P-F 的共用與每任務 bytes（fp32），格式同 moe3 F7。

### 門檻

G-CONF：seed 43、44、45、46 各自、兩序各自：CIL ACC(P-F(s), t = 4) − CIL ACC(P-main(s), t = 4) 的十折平均 ≥ +0.007 且差 > 0 的折數 ≥ 7。全部滿足才通過。
G-TRAJ：seed 43–46 各自、兩序各自：Ā(P-F(s)) − Ā(P-main(s)) 的十折平均 ≥ −0.005。全部滿足才通過。
G-ONE：每折取 seed 43–46 的 [CIL ACC(P-F(s)) − CIL ACC(P-4(s))]（t = 4）平均；十折平均 ≥ −0.003。
G-CONF、G-TRAJ 的同一組數字也對 P-4 算一次，只報告、不判定。
seed 42 的數字只作描述。H3、H4、H5 不設門檻。

---

## 第二部分：操作化細則（與第一部分同時登記；不改變第一部分）

未列於此、執行中才遇到的判斷，寫入 `outputs/navcil/mac/moe4/DECISIONS.md`（做了什麼判斷、為什麼），並印在報告內。

### 0. 共通

1. **資料、類別順序、AR**：同 PREREG-19 細則 1（8 類固定序；任務 p 的兩類為第 2p、2p+1 列；reverse 與 paper 的任務序同 PREREG-19；階段 t 的 AR = `moe1_common.ar_stage`，γ = 1e-3）。
2. **向量來源**：
   - w(42) = MOE-2 的 `one42` 快取（`moe2/cache/vec_*`；= `one_shot(s0 + g)`，seed 42 的 head）。
   - v(42) = `moe1/cache/s42v_train`（train）與 `s42cells_*`（`v_four`，validation／test）。
   - v(43…46) = MOE-2 `vec_*` 快取的 `s43`…`s46`（四輪，`four_round`）。
   - u_64 = MOE-3 `cache/u_{split}_fold{f}.pt` 的 `u[64]`（唯讀；本批不重算）。
   - w(43…46)：本批 `vec` 階段新算（細則 3）。
3. **`vec` 階段（新算 w(43…46)）**：每張 slide 讀一次（`selector.evaluate.read_slide`）。對 seed s ∈ {43…46}、任務 p：`score = head_{s,p}(Z, f_task_p)`（`I6Expert.forward`，即 s0 + g）；
   `idx = top_k_select(score, 64)`（patch 數 ≤ 64 的 slide 取全部，同 MOE-3 D3）；`w = mean_norm(Z, idx)`。train 只算自己任務的 p；validation、test 四個任務都算（`[n, 4, 512]`）。
   head 權重路徑：`moe1/i6_seed{s}/fold{f}_{task}.pt`；缺檔就停（不重訓）。存 `cache/w_{split}_fold{f}.pt`（每（split、折）一個 done 標記）。
   逐張記錄 `t_read_s`、`t_compute_s`。
   **對齊檢查**（不過就停下 `vec`）：w 快取的 slide id 與標籤，與 MOE-2 `vec_*` 快取逐張相同；所有 w 向量有限、範數與 1 的差 ≤ 1e-5。
4. **累加式統計量與判讀器**（同 MOE-3 細則 3）：x = [向量; 1]（float64）；序 o 的階段 t：依序對前 t 個任務 A += XᵀX，B 的第 c 欄 = 類別 c 的 train slides 的 x 之和（只含已學類別、固定 8 類序）。
   RDG(γ)：W = `torch.linalg.solve(A + γI, B)`（513 × 513）。分數 = xW；任務 q 的分數差 d = 第 2q 欄 − 第 2q+1 欄；d ≥ 0 判第一類（`moe2_common.pred_of`）。
   TXT：cos = 向量 · Fᵀ（float32、逐張相乘，`moe2_common.cos_rows`）；d = 任務 q 第一類 − 第二類的 cosine；d ≥ 0 判第一類。
5. **告訴任務與 CIL（每個階段 t）**：只對已學任務的 test slides 算（`moe3_common.run_cl`）。告訴任務 = 向量取自真實任務的 head（w(s)、v(s) 取真實任務的版本）；CIL = τ̂ 為階段 t 的 AR 在已學任務中的 argmax，向量取自 τ̂。
   ACC_t、Forgetting、BWT 由 `nc5_report.cil_full` 計（Table 1 同一個函式）；告訴任務的 WP_t = 已學任務的每任務 2 類正確率等權平均；Ā = 四個階段 ACC_t 的平均（每折先算，再取十折 mean ± sd）。
6. **γ 不選擇**：RDG 的 γ = 1e-3 固定。H4 的 {1e-4, 1e-2} 只作描述，不改定案。
7. **系統**：P-F(s)、P-4(s)、P-T1(s)、P-main(s)、P-0 依 PREREG-20 第一部分；P-0 與 MOE-3 S-R0 的差異只在 γ 來源（S-R0 的 γ\* 由 MOE-3 hp.json 選得 1e-3，兩者相同，見 K3）。
8. **逐折相減**：`moe1_common.paired`（平均差；贏折數 = 差 > 1e-12 的折數；Wilcoxon = `scipy.stats.wilcoxon` 雙尾，全為 0 時 p = 不適用）。mean ± sd 的 sd 為十折的樣本標準差。
9. **兩序是否逐張相同**（H2）：t = 4 的 test：告訴任務判定不同的張數、CIL 判定不同的張數、告訴任務 d 的最大絕對差（`moe3_common.order_same`）。

### 1. 一致性檢查（`chk` 階段；任何一項不過 → 該階段停止、寫 `FAILED_chk.txt`，不重試、不改期望值）

10. **K1**：P-main(42) 的 test 逐折 WP（四任務等權）、逐任務 WP、CIL ACC 對 `moe1_common.nc8_main`（= nc8/per_fold.json 第 6 列）的最大絕對差 ≤ 1e-9；
    另 P-main(42) 每序每階段 ACC 對 moe1/s2.json 的 `orders[序].main[折].acc_t` 最大絕對差 ≤ 1e-9。十折全跑時，平均四捨五入到小數第 4 位須為 WP 0.9340、CIL ACC 0.9128。
11. **K2**：P-4(42) 的 test t = 4：WP（兩序各算）與 CIL ACC 對 `moe1/s5.json` 的 `splits.test.alone.LR.wp[折]` 與 `lr_cil[序].cil[折]` 逐折最大絕對差 ≤ 1e-9；十折全跑時，WP 平均四捨五入為 0.9457、CIL 平均為 0.9252。
12. **K3**：P-0 的 `acc_t`、`wp_t`、`forgetting`、`bwt`（兩序、四階段）對 `moe3/f2.json` 的 `rows[序]["S-R0"]` 最大絕對差 ≤ 1e-9（`moe3_common.row_diff`）。
13. **K4**：P-main(s)，s = 43…46：兩序各自、每折的 t = 4 CIL ACC 對 `moe2/e1.json` 的 `seeds[s].orders[序].main[折].cil` 最大絕對差 ≤ 1e-9。
14. **K5**：w(42) 的 cosine（`cos_rows(w42, Fᵀ)`，validation 與 test 的四個任務）對 `moe1/cache/s42cells_{split}_fold{f}.pt` 的 `cells_cos8[:, :, 0]`（「一次取 64／等權」格）最大絕對差 ≤ 1e-6；
    每任務 2 類 argmax 全同（同 MOE-2 E2 的 c 檢查）。
15. **失敗處理**：`vec` 失敗 → `chk`、`f2`、`f3`、`f4`、`f5` 都不跑。`chk` 失敗 → `f2`、`f3`、`f4`、`f5` 都不跑（K 檢查的正是本批判讀器與向量的實作；不過就不產生結果）。
    各 stage 開始時讀取 `chk.json`，`pass` 不為真就拒絕執行（防止手動單獨重跑）。
16. **冒煙測試**：同一支 `moe4_run_all.sh`，`MOE4_OUT=moe4_smoke MOE4_FOLDS=1`，報告寫到 `moe4_smoke/REPORT_smoke.md`；數字不寫進 `REPORT_moe4.md`；十折平均的四捨五入檢查與門檻在非十折時不作判定。

### 2. 門檻

17. **Ā 與逐折差**：Ā 每折每系統每序 = 四個階段 ACC_t 的平均。逐折差 = 同一折的兩個系統相減（同 seed、同序、同折）。
18. **「差 > 0 的折數」**以差 > 1e-12 計（同 PREREG-19 細則 7 與 16–18 的寫法）。
19. **G-CONF**（`f2`）：s ∈ {43, 44, 45, 46}、序 ∈ {reverse, paper} 共 8 格各算：D_f = CIL ACC(P-F(s), t = 4) − CIL ACC(P-main(s), t = 4)；十折平均 ≥ +0.007 且 D > 1e-12 的折數 ≥ 7。8 格全部滿足才通過。
20. **G-TRAJ**（`f2`）：s ∈ {43…46}、序共 8 格各算：D_f = Ā(P-F(s)) − Ā(P-main(s))；十折平均 ≥ −0.005。8 格全部滿足才通過。
21. **G-ONE**（`f2`）：每序各算：每折 D_f = mean_{s∈{43…46}} [CIL ACC(P-F(s), t = 4) − CIL ACC(P-4(s), t = 4)]；十折平均 ≥ −0.003。兩序都滿足才通過。
    （指令未指定序；取與 G-CONF、G-TRAJ 相同的「全部滿足」。在 t = 4，兩序的 A、B 與告訴任務判定在精確算術下相同，兩序的數字預期一致。）
22. **P-4 的同一組數字**：G-CONF、G-TRAJ 把 P-F(s) 換成 P-4(s) 算一次，列在同表，只報告、不判定（不寫「通過／未通過」）。
23. **seed 42**：G-CONF、G-TRAJ 的 seed 42 數字只作描述（不在門檻內）；G-ONE 的平均只用 seed 43–46。

### 3. 各節

24. **H1**（`f2`）：P-F、P-4、P-T1、P-main 的 seed 42–46 各一列、兩序、t = 1…4：ACC_t、WP_t（告訴任務、已學任務等權）、每任務 WP（未學為 —）、Forgetting、BWT、Ā；十折 mean ± sd。
    「五個 seed 平均」：每折先對 seed 42–46 平均（ACC、WP、每任務 WP、Forgetting、BWT、Ā 各自平均），再算十折 mean ± sd。P-0 每序一列。
25. **H2**（`f2`）：每個 seed：t = 4 的 WP 與 CIL ACC（十折 mean ± sd）；逐折相減 P-F(s) − P-main(s)、P-F(s) − P-4(s)、P-F(s) − P-0、P-F(s) − P-T1(s)：WP 與 CIL ACC 各一組，列平均、贏折數（> 1e-12）、Wilcoxon 雙尾 p、逐折值。兩序是否逐張相同（細則 9）每 seed 一列。
26. **H3**（`f3`）：t = 4、告訴任務、reverse 序（告訴任務判定在 t = 4 與序無關，見細則 21 的理由；同 MOE-3 F4 取 reverse）、test：
    系統 P-F(s)、P-4(s)、P-main(s) 的「五個 seed 合計」（五個 seed × 十折的正確張數加總）與「seed 42」各一份；P-0 只一份（seed 無關，列十折合計）。
    每系統：8 類各自的張數、正確張數、正確率；每任務 balanced accuracy（= 該任務兩類正確率的平均）與四任務平均。
27. **H4**（`f4`）：P-F 在 γ ∈ {1e-4, 1e-3, 1e-2}：每折先對 seed 42–46 平均 CIL ACC_4 與 Ā，兩序分別列；再算十折 mean ± sd。只作描述，不改 γ。
28. **H5**（`f5`）：由實際張量形狀計算（d = 512、x = 513）。P-F 的「所有任務共用」與「每任務」存了什麼、fp32 的 bytes、總量 = S + T·p 與 T = 4 的數值；
    共用：A（w 的 XᵀX，513²）、A_mv（AR 的 XᵀX，513²）；每任務：I6 head（參數數取自 `load_head` 的實際權重）、兩類文字特徵（2 × 512；head 的 s0 與 TXT 都要用）、B_w（該任務兩欄，513 × 2）、B_ar（該任務一欄，513）。
    主系統 P-main、P-4 與 P-0 同表列為對照（P-0 不存 head）。CONCH 骨幹、patch 特徵、超參數純量列出但不計入。

### 4. 執行

29. 階段：`vec` → `chk` → `f2` → `f3` → `f4` → `f5`。每階段完成寫 `<階段>.json` 與 `<階段>.done`，重跑自動跳過。失敗不重試，寫 `FAILED_<階段>.txt`。
30. 啟動：`tmux new-session -d -s moe4 "env NAVCIL_MACHINE=mac MOE4_PY=<venv python> caffeinate -dimsu scripts/moe4_run_all.sh"`；
    `HEARTBEAT.log` 每 900 秒一行，每階段開始與結束各一行；每階段結束後重產報告（`scripts/moe4_report.py`）；跑完不關機、不做 git 操作。
31. 報告：開頭三個門檻與 K1–K5；H1–H5 各一節，表格標明 PREREG-20 的落點；最後列失敗的階段（`FAILED_*.txt` 摘要）、`DECISIONS.md` 全文、各階段耗時。只給數字與表格。
