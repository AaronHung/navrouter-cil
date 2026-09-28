# PREREG-11 — NC-11 預先註冊（逐折配對檢定、top-K ablation、選取方式 ablation）

登記時間：2026-09-29，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro，10 核、16 GB），CPU、`torch.set_num_threads(8)`；同一子實驗的所有 arm 在同一次執行中完成。
分支：`ws3-ablation`（worktree `~/research/01_navrouter-cil-ws3`，自 main @ ee2be2c）。
主方法 D3 = AR 分派器（γ = 1e-3）＋ I6(r = 2) 任務分類頭。十折、reverse 與 paper 兩序。

共用資料（gitignored，唯讀，以參數指向主工作樹）：NC-8 同一批快取 `outputs/navcil/mac/cache/nc8_fold*_{train,test}_*.pt`
（`--cache-dir`）、既有 I6(r=2) 權重 `outputs/navcil/mac/i6/r2/`（`--i6-ref-dir`）。特徵檔：`can_dataset/<task>/feats-l1-s256_CONCH/`。
AR 一律用 `selector/incremental_ridge.py`（與 NC-8 逐位元相同，REPORT_stage11），第 t 階段只讀任務 t 的 train mean_vec。
CIL 定義同 `scripts/nc8_report.py`（`nc5_report.cil_full`、`nc2_report.hard`）：task-known = oracle ＋ I6，task-inferred = AR ＋ I6，
主指標為 t = 4 的 ACC（四任務等權平均），十折 mean ± sd，兩序分列。

## 3.1 逐折配對檢定（不重新訓練、不重讀特徵）

- 資料：`outputs/navcil/mac/nc10/per_fold.json`（WS2，同一批 I6 頭下的 oracle、R0、text-organ、R3、AR）；AR 另與
  `nc8/per_fold.json` 第 6 列逐折核對（應完全相同）。
- 比較：AR vs R3(k=8)；AR vs R0（REPORT_stage12 已存在，故執行）。指標：t = 4 分派正確率（micro、macro）與 CIL 全程 ACC，兩序各一次。
- 輸出：10 列表（折 | 對照 | AR | 差）；Wilcoxon signed-rank（雙尾，`scipy.stats.wilcoxon`，預設 zero_method = "wilcox"）與
  paired t（雙尾，`scipy.stats.ttest_rel`）的 p 值；AR 勝的折數（差 > 0；平手另計）。
- AR 變差的折（任一指標差 < 0）：列出該折 AR 與對照的 4 × 4 分派混淆（由 NC-8 快取以 `nc8_report.router`／`nc10` 同一程式重算
  該折 τ̂；十折合計須與 `nc10/confusion.json` 相同），並把該折 CIL 差拆成「只有 AR 分派對的片」與「只有對照分派對的片」中
  第二站判對的張數（同一張片兩者都分派對時第二站結果相同），作為原因初判。
- 不設門檻；p 值照實報告，不做多重比較校正（另列檢定次數）。
- 執行時間估算：< 1 分鐘。

## 3.2 top-K ablation（每個 K 重新訓練 I6）

- 先確認（唯讀）：I6 以 one-shot top-64 ＋ softmax(分數) 加權訓練（`selector/cil_ops.py:87-90`，`budget` 預設 64，
  `scripts/nc7_i6.py:60-61` 未傳 budget），故每個 K 都要重新訓練。
- arm：K ∈ {16, 32, 64, 128, 全部}。
  - 訓練：與 `scripts/nc7_i6.py` `train_one` 相同（r = 2、seed 42、5 epochs、lr 5e-4、wd 1e-4、每步有限性檢查、threads 8），
    只把 `train_selector` 的 `budget` 設為 K（「全部」= budget 0，`top_k_select` 回傳全部 patch，softmax 加權涵蓋全部 patch）。
  - 推論：`four_round(Z, s, λ* = 1.5, budget = K, step = 16)`，輪數 = K／16（K = 16 為 1 輪、K = 128 為 8 輪）；所選 patch 等權平均
    L2 正規化後對 8 類文字取 cosine（同 NC-8）。patch 數少於 K 的 slide 取全部。
  - 「全部」的推論不做選取：全部 patch 等權平均（= 整片 mean_vec；此時任務分類頭不影響結果）。另報一個 arm
    「全部（softmax 加權）」：全部 patch 以 softmax(分數) 加權（與訓練時的聚合相同），不列入結論句。
- 每折 test slides 讀一次，同一次讀取中算完所有 arm 的 4 個任務分類頭證據；AR 分派結果與 K 無關。
- 指標：task-known、task-inferred（t = 4 ACC，十折 mean ± sd，兩序）；另存逐階段 ACC、Masked ACC、Forgetting。
- 重現檢查（不符即停，不繼續、不改判準）：
  a. K = 64 重新訓練的 40 個權重檔與 `i6/r2/` 逐位元相同（若不同，記錄最大差，接著看 b）；
  b. K = 64 的 task-known 兩序 = 0.9340 ± 0.0202（REPORT_stage10.md:18、:31），task-inferred 兩序 = 0.9128 ± 0.0258
     （REPORT_stage10.md:16、:29），逐折與 `nc8/per_fold.json` 第 8、6 列在小數點後四位相同。b 不符即停。
- 結論句：以 K = 64 為參照，報告「K 在某範圍內與 K = 64 的差距不超過 X pp」這類描述，不寫「最佳」。
- 執行時間估算：訓練以 NC-7 實測 r = 2 十折四任務 1,134 秒（`nc7/timing_i6.json` `train_total/r2`）為基準，K = 16／32／64／128
  各約 19–25 分鐘，「全部」因 softmax 與分類涵蓋全部 patch 估 1.5–2 倍（約 30–40 分鐘）；評估（十折 test、6 個 arm × 4 個頭）約 10–20 分鐘。
  合計約 2–2.5 小時；與其他工作同時執行時估上限 4 小時（< 12 小時）。

## 3.3 選取方式 ablation（訓練不變）

- 任務分類頭沿用既有 I6(r=2) 權重（one-shot top-64 訓練，不重新訓練）。
- arm：(i) 4 輪 × 16，λ* = 1.5 去重懲罰（現行，`selector/cil_ops.py:39-45`）；(ii) 一次取 top-64，不去重（`one_shot`，`cil_ops.py:48-49`）。
  兩者都等權平均後對 8 類文字取 cosine。
- 每折 test slides 讀一次，兩個 arm × 4 個頭在同一次讀取中算完。
- 指標：同 3.2。另報兩 arm 所選 patch 的 Jaccard（自家任務的頭）。
- 重現檢查（不符即停）：arm (i) 的 8 類 cosine 與 NC-8 快取 `I6_cos8` 逐張比對（報最大絕對差），且 task-known／task-inferred
  兩序與 REPORT_stage10.md:18、:31、:16、:29 相同，逐折與 `nc8/per_fold.json` 在小數點後四位相同。
- 報告開頭以一段白話說明去重懲罰的作用，並附機制與行號（`selector/multiround.py:139-180`）。
- 執行時間估算：十折 test 讀一次 ＋ 2 × 4 次選取，約 5–10 分鐘。

## 執行與停止規則

- 順序：3.1 → 3.3 → 3.2，各自一次執行；以 `scripts/run_stage.sh` 開 tmux，內層另加 `caffeinate -dimsu`；程式每 15 分鐘在 log 寫一行
  「子實驗、完成數／總數、失敗數」。
- 每（K、折、任務）訓練完成寫 done 標記；任何一個 run 失敗（例外、非有限值、重現檢查不符）就停下回報，不重試、不跳過。
- 每張 slide 記錄 t_read_s、t_compute_s。
- 輸出：`outputs/navcil/mac/nc11/`（3.1：`paired.json`；3.2：`topk/`；3.3：`select/`）與 `outputs/navcil/mac/REPORT_stage13.md`。
- 不改既有腳本與既有 outputs。
