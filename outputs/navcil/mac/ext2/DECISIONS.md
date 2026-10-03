EXT-2 的判斷紀錄（2026-10-03）。每條：做了什麼判斷、為什麼。D1–D9 在 PREREG-22 commit 時寫入；之後為執行中新增。

| # | 判斷 | 為什麼 |
|---|---|---|
| D1 | 【PI 裁決，承 EXT-1】外部對照表註加兩句：reverse 序為 b8（論文設定）、paper 序為 b16；匯入欄為 acc@mid（argmax、不含 test 資訊），該欄在兩序都高於 test 最佳門檻的 acc（reverse 0.9352 對 0.9240；paper 0.9261 對 0.9150；`outputs/external/` 該目錄的 PROVENANCE.md），對外部方法有利。不重跑 paper 序。 | 指令 0-1a。 |
| D2 | 【PI 裁決，承 EXT-1】MergeSlide 做法 1 不做（成本見 `outputs/external/mergeslide/DECISION.md`）；做法 2–5 不做；論文只放相關工作與表註。本批的表沒有 MergeSlide 列。 | 指令 0-1b。 |
| D3 | 【PI 裁決，承 EXT-1】M2 只報 WP；CIL 與 t < 4 的欄填「—」，不補算。本批不重算 M2。 | 指令 0-1c。 |
| D4 | 「PREREG-20 的 trajectory 驗證規則」取 PREREG-19 細則 5 的目標（兩序 × t = 1…4 的 validation 告訴任務 WP 平均，十折平均；grid {1e-5…1e-1}；同分取較大的 γ）。 | PREREG-20 本身寫「γ 固定為 1e-3，不做任何選擇」，依據是 MOE-3 F6 的逐階段 validation 目標；規則與 grid 定義在 PREREG-19 細則 5、`moe3_common.val_objective`／`select`。 |
| D5 | K 掃描取既有的 u_K（依 s0 一次取前 K 個、不扣冗餘）；「一次 top-64」＝ K = 64 那一列。四輪版本的 K ≠ 64 不算。 | FINAL_RESULTS B 表的 K 列本來就是不用 head 的 u_K。四輪的 K ≠ 64 沒有既有定義（每輪幾張未定）；不為了填表新設計算法（PI 2026-10-03 對 EXT-1 的裁決精神）。 |
| D6 | 對照列（zero-shot、LIN8、主系統、FINAL-A 的五個 seed）在本批重算一次，寫到 `ext2/a/`，並以 K9 與 EXT-1 的逐折值比對。 | AGENTS.md 紅線 4：同一張表的數字來自同一台、同一批。重算只讀快取，約十幾秒。 |
| D7 | 「每任務參數」＝ 由該任務 train 資料得到、需要保存的數值個數（head 參數、B 的兩欄、B_ar 的一欄）；兩類文字特徵計入儲存 bytes、不計入參數。 | 指令沒有定義這一欄。文字特徵由凍結的文字編碼器算出，不是從 train 資料得到的。兩個數都列，讀者可自行換算。 |
| D8 | 「每任務訓練秒數」只填本批量到的部分（讀 train 特徵 ＋ 取向量 ＋ closed-form）；head 訓練秒數是 MOE-1 批的紀錄，只寫在表註。 | 本批不訓練任何 head；把別批的秒數填進同一格會違反紅線 4。 |
| D9 | 各階段預估都在數分鐘內，以 `caffeinate -ims` 在前景執行，不開 tmux。 | 同 EXT-1 D22；AGENTS.md 的 tmux 規定針對長時間指令。既有的兩個 tmux session 不動。 |
| D10 | γ 重選的結果是 γ\_B = 1e-3（validation 目標 0.9423，五個候選中最大，不在邊界），與 FINAL-A 的定案值相同；因此沒有「兩個都報」的第二組數字，`FINALB_perfold.csv` 沒有 `FINALB_g*` 列。 | PREREG-22 細則 5。K10：目標值與 `moe3/hp.json` 的 `RDG:g0` 最大絕對差 0。 |
| D11 | 「拿掉 head」列（`ABLATION_full.csv` 的 `NOHEAD`）就是 FINAL-B 的定義：直接引用。本批仍由同一批快取把 FINAL-B 算一次（`ext2/a/FINALB/`），與 `NOHEAD` 逐折、逐階段、逐張判定差 0。 | 細則 4：(i) 程式逐條相符（REPORT「程式確認」節）；(ii) K11 通過——fold 1 train 2,273 張不載入 head、由特徵檔重算的 v 與快取最大絕對差 0.0，test 279 張的完整推論逐張判定相同。快取的 v0 當初是經 seed 42 head 的 `parts()` 取 s0，s0 不含 head 的任何參數，K11 證實。 |
| D12 | 推論計時：FINAL-B、FINAL-A(42)、FINAL-B 第二遍，依序各跑一遍 fold 1 的 279 張（各自讀檔）。主表用第一遍；第二遍並列。計時當下 load average 約 6（開始）→ 8（結束），不是本批的 job，照量照記。 | 第一遍含第一次讀檔；第二遍的檔案已在系統快取內。兩遍都報，讀者可看出順序的影響。同 EXT-1 D23。 |
| D13 | 每任務訓練秒數只在 fold 1 量（2,273 張 train），closed-form 取 reverse 序學到該任務那一步的累加與求解；讀檔與 mean_vec 每張只算一次。FINAL-A 的取向量用 seed 42 的 head。 | 細則 21。十折都量要重讀全部 train 特徵檔，量到的是同一種操作。 |
| D14 | bootstrap 只做 t = 4 的 CIL ACC（同 PREREG-21 細則 23）；其他指標只有逐折配對。外部對照沒有 bootstrap。 | 逐折輸出只存了 CIL 的逐張正確與否；外部方法的逐張預測不在本機（其 PROVENANCE 的「限制」）。 |
| D15 | D2、D3 跳過。can_dataset 四個任務的特徵檔都是單一 tensor [N, 512]（float32），沒有座標、沒有倍率欄位；can_dataset 與 `~/research/WSI_data` 下沒有 `.h5` 或含 coord 字樣的檔；openslide、tifffile 都沒有安裝。本機 10 張 .svs 都是 tcga_brca、都有對應的特徵檔，其中 2 張在 fold 1 的 test（記在 `ext2/d.json`）。 | 細則 24 的第一個條件（有逐 patch 座標）不成立。patch 的順序對應哪個座標無法由特徵檔得知，不猜。 |
| D16 | 總表的外部對照列：每任務參數、儲存、訓練秒數填「—」。 | 既有產物沒有這三項（FINAL_RESULTS 記為「待查論文」）；本批不讀其程式去算。 |
| D17 | 另做了 FINAL-B − {主系統, zero-shot, LIN8} 的逐折配對與 bootstrap（B1 附表）。 | 指令 A 要「同等完整」；FINAL-A 在 REPORT_ext1 的 D1 有這一組。 |
| D18 | A 的 γ 敏感度表（test，五個 γ）只作事後描述；γ\_B 由 validation 決定，不因這張表更動。 | 細則 6、17；同 EXT-1 D3 的處理。 |
| D19 | `.done` 標記不進版控；`ext2/a/` 的逐折 JSON、兩個 CSV、`a.json`、`b.json`、`cost.json`、`d.json`、`example.json` 進版控。 | 沿用 EXT-1 D29。 |
| D20 | （收尾，2026-10-03）REPORT 0-5 更正：原表的「y、B 的 shape、A」三列答的是 TP（AR）的 ridge，讀出的 ridge 只寫在表後一句話。現把三列標明「TP（AR）」，並新增 0-5b：讀出 ridge 的 B 是**每類一欄**（t = 4 時 [513, 8]），W [513, 8]，在 τ̂ 兩欄內 argmax（d = 第一類 − 第二類 ≥ 0 判第一類）；不是每任務一欄、閾值判類。貼出程式原文與行號；fold 1、reverse、t = 4 的 A、B、W 存成 `ext2/readout_fold1_reverse_t4.pt` 後讀回印 shape（`scripts/ext2_readout.py`、`ext2/readout.json`）。數字沒有任何更動。 | PI 要求說清楚。既有產物原本沒有存 B、W（每次由快取累加求解），所以「實際存檔」是本次才存的；`.pt` 依 `.gitignore` 不進版控。每任務一欄的是 TP（AR）的 B（[513, 4]），兩者在原版容易混淆。 |
| D21 | 【PI 裁決，2026-10-03】FINAL-B 的定義維持 PREREG-22（四輪各 16、λ = 1.5）；「一次 top-64」列為消融，不取代定義。總表加表註：四輪 − 一次 top-64 的差異不顯著（+0.0015，贏／輸／平手 7／2／1，p = 0.1797，bootstrap 95% CI [−0.0005, +0.0034] 跨 0）；外部對照的 Masked 欄為告訴任務定義。 | 定義在看 test 之前已登記；差異不顯著不構成改定義的理由，改了反而是看 test 之後才選。 |
