MOE-2 的判斷紀錄（2026-10-03）。每條：做了什麼判斷、為什麼。D1–D22 在正式執行前寫定並 commit；執行中新增的判斷接在後面並註明時間。

| # | 判斷 | 為什麼 |
|---|---|---|
| D1 | PREREG-18 = 指令原文（共同設定、一致性檢查、操作定義、門檻）全文 ＋ 同時登記的「操作化細則」。 | 沿用 PREREG-17 的寫法；指令要求「操作定義」與「門檻」全文加操作化細則，共同設定與 K 一併登記不改變內容。 |
| D2 | seed 42 的 v 沿用 MOE-1 快取（train：`s42v_train`；validation／test：`s42cells` 的 `v_four`）；seed 43–46 的 v、v0、v1(42) 在 `vec` 階段一次讀檔一起算（每張 slide 讀一次）。 | 指令「沿用 MOE-1 的程式、快取與 head 權重」；seed 43–46 的 MOE-1 快取只存了 8 類 cosine，沒有存向量，必須由特徵檔重算。一次讀檔省時間。 |
| D3 | v0 的 s0 取 seed 42 head 的 `parts(Z, f_task)[0]`。validation／test 的 v0、v1 四個任務都算；CIL 時 v0 取 τ̂ 的 s0。 | s0 與 head 參數無關（slide 內 z-score 的文字 cosine），與 MOE-0 的 g0 同一個算法；「CIL 時 v 取自 τ̂ 的 head」對 v0 的對應就是 τ̂ 的文字。 |
| D4 | 細則 4 的快取檢查中，v·Fᵀ 一律逐張計算（每個 [512] 向量各自乘 Fᵀ），不用批次矩陣乘法。 | 開發時（`_moe2_dev`，fold 1）用批次矩陣乘法算，檢查 (b) 的最大差為 1.07e-6（略超過 1e-6），(a) 的向量本身逐位元相同（差 0）；差異來自 float32 批次乘法的累加順序。原快取是逐張 `mean_norm(...) @ F.t()`，逐張計算即重現原算法，改後 (a)–(d) 的差都是 0。判準（1e-6）沒有改。 |
| D5 | K2 照 MOE-1 S5 的算法：W 依 reverse 序累加、兩序的 CIL 都用這個 W（各用自己序的 AR）。E1 的兩序結果則各用自己序的累加（細則 7）。 | K2 是對 `s5.json` 的重現檢查，必須用 s5 的定義；E1 要回答「兩序是否逐張相同」，所以兩序各自累加。 |
| D6 | γ 一律以 reverse 序、t = 4 的 validation 告訴任務 WP 選；平均值完全相等才算同分。 | 與 MOE-1 S5（D13）相同；t = 4 時兩序只差浮點累加順序。 |
| D7 | G-LR、G-HEAD、G-CAT 一律兩序各算、兩序都滿足才通過；兩序結果相同時兩者等價。 | G-LR 原文「若兩序結果不同，兩序各算、都要滿足」；G-HEAD、G-CAT 沒寫，取同樣（較嚴）的做法，與 MOE-1 的 D3 一致。 |
| D8 | D_head 每序用該序 t = 4 的 test WP；五個 seed 的 LR 共用 LR 的 γ\*。 | 原文「五個 seed 的 WP(LR(s)) 平均」；γ 依原文「選定後所有 seed 共用」。 |
| D9 | LT 在 g = 0 的 WP 與 MOE-0 B2 的比對只報最大絕對差，不是 K、不停。 | 原文「重算並比對」，不在 K1–K3；cosine 層級的同一性已由細則 4(d)（不過就停）保證。 |
| D10 | 「LR 及以下各變體各自選一次」= LR、LR0、LR1、LRG、LR-bal、GR-bal 各自選 γ；M3-bal 不另選（GR-bal 的 γ\*、β = 1）。LR 的 γ\* 供 E1–E4、E6、E7 的 LR 共用。 | 原文。M3-bal 依原文是「LT 與 GR-bal 的 σ 固定版相加（β = 1）」，沒有自己的超參數。 |
| D11 | E4 的分數層級相加：GR = LIN8（γ = 0.01、reverse、t = 4，同 MOE-1 S5 的 GR）；validation 與 test 都報；項的順序照名稱，f = 0 看第一項。 | 原文「t = 4、只作描述」；與 MOE-1 S5 的融合同一算法，方便對照。 |
| D12 | E5 的 n_c = 該折 train split 中類別 c 的張數（每類權重和 = 1）；CIL 的 TP 仍為不加權的 AR；另報 LR-bal − LR、M3-bal − M3 的逐折差（描述）。 | 原文「n_c = 該折該類別的 train slide 數」「CIL 的 TP 一律 AR（γ = 1e-3）」；差值方便讀，不設門檻。 |
| D13 | E6 用 reverse 序（M3、M3-bal 的 σ_b 固定值與序有關；ridge 的 W 用 reverse 累加）；balanced accuracy 由十折合計的兩類正確率平均。 | 原文沒指定序；與 γ 選擇、MOE-1 S5 同序。原文「十折合計」。 |
| D14 | E7 的主系統與 M3 列取自 `moe1/s2.json`；本批照 `moe1_s2.py` 重算並報最大絕對差，不設停止條件。 | 原文「取自 moe1/s2.json 並比對」；數值本身來自 s2.json，比對只是確認同一算法。 |
| D15 | E8 只算推論與繼續學習必須保存的量：ridge 存 A、B（不存 W）、head 參數、每任務文字特徵、σ；CONCH、patch 特徵、λ\*、logit scale 列出不計。另列 LRG 省去重複（A_mv 是 A_g 的子區塊）的數字。 | 封閉解的繼續學習需要的是統計量；W 隨時可解。省去重複的數字讓 LRG 的增量不被高估。 |
| D16 | LRG 的 1025 × 1025 封閉解用 float64。 | 本批指令「closed-form 一律 float64」；超出 AGENTS.md 可攜規則 3 例外的 513 × 513，在報告揭露（同 MOE-1 的 D17）。 |
| D17 | 啟動方式：`tmux new-session -d -s moe2 "env NAVCIL_MACHINE=mac MOE2_PY=<venv python> caffeinate -dimsu scripts/moe2_run_all.sh"`，不經過 `scripts/run_stage.sh`。 | 指令指定 session 名稱「moe2」與 `caffeinate -dimsu`；`run_stage.sh` 會命名為 `navcil-<name>` 並用 `-ims`（同 MOE-1 的 D18）。心跳由 `moe2_run_all.sh` 寫到 `HEARTBEAT.log`。 |
| D18 | 失敗處理：階段失敗 → 寫 `FAILED_<階段>.txt`、不重試、不改期望值；`vec` 失敗則 E1–E7 不跑（寫明原因），E8 照跑。各 E 階段各自重算所需的 γ 選擇（決定性），彼此不依賴。 | 本批指令「失敗的階段不重試」；E8 只用張量形狀，不需要向量快取。 |
| D19 | K1 在 E1–E7 每個階段開始時都跑。 | 同 MOE-1 的 D9；每個階段都有 seed 42 的主系統或以它為基準的比較，重算很便宜。 |
| D20 | 「差 > 0 的折數」以差 > 1e-12 計；HEARTBEAT 除了每 15 分鐘一行，每階段開始與結束也各寫一行。 | 同 MOE-1 的 D21、D22。 |
| D21 | 各 seed 的 M3 逐折 CIL ACC 另與 s4.json／s7.json 的 M3 比對，只報最大絕對差，不是 K。 | 指令的 K3 只要求主系統；M3 的比對是額外的確認。 |
| D22 | 開發時在暫存目錄（`_moe2_dev`，用完刪除）跑過 fold 1 的 `vec` 與 E1–E8（除錯；`vec` 的第一次執行因 D4 的批次乘法停在檢查 (b)）。開發過程看到的是 fold 1 的 validation γ 選擇記錄、K 表（fold 1 的 seed 42 主系統與 seed 42 LR 的 test 數字，MOE-1 已有）與 E8（儲存）一節；其餘 test 數字沒有讀，dev 報告只檢查有無格式化失敗。 | 揭露：門檻數字在正式執行前沒有以十折跑過。 |
| D23 | 冒煙測試（2026-10-03 09:52–09:57，`moe2_smoke/`，fold 1，經 `moe2_run_all.sh`、心跳 60 秒）：vec、E1–E8 全部 rc = 0，細則 4 的檢查 (a)–(d) 差都是 0、K1、K2、K3 通過、報告各節無格式化失敗，才 commit 並啟動十折。`vec` 每折約 4 分鐘（train 約 2 分鐘）。 | 指令：冒煙測試通過後 commit、push，再啟動。冒煙測試的數字不寫進報告、`moe2_smoke/` 不 commit。 |
