MOE-3 的判斷紀錄（2026-10-03）。每條：做了什麼判斷、為什麼。D1–D19 在正式執行前寫定並 commit；執行中新增的判斷接在後面並註明時間。

| # | 判斷 | 為什麼 |
|---|---|---|
| D1 | PREREG-19 = 指令原文（共同設定、操作定義、一致性檢查、要報的、門檻）全文 ＋ 同時登記的「操作化細則」。 | 沿用 PREREG-18 的寫法；指令要求「操作定義」與「門檻」全文加操作化細則，其餘段落一併登記不改變內容。 |
| D2 | u_K 的 s0 = `zscore(text_nav_feats(Z, f_task)[:, 0])`，直接呼叫這兩個函式，不載入任何 head 權重。 | 與 `I6Expert.parts` 的 s0 是同一段算式（MOE-2 的 v0 經 seed 42 head 的 `parts` 取 s0，結果與 head 參數無關）；指令「不用 head」。 |
| D3 | patch 數 ≤ K 的 slide 取全部 patch（`top_k_select` 的既有行為）；每個 K 下 patch 數 < K 的張數列在報告。 | 指令沒寫 patch 不足時怎麼辦；沿用既有函式（MOE-2 的「一次取 64」也是這個函式），不另訂規則。 |
| D4 | 五個 K 各自呼叫一次 `top_k_select(s0, K)`（同一個 s0）。 | 最貼近「依 s0 一次取前 K 個」；不從前 256 個截取，避免同分時的次序差異。 |
| D5 | TXT 的 cosine = 向量 · Fᵀ（float32、逐張相乘），d ≥ 0 判第一類；mean_vec 先 L2 正規化。 | 與 moe0／MOE-2 的 LT（g = 0）同一算法（K5 要逐折重現）；2 類 argmax 同分取第一類與 d ≥ 0 等價。正規化不改變兩類 cosine 差的正負。 |
| D6 | RDG、ANC 的 γI 是 513 × 513 的單位矩陣（含常數項那一維）；ANC 的 T 第 513 列 = 0。 | K2 要求 RDG(γ = 1e-3) 逐折重現 MOE-2 的 LR0，後者就是對 513 維全部加 γ；T 的定義照指令原文。 |
| D7 | 超參數在獨立的 `hp` 階段選一次、寫入 `hp.json`，之後的階段讀它；`hp` 只碰 validation。 | 指令「各自選一次」；選定在 test 之前完成並留檔。 |
| D8 | 目標值：每折先對兩序 × t = 1…4 的 8 個 validation 告訴任務 WP_t 取平均，再取十折平均；完全相等才算同分。 | 指令原文的目標加上「用十折 validation」；同分規則照原文（先較大的 γ，再較小的 α）。 |
| D9 | RDG × v(42) 的網格照算並列在 F6，但只作描述；S-LR 固定 γ = 1e-5。 | 指令 F6「每個判讀器 × 輸入的整張 validation 網格」；F1 把 S-LR 定為 RDG(γ = 1e-5)（取自 moe2 E7），兩者都照做。 |
| D10 | F5 的「全部 patch」（mean_vec）也沿用 u_64 選定的超參數，不重選。 | 指令「K ≠ 64 的向量沿用 K = 64 選定的超參數」；mean_vec 在 F5 是 K 軸上的一格，取同樣的做法。 |
| D11 | 邊界分 γ、α 兩個標示（各自是否在自己網格的最小／最大值）。 | 指令「選在邊界照報」；ANC 有兩個超參數，分開標示較清楚。 |
| D12 | K3 取 u_64、reverse、γ = 1e-2、所跑折數中的第一折；RDG 與 ANC 走各自的程式路徑。 | 指令「任取一個 γ、fold 1、t = 4」；冒煙測試與十折都是 fold 1。 |
| D13 | K5 除了十折平均對 B2，另加「每折每任務對 moe0 快取 `g0_cos8` 的 2 類 argmax 正確率」的比對（≤ 1e-9），冒煙測試也能檢查。 | B2 只存十折平均，單折無法比；加的是更細的同一項檢查，不放寬原判準。 |
| D14 | `chk` 失敗 → `hp`、`f2`、`f4`、`f5`、`f6` 不跑（`f7` 照跑）。 | AGENTS.md 紅線 2「失敗就停」；K2–K5 檢查的正是 RDG／ANC／TXT 的實作，檢查不過時不該產生這些判讀器的結果。MOE-2 的 K2、K3 放在 E1 內、只停 E1；本批的 K 檢查的是所有階段共用的判讀器，所以擋住後面。 |
| D15 | G-FINAL、G-COLD、G-ANCHOR 一律兩序各算、兩序都滿足才通過；「> 0 的折數」以差 > 1e-12 計。 | G-COLD、G-ANCHOR 原文「兩序各自」；G-FINAL 沒寫，取同樣（較嚴）的做法，與 MOE-1 D3、MOE-2 D7、D20 一致。 |
| D16 | 「取自並比對」（S-main、S-M3 對 `moe1/s2.json`；S-LR 對 `moe2/e7.json`；五個 seed 主系統對 `moe2/e1.json`）：表內用既有檔案的數值，本批重算只報最大絕對差，不設停止條件。 | 與 MOE-2 D14 相同；K1 已檢查主系統 seed 42。 |
| D17 | S-main 的告訴任務 WP_t = 真實任務 head 的 2 類正確率在已學任務上的等權平均（`s2.json` 的 `wp_task_t.main`），不是 Table 1 的 Masked ACC 欄。 | 指令 F2「告訴任務的 WP」；主系統的 Masked ACC 欄用的是 τ̂ 的 head，與 WP 定義不同（t = 4 的 0.9340 是 WP）。 |
| D18 | F4 用 reverse 序。 | S-M3 的 σ_b 固定值與序有關；與 MOE-2 E6（D13）同序。 |
| D19 | 啟動方式：`tmux new-session -d -s moe3 "env NAVCIL_MACHINE=mac MOE3_PY=<venv python> caffeinate -dimsu scripts/moe3_run_all.sh"`，不經過 `scripts/run_stage.sh`；HEARTBEAT 每 15 分鐘一行，另每階段開始與結束各一行。 | 指令指定 session 名稱「moe3」與 `caffeinate -dimsu`；`run_stage.sh` 會命名為 `navcil-<name>` 並用 `-ims`（同 MOE-2 D17、D20）。 |
| D20 | 冒煙測試（2026-10-03 12:10–12:11，`moe3_smoke/`，fold 1，經 `moe3_run_all.sh`、心跳 60 秒）：vec、chk、hp、f2、f4、f5、f6、f7 全部 rc = 0；K1–K5 通過（K5 的 B2 比對在單折不適用，逐折對 `g0_cos8` 的差為 0）；S-main、S-M3、S-LR 與五個 seed 主系統的重算差都是 0；報告各節無格式化失敗，才 commit 並啟動十折。全部階段約 70 秒（vec 約 30 秒／折）。 | 指令：冒煙測試通過後 commit、push，再啟動。冒煙測試的數字不寫進報告、`moe3_smoke/` 不 commit。 |
| D21 | 沒有另外的開發執行；冒煙測試是程式的第一次執行。冒煙測試後讀過的內容：各階段 log（K1–K5 的 fold 1 數值、fold 1 的 validation 超參數選擇）、報告的 K 表與 vec 一節、各節標題（確認沒有格式化失敗）、`f2.json` 的重算差與兩序比對；fold 1 的門檻表與 F2–F6 的 test 數字沒有讀。 | 揭露：門檻數字在正式執行前沒有以十折跑過；fold 1 的 validation 選擇（單折）不用於十折。 |
