MOE-1 的判斷紀錄（2026-10-02；過夜無人看管，沒有提問）。每條：做了什麼判斷、為什麼。D1–D24 在正式執行前寫定並 commit；執行中新增的判斷接在後面並註明時間。

| # | 判斷 | 為什麼 |
|---|---|---|
| D1 | PREREG-17 = 指令原文（共同設定、一致性檢查、操作定義、門檻）全文 ＋ 同時登記的「操作化細則」。 | 沿用 PREREG-16 的寫法（原文 ＋ 操作定義）；細則把會影響數字的解讀在執行前固定下來。 |
| D2 | Masked ACC：主系統 = Table 1 的 Masked ACC 欄（τ̂ 的 head 的證據在真實任務兩類內 argmax，0.9312）；融合系統（M3、σ 重算版、G1、G2、ensemble）= 告訴任務的融合判定。融合系統的四任務 WP 與 Masked ACC 因此是同一個數。 | K2 指定 M3 的「Masked ACC」要等於 moe0 的 z′、λ = 1.0 那一欄（0.9448），那一欄就是告訴任務的融合判定（PREREG-16 操作定義 16）。主系統沿用 Table 1 的定義，K1 的 WP 0.9340 另列。 |
| D3 | G-ENS 兩序各算、兩序都滿足才算通過。 | 門檻原文沒有寫順序；LG 的 σ_b 固定值與順序有關，所以兩序的數字不同。G-M3c、G-GATE 都是兩序都要，取同樣（較嚴）的做法。 |
| D4 | G-GATE 的代表每序各自決定（訓練資料正確率 = 每折四任務等權、再十折平均；同分取 G1）。 | gate 的訓練資料（leave-one-out 的 d_b、σ_b 固定值）與順序有關，兩序是兩次獨立的 CL 流程。 |
| D5 | G1 同分時取 \|β − 1\| 最小者；仍同分（例如 0 與 2）取較小的 β。 | 原文「同分取最接近 1 者」在 0 與 2 之間沒有定義；取較小 = 較接近主系統（β = 0 就是主系統）。 |
| D6 | G2 用 `torch.optim.LBFGS(lr=1, max_iter=200, line_search_fn="strong_wolfe")`、float64、呼叫一次 `step`；`max_eval` 用預設值（250 次函數評估）。每次 closure 檢查 loss 與梯度有限，結束時檢查參數有限。 | 「L-BFGS，最多 200 步」= 最多 200 次迭代；strong-Wolfe 是標準的線搜尋。3 個參數的問題遠在 250 次評估內收斂；實際評估次數的最大值與中位數印在報告 T6.3 下方。 |
| D7 | K3 除了 S4，在 S6、S7（seed 45／46 的部分）訓練前也各跑一次；不過就停下該階段。 | 三個階段用同一個訓練函式（`moe1_common.train_head`）；K3 不過代表訓練不可重現，後面的訓練都不可信。每次約 5 秒。 |
| D8 | K2 的逐折期望值 = `nc8/per_fold.json` 第 6 列的逐折值 ＋ `moe0/results.json` 的逐折差（`B4.zprime.paired`）。 | `results.json` 只存了十折平均與「融合 − 主系統」的逐折差，沒有存逐折絕對值。 |
| D9 | K1 在每個報告 seed 42 主系統的階段都先跑（S1–S7）；S3 另外用由特徵檔重算的現行格再跑一次。 | K1 是所有後續比較的基準；重算很便宜（只讀快取）。 |
| D10 | S1 的 patch 數 N 取自既有快取 `cache/fold{f}_test_{task}.pt` 的 `n_patch`（逐張檢查 slide id 對齊），不讀特徵檔。 | 指令寫「不重跑」；該快取十折都與 moe0 的 test 快取逐張對齊（執行前已確認）。 |
| D11 | S2 的 γ 統一 (i) 在兩序、每個階段 t = 1…4 都驗證，只報最大絕對差，≥ 1e-8 也不停（不是 K）。「LIN8 兩類分數和當 TP」同時報主系統與 M3 的 CIL ACC。 | 原文寫「預期 < 1e-8」「不設門檻」；CIL ACC 沒有指定是哪個系統，兩個都報。 |
| D12 | S4：GG、LLL 沒有單一的 seed s，增益對三個 seed 的主系統各報一次。ensemble 的 CIL：TP = AR，τ̂ 的各 head／欄位／σ。 | 原文「每個都報相對 seed s 的主系統的增益」。 |
| D13 | S5：GR 與各 σ 用 t = 4、reverse 序的 LIN8（與 MOE-0 的 B3 相同）；LR 的 A、B 依 reverse 序累加；兩兩組合的張數 validation 與 test 都報；LR 的 γ 選定之後才算 test。 | S5 是告訴任務、t = 4 的分析，兩序在 t = 4 只差浮點累加順序。 |
| D14 | S3 與 S5 共用一份 seed 42 的推論快取（`cache/s42cells_{val,test}_fold{f}.pt`：2×2 四格的 cosine 與四輪等權向量），誰先跑誰建。 | 兩個階段都要對 validation／test 的每張 slide 跑四個 head 的四輪選片；共用可省約 4 分鐘，且兩個階段仍可各自獨立執行。 |
| D15 | S6 的參考上限（LT＋GR 至少一個對）在 S6 內由同一批資料直接算。 | S6 不應依賴 S5 是否完成（指令：S6 只依賴 S2 的程式）。數值與 S5 的 T5.3「LT+GR… 」相同（順序為 GR+LT）。 |
| D16 | S7 先跑隨機特徵（幾分鐘）再跑 seed 45／46 的訓練；隨機特徵的結果先寫到 `s7_rf.json`。seed 45／46 的部分只在 S4 成功（`s4.done`）時跑。 | 指令說 S7「可以沒跑完」；先跑短的，沒跑完時報告仍有隨機特徵的結果。seed 45／46 是「重做 S4 的兩列」，S4 失敗時沒有可比的基準。 |
| D17 | S7 的 2049 × 2049 封閉解用 float64。 | 本批指令「closed-form 一律 float64」；超出 AGENTS.md 可攜規則 3 例外的 513 × 513 範圍，在此揭露。隨機特徵的 γ 以 reverse 序、t = 4 的 validation WP 選；TP 的 CIL ACC 同時報主系統與取代後的 M3。 |
| D18 | 啟動方式：`tmux new-session -d -s moe1 "env NAVCIL_MACHINE=mac MOE1_PY=<venv python> caffeinate -dimsu scripts/moe1_run_all.sh"`，不經過 `scripts/run_stage.sh`。python 路徑用環境變數 `MOE1_PY` 傳入。 | 指令指定 session 名稱「moe1」與 `caffeinate -dimsu`；`run_stage.sh` 會把 session 命名為 `navcil-<name>` 並用 `-ims`（NC-15 的 `nc15_run.sh` 是同類的先例）。心跳由 `moe1_run_all.sh` 自己寫到 `HEARTBEAT.log`。可攜規則 2：程式裡不寫絕對路徑。 |
| D19 | 失敗處理：階段失敗 → 該階段停止、寫 `FAILED_<階段>.txt`、不重試、不改期望值，接著跑不依賴它的階段。 | 本批指令（無人看管）；與 AGENTS.md 紅線 2「失敗就停」的差別只在後面互相獨立的階段會繼續。 |
| D20 | 冒煙測試用同一支 `moe1_run_all.sh`（`MOE1_OUT=moe1_smoke MOE1_FOLDS=1 MOE1_EPOCHS=1`），報告寫到 `moe1_smoke/REPORT_smoke.md`，不碰 `REPORT_moe1.md`。 | 指令：冒煙測試的數字不寫進報告；同時測到 run_all 的心跳、失敗處理與報告程式。 |
| D21 | 「差 > 0 的折數」以差 > 1e-12 計。 | 與 `moe0_report.paired` 相同；避免把浮點尾數當成贏。 |
| D22 | `HEARTBEAT.log` 除了每 15 分鐘一行，每個階段開始與結束也各寫一行（同樣的欄位）。 | 方便事後對時間；不影響 15 分鐘的心跳。 |
| D23 | 開發時在暫存目錄（`_moe1_dev`，已刪除）跑過：S1、S2 的十折（只讀 seed 42 快取，用來確認 K1、K2 逐折差為 0），以及 S3–S7 的 fold 1、1 個 epoch（除錯）。 | 揭露：S1、S2 是決定性的只讀計算，正式執行會得到相同的數字，且 seed 42 的結果本來就只作描述。S4、S6 的門檻數字在正式執行前沒有以十折或 5 個 epoch 跑過。 |
| D24 | 冒煙測試（2026-10-02 21:28–21:34，`moe1_smoke/`，fold 1、1 個 epoch）：S1–S7 全部跑完，K1、K2、K3 通過，才 commit 並啟動十折。另測過失敗路徑（以不存在的 fold 99 執行，暫存目錄已刪）：每階段寫出 `FAILED_<階段>.txt`、S2 失敗時 S6 不跑、報告列出失敗摘要。正式執行前另以十折檢查 leave-one-out 的 d_b 全部有限（h 最大 0.8982）、自行累加的 LIN8 權重與 `B8.W` 相同、分層切 3 份可行（保留份最少 39 張）。 | 指令：冒煙測試全部跑得完且一致性檢查通過才進下一步。十折的檢查只看數值是否有限與切分是否可行，沒有看任何正確率。 |
