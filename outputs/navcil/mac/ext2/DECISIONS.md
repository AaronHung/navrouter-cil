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
