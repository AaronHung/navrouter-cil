# MOE-4 判斷紀錄（2026-10-03）

本檔在十折正式執行前寫定並 commit。每條：做了什麼判斷、為什麼。執行中新增的判斷接在後面並註明時間。

| # | 判斷 | 為什麼 |
|---|---|---|
| D1 | PREREG-20 ＝ 指令原文（操作定義、一致性檢查、要報的、門檻）全文 ＋ 操作化細則（第二部分）。 | 沿用 PREREG-19 的寫法；指令要求「操作定義」「一致性檢查」「門檻」全文，「要報的」一併登記，不改變內容。 |
| D2 | 分支 `moe-4` 自 `moe-3`（fb2a6d8）開；PREREG-20 commit 3a7e2a9，程式 commit ab3d221；commit 前跑 pytest（139 passed）。 | 指令；AGENTS.md 紅線 1 與「任何 commit 前都要跑 pytest」。 |
| D3 | w(42) ＝ MOE-2 vec 快取的 `one42`（seed 42 head 的 `one_shot(s0 + g)`），不重算。 | 指令「seed 42 沿用 MOE-2 的 v1(42) 快取」；`one42` 即一次取 64 的 seed 42 向量。K5 以 moe1 s42cells「一次取 64／等權」格檢查。 |
| D4 | w(43…46) ＝ `top_k_select(head 分數 s0 + g, 64)` 後 `mean_norm`（原始 Z、等權、L2）；patch 數 ≤ 64 取全部。 | 指令操作定義「一次取前 64 個 patch（不扣冗餘）」；patch 不足的處理沿用 MOE-3 D3（`top_k_select` 既有行為）。 |
| D5 | seed 42 的 head 路徑為 `outputs/navcil/mac/i6/r2/`；seed 43–46 為 `moe1/i6_seed{s}/`。 | 指令「seed 42：i6/r2/；seed 43–46：moe1/i6_seed{s}/」；同 moe2_common.head_path。 |
| D6 | u_64 直接讀 MOE-3 的 `moe3/cache/u_*.pt`（唯讀），不重算。 | 指令「u_64 ＝ 不用 head 的一次取 64（沿用 MOE-3 快取）」。 |
| D7 | 「差 > 0 的折數」以差 > 1e-12 計。 | 同 PREREG-19 細則 7、16–18；浮點雜訊不計為勝。 |
| D8 | G-ONE 未指定序；兩序各算、兩序都滿足才通過。 | 與 G-CONF、G-TRAJ 的「全部滿足」一致（較嚴）。在 t = 4，兩序的 A、B 與告訴任務判定精確相同，預期兩序數字一致。 |
| D9 | G-CONF、G-TRAJ 的 8 格（seed 43–46 × 兩序）全部滿足才通過；seed 42 的同一組數字只作描述。P-4 的同一組數字只報告。 | 指令「seed 43、44、45、46 各自、兩序各自」；「seed 42 的數字只作描述」；「也對 P-4 算一次，只報告」。 |
| D10 | H3 用 reverse 序。 | 告訴任務的判定在 t = 4 只取決於各任務的 w(s) 與 head，與序無關（同 MOE-3 F4 取 reverse）。 |
| D11 | H3 的 P-0：seed 無關，只列十折合計一份（不乘 5）。 | P-0 不含 seed；五個 seed 合計會把同一組判定重複計 5 次。 |
| D12 | H5 由實際張量形狀與 head 參數數計算；P-F 的共用 ＝ A_w、A_mv，每任務 ＝ head、兩類文字、B_w、B_ar；另列 P-4、P-main、P-T1、P-0 作對照。 | 指令 H5「P-F 的共用與每任務 bytes，格式同 moe3 F7」；文字特徵在 head 的 s0 中要用，故計入。推論時不另存 W（W 由 A、B 解出）。 |
| D13 | `chk` 不過 → `f2`、`f3`、`f4`、`f5` 都不跑；`f2`–`f5` 彼此獨立，單一階段失敗不擋其他階段、不重試。 | 指令「K 不過時擋住後面階段」；失敗就停、不重試（AGENTS.md 紅線 2）。 |
| D14 | 各階段開始時讀取 `chk.json` 的 pass；不過就拒絕執行。 | 防止手動單獨重跑 f2–f5 時跳過 K。 |
| D15 | 階段外殼與判讀器沿用 `moe3_common`（`run_cl`、`ridge_d`、`txt_d`、`ar_stages`、`order_same`、`row_diff`）；stage_main 另寫為 MOE-4 版（只改 log 標籤）。 | 指令「不改既有程式」，但沿用共用函式；log 標籤避免與 MOE-3 混淆。 |
| D16 | H1 的「五個 seed 平均」：每折先對 seed 42–46 平均（ACC、WP、每任務 WP、Forgetting、BWT、Ā 各自），再算十折 mean ± sd。 | 指令「先對 seed 平均再算十折 mean ± sd」。 |
| D17 | 冒煙測試（2026-10-03 15:42–15:49，`moe4_smoke/`，fold 1）：先單獨跑 vec、chk、f2–f5，再以 `moe4_run_all.sh`（心跳 60 秒）從空目錄完整跑一次，全部 rc = 0；K1–K5 通過。冒煙數字未進報告。 | 指令「冒煙測試通過後 commit 並 push」。冒煙測試的 fold 1 數字已在終端機顯示過（例如 H2 的兩序判定逐張相同、fold 1 的 P-F、P-main 數字）；之後的改動只涉及下列程式問題，沒有依據數字調整任何判準、門檻或系統。 |
| D18 | 冒煙測試中發現並修正的程式問題（正式執行前完成）：（a）f5 第一次失敗：`head_path` 把 seed 42 指到 `moe1/i6_seed42/`，已改為 `i6/r2/`，之後 f5 通過（vec、chk 不受影響，因為 vec 只用 seed 43–46 的路徑）；（b）f2–f4 的心跳分母少算一次（`run.total` 改為 folds + 1）；（c）報告中 seed 42 的 P-4 門檻數字原本未算，改為與 seed 43–46 一樣算出、seed 42 只標示「描述」。 | 揭露：（a）是錯誤路徑、（b）（c）是顯示與計數。都不改變統計量、判準或門檻。 |
| D19 | 啟動：`tmux new-session -d -s moe4 "env NAVCIL_MACHINE=mac MOE4_PY=/opt/miniconda3/bin/python caffeinate -dimsu scripts/moe4_run_all.sh"`；`HEARTBEAT.log` 每 900 秒一行，每階段開始與結束各一行。 | 指令指定 session 名稱「moe4」與 `caffeinate -dimsu`；不經 `scripts/run_stage.sh`（同 MOE-3 D19）。 |
