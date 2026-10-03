EXT-1 的判斷紀錄（2026-10-03）。每條：做了什麼判斷、為什麼。D1–D6 在 PREREG-21 commit（277339b）時已寫進其第二部分；D7 起為執行中新增。

| # | 判斷 | 為什麼 |
|---|---|---|
| D1 | PREREG-21 ＝ 指令原文摘要 ＋ 操作化細則；符號檢定用 `scipy.stats.binomtest(贏, 贏 + 輸, 0.5)`，平手折（\|差\| ≤ 1e-12）不計入 n。 | 指令寫「exact binomial，雙尾」，沒有寫平手怎麼算；不計平手是符號檢定的標準做法，且與既有批次「差 > 1e-12 才算贏」一致。 |
| D2 | 指令的「M1、M2」在 repo 內沒有定義。M1 取 MOE-0 B5（任務間 soft gating），M2 取 MOE-0 B2（跨器官共用 head）。**PI 於 2026-10-03 裁決：M1 = MOE-0 B5、M2 = MOE-0 B2。** | repo 全文只有 G-M2、G-M3 兩個相關名稱；M3 是融合，依序推 M1、M2 為 MOE-0 的另外兩個設計。推定已由 PI 確認。 |
| D3 | K8 以十折平均判定（8 格都滿足才通過）。 | ESCA 每折 test 約 15 張，單折差 0.02 不到一張。B 停在 B1，K8 沒有執行。 |
| D4 | bootstrap 在（折、任務）層內重抽，兩個系統用同一組索引。 | 主表的統計量是「每折四任務等權、再十折平均」；分層重抽保留這個結構。 |
| D5 | D3 只改 ridge readout 的 γ，AR 的 γ 維持 1e-3。 | FINAL 的兩個 γ 是分開定案的；指令說的是 readout 的敏感度。 |
| D6 | B3 的 12 小時判定用指令原式；等比例估計只作揭露。 | 指令給了公式。B 停在 B1，沒有用到。 |
| D7 | A1：十折輸出不在 navipath（只有 3 折、自寫 runner），而在 `~/research/02_pathselect/outputs/exp2/sota/repro_qpmil/`。改從那裡匯入，並在 PROVENANCE 寫明。 | 指令要的是「我們之前用公開程式跑的十折輸出」；navipath 的 3 折不是十折，也不是上游的 `main.py`。02_pathselect 那三批是十折、上游程式、每折 log 有 slide ID。搜尋由子代理做，我另外逐項核對（summary.json、metrics、log 的 ID 行、commit）。 |
| D8 | reverse 有兩批（b8、b16）。主檔取 reverse_b8，reverse_b16 另存、不進配對表。 | b8 是論文的 reverse 設定；b16 的 K6 不過（fold 7 的 log 缺 ID）。選擇時沒有比較我方 test 數字。b8 的 ACC 較高（0.8681 對 0.8141），取它對外部方法較有利。 |
| D9 | K6 直接比 slide ID（log 內有 `sids_*` 清單），不用間接證據。 | 細則 7 的「間接」只在沒有 slide 清單時才用。 |
| D10 | 外部方法的 MaskedACC 取 `acc@mid`，不取 log 的 `acc`。**2026-10-03 收尾時重新核對並更正說明**：上游評估器有二元與多類兩個。二元的 `acc` 用 test 資料自己的 ROC 曲線挑門檻（含 test 資訊），`acc@mid` 用固定 0.5；多類評估器沒有 0.5 門檻，其 `acc` 即 argmax。Masked 評估把 logits 切成該任務的兩類，所以走二元的 `acc@mid`（= 兩類 argmax）；逐階段 R 在 t = 1 走二元 `acc@mid`、t ≥ 2 走多類 `acc`。兩者都是不含 test 資訊的 argmax 正確率，因此不換欄、不重新匯入；兩欄的十折 mean 都列在 qpmil_vl/PROVENANCE.md。原文與行號、獨立驗證也在該檔。 | 先前 D10 只寫了「`acc@mid` 等於兩類 argmax」，沒有交代多類評估器沒有 0.5 門檻；PI 要求重新說明。讀的是 `utils/evaluator_clf.py`、`manager/manager.py`（只讀，沒有複製或 import）。獨立驗證：用 Mac CPU 未完成執行留下的逐 slide 機率（76 張）重算，`p > 0.5` 的正確率對上 log 的 `acc@mid`，test 最佳門檻的正確率對上 `acc`。 |
| D11 | 外部方法的 Forgetting、BWT 由 R 以我方 Table 1 公式重算。 | 同一張配對表要用同一個公式。重算值與來源 `summary.json` 逐折相同。 |
| D12 | 匯入腳本放 `~/research/ext/ext1_import/`；navrouter-cil 的 `scripts/` 不出現外部方法的名字，外部目錄名在執行時由 `outputs/external/` 列出。 | `tests/test_no_banned_deps.py` 會掃 `scripts/`；`outputs/` 不掃。 |
| D13 | B 停在 B1。在等 A1 搜尋結果時先做了 B1 的 clone 與閱讀（不佔運算）。 | 補充 4 的停止條件成立（見 PORT.md）。「各階段依序、不並行」是為了 B3 計時不受干擾；讀程式不影響。B3 沒有執行。 |
| D14 | C 的兩種 Masked ACC 用同一個介面算：`first(t, pv, pc)`＝證據取自任務 pv 的版本、在任務 pc 兩類內判。Table 1 定義 pv = τ̂、pc = 真實任務；oracle 定義 pv = pc = 真實任務。 | 這就是主系統 Table 1 那一欄（`nc2_report.hard` 的 masked）的定義，推廣到 ridge 判讀器：向量取 τ̂ 的 head，ridge 分數取真實任務兩欄。K7 對主系統的 Masked 欄逐折差為 0。 |
| D15 | 【**作廢**，PI 2026-10-03 裁決】原：融合系統（M3、G1、G2、RF）的 Table 1 Masked ACC 延伸定義。**現行：融合系統的 Masked ACC 不進任何表；`ABLATION_full.csv` 的 `MaskedACC_table1` 與 `MaskedACC_oracle` 兩欄對這四列一律留空（報告表顯示「—」），逐折 JSON 的 `mk1_t` 為 null、不保存 Rm。WP（告訴任務）與 CIL ACC 照填，定義不變。** | 該欄是本批新訂的延伸定義，沒有既有定義可依；PI 裁決不採用。已重算那四列並重跑 K7（仍通過）。 |
| D16 | M1（soft gating）的兩種 Masked 都等於主系統的告訴任務判定。 | 分數是 π[任務] × q[類別]；限制在真實任務兩類內時 π 相消，只剩真實任務 head 的 q。 |
| D17 | M2 只有 t = 4 的告訴任務 WP（四個 head 各一列，另加 g = 0 一列）；CIL、Masked（Table 1）、t < 4、Forgetting、BWT 留空。 | 既有快取 `cross_cos8` 只有「真實任務的文字 × 換 head」；CIL 需要 τ̂ 的文字 × 換 head，要重讀特徵檔新算，超出「從快取或重跑既有定義」。t < 4 時「某個 head 用於全部任務」也沒有既有定義（細則 19）。 |
| D18 | G1、G2 的參數（β、c）讀 `moe1/s6.json`，不重學；延伸到 t < 4 時 gate 參數與 σ 固定、LIN8 用階段 t 的權重。M1、RF 同樣延伸到 t < 4。 | 細則 18、19：參數既有、定義可直接延伸（與 M3 在 MOE-1 S2 的 CL 流程同一算法）。t = 4 的值與 s6.json、s7_rf.json 逐折差為 0。t < 4 與 Forgetting、BWT 是這些列第一次有數字，沒有既有值可比。 |
| D19 | CONCAT 取 MOE-2 的 LRG(42)、γ\* = 1e-3（`moe2/e4.json`）；「拿掉 head」取 v0／RDG 1e-3（S-R0f）；K 掃描取 u_K／RDG 1e-3；ANC 三個輸入都列。 | 對應 FINAL_RESULTS B 表各列的來源。 |
| D20 | 儲存 bytes 只填既有報告有公式的列；M1、M2、G1、G2、RF 留空。ANC × v(42) 沿用 S-LR 的項目（要存 head）。 | 細則 20。 |
| D21 | 五 seed 平均列（FINAL、MAIN、ONE64）附在 CSV 後段，另保留每個 seed 的列。 | 細則 22；配對與 K7 都要用到逐 seed 的逐折值。 |
| D22 | C、D 各只需十幾秒，直接以 `caffeinate -ims` 在前景跑，沒有開 tmux。 | AGENTS.md 的 tmux 規定是針對長時間指令；本批最長的一步約 15 秒。既有的兩個 tmux session 沒有動。 |
| D23 | D2 計時當下系統 load average 約 5–6（macOS 的 `mediaanalysisd` 佔一個核心），沒有我方的其他 job。照量、照記。讀檔秒數是檔案已在系統快取內的數字。 | 不是我方的 job，不去停它。load 記在 `d2.json`。 |
| D24 | 訓練成本的 head 秒數只有 seed 43–46。 | `i6/r2/*_train.json`（seed 42）沒有 `wall_s` 欄（細則 24 已預留這個情況）。 |
| D25 | Forgetting 的配對表另列「FINAL 較好的折數」（Forgetting 越低越好，等於差 < 0 的折數）。 | 細則 4 的「贏」是差 > 0；對越低越好的指標直接套用會讀反。p 值不受影響。 |
| D26 | 【**已裁決**，PI 2026-10-03】外部對照不受 AGENTS.md 紅線 4 約束；每個外部對照表加表註，寫明外部數字的機器與日期（RunPod 機型、執行日）與我方的機器與日期。外部數字與我方數字仍分開列（逐折配對表是例外：兩邊用同一折、同一序逐折相減）。 | 外部數字在 RunPod GPU、我方在 Mac CPU，不可能同機；表註讓讀者知道。 |

| D27 | （2026-10-03 收尾）M1 的兩種 Masked 定義在構造上相同（D16），不是融合系統，保留；M2 只有 t = 4 的告訴任務 WP（D17）維持。 | PI 的裁決只針對融合系統。 |
| D28 | （2026-10-03 收尾）MergeSlide：停止條件成立、不跑，理由與做法 1–5 的需求及偏離寫在 `outputs/external/mergeslide/DECISION.md`。TITAN 權限只做了一次唯讀 HEAD 請求（`config.json`，用本機既有 HF token），沒有下載任何權重。 | PI 要求列出做法 1 需要的每一項；權限是否在手需要查一次才能寫。 |
| D29 | （2026-10-03 收尾）`ext1/c/` 的逐折 JSON 與 `k6_navrouter_ids.json` 進版控；`.done` 標記不進（沿用 `.gitignore` 對 `.done` 的既有做法，避免別台 clone 後誤跳過）。 | PI 要求 commit 這兩項。 |