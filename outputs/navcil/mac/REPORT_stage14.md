# REPORT — NC-12：六種學習順序下的中途軌跡與 t = 4 順序無關驗證（Mac CPU，十折）

機器：mac（Apple M1 Pro）、CPU、torch 2.11.0；所有數字來自同一台、同一次執行（`scripts/nc12_order_traj.py`，log：`logs/nc12_order_run2.log`；第一次執行見文末「執行紀錄」）。判準與定義見 `PREREG-12.md`（commit bf6cbf7）。主方法 D3 = AR 分派器（γ = 1e-3，`selector/incremental_ridge.py` 累加統計量，第 t 階段只讀該順序第 t 個任務的 train mean_vec）＋ I6(r = 2) 任務分類頭；證據與 mean_vec 為 NC-8 同一批快取。CIL 正確率只計已見任務、各任務等權平均；分派正確率 micro = 已見任務 test slides 中分派正確的比例，macro = 各已見任務比例的等權平均。數字附 fact-id 與 `outputs/navcil/mac/nc12/facts.json` 的行號。

## 順序

| 代號 | 順序 | 來源 |
|---|---|---|
| reverse | esca → rcc → brca → lung | 既有 |
| paper | lung → brca → rcc → esca | 既有 |
| perm1 | lung → rcc → esca → brca | random.Random(20260930).sample，排除重複 |
| perm2 | lung → esca → brca → rcc | random.Random(20260930).sample，排除重複 |
| perm3 | esca → rcc → lung → brca | random.Random(20260930).sample，排除重複 |
| perm4 | brca → esca → rcc → lung | random.Random(20260930).sample，排除重複 |

## T1 軌跡表：CIL 正確率（十折 mean ± sd）

| 順序 | t = 1 | t = 2 | t = 3 | t = 4 |
|---|---|---|---|---|
| reverse（esca → rcc → brca → lung） | 0.9608 ± 0.0547 | 0.9532 ± 0.0354 | 0.9360 ± 0.0232 | 0.9128 ± 0.0258 |
| paper（lung → brca → rcc → esca） | 0.8958 ± 0.0390 | 0.8964 ± 0.0336 | 0.9149 ± 0.0233 | 0.9128 ± 0.0258 |
| perm1（lung → rcc → esca → brca） | 0.8958 ± 0.0390 | 0.9246 ± 0.0215 | 0.9215 ± 0.0287 | 0.9128 ± 0.0258 |
| perm2（lung → esca → brca → rcc） | 0.8958 ± 0.0390 | 0.9049 ± 0.0431 | 0.9029 ± 0.0353 | 0.9128 ± 0.0258 |
| perm3（esca → rcc → lung → brca） | 0.9608 ± 0.0547 | 0.9532 ± 0.0354 | 0.9215 ± 0.0287 | 0.9128 ± 0.0258 |
| perm4（brca → esca → rcc → lung） | 0.9181 ± 0.0241 | 0.9252 ± 0.0426 | 0.9360 ± 0.0232 | 0.9128 ± 0.0258 |

fact-id：`nc12.traj.<順序>.t<階段>.acc`（例：`nc12.traj.reverse.t4.acc`（nc12/facts.json:20））。

## T2 軌跡表：分派正確率 micro（十折 mean ± sd；t = 1 只有一個任務，恆為 1）

| 順序 | t = 1 | t = 2 | t = 3 | t = 4 |
|---|---|---|---|---|
| reverse | 1.0000 ± 0.0000 | 0.9946 ± 0.0076 | 0.9914 ± 0.0046 | 0.9855 ± 0.0053 |
| paper | 1.0000 ± 0.0000 | 0.9885 ± 0.0088 | 0.9885 ± 0.0066 | 0.9855 ± 0.0053 |
| perm1 | 1.0000 ± 0.0000 | 0.9954 ± 0.0025 | 0.9910 ± 0.0037 | 0.9855 ± 0.0053 |
| perm2 | 1.0000 ± 0.0000 | 0.9910 ± 0.0059 | 0.9855 ± 0.0096 | 0.9855 ± 0.0053 |
| perm3 | 1.0000 ± 0.0000 | 0.9946 ± 0.0076 | 0.9910 ± 0.0037 | 0.9855 ± 0.0053 |
| perm4 | 1.0000 ± 0.0000 | 0.9918 ± 0.0080 | 0.9914 ± 0.0046 | 0.9855 ± 0.0053 |

分派正確率 macro：

| 順序 | t = 1 | t = 2 | t = 3 | t = 4 |
|---|---|---|---|---|
| reverse | 1.0000 ± 0.0000 | 0.9891 ± 0.0209 | 0.9828 ± 0.0144 | 0.9748 ± 0.0119 |
| paper | 1.0000 ± 0.0000 | 0.9883 ± 0.0088 | 0.9885 ± 0.0063 | 0.9748 ± 0.0119 |
| perm1 | 1.0000 ± 0.0000 | 0.9950 ± 0.0027 | 0.9790 ± 0.0182 | 0.9748 ± 0.0119 |
| perm2 | 1.0000 ± 0.0000 | 0.9729 ± 0.0238 | 0.9729 ± 0.0222 | 0.9748 ± 0.0119 |
| perm3 | 1.0000 ± 0.0000 | 0.9891 ± 0.0209 | 0.9790 ± 0.0182 | 0.9748 ± 0.0119 |
| perm4 | 1.0000 ± 0.0000 | 0.9786 ± 0.0223 | 0.9828 ± 0.0144 | 0.9748 ± 0.0119 |

fact-id：`nc12.traj.<順序>.t<階段>.tp_micro`／`.tp_macro`。

## T3 t = 4 一致性檢查（逐折）

| 判準 | 內容 | 結果 |
|---|---|---|
| 1 | 每張 test slide 的 τ̂ 在 6 種順序下逐張一致（依任務對齊） | 通過（不一致合計 0 張；`nc12.check.tau_mismatch_total`（nc12/facts.json:146）） |
| 2 | t = 4 ACC 在 6 種順序下位元相同，且等於 nc8/per_fold.json 第 6 列 | 通過 |
| 3 | t = 4 的 W 與 reverse 最大絕對差 ≤ 1e-9 | 通過（最大 1.25e-10；`nc12.check.w_maxabs`（nc12/facts.json:147）） |
| 另核對 | reverse、paper 的逐階段 CIL 與 nc8/per_fold.json 第 6 列 acc_t 位元相同 | 通過 |

| 折 | test 張數 | τ̂ 不一致（6 序合計） | t = 4 ACC（6 序） | 6 序位元相同 | 等於 NC-8 D3 | W 最大差（對 reverse） | fact-id |
|---|---|---|---|---|---|---|---|
| 1 | 279 | 0 | 0.9117 | 是 | 是 | 1.12e-10 | `nc12.t4.fold1.acc`（nc12/facts.json:148） |
| 2 | 286 | 0 | 0.9372 | 是 | 是 | 9.68e-11 | `nc12.t4.fold2.acc`（nc12/facts.json:149） |
| 3 | 294 | 0 | 0.9414 | 是 | 是 | 1.01e-10 | `nc12.t4.fold3.acc`（nc12/facts.json:150） |
| 4 | 300 | 0 | 0.8914 | 是 | 是 | 9.28e-11 | `nc12.t4.fold4.acc`（nc12/facts.json:151） |
| 5 | 279 | 0 | 0.9583 | 是 | 是 | 7.17e-11 | `nc12.t4.fold5.acc`（nc12/facts.json:152） |
| 6 | 279 | 0 | 0.9164 | 是 | 是 | 9.01e-11 | `nc12.t4.fold6.acc`（nc12/facts.json:153） |
| 7 | 283 | 0 | 0.8901 | 是 | 是 | 8.69e-11 | `nc12.t4.fold7.acc`（nc12/facts.json:154） |
| 8 | 280 | 0 | 0.8772 | 是 | 是 | 9.72e-11 | `nc12.t4.fold8.acc`（nc12/facts.json:155） |
| 9 | 286 | 0 | 0.8987 | 是 | 是 | 9.85e-11 | `nc12.t4.fold9.acc`（nc12/facts.json:156） |
| 10 | 269 | 0 | 0.9056 | 是 | 是 | 1.25e-10 | `nc12.t4.fold10.acc`（nc12/facts.json:157） |

十折平均 ± sd：0.9128 ± 0.0258（REPORT_stage10.md:16，`nc8.t1.d3.reverse.acc` 為 0.9128 ± 0.0258）。

## T4 觀察：中途分派正確率最低的階段

t = 2、3 之中，十折平均分派正確率 micro 最低的是 **perm2（lung → esca → brca → rcc）的 t = 3**：micro 0.9855 ± 0.0096、macro 0.9729 ± 0.0222（`nc12.traj.perm2.t3.tp_micro`（nc12/facts.json:88））；已見任務為 lung、esca、brca。主要錯分：brca→lung 12 張、lung→brca 9 張、esca→lung 5 張。

十折合計混淆矩陣（perm2、t = 3；列 = 真實、欄 = 分派，只含已見任務）：

| 真實 \ 分派 | esca | rcc | brca | lung |
|---|---|---|---|---|
| esca | 141 | — | 4 | 5 |
| brca | 0 | — | 940 | 12 |
| lung | 0 | — | 9 | 956 |

耗時：2 秒。

## 執行紀錄

1. 第一次執行（2026-09-30 09:28:48，log：`outputs/navcil/mac/logs/nc12_order.log`，exit = 1）因腳本 bug 失敗：`ValueError: '1,3' is not in list`，發生在 `scripts/nc8_report.py:58`（`B8.gather`）。
2. 原因：當時的腳本呼叫 NC-8 的 `B8.gather` 組合已見任務的 test 資料；它會依已見子集取 zero-shot 證據 `zs8_cos8` 的欄，而 NC-8 快取的子集清單只含 reverse、paper 的 7 個前綴（`selector/cil_eval.py:83-90`、`scripts/nc8_batch.py:65-67`）。perm1 在 t = 2 的已見子集 {rcc, lung}（`'1,3'`）不在清單中。D3 不使用 `zs8_cos8`。
3. 修正（commit 訊息 "fix: own gather without zs8 (first run failed, see log)"）：改用本檔自寫的 `gather`，只取 labels、mean_vec、`I6_cos8` 與任務標記；既有 `nc8_report.py` 未改。重跑前已確認本檔用到的資料來源中，依子集或順序預先算好的只有 `zs8_cos8`（依子集）、`L1_cos8`（依順序）與中繼清單 `subsets`／`orders`，本檔都不使用；`I6_cos8`、`mean_vec`、`labels` 與順序無關。
4. 失敗發生在第一個新順序（perm1）第 1 折的 t = 2，在寫出任何結果檔之前；不涉及 PREREG-12 的定義或判準變更。本報告所有數字來自修正後的整批重跑（`logs/nc12_order_run2.log`）。
