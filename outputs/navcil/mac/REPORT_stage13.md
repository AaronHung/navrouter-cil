# REPORT — NC-11：逐折配對檢定、top-K ablation、選取方式 ablation（Mac CPU，十折兩序）

機器：mac（Apple M1 Pro）、CPU、`torch.set_num_threads(8)`、torch 2.11.0；每個子實驗的所有 arm 在同一次執行中完成（log：`logs/nc11_paired.log`、`logs/nc11_select.log`、`logs/nc11_topk.log`）。判準與定義見 `PREREG-11.md`（commit 16275bc）。主方法 D3 = AR 分派器（γ = 1e-3，累加統計量）＋ I6(r = 2) 任務分類頭。task-known = oracle ＋ I6、task-inferred = AR ＋ I6，指標為 t = 4 CIL ACC（四任務等權平均），十折。數字附 fact-id 與其所在的 facts.json 行號（路徑相對於 `outputs/navcil/mac/`）。

## 0 機制說明（唯讀，寫在最前面）

**去重懲罰在做什麼（白話）**：任務分類頭先替每個 patch 打一個「像不像該任務亞型」的分數。一次挑 64 個分數最高的 patch 時，常常會挑到一大堆彼此幾乎一模一樣的相鄰 patch（同一塊組織），證據重複。現行做法改成分 4 輪、每輪挑 16 個：從第 2 輪起，每個還沒被挑的 patch 的分數，會扣掉「λ = 1.5 × 它跟已挑 patch 中最像那一個的 cosine」。跟已挑的很像的 patch 被扣得多，於是後面幾輪會轉去挑分數也高、但長得不一樣的區域，最後 64 個 patch 涵蓋的組織比較多樣。

**每輪之間更新了什麼（附行號）**：呼叫鏈 `scripts/nc8_batch.py:63` → `selector/cil_ops.py:39-45` `four_round`（budget = K、step = 16、redundancy_weight = λ\* = 1.5、normalize_base = True、redundancy_mode = "maxsim"）→ `selector/multiround.py:134-203` `SequentialBudgetedObserver.observe`。
1. 迴圈前：任務分類頭的分數 s 只算一次（`nc8_batch.py:72`），做一次 z-score（`multiround.py:146-150`）；patch 特徵 L2 正規化（`:139`）；「對已選集合的最大 cosine」max_sim_seen 初始為 0（`:152`）。
2. 每輪：調整後分數 = z(s) − λ · max_sim_seen（`:156-163`，第 1 輪 seen 為空，不扣）；已選 patch 設為 −∞，不重選（`:164`）；取最高的 16 個（`:166-169`）加入已選集合（`:173`）。
3. 輪與輪之間：只更新「已選集合」與 max_sim_seen = max(max_sim_seen, 各 patch 對本輪新選 16 個的最大 cosine)（`:178-180`）。分數 s 不重算、任務分類頭不再呼叫、沒有查詢向量或文字向量更新；迴圈內不做分類（`:184`，confidence_threshold = None）。
4. 選完後所選 patch 等權平均並 L2 正規化（`cil_ops.py:31-36`），對 8 類文字取 cosine（`nc8_batch.py:63`）。

**I6 是否以 top-K 的輸出訓練（3.2 的前提）**：是。`selector/cil_ops.py:87-90`：每步以分數取 one-shot top-K（`top_k_select(s.detach(), budget)`），所選 patch 以 softmax(分數) 加權後分類、算 CE；`budget` 預設 64（`cil_ops.py:22`、`:66-67`），`scripts/nc7_i6.py:60-61` 未傳 budget。訓練時是一輪、不去重；推論時才用上面的 4 輪 × 16。因此 3.2 每個 K 都以 one-shot top-K 重新訓練（「全部」= budget 0，全部 patch 參與 softmax 加權），推論每輪 16 個、輪數 = K／16。3.3 不重新訓練，只換推論的挑法。

## 1 重現檢查（驗收 3）

| 子實驗 | 項目 | 基準 | 基準值 | 本輪 | 結果 |
|---|---|---|---|---|---|
| 3.3 arm (i) 4 輪 × 16（既有 I6 權重） | oracle ＋ I6（task-known），reverse | REPORT_stage10.md:18 | 0.9340 ± 0.0202 | 0.9340 ± 0.0202；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| 3.3 arm (i) 4 輪 × 16（既有 I6 權重） | oracle ＋ I6（task-known），paper | REPORT_stage10.md:31 | 0.9340 ± 0.0202 | 0.9340 ± 0.0202；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| 3.3 arm (i) 4 輪 × 16（既有 I6 權重） | AR ＋ I6（task-inferred），reverse | REPORT_stage10.md:16 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| 3.3 arm (i) 4 輪 × 16（既有 I6 權重） | AR ＋ I6（task-inferred），paper | REPORT_stage10.md:29 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| 3.2 K = 64（重新訓練） | oracle ＋ I6（task-known），reverse | REPORT_stage10.md:18 | 0.9340 ± 0.0202 | 0.9340 ± 0.0202；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| 3.2 K = 64（重新訓練） | oracle ＋ I6（task-known），paper | REPORT_stage10.md:31 | 0.9340 ± 0.0202 | 0.9340 ± 0.0202；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| 3.2 K = 64（重新訓練） | AR ＋ I6（task-inferred），reverse | REPORT_stage10.md:16 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| 3.2 K = 64（重新訓練） | AR ＋ I6（task-inferred），paper | REPORT_stage10.md:29 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |

- 3.3 arm (i) 的 8 類 cosine 與 NC-8 快取 `I6_cos8` 逐張最大絕對差：0.00e+00（`nc11.select.cos8_four16_vs_nc8_maxabs`（nc11/select/facts.json:19））。
- 3.2 K = 64 重新訓練的權重與既有 `i6/r2/` 逐位元相同：40/40（最大差 0.00e+00；`nc11.topk.weights_K64_equal`（nc11/topk/facts.json:51））；K = 64 的 8 類 cosine 與 `I6_cos8` 最大差 0.00e+00（`nc11.topk.cos8_K64_vs_nc8_maxabs`（nc11/topk/facts.json:50））。
- 3.1：`nc10/per_fold.json` 的 AR 與 `nc8/per_fold.json` 第 6 列逐折相同：True；重算的每折混淆十折合計與 `nc10/confusion.json` 相同：True。

## 2 逐折配對檢定（3.1；驗收 1）

資料：`nc10/per_fold.json`（REPORT_stage12，同一批 I6 任務分類頭下的 AR、R3(k=8)、R0）。t = 4。Wilcoxon signed-rank 雙尾（`scipy.stats.wilcoxon`，n = 10、無平手時為精確檢定，最小可能 p = 2／1024 ≈ 0.0020）、paired t 雙尾。共做 12 次檢定，未做多重比較校正。t = 4 時兩序的分派結果與 CIL 逐折相同（見下表），兩序各列一次。

### AR vs R3，reverse

| 折 | R3 分派正確率 micro | AR | 差 | R3 分派正確率 macro | AR | 差 | R3 CIL 全程 | AR | 差 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9534 | 0.9821 | +0.0287 | 0.9503 | 0.9720 | +0.0218 | 0.8732 | 0.9117 | +0.0384 |
| 2 | 0.9720 | 0.9930 | +0.0210 | 0.9780 | 0.9802 | +0.0022 | 0.9350 | 0.9372 | +0.0022 |
| 3 | 0.9558 | 0.9932 | +0.0374 | 0.9527 | 0.9941 | +0.0414 | 0.9001 | 0.9414 | +0.0414 |
| 4 | 0.9567 | 0.9833 | +0.0267 | 0.9530 | 0.9608 | +0.0078 | 0.8862 | 0.8914 | +0.0052 |
| 5 | 0.9319 | 0.9857 | +0.0538 | 0.9355 | 0.9887 | +0.0532 | 0.9050 | 0.9583 | +0.0532 |
| 6 | 0.9606 | 0.9892 | +0.0287 | 0.9175 | 0.9659 | +0.0484 | 0.8709 | 0.9164 | +0.0456 |
| 7 | 0.9611 | 0.9859 | +0.0247 | 0.9709 | 0.9608 | -0.0101 | 0.9054 | 0.8901 | -0.0153 |
| 8 | 0.9536 | 0.9821 | +0.0286 | 0.9366 | 0.9860 | +0.0494 | 0.8638 | 0.8772 | +0.0134 |
| 9 | 0.9580 | 0.9755 | +0.0175 | 0.9680 | 0.9664 | -0.0016 | 0.9003 | 0.8987 | -0.0016 |
| 10 | 0.9554 | 0.9851 | +0.0297 | 0.9643 | 0.9732 | +0.0089 | 0.9023 | 0.9056 | +0.0034 |

| 指標 | 平均差 | AR 勝／平／負 | Wilcoxon p | paired t p | fact-id |
|---|---|---|---|---|---|
| 分派正確率 micro | +0.0297 | 10／0／0 | 0.0020 | 6.10e-06 | `nc11.paired.r3.reverse.tp_micro.p_wilcoxon`（nc11/paired/facts.json:5）；`nc11.paired.r3.reverse.tp_micro.p_ttest`（nc11/paired/facts.json:6） |
| 分派正確率 macro | +0.0221 | 8／0／2 | 0.0273 | 1.68e-02 | `nc11.paired.r3.reverse.tp_macro.p_wilcoxon`（nc11/paired/facts.json:10）；`nc11.paired.r3.reverse.tp_macro.p_ttest`（nc11/paired/facts.json:11） |
| CIL 全程 | +0.0186 | 8／0／2 | 0.0371 | 3.56e-02 | `nc11.paired.r3.reverse.acc.p_wilcoxon`（nc11/paired/facts.json:15）；`nc11.paired.r3.reverse.acc.p_ttest`（nc11/paired/facts.json:16） |

### AR vs R3，paper

| 折 | R3 分派正確率 micro | AR | 差 | R3 分派正確率 macro | AR | 差 | R3 CIL 全程 | AR | 差 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9534 | 0.9821 | +0.0287 | 0.9503 | 0.9720 | +0.0218 | 0.8732 | 0.9117 | +0.0384 |
| 2 | 0.9720 | 0.9930 | +0.0210 | 0.9780 | 0.9802 | +0.0022 | 0.9350 | 0.9372 | +0.0022 |
| 3 | 0.9558 | 0.9932 | +0.0374 | 0.9527 | 0.9941 | +0.0414 | 0.9001 | 0.9414 | +0.0414 |
| 4 | 0.9567 | 0.9833 | +0.0267 | 0.9530 | 0.9608 | +0.0078 | 0.8862 | 0.8914 | +0.0052 |
| 5 | 0.9319 | 0.9857 | +0.0538 | 0.9355 | 0.9887 | +0.0532 | 0.9050 | 0.9583 | +0.0532 |
| 6 | 0.9606 | 0.9892 | +0.0287 | 0.9175 | 0.9659 | +0.0484 | 0.8709 | 0.9164 | +0.0456 |
| 7 | 0.9611 | 0.9859 | +0.0247 | 0.9709 | 0.9608 | -0.0101 | 0.9054 | 0.8901 | -0.0153 |
| 8 | 0.9536 | 0.9821 | +0.0286 | 0.9366 | 0.9860 | +0.0494 | 0.8638 | 0.8772 | +0.0134 |
| 9 | 0.9580 | 0.9755 | +0.0175 | 0.9680 | 0.9664 | -0.0016 | 0.9003 | 0.8987 | -0.0016 |
| 10 | 0.9554 | 0.9851 | +0.0297 | 0.9643 | 0.9732 | +0.0089 | 0.9023 | 0.9056 | +0.0034 |

| 指標 | 平均差 | AR 勝／平／負 | Wilcoxon p | paired t p | fact-id |
|---|---|---|---|---|---|
| 分派正確率 micro | +0.0297 | 10／0／0 | 0.0020 | 6.10e-06 | `nc11.paired.r3.paper.tp_micro.p_wilcoxon`（nc11/paired/facts.json:20）；`nc11.paired.r3.paper.tp_micro.p_ttest`（nc11/paired/facts.json:21） |
| 分派正確率 macro | +0.0221 | 8／0／2 | 0.0273 | 1.68e-02 | `nc11.paired.r3.paper.tp_macro.p_wilcoxon`（nc11/paired/facts.json:25）；`nc11.paired.r3.paper.tp_macro.p_ttest`（nc11/paired/facts.json:26） |
| CIL 全程 | +0.0186 | 8／0／2 | 0.0371 | 3.56e-02 | `nc11.paired.r3.paper.acc.p_wilcoxon`（nc11/paired/facts.json:30）；`nc11.paired.r3.paper.acc.p_ttest`（nc11/paired/facts.json:31） |

### AR vs R0，reverse

| 折 | R0 分派正確率 micro | AR | 差 | R0 分派正確率 macro | AR | 差 | R0 CIL 全程 | AR | 差 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9247 | 0.9821 | +0.0573 | 0.8863 | 0.9720 | +0.0857 | 0.8233 | 0.9117 | +0.0883 |
| 2 | 0.9266 | 0.9930 | +0.0664 | 0.9158 | 0.9802 | +0.0645 | 0.8727 | 0.9372 | +0.0645 |
| 3 | 0.9184 | 0.9932 | +0.0748 | 0.9127 | 0.9941 | +0.0814 | 0.8624 | 0.9414 | +0.0791 |
| 4 | 0.9367 | 0.9833 | +0.0467 | 0.9394 | 0.9608 | +0.0214 | 0.8722 | 0.8914 | +0.0192 |
| 5 | 0.9283 | 0.9857 | +0.0573 | 0.9322 | 0.9887 | +0.0565 | 0.8991 | 0.9583 | +0.0592 |
| 6 | 0.9427 | 0.9892 | +0.0466 | 0.8915 | 0.9659 | +0.0744 | 0.8449 | 0.9164 | +0.0716 |
| 7 | 0.9152 | 0.9859 | +0.0707 | 0.9091 | 0.9608 | +0.0517 | 0.8603 | 0.8901 | +0.0298 |
| 8 | 0.9214 | 0.9821 | +0.0607 | 0.9253 | 0.9860 | +0.0606 | 0.8359 | 0.8772 | +0.0413 |
| 9 | 0.9371 | 0.9755 | +0.0385 | 0.9059 | 0.9664 | +0.0605 | 0.8382 | 0.8987 | +0.0605 |
| 10 | 0.9368 | 0.9851 | +0.0483 | 0.9354 | 0.9732 | +0.0379 | 0.8705 | 0.9056 | +0.0351 |

| 指標 | 平均差 | AR 勝／平／負 | Wilcoxon p | paired t p | fact-id |
|---|---|---|---|---|---|
| 分派正確率 micro | +0.0567 | 10／0／0 | 0.0020 | 9.61e-08 | `nc11.paired.r0.reverse.tp_micro.p_wilcoxon`（nc11/paired/facts.json:35）；`nc11.paired.r0.reverse.tp_micro.p_ttest`（nc11/paired/facts.json:36） |
| 分派正確率 macro | +0.0595 | 10／0／0 | 0.0020 | 4.75e-06 | `nc11.paired.r0.reverse.tp_macro.p_wilcoxon`（nc11/paired/facts.json:40）；`nc11.paired.r0.reverse.tp_macro.p_ttest`（nc11/paired/facts.json:41） |
| CIL 全程 | +0.0549 | 10／0／0 | 0.0020 | 3.10e-05 | `nc11.paired.r0.reverse.acc.p_wilcoxon`（nc11/paired/facts.json:45）；`nc11.paired.r0.reverse.acc.p_ttest`（nc11/paired/facts.json:46） |

### AR vs R0，paper

| 折 | R0 分派正確率 micro | AR | 差 | R0 分派正確率 macro | AR | 差 | R0 CIL 全程 | AR | 差 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.9247 | 0.9821 | +0.0573 | 0.8863 | 0.9720 | +0.0857 | 0.8233 | 0.9117 | +0.0883 |
| 2 | 0.9266 | 0.9930 | +0.0664 | 0.9158 | 0.9802 | +0.0645 | 0.8727 | 0.9372 | +0.0645 |
| 3 | 0.9184 | 0.9932 | +0.0748 | 0.9127 | 0.9941 | +0.0814 | 0.8624 | 0.9414 | +0.0791 |
| 4 | 0.9367 | 0.9833 | +0.0467 | 0.9394 | 0.9608 | +0.0214 | 0.8722 | 0.8914 | +0.0192 |
| 5 | 0.9283 | 0.9857 | +0.0573 | 0.9322 | 0.9887 | +0.0565 | 0.8991 | 0.9583 | +0.0592 |
| 6 | 0.9427 | 0.9892 | +0.0466 | 0.8915 | 0.9659 | +0.0744 | 0.8449 | 0.9164 | +0.0716 |
| 7 | 0.9152 | 0.9859 | +0.0707 | 0.9091 | 0.9608 | +0.0517 | 0.8603 | 0.8901 | +0.0298 |
| 8 | 0.9214 | 0.9821 | +0.0607 | 0.9253 | 0.9860 | +0.0606 | 0.8359 | 0.8772 | +0.0413 |
| 9 | 0.9371 | 0.9755 | +0.0385 | 0.9059 | 0.9664 | +0.0605 | 0.8382 | 0.8987 | +0.0605 |
| 10 | 0.9368 | 0.9851 | +0.0483 | 0.9354 | 0.9732 | +0.0379 | 0.8705 | 0.9056 | +0.0351 |

| 指標 | 平均差 | AR 勝／平／負 | Wilcoxon p | paired t p | fact-id |
|---|---|---|---|---|---|
| 分派正確率 micro | +0.0567 | 10／0／0 | 0.0020 | 9.61e-08 | `nc11.paired.r0.paper.tp_micro.p_wilcoxon`（nc11/paired/facts.json:50）；`nc11.paired.r0.paper.tp_micro.p_ttest`（nc11/paired/facts.json:51） |
| 分派正確率 macro | +0.0595 | 10／0／0 | 0.0020 | 4.75e-06 | `nc11.paired.r0.paper.tp_macro.p_wilcoxon`（nc11/paired/facts.json:55）；`nc11.paired.r0.paper.tp_macro.p_ttest`（nc11/paired/facts.json:56） |
| CIL 全程 | +0.0549 | 10／0／0 | 0.0020 | 3.10e-05 | `nc11.paired.r0.paper.acc.p_wilcoxon`（nc11/paired/facts.json:60）；`nc11.paired.r0.paper.acc.p_ttest`（nc11/paired/facts.json:61） |

### AR 變差的折與原因初判

**AR vs R3，第 7 折**：變差的指標 = 分派正確率 macro、CIL 全程（兩序皆然）

| 真實 \ 分派 | esca（R3／AR） | rcc（R3／AR） | brca（R3／AR） | lung（R3／AR） |
|---|---|---|---|---|
| esca | 15／13 | 0／0 | 0／1 | 0／1 |
| rcc | 0／0 | 74／74 | 1／1 | 0／0 |
| brca | 1／0 | 0／0 | 95／96 | 0／0 |
| lung | 4／0 | 1／0 | 4／1 | 88／96 |

| 任務 | test 張數 | TP R3 | TP AR | 只有 AR 分派對（其中第二站判對） | 只有 R3 分派對（其中第二站判對） | 對 CIL 的貢獻（pp） |
|---|---|---|---|---|---|---|
| tcga_esca | 15 | 1.0000 | 0.8667 | 0（0） | 2（2） | -3.33 |
| tcga_rcc | 75 | 0.9867 | 0.9867 | 0（0） | 0（0） | +0.00 |
| tcga_brca | 96 | 0.9896 | 1.0000 | 1（1） | 0（0） | +0.26 |
| tcga_lung | 97 | 0.9072 | 0.9897 | 8（6） | 0（0） | +1.55 |

初判：CIL 差 = Σ_任務（只有 AR 分派對且第二站判對 − 只有 R3 分派對且第二站判對）／張數／4；兩者都分派對的片第二站結果相同，不影響差值。本折負貢獻最大的是 esca（-3.33 pp；15 張中 AR 多分派錯 2 張，AR 多出的錯誤去向：brca 1、lung 1）。esca 每折只有約 15 張 test，一張分派錯就讓該任務正確率變動約 6.7 pp、四任務等權的 CIL 變動約 1.7 pp。

**AR vs R3，第 9 折**：變差的指標 = 分派正確率 macro、CIL 全程（兩序皆然）

| 真實 \ 分派 | esca（R3／AR） | rcc（R3／AR） | brca（R3／AR） | lung（R3／AR） |
|---|---|---|---|---|
| esca | 14／13 | 0／0 | 0／0 | 0／1 |
| rcc | 0／0 | 81／81 | 0／0 | 0／0 |
| brca | 0／0 | 0／0 | 96／96 | 3／3 |
| lung | 4／0 | 0／0 | 5／3 | 83／89 |

| 任務 | test 張數 | TP R3 | TP AR | 只有 AR 分派對（其中第二站判對） | 只有 R3 分派對（其中第二站判對） | 對 CIL 的貢獻（pp） |
|---|---|---|---|---|---|---|
| tcga_esca | 14 | 1.0000 | 0.9286 | 0（0） | 1（1） | -1.79 |
| tcga_rcc | 81 | 1.0000 | 1.0000 | 0（0） | 0（0） | +0.00 |
| tcga_brca | 99 | 0.9697 | 0.9697 | 1（1） | 1（1） | +0.00 |
| tcga_lung | 92 | 0.9022 | 0.9674 | 6（6） | 0（0） | +1.63 |

初判：CIL 差 = Σ_任務（只有 AR 分派對且第二站判對 − 只有 R3 分派對且第二站判對）／張數／4；兩者都分派對的片第二站結果相同，不影響差值。本折負貢獻最大的是 esca（-1.79 pp；14 張中 AR 多分派錯 1 張，AR 多出的錯誤去向：lung 1）。esca 每折只有約 15 張 test，一張分派錯就讓該任務正確率變動約 6.7 pp、四任務等權的 CIL 變動約 1.7 pp。

## 3 top-K ablation（3.2；驗收 2）

每個 K 都以 one-shot top-K 重新訓練 I6（其餘設定同 NC-7：r = 2、seed 42、5 epochs、lr 5e-4、wd 1e-4），推論每輪 16 個、輪數 = K／16、λ\* = 1.5。分派器（AR）與 K 無關。參照 = K = 64。

| 設定 | task-known（oracle ＋ I6） | task-inferred（AR ＋ I6） | 每折標準差（known／inferred） | 與參照差（pp，known／inferred） | 高於／低於參照的折數（known；inferred） | fact-id（reverse） |
|---|---|---|---|---|---|---|
| K = 16（1 輪 × 16） | 0.9318 | 0.9112 | 0.0155／0.0223 | -0.22／-0.16 | 4／6；4／6 | `nc11.topk.16.known.reverse.acc`（nc11/topk/facts.json:2）；`nc11.topk.16.inferred.reverse.acc`（nc11/topk/facts.json:6） |
| K = 32（2 輪 × 16） | 0.9338 | 0.9135 | 0.0188／0.0240 | -0.01／+0.07 | 5／5；6／4 | `nc11.topk.32.known.reverse.acc`（nc11/topk/facts.json:10）；`nc11.topk.32.inferred.reverse.acc`（nc11/topk/facts.json:14） |
| K = 64（4 輪 × 16，現行） | 0.9340 | 0.9128 | 0.0202／0.0258 | +0.00／+0.00 | 0／0；0／0 | `nc11.topk.64.known.reverse.acc`（nc11/topk/facts.json:18）；`nc11.topk.64.inferred.reverse.acc`（nc11/topk/facts.json:22） |
| K = 128（8 輪 × 16） | 0.9369 | 0.9157 | 0.0192／0.0239 | +0.29／+0.29 | 9／1；9／1 | `nc11.topk.128.known.reverse.acc`（nc11/topk/facts.json:26）；`nc11.topk.128.inferred.reverse.acc`（nc11/topk/facts.json:30） |
| 全部（訓練與推論都不選取；推論 = 全部 patch 等權平均） | 0.7763 | 0.7597 | 0.0288／0.0338 | -15.76／-15.31 | 0／10；0／10 | `nc11.topk.all.known.reverse.acc`（nc11/topk/facts.json:34）；`nc11.topk.all.inferred.reverse.acc`（nc11/topk/facts.json:38） |
| 另報：全部（推論以 softmax(分數) 加權） | 0.9348 | 0.9136 | 0.0230／0.0269 | +0.08／+0.08 | 6／4；6／4 | `nc11.topk.all_softmax.known.reverse.acc`（nc11/topk/facts.json:42）；`nc11.topk.all_softmax.inferred.reverse.acc`（nc11/topk/facts.json:46） |

兩序分列（t = 4 ACC，十折 mean ± sd）：

| 設定 | known reverse | known paper | inferred reverse | inferred paper |
|---|---|---|---|---|
| K = 16（1 輪 × 16） | 0.9318 ± 0.0155 | 0.9318 ± 0.0155 | 0.9112 ± 0.0223 | 0.9112 ± 0.0223 |
| K = 32（2 輪 × 16） | 0.9338 ± 0.0188 | 0.9338 ± 0.0188 | 0.9135 ± 0.0240 | 0.9135 ± 0.0240 |
| K = 64（4 輪 × 16，現行） | 0.9340 ± 0.0202 | 0.9340 ± 0.0202 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258 |
| K = 128（8 輪 × 16） | 0.9369 ± 0.0192 | 0.9369 ± 0.0192 | 0.9157 ± 0.0239 | 0.9157 ± 0.0239 |
| 全部（訓練與推論都不選取；推論 = 全部 patch 等權平均） | 0.7763 ± 0.0288 | 0.7763 ± 0.0288 | 0.7597 ± 0.0338 | 0.7597 ± 0.0338 |
| 另報：全部（推論以 softmax(分數) 加權） | 0.9348 ± 0.0230 | 0.9348 ± 0.0230 | 0.9136 ± 0.0269 | 0.9136 ± 0.0269 |

Forgetting（t = 4，十折平均；task-inferred，reverse／paper）：

| 設定 | reverse | paper |
|---|---|---|
| K = 16（1 輪 × 16） | 0.0219 | 0.0037 |
| K = 32（2 輪 × 16） | 0.0219 | 0.0033 |
| K = 64（4 輪 × 16，現行） | 0.0224 | 0.0041 |
| K = 128（8 輪 × 16） | 0.0224 | 0.0041 |
| 全部（訓練與推論都不選取；推論 = 全部 patch 等權平均） | 0.0171 | 0.0041 |
| 另報：全部（推論以 softmax(分數) 加權） | 0.0224 | 0.0041 |

**結論**：K ∈ {16, 32, 64, 128} 內，task-known 與 K = 64 的差距不超過 0.29 pp、task-inferred 不超過 0.29 pp；不做選取（全部 patch 等權平均）時 task-known -15.76 pp、task-inferred -15.31 pp。

註：「全部」的推論是全部 patch 等權平均（= 整片 mean_vec），任務分類頭不參與，所以這一列反映的是「不用任務分類頭挑證據」；它的任務分類頭雖以全部 patch 訓練，只在「另報：softmax 加權」一列被使用。

訓練耗時（秒，十折四任務合計）：K = 16 954；K = 32 945；K = 64 1,089；K = 128 991；K = 全部 1,171；評估 434；全部 5,584。

## 4 選取方式 ablation（3.3；驗收 2）

訓練不變（既有 I6 r = 2，one-shot top-64 訓練）。只換推論時的挑法；參照 = (i)。

| 設定 | task-known（oracle ＋ I6） | task-inferred（AR ＋ I6） | 每折標準差（known／inferred） | 與參照差（pp，known／inferred） | 高於／低於參照的折數（known；inferred） | fact-id（reverse） |
|---|---|---|---|---|---|---|
| (i) 4 輪 × 16，去重懲罰 λ = 1.5（現行） | 0.9340 | 0.9128 | 0.0202／0.0258 | +0.00／+0.00 | 0／0；0／0 | `nc11.select.four16.known.reverse.acc`（nc11/select/facts.json:2）；`nc11.select.four16.inferred.reverse.acc`（nc11/select/facts.json:6） |
| (ii) 一次取 top-64，不去重 | 0.9372 | 0.9160 | 0.0209／0.0264 | +0.32／+0.32 | 8／2；8／2 | `nc11.select.oneshot64.known.reverse.acc`（nc11/select/facts.json:10）；`nc11.select.oneshot64.inferred.reverse.acc`（nc11/select/facts.json:14） |

兩序分列（t = 4 ACC，十折 mean ± sd）：

| 設定 | known reverse | known paper | inferred reverse | inferred paper |
|---|---|---|---|---|
| (i) 4 輪 × 16，去重懲罰 λ = 1.5（現行） | 0.9340 ± 0.0202 | 0.9340 ± 0.0202 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258 |
| (ii) 一次取 top-64，不去重 | 0.9372 ± 0.0209 | 0.9372 ± 0.0209 | 0.9160 ± 0.0264 | 0.9160 ± 0.0264 |

Forgetting（t = 4，十折平均；task-inferred，reverse／paper）：

| 設定 | reverse | paper |
|---|---|---|
| (i) 4 輪 × 16，去重懲罰 λ = 1.5（現行） | 0.0224 | 0.0041 |
| (ii) 一次取 top-64，不去重 | 0.0224 | 0.0041 |

兩種挑法所選 64 個 patch 的 Jaccard（自家任務的頭，2,835 張 test slide）：平均 0.8333、中位數 0.8551、最小 0.3617（`nc11.select.jaccard_own.mean`（nc11/select/facts.json:18））。

**結論**：一次取 top-64 不去重與 4 輪 × 16 去重相比，task-known 差 +0.32 pp、task-inferred 差 +0.32 pp。
