# PROVENANCE — QPMIL-VL 逐折數字（EXT-1 A，2026-10-03 匯入）

本目錄只有數字。沒有 import、複製任何 QPMIL-VL 的程式；匯入腳本在 navrouter-cil 之外（`~/research/ext/ext1_import/import_qpmil.py`），只解析既有執行留下的 `metrics/*.txt` 與 `log.txt`。

## 來源

指令要找的是 navipath clone 裡的十折輸出。實際情況：

| 位置 | 內容 | 是否採用 |
|---|---|---|
| `~/research/01_navipath/outputs/qpmil_{paper,reverse}_fold{1,2,3}.json` | 只有 3 折 × 2 序；自寫的精簡 runner（`train_qpmil_runner.py`，沒有 validation、early stopping、LR scheduler，seed 固定為 1）；沒有 Masked ACC | 否（不是十折，也不是上游的 `main.py`） |
| `~/research/02_pathselect/outputs/exp2/sota/repro_qpmil/` | 十折 × 3 批；上游公開程式 `main.py`；有 ACC、Masked ACC、逐階段正確率矩陣、每折 log 內的 slide ID 清單 | **是** |

採用的三批（都在 `02_pathselect`，repo `nsysu-aivisual/pathselect`，匯入時 HEAD `2435d0d`，該目錄工作樹無未提交變更）：

| 批次 | 序 | 路徑（相對 `repro_qpmil/`） | `bp_every_batch` | GPU（文件記載） | 執行時間（log） | 進版控的 commit | 本目錄的檔 |
|---|---|---|---|---|---|---|---|
| reverse_b8 | reverse | `pod_reverse_b8/revb8/train-data_split_seed_{k}/` | 8（論文 reverse 設定） | A100-SXM4-80GB，RunPod `fazrkhdokjjewd` | 2026-09-08 01:48–03:57 | `3f527fc`（2026-09-08） | `perfold.csv`（reverse） |
| paper_b16 | paper（forward） | `pod_forward/fwd10/train-data_split_seed_{k}/` | 16（程式預設） | RTX 4090，RunPod `uh30ur1kz1suoh` | 2026-09-07 07:13–08:39 | `39eaff6`（2026-09-07） | `perfold.csv`（paper） |
| reverse_b16 | reverse | `pod/fold{k}/train-data_split_seed_{k}/` | 16 | RTX 4090，RunPod `jnd934wacl9pk6` | 2026-09-07 02:49–03:50 | `16805f5`（2026-09-07） | `perfold_reverse_b16.csv`（不進配對表） |

- 上游程式：`can-can-ya/QPMIL-VL` @ `3a7a769`；GPU 路徑的相容性修補只有 `np.long` 等別名與 reverse 用的 `class_ensemble_reverse.json`（`02_pathselect/docs/repro/SHIMS.md`）。
- 三批共同設定（各折 `log.txt` 開頭）：epochs 12 × 4、pool 20、prompt 24、match 5、lr 1e-3、CONCH、`feats-l1-s256_CONCH`、seed = 折號。
- GPU 型號來自 `02_pathselect/docs/ledger/DR-048.md`（:442、:492、:587–592）；`log.txt` 只寫 `cuda:0`。paper_b16 的型號是由「前兩輪是 RTX 4090」（DR-048:587）推得，沒有單獨記載。
- reverse 有兩批：reverse_b8 是論文的 reverse 設定，reverse_b16 是程式預設值。主檔取 reverse_b8，理由是它對應發表表格的設定，且 reverse_b16 的 K6 不過（見下）。兩批之間 GPU 也同時換了（4090 → A100），兩個變數沒有分離。這個選擇沒有看我方任何 test 數字。

## K6（折一致性）

每個 (fold, task, split) 的 slide ID 集合排序後 sha256，與 navrouter-cil 實際載入的清單（`outputs/navcil/mac/ext1/k6_navrouter.json`）比對；ID 取自各折 `log.txt` 的 `sids_train／sids_val／sids_test` 行。結果在 `k6.json`。

| 批次 | 相同格數／120 | test 相同格數／40 | 判定 |
|---|---|---|---|
| reverse_b8 | 120 | 40 | 通過 |
| paper_b16 | 120 | 40 | 通過 |
| reverse_b16 | 114 | 38 | **不過**：fold 7 的 log 是失敗那次的（停在 BRCA），沒有 BRCA、LUNG 的 ID 清單；6 格無法比對 |

特徵檔本身（pod 上的 `can_dataset` 與本機）沒有逐位元比對；只有切分的 slide ID 一致。

## 指標

| 欄位 | 定義 | 來源 |
|---|---|---|
| ACC（t = 1…4） | 四任務（已學任務）等權：`test_acc.txt` 第 t 列的平均。上游在 t = 1（兩類）用二元評估器的 `acc@mid`，t ≥ 2（≥ 4 類）用多類評估器的 `acc`（argmax）；兩者都不用 test 資訊（見下節） | `metrics/test_acc.txt`（下三角 R[t][j]，未遮罩） |
| MaskedACC | 告訴真實任務、logits 只取該任務兩類後的二元 `acc@mid`（softmax 後 p(類 1) > 0.5，等同兩類 argmax），已學任務等權。t = 4 等於 `test_mask_acc.txt` 的平均；t < 4 由 log 的 `[<task>/test/masked/pred]` 行的 `acc@mid` 取得 | `metrics/test_mask_acc.txt`、`log.txt` |
| Forgetting、BWT（只在 t = 4） | 由 R 以 navrouter-cil Table 1 的同一公式重算（`nc5_report.cil_full`：Forgetting 的 max 取階段 j…T−1）。與來源 `summary.json` 的值逐折相同（差 ≤ 2e-6） | 重算 |
| MaskedACC_optthr | 同一行 log 的 `acc`：二元評估器預設以 **test 資料自己的 ROC 曲線**挑出的門檻判的正確率（用到 test 標籤）。**不用於任何比較**，只照錄 | `log.txt` |

- 它的 Masked ACC 是「告訴任務」的版本，對應我方的 oracle 定義；我方主表用的 Table 1 定義（τ̂ 的 expert 證據）在它身上沒有對應物。配對表兩種我方定義都列。
- 數值精度：來源檔為六位小數。

## 上游評估器如何定義 `acc` 與 `acc@mid`（2026-10-03 為 EXT-1 收尾讀取；只讀，未複製或 import；`~/research/ext/QPMIL-VL` @ `3a7a769`）

檔案 `utils/evaluator_clf.py`（工作樹只有 `np.long → np.int64` 的相容性修補，行號與上游相同）與 `manager/manager.py`。

**二元評估器 `BinClf_Evaluator`**（`evaluator_clf.py`）：

```
58:   self.y_hat = F.softmax(data['y_hat'], dim=1)[:, 1]  # apply softmax to get prob.
78:   self.fpr, self.tpr, self.thresholds = metrics.roc_curve(self.y, self.y_hat, pos_label=self.pos_label, drop_intermediate=False)
79:   self.fpr_optimal, self.tpr_optimal, self.threshold_optimal = self._optimal_thresh(self.fpr, self.tpr, self.thresholds)
95-98: def _optimal_thresh(fpr, tpr, thresholds, p=0):
         loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
         idx = np.argmin(loss, axis=0)
         return fpr[idx], tpr[idx], thresholds[idx]
103-109: def _acc(self, threshold=None):
         if threshold is None:
             threshold = self.threshold_optimal
         pred_logit = self.y_hat > threshold
         pred_logit = pred_logit.astype(np.int64)
         acc = np.sum(pred_logit == self.y) / self.y.shape[0]
         return acc
145-146: def _acc_mid_threshold(self):
         return self._acc(threshold=0.5)
```

- `acc`：門檻 = `threshold_optimal`，由 **這批 test 資料自己的標籤**畫出 ROC、取 TPR − FPR 最大的點（行 78–79、95–98）。含 test 資訊，不是標準正確率。
- `acc@mid`：門檻固定 0.5，作用在 softmax 後 p(類 1) 上（行 58、145–146）。兩類時 p(類 1) > 0.5 等同「類 1 的 logit > 類 0 的 logit」，與兩類 argmax 相同（平手時判類 0，與 `np.argmax` 一致）。不含 test 資訊。

**多類評估器 `MultiClf_Evaluator`**（`evaluator_clf.py:253–256`）：

```
253:  def _acc(self):
254:      pred_cls = np.argmax(self.y_hat, axis=-1).astype(np.int64)
255:      acc = np.sum(pred_cls == self.y) / self.y.shape[0]
256:      return acc
```

多類沒有 `acc@mid`，`acc` 本身就是 argmax 正確率，也不含 test 資訊。

**`manager/manager.py` 用哪一個**：

| 用途 | 行 | 取法 | 原因 |
|---|---|---|---|
| Masked（`test_mask_acc.txt`） | 229–235 | `if_binary = v_cltor['y_hat'].shape[1] == 2`（前一行 228 把 logits 切成該任務的兩類）；二元 → `acc@mid` | 各任務 `dataset_subtype_num` 都是 2，所以 Masked 一律走二元評估器的 `acc@mid` |
| 逐階段 R（`test_acc.txt`） | 267、274 | `if_binary = len(current_ensemble_classes['count']) == 2`；t = 1（兩類）取 `acc@mid`，t ≥ 2（≥ 4 類）取多類的 `acc` | 同上 |
| validation | 382 | 同上 | 同上 |

所以本目錄匯入的 ACC 與 MaskedACC 兩欄都是「不含 test 資訊的標準 argmax 正確率」，**不需要換欄、不需要重新匯入**。先前 DECISIONS D10 說「`acc@mid` 等於兩類 argmax」的結論不變，但該句沒有交代多類評估器沒有 0.5 門檻；上表補上。

**獨立驗證**（不經 log 的彙總，直接用逐 slide 機率重算）：`02_pathselect/.../config_full/20260907-104727/train-data_split_seed_1/tcga_rcc-task2/eval_results/masked-tcga_rcc-test.csv`（Mac CPU 的未完成執行，76 張）：`p > 0.5` 的正確率 = 0.947368 = 同次 log 的 `acc@mid`；以該 CSV 自己的 ROC 曲線取 TPR − FPR 最大的門檻（0.026434）的正確率 = 0.934211 = 同次 log 的 `acc`。兩欄的定義與數值都對上。

**兩欄的十折 mean ± sd（t = 4，Masked）**

| 批次 | `acc@mid`（匯入為 MaskedACC，不含 test 資訊） | `acc`（MaskedACC_optthr，test 最佳門檻） |
|---|---|---|
| reverse_b8 | 0.9352 ± 0.0184（n = 10） | 0.9240 ± 0.0167（n = 10） |
| paper_b16 | 0.9261 ± 0.0247（n = 10） | 0.9150 ± 0.0229（n = 10） |
| reverse_b16（K6 不過） | 0.9085 ± 0.0429（n = 10） | 0.9200 ± 0.0195（n = 9；fold 7 的 log 缺逐階段行） |

## 十折 mean ± sd（t = 4）

| 批次 | ACC | MaskedACC | Forgetting | BWT |
|---|---|---|---|---|
| reverse_b8 | 0.8681 ± 0.0463 | 0.9352 ± 0.0184 | 0.0638 ± 0.0542 | −0.0638 ± 0.0542 |
| paper_b16 | 0.8836 ± 0.0326 | 0.9261 ± 0.0247 | 0.0381 ± 0.0179 | −0.0381 ± 0.0179 |
| reverse_b16（K6 不過） | 0.8141 ± 0.0647 | 0.9085 ± 0.0429 | 0.1473 ± 0.0905 | −0.1473 ± 0.0905 |
| 發表值 reverse（Tab. 2） | 0.859 ± 0.032 | 0.925 ± 0.018 | 0.064 ± 0.031 | — |
| 發表值 paper（Tab. 1） | 0.890 ± 0.021 | 0.930 ± 0.018 | 0.027 ± 0.014 | — |

## 表註用（機器與日期）

- reverse：RunPod A100-SXM4-80GB 上於 2026-09-08 01:48–03:57 產生（log 時間）；paper：RunPod RTX 4090 上於 2026-09-07 07:13–08:39 產生。我方 FINAL 在 Mac（Apple M1 Pro、CPU）上於 2026-10-02／03 產生。
- 外部對照不受 AGENTS.md 紅線 4（同一張表的數字來自同一台機器、同一批）約束（PI 2026-10-03 裁決）；表註一律寫明上述機器與日期。

## 限制

- 這些數字在 RunPod GPU 上產生，與 navrouter-cil 的 Mac CPU 數字不是同一台機器；配對表因此跨機器（PI 已裁決外部對照不受紅線 4 約束，見上節）。
- 逐 slide 預測沒有留在本機，無法做 slide 層級的配對或 bootstrap。
- 每個設定只有一次執行（seed = 折號），沒有多 seed。
