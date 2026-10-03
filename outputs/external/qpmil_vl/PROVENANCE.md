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
| ACC（t = 1…4） | 四任務（已學任務）等權：`test_acc.txt` 第 t 列的平均 | `metrics/test_acc.txt`（下三角 R[t][j]，未遮罩） |
| MaskedACC | 告訴真實任務、在其兩類內判（門檻 0.5，等同兩類 argmax）的正確率，已學任務等權。t = 4 等於 `test_mask_acc.txt` 的平均；t < 4 由 log 的 `[<task>/test/masked/pred]` 行的 `acc@mid` 取得 | `metrics/test_mask_acc.txt`、`log.txt` |
| Forgetting、BWT（只在 t = 4） | 由 R 以 navrouter-cil Table 1 的同一公式重算（`nc5_report.cil_full`：Forgetting 的 max 取階段 j…T−1）。與來源 `summary.json` 的值逐折相同（差 ≤ 2e-6） | 重算 |
| MaskedACC_optthr | log 同一行的 `acc`：以 test 上 ROC 最佳門檻判的正確率。門檻由 test 決定，**不用於任何比較**，只照錄 | `log.txt` |

- 它的 Masked ACC 是「告訴任務」的版本，對應我方的 oracle 定義；我方主表用的 Table 1 定義（τ̂ 的 expert 證據）在它身上沒有對應物。配對表兩種我方定義都列。
- 數值精度：來源檔為六位小數。

## 十折 mean ± sd（t = 4）

| 批次 | ACC | MaskedACC | Forgetting | BWT |
|---|---|---|---|---|
| reverse_b8 | 0.8681 ± 0.0463 | 0.9352 ± 0.0184 | 0.0638 ± 0.0542 | −0.0638 ± 0.0542 |
| paper_b16 | 0.8836 ± 0.0326 | 0.9261 ± 0.0247 | 0.0381 ± 0.0179 | −0.0381 ± 0.0179 |
| reverse_b16（K6 不過） | 0.8141 ± 0.0647 | 0.9085 ± 0.0429 | 0.1473 ± 0.0905 | −0.1473 ± 0.0905 |
| 發表值 reverse（Tab. 2） | 0.859 ± 0.032 | 0.925 ± 0.018 | 0.064 ± 0.031 | — |
| 發表值 paper（Tab. 1） | 0.890 ± 0.021 | 0.930 ± 0.018 | 0.027 ± 0.014 | — |

## 限制

- 這些數字在 RunPod GPU 上產生，與 navrouter-cil 的 Mac CPU 數字不是同一台機器。配對表因此跨機器；是否與 AGENTS.md 紅線 4 相容由 PI 決定。
- 逐 slide 預測沒有留在本機，無法做 slide 層級的配對或 bootstrap。
- 每個設定只有一次執行（seed = 折號），沒有多 seed。
