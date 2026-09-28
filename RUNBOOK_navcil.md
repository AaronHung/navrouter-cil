# RUNBOOK — NavRouter-CIL stage9／stage10 報告的產出程式（2026-09-28 盤點，read-only）

## 1. 一句話結論
兩份報告都不在 navipath、mllm_hwsi、pathselect 三個 repo，而是由第四個 repo `AaronHung/navrouter-cil`（本機 `~/research/01_navrouter-cil`）產生。stage9 在 commit `6ba9a16` 由 `scripts/nc7_i6.py` → `scripts/nc7_report.py` 產生（報告內稱 NC-7）；stage10、`per_fold.json`、`deferral_curve.csv` 在 commit `5751568`（= 目前 HEAD）由 `scripts/nc8_batch.py` → `scripts/nc8_report.py` 產生（報告內稱 NC-8）。

## 2. Repo 對照表
| repo | 本機路徑 | branch | HEAD | 未 commit 變更 | 在本研究的角色 |
|---|---|---|---|---|---|
| AaronHung/navrouter-cil | ~/research/01_navrouter-cil | main | 5751568 | 無（只多了本檔） | **正本**（報告與全部產出程式） |
| AaronHung/pathselect | ~/research/02_pathselect | cockpit | 2435d0d | 4 個 untracked（logs/pod/*.stdout、paper/figures/archive/） | 參考：`configs/base.yaml:2` 記載設定取自 `pathselect@081bdcd`；6 個模組逐位元相同（見 §4 末） |
| （同上，另兩個 clone） | ~/research/pathselect-paper、~/research/pathselect-exp5 | paper/v1.0-m64、exp/m64-5090-backup | 081bdcd、1529c0b | 無 | 參考（081bdcd 是 base.yaml 記載的來源版本） |
| AaronHung/navipath | ~/research/01_navipath | main | 9dbfe2c | 20 項（M RUNPOD_SETUP.md、outputs/figs/*.pdf 等） | 未使用程式；只借用 CONCH 權重路徑（`configs/machine_mac.yaml:4`） |
| AaronHung/mllm_hwsi_ah | ~/research/01_mllm_hwsi/mllm_hwsi_ah | main | 2295fba | 無 | 未使用 |

- 題目給的 `find -name navipath/mllm_hwsi/pathselect` 只找到 `01_navipath/pathselect`（是 navipath 的子目錄，不是獨立 repo）和 `01_research-cockpit-old/projects/pathselect`；上表是改用 git remote 辨識的結果。mllm_hwsi 的 remote 名稱是 `mllm_hwsi_ah`。
- 報告只有一份：`find ~ -path '*outputs/navcil/mac*'` 只找到 01_navrouter-cil 裡的檔案。檔案 mtime（09-26 11:48、14:46）與 commit 時間（11:48:58、14:46:56）吻合，工作樹與 HEAD 沒有差異。
- 三個 repo grep `REPORT_stage`、`deferral_curve`、`navcil` 都是 0 筆。navipath 裡的 42 筆與 pathselect 裡的 5 筆 `per_fold` 屬於各自的實驗，與本報告無關。

## 3. 執行入口（Mac；會用到的路徑都已用 ls 確認存在）
```bash
cd /Users/aaron/research/01_navrouter-cil
export NAVCIL_MACHINE=mac PYTHONNOUSERSITE=1      # run_stage.sh 內部也會設定（run_stage.sh:38）
PY=/Users/aaron/venvs/navcil/bin/python           # conda env navcil；log 記錄的就是這個直譯器
# stage9（NC-7）
scripts/run_stage.sh nc7_i6 $PY scripts/nc7_i6.py --device cpu   # 來源：log 有記載（logs/nc7_i6.log:1，exit=0 在第 740 行）
$PY scripts/nc7_report.py                                         # 從程式推回（nc7_report.py:5 docstring；沒有 log）
# stage10（NC-8）
scripts/run_stage.sh nc8_batch $PY scripts/nc8_batch.py --device cpu  # 來源：log 有記載（logs/nc8_batch.log:1，exit=0 在第 23 行）
$PY scripts/nc8_report.py                                         # 從程式推回（nc8_report.py:5 docstring；沒有 log）
```
- 兩份報告本身都沒有寫出指令或腳本名稱，只寫了 `--device cpu`、threads 8、PREREG commit（stage9 第 3 行 → PREREG-7 @3d70d29；stage10 第 3 行 → PREREG-8 @6ff3315）。這段機器敘述是報告程式寫死的字串（nc7_report.py:161、nc8_report.py:275），不是執行時讀取的值。
- 已存在 `<task>.done` 的步驟會被跳過（例如 nc8_batch.py:78、nc7_i6.py:44、nc7_i6.py:77），所以上面的指令在現況下不會重算。
- 前置產物（gitignored，都在 `outputs/navcil/mac/`）：`bank/`（nc1_pipeline.py）、`nc1/lambda.json`（λ\*=1.5）、`cache/fold*`（nc1_pipeline.py）、`cache/nc2_*`（nc2_r_cache.py）、`cache/nc5_*`（nc5_ctx_cache.py）、`lora_v2/r2/{reverse,paper}/`（`nc2_lora.py --tag lora_v2 --r 2`）、`i6/r2/`（nc7_i6.py），以及 nc1、nc3、nc5、nc6、nc7 的 `metrics.json`（各關的 report 腳本產生；nc8_report 的 T5 比對會用到，見 nc8_report.py:204-208）。各關的實際指令記在 `outputs/navcil/mac/logs/*.log` 第 1 行。

## 4. 元件位置表
| 元件 | 檔案:行號 | 函式／類別 | 備註 |
|---|---|---|---|
| mean_vec 計算 | selector/cil_ops.py:31-36 | `mean_norm` | 全部 patch 平均後做 L2 正規化。stage10 的呼叫點：nc8_batch.py:116（train）、:62（test）；stage9 的呼叫點：nc1_pipeline.py:149（val/test）、:247（train），經由 nc6_report.py:44-49 `feat` 讀取 |
| patch feature 來源 | configs/base.yaml:35-40；selector/evaluate.py:29-54；data/wsi_dataset.py:29-31 | `slide_dataset`、`read_slide`、`WSIClf` | `/Users/aaron/research/can_dataset/<task>/feats-l1-s256_CONCH/pt_files/*.pt`。backbone 是 CONCH（base.yaml:35 註解、:40 `conch_path_feat: CONCH` 與目錄名）。實測 tensor 形狀 `(1876, 512)`、float32，維度 512（base.yaml:8） |
| text encoder | selector/text_encoder.py:149-175；third_party/conch/text_tower.py:38 | `build_f_txt`、`build_text_tower` | CONCH ViT-B-16 text tower。執行時直接讀已 commit 的 `cache/text/f_txt_<task>.pt`（text_encoder.py:156），不載入權重 |
| 群中心法分派器（推測對應 R3，k=8） | scripts/nc8_report.py:66-69、91-94；scripts/nc4_report.py:121-124 | `B8.r3_keys`、`router("R3")`、`kmeans_keys` | 每任務對 train mean_vec 做 KMeans(k=8, n_init=10, random_state=0)，取最大 cosine。stage10 主表第 3 列使用；stage9 沒有使用。「群中心法 = R3」是從描述推測的對應 |
| AR 分派器 | stage10：scripts/nc8_report.py:71-88、91-97；stage9：scripts/nc6_report.py:65-106（由 nc7_report.py:75-76 呼叫）；scripts/nc5_report.py:151-154 | `B8.W`、`router`、`ARX.A/W/fn`、`aug` | γ=1e-3（nc8_report.py:95、nc7_report.py:30），float64，513 維（mean_vec 接常數 1）。**會讀取舊任務舊片的 mean_vec**：在第 t 階段，程式對所有已見任務 `pos[:t]` 逐一載入該任務 train slides 的 mean_vec 快取，從零重算 A=ΣXᵀX 與 b（nc8_report.py:74-81；nc6_report.py:81-88）。**不讀舊任務的片（特徵檔）**：報告程式只讀快取（nc8_report.py:53）；特徵檔只在 nc8_batch 裡每折讀一次，而且四個任務一次讀完、與任務順序無關（nc8_batch.py:113-117）。推論時只用 test slide 自己的 mean_vec（nc8_report.py:97） |
| r=2 任務分類頭（I6） | selector/i6_expert.py:25-52 | `I6Expert` | A (2×514)=1,028、b1 (2)、w2 (2)、b2 (標量 1)，合計 1,033 = 516r+1。計算處：i6_expert.py:51-52 `n_params`、nc7_report.py:66 `516 * r + 1`、nc7_i6.py:51 log 輸出。輸入 u=[Z(512); text_nav_feats(2)]（i6_expert.py:39） |
| top-64／4 輪×16 證據選取 | selector/cil_ops.py:22-23、39-45；selector/multiround.py:154-180 | `four_round`、`SequentialBudgetedObserver.observe` | BUDGET=64、STEP=16；從第 2 輪起扣 λ·max cos（λ\*=1.5，`outputs/navcil/mac/nc1/lambda.json`）。I6 的呼叫點：nc8_batch.py:63、72；nc7_i6.py:104-105。訓練時改用 one-shot top-64（cil_ops.py:88） |
| 十折切分 | configs/base.yaml:37；data/table_utils.py:14-26；selector/evaluate.py:39-41 | `read_datasplit_npz` | `can_dataset/<task>/datasplit/fold_{1..10}.npz`，鍵為 train/val/test_patients（esca fold1：118／15／15）。產生邏輯與 seed：**未找到**（見 §8） |
| reverse／paper 順序 | selector/cil_ops.py:18-21 | `ORDERS` | reverse = esca→rcc→brca→lung；paper = lung→brca→rcc→esca。8 類固定依 base.yaml:29-33 的順序排列 |
| 亞型文字描述 | data/class_prompts.json:4（classnames）、:60（templates）；configs/base.yaml:15-16 | `class_prompt_ensemble`（text_encoder.py:75） | 每類多個同義名 × 22 個模板，編碼後取平均。I6 與四輪選取用的是該任務自己 2 類的文字（nc1_pipeline.py:75-76 `f_task`） |
| 任務文字描述 | scripts/nc1_organ_keys.py:23-25 → cache/text/organ_keys.pt | `ORGANS`、`TEMPLATES` | 只有 NC-1 的 text-organ 分派器使用（selector/router.py:24-25）；stage9／10 沒有呼叫 |
| deferral 5% | stage10：scripts/nc8_report.py:175-201（QS 在 :32）；stage9：scripts/nc7_report.py:102-135（QS 在 :31） | `main` 內迴圈 | margin = AR 分數第一名減第二名；margin 最小的 ⌈q·N⌉ 張轉交，剩下的片計算四任務等權 ACC。stage10 搭配 I6；stage9 搭配 D1（L1 v2）。CSV 寫出位置：nc8_report.py:248-256 |
| fact-id 慣例 | ~/research/01_research-cockpit/docs/UPDATE_SOP.md:40；tools/import_facts.py:204、217；tools/check_claims.mjs:356 | `add()`、`FACT_ID` regex | 格式如 `nc8.t1.<row>.<order>.<metric>`、`nc8.defer.q00.reverse.acc`；fact 檔的 `source_head` = 5751568（projects/navrouter-cil/facts/nc8_deferral.json）。本 repo 內沒有 fact-id |
| PREREG-7 | ~/research/01_navrouter-cil/PREREG-7.md | — | commit 3d70d29（在 6ba9a16 之前）；PREREG-8 為 6ff3315 |

重複實作（stage9／10 是否實際呼叫）：
- AR 在本 repo 有三份：`nc8_report.B8.W` 由 stage10 呼叫；`nc6_report.ARX` 由 stage9 呼叫；`nc5_report.Ridge`（:157-202）兩者都沒有呼叫，只用在 NC-5。
- 群中心：`nc4_report.kmeans_keys` 由 stage10 呼叫；`selector/router.py:66-73 key_multi_proto` 與 `:93-95 R3` 兩者都沒有呼叫（NC-2 版本）。
- 低秩頭：`I6Expert` 與 `selector/lora_expert.py:19 LowRankExpert` 兩者都有呼叫（後者是對照組 D1 的 L1 v2，見 nc8_batch.py:47）。
- 跨 repo：02_pathselect 的 `selector/{flat_selector,multiround,evaluate,classifier}.py` 與 `data/{table_utils,wsi_dataset}.py` 和本 repo 逐位元相同（cmp），`selector/text_encoder.py`、`device.py` 內容不同。navipath 的 `zeronav/router.py` 是逐 patch 的 `TextNavRouter`，不是分派器。mllm_hwsi 的 `nav/device.py` 內容不同。這些跨 repo 的檔案都**沒有被呼叫**：本 repo 只把自己的根目錄與 `scripts/` 加進 sys.path（nc8_report.py:20-21），AGENTS.md 紅線 1 也禁止 import navipath（`tests/test_no_banned_deps.py` 另外禁用 QPMIL／zeronav 識別字）。

## 5. 資料與產物位置
- features：`/Users/aaron/research/can_dataset/{tcga_esca,tcga_rcc,tcga_brca,tcga_lung}/feats-l1-s256_CONCH/pt_files/`（158／937／1133／1054 個檔）
- splits：`…/can_dataset/<task>/datasplit/fold_{1..10}.npz`；標籤表：`…/<task>/table/TCGA_<TASK>_path_subtype_x10_processed.csv`
- text prompts：`data/class_prompts.json`、`scripts/nc1_organ_keys.py`；文字特徵快取：`cache/text/f_txt_<task>.pt`、`organ_keys.pt`（已 commit）
- CONCH 權重（只有重建文字快取時才需要）：`/Users/aaron/research/01_navipath/checkpoints/conch/pytorch_model.bin`（802 MB）
- outputs：`outputs/navcil/mac/REPORT_stage{9,10}.md`、`nc8/per_fold.json`、`nc8/fig/*.csv`、`nc7/metrics.json`、`i6/r{1,2,3}/`、`cache/nc8_*`（80 檔）、`logs/`

## 6. 環境
- **Mac 實測**：conda env `navcil`（`/Users/aaron/venvs/navcil`，有 conda-meta）；Python 3.12.2；torch 2.11.0（mps available: True，但兩關都強制 `--device cpu`：nc7_i6.py:134、nc8_batch.py:104）；threads 8（machine_mac.yaml:6 → nc1_pipeline.py:57-58）。numpy 2.4.4、pandas 3.0.2、h5py 3.16.0、PyYAML 6.0.1、transformers 5.5.3、tokenizers 0.22.2、scikit-learn 1.8.0、scipy 1.18.1、pytest 9.1.1、safetensors 0.8.0、huggingface_hub 1.33.0。
- **與 requirements.txt 的差異**：列出的 9 個套件版本全部一致。scipy 1.18.1 沒有列在 requirements.txt（MACHINE.md 有列），程式沒有直接 import scipy，是 scikit-learn 的相依套件。
- **MSI**：`~/.ssh/config` 沒有 MSI／WSL 的 host alias（只有 Pod_*、github、devops 和幾個 IP），依規定沒有連線。在 MSI 重建需要：(1) clone navrouter-cil @5751568；(2) 自備 `configs/machine_msi.yaml`（repo 內沒有；需要 machine、dataset_root_dir、device、threads 這幾個鍵）；(3) Python 3.12 + torch 2.11.x CUDA wheel + requirements.txt（requirements.txt:3-4）；(4) 複製 can_dataset 四個任務的 feats、datasplit、table；(5) 不需要 CONCH（cache/text 已 commit）；(6) 依 AGENTS.md 紅線 4，MSI 的數字必須從 nc1_pipeline 起在 MSI 上整鏈重跑，不能沿用 Mac 的 bank、lora_v2、i6 產物混表。

## 7. 報告產生後程式是否變動
- stage9（6ba9a16）→ HEAD：`git diff --stat 6ba9a16 HEAD -- scripts selector data configs requirements.txt` 只顯示**新增**的 `scripts/nc8_batch.py`（+124）與 `scripts/nc8_report.py`（+343）。nc7_i6、nc7_report 及其 import 的 nc2、nc5、nc6_report 與 selector/* 都沒有變動。
- stage10（5751568）= HEAD：diff 為空，工作樹乾淨。
- 結論：產出程式在報告 commit 之後**沒有變動**，重跑結果不會因程式變動而不同。不在版控內的輸入（can_dataset 特徵與 split、gitignored 的中間產物）沒有 hash 紀錄，無法用 git 證明它們沒變。

## 8. 未解問題
1. **十折切分的產生邏輯與 seed：未找到。** npz 只有三個 patient 名單鍵，沒有 seed；檔案 mtime 為 2026-05-20。已查：本 repo 全部 `*.py`；navipath、pathselect、mllm_hwsi grep `train_patients`（只找到讀取端：02_pathselect/scripts/audit_benchmark_protocol.py:47、mllm_hwsi_ah/nav/candata.py:73）。推測來自外部官方切分，依據是 audit_benchmark_protocol.py:2、:42 把 `path_split` 對照到外部官方設定，而且 tests/test_no_banned_deps.py:4-5 寫明對該外部方法「只能讀其切分／manifest 檔」；依本 repo 紅線，沒有進一步查該外部 repo。
2. **「群中心法」對應哪一個變體：推測為 R3(k=8)。** 程式與報告裡沒有「群中心」字樣，只有 R0–R5（selector/router.py:53）；stage10 除了 AR 以外，唯一使用 mean_vec 的分派器是 R3。
3. **nc7_report.py／nc8_report.py 的實際執行指令：沒有 log。** logs/ 裡只有 nc7_i6.log、nc8_batch.log。報告指令是從 docstring 推回來的，當初是否設定了 `PYTHONNOUSERSITE=1` 無法確認。
4. **MSI 實測值：沒有取得。** 沒有 ssh alias（已查 ~/.ssh/config 的全部 `Host` 行）。`192.168.10.107` 是否就是 MSI 無從判斷，沒有嘗試連線。
5. **「任務文字描述」：** I6／AR 路徑沒有使用任務層級的文字，只用亞型文字。organ_keys 只在 NC-1 使用。如果報告所說的「任務描述」另有所指，目前沒有找到。
