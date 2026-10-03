# PORT.md — MergeSlide 移植到 navrouter-cil 設定的可行性（EXT-1 B1）

日期：2026-10-03。上游：https://github.com/caodoanh2001/MergeSlide @ `96e7d67693c37bee518b61760c0af5b57b00f865`（clone 於 `~/research/03_mergeslide`，未改動任何檔案）。
依據：PREREG-21（navrouter-cil @ 277339b）的 MergeSlide 移植規則與補充 3、4。

## 結論

**B 停在 B1（補充 4 的停止條件成立）。** MergeSlide 微調與合併的對象是 TITAN 預訓練的 slide encoder（`titan_model.vision_encoder`）；
它吃 CONCH v1.5 的 768 維 patch 特徵加座標。我們的特徵是 CONCH v1 的 512 維、沒有座標，CONCH v1 也沒有對應的預訓練 slide encoder。
「特徵維度 768 → 512」因此不是改一個數字就能成立的改動。沒有做 B2、B3，沒有估時，沒有用任何替代 backbone。

下列凡標「程式」者為讀上游程式所得（附檔名與行號）；標「背景」者為 TITAN／CONCH 的一般知識，本次沒有另外查證。

## 輸入格式

- 每張 slide 一個 `h5_files/<slide_id>.h5`，內含 `features`（[N, 768]）與 `coords`（[N, 2]）；`features` 讀不到時改讀 `pt_files/<slide_id>.pt`，座標仍取自 h5（程式：`datasets.py:430-436`、`:484-486`）。
- 特徵由 TITAN 的 patch encoder（CONCH v1.5）抽取（程式：`README.md:39`；`WSI_processing.ipynb` 第 256、276、306 行「use vision encoder of TITAN (CONCHv1.5)」、`--model "conch15"`）。
- 前向呼叫是 `model.backbone(features, coords, 1024)`，第三個參數是 level 0 的 patch 大小（程式：`train_random_sampling.py:189`）。座標是必要輸入。
- 切分：每任務一個 `splits_{fold}.csv`，欄位 train／val／test（另一類資料集多 `*_label` 欄）（程式：`datasets.py:457-463`、`:563`）。
- 我們的資料：`can_dataset/<task>/feats-l1-s256_CONCH/pt_files/*.pt`，單一 tensor，實測 `(7060, 512)` float32；`can_dataset` 下沒有任何 `.h5`，沒有座標。本機 `WSI_data/slides` 只有 12 個資料夾，沒有四個 cohort 的原始 WSI。

## (a) 微調的 backbone 與初始權重

- backbone = `model.vision_encoder`，其中 `model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)`（程式：`train_random_sampling.py:54-63`、`:329`）。即 TITAN 的 slide encoder（README 稱 slide aggregator，`README.md:69`）。
- 初始權重 = TITAN 的預訓練權重，每個任務都從同一份預訓練權重重新載入後微調（程式：`:326-330`，迴圈內每個任務重新 `from_pretrained`）。不是從零開始。HuggingFace 上需要申請權限（`README.md:71`）。
- 分類頭是一個 `nn.Linear(768, 類別數)`，權重設為文字 prototype、bias 0，並凍結；只有 backbone 被訓練（程式：`:333-343`）。
- **CONCH 設定下用什麼**：沒有對應物。CONCH v1 只有 patch 層級的影像／文字編碼器，沒有預訓練的 slide encoder（背景）。TITAN 的 slide encoder 是在 CONCH v1.5 的 768 維特徵上預訓練的，不能直接吃 CONCH v1 的 512 維特徵（維度與特徵分佈都不同）。
- 這正是補充 4 寫的情況：「backbone 必須是 TITAN 預訓練的 slide encoder，CONCH 沒有對應物」。

## (b) 合併的對象與基準權重

- 合併對象：`vision_encoder` 的**全部參數**（各任務 checkpoint 去掉最後兩個 key，即 MLP 的 weight 與 bias）（程式：`opcm_mergeslide.py:107`、`:112`）。
- 基準權重：TITAN 預訓練的 `vision_encoder`（程式：`:95`、`:102`、`:148`、`:158`、`:168`）。task vector = 微調後權重 − 預訓練權重（`:38-39`、`:68-69`）。
- 做法（依學習順序逐一併入）：
  - 起點 = 第一個任務的微調權重（`:116-117`）。
  - `nn.Linear` 的 weight：對「目前已合併的 task vector」做 SVD，把新任務的 task vector 投影到該基底、對角線歸零後轉回，再相加並除以 λ_t（`:23-52`、`:145-154`）。
  - 其餘參數（bias、非 Linear 模組的參數）：task vector 直接相加後除以 λ_t（`:54-74`、`:155-175`）。
  - 每併入一個任務後，把合併後的 task vector 範數縮放到各任務 task vector 範數的平均（`:177-184`）。
- 每個階段存一份合併後的 backbone（`:186`）。每任務另存的東西：凍結的 MLP（= 該任務的文字 prototype），評估時從各任務的微調 checkpoint 讀出（`test_classIL_task_prompt.py:178`）。
- `alpha = 0.5`（`:120`）傳入 `merge_linear_weights` 但函式內沒有使用。
- **CONCH 設定下**：沒有共同的預訓練基準，task vector 無從定義。

## (c) task-to-class 推論的文字向量

- 類別 prototype：`titan_model.zero_shot_classifier(CLASS_PROMPTS, TEMPLATES)`，用 TITAN 的文字編碼器，對每類多個同義名 × 模板取 ensemble（程式：`train_random_sampling.py:28-37`；提示在 `prompts_zeroshot.py`，每類 5 個同義名、每任務一組模板）。
- 任務 prototype：直接讀 repo 內的 `task_prompts.pt`（程式：`test_classIL_task_prompt.py:288`；`test_classIL_task_prompt_other_metrics.py:253`）。實測為 `[6, 768]` float32，各列範數 0.89–0.95。
  **產生它的程式不在 repo 內**（全 repo 只有載入它的兩處）。範數小於 1，與「多個單位向量取平均」相符，但這是推測，無法由程式確認。
- 推論：`predicted_task_id = argmax(slide_embed @ task_prompts.T)`，再用該任務的 MLP 在其類別內 argmax（`test_classIL_task_prompt.py:195-207`）。
- **CONCH 設定下改用哪個檔**：
  - 類別 prototype → `cache/text/f_txt_tcga_{esca,rcc,brca,lung}.pt`（每檔 2 × 512）。這一項對得上。
  - 任務 prototype → `cache/text/` 裡**沒有以同樣方式產生的檔**，因為上游的產生方式未公開。可選的只有：每任務兩列 `f_txt` 的平均（自行推導），或 `organ_keys.pt`（NC-1 的器官文字 key，提示設計不同）。兩者都是我們自己的選擇，不是上游的定義。

## (d) slide 向量與文字是否同一空間

- 上游：是。TITAN 的 slide encoder 與文字編碼器經過視覺－語言對齊預訓練（背景），所以 `slide_embed @ prototype.T` 在微調前就有意義；凍結的文字 prototype 當分類頭也因此成立（程式：`train_random_sampling.py:336-343`）。
- CONCH 設定下：CONCH 對齊的是 **patch** 影像向量與文字。patch 向量的平均仍落在同一空間（我們的 zero-shot 就是這樣做），但那不是一個可微調、有預訓練權重的 slide encoder。
  任何新加的可訓練聚合器在初始化時都沒有與文字對齊，除非刻意把它初始化成恆等映射（見「可能做法」第 3 項）。

## (e) 指標定義與我們的差異

| 指標 | 上游定義（程式） | 我們的定義 | 差異 |
|---|---|---|---|
| bACC（主結果） | 每任務 `balanced_accuracy_score(任務內標籤, 預測的任務內索引)`，再對任務平均（`test_classIL_task_prompt.py:369`、`:387`） | ACC：8 類標籤相等的比例，每任務計後四任務等權 | 上游是 balanced、我們是一般正確率；另見下一列 |
| class-IL 的「正確」 | 比的是**任務內索引**：預測取自「預測任務」的 MLP，標籤是「真實任務」的任務內標籤，兩者直接比（`:207`、`:354`）；沒有檢查預測任務是否等於真實任務 | 必須 8 類全域標籤相等（任務判錯就算錯） | 上游在任務判錯、但任務內索引剛好相同時會算對，數字不低於我們的定義 |
| ACC | 每任務一般正確率的平均（`:371`、`:388`） | 同上（四任務等權） | 同上一列的索引問題 |
| Masked bACC | task-IL：給定真實任務，用該任務的 MLP 在合併後的 backbone 上判，取 bACC 的任務平均（`test_taskIL.py:237-258`） | Table 1 定義：τ̂ 的 expert 證據、在真實任務兩類內 argmax；oracle 定義：告訴任務 | 上游的 Masked 等於我們的 oracle 定義（它沒有 per-task expert，兩種定義在它身上相同），但用 bACC |
| Forgetting | 每任務取各階段一般正確率的最大值（未學階段補 0）減最終值，對前 T−1 個任務平均（`test_classIL_task_prompt_other_metrics.py:164-174`） | `nc5_report.cil_full`：max 取自階段 j…T−1（不含最終階段）減最終值，前 T−1 個任務平均 | 上游的 max 含最終階段，所以每項 ≥ 0；我們的不含。用的都是一般正確率（`README.md:140`） |
| BWT | 最終值 − 剛學完的值，前 T−1 個任務平均（`:176-182`） | 相同 | 無 |
| mACC | 各階段「全部已學任務的張數合計正確率」的平均（`:295-306`） | Ā：各階段四任務等權 ACC 的平均 | 上游依張數加權，我們依任務等權 |

另：上游在訓練與評估時每張 slide 隨機取 K 個 patch（訓練 400、class-IL 評估 300、另一支評估 400），所以評估結果與亂數種子有關（`train_random_sampling.py:170`、`:183`；`test_classIL_task_prompt.py:166`、`:189`；`test_classIL_task_prompt_other_metrics.py:188`、`:195`）。

## (f) 預設超參數（出處行號）

| 項目 | 值 | 出處 |
|---|---|---|
| 實際訓練 epoch 數 | **1**（迴圈寫死 `range(1)`） | `train_random_sampling.py:171` |
| `--num_epochs` | 10（只用來算學習率排程的總步數） | `:315`、`:157-162` |
| 學習率 | 1e-5 | `:316` |
| 排程 | 線性 warmup（總步數的 10%）接 cosine；因為只跑 1 個 epoch 而排程以 10 個 epoch 計，整個訓練都在 warmup 內 | `:81-104`、`:157-162` |
| weight decay | 1e-4（bias、norm、一維參數為 0） | `:317`、`:143-156` |
| optimizer | AdamW | `:156` |
| batch size | 1 | `:313`；`datasets.py:564` |
| 每張 slide 取的 patch 數 | 訓練 400（隨機）；class-IL 評估 300；另一支評估 400 | `:170`、`:183`；`test_classIL_task_prompt.py:166` |
| 混合精度 | bfloat16 autocast ＋ GradScaler（CUDA） | `:167`、`:188` |
| loss | CrossEntropy | `:163` |
| validation／early stopping | patience 2，但只在 `epoch > 1` 時跑 validation；只有 1 個 epoch，所以**從未執行** | `:169`、`:208` |
| 分類頭 | 凍結的文字 prototype，bias 0 | `:333-343` |
| seed | `torch.manual_seed(42)`、`seed_torch(device, 0)` | `:10`、`:308` |
| 任務數／類別數 | 6；`[2, 3, 2, 2, 2, 2]`；類別索引範圍 `dict_classes` | `:322-323`、`:39-46` |
| 任務迴圈 | `range(3)`（公開版只跑前三個任務） | `:326` |
| 裝置 | 寫死 `cuda:0` | `:24` |
| 合併 | `alpha = 0.5`（未使用）、`previous_lambda_t = 1` 起始 | `opcm_mergeslide.py:120-121` |
| 折數 | 10（`range(0, 10)`） | `train_random_sampling.py:320`；`opcm_mergeslide.py:96` |

## PREREG-21 的四件事各改在哪裡

| 改動 | 檔案與行 | 能否成立 |
|---|---|---|
| 1. 特徵維度 768 → 512 | 分類頭 `nn.Linear(768, …)`：`train_random_sampling.py:333`、`test_classIL_task_prompt.py:202`、`test_classIL_task_prompt_other_metrics.py:203`、`test_taskIL.py:237` | **不能**。768 同時是 TITAN slide encoder 的輸入與輸出維度，寫在預訓練權重裡；改分類頭的數字不會讓 backbone 接受 512 維輸入 |
| 2. 文字向量改用 `cache/text/` | 類別 prototype：`train_random_sampling.py:25-37`（及兩支評估程式的同一段）；任務 prototype：`test_classIL_task_prompt.py:288`、`…_other_metrics.py:253` | 類別 prototype 可以；任務 prototype 沒有對應檔（見 (c)） |
| 3. 任務數 6 → 4 與類別定義 | `train_random_sampling.py:23`、`:32`、`:39-46`、`:322-323`、`:326`；`test_classIL_task_prompt.py:272-281`；`…_other_metrics.py:189`、`:244-248`；`opcm_mergeslide.py --num_tasks`；`datasets.py:552-559` | 可以 |
| 4. 切分改用我們的十折 | `datasets.py:552-563`（資料集清單、`split_dirs`、`splits_{fold}.csv`） | 可以（需把 npz 的 patient 清單展開成 slide 清單的 csv） |

另有與四件事無關、但在 Mac 上必須處理的地方：裝置寫死 `cuda:0`（`:24`）、`torch.cuda.amp` 的 autocast 與 GradScaler（`:167`、`:188`）、DataLoader `num_workers=4`。這些屬於執行環境，不屬於演算法。

## 可能做法與各自偏離原方法的地方（等 PI 決定；本批都沒有做）

| # | 做法 | 需要什麼 | 偏離原方法的地方 | 與 FINAL 的可比性 |
|---|---|---|---|---|
| 1 | 照原方法跑：為四個 cohort 取得 CONCH v1.5 的 768 維特徵與座標，用 TITAN slide encoder，只改任務數與切分 | TITAN 的 HuggingFace 權限；四個 cohort 的 CONCH v1.5 特徵（從原始 WSI 重抽，或用作者釋出的預處理特徵，README 該連結目前寫「Updating」）；GPU | 方法本身不偏離；移植規則的第 1、2 項不適用（維度維持 768、文字用 TITAN） | 同折、同 slide，但 patch 特徵不同（CONCH v1.5 對 CONCH v1），差異包含基礎模型的差異 |
| 2 | 把 backbone 換成隨機初始化的聚合器（例如 ABMIL 或小型 Transformer），輸入 CONCH 512 維，文字 prototype 用 `cache/text/` | 只需現有特徵；Mac 可跑 | 沒有預訓練的 slide encoder；task vector 相對於隨機初始值；初始時 slide 向量與文字不對齊；1 個 epoch、lr 1e-5 的預設對從零訓練很可能不夠。合併演算法保留，其餘已不是 MergeSlide | 同折、同特徵；但對照的是「我們改寫的方法」，不能稱為 MergeSlide |
| 3 | backbone 換成「patch 平均 ＋ 一層 512 → 512 線性層，初始為恆等」，文字 prototype 用 `cache/text/` | 只需現有特徵；Mac 可跑 | 保留「有共同基準、且基準與文字對齊」這個性質（基準即 mean-pool zero-shot），合併演算法可照用；但 backbone 的容量從整個 slide Transformer 變成一個矩陣 | 同折、同特徵；同樣不能稱為 MergeSlide，只能稱為其合併與推論規則在 CONCH 上的簡化版 |
| 4 | 在 TITAN slide encoder 前加 512 → 768 的轉接層，輸入 CONCH v1 特徵 | TITAN 權限、GPU、座標 | 預訓練權重看到的是分佈不同的輸入；轉接層需要另外訓練；缺座標 | 不建議；偏離最大且無座標可用 |
| 5 | 不跑；主表維持發表值加表註 | 無 | 無 | 非同折；發表值是六任務、bACC，與我們的四任務 ACC 定義不同 |

任務 prototype（見 (c)）在做法 1 以外都需要 PI 指定來源。做法 1 若只取四個任務，`task_prompts.pt` 的對應列可直接沿用（列序 BRCA、RCC、NSCLC、ESCA、TGCT、CESC，依 `train_random_sampling.py:32` 的提示順序推定，未能由程式確認）。
