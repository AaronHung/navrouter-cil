# DECISION — MergeSlide 不跑（EXT-1，2026-10-03）

## 裁決

PREREG-21 補充 4 的停止條件成立：MergeSlide 微調與合併的是 TITAN 預訓練的 slide encoder，CONCH v1（512 維、無座標）沒有對應物（理由與逐行出處見同目錄 `PORT.md`）。**B2–B5、K8 不跑，沒有使用替代 backbone，主表的 MergeSlide 列沒有同折數字。** 本檔列出若要跑「做法 1」還缺什麼，以及做法 2–5 各自偏離原方法之處，供 PI 決定。

標「查」者是 2026-10-03 在本機實際查到的；標「估」者是未實測的粗估，假設寫在旁邊；標「背景」者是我記憶中的 TITAN／CONCH 公開資訊，本次沒有查證。

## 做法 1（照原方法）需要的每一項

| 項目 | 現況 | 缺口 |
|---|---|---|
| TITAN 權限 | 查：本機有 HF token；用它對 `MahmoodLab/TITAN` 的 `config.json` 做一次 HEAD 請求成功（6,372 bytes）。沒有下載權重，所以「可下載權重」沒有驗證 | 大概率在手；GPU 機上要另外放 token（或把權重帶過去） |
| 四 cohort 原始 WSI 的張數 | 查：我們的 CONCH 特徵檔數 = ESCA 158、RCC 937、BRCA 1,133、LUNG 1,054，共 **3,282 張**（`can_dataset/<task>/feats-l1-s256_CONCH/pt_files/`）。原始 WSI 要與這 3,282 張同一批，才能沿用我們的十折切分 | — |
| 原始 WSI 是否在手 | 查：本機只有 10 張 `.svs`（`~/research/WSI_data/slides/`，55 MB–1.8 GB，其餘是 openslide 示範檔），**不是四個 cohort 的 3,282 張**。RunPod 上的網路磁碟區放的是特徵（`/workspace/datasets/can_dataset`），沒有查到原始 WSI | 3,282 張都要重新取得（GDC 開放資料，需知道 slide ID，我們的切分表已有） |
| 原始 WSI 的容量 | 估：本機 10 張平均約 0.8 GB（總 8.1 GB）→ 3,282 張約 **2.6 TB**（樣本只有 10 張，誤差可到 ±30%）。查：本機可用空間 58 GiB（94% 已用），放不下 | 需要 ≥ 3 TB 的暫存空間（RunPod 網路磁碟區或外接硬碟），或逐張下載、抽完特徵就刪 |
| 特徵抽取：CONCH v1.5 | 查：本機沒有 CONCH v1.5（我們只有 v1，權重在 `01_navipath/checkpoints/conch/`）。背景：CONCH v1.5 隨 TITAN 的 HF repo 釋出，輸出 768 維 | 需用 TITAN 權限取得 v1.5 |
| 抽取設定 | 指令寫「10×」。背景：TITAN 論文的設定是 20×、512 × 512 的 patch，並需要每個 patch 的座標；我們現有特徵是 `l1-s256`（level 1、256 px），沒有座標。若 level 1 為 4 倍降採樣（40× 的底片 → 10×）：256 px @ 10× 與 512 px @ 20× 覆蓋的 level 0 面積相同（都是 1024 × 1024 px），patch 數量級相同。**這個等價成立與否取決於這批 TCGA 底片的底層倍率，沒有查證** | 抽取設定要照 TITAN 的規定，不能沿用現有特徵 |
| patch 總數 | 查：現有 CONCH v1 特徵共約 **10.25 M 個 patch**（四個 cohort 的特徵檔總容量 20.5 GB ÷ 每個 patch 2,048 bytes）。fold 1 test 每張平均 3,147 個 | 若面積等價成立，v1.5 約同量級 |
| 抽取的 GPU 時數 | 估：CONCH v1.5 為 ViT-L 級；單 patch 前向約 0.12 TFLOP（輸入 224）至 0.5 TFLOP（輸入 448）。A100 fp16 有效 100 TFLOPs、RTX 4090 約 70 TFLOPs 的假設下，10.25 M 個 patch 約需 **A100 約 3.5–14 小時、4090 約 5–20 小時**。**輸入解析度我不確定，所以範圍是 4 倍寬。** 另外底片讀取與切 patch 在 CPU 端，8 核每核 50 patch/s 時約 7 小時，通常與 GPU 重疊進行，實際牆鐘以較慢者為準 | 一台 ≥ 8 核的 GPU pod 一到數天；需先量一張 |
| 訓練與合併 | 未估。PORT.md (f)：每任務每折微調 1 個 epoch、每張隨機取 400 個 patch、bf16；4 任務 × 10 折 = 40 次微調，合併與評估另計。要用 CUDA（程式寫死 `cuda:0`、`GradScaler`） | B3 的計時要在 GPU 上做；Mac 上不適用 |
| 任務 prototype（`task_prompts.pt`） | 查：上游 repo 內有 6 × 768 的檔，但產生它的程式不在 repo 內；前四列（BRCA、RCC、NSCLC、ESCA）可沿用，列序依提示順序推定，未由程式確認 | 若要嚴格照原方法，需向作者確認產生方式 |
| 與我方資料的對齊 | 我方 LUNG 是 LUAD／LUSC（對應其 NSCLC）、RCC 在我方是 2 類（CCRCC／PRCC），而其 RCC 是 3 類（含 CHRCC）；任務順序與類別數要改成 2/2/2/2 | PORT.md 的四件事之 3 |

做法 1 的偏離：方法本身不偏離，但特徵是 CONCH v1.5（768 維）而不是我方的 CONCH v1（512 維），所以與 FINAL 的差異包含基礎模型的差異。PREREG-21 的移植規則第 1、2 項（維度 768 → 512、文字向量用我方 `cache/text/`）不適用，需另立 PREREG 修訂。

## 做法 2–5 各自偏離原方法之處

| # | 做法 | 偏離原方法之處 | 能否稱為 MergeSlide |
|---|---|---|---|
| 2 | backbone 換成隨機初始化的聚合器（如 ABMIL 或小型 Transformer），輸入我方 512 維特徵，文字 prototype 用 `cache/text/` | ① 沒有預訓練的 slide encoder，task vector 是相對於隨機初始值，各任務的初始值要共用同一份才有意義；② 初始時 slide 向量與文字不在同一空間（原方法靠預訓練的視覺－語言對齊）；③ 原方法 1 個 epoch、lr 1e-5 的預設對從零訓練幾乎一定不夠，要改超參數，而 PREREG-21 規定不調超參數；④ 保留的只有合併演算法（SVD 投影）與 task-to-class 推論的形式 | 否 |
| 3 | backbone = 全部 patch 平均 ＋ 一層 512 → 512 線性層，初始為恆等；文字 prototype 用 `cache/text/` | ① backbone 容量從整個 slide Transformer 變成一個矩陣；② 基準權重 = 恆等映射，等於 zero-shot 的 mean-pool，所以「共同基準且與文字對齊」這一性質保留；③ 沒有使用座標；④ 合併演算法可照用，但在單一矩陣上做 SVD 投影，行為與在多層 Transformer 上不同 | 否，只能稱為「其合併與推論規則在 CONCH 上的簡化版」 |
| 4 | 在 TITAN slide encoder 前加 512 → 768 的轉接層，輸入我方特徵 | ① TITAN 預訓練時看到的是 CONCH v1.5 的特徵分佈，轉接層另外要訓練，初始時等於餵給它雜訊；② 缺座標，TITAN 的位置編碼無法使用（要用假座標，偏離大）；③ 需要 TITAN 權限與 GPU | 否；偏離最大，不建議 |
| 5 | 不跑；主表維持發表值加表註 | 無偏離，但沒有同折數字。發表值是六任務（BRCA、RCC〔3 類〕、NSCLC、ESCA、TGCT、CESC）的 bACC，與我方四任務的 ACC 定義不同，只能放表註 | 引用而已 |

## 需要 PI 決定

1. 做法 1–5 選哪一個；選 1 需要先確認抽取的倍率與 patch 大小（上表「抽取設定」一列）與能取得 ≥ 3 TB 的暫存空間。
2. 選 1 時，PREREG-21 的移植規則第 1、2 項要由新的 PREREG 取代（特徵是 768 維、文字向量用 TITAN 的文字編碼器）。
3. 選 2 或 3 時，需明訂不調超參數下的訓練設定（epoch、lr），否則結果取決於我們的設定。

B4 的「go」仍須由 PI 另行下達；本檔不是 go。
