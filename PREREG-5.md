# PREREG-5 — NC-5 預先註冊（背景分派 ctx、機制診斷、對照組 AR／LIN8、主系統指標）

登記時間：2026-09-25，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`、`torch.set_num_threads(8)`；本輪所有數字來自同一台、同一批。
全部只做推論，不訓練任何 expert。

## 判準與設定（原文）

A. 背景分派 ctx（只推論；expert 用 L0，Hard，與第四、五關的 router 評估相同）
- 每個 patch 的腫瘤度 u，三種來源：
  z（通用文字）：腫瘤組 prompts = ["tumor", "carcinoma", "malignant tumor", "an H&E stained image of tumor", "an H&E stained image of carcinoma"]；非腫瘤組 = ["normal tissue", "benign tissue", "non-neoplastic tissue", "an H&E stained image of normal tissue", "an H&E stained image of benign tissue"]。兩組各走 encode_prompt_groups（與 scripts/nc1_organ_keys.py 同一條路徑）得 k_tumor、k_normal，存 cache/text/tumor_normal_keys.pt 並 commit。u = cos(x, k_tumor) − cos(x, k_normal)。與順序、階段無關。
  c（已見類別文字）：u = 對已見任務全部類別文字（f_txt）的最大 cosine。建任務 j 的 key 時，已見 = 該序中到任務 j 為止的任務；測試 t = 4 時為全部 4 個任務。
  e（已見 expert）：u = 已見任務的 L0 expert 對該 patch 的 one-shot 分數（text_nav_feats 用該 expert 自己任務的 2 類文字），每個 expert 的分數先在該 slide 內做 z-score，再取最大。已見集合同 c。
- 背景向量：slide 內依 u 由小到大排序，取最小的 ceil(q·n) 個 patch，平均後 L2 正規化；q ∈ {0.25, 0.5, 0.75}。
- 腫瘤半向量（只作機制診斷）：依 u 最大的 ceil(0.5·n) 個 patch，平均後正規化。
- key：每任務對其 train slides 的背景向量做 KMeans(k=8, n_init=10, random_state=0)，centroid 正規化（與 R3 同規格，16,384 bytes／任務）。測試：slide 背景向量對每個已見任務所有 key 的最大 cosine，取最大者為分派任務。
- 候選：基準 R3(k=8)（全部 patch 的 mean_vec，沿用既有 key）＋ {z, c, e} × {0.25, 0.5, 0.75}，共 10 個。
- 選法：十折 validation 的 t = 4 Hard CIL 平均。c、e 與順序有關，取兩序平均；z 與基準兩序相同。同分依「基準、z、c、e；同來源 q 由大到小」取前者。
- ctx-pass：選出的不是基準；兩序各自 test t = 4 CIL 十折平均 − 基準 ≥ +0.01，且差 > 0 的折數 ≥ 7；每任務儲存 ≤ 16,384 bytes（z 另有兩個共用文字向量 4,096 bytes，揭露，不計入）。

B. 機制診斷（不設門檻，必報；test，十折）
- 每個候選：TP micro、macro、每任務 TP、ESCA→Lung 與 Lung→ESCA 張數（十折合計）、t = 4 CIL。
- 基準的 Lung→ESCA 錯誤中，真實類別 LUAD／LUSC 各幾張；ESCA→Lung 錯誤中，ESAD／ESCC 各幾張。對選出的候選也報同一件事，並報相對基準修好／弄壞的張數。
- 腫瘤半向量當 router（三種來源，key 規格同上）：TP micro 與兩個方向的混淆張數。
- 真實任務的 L0 expert 四輪選中的 64 個 patch，有多少比例落在 q = 0.5 的背景集合（三種來源，十折平均）。

C. 兩個對照組（不設門檻，必報；不參與 ctx-pass）
- AR 解析式 router（Any-SSR 式）：x = slide 的 mean_vec 後面接常數 1（513 維）。依該序每學一個任務 j：A += Xjᵀ Xj（A 從 0 開始），bj = Xjᵀ 1（Xj = 任務 j 全部 train slides，1 = 全 1 向量）。第 t 階段 W = (A + γI)⁻¹ [b1 … bt]，測試分派 = argmax(x W)。γ ∈ {0.01, 0.1, 1, 10}，以十折 validation 平均 t = 4 CIL 選。儲存：A 共用 513 × 513 × 4 bytes，每任務 bj 2,052 bytes。檢查（fold 1）：t = 4 的 W 與「四任務 train 一次解」的 W 最大絕對差 < 1e-4。報 TP（同 B 的欄位）與 CIL（L0 expert、Hard）。
- 若 A 選出的不是基準：再報 AR 改用選出的背景向量（同來源、同 q，A 與 bj 用背景向量重算）的 TP 與 CIL，組成 2 × 2 表（router：R3／AR × 輸入：全部平均／背景）。
- LIN8 8 類線性基線：同 AR 的累加法，目標改為 8 類 one-hot（每任務 2 欄，依該序的已見類別），輸入 mean_vec 接常數 1；γ 同樣四選一，以十折 validation 平均 t = 4 ACC 選。不用 expert、不用文字頭。報：t = 4 ACC、Masked ACC（真實任務的 2 類內 argmax）、兩序的 Forgetting（每階段只在已見類別中 argmax）；並與目前完整系統（R3(k=8)＋L1(r=2) v2，兩序）逐折相減，報十折平均差與贏折數。

D. 補齊主系統指標（不設門檻，必報）
- 主系統 = R3(k=8)＋L1(r=2) v2，Hard，兩序：每學完一個任務（t = 1、2、3、4）的 ACC 與 Masked ACC（十折 mean ± sd）；BWT = 平均_{j<4}(R[4][j] − R[j][j])，並檢查每折是否 BWT = −Forgetting。

操作細節：
- 每張 slide 每折只讀一次，同時算出所有來源、所有 q、腫瘤半、以及 u 的統計；train slides 的 c、e 依兩個順序各算一次（已見集合不同）。快取放 outputs/navcil/mac/cache/nc5_*，加 .done。
- 其餘設定（K=64、每輪 16、λ*=1.5、Hard 的定義、Masked ACC 的算法）與 PREREG-3、PREREG-4 相同。

## 操作定義（與判準同時登記）

1. **各階段的已見集合**：test 的第 t 階段（Forgetting 需要 t = 1…4），c、e 的背景向量與腫瘤半向量
   以該序前 t 個任務為已見集合計算；t = 4 為全部 4 個任務。validation 只用 t = 4。z 與階段無關。
   任務 j 的 key 用該序中到任務 j 為止的已見集合計算其 train slides 的向量（c、e 因此兩序各一套 key）。
2. **e 的 z-score**：同一 slide 內 (s − mean(s)) / (std(s) + 1e-6)（std 為樣本標準差），與 E-max 相同。
3. **排序**：最小 ceil(q·n) 個以 `torch.topk(u, k, largest=False)` 取、最大 ceil(0.5·n) 個以
   `torch.topk(u, k)` 取；同值順序由 topk 決定。
4. **CIL 與 TP**：沿用 PREREG-3 操作定義 9、10（Hard、分派後 expert 的證據、Forgetting 定義）。
   c、e 與順序有關，TP、混淆張數與 CIL 兩序分列；z、基準在 t = 4 兩序相同。
5. **修好／弄壞**：test t = 4、十折合計，以 TP（分派任務是否等於真實任務）計，另報以 CIL（最終類別
   是否正確）計的同一數字；選出者為基準時不報。
6. **背景集合比例**：test t = 4，真實任務的 L0 expert 四輪選中的 64 個 patch（第一關 test 快取的
   four_idx）中，落在該 slide q = 0.5 背景集合（c、e 以全部 4 個任務為已見集合）的比例；逐張算，
   每折平均後再十折平均。
7. **u 的統計**：test t = 4，每張 slide 的 u 平均與標準差（三種來源），報十折平均。
8. **AR／LIN8 數值**：依 AGENTS.md 可攜規則 3 一律 float32；A、bj 依該序逐任務累加，
   W = `torch.linalg.solve(A + γI, B)`。fold 1 的一致性檢查以選出的 γ 判定（其餘 γ 一併報告），
   未達 < 1e-4 即停下回報。γ 同分取小。LIN8 的第 t 階段只在已見任務的 2t 個類別欄中取 argmax。
9. **LIN8 與完整系統**：兩序各自逐折計算 t = 4 ACC 之差（LIN8 − 完整系統）；報十折平均差與
   差 > 0 的折數。完整系統的數字由本輪同一批重算（R3(k=8) key、L1(r=2) v2 證據）。
10. **主系統逐階段**：ACC_t = 平均_{j≤t} R[t][j]；Masked ACC_t = 平均_{j≤t}（第 t 階段、任務 j 的
    Masked ACC）。BWT 以 1 起算的 j = 1、2、3。BWT = −Forgetting 的檢查容差 1e-9。
11. **儲存揭露**：ctx 與腫瘤半 key 16,384 bytes／任務；z 的共用文字向量 2 × 512 × 4 = 4,096 bytes；
    AR 的 A 共用 1,052,676 bytes、每任務 2,052 bytes；LIN8 的 A 共用、每任務 2 × 2,052 bytes。
