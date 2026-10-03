# AUDIT — WP（第二站：residual scoring head I6 ＋ 證據選取 ＋ 凍結 CONCH 頭）稽核

只讀稽核（MOE-0 A 段）：不訓練、不改程式；A9 的實例以 `scripts/moe0_a9.py`（新檔，只推論）算出。
基準 commit：main @ 70d20d1（分支 moe-0）。「主系統」= AR（γ = 1e-3）＋ I6(r = 2)＝ 9/30 Table 1 第 6 列（REPORT_stage10.md:359，ACC 0.9128）。
行號皆為本分支檔案的行號。標「不確定」者是程式與文件都查不到答案。

---

## A1 head 的輸入 514 維

來源：`selector/i6_expert.py:39`（`u = cat([Z, tf])`）、`selector/flat_selector.py:19-33`（`text_nav_feats`）。

| 維度 | 量 | 計算 | 正規化 |
|---|---|---|---|
| 1–512 | Z：CONCH patch 特徵（凍結） | 直接取特徵檔 `feats-l1-s256_CONCH/pt_files/*.pt` 的 [n, 512]（`selector/evaluate.py:49-54`，`.float()`） | **沒有正規化**：實例 slide 的 ‖z‖ 為 24.55／25.24／25.80（最小／中位／最大） |
| 513 | 最大文字相似度 tf[:,0] | `z = normalize(Z)`、`t = normalize(f_txt)`、`txt = z @ tᵀ`（[n, 2]，該任務 2 個亞型文字），取 `amax(-1)`（`flat_selector.py:28-30、33`） | 兩邊 L2 正規化的 cosine；**沒有**做 slide 內 z-score（z-score 只用在 s0，見 A2） |
| 514 | 文字相似度分佈的 entropy tf[:,1] | `softmax(txt, -1)`（**無 temperature**）後 `-Σ p log(p + 1e-9)`（`flat_selector.py:31-32`） | 無；只有 2 類，所以值域是 [0, ln 2]；實例 slide 的範圍 0.6732–0.6931（ln 2 = 0.6931），幾乎是常數 |

沒有其他正規化（沒有 layer norm、沒有對 u 做標準化）。

## A2 zero-shot 分數 s0

- 定義：`selector/i6_expert.py:35-38`：`tf = text_nav_feats(Z, f_task)`，`s0 = zscore(tf[:, 0])`；`zscore(s) = (s − mean)/(std + 1e-6)`（`i6_expert.py:21-22`，std 為樣本標準差，在**該張 slide 的全部 patch** 上算）。
- patch 與兩個亞型文字的相似度如何變成一個分數：patch 對該任務**兩個**亞型文字各算 cosine，取較大者（`amax`，`flat_selector.py:30、33`）；沒有 softmax、沒有 temperature；兩類之間不做相減。
- prompt：`data/class_prompts.json`（`classnames`、`templates`），`selector/text_encoder.py:75-97` `class_prompt_ensemble`：每個亞型的 prompt = 每個同義名 × **22 個模板**（`CLASSNAME` 置換，句尾補 `.`，`text_encoder.py:94`）。
  各類 prompt 數：LUAD 66、LUSC 66（各 3 個同義名）；ESAD 66、ESCC 66（各 3）；CCRCC 88、PRCC 88（各 4）；IDC 110、ILC 110（各 5）。
  22 個模板：`CLASSNAME`、`a photomicrograph showing/of CLASSNAME`、`an image of/showing CLASSNAME`、`an example of CLASSNAME`、`CLASSNAME is shown`、`this is CLASSNAME`、`there is CLASSNAME`、`a histopathological image showing/of CLASSNAME`、`a histopathological photograph showing/of CLASSNAME`、`shows CLASSNAME`、`presence of CLASSNAME`、`CLASSNAME is present`、`an H&E stained image of/showing CLASSNAME`、`an H&E image showing/of CLASSNAME`、`CLASSNAME, H&E stain`、`CLASSNAME, H&E`。
- 合併：每條 prompt 經 CONCH text tower 編碼並 L2 正規化（`text_encoder.py:100-107`），同一亞型的所有 prompt 取平均再 L2 正規化（`text_encoder.py:170`），得 `f_txt` [2, 512]；快取於 `cache/text/f_txt_<task>.pt`，執行時直接讀（`text_encoder.py:149-175`），不載入 CONCH 權重。
- s0 與 head 的輸入 u 用的是**同一份** `f_txt`（該任務的 2 類，`nc1_pipeline.py:75-76` `f_task`）。

## A3 head 的結構與參數量

`selector/i6_expert.py:25-52`（r = 2）：

| 層 | 形狀 | bias | activation | 初始化（行） |
|---|---|---|---|---|
| A（514 → r） | 2 × 514 = 1,028 | b1 [2]（`:31`） | GELU（`F.gelu`，預設 exact／erf 版；`:44`） | A：`kaiming_uniform_(a=√5)`（`:29-30`）；b1 = 0（`:31`） |
| w2（r → 1） | [2]，逐元素乘加 `(GELU(·) * w2).sum(-1)`（`:44`） | b2 標量 [1]（`:33`） | 無 | w2 = 0（`:32`）；b2 = 0（`:33`） |

- 1,033 的算式：A 2×514 = 1,028；b1 2；w2 2；b2 1；1,028 + 2 + 2 + 1 = 1,033 = 516r + 1（r = 2；`i6_expert.py:51-52` `n_params`，`scripts/nc7_report.py:66`）。
- 初始 w2 = 0、b2 = 0 → 初始 g ≡ 0，score = s0（`i6_expert.py:6` docstring）。r = 2 時 A u 走 gemm（`:43`），r = 1 走逐元素乘加（`:40-41`）。
- b2 的觀察：b2 是加在 g（因此加在每個 patch 的分數 s）上的常數；softmax（訓練時的權重）、top-k 的排序、四輪選片前的 z-score（`multiround.py:146-150`）對常數平移都不變，所以 b2 不影響任何輸出。40 個權重檔（10 折 × 4 任務）的 |b2| 最大值 3.5e-4（w2 的 |值| 最大 0.28）。b2 為何不是 0（梯度理論上為 0）：**不確定**，沒有量測。
- 所有 (fold, task) 的訓練都先 `torch.manual_seed(42)` 再建模型（`scripts/nc7_i6.py:49-50`），所以 A 的初始值在 40 次訓練中相同（由程式推得，沒有逐檔驗證）。

## A4 patch 最終分數與選出的 64 個

- 最終分數 s = s0 + g（`i6_expert.py:47-49` `forward`；推論呼叫點 `scripts/nc8_batch.py:72` 的 `m(Z, ctx.f_task(p))`，每張 slide、每個 expert 只算一次）。
- 主系統是**四輪各 16**，不是一次取 64：`selector/cil_ops.py:39-45` `four_round(Z, base, lam, budget=64, step=16)`（`BUDGET = 64`、`STEP = 16`，`cil_ops.py:22-23`），λ\* = 1.5（`outputs/navcil/mac/nc1/lambda.json`），`redundancy_mode="maxsim"`、`normalize_base=True`。
- 流程（`selector/multiround.py:134-203`）：
  1. 先把 s 在 slide 內 z-score（`:146-150`，std ≤ 1e-6 時不做）→ `norm_base`；`max_sim_seen` 初始為 0（`:152`）。
  2. 每輪：`adj = norm_base − λ · max_sim_seen`（`:157-163`；第 1 輪 `state.seen` 為空，不扣）；已選者設 −∞（`:164`）；取 `adj` 前 16（`:166-169`）。
  3. 每輪後更新 `max_sim_seen = max(max_sim_seen, 每個 patch 對本輪新選 16 個的最大 cosine)`（`:177-180`，cosine 用 L2 正規化的 Z，`:139`）。
  4. 4 輪 × 16 = 64；若 patch 數 n < 64，budget = n（`:138`）。
- 輪與輪之間有狀態更新：有，狀態 = 已選集合（`seen_mask`）與 `max_sim_seen`（patch 對已選者的最大 cosine）；**分數 s 本身不重算**，head 不再被呼叫（REPORT_stage11.md:212 同此描述）。
- λ\* = 1.5 的來源：fold 1 validation、用第一關 L0 selectors 在 λ ∈ {0, 0.5, 1, 1.5, 2} 上選的（`scripts/nc1_pipeline.py:263-284` `choose_lambda`），之後十折固定，也套用在 I6；I6 本身沒有重選 λ。
- 訓練時**不是**四輪：訓練用 `top_k_select(s.detach(), 64)` 一次取 64（`cil_ops.py:88`），沒有冗餘扣分（見 A6）。

## A5 slide 向量與 logits

- 彙整：等權平均。`selector/cil_ops.py:31-36` `mean_norm(Z, idx)`：`Z.index_select(0, idx).mean(0)`（**未正規化的原始 Z** 的算術平均），再 L2 正規化 → [512]（`w=None`）。主系統呼叫：`scripts/nc8_batch.py:63`。
- 與文字比較：`slide_vec @ ctx.F.t()`（`nc8_batch.py:63`；`ctx.F` = 四個任務的 `f_txt` 串成 [8, 512]，`nc1_pipeline.py:62`）→ 8 個 cosine。WP 在該任務 2 類（第 2p、2p+1 列）內取 argmax（`scripts/nc2_report.py:89-96` `hard`）。
- **推論沒有 temperature、沒有 logit scale**：直接比 cosine（2 類 argmax 與正的 scale 無關）。
- 訓練才用 logit scale：`selector/classifier.py:40-41` `conch_classify`：`logit_scale · normalize(Σ w_i z_i) · f_txtᵀ`，`logit_scale` 取自 CONCH（`cache/text/f_txt_*.pt` 內，值 56.3477；`nc1_pipeline.py:63` `ctx.ls`）。

## A6 訓練

- loss（`selector/cil_ops.py:87-91`，每步 1 張 slide）：
  `s = s0 + g`；`idx = top_k_select(s.detach(), 64)`；`w = softmax(s[idx])`（在 64 個選中者上）；
  `logits = 56.3477 · normalize(Σ_i w_i z_i) · f_task[2×512]ᵀ`（`classifier.py:40-41`）；
  `loss = CE(logits, y)`，y = 該任務內的 0／1（`nc7_i6.py:57` `rec.label − shift`），只用該任務自己的 2 類文字。
- top-K 不可微，梯度走哪條路：選取 `idx` 用 `s.detach()`（無梯度）；梯度只經 **softmax 權重 w = softmax(s[idx])** 回到 s[idx] = s0[idx] + g[idx]，再到 g 的參數（A、b1、w2、b2）。s0 無參數。沒被選進 top-64 的 patch 不貢獻任何梯度；選取本身（誰進前 64）沒有梯度訊號。w2 初始為 0，所以第一步 A 的梯度為 0（g 對 A 的導數乘 w2），w2 先動。
- 訓練與推論的彙整方式**不相同**：

  | | 選片 | 彙整 | 分類 |
  |---|---|---|---|
  | 訓練（`cil_ops.py:87-91`） | 一次 top-64（無冗餘扣分） | softmax(s[idx]) 加權 | `logit_scale × cosine` 後 CE |
  | 推論（`nc8_batch.py:63`） | 四輪 16，λ = 1.5 的 MMR 扣分 | 等權平均 | 8 類 cosine，無 logit scale |

  （REPORT_stage13.md:15、PREREG-11.md:29 已寫出訓練用 softmax 加權；推論用等權見 PREREG-8.md:37。）
- optimizer／超參數：`torch.optim.Adam`（`cil_ops.py:78-79`），lr 5e-4、weight_decay 1e-4（Adam 的耦合 L2）、5 epochs、batch = 1 張 slide、`SEED, EPOCHS, LR, WD = 42, 5, 5e-4, 1e-4`（`nc1_pipeline.py:40`）。
- 每個 epoch 的 slide 順序：`torch.randperm(len(ds), generator=Generator().manual_seed(42 + ep))`（`nc7_i6.py:53-57`、`cil_ops.py:84`）。
- seed：42（`nc7_i6.py:49` 建模型前、`cil_ops.py:76` 訓練開始時各 `manual_seed(42)`）。
- 資料與挑 checkpoint：訓練只用該折該任務的 **train split**（`nc7_i6.py:48`）。**沒有**挑 checkpoint：存的是第 5 個 epoch 後的權重（`nc7_i6.py:63`），沒有用 validation 或 test 選 epoch。validation 只用在兩處：選 r（r\* = 2，PREREG-7 操作定義 4）與選 λ\*（fold 1，用 L0，見 A4）。
- 每步檢查梯度與參數有限（`cil_ops.py:94、96`）。

## A7 推論時從 TP 輸出到 8 類標籤

1. TP 輸入：整張 slide 全部 patch 的平均後 L2 正規化 `mean_vec`（`cil_ops.py:31-36`；`nc8_batch.py:62`）。
2. AR：`aug(mean_vec)` = 接常數 1 的 513 維 float64（`scripts/nc5_report.py:151-154`）× W [513, 4]；W = solve(A + γI, B)，A = ΣXᵀX（四個任務的 train mean_vec），B 的第 j 欄 = X_jᵀ1（任務 j 的 train slides 為 1、其餘為 0 的指示迴歸），γ = 1e-3（`scripts/nc8_report.py:71-88`、`:95`）。
3. τ̂ = AR 4 個分數 argmax（`nc8_report.py:125`）。
4. 取 τ̂ 的 I6 expert 對這張 slide 的四輪證據（`nc8_batch.py:72`）→ 8 類 cosine（A4、A5）。
5. 最終標籤 = 在 τ̂ 的 2 類（第 2τ̂、2τ̂+1 列）內 argmax（`nc2_report.py:89-96` `hard`，`pred`）。8 類標籤 = 2τ̂ + 該 argmax。
- TP 判錯（τ̂ ≠ 真實任務）時：WP 的輸入是 **τ̂ 的 expert 以 τ̂ 的兩個亞型文字**對這張 slide 算出的 s0、u、g，四輪選片與等權平均也都是 τ̂ expert 的選擇；輸出是 τ̂ 的 2 類之一。真實標籤不在 τ̂ 的 2 類內，所以 CIL 判定必為錯（`hard` 的 `pred` 只在 C_τ̂ 取 argmax）。Table 1 的 Masked ACC 欄（`hard` 的 `masked`）則用 τ̂ expert 的證據、在**真實任務**的 2 類內取 argmax（PREREG-8.md:44）。
- 兩序在 t = 4 的 TP 與判定相同（B1 已驗證逐張相同）。

## A8 檔案與 bytes

per-task（每個 fold × 任務各一份）：

| 檔案 | bytes（fold 1） | 內容 |
|---|---|---|
| `outputs/navcil/mac/i6/r2/fold{f}_{task}.pt` | esca 6,573；rcc 6,563；brca 6,573；lung 6,573 | state_dict：A [2,514]、b1 [2]、w2 [2]、b2 [] |
| 同目錄 `fold{f}_{task}.done` | 20 | 完成標記 |
| 同目錄 `fold{f}_{task}_train.json` | esca 52,152；rcc 264,833；brca 327,914；lung 332,590 | 每 epoch 的 loss 與逐張讀檔／計算秒數 |

- 參數本身 fp32：1,033 × 4 = 4,132 bytes／任務；四任務合計 16,528 bytes（`nc8_report.py:36` `EXPERT_BYTES[6]`）。檔案較大是 torch zip 容器開銷。
- 40 個 `.pt`（10 折 × 4 任務）合計 262,860 bytes。i6 目錄另有 `eval_fold{f}.pt`（fold 1 為 620,321 bytes，nc7 的評估輸出，非權重）。
- 主系統不使用第一關的 L0 bank 或 L1 權重（I6 的 s0 無參數；`nc8_batch.py:36-56` `experts_for` 載入 L0／L1 是為了其他列）。

所有任務共用：

| 項目 | bytes | 說明 |
|---|---|---|
| AR 分派器統計量（TP） | fp32 計：A 1,052,676 ＋ 4 × 2,052 = 1,060,884（`nc8_report.py:37`）；實際落檔（NC-9 `nc9/state/fold1_reverse_AR.pt`，float64）2,123,757 | **Table 1 的主系統沒有落檔**：`nc8_report.B8.W` 每次由 `cache/nc8_fold{f}_train_{task}.pt` 的 train mean_vec 從零求解（`nc8_report.py:71-88`）；落檔版本只在 NC-9（`selector/incremental_ridge.py:52-58`） |
| 類別文字 `f_txt` 與 logit_scale | `cache/text/f_txt_tcga_{esca,rcc,brca,lung}.pt`：6,117／6,109／6,053／6,117（合計 24,396） | 每檔含 f_txt [2,512]、logit_scale 標量、class_names；四個任務、所有 fold 共用；已 commit |
| λ\* | `outputs/navcil/mac/nc1/lambda.json` 1,084 | λ\* = 1.5（含 fold 1 validation 的網格） |
| prompt 來源 | `data/class_prompts.json` 3,150 | 只在重建 `f_txt` 時需要 |
| 重算 AR 所需的 train mean_vec | `cache/nc8_fold1_train_*.pt`：esca 258,449；rcc 1,316,679；brca 1,630,353；lung 1,653,905（fold 1 合計 4,859,386） | 衍生資料，不是權重；gitignored |

CONCH 權重（802 MB）只在重建 `f_txt` 時需要，執行主系統不載入。

## A9 實例：fold 1、tcga_lung、test 第 0 張（資料集順序）

slide id：`TCGA-51-6867-01Z-00-DX1.5f3a0562-efbe-413f-8e13-9826aaefa298`（`scripts/moe0_a9.py --index 0`，取法：fold 1 LUNG test split 的第 0 張，非挑選）。標籤 LUSC（全域 label 7）。

| 步驟 | shape | 數值 |
|---|---|---|
| 讀特徵 Z | (1263, 512) | patch 數 N = **1263**；‖z‖ 24.55／25.24／25.80 |
| text_nav_feats | (1263, 2) | max cos：0.0240／0.4891／0.7466（min／median／max）；entropy：0.6732／0.6905／0.6931 |
| u | (1263, 514) | — |
| s0（slide 內 z-score） | (1263,) | min −3.7532／median 0.1281／max 2.2774 |
| g | (1263,) | min −3.3021／median 0.7384／max 4.6674；std(g)/std(s0) = 1.6517 |
| s = s0 + g | (1263,) | min −5.6860／median 0.7849／max 6.1359 |
| 第 64 名分數（s 的純排序，未扣冗餘） | — | **3.9449**（s0 的第 64 名為 1.4731） |
| 四輪選出（λ = 1.5） | (64,) | 選出者在 s 的純排序名次：min 1／median 32／max 106；其中 s 純排序名次 > 64（因冗餘扣分而進入）者 5 個 |
| 與 s0 純前 64 的差異 | — | 最終 64 個中有 **46 個**不在 s0 純排序前 64 |
| 因 g 進入前 64 的 patch | — | patch 218：s0 = 1.3736、g = 4.2548、s = 5.6284；**加 g 前名次（s0）86 → 加 g 後名次（s0+g）3**；於第 1 輪被選。另：patch 518（231 → 4）、patch 1200（130 → 5） |
| slide 向量 v | (512,) | 64 個原始 Z 的平均後 L2 正規化，‖v‖ = 1.000000 |
| 與 8 類文字的 cosine | (8,) | ESAD 0.5026、ESCC 0.3916、CCRCC 0.0849、PRCC 0.1634、IDC 0.3588、ILC 0.1101、**LUAD 0.5775、LUSC 0.6671** |
| 與兩個亞型文字的 cosine | (2,) | LUAD 0.577495、LUSC 0.667105；差（LUAD − LUSC）= −0.089611 |
| WP（告訴任務）判定 | — | LUSC；正確答案 LUSC → **正確** |
| TP（AR）分數 [esca, rcc, brca, lung] | (4,) | 0.0280, 0.0230, −0.1231, **1.0721** → 分派 tcga_lung |
| CIL 最終判定 | — | LUSC；正確答案 LUSC → **正確** |

與 NC-8 快取比對：I6_cos8 與 mean_vec 重算值最大絕對差 0.00e+00。

## A10 RUNBOOK 與既有 REPORT 對 WP 的描述，與程式不一致之處

逐項核對 RUNBOOK_navcil.md §4（`I6`、`top-64／4 輪`、`mean_vec`、`AR 分派器`、`亞型文字描述`列）與 REPORT_stage9／10／11／13、PREREG-7／8／11 對 WP 的描述：

- 與程式一致、未發現不一致者：參數量 1,033 = 516r + 1（RUNBOOK §4）；u = [Z; text_nav_feats(2)]（i6_expert.py:39）；四輪 K = 64、每輪 16、λ\* = 1.5、等權平均後 L2 正規化（PREREG-7、PREREG-8:37、REPORT_stage13.md:13）；訓練用 one-shot top-64＋softmax 加權、推論用等權（REPORT_stage13.md:15、PREREG-11:29；行號 `cil_ops.py:87-90`、`:22`、`:66-67` 與本分支相符）；REPORT_stage11.md:212 對四輪選片機制的描述（`multiround.py:146-150、:139、:152`）與程式相符。
- **不一致 1（程式內文件與實際行為）**：`selector/classifier.py:3-4` docstring 寫「訓練與評估共用同一條路徑（舊 code 兩邊不一致，是 bug）」，`:50-55` 寫主線權重 softmax「與訓練一致」。主系統的推論彙整是等權平均（`cil_ops.py:31-36` → `nc8_batch.py:63`），訓練是 softmax 加權（`cil_ops.py:89`）；兩邊並不相同（見 A6）。這是檔案內的 docstring，不在 RUNBOOK／REPORT。
- **不一致 2（設定檔）**：`configs/base.yaml:18-26` 寫操作點 `budget: 8`、`chunk: 1`（「B=8 是最佳點」），並說兩者仍為 CLI 參數（`--budget／--chunk`）。主系統實際用 `cil_ops.py:22-23` 的常數 64／16；全 repo 沒有程式讀 `cfg["budget"]`／`cfg["chunk"]`，`scripts/run_1a.py:107` 的 `--budget` 預設為 64，也沒有 `--chunk` 參數。RUNBOOK §4 的「top-64／4 輪×16」列與程式一致，只有 base.yaml 的註解與設定值過時。
- **用語提醒（不是描述錯誤）**：PREREG-2.md:46 寫「WP：四任務 Masked ACC 的平均」；Table 1 第 6 列的 Masked ACC 欄是 0.9312（用分派 expert 的證據、在真實任務 2 類內取 argmax，PREREG-8.md:44），而 WP 欄的四任務平均是 0.9340（真實任務 expert，PREREG-8.md:47；與第 8 列 oracle 的 ACC = Masked ACC = 0.9340 相同）。兩者在 oracle 分派下相等，在有 router 的列不相等；REPORT_stage10.md 的表頭沒有再定義這兩欄。
- RUNBOOK 與 REPORT 對 WP 的其他描述：未發現與程式不一致。
