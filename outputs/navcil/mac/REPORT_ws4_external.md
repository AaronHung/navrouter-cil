# REPORT — WS4：外部方法對照表（唯讀查閱論文，不跑實驗、不改程式）

來源：`reference/papers/` 九篇 PDF（2026-09-30 讀取；頁碼 = PDF 頁序，以 `pdftotext -f N -l N` 逐頁確認）。本研究的設定取自 `REPORT_stage10.md`、`REPORT_stage11.md`、`selector/cil_ops.py:18-21`、`configs/base.yaml:35-40`。本報告沒有新數字，全部是引用。

## T1 外部方法逐篇登錄

「論文未載明」= PDF 內找不到該項。ACC 一律照論文原單位（小數或 %）。

| 方法 | 發表處與年份 | backbone | 資料集與任務順序 | 切分來源 | replay（存舊 WSI 或其特徵）及數量 | 可訓練參數量 | 報告的 ACC（各順序） | 頁碼／表號 |
|---|---|---|---|---|---|---|---|---|
| QPMIL-VL（Gou et al.） | AAAI 2025（p.1 版權頁） | CONCH 影像／文字編碼器（凍結）；CLAM 切 256 × 256、10×（p.5） | TCGA NSCLC、BRCA、RCC、ESCA，各 2 類（p.10 Tab. 5：slides 965／952／768／150）；forward = NSCLC→BRCA→RCC→ESCA，reverse = ESCA→RCC→BRCA→NSCLC | 十折交叉驗證（p.5）；切分檔來源與 seed 論文未載明 | 無 buffer，0 張（p.1–2）；存 prototype pool（M = 20 組 key／prompt，p.5） | 0.365 M（Tab. 1「Params」，p.5） | forward 0.890 ± 0.021；reverse 0.859 ± 0.032 | Tab. 1 p.5；Tab. 2 p.6；Tab. 10 p.12（MaxPooling 列同值） |
| ConSlide（Huang et al.） | ICCV 2023（PDF 為 arXiv 版未印；依 QPMIL-VL 參考文獻 p.8） | ConvNeXt（預訓練資料論文未載明）抽 patch 與 region 特徵；20×，4,096² region → 64 個 512² patch；HIT transformer（p.6） | 同四個 TCGA 資料集，但 Tab. 1 的 case 數不同（LUAD 492、LUSC 466、IDC 726、ILC 149、CCRCC 498、PRCC 289、ESAD 65、ESCC 89，p.6）；NSCLC→BRCA→RCC→ESCA（p.6），另報反序（p.8） | 十折交叉驗證（p.6）；切分來源論文未載明 | 有：region buffer 1,100／2,200／6,600 regions（≈ 5／10／30 WSIs）（Tab. 3 p.7） | 論文未載明（QPMIL-VL Tab. 1 列 39.446 M，p.5） | buffer 30：forward 0.659 ± 0.022、reverse 0.499 ± 0.025；buffer 10：0.594 ± 0.053、0.453 ± 0.052；buffer 5：0.553 ± 0.033、0.426 ± 0.047 | Tab. 3 p.7；Tab. 5 p.8 |
| ZeroSlide（Bui et al.） | arXiv 2504.15627v1，2025；發表處論文未載明 | TITAN（patch 特徵 768 維，slide encoder 與 text encoder 皆凍結）（p.3–5） | 六個 TCGA：BRCA→RCC（3 類，含 chromophobe）→NSCLC→ESCA→TGCT→CESC，單一順序（p.5–6） | 每資料集 10 折 train／val／test（p.5）；來源論文未載明 | 0（training-free）（p.4、p.7） | 0（p.4） | 64.129 ± 1.591 %（單一順序） | Tab. 1 p.7 |
| MergeSlide（Bui et al.） | arXiv 2511.13099v1，2025；發表處論文未載明 | TITAN；每任務微調 slide aggregator（TITAN 預訓練初始化），再做正交合併（p.4–6） | 同 ZeroSlide 的六個 TCGA；B→R→N→E→T→C 與反序 C→T→E→N→R→B，另四種交錯序（p.6–7） | 每 cohort 10 折（p.6）；來源論文未載明 | 0 WSIs（Tab. 2 p.7）；存一個合併後的模型 | 論文未載明（整個 slide aggregator 都在訓練） | 以 bACC 報：w/ TCP 87.929 ± 2.110 %（正序）、87.930 ± 2.112 %（反序）；Tab. 5 四種交錯序 ACC 91.933–91.975 % | Tab. 2、Tab. 3 p.7；Tab. 5 p.8 |
| CLIP-Adapter（Gao et al.） | IJCV 2024（依 QPMIL-VL 參考文獻 p.8；PDF 為 arXiv v2） | CLIP ResNet-50 | 11 個自然影像資料集 few-shot（1／2／4／8／16-shot）；不是持續學習 | 用各資料集官方 test split；few-shot 取樣來源論文未載明 | 不適用（非 CIL） | 0.52 M（Tab. 1 p.8） | ImageNet 16-shot 61.33 % | Tab. 1 p.8 |
| Tip-Adapter（Zhang et al.） | arXiv 2111.03930v2，2021；發表處論文未載明 | CLIP RN50（另報 RN101、ViT-B/32、ViT-B/16、RN50×16） | 11 個自然影像資料集 few-shot；不是持續學習 | 同 CoOp；論文未另載明 | 不適用（非 CIL）；cache 存全部 N·K 筆 few-shot 訓練特徵（p.4） | training-free 版 0；-F 版微調 cache keys，數量論文未載明 | ImageNet 16-shot RN50：Tip-Adapter 62.03 %、Tip-Adapter-F 65.51 % | Tab. 1、Tab. 3 p.7 |
| ACIL（Zhuang et al.） | arXiv 2205.14922v2，2022；發表處論文未載明 | ResNet-32（CIFAR-100）、ResNet-18（ImageNet），base phase 訓練後凍結＋解析解分類器 | CIFAR-100、ImageNet-Subset、ImageNet-Full；先學一半類別，再分 K = 5／10／25／50 階段（p.5） | 沿用 [9]、[15] 的協定；類別順序論文未載明 | 0（exemplar-free，Tab. I 標記，p.6） | 論文未載明（FE 擴展維度 8k／15k／15k） | CIFAR-100 平均增量準確率 66.30／66.07／65.95／66.01 %（K = 5／10／25／50） | Tab. I p.6 |
| Kim et al.（WP × TP 理論） | NeurIPS 2022（p.1） | AlexNet 類（MNIST）、ResNet-18（CIFAR、Tiny-ImageNet），不預訓練 | M-5T、C10-5T、C100-10T、C100-20T、T-5T、T-10T | 5 個隨機 seed（p.7）；切分論文未載明 | HAT+CSI／Sup+CSI：0；「+c」版與 replay 基線：buffer 200（MNIST、CIFAR-10）、2,000（CIFAR-100、Tiny-ImageNet） | 論文未載明 | HAT+CSI C100-10T 63.3 ± 1.00 %；Sup+CSI 65.1 ± 0.39 % | Tab. 3 p.9 |
| Any-SSR（Tong et al.） | arXiv 2503.13575v2，2025；發表處論文未載明 | LLaMA-2-7B-Chat（凍結）＋每任務 LoRA（rank 8）＋解析（RLS）router | TRACE 八個文字任務，Order1／Order2（Tab. 1 p.6） | 沿用 TRACE；每任務 5,000 筆 | 0（exemplar-free） | 論文未載明 | OP：Order1 55.69 %、Order2 55.69 % | Tab. 2 p.7 |

## T2 QPMIL-VL 已發表值核對

| 量 | 序 | `reference/external_baselines.json` | `REPORT_stage5.md:161-162` | 論文 | 表號（頁） | 一致 |
|---|---|---|---|---|---|---|
| ACC | reverse | 0.859 | 0.859（:161） | 0.859 ± 0.032 | Tab. 2（p.6）；Tab. 10 MaxPooling（p.12） | 一致 |
| ACC | paper（= forward） | 0.89 | 0.890（:162） | 0.890 ± 0.021 | Tab. 1（p.5）；Tab. 10 MaxPooling（p.12） | 一致 |
| Masked ACC | reverse | 0.925 | 0.925 | 0.925 ± 0.018 | Tab. 2（p.6） | 一致 |
| Masked ACC | paper | 0.93 | 0.930 | 0.930 ± 0.018 | Tab. 1（p.5） | 一致 |
| Forgetting | reverse | 0.064 | 0.064 | 0.064 ± 0.031 | Tab. 2（p.6） | 一致 |
| Forgetting | paper | 0.027 | 0.027 | 0.027 ± 0.014 | Tab. 1（p.5） | 一致 |
| 表號欄位 | — | reverse「Tab. 2 (0.859±0.032)」、paper「Tab. 1 (0.890±0.021)」 | — | — | — | 一致 |
| 順序對應 | — | 「paper 序即 forward」 | — | forward = NSCLC→BRCA→RCC→ESCA（p.5）；reverse = ESCA→RCC→BRCA→NSCLC（p.6） | — | 與 `selector/cil_ops.py:19-20` 一致 |

## T3 與本研究的設定逐項對照

本研究：主系統 D3 = AR（γ = 0.001）＋ I6(r = 2)。ACC reverse 0.9128 ± 0.0258、paper 0.9128 ± 0.0258（`REPORT_stage10.md:16`、`:29`）；Masked ACC 0.9312 ± 0.0215；Forgetting reverse 0.0224 ± 0.0172、paper 0.0041 ± 0.0055。

| 項目 | 本研究（出處） | QPMIL-VL | ConSlide | ZeroSlide | MergeSlide |
|---|---|---|---|---|---|
| 資料集 | TCGA esca／rcc／brca／lung，各 2 類；test 十折合計 150／768／952／965 張（`REPORT_stage10.md:103-106` 列和） | 相同（Tab. 5 p.10 slides 150／768／952／965，逐任務相同） | 不同（同四個 cohort，但 case 數不同，p.6 Tab. 1） | 不同（六任務；RCC 3 類；NSCLC、ESCA 類別數與張數不同，p.6 Fig. 2） | 不同（同 ZeroSlide，p.6 Tab. 1） |
| 任務數 | 4 | 相同 | 相同 | 不同（6） | 不同（6） |
| reverse 序 esca→rcc→brca→lung（`selector/cil_ops.py:19`） | — | 相同（Tab. 2） | 相同（Tab. 5「reversed」） | 不同（無此序） | 不同（無此序） |
| paper 序 lung→brca→rcc→esca（`selector/cil_ops.py:20`） | — | 相同（Tab. 1 forward） | 相同（Tab. 3） | 不同 | 不同 |
| 十折 | 十折（`REPORT_stage10.md:1`） | 相同 | 相同 | 相同（10 折） | 相同（10 折） |
| 切分檔 | `can_dataset/<task>/datasplit/fold_{1..10}.npz`；產生邏輯與 seed 未找到（`REPORT_stage11.md` R4） | 論文未載明 | 論文未載明 | 論文未載明 | 論文未載明 |
| 特徵 backbone | CONCH（`configs/base.yaml:35-40`，`feats-l1-s256`） | 相同（CONCH） | 不同（ConvNeXt） | 不同（TITAN） | 不同（TITAN） |
| patch 大小／放大倍率 | 256；放大倍率本 repo 設定未寫 | 256 相同；10× 無法核對 | 不同（512 patch、4,096 region、20×） | 不同（256 patch＋1,024 region、10×） | 256、10×；backbone 已不同 |
| replay | 零 replay：不存舊 WSI 或逐片特徵；router 只存累加統計量 A（513 × 513）與 b_t（`REPORT_stage11.md` R1、T5） | 相同（無 buffer）；另存 prototype pool | 不同（buffer ≈ 5／10／30 WSIs） | 相同（0） | 相同（0 WSIs）；另存合併模型 |
| 可訓練參數 | 每任務 1,033（I6，`REPORT_stage11.md` R2），四任務合計 4,132；AR router 為封閉解，W = 513 × 4 | 不同（0.365 M 合計） | 不同（論文未載明；QPMIL-VL 列 39.446 M） | 不同（0） | 論文未載明 |
| ACC 定義 | 平均_j R[4][j]（`PREREG.md:36`） | 相同（Eq. 11 p.10） | 論文未載明是否逐資料集平均（p.6「ACC of all the previous and current datasets」） | 論文未載明計算式（p.5） | 不同（以 bACC 為主指標，p.6） |
| Masked ACC 定義 | 真實任務 2 類內 argmax，證據用分派後的 expert（PREREG-3 定義 9） | 類別遮罩相同；「分派後 expert」這一步 QPMIL-VL 沒有（單一模型遮 logits，p.5） | 遮罩相同（p.6） | 相同（只比當前任務 prototypes，p.5） | 不同（Masked bACC） |
| Forgetting 定義 | 平均_{j<4}(max_{j≤t<4} R[t][j] − R[4][j])（`PREREG.md:36`），單項可為負 | 不同：Eq. 13（p.10）的 max 含 i = T，單項 ≥ 0 | 論文未載明（「following [9, 24]」） | 文字描述相同（p.5），公式論文未載明 | 論文未載明 |
| 驗證集用途 | λ\*、r\*、γ 事先由 validation 選定（`REPORT_stage11.md` R1） | 論文未載明 | 論文未載明 | 用 val 選 checkpoint（p.5） | 論文未載明 |
| 統計 | 十折 mean ± sd | 相同（mean ± sd，十折） | 相同 | 相同 | 相同 |
| 逐折數值 | 有（`nc8/per_fold.json`） | 論文未載明（只給 mean ± sd，無法 paired） | 論文未載明 | 論文未載明 | 論文未載明 |

另外五篇（CLIP-Adapter、Tip-Adapter、ACIL、Kim et al.、Any-SSR）在上表每一項都「不同」：沒有 WSI，其中兩篇（CLIP-Adapter、Tip-Adapter）不是持續學習，Any-SSR 是文字 LLM。

## T4 結論：哪些數字進主表、哪些只放附註

| 外部數字 | 放哪裡 | 理由（對應 T3） |
|---|---|---|
| QPMIL-VL ACC reverse 0.859、paper 0.890；Masked ACC 0.925／0.930；Forgetting 0.064／0.027 | **主表**，照 `REPORT_stage5.md:161-162` 的格式放「外部參考、已發表值、非 paired」列 | 資料集（逐任務張數相同）、兩序、十折、CONCH 特徵、零 WSI replay、ACC 定義都相同。附註要寫：(i) 切分檔是否與論文同一份，論文沒寫，無法確認；(ii) 只有 mean ± sd，不能做逐折檢定；(iii) Forgetting 公式有差異（Eq. 13 的 max 含 i = T），這兩列 Forgetting 差距解讀時要注意；(iv) 可訓練參數 0.365 M 對 4,132 |
| QPMIL-VL Tab. 1／Tab. 2 內自己在 CONCH 上重跑的列（MI-Zero 0.839、AttriCLIP、EWC、LwF、ER／ER-ACE／DER++／A-GEM） | 可進主表，但要標「引自 QPMIL-VL Tab. 1／2」與 buffer 大小（/30、/100） | 協定與 QPMIL-VL 相同（p.5「all methods are adapted to VLM」）。rehearsal 列有 buffer，要在 replay 欄註明；本報告沒有另外取得這些方法的原論文 |
| QPMIL-VL 表中的 ConSlide/30（0.659／0.499，灰字） | **只放附註** | 論文註明灰字「直接引自已發表論文」（Tab. 1 註，p.5），與 ConSlide Tab. 3／Tab. 5 的 buffer 30 值相同，所以是 ConvNeXt、20× 的設定，不是 CONCH |
| ConSlide 原論文所有 ACC（buffer 5／10／30，兩序） | **只放附註** | backbone（ConvNeXt）、放大倍率（20×）、case 數、replay（buffer ≈ 5–30 WSIs）都不同 |
| ZeroSlide 64.129 % | **只放附註** | 六任務、不同順序、TITAN、類別定義不同；無 reverse／paper 序 |
| MergeSlide bACC 87.929／87.930 %、Tab. 5 ACC 91.9 % | **只放附註** | 同上，且主指標是 bACC；參數量論文未載明 |
| CLIP-Adapter、Tip-Adapter、ACIL、Kim et al.、Any-SSR 的全部數字 | **不列數字** | 非 WSI（其中兩篇也不是 CIL），只能在方法段落當技術來源引用 |
