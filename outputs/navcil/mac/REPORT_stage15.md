# REPORT — NC-13：修正頭的貢獻（修正量設為 0，只重新推論；Mac CPU，十折兩序）

機器：mac（Apple M1 Pro）、CPU、`torch.set_num_threads(8)`、torch 2.11.0；四個 arm 與附加分析在同一次執行中完成（log：`logs/nc13_head.log`）。判準與定義見 `PREREG-13.md`。不訓練：修正頭沿用既有 I6(r = 2) 權重，分派器為 AR（γ = 1e-3，累加統計量）。task-known = oracle 分派、task-inferred = AR 分派，指標為 t = 4 CIL ACC（四任務等權平均），十折。數字附 fact-id 與其所在的 facts.json 行號（路徑相對於 `outputs/navcil/mac/`）。另一個 session 同時在本機執行，耗時僅供參考。

## 0 白話摘要

在現行的挑法（4 輪 × 16、去重）下，把修正量拿掉（只用 zero-shot 分數 s0）後，task-known 由 0.9340 變為 0.8966，也就是修正頭帶來 +3.74 pp（`nc13.a.known.reverse.acc`（nc13/facts.json:2）、`nc13.b.known.reverse.acc`（nc13/facts.json:6））；task-inferred 帶來 +3.57 pp。十折中加修正量較高的折數為 10／10（平 0、低 0），事先登記的方向性預測（至少 8 折）成立。改用一次挑 top-64 時，修正頭帶來 task-known +3.67 pp、task-inferred +3.50 pp。

瓶頸的兩個維度（h1、h2）：fold 1 各任務修正頭在全部 test patch 上的相關係數見 T6（以 |r| ≥ 0.95 描述「幾乎同向」，這是描述用的門檻，不是事先登記的判準）。ESCA、RCC 的 |r| ≥ 0.95，兩個維度幾乎同向，實際上接近只用一個方向；BRCA -0.82、LUNG -0.84，兩個維度並未重合。另外，h1、h2 同時大於 3 的 patch 比例為 ESCA 31%、RCC 77%、BRCA 1%、LUNG 41%；在這個區間 GELU 近似恆等，修正量近似為 u 的單一線性函數（(w2₁A₁ + w2₂A₂)· u 加上常數），也就是說不論相關係數多少，這些 patch 上的修正量都只沿一個方向變化（由 GELU 定義推導）。

## 1 一致性檢查（PREREG-13）

| 項目 | 基準 | 基準值 | 本輪 | 結果 |
|---|---|---|---|---|
| (a) task-known（oracle 分派），reverse | REPORT_stage10.md:18 | 0.9340 ± 0.0202 | 0.9340 ± 0.0202；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| (a) task-known（oracle 分派），paper | REPORT_stage10.md:31 | 0.9340 ± 0.0202 | 0.9340 ± 0.0202；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| (a) task-inferred（AR 分派），reverse | REPORT_stage10.md:16 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| (a) task-inferred（AR 分派），paper | REPORT_stage10.md:29 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| (c) task-known（oracle 分派），reverse | REPORT_stage13.md:215（bf488ff） | 0.9372 | 0.9372 ± 0.0209；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| (c) task-known（oracle 分派），paper | REPORT_stage13.md:215（bf488ff） | 0.9372 | 0.9372 ± 0.0209；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| (c) task-inferred（AR 分派），reverse | REPORT_stage13.md:215（bf488ff） | 0.9160 | 0.9160 ± 0.0264；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |
| (c) task-inferred（AR 分派），paper | REPORT_stage13.md:215（bf488ff） | 0.9160 | 0.9160 ± 0.0264；逐折四位相同 10/10（最大差 0.0e+00） | 符合 |

- (a) 的 8 類 cosine 與 NC-8 快取 `I6_cos8` 的逐張最大絕對差：0.00e+00（`nc13.check.cos8_a_vs_nc8_maxabs`（nc13/facts.json:70））。
- REPORT_stage13 位於分支 `ws3-ablation` @ bf488ff（尚未合併）；(c) 的逐折基準以 `git show` 讀取該 commit 的 `nc11/select/result.json`（`oneshot64`）。
- 另報（非判準）：(d) g = 0、一次 top-64 的 task-known 逐折值與 NC-1 第一關 (b) zero-shot top-64 在小數點後四位相同 10/10（REPORT_stage1-3.md:54-63，十折平均 REPORT_stage1-3.md:38）。s0 是最大 cosine 的片內 z-score，排序與最大 cosine 相同，所以一次 top-64 選到同一組 patch。

## 2 主表（t = 4、十折 mean ± sd；reverse 序，t = 4 時兩序逐折相同，兩序分列見 T2）

| arm | task-known | task-inferred | 與 (a) 的差（pp，known／inferred） | 高於／低於 (a) 的折數（known；inferred） | fact-id |
|---|---|---|---|---|---|
| (a) s0 + g，4 輪 × 16 去重（現行） | 0.9340 ± 0.0202 | 0.9128 ± 0.0258 | +0.00／+0.00 | 0／0；0／0 | `nc13.a.known.reverse.acc`（nc13/facts.json:2）；`nc13.a.inferred.reverse.acc`（nc13/facts.json:4） |
| (b) g = 0，4 輪 × 16 去重 | 0.8966 ± 0.0138 | 0.8771 ± 0.0176 | -3.74／-3.57 | 0／10；0／10 | `nc13.b.known.reverse.acc`（nc13/facts.json:6）；`nc13.b.inferred.reverse.acc`（nc13/facts.json:8） |
| (c) s0 + g，一次 top-64 | 0.9372 ± 0.0209 | 0.9160 ± 0.0264 | +0.32／+0.32 | 8／2；8／2 | `nc13.c.known.reverse.acc`（nc13/facts.json:10）；`nc13.c.inferred.reverse.acc`（nc13/facts.json:12） |
| (d) g = 0，一次 top-64 | 0.9005 ± 0.0162 | 0.8810 ± 0.0198 | -3.35／-3.18 | 0／10；0／10 | `nc13.d.known.reverse.acc`（nc13/facts.json:14）；`nc13.d.inferred.reverse.acc`（nc13/facts.json:16） |

## T2 兩序分列

| arm | known reverse | known paper | inferred reverse | inferred paper |
|---|---|---|---|---|
| (a) s0 + g，4 輪 × 16 去重（現行） | 0.9340 ± 0.0202 | 0.9340 ± 0.0202 | 0.9128 ± 0.0258 | 0.9128 ± 0.0258 |
| (b) g = 0，4 輪 × 16 去重 | 0.8966 ± 0.0138 | 0.8966 ± 0.0138 | 0.8771 ± 0.0176 | 0.8771 ± 0.0176 |
| (c) s0 + g，一次 top-64 | 0.9372 ± 0.0209 | 0.9372 ± 0.0209 | 0.9160 ± 0.0264 | 0.9160 ± 0.0264 |
| (d) g = 0，一次 top-64 | 0.9005 ± 0.0162 | 0.9005 ± 0.0162 | 0.8810 ± 0.0198 | 0.8810 ± 0.0198 |

## T3 每折值（reverse）

**task-known（oracle 分派）**

| 折 | (a) | (b) | (c) | (d) | (a) − (b) | (c) − (d) |
|---|---|---|---|---|---|---|
| 1 | 0.9203 | 0.8769 | 0.9230 | 0.8743 | +0.0434 | +0.0486 |
| 2 | 0.9570 | 0.9026 | 0.9649 | 0.9105 | +0.0544 | +0.0544 |
| 3 | 0.9473 | 0.9076 | 0.9476 | 0.9056 | +0.0397 | +0.0420 |
| 4 | 0.9306 | 0.8997 | 0.9351 | 0.8997 | +0.0309 | +0.0354 |
| 5 | 0.9643 | 0.9186 | 0.9617 | 0.9273 | +0.0458 | +0.0344 |
| 6 | 0.9349 | 0.8885 | 0.9505 | 0.9099 | +0.0464 | +0.0406 |
| 7 | 0.9293 | 0.9048 | 0.9241 | 0.9014 | +0.0246 | +0.0227 |
| 8 | 0.8912 | 0.8745 | 0.8938 | 0.8751 | +0.0167 | +0.0187 |
| 9 | 0.9323 | 0.9016 | 0.9354 | 0.9075 | +0.0307 | +0.0280 |
| 10 | 0.9324 | 0.8912 | 0.9358 | 0.8939 | +0.0412 | +0.0418 |

**task-inferred（AR 分派）**

| 折 | (a) | (b) | (c) | (d) | (a) − (b) | (c) − (d) |
|---|---|---|---|---|---|---|
| 1 | 0.9117 | 0.8710 | 0.9143 | 0.8683 | +0.0407 | +0.0460 |
| 2 | 0.9372 | 0.8828 | 0.9451 | 0.8907 | +0.0544 | +0.0544 |
| 3 | 0.9414 | 0.9043 | 0.9417 | 0.9023 | +0.0371 | +0.0394 |
| 4 | 0.8914 | 0.8658 | 0.8959 | 0.8658 | +0.0256 | +0.0300 |
| 5 | 0.9583 | 0.9125 | 0.9556 | 0.9212 | +0.0458 | +0.0344 |
| 6 | 0.9164 | 0.8700 | 0.9321 | 0.8914 | +0.0464 | +0.0406 |
| 7 | 0.8901 | 0.8655 | 0.8849 | 0.8622 | +0.0246 | +0.0227 |
| 8 | 0.8772 | 0.8605 | 0.8798 | 0.8611 | +0.0167 | +0.0187 |
| 9 | 0.8987 | 0.8681 | 0.9018 | 0.8739 | +0.0307 | +0.0280 |
| 10 | 0.9056 | 0.8706 | 0.9090 | 0.8733 | +0.0351 | +0.0357 |

## T4 配對比較（Wilcoxon signed-rank 雙尾、paired t 雙尾）

`scipy.stats.wilcoxon`（zero_method = "wilcox"，n = 10 無平手時為精確檢定，最小可能 p = 2／1024 ≈ 0.0020）、`scipy.stats.ttest_rel`。共 8 次檢定（2 組比較 × 2 種分派 × 2 序），未做多重比較校正；t = 4 時兩序逐折相同，兩序各列一次。

| 比較 | 分派 | 序 | 平均差（pp） | 前者 勝／平／負 | Wilcoxon p | paired t p | fact-id |
|---|---|---|---|---|---|---|---|
| (a) vs (b) | task-known（oracle 分派） | reverse | +3.74 | 10／0／0 | 0.0020 | 2.73e-06 | `nc13.paired.a_vs_b.known.reverse.p_wilcoxon`（nc13/facts.json:19）；`nc13.paired.a_vs_b.known.reverse.p_ttest`（nc13/facts.json:20） |
| (a) vs (b) | task-known（oracle 分派） | paper | +3.74 | 10／0／0 | 0.0020 | 2.73e-06 | `nc13.paired.a_vs_b.known.paper.p_wilcoxon`（nc13/facts.json:23）；`nc13.paired.a_vs_b.known.paper.p_ttest`（nc13/facts.json:24） |
| (a) vs (b) | task-inferred（AR 分派） | reverse | +3.57 | 10／0／0 | 0.0020 | 4.43e-06 | `nc13.paired.a_vs_b.inferred.reverse.p_wilcoxon`（nc13/facts.json:27）；`nc13.paired.a_vs_b.inferred.reverse.p_ttest`（nc13/facts.json:28） |
| (a) vs (b) | task-inferred（AR 分派） | paper | +3.57 | 10／0／0 | 0.0020 | 4.43e-06 | `nc13.paired.a_vs_b.inferred.paper.p_wilcoxon`（nc13/facts.json:31）；`nc13.paired.a_vs_b.inferred.paper.p_ttest`（nc13/facts.json:32） |
| (c) vs (d) | task-known（oracle 分派） | reverse | +3.67 | 10／0／0 | 0.0020 | 2.70e-06 | `nc13.paired.c_vs_d.known.reverse.p_wilcoxon`（nc13/facts.json:35）；`nc13.paired.c_vs_d.known.reverse.p_ttest`（nc13/facts.json:36） |
| (c) vs (d) | task-known（oracle 分派） | paper | +3.67 | 10／0／0 | 0.0020 | 2.70e-06 | `nc13.paired.c_vs_d.known.paper.p_wilcoxon`（nc13/facts.json:39）；`nc13.paired.c_vs_d.known.paper.p_ttest`（nc13/facts.json:40） |
| (c) vs (d) | task-inferred（AR 分派） | reverse | +3.50 | 10／0／0 | 0.0020 | 2.82e-06 | `nc13.paired.c_vs_d.inferred.reverse.p_wilcoxon`（nc13/facts.json:43）；`nc13.paired.c_vs_d.inferred.reverse.p_ttest`（nc13/facts.json:44） |
| (c) vs (d) | task-inferred（AR 分派） | paper | +3.50 | 10／0／0 | 0.0020 | 2.82e-06 | `nc13.paired.c_vs_d.inferred.paper.p_wilcoxon`（nc13/facts.json:47）；`nc13.paired.c_vs_d.inferred.paper.p_ttest`（nc13/facts.json:48） |

方向性預測（PREREG-13）：(a) 的 task-known 高於 (b) 的折數 = 10／10（門檻 ≥ 8）→ **成立**（`nc13.paired.a_vs_b.known.reverse.wins`（nc13/facts.json:21））。

## T5 附加分析 1：修正量對選取的影響（自家修正頭，所選 64 個 patch 的重疊數）

每折先對該折全部 test slides（四任務）平均，再取十折 mean ± sd。

| 比較 | 重疊數 | fact-id |
|---|---|---|
| (a) vs (b)：4 輪 × 16，有／無修正量 | 30.32 ± 6.02（共 64） | `nc13.overlap.ab`（nc13/facts.json:50） |
| (c) vs (d)：一次 top-64，有／無修正量 | 31.08 ± 6.03（共 64） | `nc13.overlap.cd`（nc13/facts.json:51） |
| (a) vs (c)：有修正量，兩種挑法 | 58.00 ± 0.42（共 64） | `nc13.overlap.ac`（nc13/facts.json:52） |
| (b) vs (d)：無修正量，兩種挑法 | 58.25 ± 0.15（共 64） | `nc13.overlap.bd`（nc13/facts.json:53） |

## T6 附加分析 2：瓶頸是否退化（fold 1，自家修正頭，該任務全部 test slides 的所有 patch）

h = A u + b1（`selector/i6_expert.py:35-45`）；GELU 線性區以 h > 3 計（GELU(3) ≈ 2.996）。

| 任務 | patch 數 | Pearson r(h1, h2) | h1 > 3 | h2 > 3 | 兩者皆 > 3 | w2 | fact-id |
|---|---|---|---|---|---|---|---|
| tcga_esca | 56,596 | +0.9948 | 0.322 | 0.345 | 0.314 | (+0.0559, +0.0537) | `nc13.bottleneck.esca.pearson`（nc13/facts.json:54） |
| tcga_rcc | 246,652 | +0.9996 | 0.769 | 0.769 | 0.767 | (+0.0855, +0.0897) | `nc13.bottleneck.rcc.pearson`（nc13/facts.json:58） |
| tcga_brca | 288,350 | -0.8244 | 0.214 | 0.666 | 0.015 | (-0.1115, -0.1655) | `nc13.bottleneck.brca.pearson`（nc13/facts.json:62） |
| tcga_lung | 286,519 | -0.8426 | 0.531 | 0.874 | 0.406 | (+0.1841, -0.1538) | `nc13.bottleneck.lung.pearson`（nc13/facts.json:66） |

## T7 耗時（秒，wall clock；與其他工作同時執行）

| 項目 | 秒 |
|---|---|
| 推論與附加分析（十折，2,835 張 test slide） | 161 |
| 其中讀檔（逐張加總） | 9 |
| 其中計算（逐張加總） | 151 |
| 報告 | 1 |
