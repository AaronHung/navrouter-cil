# AMENDMENT-2 — NC-5 C 段（AR／LIN8）改用 float64

日期：2026-09-25。**PREREG-5.md 不修改**。

## 問題

PREREG-5 操作定義 8 規定 AR／LIN8 用 float32。fold 1、t = 4 的一致性檢查未過：選出的 γ = 0.01 時，
依序累加與一次解的 W 最大差 1.98e-2（門檻 1e-4）。診斷見
`outputs/navcil/mac/nc5/ar_check_failed.json`：誤差約等於 float32 機器精度 × 條件數（γ = 0.01 時
3.4 × 10⁵）；float64 下每個 γ 都 ≤ 4e-11，累加寫法正確。

## 修正

只有 AR 與 LIN8 的累加（A、bj）、求解（`torch.linalg.solve`）與打分（xW）改用 float64，在 CPU 上算。
其餘照 PREREG-5：γ 候選 {0.01, 0.1, 1, 10}、選法（十折 validation 平均 t = 4 CIL，同分取小）、
一致性檢查門檻 < 1e-4、所有報告欄位都不變。

AGENTS.md 可攜規則 3 加一條例外：513 × 513 以內的封閉解可在 CPU 上用 float64。

## 影響範圍

只有 C 段。A、B、D 不使用 AR／LIN8，不受影響。

float32 下的 AR／LIN8 結果（包括 γ 的 validation 選擇）全部作廢，不報、不引用；γ 在 float64 下從頭重選。
