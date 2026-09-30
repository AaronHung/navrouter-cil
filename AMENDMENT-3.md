# AMENDMENT-3 — PREREG-14 檢查 4：NC-12 `acc_task` 的比對方式

日期：2026-09-30。**PREREG-14.md 不修改**。

## 問題

PREREG-14 一致性檢查 4 要求 NC-8 的 R 與 NC-12 的 `per_fold[序][折][階段].acc_task` 逐折、逐格位元相同。
第一次執行（`scripts/nc14_ext_metrics.py`，未 commit 版）在 paper 序不符，其餘八項兩序皆通過。

原因：NC-12 的 `acc_task` 以**任務編號**為鍵（0 = esca、1 = rcc、2 = brca、3 = lung，
`scripts/nc12_order_traj.py:83` `dict(zip(map(int, seen), acc_task))`），而核對程式把鍵當成該序中的**位置**。
reverse 序（esca→rcc→brca→lung）位置與任務編號相同，所以通過；paper 序（lung→brca→rcc→esca）錯位。
唯讀檢視 fold 1 paper 序：依任務編號換算後各格與 NC-8 位元相同。

## 修正

檢查 4 中 NC-12 的比對方式改為：該序第 j 個位置的任務 = `ORDERS[序][j]`，其任務編號 = 該任務在
`configs/base.yaml` `tasks` 中的索引；R_{t,j} 與 `acc_task[str(任務編號)]` 比對。

此修正只釐清比對方式，不改變 PREREG-14 的任何判準、公式、資料來源或期望值。十折、兩序逐一重新檢查九項。
