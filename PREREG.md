# PREREG — NC-1 預先註冊（第一～三關）

登記時間：2026-09-24，開跑前 commit。本檔 commit 之後不得修改。
機器：Mac（Apple M1 Pro），`--device cpu`；本輪所有數字來自同一台、同一批。

## 判準

### 第一關通過
(e) 導覽器四輪的 Masked ACC 十折平均 ≥ 0.90，比 (b) zero-shot top-64 高 ≥ 0.03，且贏的折數 ≥ 7/10。

### 第二關判讀（以 t = 4、兩序、十折平均判定）
- TP-text-class ≥ 0.97 → N1 降為消融，第四關主做 N2；
- TP-text-class ≤ 0.95、TP-nav ≥ TP-text-class + 0.03 且 TP-nav ≥ TP-proto − 0.005 → N1 為主線；
- 其他 → N1 列為消融，主做 N2。

### 第三關通過
無儲存 TP（text-class 或 nav）＋Hard 的 ACC，reverse ≥ 0.859 且 paper ≥ 0.890（QPMIL-VL 已發表值，只當外部參考），零 replay。

## 操作定義（與判準同時登記）

1. **任務序**：reverse = esca → rcc → brca → lung；paper = lung → brca → rcc → esca
   （QPMIL-VL Tab. 1 的順序，其 ACC 0.890；reverse 對應 Tab. 2 的 0.859）。
2. **類別編號**：8 類固定依 esca、rcc、brca、lung 疊放（每任務 2 類），與任務序無關。
3. **Masked ACC**：只在該 slide 真實任務的 2 類之間取 argmax。
4. **第一關 (b)(d)(e) 的比較與判準**一律用主聚合：所選 patch 等權平均後 L2 正規化；
   softmax 加權聚合只另報，不進判準。
5. **十折平均**：每折先取四任務 Masked ACC 的平均，再對十折取平均；sd 為十折樣本標準差。
   「贏的折數」＝該折四任務平均 (e) > (b)（嚴格大於）。
6. **四輪選片**：K = 64、每輪 16，第 2 輪起分數為 z-score 後的導覽器分數 − λ·max cos（與已選
   patch），沿用 `selector/multiround.py` 的 `SequentialBudgetedObserver`（maxsim 模式）。
7. **λ\***：fold 1 validation split 上四任務 (e) Masked ACC 平均最高者，同分取小；之後十折固定。
   **T\***：fold 1 validation split、t = 4，Soft 的 ACC 最高者，同分取小，每個 TP 變體各選一次。
8. **第二關 t = 4 的判定值**：兩序在 t = 4 的候選集合相同，取兩序、十折 TP 正確率的平均。
9. **argmax 同分**：取候選順序（該序中較早學到的任務）中的第一個；patch-vote 票數同分時取
   投票 patch 的 max cosine 總和較大者。
10. **第三關 ACC** = 平均_j R[4][j]；Forgetting = 平均_{j<4}(max_{j≤t<4} R[t][j] − R[4][j])，
    t、j 依該序的任務位置編號。
