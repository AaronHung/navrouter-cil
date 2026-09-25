# AGENTS.md — 常駐規則

每一個在本 repo 工作的 agent（人或模型）都必須遵守以下規則。

## 紅線

1. **禁用 QPMIL**。不得 import、複製或引用 QPMIL / QPMIL-VL 的任何程式碼；也不得從
   navipath 複製檔案或 import navipath。`tests/test_no_banned_deps.py` 必須通過，任何
   commit 前都要跑 `pytest`。
2. **失敗就停**。任何步驟失敗（指令非零結束、數字與期望不符、測試失敗）就停下來回報，
   不重試、不跳過、不自行修改期望值。
3. **跑完不關機**。長時間工作結束後不關機、不休眠；在 MSI 上**不下 `wsl --shutdown`**。
4. **同一張表的數字只來自同一台機器、同一批執行**。不同機器或不同批次的數字不得混在
   同一張表。
5. **輸出一律放 `outputs/navcil/<machine>/`**。

## 命名原則（NC-2 起）

- 每任務的選片器稱 **expert**；程式沿用 `selector`（`EvidenceSelector`、`SelectorBank`）。
- 把 slide 分派到某個 expert 的模組稱 **router**（`selector/router.py`）。
- 方法名稱不用 navigation、zero 字樣（zero-shot 基線的描述不在此限）。
- `zeronav` 與 QPMIL 相關識別字仍由 `tests/test_no_banned_deps.py` 禁用；`router` 已解禁。

## 執行方式

- 長時間指令一律在 tmux 裡跑，用 `scripts/run_stage.sh`（macOS 上自動包
  `caffeinate -ims`）。
- 每 15 分鐘心跳一次：`run_stage.sh` 會每 900 秒在 log 寫一行 `heartbeat`；負責監看的
  agent 每 15 分鐘檢查一次 log 並回報進度。
- 回報只給數字與表格，不寫論文文字。

## 可攜規則（Mac 的 mps／cpu 與 MSI 的 cuda 跑同一份程式）

1. 裝置自動選 cuda → mps → cpu，可用 `--device` 覆寫（`selector/device.py`）。
2. 程式裡不寫絕對路徑；每台一個 `configs/machine_<name>.yaml`，以環境變數
   `NAVCIL_MACHINE` 選擇，覆寫 `configs/base.yaml` 的同名鍵。
3. 一律 float32；`torch.load(..., map_location="cpu")` 後再 `.to(device)`；不用 CUDA
   專用 API（計時同步用 `torch.accelerator.synchronize()`）。
   例外（AMENDMENT-2）：513 × 513 以內的封閉解（例如 AR／LIN8 的累加與求解）可在 CPU 上用 float64。
4. 兩台用同一個 PyTorch minor 版本（目前 2.11.x），寫在 `requirements.txt`。
5. 每個（關卡、fold、任務）完成就寫 done 標記（`<task>.done`），重跑自動跳過。
6. 輸出放 `outputs/navcil/<machine>/`。
7. 文字特徵只在 Mac 用 CONCH 算一次，存在 `cache/text/` 並 commit；MSI 不載入 CONCH
   （`scripts/verify_text_cache.py` 只在 Mac 跑）。
8. `scripts/run_stage.sh` 兩台共用：開 tmux，在 macOS 上自動加 `caffeinate -ims`。
9. 每次執行都記錄每張 slide 的讀檔秒數與計算秒數（`t_read_s`、`t_compute_s`）。

## 範例

```bash
export NAVCIL_MACHINE=mac        # MSI 上為 msi（需自備 configs/machine_msi.yaml）
PYTHONNOUSERSITE=1 python -m pytest
scripts/run_stage.sh 1a python scripts/run_1a.py --device cpu --tag 1a
```
