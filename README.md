# navrouter-cil

WSI patch evidence selection 的 class-incremental 實驗 repo。規則見 [AGENTS.md](AGENTS.md)。

## 來源

所有程式與資料檔取自 **AaronHung/pathselect @ `081bdcd165bdc02180c7544962e09ef80d80b98f`**
（main，2026-09-16，"exp3: evidence budget sweep B in {4,16} (DR-055)"）。
未從 navipath 複製任何檔案，也不 import navipath 或 QPMIL-VL。

| 本 repo | pathselect@081bdcd | 備註 |
|---|---|---|
| `selector/text_encoder.py` | 同路徑 | 改：讀 `configs/base.yaml` + `configs/machine_<NAVCIL_MACHINE>.yaml`；`torch.load` 改 `map_location="cpu"`；錯誤訊息 |
| `selector/evaluate.py` | 同路徑 | 未改 |
| `selector/flat_selector.py` | 同路徑 | 未改 |
| `selector/multiround.py` | 同路徑 | 未改 |
| `selector/device.py` | 同路徑 | 改：docstring 用法（原文指向 navipath 的模組，僅文字、非 import）；`get_device` 檢查參數 |
| `third_party/conch/` | 同路徑 | 未改 |
| `data/` | 同路徑 | 未改 |
| `configs/base.yaml` | `configs/pathselect.yaml` | 改名；移除絕對路徑（移到 `configs/machine_mac.yaml`）；`f_txt_cache_dir: cache/text` |
| `reference/v9/`、`reference/SHA256SUMS.txt` | `reference/` | 已依 SHA256SUMS.txt 驗證，六檔皆 OK |
| `cache/text/f_txt_*.pt` | `outputs/cache/f_txt_*.pt` | 四個 task；已驗證與 CONCH text tower 重算逐元素相同 |
| `tests/test_no_banned_deps.py` | 同路徑 | 改：EXEMPT 只保留本 repo 存在的兩檔 |
| `tests/test_label_space_alignment.py` | 同路徑 | 改：未設 `NAVCIL_MACHINE` 時資料相關測試 skip |
| `reference/verify_v9_delta.py` | `scripts/verify_v9_delta.py` | 1a 的對照；未改 |

### 為了 import 而補的最少必要檔案

| 檔案 | 被誰需要 |
|---|---|
| `selector/__init__.py` | `selector` 套件 |
| `selector/classifier.py` | `selector/evaluate.py`（`conch_classify`、`softmax_weights`） |
| `scripts/v9_reference.py` | `tests/test_label_space_alignment.py`、`reference/verify_v9_delta.py` |

### 本 repo 新增

`configs/machine_mac.yaml`、`scripts/run_stage.sh`、`scripts/run_1a.py`、
`scripts/verify_text_cache.py`、`pytest.ini`、`requirements.txt`、`AGENTS.md`、`.gitignore`。

## 使用

```bash
conda activate navcil                      # Mac：Python 3.12.2、torch 2.11.0
export NAVCIL_MACHINE=mac
PYTHONNOUSERSITE=1 python -m pytest
python scripts/verify_text_cache.py        # 只在 Mac（需要 CONCH 權重）
scripts/run_stage.sh 1a python scripts/run_1a.py --device cpu --tag 1a
```
