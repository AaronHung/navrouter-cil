# Vendored CONCH (text tower only)

上游：<https://github.com/mahmoodlab/CONCH> — "CONCH custom API"，
目錄 `conch/open_clip_custom/`，
commit `141cc09c7d4ff33d8eda562bd75169b457f71a62`（2025-03-26）。

只建 text tower，所以不需要 CONCH vision tower 依賴的 `timm` / `torchvision`。
模型權重**不**隨 repo 發佈；`configs/machine_<name>.yaml` 的 `conch_ckpt_path`（navcil；原為 pathselect 的 `configs/pathselect.yaml`） 指向自備的
CONCH checkpoint。

## 來源查核（V3）

本 repo 的檔案最初是從本機的 `01_navipath/QPMIL-VL/models/conch/` 複製的，因此必須
確認 QPMIL 沒有改過 CONCH 原始碼。做法：從官方 repo 直接抓對應檔案，逐檔 diff。

**結論：QPMIL-VL 的那份 CONCH 副本與官方完全相同，四個檔案 byte-for-byte 一致，
沒有任何改動。** 本 repo 的副本目前也直接對得上官方 sha256（見
`UPSTREAM_SHA256SUMS.txt`），唯一例外是下面記錄的 tokenizer 相容性修改。

| 檔案 | vs 官方 | vs QPMIL 副本 | 說明 |
|---|---|---|---|
| `transformer.py` | **identical** | identical | 原樣複製，`TextTransformer` 在此 |
| `custom_tokenizer.py` | **1 處修改**（見下） | 其餘 identical | 原樣複製 + transformers 5 相容性 |
| `model_configs/conch_ViT-B-16.json` | **identical** | identical | 原樣複製 |
| `tokenizers/conch_byte_level_bpe_uncased.json` | **identical** | identical | 原樣複製（BPE 詞表） |
| `text_tower.py` | — | — | **本 repo 撰寫**，非上游檔案 |
| `__init__.py` | — | — | **本 repo 撰寫**，只 re-export 三個名字 |

### 唯一的本地修改

`custom_tokenizer.py`，函式 `tokenize()`：

```diff
--- upstream/custom_tokenizer.py
+++ third_party/conch/custom_tokenizer.py
@@ -21,7 +21,9 @@
 def tokenize(tokenizer, texts):
     # model context length is 128, but last token is reserved for <cls>
     # so we use 127 and insert <pad> at the end as a temporary placeholder
-    tokens = tokenizer.batch_encode_plus(texts,
+    # NOTE(pathselect): upstream calls `tokenizer.batch_encode_plus(...)`, removed in
+    # transformers>=5.  `tokenizer(...)` is the documented equivalent; args unchanged.
+    tokens = tokenizer(texts,
                                         max_length = 127,
                                         add_special_tokens=True,
                                         return_token_type_ids=False,
                                         truncation = True,
                                         padding = 'max_length',
                                         return_tensors = 'pt')
```

原因：本機 `transformers` 是 5.5.3，`PreTrainedTokenizerFast.batch_encode_plus` 已移除
（`AttributeError: TokenizersBackend has no attribute batch_encode_plus`）。
`tokenizer(...)` 是官方對等入口，參數一字未改，tokenization 結果相同。

`text_tower.py` 是本 repo 撰寫的 text-only 載入器，對應上游
`coca_model._build_text_tower` 與 `factory.read_state_dict` 的邏輯，不含 vision tower。

### 重新查核

```bash
cd third_party/conch && shasum -a 256 -c UPSTREAM_SHA256SUMS.txt
# 預期：custom_tokenizer.py FAILED（上面那處修改），其餘三個 OK
```

## 授權

CONCH 的程式碼與權重由 Mahmood Lab 以 CC-BY-NC-ND 4.0 釋出（非商業、學術用途）。
