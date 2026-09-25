#!/usr/bin/env python3
"""NC-5 A 的 z 來源：腫瘤／非腫瘤文字 key（只在 Mac 用 CONCH text tower 算一次並 commit）。

兩組 prompt 各走 `encode_prompt_groups`（與 scripts/nc1_organ_keys.py 同一條路徑：CONCH text tower
→ 每條 L2 normalize → 組內平均 → L2 normalize）。prompt 照 PREREG-5 原文，不另加句點。

    NAVCIL_MACHINE=mac python scripts/nc5_tumor_keys.py
輸出：cache/text/tumor_normal_keys.pt  {"features": [2, 512]（0 = tumor、1 = normal）, "prompts"}
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.text_encoder import _abs, encode_prompt_groups, load_config  # noqa: E402

TUMOR = ["tumor", "carcinoma", "malignant tumor", "an H&E stained image of tumor",
         "an H&E stained image of carcinoma"]
NORMAL = ["normal tissue", "benign tissue", "non-neoplastic tissue",
          "an H&E stained image of normal tissue", "an H&E stained image of benign tissue"]


def main() -> int:
    cfg = load_config()
    feats = encode_prompt_groups([TUMOR, NORMAL], cfg, device="cpu")
    assert feats.shape == (2, cfg["feat_dim"]) and feats.dtype == torch.float32
    out = _abs(cfg["f_txt_cache_dir"]) / "tumor_normal_keys.pt"
    torch.save({"features": feats.cpu(), "names": ["tumor", "normal"],
                "prompts": {"tumor": TUMOR, "normal": NORMAL}}, out)
    print(f"tumor/normal keys {tuple(feats.shape)} → {out}；cos(tumor, normal) = "
          f"{float(feats[0] @ feats[1]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
