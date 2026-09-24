#!/usr/bin/env python3
"""第二關 (2) text-organ 的任務 key：只在 Mac 用 CONCH text tower 算一次並 commit。

模板 "{organ}"、"{organ} tissue"、"an H&E stained image of {organ} tissue"；每任務的
三條 prompt 走 `encode_prompt_groups`（與 build_f_txt 同一條路徑：CONCH text tower →
每條 L2 normalize → 組內平均 → L2 normalize）。模板照原文，不另加句點。

    NAVCIL_MACHINE=mac python scripts/nc1_organ_keys.py
輸出：cache/text/organ_keys.pt  {"features": [4, 512], "tasks", "organs", "templates"}
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.text_encoder import _abs, encode_prompt_groups, load_config  # noqa: E402

ORGANS = {"tcga_esca": "esophagus", "tcga_rcc": "kidney",
          "tcga_brca": "breast", "tcga_lung": "lung"}
TEMPLATES = ["{organ}", "{organ} tissue", "an H&E stained image of {organ} tissue"]


def main() -> int:
    cfg = load_config()
    tasks = list(cfg["tasks"])
    groups = [[t.format(organ=ORGANS[task]) for t in TEMPLATES] for task in tasks]
    feats = encode_prompt_groups(groups, cfg, device="cpu")
    assert feats.shape == (len(tasks), cfg["feat_dim"]) and feats.dtype == torch.float32
    out = _abs(cfg["f_txt_cache_dir"]) / "organ_keys.pt"
    torch.save({"features": feats.cpu(), "tasks": tasks,
                "organs": [ORGANS[t] for t in tasks], "templates": TEMPLATES,
                "prompts": groups}, out)
    print(f"organ keys {tuple(feats.shape)} → {out}")
    print("cos(organ_i, organ_j):")
    print((feats @ feats.t()).numpy().round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
