#!/usr/bin/env python3
"""驗證 cache/text/f_txt_*.pt 與 CONCH text tower 重算結果逐元素相同（只在 Mac 跑）。

重算走 selector.text_encoder 的同一條路徑（class_prompts.json → CONCH text tower
→ 組內平均 → L2 normalize），但不寫回 cache。比對 f_txt、logit_scale、class_names。

    NAVCIL_MACHINE=mac python scripts/verify_text_cache.py
輸出：outputs/navcil/<machine>/text_cache_check.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.text_encoder import (_abs, _encode, _require_ckpt,  # noqa: E402
                                   class_prompt_ensemble, load_config)
from third_party.conch import build_text_tower                    # noqa: E402


@torch.no_grad()
def main() -> int:
    cfg = load_config()
    machine = cfg.get("machine")
    if not machine:
        raise SystemExit("請設定 NAVCIL_MACHINE")
    device = torch.device("cpu")
    tower, logit_scale, _dim = build_text_tower(
        _require_ckpt(cfg), model_cfg=cfg.get("conch_model_cfg", "conch_ViT-B-16"),
        device=device)
    rows, ok_all = [], True
    for task in cfg["tasks"]:
        cached = torch.load(_abs(cfg["f_txt_cache_dir"]) / f"f_txt_{task}.pt",
                            map_location="cpu")
        names, prompts = class_prompt_ensemble(task, cfg["class_prompt_path"])
        f_txt = torch.stack([F.normalize(_encode(tower, p, device).mean(0), dim=-1)
                             for p in prompts], 0)
        diff = (f_txt - cached["f_txt"]).abs().max().item()
        row = dict(task=task, shape=list(f_txt.shape),
                   f_txt_equal=bool(torch.equal(f_txt, cached["f_txt"])),
                   f_txt_max_abs_diff=diff,
                   logit_scale_equal=bool(torch.equal(logit_scale.cpu(),
                                                      cached["logit_scale"])),
                   logit_scale=float(logit_scale),
                   class_names_equal=names == list(cached["class_names"]))
        ok = row["f_txt_equal"] and row["logit_scale_equal"] and row["class_names_equal"]
        ok_all &= ok
        rows.append(row)
        print(f"  {task:10s} f_txt_equal={row['f_txt_equal']} max|Δ|={diff:.3e} "
              f"logit_scale_equal={row['logit_scale_equal']} names={row['class_names_equal']}")
    out = REPO_ROOT / "outputs" / "navcil" / machine
    out.mkdir(parents=True, exist_ok=True)
    (out / "text_cache_check.json").write_text(json.dumps(
        dict(torch=torch.__version__, all_equal=ok_all, tasks=rows), indent=1))
    print("ALL EQUAL" if ok_all else "MISMATCH — 停下回報")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
