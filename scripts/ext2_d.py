#!/usr/bin/env python3
"""EXT-2 D1 與 D2／D3 的條件檢查（PREREG-22 細則 23、24）。只列目錄、讀特徵檔的型別與 shape、比對檔名；不讀切片、不下載。

    NAVCIL_MACHINE=mac python scripts/ext2_d.py
輸出：outputs/navcil/<machine>/ext2/d.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from selector.evaluate import slide_dataset                               # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

SLIDE_EXT = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs")
FOLD = 1


def main() -> int:
    cfg = load_config()
    root = Path(cfg["dataset_root_dir"])
    wsi = root.parent / "WSI_data"
    out = {"stage": "d", "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_root": str(root), "path_feat": cfg["path_feat"],
           "conch_path_feat": cfg["conch_path_feat"], "feat_format": cfg["feat_format"], "tasks": {}}
    feat_sids = {}
    for task in cfg["tasks"]:
        tdir = root / task
        files = [p for p in tdir.rglob("*") if p.is_file()]
        ext = Counter(p.suffix for p in files)
        fdir = Path(str(root) + cfg["path_feat"].format(task, cfg["conch_path_feat"]))
        pts = sorted(fdir.glob("*.pt"))
        x = torch.load(pts[0], map_location="cpu", weights_only=False)
        rec = {"subdirs": sorted(str(p.relative_to(tdir)) for p in tdir.rglob("*") if p.is_dir()), "file_ext_counts": dict(ext),
               "feat_dir": str(fdir.relative_to(root)), "n_feat_files": len(pts), "example_file": pts[0].name,
               "python_type": type(x).__name__, "shape": list(x.shape) if hasattr(x, "shape") else None,
               "dtype": str(x.dtype) if hasattr(x, "dtype") else None, "keys": sorted(map(str, x.keys())) if isinstance(x, dict) else None,
               "coord_like_files": sorted(str(p.relative_to(tdir)) for p in files if p.suffix == ".h5" or "coord" in p.name.lower())}
        rec["has_coords"] = bool(rec["coord_like_files"]) or bool(rec["keys"] and any("coord" in k.lower() for k in rec["keys"]))
        out["tasks"][task] = rec
        feat_sids[task] = {p.stem for p in pts}
    out["has_coords_any"] = any(v["has_coords"] for v in out["tasks"].values())
    out["magnification_strings"] = {"feat_dir_name": cfg["conch_path_feat"] and f"feats-l1-s256_{cfg['conch_path_feat']}",
                                    "table_name": "TCGA_<TASK>_path_subtype_x10_processed.csv",
                                    "note": "目錄名的 l1、s256 與標籤表檔名的 x10 是檔名字樣；特徵檔內沒有倍率欄位"}

    # 本機切片（只比對檔名）
    slides = sorted(p for p in wsi.rglob("*") if p.suffix.lower() in SLIDE_EXT) if wsi.is_dir() else []
    loc = []
    test_sids = {}
    for p_, task in enumerate(cfg["tasks"]):
        ds, _ = slide_dataset(dict(cfg, fold=FOLD), task, p_, "test")
        test_sids[task] = {str(s) for s in ds.sids}
    for p in slides:
        sid = p.stem
        loc.append({"file": str(p.relative_to(wsi)), "size_bytes": p.stat().st_size,
                    "in_feats": [t for t in cfg["tasks"] if sid in feat_sids[t]], "in_fold1_test": [t for t in cfg["tasks"] if sid in test_sids[t]]})
    # 重疊的 slide 在 fold 1、reverse 序、t = 4 的 CIL 判定（讀 A 階段的逐折輸出；只供之後取座標時參考）
    base = REPO_ROOT / "outputs" / "navcil" / cfg["machine"]
    a_dir, m0 = base / "ext2" / "a", base / "moe0" / f"test_fold{FOLD}.pt"
    if m0.exists() and (a_dir / "FINALB" / f"reverse_fold{FOLD}.json").exists():
        raw = torch.load(m0, map_location="cpu", weights_only=False)
        order = [s for t in cfg["tasks"] for s in raw[t]["sids"]]
        okB = json.loads((a_dir / "FINALB" / f"reverse_fold{FOLD}.json").read_text())["ok4"]
        okA = json.loads((a_dir / "FINAL" / f"reverse_fold{FOLD}.json").read_text())["ok4"]
        for x in loc:
            sid = Path(x["file"]).stem
            if x["in_fold1_test"] and sid in order:
                i = order.index(sid)
                x["fold1_cil_correct"] = {"FINALB": bool(okB[i]), "FINALA_seed42": bool(okA[i])}
    out["local_slides"] = loc
    out["n_local_slides"] = len(loc)
    out["n_local_in_feats"] = sum(bool(x["in_feats"]) for x in loc)
    out["n_local_in_fold1_test"] = sum(bool(x["in_fold1_test"]) for x in loc)
    out["other_coord_files_under_WSI_data"] = sorted(str(p.relative_to(wsi)) for p in wsi.rglob("*")
                                                     if p.is_file() and (p.suffix == ".h5" or "coord" in p.name.lower()))[:50] if wsi.is_dir() else []
    out["libs"] = {m: importlib.util.find_spec(m) is not None for m in ("openslide", "tifffile")}
    out["D2"] = {"done": False, "reason": "特徵檔是單一 tensor [N, 512]，沒有逐 patch 座標；can_dataset 下沒有 .h5 或含 coord 字樣的檔"} \
        if not out["has_coords_any"] else {"done": False, "reason": "有座標；D2 需另寫程式（本檔只做條件檢查）"}
    out["D3"] = {"done": False, "reason": "D2 沒有做（沒有座標可畫）" + ("；且 openslide、tifffile 都未安裝" if not any(out["libs"].values()) else "")}
    p = REPO_ROOT / "outputs" / "navcil" / cfg["machine"] / "ext2" / "d.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("has_coords_any", "n_local_slides", "n_local_in_feats", "n_local_in_fold1_test", "libs", "D2", "D3",
                                           "other_coord_files_under_WSI_data")}, indent=1, ensure_ascii=False))
    for t, v in out["tasks"].items():
        print(t, v["file_ext_counts"], v["python_type"], v["shape"], v["dtype"], v["coord_like_files"])
    for x in loc:
        print(x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
