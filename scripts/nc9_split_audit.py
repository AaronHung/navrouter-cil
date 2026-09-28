#!/usr/bin/env python3
"""NC-9 唯讀確認：本 repo 十折 split 檔與 pathselect repo 所用 split 檔的 sha256 與逐折病人名單比對。

pathselect 側只讀其 configs/pathselect.yaml 的 dataset_root_dir、path_split 與 split 檔本身，不 import 其程式。
    NAVCIL_MACHINE=mac python scripts/nc9_split_audit.py --pathselect-root <repo> [--pathselect-root <repo> ...]
輸出：outputs/navcil/<machine>/nc9/split_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.table_utils import read_datasplit_npz                           # noqa: E402
from selector.text_encoder import load_config                             # noqa: E402

FOLDS = list(range(1, 11))
KEYS = ("train", "val", "test")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pids(p: Path) -> dict:
    return dict(zip(KEYS, read_datasplit_npz(str(p))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathselect-root", action="append", required=True)
    args = ap.parse_args()
    cfg = load_config()
    tasks = list(cfg["tasks"])
    ours = {(t, f): Path(cfg["dataset_root_dir"] + cfg["path_split"].format(t, f)) for t in tasks for f in FOLDS}
    out = {"navcil": {"config": "configs/base.yaml + configs/machine_%s.yaml" % cfg["machine"],
                      "dataset_root_dir": cfg["dataset_root_dir"], "path_split": cfg["path_split"],
                      "files": {f"{t}/fold_{f}": {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size,
                                                  "n": {k: len(v or []) for k, v in pids(p).items()}}
                                for (t, f), p in ours.items()}},
           "pathselect": {}}
    for root in args.pathselect_root:
        root = Path(os.path.expanduser(root)).resolve()
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        pc = yaml.safe_load((root / "configs" / "pathselect.yaml").read_text())
        rows, n_diff = {}, 0
        for (t, f), p_ours in ours.items():
            p = Path(pc["dataset_root_dir"] + pc["path_split"].format(t, f))
            a, b = pids(p_ours), pids(p)
            diff = {k: {"only_navcil": sorted(set(a[k] or []) - set(b[k] or [])),
                        "only_pathselect": sorted(set(b[k] or []) - set(a[k] or []))} for k in KEYS}
            same_lists = all(a[k] == b[k] for k in KEYS)
            rows[f"{t}/fold_{f}"] = {"path": str(p), "same_realpath": os.path.realpath(p) == os.path.realpath(p_ours),
                                     "sha256": sha256(p), "sha256_equal": sha256(p) == sha256(p_ours),
                                     "patient_lists_equal": same_lists,
                                     "diff": {k: v for k, v in diff.items() if v["only_navcil"] or v["only_pathselect"]}}
            n_diff += not (rows[f"{t}/fold_{f}"]["sha256_equal"] and same_lists)
        out["pathselect"][str(root)] = {"head": head, "config": "configs/pathselect.yaml",
                                        "dataset_root_dir": pc["dataset_root_dir"], "path_split": pc["path_split"],
                                        "n_files": len(rows), "n_different": n_diff, "files": rows}
    d = REPO_ROOT / "outputs" / "navcil" / cfg["machine"] / "nc9"
    d.mkdir(parents=True, exist_ok=True)
    (d / "split_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for r, v in out["pathselect"].items():
        print(f"{r} @{v['head'][:7]}: {v['n_files']} files, {v['n_different']} different")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
