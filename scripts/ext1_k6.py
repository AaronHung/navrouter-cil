"""EXT-1 K6（PREREG-21 細則 7）：navrouter-cil 一方每折實際載入的 slide ID 集合的 sha256。

每個 (fold, task, split)：`Ctx.ds`（= selector.evaluate.slide_dataset）列出的 slide ID 排序後以 "\\n" 連接、UTF-8 編碼的 sha256。
另記切分檔與標籤表本身的 sha256（供只留下 patient 層級切分的一方做間接比對）。只建 dataset，不讀特徵檔。

    PYTHONNOUSERSITE=1 NAVCIL_MACHINE=mac python scripts/ext1_k6.py
輸出：outputs/navcil/<machine>/ext1/k6_navrouter.json（雜湊與張數）、k6_navrouter_ids.json（ID 清單，供列差異）
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import nc1_pipeline as P                                                  # noqa: E402

SPLITS = ("train", "val", "test")
FOLDS = range(1, 11)


def set_hash(ids) -> str:
    return hashlib.sha256("\n".join(sorted(str(s) for s in ids)).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ctx = P.Ctx(torch.device("cpu"))
    root = Path(ctx.cfg["dataset_root_dir"])
    out = ctx.out / "ext1"
    out.mkdir(parents=True, exist_ok=True)
    cells, ids, files = {}, {}, {}
    for task in ctx.tasks:
        table = Path(str(root) + ctx.cfg["path_table"].format(task, task.upper()))
        files[f"{task}/table"] = {"name": table.name, "sha256": file_hash(table)}
        for f in FOLDS:
            npz = Path(str(root) + ctx.cfg["path_split"].format(task, f))
            z = np.load(npz, allow_pickle=True)
            files[f"{task}/fold_{f}.npz"] = {
                "sha256": file_hash(npz),
                "patients": {k: {"n": len(z[f"{k}_patients"]), "sha256": set_hash(z[f"{k}_patients"])} for k in SPLITS}}
            for split in SPLITS:
                ds, _ = ctx.ds(f, task, split)
                sids = [str(s) for s in ds.sids]
                key = f"{f}|{task}|{split}"
                cells[key] = {"n": len(sids), "n_unique": len(set(sids)), "sha256": set_hash(sids)}
                ids[key] = sorted(sids)
    res = {"definition": "sha256('\\n'.join(sorted(slide_ids)).encode('utf-8'))；slide ID = WSIClf.sids 的字串",
           "tasks": ctx.tasks, "folds": list(FOLDS), "splits": list(SPLITS), "cells": cells, "files": files}
    (out / "k6_navrouter.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "k6_navrouter_ids.json").write_text(json.dumps(ids, ensure_ascii=False))
    n = {s: sum(v["n"] for k, v in cells.items() if k.endswith(s) and k.startswith("1|")) for s in SPLITS}
    print(f"cells={len(cells)} fold1 張數 {n}；test 十折合計 {sum(v['n'] for k, v in cells.items() if k.endswith('|test'))}")


if __name__ == "__main__":
    main()
