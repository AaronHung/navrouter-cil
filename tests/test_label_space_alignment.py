"""P-B — 8-way label space 疊放順序必須與 reference/v9 對齊。

評估在 8-way label space：f_txt 每個 task 是 [2, 512]，評估時依任務序疊成 [8, 512]。
只要疊放順序與存檔不同，排在後面的 task 錯位最嚴重 —— 所以這件事必須先釘死，
才有資格談 DELTA_v9 的 delta。

對齊的四個環節，缺一不可：
  1. 任務序          reverse = esca → rcc → brca → lung
  2. 每個 task 內的類別序   來自 class_prompts.json 的 classnames 插入序
  3. label shift     shift = 2 * task_pos
  4. 表格 label 語意  label 0/1 對應該 task 的第 1/2 個 classname
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.v9_reference import task_order as v9_task_order
from selector.text_encoder import class_prompt_ensemble, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# reference/v9 產出時使用的 class ensemble JSON 的 sha256。
# 來源：v9 checkpoint qpmil_reverse_fold1.pt 的 qpmil_cfg["class_ensemble_path"]
# （本機對應 01_navipath/QPMIL-VL/class_ensemble/class_ensemble.json）。
# 本 repo 的 data/class_prompts.json 必須與它 byte-for-byte 相同。
V9_CLASS_ENSEMBLE_SHA256 = \
    "09b6a32d6a572516abde28be494db6e0c67974f663b7ccc8e257c88c687a6a8a"

# v9 的 8-way label space（依 reverse 任務序展開），列索引即 label。
EXPECTED_LABEL_SPACE = [
    (0, "tcga_esca", "ESAD"), (1, "tcga_esca", "ESCC"),
    (2, "tcga_rcc", "CCRCC"), (3, "tcga_rcc", "PRCC"),
    (4, "tcga_brca", "IDC"),  (5, "tcga_brca", "ILC"),
    (6, "tcga_lung", "LUAD"), (7, "tcga_lung", "LUSC"),
]


def stacked_label_space(cfg) -> list[tuple[int, str, str]]:
    """我們疊出來的 8 個類別：(列索引, task, class name)。"""
    out = []
    for task_pos, task in enumerate(cfg["tasks"]):
        names, _prompts = class_prompt_ensemble(task, cfg["class_prompt_path"])
        for j, name in enumerate(names):
            out.append((2 * task_pos + j, task, name))
    return out


def test_class_prompts_file_is_the_one_v9_used():
    """data/class_prompts.json 必須與 v9 用的那份完全相同，否則類別序無從對齊。"""
    blob = (REPO_ROOT / "data" / "class_prompts.json").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == V9_CLASS_ENSEMBLE_SHA256


def test_task_order_matches_v9_archive():
    """任務序必須等於存檔檔名 task0..task3 的順序。"""
    cfg = load_config()
    assert cfg["tasks"] == v9_task_order()
    assert cfg["tasks"] == ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"]


def test_stacked_label_space_matches_v9():
    """逐列比對疊放結果。"""
    assert stacked_label_space(load_config()) == EXPECTED_LABEL_SPACE


def test_each_task_occupies_rows_2p_and_2p_plus_1():
    """shift = 2 * task_pos：每個 task 剛好佔第 2p、2p+1 列。"""
    cfg = load_config()
    rows = stacked_label_space(cfg)
    for task_pos, task in enumerate(cfg["tasks"]):
        owned = [r for r in rows if r[1] == task]
        assert [r[0] for r in owned] == [2 * task_pos, 2 * task_pos + 1]
    assert [r[0] for r in rows] == list(range(8))


def test_lung_is_last_and_owns_rows_6_and_7():
    """錯位最嚴重的必然是排最後的 task，單獨釘死 lung。"""
    cfg = load_config()
    assert cfg["tasks"][3] == "tcga_lung"
    lung = [r for r in stacked_label_space(cfg) if r[1] == "tcga_lung"]
    assert lung == [(6, "tcga_lung", "LUAD"), (7, "tcga_lung", "LUSC")]


def _table_path(cfg, task):
    if "dataset_root_dir" not in cfg:           # navcil：未設 NAVCIL_MACHINE
        pytest.skip("NAVCIL_MACHINE not set; dataset_root_dir unknown")
    return Path(cfg["dataset_root_dir"] + cfg["path_table"].format(task, task.upper()))


@pytest.mark.parametrize("task", ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"])
def test_table_label_semantics_match_classname_order(task):
    """表格的 label 0/1 必須對應 classnames 的第 1/2 個名字。"""
    cfg = load_config()
    path = _table_path(cfg, task)
    if not path.exists():
        pytest.skip(f"dataset not available: {path}")
    import pandas as pd

    df = pd.read_csv(path)
    names, _ = class_prompt_ensemble(task, cfg["class_prompt_path"])
    for local_label, name in enumerate(names):
        subtypes = set(df.loc[df["label"] == local_label, "subtype"])
        assert subtypes == {name}, f"{task} label={local_label} → {subtypes}, 期望 {name}"


@pytest.mark.parametrize("task_pos,task", list(enumerate(
    ["tcga_esca", "tcga_rcc", "tcga_brca", "tcga_lung"])))
def test_test_split_labels_land_in_the_owned_rows(task_pos, task):
    """test split 的全域 label 必須全部落在該 task 自己的兩列上。

    rcc 的表格另有第三個 subtype（CHRCC, label=2），若它混進 test split，
    shift 後會落到別的 task 的列上 —— 這條測試就是為了擋這件事。
    """
    cfg = load_config()
    if not _table_path(cfg, task).exists():
        pytest.skip("dataset not available")
    from selector.evaluate import iter_test_slides

    owned = {2 * task_pos, 2 * task_pos + 1}
    seen = {rec.label for rec in iter_test_slides(cfg, task, task_pos)}
    assert seen <= owned, f"{task} 出現不屬於本 task 的 label：{sorted(seen - owned)}"
