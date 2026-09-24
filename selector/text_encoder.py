"""f_txt — class text features, rebuilt from the CONCH text tower alone.

以 CONCH 的 **text tower**（不碰 vision tower、不碰任何外部方法模組）重建每個
類別的文字特徵：

    data/class_prompts.json  →  當前 task 的 class prompt ensemble
                             →  CONCH text tower 編碼
                             →  每個類別的多個 prompt 取平均
                             →  L2 normalize
                             →  f_txt [C, 512]

logit_scale 直接取自 CONCH checkpoint 內的 `logit_scale`（已 exp），不自訂常數。
結果 cache 到 cache/text/f_txt_{task}.pt（commit 進 repo），之後直接讀。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:                # 讓 third_party/ 可被 import
    sys.path.insert(0, str(_REPO_ROOT))

from third_party.conch import build_text_tower, get_tokenizer, tokenize  # noqa: E402

DEFAULT_CONFIG = _REPO_ROOT / "configs" / "base.yaml"
MACHINE_ENV = "NAVCIL_MACHINE"                # 選 configs/machine_<name>.yaml


@dataclass
class ClassTextFeatures:
    """一個 task 的類別文字特徵。"""
    f_txt: torch.Tensor            # [C, D]，已 L2 normalize
    logit_scale: torch.Tensor      # scalar，取自 CONCH（已 exp）
    class_names: list[str]         # 長度 C，順序即 label 順序
    task: str

    def to(self, device) -> "ClassTextFeatures":
        return ClassTextFeatures(self.f_txt.to(device), self.logit_scale.to(device),
                                 self.class_names, self.task)


def load_config(path: str | os.PathLike | None = None,
                machine: str | None = None) -> dict:
    """base.yaml，再以 configs/machine_<name>.yaml 覆寫同名鍵。

    machine 未指定時讀環境變數 NAVCIL_MACHINE；兩者皆無則只回 base（不含機器
    相關路徑，例如 dataset_root_dir）。程式裡不寫任何絕對路徑。
    """
    with open(path or DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    machine = machine or os.environ.get(MACHINE_ENV)
    if machine:
        mpath = _REPO_ROOT / "configs" / f"machine_{machine}.yaml"
        if not mpath.exists():
            raise FileNotFoundError(f"{MACHINE_ENV}={machine} 但 {mpath} 不存在")
        with open(mpath) as f:
            cfg.update(yaml.safe_load(f) or {})
        cfg["machine"] = machine
    return cfg


def _abs(cfg_path: str | os.PathLike) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else _REPO_ROOT / p


def class_prompt_ensemble(task: str, class_prompt_path: str | os.PathLike
                          ) -> tuple[list[str], list[list[str]]]:
    """回傳 (class_names, prompts_per_class)。

    prompts_per_class[i] = 第 i 類的所有 prompt 字串
    （每個 classname × 每個 template，句尾補 '.'，與 CONCH zero-shot 慣例一致）。
    """
    with open(_abs(class_prompt_path)) as f:
        spec = json.load(f)["0"]
    classnames = spec["classnames"]
    templates = spec["templates"]
    if task not in classnames:
        raise KeyError(f"task '{task}' not in {class_prompt_path}; "
                       f"available: {sorted(classnames)}")

    class_names, prompts_per_class = [], []
    for subtype, names in classnames[task].items():
        class_names.append(subtype)
        prompts_per_class.append([
            t.replace("CLASSNAME", name) + "." for name in names for t in templates
        ])
    return class_names, prompts_per_class


@torch.no_grad()
def _encode(text_tower, prompts: list[str], device, batch_size: int = 64) -> torch.Tensor:
    """CONCH text tower 編碼 → [N, D]，每條 prompt 已 L2 normalize。"""
    tokenizer = get_tokenizer()
    out = []
    for i in range(0, len(prompts), batch_size):
        tokens = tokenize(tokenizer, prompts[i:i + batch_size]).to(device)
        pooled, _ = text_tower(tokens[:, :-1])     # 末位 pad 讓出 <cls>，同 CoCa.encode_text
        out.append(F.normalize(pooled.float(), dim=-1))
    return torch.cat(out, dim=0)


def encode_prompt_groups(groups: list[list[str]], cfg: dict | None = None, *,
                         device: str | torch.device = "cpu",
                         cache_name: str | None = None,
                         refresh: bool = False) -> torch.Tensor:
    """把「每組多條 prompt」編成 [G, D]：組內平均後 L2 normalize。

    與 build_f_txt 同一條路徑（CONCH text tower、同樣的平均與正規化），只是
    prompt 來源由呼叫端給定，供 tissue prototype 之類的用途重用。
    cache_name 給定時結果快取到 {f_txt_cache_dir}/{cache_name}.pt。
    """
    cfg = cfg or load_config()
    cache_path = (_abs(cfg["f_txt_cache_dir"]) / f"{cache_name}.pt") if cache_name else None
    if cache_path is not None and cache_path.exists() and not refresh:
        return torch.load(cache_path, map_location="cpu")["features"].to(device)

    text_tower, _logit_scale, _dim = build_text_tower(
        _require_ckpt(cfg), model_cfg=cfg.get("conch_model_cfg", "conch_ViT-B-16"),
        device=device)
    rows = [F.normalize(_encode(text_tower, prompts, device).mean(0), dim=-1)
            for prompts in groups]
    feats = torch.stack(rows, dim=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"features": feats.cpu(), "n_groups": len(groups)}, cache_path)
        print(f"[text_encoder] {cache_name}{tuple(feats.shape)} → {cache_path}")
    return feats


def _require_ckpt(cfg: dict) -> str:
    ckpt = cfg.get("conch_ckpt_path")
    if not ckpt or not Path(ckpt).exists():
        raise FileNotFoundError(
            f"CONCH checkpoint not found: {ckpt}\n"
            f"請在 configs/machine_<name>.yaml 設定 conch_ckpt_path"
            f"（只有 Mac 需要；其他機器直接讀 cache/text/）。")
    return ckpt


def build_f_txt(task: str, cfg: dict | None = None, *,
                              device: str | torch.device = "cpu",
                              refresh: bool = False) -> ClassTextFeatures:
    """建立（或讀 cache）某個 task 的 f_txt。"""
    cfg = cfg or load_config()
    cache_path = _abs(cfg["f_txt_cache_dir"]) / f"f_txt_{task}.pt"

    if cache_path.exists() and not refresh:
        blob = torch.load(cache_path, map_location="cpu")
        return ClassTextFeatures(blob["f_txt"].to(device),
                                 blob["logit_scale"].to(device),
                                 list(blob["class_names"]), task)

    text_tower, logit_scale, embed_dim = build_text_tower(
        _require_ckpt(cfg), model_cfg=cfg.get("conch_model_cfg", "conch_ViT-B-16"),
        device=device)

    class_names, prompts_per_class = class_prompt_ensemble(task, cfg["class_prompt_path"])
    rows = []
    for prompts in prompts_per_class:
        emb = _encode(text_tower, prompts, device)          # [P, D]
        rows.append(F.normalize(emb.mean(0), dim=-1))       # 取平均 → L2 normalize
    f_txt = torch.stack(rows, dim=0)                        # [C, D]
    assert f_txt.shape[1] == embed_dim == cfg["feat_dim"], (f_txt.shape, embed_dim)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"f_txt": f_txt.cpu(), "logit_scale": logit_scale.cpu(),
                "class_names": class_names, "task": task}, cache_path)
    print(f"[text_encoder] f_txt{tuple(f_txt.shape)} for '{task}' → {cache_path}")
    return ClassTextFeatures(f_txt, logit_scale, class_names, task)


if __name__ == "__main__":                                   # 建 cache 用
    conf = load_config()
    for t in sys.argv[1:] or conf["tasks"]:
        ctf = build_f_txt(t, conf, refresh=True)
        print(f"  {t}: {ctf.class_names}  f_txt{tuple(ctf.f_txt.shape)}  "
              f"logit_scale={float(ctf.logit_scale):.4f}  first3={ctf.f_txt[0, :3].tolist()}")
