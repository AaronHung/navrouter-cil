"""Text-only loader for the CONCH VLM.

We deliberately build **only** the text tower of CONCH's CoCa model: the class
text features f_txt need nothing else, and skipping the vision tower keeps this
package free of the timm / torchvision dependencies that the full CONCH API
pulls in.

The tower itself (`transformer.TextTransformer`) and the tokenizer are verbatim
copies of the official CONCH API — see PROVENANCE.md.  The construction below
mirrors `conch.coca_model._build_text_tower` and `conch.factory.read_state_dict`.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .transformer import TextTransformer

CFG_DIR = Path(__file__).parent / "model_configs"
DEFAULT_MODEL_CFG = "conch_ViT-B-16"


def read_state_dict(checkpoint_path: str, map_location="cpu") -> dict:
    """Verbatim from conch.factory.read_state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def build_text_tower(checkpoint_path: str,
                     model_cfg: str = DEFAULT_MODEL_CFG,
                     device: str | torch.device = "cpu"):
    """Load the CONCH text tower and its logit_scale from a CoCa checkpoint.

    Returns (text_tower, logit_scale, embed_dim) where
      text_tower(tokens) -> (pooled [N, embed_dim], token_embs)
      logit_scale        -> scalar tensor, already exp()'d (i.e. 1 / temperature)
    """
    with open(CFG_DIR / f"{model_cfg}.json") as f:
        cfg = json.load(f)
    embed_dim = cfg["embed_dim"]
    text_cfg = cfg["text_cfg"]

    # mirrors conch.coca_model._build_text_tower (act/norm layers included)
    text = TextTransformer(
        context_length=text_cfg["context_length"],
        vocab_size=text_cfg["vocab_size"],
        width=text_cfg["width"],
        heads=text_cfg["heads"],
        layers=text_cfg["layers"],
        ls_init_value=text_cfg.get("ls_init_value"),
        output_dim=embed_dim,
        embed_cls=text_cfg.get("embed_cls", False),
        output_tokens=text_cfg.get("output_tokens", False),
        pad_id=text_cfg.get("pad_id", 0),
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    )

    state = read_state_dict(checkpoint_path)
    text_state = {k[len("text."):]: v for k, v in state.items() if k.startswith("text.")}
    if not text_state:
        raise RuntimeError(
            f"no 'text.*' weights found in {checkpoint_path} — not a CONCH CoCa checkpoint?"
        )
    missing, unexpected = text.load_state_dict(text_state, strict=False)
    missing = [k for k in missing if k != "attn_mask"]        # non-persistent buffer
    if missing:
        raise RuntimeError(f"CONCH text tower is missing weights: {missing}")

    if "logit_scale" not in state:
        raise RuntimeError(f"no 'logit_scale' in {checkpoint_path}")
    logit_scale = state["logit_scale"].to(device).float().exp()

    text = text.to(device).eval()
    for p in text.parameters():
        p.requires_grad_(False)
    return text, logit_scale, embed_dim
