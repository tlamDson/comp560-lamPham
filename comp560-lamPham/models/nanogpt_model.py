from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NANOGPT_ROOT = _REPO_ROOT / "comp560-nanoGPT"
if str(_NANOGPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_NANOGPT_ROOT))

from model import GPT, GPTConfig  # noqa: E402


def build_nanogpt_model(
    *,
    init_from: str,
    model_args: Dict,
    meta_vocab_size: int | None,
    dropout: float,
    out_dir: str,
    device: str,
) -> Tuple[torch.nn.Module, Dict, dict | None, int, float]:
    iter_num = 0
    best_val_loss = 1e9
    checkpoint = None

    if init_from == "scratch":
        local_args = dict(model_args)
        if meta_vocab_size is None:
            print("defaulting to vocab_size=50304")
        local_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**local_args)
        model = GPT(gptconf)
        return model, local_args, checkpoint, iter_num, best_val_loss

    if init_from == "resume":
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]

        local_args = dict(model_args)
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            local_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**local_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        return model, local_args, checkpoint, iter_num, best_val_loss

    if init_from.startswith("gpt2"):
        model = GPT.from_pretrained(init_from, dict(dropout=dropout))
        local_args = dict(model_args)
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            local_args[k] = getattr(model.config, k)
        return model, local_args, checkpoint, iter_num, best_val_loss

    raise ValueError(f"Unsupported init_from mode for NanoGPT: {init_from}")
