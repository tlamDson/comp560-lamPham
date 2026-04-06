from __future__ import annotations

from typing import Dict, Tuple

import torch

from .nanogpt_model import build_nanogpt_model
from .phi_wrapper import build_phi_model


def build_model(
    *,
    model_family: str,
    init_from: str,
    model_args: Dict,
    meta_vocab_size: int | None,
    dropout: float,
    out_dir: str,
    device: str,
    phi_model_source: str,
) -> Tuple[torch.nn.Module, Dict, dict | None, int, float]:
    family = model_family.lower().strip()

    local_args = dict(model_args)
    if local_args.get("vocab_size") is None:
        local_args["vocab_size"] = meta_vocab_size

    if family == "nanogpt":
        return build_nanogpt_model(
            init_from=init_from,
            model_args=local_args,
            meta_vocab_size=meta_vocab_size,
            dropout=dropout,
            out_dir=out_dir,
            device=device,
        )

    if family == "phi":
        return build_phi_model(
            init_from=init_from,
            model_args=local_args,
            out_dir=out_dir,
            device=device,
            phi_model_source=phi_model_source,
        )

    raise ValueError(f"Unsupported model_family: {model_family}. Use 'nanogpt' or 'phi'.")
