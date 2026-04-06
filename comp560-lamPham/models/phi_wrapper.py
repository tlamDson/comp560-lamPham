from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PhiAdapterConfig:
    n_layer: int
    n_head: int
    n_embd: int
    block_size: int
    vocab_size: int
    dropout: float = 0.0
    intermediate_multiplier: int = 4


class PhiAdapter(nn.Module):
    """Adapter to make Phi CausalLM compatible with the existing training loop."""

    def __init__(self, config: PhiAdapterConfig, phi_source_dir: str):
        super().__init__()

        phi_path = Path(phi_source_dir)
        phi_dir = phi_path if phi_path.is_absolute() else (_WORKSPACE_ROOT / phi_path)
        phi_dir = phi_dir.resolve()
        if not phi_dir.exists():
            raise FileNotFoundError(
                f"Phi source/model directory not found: {phi_dir}. "
                "Set phi_model_source in your config to the Phi model folder."
            )

        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head for Phi attention")

        hf_cfg = AutoConfig.from_pretrained(str(phi_dir), trust_remote_code=True)
        hf_cfg.num_hidden_layers = config.n_layer
        hf_cfg.num_attention_heads = config.n_head
        hf_cfg.num_key_value_heads = config.n_head
        hf_cfg.hidden_size = config.n_embd
        hf_cfg.intermediate_size = config.n_embd * config.intermediate_multiplier
        if isinstance(getattr(hf_cfg, "rope_scaling", None), dict):
            if "type" not in hf_cfg.rope_scaling and "rope_type" in hf_cfg.rope_scaling:
                hf_cfg.rope_scaling["type"] = hf_cfg.rope_scaling["rope_type"]
            if hf_cfg.rope_scaling.get("type") not in {"longrope"}:
                hf_cfg.rope_scaling = None
        hf_cfg.max_position_embeddings = config.block_size
        hf_cfg.original_max_position_embeddings = config.block_size
        hf_cfg.vocab_size = config.vocab_size
        if getattr(hf_cfg, "pad_token_id", 0) is None or hf_cfg.pad_token_id >= config.vocab_size:
            hf_cfg.pad_token_id = 0
        if getattr(hf_cfg, "eos_token_id", 0) is None or hf_cfg.eos_token_id >= config.vocab_size:
            hf_cfg.eos_token_id = max(0, config.vocab_size - 1)
        if getattr(hf_cfg, "bos_token_id", 0) is None or hf_cfg.bos_token_id >= config.vocab_size:
            hf_cfg.bos_token_id = 0
        hf_cfg.resid_pdrop = config.dropout
        hf_cfg.embd_pdrop = config.dropout
        hf_cfg.attention_dropout = config.dropout

        self.model = AutoModelForCausalLM.from_config(hf_cfg, trust_remote_code=True)

        # Keep compatibility with existing code that accesses model.config.block_size.
        self.config = SimpleNamespace(block_size=config.block_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        labels = None
        if targets is not None:
            labels = targets.clone()
            labels[labels < 0] = -100

        outputs = self.model(input_ids=idx, labels=labels, use_cache=False)
        return outputs.logits, outputs.loss

    def crop_block_size(self, block_size: int):
        self.config.block_size = block_size
        if hasattr(self.model.config, "max_position_embeddings"):
            self.model.config.max_position_embeddings = block_size
        if hasattr(self.model.config, "original_max_position_embeddings"):
            self.model.config.original_max_position_embeddings = block_size

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        use_fused = fused_available and device_type == "cuda"
        extra_args = {"fused": True} if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # Phi adapter does not yet expose a precise MFU estimator compatible with NanoGPT's method.
        return 0.0


def build_phi_model(
    *,
    init_from: str,
    model_args: Dict,
    out_dir: str,
    device: str,
    phi_model_source: str,
) -> Tuple[nn.Module, Dict, dict | None, int, float]:
    iter_num = 0
    best_val_loss = 1e9
    checkpoint = None

    if init_from not in {"scratch", "resume"}:
        raise ValueError("Phi adapter currently supports init_from='scratch' or 'resume'")

    local_args = dict(model_args)
    local_args["model_family"] = "phi"
    local_args["phi_model_source"] = phi_model_source

    if local_args.get("vocab_size") is None:
        raise ValueError("vocab_size must be resolved from dataset meta before building Phi model")

    phi_cfg = PhiAdapterConfig(
        n_layer=local_args["n_layer"],
        n_head=local_args["n_head"],
        n_embd=local_args["n_embd"],
        block_size=local_args["block_size"],
        vocab_size=local_args["vocab_size"],
        dropout=local_args.get("dropout", 0.0),
    )
    model = PhiAdapter(config=phi_cfg, phi_source_dir=phi_model_source)

    if init_from == "resume":
        ckpt_path = Path(out_dir) / "ckpt.pt"
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    return model, local_args, checkpoint, iter_num, best_val_loss
