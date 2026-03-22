"""
Project-owned training entrypoint for arithmetic tasks.

This keeps task-specific training behavior in comp560-lamPham while
continuing to reuse the upstream nanoGPT model implementation.
"""

import os
import sys
import time
import math
import pickle
import gc
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Runtime speed knobs (same intent as upstream train.py)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_sync_debug_mode(0)

# Import model from read-only upstream dependency.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NANOGPT_ROOT = _REPO_ROOT / "comp560-nanoGPT"
sys.path.insert(0, str(_NANOGPT_ROOT))

from model import GPTConfig, GPT  # noqa: E402


# -----------------------------------------------------------------------------
# Defaults (overridden by task config file and optional --key=value CLI args)
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"

# wandb logging
wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2"

# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
sample_stride = 16

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# DDP settings
backend = "nccl"

# system
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
compile = True

# task-specific policy toggles
early_stop_loss = 0.0
early_stop_acc = 0.99
answer_only_loss = False


# -----------------------------------------------------------------------------
def _coerce_override(raw_value, old_value):
    if isinstance(old_value, bool):
        return raw_value.lower() in {"1", "true", "yes", "on"}
    if isinstance(old_value, int):
        return int(raw_value)
    if isinstance(old_value, float):
        return float(raw_value)
    return raw_value


def load_runtime_config():
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]

    config_file = os.environ.get("LAM_CONFIG")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        config_file = sys.argv[1]
    if config_file is None:
        raise ValueError("No config file provided. Use: python common/train.py <config.py> [--key=value]")

    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Apply config file values.
    exec(config_path.read_text(), globals())

    # Apply command-line overrides.
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            continue
        key, value = arg.split("=", 1)
        key = key[2:]
        if key not in config_keys:
            raise ValueError(f"Unknown config key: {key}")
        globals()[key] = _coerce_override(value, globals()[key])

    return {k: globals()[k] for k in config_keys}


config = load_runtime_config()

# -----------------------------------------------------------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

data_dir = os.path.join("data", dataset)
train_data = torch.from_numpy(np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16).astype(np.int64))
val_data = torch.from_numpy(np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16).astype(np.int64))


iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
eq_token_id = None
nl_token_id = None

if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    if answer_only_loss and "stoi" in meta:
        stoi = meta["stoi"]
        eq_token_id = stoi.get("=")
        nl_token_id = stoi.get("\n")
        if eq_token_id is not None:
            print(
                f"answer_only_loss ENABLED: '=' token_id={eq_token_id}, '\\n' token_id={nl_token_id}"
            )
            print("  -> Loss computed on answer region only.")
        else:
            print("WARNING: answer_only_loss enabled but '=' not found. Falling back to full loss.")
            answer_only_loss = False


def get_batch(split):
    data = train_data if split == "train" else val_data
    nums_lines = (len(data) - block_size) // sample_stride
    ix = torch.randint(nums_lines, (batch_size,)) * sample_stride
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])

    if answer_only_loss and eq_token_id is not None:
        transitions = (x == eq_token_id).long() - (x == nl_token_id).long()
        state = transitions.cumsum(dim=1)
        prompt_mask = state <= 0
        if nl_token_id is not None:
            prompt_mask = prompt_mask | (y == nl_token_id)
        y = y.clone()
        y[prompt_mask] = -1

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size=50304")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"Unsupported init_from mode: {init_from}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])

checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        accuracies = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                logits, loss = model(x, y)
            losses[k] = loss.item()

            preds = torch.argmax(logits, dim=-1)
            mask = y != -1
            match = (preds == y) | ~mask
            seq_acc = match.all(dim=-1).float().mean()
            accuracies[k] = seq_acc.item()

        out[split] = losses.mean()
        out[f"{split}_acc"] = accuracies.mean()

    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


gc.collect()
gc.disable()

x, y = get_batch("train")
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}, val acc {losses['val_acc']*100:.2f}%"
        )

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "val/acc": losses["val_acc"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                }
            )

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

        if losses["val_acc"] >= early_stop_acc:
            print(
                f"Early stopping triggered! val_acc {losses['val_acc']*100:.2f}% "
                f">= threshold {early_stop_acc*100:.2f}%"
            )
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": losses["val"],
                "config": config,
            }
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            print(f"Final checkpoint saved. Training complete in {iter_num} iterations.")
            break

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        x, y = get_batch("train")
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
