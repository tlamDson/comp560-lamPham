# train a miniature 3-digit addition model (SPEEDRUN MODE)

out_dir = 'out'
eval_interval = 1000  # Minimal eval
eval_iters = 5  # Very fast eval
log_interval = 200  # Minimal logging

always_save_checkpoint = False

wandb_log = False
wandb_project = 'arithmetic'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 1024  # Max out VRAM
block_size = 16

# Smaller model = faster training
n_layer = 3  # Reduced from 4
n_head = 4
n_embd = 192  # Reduced from 256
dropout = 0.0

# Training settings - AGGRESSIVE
learning_rate = 6e-3  # Very high LR
max_iters = 5000  # Slightly more for 100%
lr_decay_iters = 5000
min_lr = 6e-4
beta2 = 0.99  # More momentum

warmup_iters = 50  # Very short warmup

# Early Stopping: Disabled for now - low loss doesn't guarantee 100% accuracy
# Set to 0 to disable, or use very low value like 0.0001
early_stop_loss = 0.0  # Disabled - let it train full iterations

import torch
import sys

# Auto-detect CUDA availability (fallback to CPU if CUDA not available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

# Disable compile - for short runs, JIT overhead doesn't pay off
compile = False
