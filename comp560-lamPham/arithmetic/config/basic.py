# train a miniature 3-digit addition model (SPEEDRUN MODE)

out_dir = 'out'
eval_interval = 50
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'arithmetic'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 256  # Increased from 64 to maximize GPU utilization
block_size = 16

n_layer = 6
n_head = 6
n_embd = 192
dropout = 0.0

learning_rate = 3e-3  # Balanced: faster than 1e-3, safer than 5e-3
max_iters = 6000  # Increased from 4000 (needs more iterations to converge)
lr_decay_iters = 6000
min_lr = 3e-4  # Proportional to new learning_rate
beta2 = 0.95

warmup_iters = 200  # Increased from 20 for stability with higher LR

import torch
import sys

# Auto-detect CUDA availability (fallback to CPU if CUDA not available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

# Disable compile on Windows (Triton issues) or if CUDA not available
# On Windows, torch.compile requires Triton which has limited support
is_windows = sys.platform == 'win32'
compile = False if is_windows or not torch.cuda.is_available() else True
