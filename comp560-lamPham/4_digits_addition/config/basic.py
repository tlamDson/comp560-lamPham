# train a miniature 4-digit addition model (SPEEDRUN MODE)

out_dir = 'out'
eval_interval = 50
eval_iters = 20
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = '4_digits_arithmetic'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 256  # Increased from 64 to maximize GPU utilization
block_size = 20   # Increased from 16 (longer examples: 4+1+4+1+5=15 chars + newline)

# Model Pruning: 4 layers with wider embedding (faster than 6 layers)
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0

# Training settings
learning_rate = 4e-3  # Slightly higher with smaller model
max_iters = 8000  # Increased for 4-digit (more complex task)
lr_decay_iters = 8000
min_lr = 4e-4  # Proportional to new learning_rate
beta2 = 0.95

warmup_iters = 200  # Slightly increased for more complex task

# Early Stopping: Disabled for now - low loss doesn't guarantee 100% accuracy
# Set to 0 to disable, or use very low value like 0.0001
early_stop_loss = 0.0  # Disabled - let it train full iterations

import torch
import sys

# Auto-detect CUDA availability (fallback to CPU if CUDA not available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

# Disable compile on Windows (Triton issues) or if CUDA not available
# On Windows, torch.compile requires Triton which has limited support
is_windows = sys.platform == 'win32'
compile = False if is_windows or not torch.cuda.is_available() else True
