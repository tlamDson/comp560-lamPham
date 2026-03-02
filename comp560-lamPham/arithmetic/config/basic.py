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
batch_size = 256
block_size = 16

# Model: 4 layers
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0

# Training settings
learning_rate = 4e-3
max_iters = 6000
lr_decay_iters = 6000
min_lr = 4e-4
beta2 = 0.95

warmup_iters = 150

# Early Stopping: Disabled (low loss doesn't guarantee 100% accuracy)
early_stop_loss = 0.0

import torch

# Auto-detect CUDA availability (fallback to CPU if CUDA not available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

# Enable compile only if CUDA available
compile = torch.cuda.is_available()
