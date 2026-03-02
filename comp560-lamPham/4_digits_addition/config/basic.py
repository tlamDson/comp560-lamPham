out_dir = 'out'
eval_interval = 200
eval_iters = 5  
log_interval = 200

always_save_checkpoint = False

wandb_log = False
wandb_project = '4_digits_arithmetic'
wandb_entity = 'dickinson-comp560-sp26'
wandb_run_name = '4digit-addition'

dataset = 'basic'
gradient_accumulation_steps = 1  
batch_size = 1024  
block_size = 20   # 4+1+4+1+5=15 chars + padding

n_layer = 4
n_head = 8
n_embd = 64
dropout = 0.0
weight_decay = 0.0        # Allow weights to grow sharp to calculate carries

# Training settings - STABLE convergence (30 min run)
learning_rate = 6e-3  
max_iters = 5000    
lr_decay_iters = 5000
min_lr = 6e-4
beta2 = 0.99
grad_clip = 1.0      # Prevents math logic from destabilizing
warmup_iters = 100
early_stop_loss = 1.12
answer_only_loss = True

import torch
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

compile = True
