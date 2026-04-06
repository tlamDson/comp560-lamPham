out_dir = 'out'
eval_interval = 200
eval_iters = 5
log_interval = 200

always_save_checkpoint = False

wandb_log = False
wandb_project = '50_digits_arithmetic'
wandb_entity = 'dickinson-comp560-sp26'
wandb_run_name = '50digit-addition'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 248   # 50+1+50+1+51 + padding
sample_stride = 244  # One fixed example length including newline in token stream

n_layer = 6
n_head = 8
n_embd = 256
dropout = 0.0
weight_decay = 0.0

learning_rate = 1e-3
max_iters = 30000
lr_decay_iters = 30000
min_lr = 1e-4
beta2 = 0.99
grad_clip = 1.0
warmup_iters = 1500
early_stop_loss = 0.001
early_stop_acc = 0.99
answer_only_loss = True

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

compile = True
