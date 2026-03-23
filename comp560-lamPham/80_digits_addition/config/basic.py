out_dir = 'out'
eval_interval = 200
eval_iters = 5
log_interval = 200

always_save_checkpoint = False

wandb_log = False
wandb_project = '80_digits_arithmetic'
wandb_entity = 'dickinson-comp560-sp26'
wandb_run_name = '80digit-addition'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 512
block_size = 248   # 80+1+80+1+81 + padding
sample_stride = 244  # One fixed example length including newline in token stream

n_layer = 4
n_head = 8
n_embd = 64
dropout = 0.0
weight_decay = 0.0

learning_rate = 3e-3
max_iters = 12000
lr_decay_iters = 12000
min_lr = 3e-4
beta2 = 0.99
grad_clip = 1.0
warmup_iters = 200
early_stop_loss = 0.0
early_stop_acc = 0.99
answer_only_loss = True

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

compile = True
