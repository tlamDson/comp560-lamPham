out_dir = 'out_phi'
eval_interval = 500
eval_iters = 1
log_interval = 100

always_save_checkpoint = False

wandb_log = False
wandb_project = '100_digits_arithmetic'
wandb_entity = 'dickinson-comp560-sp26'
wandb_run_name = '100digit-addition-phi'

dataset = 'basic'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 608
sample_stride = 604

# Switch the trainer to Phi through model factory.
model_family = 'phi'
phi_model_source = 'Phi-3-mini-4k-instruct'

# Small Phi-shaped config for this arithmetic setup.
n_layer = 8
n_head = 8
n_embd = 512
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
