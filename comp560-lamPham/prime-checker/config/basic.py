# train a miniature prime-checker model
# good for debugging and playing

out_dir = 'out'
eval_interval = 50 # Number of training iterations between evaluating accuracy
eval_iters = 20 # Number of iterations to perform when evaluating accuracy
log_interval = 10 # Number of iterations between printing results

# Should we save the checkpoint on every evaluation or only when accuracy improves?
always_save_checkpoint = False

wandb_log = False # Should we log results to wandb.ai?
wandb_project = 'prime-checker' # A wandb project is a collection of related experiments

dataset = 'basic' # The name of the directory containing Prepared data including meta.pkl, train.bin, val.bin
gradient_accumulation_steps = 1
batch_size = 12 # The number of training instances in a micro batch.
block_size = 64 # The number of tokens in the context window of the GPT

# baby GPT model :)
n_layer = 4 # The number of layers in the GPT
n_head = 4 # The number of attention heads in the GPT
n_embd = 128 # The embedding size of the GPT
dropout = 0.0 # The fraction of network connections that are randomly disabled during training

learning_rate = 1e-3 # Initial learning rate
max_iters = 5000 # Increased for better learning
lr_decay_iters = 5000 # The number of iterations over which the learning rate will decay
min_lr = 1e-4 # The final learning rate after it has been decayed
beta2 = 0.95

warmup_iters = 20 # warm up iterations

device = 'cpu'  # run on cpu only (not GPU)
compile = False # do not torch compile the model
