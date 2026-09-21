# config.py — every variable here is read AND used by train.py
# edit this file instead of passing CLI flags

# ---------------------------- I/O ----------------------------
out_dir = 'out_fineweb20mb'
eval_interval = 50
log_interval = 5
eval_iters = 25
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# ---------------------------- wandb ----------------------------
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'fw20mb'

# ---------------------------- data -----------------------------
data_root = 'datasets'
dataset = 'fineweb20mb'
preload_data_to_gpu = False

# ---------------------------- model ----------------------------
vocab_size = 8192
n_layer = 8
n_head = 16
n_embd = 512
block_size = 256
dropout = 0.0
bias = False

# ---------------------------- batching -------------------------
batch_size = 64
gradient_accumulation_steps = 16

# --------------------------- optimizer -------------------------
learning_rate = 1e-3
max_iters = 250
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ------------------------- LR schedule -------------------------
decay_lr = True
warmup_iters = 50
lr_decay_iters = 250
min_lr = 5e-4

# ---------------------------- system ---------------------------
device = 'cuda'
dtype = 'float16'
compile = False
compile_mode = 'default'
seed = 1337
