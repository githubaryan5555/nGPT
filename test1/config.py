
# config.py — every variable here is read AND used by train.py
# edit this file instead of passing CLI flags

# ---------------------------- I/O ----------------------------
out_dir = 'out_fineweb20mb'
eval_interval = 100
log_interval = 10
eval_iters = 25
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# ---------------------------- wandb --------------------------
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'fw20mb'

# ---------------------------- data ----------------------------
data_root = 'datasets'
dataset = 'fineweb20mb'

gradient_accumulation_steps = 4 * 4 # used to simulate larger batch sizes
batch_size = 128 # micro-batch size
block_size = 256

# ---------------------------- model ---------------------------
n_layer = 8
n_head = 8
n_embd = 256
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# ---------------------------- adamw optimizer -----------------
learning_rate = 1e-3
max_iters = 1000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# ------------------------- LR decay settings ------------------
decay_lr = True
warmup_iters = 50
lr_decay_iters = 1000
min_lr = 5e-4

# ---------------------------- DDP settings --------------------
backend = 'nccl' # 'nccl', 'gloo', etc.

# ---------------------------- system ---------------------------
device = 'cuda'

dtype = (
    'float16'
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else 'float16'
) # 'float32', 'bfloat16', or 'float16'

compile = True # use PyTorch 2.0 to compile the model to be faster
