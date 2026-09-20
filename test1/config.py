# config.py — every variable here is read AND used by train.py
# edit this file instead of passing CLI flags

# ---------------------------- I/O ----------------------------
out_dir = 'out_fineweb'          # checkpoint output directory
eval_interval = 100              # eval + checkpoint every N iters
log_interval = 10                # log every N iters
eval_iters = 25                  # batches per eval split
eval_only = False                # if True, exit after first eval
always_save_checkpoint = True    # save after every eval, not just on improvement
init_from = 'scratch'            # 'scratch' | 'resume' | 'gpt2*'

# ---------------------------- wandb ----------------------------
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'fw'

# ---------------------------- data -----------------------------
data_root = 'datasets'           # bins expected at datasets/<dataset>/*.bin or datasets/*.bin
dataset = 'fineweb'              # subfolder name under data_root (falls back to data_root itself)
preload_data_to_gpu = False      # if True and bins fit in VRAM, load tokens fully onto GPU (fastest loader)

# ---------------------------- model ----------------------------
n_layer = 8
n_head = 8
n_embd = 256
block_size = 256
dropout = 0.0                    # 0.0 for pretraining, try 0.1+ for finetuning
bias = False                     # bias inside LayerNorm and Linear layers?

# ---------------------------- batching -------------------------
batch_size = 64                  # micro-batch size
gradient_accumulation_steps = 16 # effective batch = batch_size * grad_accum

# --------------------------- optimizer -------------------------
learning_rate = 1e-3             # max LR
max_iters = 1000                 # total training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0                  # 0.0 disables

# ------------------------- LR schedule -------------------------
decay_lr = True
warmup_iters = 50
lr_decay_iters = 1000            # ~= max_iters per Chinchilla
min_lr = 5e-4                    # ~= learning_rate/10 per Chinchilla

# ---------------------------- system ---------------------------
device = 'cuda'                  # 'cpu', 'cuda', 'cuda:0', ...
dtype = 'auto'                   # 'auto' | 'float32' | 'bfloat16' | 'float16'
compile = True                   # torch.compile the model
compile_mode = 'default'         # 'default' | 'max-autotune' | 'max-autotune-no-cudagraphs' | ...
seed = 1337
