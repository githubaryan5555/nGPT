# config.py
# All trainer variables are read by train.py.
# model_kwargs are passed directly to the selected model.

# =============================================================================
# I/O
# =============================================================================

out_dir = "out_fineweb20mb"

eval_interval = 100
log_interval = 10
eval_iters = 25

eval_only = False
always_save_checkpoint = True
init_from = "scratch"


# =============================================================================
# W&B
# =============================================================================

wandb_log = False
wandb_project = "owt"
wandb_run_name = "fw20mb"


# =============================================================================
# DATA
# =============================================================================

data_root = "datasets"
dataset = "fineweb20mb"

preload_data_to_gpu = False


# =============================================================================
# MODEL SELECTION
# =============================================================================

model_module = "model"
model_class = "Transformer"
model_config_class = "ModelArgs"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

model_kwargs = {
    "dim": 512,
    "n_layers": 8,
    "n_heads": 16,
    "n_kv_heads": 4,

    "vocab_size": 8192,

    "multiple_of": 256,
    "ffn_dim_multiplier": None,
    "norm_eps": 1e-5,

    "max_batch_size": 128,
    "max_seq_len": 256,
}


# =============================================================================
# TRAINING / BATCHING
# =============================================================================

batch_size = 128
gradient_accumulation_steps = 16


# =============================================================================
# OPTIMIZER
# =============================================================================

learning_rate = 1e-3
max_iters = 1000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

grad_clip = 1.0


# =============================================================================
# LR SCHEDULE
# =============================================================================

decay_lr = True

warmup_iters = 50
lr_decay_iters = 1000
min_lr = 5e-4


# =============================================================================
# SYSTEM
# =============================================================================

device = "cuda"
dtype = "float16"

compile = True
compile_mode = "default"

seed = 1337


# =============================================================================
# GENERIC DATASET SEQUENCE LENGTH
# =============================================================================
#
# train.py needs this because it creates:
#
# X = tokens[start : start + block_size]
# Y = tokens[start + 1 : start + block_size + 1]
#
# Keep this even if the model calls it max_seq_len internally.

vocab_size = 8192
block_size = 256
