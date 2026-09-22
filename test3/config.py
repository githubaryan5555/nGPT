# ============================================================
# MODEL CONFIG
# ============================================================

# Vocabulary / Model
vocab_size = 8192
hidden_size = 256
num_hidden_layers = 16

# MLP / SwiGLU
intermediate_size = 1024

# Attention / GQA
num_attention_heads = 16
num_key_value_heads = 4
attention_dropout = 0.0

# Model dropout
hidden_dropout = 0.0

# RMSNorm
rms_norm_eps = 1e-5

# RoPE
rope_theta = 10000.0
max_seq_len = 256

# Output
tie_word_embeddings = True

# Initialization
initializer_range = 0.02


# ============================================================
# DERIVED MODEL VALUES
# ============================================================

head_dim = hidden_size // num_attention_heads
num_queries_per_kv = num_attention_heads // num_key_value_heads


# ============================================================
# DATASET
# ============================================================

data_root = "datasets"
dataset = "fineweb20mb"

train_data = f"{data_root}/{dataset}/train.bin"
val_data = f"{data_root}/{dataset}/val.bin"

preload_data_to_gpu = False


# ============================================================
# TRAINING
# ============================================================

batch_size = 128
gradient_accumulation_steps = 16

block_size = max_seq_len

max_iters = 1000

eval_interval = 100
eval_iters = 25
log_interval = 10


# ============================================================
# OPTIMIZER
# ============================================================

learning_rate = 1e-3

weight_decay = 0.1

beta1 = 0.9
beta2 = 0.95

grad_clip = 1.0


# ============================================================
# LEARNING-RATE SCHEDULE
# ============================================================

decay_lr = True

warmup_iters = 50
lr_decay_iters = max_iters
min_lr = 5e-4


# ============================================================
# PRECISION / DEVICE
# ============================================================

device = "cuda"
dtype = "float16"


# ============================================================
# REPRODUCIBILITY
# ============================================================

seed = 1337


# ============================================================
# CHECKPOINTS
# ============================================================

out_dir = "out"

always_save_checkpoint = True
init_from = "scratch"


# ============================================================
# COMPILE
# ============================================================

compile = True
compile_mode = "default"


# ============================================================
# OPTIONAL LOGGING
# ============================================================

wandb_log = False
wandb_project = "model5555"
wandb_run_name = "quick-test"
