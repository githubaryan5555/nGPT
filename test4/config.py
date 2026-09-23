# ============================================================
# CONFIG.PY
# Model5555 training configuration
# ============================================================


# ============================================================
# DATASET
# ============================================================

# Folder containing the already-tokenized dataset.
#
# Expected:
#
# dataset/
# ├── train.bin
# └── val.bin
#
# No meta.pkl is required.
# No tokenizer is loaded by train.py.

dataset_dir = "dataset"

train_bin = f"{dataset_dir}/train.bin"

val_bin = f"{dataset_dir}/val.bin"


# ============================================================
# MODEL
# ============================================================

vocab_size = 8192

hidden_size = 256

num_hidden_layers = 16

intermediate_size = 1024

num_attention_heads = 8

num_key_value_heads = 2

attention_dropout = 0.0

hidden_dropout = 0.0

rms_norm_eps = 1e-5

rope_theta = 10000.0

max_seq_len = 256

tie_word_embeddings = True

initializer_range = 0.02


# ============================================================
# TRAINING
# ============================================================

batch_size = 128

gradient_accumulation_steps = 16

max_iters = 1000

eval_interval = 100

eval_iters = 50

log_interval = 10

eval_only = False


# ============================================================
# OPTIMIZER
# ============================================================

learning_rate = 3e-4

min_lr = 3e-5

weight_decay = 0.1

beta1 = 0.9

beta2 = 0.95

grad_clip = 1.0


# ============================================================
# LR SCHEDULE
# ============================================================

lr_decay = True

warmup_iters = 200

lr_decay_iters = max_iters


# ============================================================
# PRECISION / DEVICE
# ============================================================

device = "cuda"

# Options:
# "float32"
# "float16"
# "bfloat16"

dtype = "float16"

compile_model = True


# ============================================================
# DISTRIBUTED TRAINING
# ============================================================

# "nccl" for NVIDIA CUDA GPUs.
# "gloo" can be used for CPU-based distributed training.

backend = "nccl"


# ============================================================
# CHECKPOINTS
# ============================================================

out_dir = "checkpoints"

checkpoint_name = "ckpt.pt"

# Master switch for checkpoint saving.
save_checkpoint = True

# If True, save after every evaluation.
# If False, save only when validation loss improves.

always_save_checkpoint = False

# "scratch" or "resume"

init_from = "scratch"


# ============================================================
# RANDOMNESS
# ============================================================

seed = 1337


# ============================================================
# LOGGING / MODEL INFORMATION
# ============================================================

print_model_info = True

estimate_vram = True


# ============================================================
# HARDWARE MFU
# ============================================================

# Peak theoretical GPU FLOPS.
#
# Leave as None if you don't want MFU calculation.
#
# Example for a T4 FP16:
# peak_flops = 8.1e12
#
# This value should eventually be chosen based on the
# actual precision and hardware being used.

peak_flops = None
