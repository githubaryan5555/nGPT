# ============================================================
# CONFIG.PY
# Model5555 training configuration
# ============================================================

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------

dataset_dir = "dataset"

train_bin = f"{dataset_dir}/train.bin"
val_bin = f"{dataset_dir}/val.bin"

tokenizer_json = f"{dataset_dir}/tokenizer.json"
tokenizer_config_json = f"{dataset_dir}/tokenizer_config.json"


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

# These are the architectural values consumed directly by
# Model5555LM / Config in model.py.

vocab_size = 8192

hidden_size = 256

num_hidden_layers = 16

intermediate_size = 768

num_attention_heads = 8

num_key_value_heads = 2

attention_dropout = 0.0

hidden_dropout = 0.0

rms_norm_eps = 1e-5

rope_theta = 10000.0

max_seq_len = 256

tie_word_embeddings = True

initializer_range = 0.02


# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------

batch_size = 64

gradient_accumulation_steps = 1

max_iters = 5000

eval_interval = 250

eval_iters = 50

log_interval = 10


# ------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------

learning_rate = 3e-4

min_lr = 3e-5

weight_decay = 0.1

beta1 = 0.9

beta2 = 0.95

grad_clip = 1.0


# ------------------------------------------------------------
# LR SCHEDULE
# ------------------------------------------------------------

lr_decay = True

warmup_iters = 200

lr_decay_iters = max_iters


# ------------------------------------------------------------
# PRECISION / DEVICE
# ------------------------------------------------------------

device = "cuda"

# Options:
# "float32"
# "float16"
# "bfloat16"
dtype = "float16"

compile_model = False


# ------------------------------------------------------------
# CHECKPOINTS
# ------------------------------------------------------------

out_dir = "checkpoints"

save_checkpoint = True

always_save_checkpoint = False


# ------------------------------------------------------------
# RANDOMNESS
# ------------------------------------------------------------

seed = 1337


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

print_model_info = True

estimate_vram = True
