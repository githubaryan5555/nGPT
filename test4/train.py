import os
import json
import math
import time
import random

import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from model import Model5555LM, Config


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)


# ============================================================
# DEVICE
# ============================================================

if cfg.device == "cuda" and not torch.cuda.is_available():
    print("CUDA requested but unavailable.")
    print("Falling back to CPU.")
    device = torch.device("cpu")
else:
    device = torch.device(cfg.device)

print("=" * 64)
print("MODEL5555 TRAINING")
print("=" * 64)
print(f"Device           : {device}")


# ============================================================
# DTYPE
# ============================================================

if cfg.dtype == "float32":
    torch_dtype = torch.float32

elif cfg.dtype == "float16":
    torch_dtype = torch.float16

elif cfg.dtype == "bfloat16":
    torch_dtype = torch.bfloat16

else:
    raise ValueError(
        f"Unknown dtype: {cfg.dtype}"
    )

print(f"Dtype            : {cfg.dtype}")


# ============================================================
# DATASET PATHS
# ============================================================

if not os.path.exists(cfg.train_bin):
    raise FileNotFoundError(
        f"Missing training dataset:\n{cfg.train_bin}"
    )

if not os.path.exists(cfg.val_bin):
    raise FileNotFoundError(
        f"Missing validation dataset:\n{cfg.val_bin}"
    )


# ============================================================
# TOKENIZER CONFIG
# ============================================================

def load_tokenizer_config(path):
    if not os.path.exists(path):
        print(
            f"Tokenizer config not found:\n{path}"
        )
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Tokenizer config must contain a JSON object."
        )

    return data


tokenizer_config = load_tokenizer_config(
    cfg.tokenizer_config_json
)


# ============================================================
# FIND VOCAB SIZE
# ============================================================

def find_vocab_size(data):
    possible_keys = (
        "vocab_size",
        "vocabulary_size",
        "n_vocab",
    )

    for key in possible_keys:
        value = data.get(key)

        if isinstance(value, int):
            return value

    # Some tokenizer configs nest the information.
    for key in (
        "tokenizer",
        "model",
        "config",
    ):
        nested = data.get(key)

        if isinstance(nested, dict):
            result = find_vocab_size(nested)

            if result is not None:
                return result

    return None


detected_vocab_size = find_vocab_size(
    tokenizer_config
)

if detected_vocab_size is not None:

    if detected_vocab_size != cfg.vocab_size:
        print(
            "WARNING:"
        )

        print(
            f"config.py vocab_size = "
            f"{cfg.vocab_size}"
        )

        print(
            f"tokenizer config vocab_size = "
            f"{detected_vocab_size}"
        )

        raise ValueError(
            "Model vocabulary size does not match "
            "tokenizer vocabulary size."
        )

    print(
        f"Tokenizer vocab   : "
        f"{detected_vocab_size}"
    )

else:
    print(
        f"Tokenizer vocab   : "
        f"{cfg.vocab_size} "
        "(from config.py)"
    )


# ============================================================
# LOAD BIN DATA
# ============================================================

print("Loading datasets...")

train_data = np.memmap(
    cfg.train_bin,
    dtype=np.uint16,
    mode="r",
)

val_data = np.memmap(
    cfg.val_bin,
    dtype=np.uint16,
    mode="r",
)

print(
    f"Train tokens      : {len(train_data):,}"
)

print(
    f"Val tokens        : {len(val_data):,}"
)


# ============================================================
# DATASET VALIDATION
# ============================================================

if len(train_data) <= cfg.max_seq_len:
    raise ValueError(
        "Training dataset is smaller than "
        "max_seq_len."
    )

if len(val_data) <= cfg.max_seq_len:
    raise ValueError(
        "Validation dataset is smaller than "
        "max_seq_len."
    )


# Check a small sample for invalid IDs.

sample_count = min(
    1_000_000,
    len(train_data),
)

train_sample = train_data[:sample_count]

train_min = int(train_sample.min())
train_max = int(train_sample.max())

print(
    f"Train token range : "
    f"{train_min} .. {train_max}"
)

if train_min < 0:
    raise ValueError(
        "Negative token ID found."
    )

if train_max >= cfg.vocab_size:
    raise ValueError(
        "Training dataset contains token IDs "
        "outside the model vocabulary."
    )


# ============================================================
# BATCH FUNCTION
# ============================================================

def get_batch(data):

    max_start = len(data) - cfg.max_seq_len - 1

    ix = torch.randint(
        0,
        max_start + 1,
        (
            cfg.batch_size,
        ),
    )

    x = torch.stack(
        [
            torch.from_numpy(
                data[i:i + cfg.max_seq_len].astype(
                    np.int64
                )
            )
            for i in ix.tolist()
        ]
    )

    y = torch.stack(
        [
            torch.from_numpy(
                data[
                    i + 1:
                    i + 1 + cfg.max_seq_len
                ].astype(np.int64)
            )
            for i in ix.tolist()
        ]
    )

    return (
        x.to(device=device, dtype=torch.long),
        y.to(device=device, dtype=torch.long),
    )


# ============================================================
# MODEL CONFIG
# ============================================================

model_config = Config(
    vocab_size=cfg.vocab_size,
    hidden_size=cfg.hidden_size,
    num_hidden_layers=cfg.num_hidden_layers,
    intermediate_size=cfg.intermediate_size,
    num_attention_heads=cfg.num_attention_heads,
    num_key_value_heads=cfg.num_key_value_heads,
    attention_dropout=cfg.attention_dropout,
    hidden_dropout=cfg.hidden_dropout,
    rms_norm_eps=cfg.rms_norm_eps,
    rope_theta=cfg.rope_theta,
    max_seq_len=cfg.max_seq_len,
    tie_word_embeddings=cfg.tie_word_embeddings,
    initializer_range=cfg.initializer_range,
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating model...")

model = Model5555LM(
    config=model_config
)

model = model.to(
    device=device,
    dtype=torch_dtype,
)


# ============================================================
# MODEL INFO
# ============================================================

if cfg.print_model_info:

    model.print_model_info(
        batch_size=cfg.batch_size,
        seq_len=cfg.max_seq_len,
        dtype_bytes=2
        if torch_dtype in (
            torch.float16,
            torch.bfloat16,
        )
        else 4,
        optimizer="adamw",
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.learning_rate,
    betas=(
        cfg.beta1,
        cfg.beta2,
    ),
    weight_decay=cfg.weight_decay,
)


# ============================================================
# LEARNING RATE
# ============================================================

def get_lr(iteration):

    if not cfg.lr_decay:
        return cfg.learning_rate

    # Warmup
    if iteration < cfg.warmup_iters:

        return (
            cfg.learning_rate
            * (iteration + 1)
            / cfg.warmup_iters
        )

    # After decay period
    if iteration >= cfg.lr_decay_iters:

        return cfg.min_lr

    # Cosine decay
    decay_ratio = (
        iteration - cfg.warmup_iters
    ) / (
        cfg.lr_decay_iters
        - cfg.warmup_iters
    )

    coeff = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi * decay_ratio
            )
        )
    )

    return (
        cfg.min_lr
        + coeff
        * (
            cfg.learning_rate
            - cfg.min_lr
        )
    )


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

os.makedirs(
    cfg.out_dir,
    exist_ok=True,
)


# ============================================================
# CHECKPOINT PATH
# ============================================================

checkpoint_path = os.path.join(
    cfg.out_dir,
    "model5555.pt",
)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split, data in (
        ("train", train_data),
        ("val", val_data),
    ):

        losses = torch.zeros(
            cfg.eval_iters,
            device=device,
        )

        for k in range(cfg.eval_iters):

            x, y = get_batch(data)

            with torch.autocast(
                device_type=device.type,
                dtype=torch_dtype,
                enabled=(
                    device.type == "cuda"
                    and torch_dtype
                    in (
                        torch.float16,
                        torch.bfloat16,
                    )
                ),
            ):

                logits = model(x)

                loss = F.cross_entropy(
                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),
                    y.reshape(-1),
                )

            losses[k] = loss.detach()

        results[split] = (
            losses.mean().item()
        )

    model.train()

    return results


# ============================================================
# OPTIONAL COMPILE
# ============================================================

if cfg.compile_model:

    if hasattr(torch, "compile"):

        print("Compiling model...")

        model = torch.compile(
            model
        )

    else:

        print(
            "torch.compile unavailable."
        )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 64)
print("TRAINING")
print("=" * 64)

model.train()

best_val_loss = float("inf")

start_time = time.time()

for iteration in range(
    cfg.max_iters
):

    # --------------------------------------------------------
    # LEARNING RATE
    # --------------------------------------------------------

    lr = get_lr(iteration)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # --------------------------------------------------------
    # GRADIENT ACCUMULATION
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulated_loss = 0.0

    for micro_step in range(
        cfg.gradient_accumulation_steps
    ):

        x, y = get_batch(
            train_data
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch_dtype,
            enabled=(
                device.type == "cuda"
                and torch_dtype
                in (
                    torch.float16,
                    torch.bfloat16,
                )
            ),
        ):

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                y.reshape(-1),
            )

            loss = (
                loss
                / cfg.gradient_accumulation_steps
            )

        accumulated_loss += loss.item()

        loss.backward()

    # --------------------------------------------------------
    # GRADIENT CLIPPING
    # --------------------------------------------------------

    if cfg.grad_clip > 0:

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip,
        )

    else:

        grad_norm = torch.tensor(
            0.0,
            device=device,
        )

    # --------------------------------------------------------
    # OPTIMIZER STEP
    # --------------------------------------------------------

    optimizer.step()

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    if (
        iteration % cfg.log_interval == 0
        or iteration == cfg.max_iters - 1
    ):

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"step {iteration:6d} | "
            f"loss {accumulated_loss:.4f} | "
            f"lr {lr:.6e} | "
            f"grad {float(grad_norm):.4f} | "
            f"time {elapsed:.1f}s"
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    if (
        iteration % cfg.eval_interval == 0
        or iteration == cfg.max_iters - 1
    ):

        losses = estimate_loss()

        print(
            f"           train loss: "
            f"{losses['train']:.4f}"
        )

        print(
            f"           val loss  : "
            f"{losses['val']:.4f}"
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        should_save = (
            losses["val"] < best_val_loss
            or cfg.always_save_checkpoint
        )

        if should_save:

            best_val_loss = losses["val"]

            if cfg.save_checkpoint:

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_config": model_config.__dict__,
                    "train_config": {
                        key: value
                        for key, value
                        in vars(cfg).items()
                        if not key.startswith("__")
                    },
                    "iteration": iteration,
                    "best_val_loss": best_val_loss,
                }

                torch.save(
                    checkpoint,
                    checkpoint_path,
                )

                print(
                    f"           checkpoint saved: "
                    f"{checkpoint_path}"
                )


# ============================================================
# FINISHED
# ============================================================

elapsed = (
    time.time()
    - start_time
)

print()
print("=" * 64)
print("TRAINING COMPLETE")
print("=" * 64)

print(
    f"Final step       : {cfg.max_iters - 1}"
)

print(
    f"Best val loss    : {best_val_loss:.6f}"
)

print(
    f"Training time    : {elapsed:.2f} seconds"
)

print(
    f"Checkpoint       : {checkpoint_path}"
  )
