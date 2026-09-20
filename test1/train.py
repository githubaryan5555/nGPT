"""
Single-GPU training script.

All user-configurable settings live in config.py.

Expected layout:

test1/
├── config.py
├── model.py
├── train.py
└── datasets/
    └── fineweb20mb/
        ├── train.bin
        ├── val.bin
        ├── dataset_config.json
        ├── tokenizer.json
        └── tokenizer_config.json

The .bin files must already contain token IDs.
The tokenizer is NOT used during training.
"""

import os
import time
import math
import json
from contextlib import nullcontext

import numpy as np
import torch

from model import GPTConfig, GPT


# =============================================================================
# Load config.py
# =============================================================================

import config as user_config

_config_keys = [
    k for k, v in vars(user_config).items()
    if not k.startswith("_")
    and isinstance(v, (int, float, bool, str))
]

config = {
    k: getattr(user_config, k)
    for k in _config_keys
}

globals().update(config)


# =============================================================================
# Basic configuration validation
# =============================================================================

def fail(message):
    raise RuntimeError(f"\nCONFIGURATION ERROR:\n{message}\n")


def require(condition, message):
    if not condition:
        fail(message)


require(vocab_size > 0, "vocab_size must be > 0.")
require(n_layer > 0, "n_layer must be > 0.")
require(n_head > 0, "n_head must be > 0.")
require(n_embd > 0, "n_embd must be > 0.")
require(block_size > 0, "block_size must be > 0.")

require(batch_size > 0, "batch_size must be > 0.")
require(
    gradient_accumulation_steps > 0,
    "gradient_accumulation_steps must be > 0."
)

require(max_iters >= 0, "max_iters must be >= 0.")
require(eval_iters > 0, "eval_iters must be > 0.")
require(eval_interval > 0, "eval_interval must be > 0.")
require(log_interval > 0, "log_interval must be > 0.")

require(learning_rate > 0, "learning_rate must be > 0.")
require(weight_decay >= 0, "weight_decay must be >= 0.")
require(0 <= beta1 < 1, "beta1 must be in [0, 1).")
require(0 <= beta2 < 1, "beta2 must be in [0, 1).")

if grad_clip < 0:
    fail("grad_clip must be >= 0.")

if decay_lr:
    require(warmup_iters >= 0, "warmup_iters must be >= 0.")
    require(lr_decay_iters > warmup_iters,
            "lr_decay_iters must be greater than warmup_iters.")
    require(min_lr >= 0, "min_lr must be >= 0.")
    require(min_lr <= learning_rate,
            "min_lr must be <= learning_rate.")

require(init_from in {"scratch", "resume"},
        "init_from must be either 'scratch' or 'resume'.")

require(
    isinstance(preload_data_to_gpu, bool),
    "preload_data_to_gpu must be True or False."
)

if preload_data_to_gpu:
    print(
        "WARNING: preload_data_to_gpu=True requests full dataset "
        "preloading, which is intentionally disabled for the general "
        "streaming loader."
    )


# =============================================================================
# Device
# =============================================================================

requested_device = str(device)

if requested_device.startswith("cuda"):
    if not torch.cuda.is_available():
        fail(
            f"config requests device='{requested_device}', "
            "but CUDA is not available."
        )

    device = torch.device("cuda:0")

elif requested_device == "cpu":
    device = torch.device("cpu")

elif requested_device.startswith("mps"):
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        fail(
            f"config requests device='{requested_device}', "
            "but MPS is not available."
        )

    device = torch.device(requested_device)

else:
    fail(
        f"Unsupported device '{requested_device}'. "
        "Use 'cuda', 'cuda:0', 'cpu', or 'mps'."
    )


device_type = device.type
using_cuda = device_type == "cuda"


# =============================================================================
# CUDA setup
# =============================================================================

if using_cuda:
    torch.cuda.set_device(device)

    # These are safe performance settings for modern NVIDIA GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Let cuDNN benchmark choose kernels when shapes stay fixed.
    torch.backends.cudnn.benchmark = True


# =============================================================================
# Dtype / AMP configuration
# =============================================================================

requested_dtype = str(dtype).lower()

if requested_dtype not in {
    "auto",
    "float32",
    "fp32",
    "float16",
    "fp16",
    "bfloat16",
    "bf16",
}:
    fail(
        f"Unsupported dtype='{dtype}'. "
        "Use 'auto', 'float32', 'float16', or 'bfloat16'."
    )


def choose_dtype():
    if requested_dtype in {"float32", "fp32"}:
        return torch.float32

    if requested_dtype in {"float16", "fp16"}:
        if not using_cuda:
            print(
                "WARNING: float16 AMP is not used on CPU. "
                "Falling back to float32."
            )
            return torch.float32

        return torch.float16

    if requested_dtype in {"bfloat16", "bf16"}:
        if using_cuda:
            if not torch.cuda.is_bf16_supported():
                fail(
                    "dtype='bfloat16' was requested, but this CUDA "
                    "device does not report BF16 support."
                )
            return torch.bfloat16

        # CPU BF16 support depends on the actual CPU/PyTorch build.
        # PyTorch can generally execute BF16 CPU operations, but it may
        # be substantially slower than FP32.
        return torch.bfloat16

    # auto
    if using_cuda:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


ptdtype = choose_dtype()

if ptdtype == torch.float16 and not using_cuda:
    fail("float16 training requires CUDA in this training script.")


# AMP is useful for CUDA. CPU BF16 uses autocast as well.
if device_type == "cuda":
    autocast_enabled = ptdtype in {
        torch.float16,
        torch.bfloat16,
    }

    autocast_context = (
        torch.autocast(
            device_type="cuda",
            dtype=ptdtype,
            enabled=autocast_enabled,
        )
        if autocast_enabled
        else nullcontext()
    )

elif device_type == "cpu" and ptdtype == torch.bfloat16:
    autocast_context = torch.autocast(
        device_type="cpu",
        dtype=torch.bfloat16,
    )

else:
    autocast_context = nullcontext()


# GradScaler is required/useful for FP16, but NOT BF16.
use_grad_scaler = (
    using_cuda
    and ptdtype == torch.float16
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_grad_scaler,
)


# =============================================================================
# Random seeds
# =============================================================================

torch.manual_seed(seed)

if using_cuda:
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Dataset paths
# =============================================================================

data_dir = os.path.join(
    str(data_root),
    str(dataset),
)

train_path = os.path.join(data_dir, "train.bin")
val_path = os.path.join(data_dir, "val.bin")
dataset_config_path = os.path.join(data_dir, "dataset_config.json")
tokenizer_config_path = os.path.join(data_dir, "tokenizer_config.json")
tokenizer_path = os.path.join(data_dir, "tokenizer.json")

require(
    os.path.isdir(data_dir),
    f"Dataset directory does not exist:\n{data_dir}"
)

require(
    os.path.isfile(train_path),
    f"Missing training file:\n{train_path}"
)

require(
    os.path.isfile(val_path),
    f"Missing validation file:\n{val_path}"
)


# =============================================================================
# Dataset metadata
# =============================================================================

dataset_config = {}

if os.path.isfile(dataset_config_path):
    try:
        with open(dataset_config_path, "r", encoding="utf-8") as f:
            dataset_config = json.load(f)
    except Exception as e:
        fail(
            f"Could not read dataset_config.json:\n{e}"
        )


# =============================================================================
# Determine token storage dtype
# =============================================================================

def normalize_numpy_dtype(value):
    """
    Convert a metadata dtype string into a NumPy dtype.
    """
    if value is None:
        return None

    value = str(value).lower().strip()

    aliases = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "long": np.int64,
    }

    return aliases.get(value)


def find_dataset_dtype(metadata):
    """
    Look for common dtype field names in dataset_config.json.
    """
    if not isinstance(metadata, dict):
        return None

    candidates = [
        metadata.get("dtype"),
        metadata.get("token_dtype"),
        metadata.get("storage_dtype"),
        metadata.get("numpy_dtype"),
        metadata.get("data_dtype"),
    ]

    for candidate in candidates:
        dtype_value = normalize_numpy_dtype(candidate)
        if dtype_value is not None:
            return dtype_value

    return None


storage_dtype = find_dataset_dtype(dataset_config)

if storage_dtype is None:
    # Your current FineWeb binary format uses uint16.
    #
    # We do NOT silently assume GPT-2 or any tokenizer vocabulary.
    # uint16 is only selected as the fallback because the existing
    # dataset format stores these token IDs as uint16.
    storage_dtype = np.uint16

    print(
        "Dataset dtype metadata not found; using uint16 for .bin files."
    )

storage_dtype = np.dtype(storage_dtype)


# =============================================================================
# Validate storage dtype
# =============================================================================

supported_storage_dtypes = {
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.int64),
}

require(
    storage_dtype in supported_storage_dtypes,
    (
        f"Unsupported dataset storage dtype: {storage_dtype}. "
        "Supported types are uint16, uint32, int16, int32, and int64."
    )
)


if np.issubdtype(storage_dtype, np.signedinteger):
    print(
        f"Dataset storage dtype: {storage_dtype} "
        "(signed integer; token IDs will be validated)"
    )
else:
    print(f"Dataset storage dtype: {storage_dtype}")


# =============================================================================
# Dataset length / memmap
# =============================================================================

train_data = np.memmap(
    train_path,
    dtype=storage_dtype,
    mode="r",
)

val_data = np.memmap(
    val_path,
    dtype=storage_dtype,
    mode="r",
)


require(
    len(train_data) > block_size,
    (
        f"train.bin contains only {len(train_data):,} tokens, "
        f"but block_size={block_size}."
    )
)

require(
    len(val_data) > block_size,
    (
        f"val.bin contains only {len(val_data):,} tokens, "
        f"but block_size={block_size}."
    )
)


# =============================================================================
# Dataset token validation
# =============================================================================

def validate_token_range(data, name):
    """
    Validate a sample of token IDs without scanning an arbitrarily huge file.

    For normal datasets, the full range can optionally be checked through
    dataset metadata. A small deterministic sample catches corrupt files
    without turning startup into a giant dataset scan.
    """

    sample_size = min(len(data), 1_000_000)

    if sample_size == len(data):
        sample = data
    else:
        # Deterministic evenly spaced sample.
        indices = np.linspace(
            0,
            len(data) - 1,
            num=sample_size,
            dtype=np.int64,
        )
        sample = data[indices]

    if sample.size == 0:
        fail(f"{name} is empty.")

    if np.issubdtype(storage_dtype, np.signedinteger):
        min_token = int(sample.min())
        if min_token < 0:
            fail(
                f"{name} contains a negative token ID: {min_token}."
            )

    max_token = int(sample.max())

    if max_token >= vocab_size:
        fail(
            f"{name} contains token ID {max_token}, "
            f"but vocab_size={vocab_size}. "
            f"Valid IDs are 0..{vocab_size - 1}."
        )

    print(
        f"{name}: {len(data):,} tokens | "
        f"sample max ID={max_token:,} | "
        f"range OK"
    )


validate_token_range(train_data, "train.bin")
validate_token_range(val_data, "val.bin")


# =============================================================================
# Dataset information
# =============================================================================

print("=" * 70)
print("DATASET")
print("=" * 70)
print(f"directory       : {data_dir}")
print(f"storage dtype   : {storage_dtype}")
print(f"train tokens    : {len(train_data):,}")
print(f"val tokens      : {len(val_data):,}")
print(f"vocab size      : {vocab_size:,}")
print(f"block size      : {block_size:,}")
print(f"streaming       : enabled")
print(f"preload to GPU  : disabled")
print("=" * 70)


# =============================================================================
# Batch loader
# =============================================================================

# Reusable CPU tensors avoid repeatedly allocating the same tensor shapes.
cpu_x = torch.empty(
    (batch_size, block_size),
    dtype=torch.long,
)

cpu_y = torch.empty(
    (batch_size, block_size),
    dtype=torch.long,
)


def get_batch(split):
    """
    Random contiguous batches from a memory-mapped token stream.

    No full dataset is loaded into RAM.
    """

    data = train_data if split == "train" else val_data

    max_start = len(data) - block_size - 1

    starts = torch.randint(
        0,
        max_start + 1,
        (batch_size,),
        dtype=torch.int64,
    ).numpy()

    # Fill reusable tensors.
    #
    # Each sample is contiguous in the original mmap.
    for row, start in enumerate(starts):
        cpu_x[row].copy_(
            torch.from_numpy(
                np.asarray(
                    data[start:start + block_size],
                    dtype=storage_dtype,
                )
            ).to(torch.long)
        )

        cpu_y[row].copy_(
            torch.from_numpy(
                np.asarray(
                    data[start + 1:start + 1 + block_size],
                    dtype=storage_dtype,
                )
            ).to(torch.long)
        )

    if using_cuda:
        # Pin once, then reuse.
        x = cpu_x.pin_memory().to(
            device,
            non_blocking=True,
        )

        y = cpu_y.pin_memory().to(
            device,
            non_blocking=True,
        )

        return x, y

    return cpu_x.to(device), cpu_y.to(device)


# =============================================================================
# Training information
# =============================================================================

tokens_per_iter = (
    batch_size
    * block_size
    * gradient_accumulation_steps
)

effective_batch_tokens = tokens_per_iter

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)
print(f"device                  : {device}")
print(f"dtype                   : {ptdtype}")
print(f"AMP                     : {use_grad_scaler or autocast_enabled if 'autocast_enabled' in globals() else False}")
print(f"GradScaler              : {use_grad_scaler}")
print(f"batch size              : {batch_size}")
print(f"gradient accumulation   : {gradient_accumulation_steps}")
print(f"effective tokens/step   : {effective_batch_tokens:,}")
print(f"learning rate           : {learning_rate:g}")
print(f"max iterations          : {max_iters:,}")
print(f"compile                 : {compile}")
print("=" * 70)


# =============================================================================
# Model initialization
# =============================================================================

iter_num = 0
best_val_loss = float("inf")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
)


checkpoint = None

if init_from == "scratch":

    print("\nInitializing model from scratch.")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == "resume":

    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    require(
        os.path.isfile(ckpt_path),
        f"Cannot resume: checkpoint does not exist:\n{ckpt_path}"
    )

    print(f"\nResuming from: {ckpt_path}")

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    require(
        "model_args" in checkpoint,
        "Checkpoint does not contain model_args."
    )

    checkpoint_model_args = checkpoint["model_args"]

    # These define the model architecture.
    architecture_keys = [
        "n_layer",
        "n_head",
        "n_embd",
        "block_size",
        "bias",
        "vocab_size",
    ]

    for key in architecture_keys:
        checkpoint_value = checkpoint_model_args[key]
        current_value = model_args[key]

        if checkpoint_value != current_value:
            fail(
                f"Checkpoint/model mismatch for '{key}':\n"
                f"  checkpoint: {checkpoint_value}\n"
                f"  config:     {current_value}\n"
                "\n"
                "Change config.py to match the checkpoint "
                "or start with init_from='scratch'."
            )

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]

    # torch.compile can add this prefix to state dict keys.
    compiled_prefix = "_orig_mod."

    if any(k.startswith(compiled_prefix) for k in state_dict):
        state_dict = {
            (
                k[len(compiled_prefix):]
                if k.startswith(compiled_prefix)
                else k
            ): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)

    iter_num = int(checkpoint.get("iter_num", 0))
    best_val_loss = float(
        checkpoint.get("best_val_loss", float("inf"))
    )


# =============================================================================
# Model validation
# =============================================================================

require(
    model.config.vocab_size == vocab_size,
    (
        f"Model vocab_size={model.config.vocab_size} does not match "
        f"config vocab_size={vocab_size}."
    )
)

require(
    model.config.block_size >= block_size,
    (
        f"Model block_size={model.config.block_size} is smaller than "
        f"configured block_size={block_size}."
    )
)


if block_size < model.config.block_size:
    if hasattr(model, "crop_block_size"):
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size
    else:
        fail(
            "Configured block_size is smaller than the model's block_size, "
            "but model.crop_block_size() is unavailable."
        )


# =============================================================================
# Move model to device
# =============================================================================

model.to(device)


# =============================================================================
# Parameter count
# =============================================================================

num_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"number of parameters: "
    f"{num_params:,} "
    f"({num_params / 1e6:.2f}M)"
)


# =============================================================================
# Optimizer
# =============================================================================

optimizer = model.configure_optimizers(
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type,
)

if checkpoint is not None and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Optimizer states were loaded onto CPU by map_location.
    # Move tensor states to the training device.
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


checkpoint = None


# =============================================================================
# torch.compile
# =============================================================================

if compile:

    if not hasattr(torch, "compile"):
        fail(
            "compile=True but this PyTorch installation "
            "does not provide torch.compile."
        )

    print(
        f"compiling model with mode='{compile_mode}'..."
    )

    model = torch.compile(
        model,
        mode=compile_mode,
    )


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ("train", "val"):

        losses = torch.empty(
            eval_iters,
            dtype=torch.float32,
        )

        for k in range(eval_iters):

            X, Y = get_batch(split)

            with autocast_context:
                _, loss = model(X, Y)

            losses[k] = loss.detach().float().cpu()

        results[split] = losses.mean().item()

    model.train()

    return results


# =============================================================================
# Learning-rate scheduler
# =============================================================================

def get_lr(iteration):

    if not decay_lr:
        return learning_rate

    if iteration < warmup_iters:
        return (
            learning_rate
            * (iteration + 1)
            / (warmup_iters + 1)
        )

    if iteration >= lr_decay_iters:
        return min_lr

    decay_ratio = (
        (iteration - warmup_iters)
        / (lr_decay_iters - warmup_iters)
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return min_lr + coeff * (
        learning_rate - min_lr
    )


# =============================================================================
# W&B
# =============================================================================

if wandb_log:

    try:
        import wandb
    except ImportError:
        fail(
            "wandb_log=True but wandb is not installed."
        )

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
    )


# =============================================================================
# Checkpoint helper
# =============================================================================

def get_raw_model():
    """
    torch.compile wraps the model, so unwrap it when possible.
    """
    model_to_save = model

    if hasattr(model_to_save, "_orig_mod"):
        model_to_save = model_to_save._orig_mod

    return model_to_save


def save_checkpoint(val_loss):

    raw_model = get_raw_model()

    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }

    path = os.path.join(
        out_dir,
        "ckpt.pt",
    )

    print(f"saving checkpoint to {path}")

    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# Create output directory
# =============================================================================

os.makedirs(out_dir, exist_ok=True)


# =============================================================================
# Initial batch
# =============================================================================

X, Y = get_batch("train")


# =============================================================================
# Training loop
# =============================================================================

t0 = time.time()

local_iter_num = 0
running_mfu = -1.0

while True:

    # -------------------------------------------------------------------------
    # Learning rate
    # -------------------------------------------------------------------------

    lr = get_lr(iter_num)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


    # -------------------------------------------------------------------------
    # Evaluation / checkpointing
    # -------------------------------------------------------------------------

    if iter_num % eval_interval == 0:

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]

        print(
            f"step {iter_num}: "
            f"train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}"
        )

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": lr,
                "mfu": (
                    running_mfu * 100
                    if running_mfu >= 0
                    else 0.0
                ),
            })

        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss

        if always_save_checkpoint or improved:

            if iter_num > 0:
                save_checkpoint(val_loss)


    # -------------------------------------------------------------------------
    # Evaluation-only mode
    # -------------------------------------------------------------------------

    if iter_num == 0 and eval_only:
        break


    # -------------------------------------------------------------------------
    # Gradient accumulation
    # -------------------------------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    last_loss = None

    for micro_step in range(
        gradient_accumulation_steps
    ):

        with autocast_context:

            logits, loss = model(X, Y)

            loss_for_backward = (
                loss
                / gradient_accumulation_steps
            )

        last_loss = loss.detach()

        # Fetch next batch while the current forward/backward
        # computation is progressing.
        X, Y = get_batch("train")

        if use_grad_scaler:
            scaler.scale(
                loss_for_backward
            ).backward()
        else:
            loss_for_backward.backward()


    # -------------------------------------------------------------------------
    # Gradient clipping
    # -------------------------------------------------------------------------

    if grad_clip > 0:

        if use_grad_scaler:
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )


    # -------------------------------------------------------------------------
    # Optimizer update
    # -------------------------------------------------------------------------

    if use_grad_scaler:

        scaler.step(optimizer)
        scaler.update()

    else:

        optimizer.step()


    # -------------------------------------------------------------------------
    # Timing / logging
    # -------------------------------------------------------------------------

    t1 = time.time()

    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:

        loss_value = (
            last_loss.float().item()
        )

        tokens_this_step = tokens_per_iter

        tok_per_sec = (
            tokens_this_step / dt
            if dt > 0
            else 0.0
        )

        mfu_text = "N/A"

        if local_iter_num >= 5 and hasattr(
            get_raw_model(),
            "estimate_mfu",
        ):

            try:

                mfu = get_raw_model().estimate_mfu(
                    batch_size
                    * gradient_accumulation_steps,
                    dt,
                )

                running_mfu = (
                    mfu
                    if running_mfu < 0
                    else 0.9 * running_mfu
                    + 0.1 * mfu
                )

                mfu_text = (
                    f"{running_mfu * 100:.2f}%"
                )

            except Exception:
                mfu_text = "N/A"

        print(
            f"iter {iter_num}: "
            f"loss {loss_value:.4f}, "
            f"lr {lr:.6g}, "
            f"time {dt * 1000:.2f}ms, "
            f"tok/s {tok_per_sec:,.0f}, "
            f"mfu {mfu_text}"
        )

    iter_num += 1
    local_iter_num += 1


    # -------------------------------------------------------------------------
    # Termination
    # -------------------------------------------------------------------------

    if iter_num >= max_iters:
        break


# =============================================================================
# Final checkpoint
# =============================================================================

print("\nTraining finished.")

if wandb_log:
    wandb.finish()
