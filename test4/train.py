"""
Training script for Model5555LM.

This script keeps the useful structure of the original Karpathy GPT
training loop while adapting it to the current Model5555LM interface.

============================================================
DATASET INPUT STANDARD
============================================================

The dataset must already be tokenized.

Example:

data/
└── fineweb/
    ├── train.bin
    └── val.bin

Both files contain token IDs stored as uint16.

No tokenizer is loaded or required by this script.

The trainer does NOT know whether the tokenizer was:
- BPE
- byte-level BPE
- word-level
- character-level
- custom

It only expects integer token IDs.

For a vocab_size of 8192:

    valid token IDs = 0 ... 8191

============================================================
BATCH FORMAT
============================================================

For a sequence length T:

X:
    [token_0, token_1, ..., token_(T-1)]

Y:
    [token_1, token_2, ..., token_T]

Shapes:

    X = [batch_size, block_size]
    Y = [batch_size, block_size]

dtype:

    torch.long

The model receives only X:

    logits = model(X)

and returns:

    logits.shape == [B, T, vocab_size]

The training script calculates cross entropy against Y.

============================================================
OUTPUT
============================================================

Example:

out/
└── ckpt.pt

The checkpoint contains:

    model
    optimizer
    model_config
    train_config
    iter_num
    best_val_loss
    total_tokens

============================================================
SINGLE GPU
============================================================

python train.py

============================================================
DDP
============================================================

torchrun --standalone --nproc_per_node=4 train.py

============================================================
CONFIG
============================================================

Training and model settings are read from config.py.

The dataset folder should be configured there.

No meta.pkl is required.
"""


import os
import time
import math
import random
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group,
    destroy_process_group,
)

import config as cfg

from model import Model5555LM, Config


# ============================================================
# CONFIG HELPERS
# ============================================================

def cfg_get(name, default):
    """
    Read a value from config.py.

    If the value does not exist, use the supplied default.
    """
    return getattr(cfg, name, default)


# ============================================================
# OUTPUT / DATA
# ============================================================

out_dir = cfg_get(
    "out_dir",
    "out",
)

dataset_dir = cfg_get(
    "dataset_dir",
    cfg_get("data_dir", "data/fineweb"),
)

train_file = cfg_get(
    "train_file",
    "train.bin",
)

val_file = cfg_get(
    "val_file",
    "val.bin",
)


# ============================================================
# EVALUATION / LOGGING
# ============================================================

eval_interval = cfg_get(
    "eval_interval",
    100,
)

log_interval = cfg_get(
    "log_interval",
    10,
)

eval_iters = cfg_get(
    "eval_iters",
    25,
)

eval_only = cfg_get(
    "eval_only",
    False,
)

always_save_checkpoint = cfg_get(
    "always_save_checkpoint",
    True,
)


# ============================================================
# RESUME
# ============================================================

init_from = cfg_get(
    "init_from",
    "scratch",
)

checkpoint_name = cfg_get(
    "checkpoint_name",
    "ckpt.pt",
)


# ============================================================
# DATA
# ============================================================

batch_size = cfg_get(
    "batch_size",
    64,
)

block_size = cfg_get(
    "block_size",
    cfg_get("max_seq_len", 256),
)

gradient_accumulation_steps = cfg_get(
    "gradient_accumulation_steps",
    1,
)


# ============================================================
# OPTIMIZER
# ============================================================

learning_rate = cfg_get(
    "learning_rate",
    1e-3,
)

max_iters = cfg_get(
    "max_iters",
    1000,
)

weight_decay = cfg_get(
    "weight_decay",
    1e-1,
)

beta1 = cfg_get(
    "beta1",
    0.9,
)

beta2 = cfg_get(
    "beta2",
    0.95,
)

grad_clip = cfg_get(
    "grad_clip",
    1.0,
)


# ============================================================
# LEARNING RATE DECAY
# ============================================================

decay_lr = cfg_get(
    "decay_lr",
    True,
)

warmup_iters = cfg_get(
    "warmup_iters",
    50,
)

lr_decay_iters = cfg_get(
    "lr_decay_iters",
    max_iters,
)

min_lr = cfg_get(
    "min_lr",
    learning_rate / 10.0,
)


# ============================================================
# SYSTEM
# ============================================================

backend = cfg_get(
    "backend",
    "nccl",
)

device = cfg_get(
    "device",
    "cuda" if torch.cuda.is_available() else "cpu",
)

seed = cfg_get(
    "seed",
    1337,
)

compile_model = cfg_get(
    "compile",
    True,
)


# ============================================================
# PRECISION
# ============================================================

dtype_name = cfg_get(
    "dtype",
    "float16",
)


# ============================================================
# MFU
# ============================================================

peak_flops = cfg_get(
    "peak_flops",
    None,
)


# ============================================================
# VALIDATION
# ============================================================

if batch_size <= 0:
    raise ValueError("batch_size must be > 0")

if block_size <= 0:
    raise ValueError("block_size must be > 0")

if gradient_accumulation_steps <= 0:
    raise ValueError(
        "gradient_accumulation_steps must be > 0"
    )

if eval_iters <= 0:
    raise ValueError("eval_iters must be > 0")

if max_iters < 0:
    raise ValueError("max_iters must be >= 0")

if learning_rate <= 0:
    raise ValueError("learning_rate must be > 0")

if min_lr < 0:
    raise ValueError("min_lr must be >= 0")


# ============================================================
# DDP INITIALIZATION
# ============================================================

ddp = int(
    os.environ.get("RANK", -1)
) != -1


if ddp:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DDP currently expects CUDA/NCCL."
        )

    init_process_group(
        backend=backend
    )

    ddp_rank = int(
        os.environ["RANK"]
    )

    ddp_local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    ddp_world_size = int(
        os.environ["WORLD_SIZE"]
    )

    device = f"cuda:{ddp_local_rank}"

    torch.cuda.set_device(device)

    master_process = (
        ddp_rank == 0
    )

    seed_offset = ddp_rank

    if (
        gradient_accumulation_steps
        % ddp_world_size
        != 0
    ):
        raise ValueError(
            "gradient_accumulation_steps must "
            "be divisible by DDP world size."
        )

    gradient_accumulation_steps //= (
        ddp_world_size
    )

else:

    master_process = True

    seed_offset = 0

    ddp_rank = 0

    ddp_local_rank = 0

    ddp_world_size = 1


# ============================================================
# DEVICE
# ============================================================

device_type = (
    "cuda"
    if "cuda" in device
    and torch.cuda.is_available()
    else "cpu"
)


if device_type == "cuda":

    torch.cuda.set_device(device)


# ============================================================
# PRECISION SETUP
# ============================================================

dtype_name = dtype_name.lower()


dtype_map = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


if dtype_name not in dtype_map:
    raise ValueError(
        "dtype must be one of: "
        "float32, float16, bfloat16"
    )


ptdtype = dtype_map[dtype_name]


if device_type == "cpu":

    ctx = nullcontext()

else:

    ctx = torch.amp.autocast(
        device_type="cuda",
        dtype=ptdtype,
    )


# FP16 needs GradScaler.
# BF16 and FP32 do not.
use_grad_scaler = (
    device_type == "cuda"
    and ptdtype == torch.float16
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_grad_scaler,
)


# ============================================================
# RANDOM SEEDS
# ============================================================

final_seed = seed + seed_offset

random.seed(final_seed)

np.random.seed(final_seed)

torch.manual_seed(final_seed)

if device_type == "cuda":
    torch.cuda.manual_seed_all(
        final_seed
    )


# ============================================================
# CUDA PERFORMANCE SETTINGS
# ============================================================

if device_type == "cuda":

    torch.backends.cuda.matmul.allow_tf32 = True

    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# TOKENS PER ITERATION
# ============================================================

tokens_per_iter = (
    gradient_accumulation_steps
    * ddp_world_size
    * batch_size
    * block_size
)


if master_process:

    print(
        f"tokens per iteration : "
        f"{tokens_per_iter:,}"
    )

    print(
        f"dataset directory    : "
        f"{dataset_dir}"
    )

    print(
        f"train file           : "
        f"{train_file}"
    )

    print(
        f"val file             : "
        f"{val_file}"
    )


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

if master_process:

    os.makedirs(
        out_dir,
        exist_ok=True,
    )


# ============================================================
# DATASET PATHS
# ============================================================

train_path = os.path.join(
    dataset_dir,
    train_file,
)

val_path = os.path.join(
    dataset_dir,
    val_file,
)


if not os.path.exists(train_path):

    raise FileNotFoundError(
        f"Training dataset not found:\n"
        f"{train_path}"
    )


if not os.path.exists(val_path):

    raise FileNotFoundError(
        f"Validation dataset not found:\n"
        f"{val_path}"
    )


# ============================================================
# DATA LOADER
# ============================================================

train_data = np.memmap(
    train_path,
    dtype=np.uint16,
    mode="r",
)

val_data = np.memmap(
    val_path,
    dtype=np.uint16,
    mode="r",
)


if len(train_data) <= block_size:

    raise ValueError(
        "train.bin is too small for "
        f"block_size={block_size}"
    )


if len(val_data) <= block_size:

    raise ValueError(
        "val.bin is too small for "
        f"block_size={block_size}"
    )


def get_batch(split):
    """
    Returns:

        X: [batch_size, block_size]
        Y: [batch_size, block_size]

    Both are torch.long.
    """

    data = (
        train_data
        if split == "train"
        else val_data
    )

    max_start = (
        len(data) - block_size
    )

    ix = torch.randint(
        low=0,
        high=max_start,
        size=(batch_size,),
    )

    # Convert selected sequences to int64.
    #
    # This is required because nn.Embedding
    # expects integer indices.
    x = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[
                        int(i):
                        int(i) + block_size
                    ],
                    dtype=np.int64,
                )
            )
            for i in ix
        ]
    )

    y = torch.stack(
        [
            torch.from_numpy(
                np.asarray(
                    data[
                        int(i) + 1:
                        int(i) + 1 + block_size
                    ],
                    dtype=np.int64,
                )
            )
            for i in ix
        ]
    )

    # --------------------------------------------------------
    # Optional dataset/model compatibility validation.
    #
    # We deliberately do NOT inspect a tokenizer.
    # --------------------------------------------------------

    if cfg_vocab_size is not None:

        if torch.any(
            x >= cfg_vocab_size
        ) or torch.any(
            y >= cfg_vocab_size
        ):

            bad_x = int(
                x.max().item()
            )

            bad_y = int(
                y.max().item()
            )

            raise ValueError(
                "Dataset contains token IDs "
                "outside configured vocab_size.\n"
                f"vocab_size = {cfg_vocab_size}\n"
                f"maximum X token = {bad_x}\n"
                f"maximum Y token = {bad_y}"
            )

    if device_type == "cuda":

        x = (
            x.pin_memory()
            .to(
                device,
                non_blocking=True,
            )
        )

        y = (
            y.pin_memory()
            .to(
                device,
                non_blocking=True,
            )
        )

    else:

        x = x.to(device)

        y = y.to(device)

    return x, y


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Model5555LM.Config already reads the architecture
# configuration from config.py.
#
# Therefore train.py does NOT duplicate:
#
# hidden_size
# num_layers
# attention heads
# KV heads
# intermediate size
# RoPE
# etc.
#
# Those belong to model.py/config.py.

model_config = Config()


cfg_vocab_size = model_config.vocab_size


if block_size > model_config.max_seq_len:

    raise ValueError(
        f"block_size ({block_size}) is greater than "
        f"model max_seq_len ({model_config.max_seq_len})"
    )


# ============================================================
# TRAINING CONFIG SNAPSHOT
# ============================================================

train_config = {
    "out_dir": out_dir,
    "dataset_dir": dataset_dir,
    "train_file": train_file,
    "val_file": val_file,

    "eval_interval": eval_interval,
    "log_interval": log_interval,
    "eval_iters": eval_iters,
    "eval_only": eval_only,
    "always_save_checkpoint":
        always_save_checkpoint,

    "init_from": init_from,
    "checkpoint_name": checkpoint_name,

    "batch_size": batch_size,
    "block_size": block_size,
    "gradient_accumulation_steps":
        gradient_accumulation_steps,

    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "weight_decay": weight_decay,
    "beta1": beta1,
    "beta2": beta2,
    "grad_clip": grad_clip,

    "decay_lr": decay_lr,
    "warmup_iters": warmup_iters,
    "lr_decay_iters": lr_decay_iters,
    "min_lr": min_lr,

    "backend": backend,
    "device": device,
    "seed": seed,
    "dtype": dtype_name,
    "compile": compile_model,

    "peak_flops": peak_flops,
}


# ============================================================
# CHECKPOINT STATE
# ============================================================

iter_num = 0

best_val_loss = float("inf")

total_tokens = 0


# ============================================================
# MODEL CREATION
# ============================================================

if init_from == "scratch":

    if master_process:

        print()
        print(
            "Initializing Model5555LM "
            "from scratch..."
        )

    model = Model5555LM(
        config=model_config
    )


elif init_from == "resume":

    ckpt_path = os.path.join(
        out_dir,
        checkpoint_name,
    )

    if master_process:

        print(
            f"Resuming from:\n"
            f"{ckpt_path}"
        )

    if not os.path.exists(ckpt_path):

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{ckpt_path}"
        )

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
    )

    saved_model_config = (
        checkpoint["model_config"]
    )

    saved_train_config = (
        checkpoint.get(
            "train_config",
            {},
        )
    )

    # --------------------------------------------------------
    # The model architecture is restored from the checkpoint.
    # --------------------------------------------------------

    model_config = Config(
        **saved_model_config
    )

    model = Model5555LM(
        config=model_config
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    iter_num = checkpoint.get(
        "iter_num",
        0,
    )

    best_val_loss = checkpoint.get(
        "best_val_loss",
        float("inf"),
    )

    total_tokens = checkpoint.get(
        "total_tokens",
        0,
    )

    if master_process:

        print(
            f"Resumed at iteration "
            f"{iter_num:,}"
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.8f}"
        )

        print(
            f"Tokens processed: "
            f"{total_tokens:,}"
        )


else:

    raise ValueError(
        "init_from must be either "
        "'scratch' or 'resume'"
    )


# ============================================================
# FINAL MODEL VALIDATION
# ============================================================

if block_size > model.config.max_seq_len:

    raise ValueError(
        f"block_size={block_size} exceeds "
        f"model max_seq_len="
        f"{model.config.max_seq_len}"
    )


if cfg_vocab_size != model.config.vocab_size:

    if master_process:

        print(
            "WARNING: model vocab_size changed "
            "because a checkpoint was resumed."
        )

    cfg_vocab_size = (
        model.config.vocab_size
    )


# ============================================================
# MODEL INFO
# ============================================================

if master_process:

    print()

    print(
        "=" * 72
    )

    print(
        "MODEL READY"
    )

    print(
        "=" * 72
    )

    print(
        f"parameters         : "
        f"{model.get_num_params():,}"
    )

    print(
        f"trainable params   : "
        f"{model.get_trainable_params():,}"
    )

    print(
        f"vocab size         : "
        f"{model.config.vocab_size:,}"
    )

    print(
        f"hidden size        : "
        f"{model.config.hidden_size}"
    )

    print(
        f"layers             : "
        f"{model.config.num_hidden_layers}"
    )

    print(
        f"attention heads    : "
        f"{model.config.num_attention_heads}"
    )

    print(
        f"KV heads           : "
        f"{model.config.num_key_value_heads}"
    )

    print(
        f"intermediate size  : "
        f"{model.config.intermediate_size}"
    )

    print(
        f"context length     : "
        f"{model.config.max_seq_len}"
    )

    print(
        f"device             : "
        f"{device}"
    )

    print(
        f"dtype              : "
        f"{dtype_name}"
    )

    print(
        f"DDP world size     : "
        f"{ddp_world_size}"
    )

    print(
        "=" * 72
    )


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

# Model5555LM currently does not provide a
# configure_optimizers() method.
#
# So the trainer owns optimizer creation.

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)


# ============================================================
# RESUME OPTIMIZER
# ============================================================

if init_from == "resume":

    if "optimizer" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'optimizer' state."
        )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )


# Free checkpoint memory after loading.

if init_from == "resume":

    checkpoint = None


# ============================================================
# COMPILE
# ============================================================

if compile_model:

    if master_process:

        print()
        print(
            "Compiling model..."
        )

    unoptimized_model = model

    model = torch.compile(
        model
    )

else:

    unoptimized_model = model


# ============================================================
# DDP
# ============================================================

if ddp:

    model = DDP(
        model,
        device_ids=[
            ddp_local_rank
        ],
    )


# ============================================================
# RAW MODEL
# ============================================================

raw_model = (
    model.module
    if ddp
    else model
)


# ============================================================
# LOSS FUNCTION
# ============================================================

def compute_loss(logits, targets):
    """
    logits:
        [B, T, vocab_size]

    targets:
        [B, T]

    returns:
        scalar loss
    """

    if logits.ndim != 3:

        raise RuntimeError(
            "Model must return logits with "
            "shape [B, T, vocab_size]. "
            f"Received: {tuple(logits.shape)}"
        )

    if targets.ndim != 2:

        raise RuntimeError(
            "Targets must have shape [B, T]. "
            f"Received: {tuple(targets.shape)}"
        )

    if (
        logits.shape[0]
        != targets.shape[0]
        or logits.shape[1]
        != targets.shape[1]
    ):

        raise RuntimeError(
            "Logits and targets have "
            "incompatible shapes.\n"
            f"logits  : {tuple(logits.shape)}\n"
            f"targets : {tuple(targets.shape)}"
        )

    # Compute CE in float32 for improved numerical
    # stability when using FP16.
    #
    # This does not change the model's parameter dtype.

    logits_for_loss = logits.float()

    return F.cross_entropy(
        logits_for_loss.reshape(
            -1,
            logits.shape[-1],
        ),
        targets.reshape(-1),
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    results = {}

    model.eval()

    for split in (
        "train",
        "val",
    ):

        losses = torch.zeros(
            eval_iters,
            dtype=torch.float32,
        )

        for k in range(eval_iters):

            X, Y = get_batch(
                split
            )

            with ctx:

                logits = model(X)

                loss = compute_loss(
                    logits,
                    Y,
                )

            losses[k] = (
                loss.detach()
                .float()
                .cpu()
            )

        results[split] = (
            losses.mean().item()
        )

    model.train()

    return results


# ============================================================
# LEARNING RATE
# ============================================================

def get_lr(it):

    if not decay_lr:

        return learning_rate

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    if it < warmup_iters:

        return (
            learning_rate
            * (it + 1)
            / (warmup_iters + 1)
        )

    # --------------------------------------------------------
    # After decay period
    # --------------------------------------------------------

    if it > lr_decay_iters:

        return min_lr

    # --------------------------------------------------------
    # Cosine decay
    # --------------------------------------------------------

    if (
        lr_decay_iters
        <= warmup_iters
    ):

        return min_lr

    decay_ratio = (
        (it - warmup_iters)
        / (lr_decay_iters - warmup_iters)
    )

    decay_ratio = max(
        0.0,
        min(1.0, decay_ratio),
    )

    coeff = (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * decay_ratio
            )
        )
    )

    return (
        min_lr
        + coeff
        * (
            learning_rate
            - min_lr
        )
    )


# ============================================================
# CUDA MEMORY HELPERS
# ============================================================

def get_cuda_memory():

    if device_type != "cuda":

        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "peak_gb": 0.0,
        }

    allocated = (
        torch.cuda.memory_allocated(
            device
        )
    )

    reserved = (
        torch.cuda.memory_reserved(
            device
        )
    )

    peak = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    divisor = 1024 ** 3

    return {
        "allocated_gb":
            allocated / divisor,

        "reserved_gb":
            reserved / divisor,

        "peak_gb":
            peak / divisor,
    }


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    current_iter,
    current_best_val_loss,
):

    if not master_process:

        return

    checkpoint = {

        "model":
            raw_model.state_dict(),

        "optimizer":
            optimizer.state_dict(),

        "model_config":
            asdict(
                raw_model.config
            ),

        "train_config":
            train_config,

        "iter_num":
            current_iter,

        "best_val_loss":
            current_best_val_loss,

        "total_tokens":
            total_tokens,
    }

    checkpoint_path = os.path.join(
        out_dir,
        checkpoint_name,
    )

    print(
        f"saving checkpoint to "
        f"{checkpoint_path}"
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )


# ============================================================
# TRAINING START
# ============================================================

X, Y = get_batch(
    "train"
)


t0 = time.perf_counter()

local_iter_num = 0

running_mfu = -1.0


# Reset CUDA peak memory before actual training.

if device_type == "cuda":

    torch.cuda.reset_peak_memory_stats(
        device
    )


# ============================================================
# TRAINING LOOP
# ============================================================

while True:

    # ========================================================
    # LEARNING RATE
    # ========================================================

    lr = get_lr(
        iter_num
    )

    for param_group in (
        optimizer.param_groups
    ):

        param_group["lr"] = lr


    # ========================================================
    # EVALUATION / CHECKPOINT
    # ========================================================

    if (
        iter_num
        % eval_interval
        == 0
        and master_process
    ):

        losses = estimate_loss()

        print()
        print(
            "=" * 72
        )

        print(
            f"step {iter_num:,}"
        )

        print(
            f"train loss : "
            f"{losses['train']:.8f}"
        )

        print(
            f"val loss   : "
            f"{losses['val']:.8f}"
        )

        print(
            f"learning rate : "
            f"{lr:.10e}"
        )

        print(
            "=" * 72
        )

        if (
            losses["val"]
            < best_val_loss
            or always_save_checkpoint
        ):

            if (
                losses["val"]
                < best_val_loss
            ):

                best_val_loss = (
                    losses["val"]
                )

            if iter_num > 0:

                save_checkpoint(
                    iter_num,
                    best_val_loss,
                )


    # ========================================================
    # EVAL ONLY
    # ========================================================

    if (
        iter_num == 0
        and eval_only
    ):

        break


    # ========================================================
    # GRADIENT ACCUMULATION
    # ========================================================

    optimizer.zero_grad(
        set_to_none=True
    )


    last_loss = None


    for micro_step in range(
        gradient_accumulation_steps
    ):

        # ----------------------------------------------------
        # DDP optimization:
        #
        # Only synchronize gradients on the final
        # micro-step.
        # ----------------------------------------------------

        if ddp:

            model.require_backward_grad_sync = (
                micro_step
                == gradient_accumulation_steps - 1
            )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with ctx:

            logits = model(X)

            loss = compute_loss(
                logits,
                Y,
            )

            loss = (
                loss
                / gradient_accumulation_steps
            )


        # ----------------------------------------------------
        # Prefetch next batch.
        #
        # CUDA transfers can overlap with computation
        # when using pinned memory + non_blocking=True.
        # ----------------------------------------------------

        X, Y = get_batch(
            "train"
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()


        last_loss = loss


    # ========================================================
    # GRADIENT CLIPPING
    # ========================================================

    grad_norm = None

    if grad_clip != 0.0:

        scaler.unscale_(
            optimizer
        )

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )
        )

    else:

        if use_grad_scaler:

            scaler.unscale_(
                optimizer
            )


    # ========================================================
    # OPTIMIZER STEP
    # ========================================================

    scaler.step(
        optimizer
    )

    scaler.update()


    # ========================================================
    # TIMING
    # ========================================================

    if device_type == "cuda":

        # Make timing accurate only after the actual
        # optimizer work has completed.

        torch.cuda.synchronize(
            device
        )


    t1 = time.perf_counter()

    dt = t1 - t0

    t0 = t1


    # ========================================================
    # TOKEN ACCOUNTING
    # ========================================================

    step_tokens = (
        batch_size
        * block_size
        * gradient_accumulation_steps
        * ddp_world_size
    )

    total_tokens += step_tokens


    tokens_per_second = (
        step_tokens / dt
        if dt > 0
        else 0.0
    )


    # ========================================================
    # MFU
    # ========================================================

    mfu = None

    if (
        peak_flops is not None
        and tokens_per_second > 0
    ):

        try:

            mfu = raw_model.estimate_mfu(
                tokens_per_second=
                    tokens_per_second,
                peak_flops=
                    peak_flops,
            )

        except Exception:

            mfu = None


    if mfu is not None:

        if running_mfu < 0:

            running_mfu = mfu

        else:

            running_mfu = (
                0.9 * running_mfu
                + 0.1 * mfu
            )


    # ========================================================
    # LOGGING
    # ========================================================

    if (
        iter_num
        % log_interval
        == 0
        and master_process
    ):

        # Undo gradient accumulation scaling.
        loss_value = (
            last_loss
            .detach()
            .float()
            .item()
            * gradient_accumulation_steps
        )

        memory = get_cuda_memory()


        if grad_norm is not None:

            grad_norm_value = float(
                grad_norm.detach()
                .float()
                .item()
            )

        else:

            grad_norm_value = 0.0


        print(
            f"iter {iter_num:6d} | "
            f"loss {loss_value:.8f} | "
            f"lr {lr:.3e} | "
            f"grad {grad_norm_value:.5f} | "
            f"{tokens_per_second:,.0f} tok/s | "
            f"{dt * 1000:.2f} ms"
        )


        print(
            f"             "
            f"tokens {total_tokens:,} | "
            f"peak {memory['peak_gb']:.3f} GB | "
            f"alloc {memory['allocated_gb']:.3f} GB | "
            f"reserved {memory['reserved_gb']:.3f} GB"
        )


        if running_mfu >= 0:

            print(
                f"             "
                f"MFU {running_mfu:.4f}%"
            )


    # ========================================================
    # ITERATION UPDATE
    # ========================================================

    iter_num += 1

    local_iter_num += 1


    # ========================================================
    # TERMINATION
    # ========================================================

    if iter_num > max_iters:

        break


# ============================================================
# FINAL CHECKPOINT
# ============================================================

if master_process:

    print()
    print(
        "=" * 72
    )

    print(
        "TRAINING FINISHED"
    )

    print(
        "=" * 72
    )

    print(
        f"iterations       : "
        f"{iter_num:,}"
    )

    print(
        f"total tokens     : "
        f"{total_tokens:,}"
    )

    print(
        f"best val loss    : "
        f"{best_val_loss:.8f}"
    )

    print(
        f"checkpoint       : "
        f"{os.path.join(out_dir, checkpoint_name)}"
    )

    print(
        "=" * 72
    )


# ============================================================
# DDP CLEANUP
# ============================================================

if ddp:

    destroy_process_group()
