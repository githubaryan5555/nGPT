import math
import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F

import config
from model import Model5555


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.seed)


# ============================================================
# DEVICE
# ============================================================

if config.device == "cuda" and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"device: {device}")


# ============================================================
# DTYPE
# ============================================================

if config.dtype == "float32":
    dtype = torch.float32
elif config.dtype == "float16":
    dtype = torch.float16
elif config.dtype == "bfloat16":
    dtype = torch.bfloat16
elif config.dtype == "auto":
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32
else:
    raise ValueError(f"Unknown dtype: {config.dtype}")

print(f"dtype:  {dtype}")


# ============================================================
# DATA
# ============================================================

train_data = np.memmap(
    config.train_data,
    dtype=np.uint16,
    mode="r",
)

val_data = np.memmap(
    config.val_data,
    dtype=np.uint16,
    mode="r",
)

print(f"train tokens: {len(train_data):,}")
print(f"val tokens:   {len(val_data):,}")


def get_batch(split):
    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - config.block_size - 1,
        (config.batch_size,),
    )

    x = torch.stack(
        [
            torch.from_numpy(
                data[i:i + config.block_size].astype(np.int64)
            )
            for i in ix
        ]
    )

    y = torch.stack(
        [
            torch.from_numpy(
                data[i + 1:i + 1 + config.block_size].astype(np.int64)
            )
            for i in ix
        ]
    )

    if config.preload_data_to_gpu:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


# ============================================================
# MODEL
# ============================================================

model = Model5555(
    vocab_size=config.vocab_size,
    hidden_size=config.hidden_size,
    num_hidden_layers=config.num_hidden_layers,
    hidden_dropout=config.hidden_dropout,
    tie_word_embeddings=config.tie_word_embeddings,
)

model = model.to(device)

print(
    f"parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)


# ============================================================
# OPTIONAL COMPILE
# ============================================================

if config.compile and hasattr(torch, "compile"):
    print("compiling model...")
    model = torch.compile(
        model,
        mode=config.compile_mode,
    )


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    betas=(config.beta1, config.beta2),
    weight_decay=config.weight_decay,
)


# ============================================================
# LEARNING RATE
# ============================================================

def get_lr(iteration):
    if not config.decay_lr:
        return config.learning_rate

    if iteration < config.warmup_iters:
        return config.learning_rate * (
            iteration + 1
        ) / config.warmup_iters

    if iteration > config.lr_decay_iters:
        return config.min_lr

    decay_ratio = (
        iteration - config.warmup_iters
    ) / (
        config.lr_decay_iters - config.warmup_iters
    )

    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )

    return config.min_lr + coeff * (
        config.learning_rate - config.min_lr
    )


# ============================================================
# AMP
# ============================================================

use_amp = (
    device.type == "cuda"
    and dtype in (torch.float16, torch.bfloat16)
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(dtype == torch.float16 and use_amp),
)


def autocast_context():
    if use_amp:
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
        )

    return torch.autocast(
        device_type=device.type,
        enabled=False,
    )


# ============================================================
# LOSS
# ============================================================

def compute_loss(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():
    model.eval()

    losses = {}

    for split in ("train", "val"):
        values = torch.zeros(config.eval_iters)

        for k in range(config.eval_iters):
            X, Y = get_batch(split)

            with autocast_context():
                logits = model(X)
                loss = compute_loss(logits, Y)

            values[k] = loss.detach().float().cpu()

        losses[split] = values.mean().item()

    model.train()

    return losses


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(iteration, best_val_loss):
    os.makedirs(config.out_dir, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iteration,
        "best_val_loss": best_val_loss,
        "config": {
            name: getattr(config, name)
            for name in dir(config)
            if not name.startswith("_")
        },
    }

    path = os.path.join(
        config.out_dir,
        "ckpt.pt",
    )

    torch.save(checkpoint, path)

    print(f"saved checkpoint: {path}")


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

model.train()

for iteration in range(config.max_iters):

    lr = get_lr(iteration)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()

    total_loss = 0.0

    for micro_step in range(
        config.gradient_accumulation_steps
    ):
        X, Y = get_batch("train")

        with autocast_context():
            logits = model(X)
            loss = compute_loss(logits, Y)

            loss = (
                loss
                / config.gradient_accumulation_steps
            )

        total_loss += loss.detach().float().item()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    if config.grad_clip > 0:

        if scaler.is_enabled():
            scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if iteration % config.log_interval == 0:
        print(
            f"iter {iteration:5d} | "
            f"loss {total_loss:.4f} | "
            f"lr {lr:.6g} | "
            f"time {elapsed:.2f}s"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        iteration % config.eval_interval == 0
        or iteration == config.max_iters - 1
    ):
        losses = estimate_loss()

        print(
            f"eval {iteration:5d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

            save_checkpoint(
                iteration,
                best_val_loss,
            )


print("training complete.")
