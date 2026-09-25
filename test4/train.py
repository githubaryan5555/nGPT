"""Training entry point for Model5555LM.

The dataset is expected to already contain token ids; no tokenisation is performed
here.  Paths and model/training defaults live in ``config.py``.
"""

import argparse
import glob
import json
import math
import os
import re
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, init_process_group

import config as cfg
from model import Config, Model5555LM


def _cli_overrides():
    """Apply simple ``--name=value`` overrides without requiring configurator.py."""
    aliases = {"compile": "compile_model", "decay_lr": "lr_decay"}
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--resume", action="store_true", help="resume the newest checkpoint")
    parser.add_argument("--eval_only", action="store_true")
    args, unknown = parser.parse_known_args()
    for item in unknown:
        if not item.startswith("--") or "=" not in item:
            continue
        name, value = item[2:].split("=", 1)
        name = aliases.get(name, name)
        if not hasattr(cfg, name):
            continue
        old = getattr(cfg, name)
        if isinstance(old, bool):
            value = value.lower() in ("1", "true", "yes", "y", "on")
        elif isinstance(old, int) and not isinstance(old, bool):
            value = int(value)
        elif isinstance(old, float):
            value = float(value)
        setattr(cfg, name, value)
    if args.resume:
        cfg.init_from = "resume"
    if args.eval_only:
        cfg.eval_only = True


_cli_overrides()

# Centralised config values, with backwards-compatible fallbacks for config.py files
# that predate this trainer.
model_name = getattr(cfg, "model_name", "model5555")
dataset_dir = getattr(cfg, "dataset_dir", "dataset")
train_bin = getattr(cfg, "train_bin", os.path.join(dataset_dir, "train.bin"))
val_bin = getattr(cfg, "val_bin", os.path.join(dataset_dir, "val.bin"))
dataset_dtype_name = getattr(cfg, "dataset_dtype", "uint16")
max_seq_len = getattr(cfg, "max_seq_len", 256)
out_dir = getattr(cfg, "out_dir", f"out_{model_name}")
if out_dir == "checkpoints" and not hasattr(cfg, "model_name"):
    out_dir = f"out_{model_name}"

# Keep the requested convention: model-name_ckpt_step-number.pt.
checkpoint_pattern = os.path.join(out_dir, f"{model_name}_ckpt_*.pt")

try:
    dataset_dtype = np.dtype(dataset_dtype_name)
except TypeError as exc:
    raise ValueError(f"Unsupported dataset_dtype: {dataset_dtype_name!r}") from exc
if dataset_dtype.kind not in "iu":
    raise ValueError("dataset_dtype must be an integer NumPy dtype")

# DDP setup.
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    init_process_group(backend=cfg.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    if cfg.gradient_accumulation_steps % ddp_world_size:
        raise ValueError("gradient_accumulation_steps must be divisible by DDP world size")
    gradient_accumulation_steps = cfg.gradient_accumulation_steps // ddp_world_size
else:
    ddp_rank, ddp_world_size, seed_offset = 0, 1, 0
    master_process = True
    device = cfg.device
    gradient_accumulation_steps = cfg.gradient_accumulation_steps

if "cuda" in device and not torch.cuda.is_available():
    raise RuntimeError("config.device requests CUDA, but CUDA is not available")
device_type = "cuda" if "cuda" in device else "cpu"
if device_type == "cpu" and cfg.dtype != "float32":
    print("CPU does not support the configured mixed precision reliably; using float32")
    dtype_name = "float32"
else:
    dtype_name = cfg.dtype
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
use_scaler = device_type == "cuda" and dtype_name == "float16"
scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * cfg.batch_size * max_seq_len
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"tokens per iteration: {tokens_per_iter:,}")

torch.manual_seed(getattr(cfg, "seed", 1337) + seed_offset)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

for path in (train_bin, val_bin):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing tokenized dataset file: {path}")

# Memmaps are recreated per batch to avoid long-lived worker/memmap leaks.
def get_batch(split):
    path = train_bin if split == "train" else val_bin
    data = np.memmap(path, dtype=dataset_dtype, mode="r")
    if len(data) <= max_seq_len:
        raise ValueError(f"{path} must contain more than max_seq_len tokens")
    ix = torch.randint(len(data) - max_seq_len, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[int(i):int(i) + max_seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[int(i) + 1:int(i) + 1 + max_seq_len].astype(np.int64)) for i in ix])
    if device_type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

# Optional BPE tokenizer, used only for qualitative samples.
tokenizer = None
tokenizer_path = getattr(cfg, "tokenizer_path", "tokenizer/tokenizer.json")
try:
    from tokenizers import Tokenizer
    if os.path.isfile(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    elif master_process:
        print(f"warning: tokenizer not found at {tokenizer_path}; sample generation disabled")
except ImportError:
    if master_process:
        print("warning: install `tokenizers` to enable generated text samples")

def decode_ids(ids):
    try:
        return tokenizer.decode([int(x) for x in ids], skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode([int(x) for x in ids])

def sample_text(raw_model, count=5):
    if tokenizer is None:
        return
    print("generated samples:")
    was_training = raw_model.training
    raw_model.eval()
    for sample_index in range(count):
        source, _ = get_batch("val")
        prompt_ids = source[0, :max(1, min(16, max_seq_len // 4))].detach().cpu()
        prompt = decode_ids(prompt_ids)
        try:
            text = raw_model.generate(
                prompt, tokenizer,
                max_new_tokens=getattr(cfg, "sample_new_tokens", 80),
                temperature=getattr(cfg, "sample_temperature", 0.8),
                top_k=getattr(cfg, "sample_top_k", 50),
                top_p=getattr(cfg, "sample_top_p", 0.95),
                eos_token_id=getattr(cfg, "eos_token_id", None),
            )
            print(f"  [{sample_index + 1}] {text}")
        except Exception as exc:
            print(f"  [{sample_index + 1}] generation failed: {exc}")
    raw_model.train(was_training)

# Build the model using the names and defaults expected by model.py.
def make_model(model_values=None):
    values = {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "intermediate_size": cfg.intermediate_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "attention_dropout": cfg.attention_dropout,
        "hidden_dropout": cfg.hidden_dropout,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_seq_len": max_seq_len,
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "initializer_range": cfg.initializer_range,
    }
    if model_values:
        values.update(model_values)
    return Model5555LM(Config(**values))


def newest_checkpoint():
    files = glob.glob(checkpoint_pattern)
    if not files:
        # Also accept a legacy checkpoint if one exists in the configured output dir.
        legacy = os.path.join(out_dir, getattr(cfg, "checkpoint_name", "ckpt.pt"))
        return legacy if os.path.isfile(legacy) else None
    return max(files, key=lambda p: int(re.search(r"_(\d+)\.pt$", p).group(1)))

iter_num = 0
best_val_loss = float("inf")
checkpoint = None
if getattr(cfg, "init_from", "scratch") == "resume":
    ckpt_path = newest_checkpoint()
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in {out_dir} matching {checkpoint_pattern}")
    print(f"resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = make_model(checkpoint.get("model_args") or checkpoint.get("model_config"))
    state = checkpoint["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    model.load_state_dict(state)
    iter_num = int(checkpoint.get("iter_num", checkpoint.get("step", 0)))
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
else:
    print("initializing a new model from scratch")
    model = make_model()

model.to(device)
if master_process and getattr(cfg, "print_model_info", True):
    print(f"parameters: {model.get_num_params():,}")
    print(f"model size (fp16): {model.get_model_size_mb():.8g} MB")

# AdamW implementation kept in train.py because model.py is intentionally unchanged.
decay, no_decay = [], []
for param in model.parameters():
    if param.requires_grad:
        (decay if param.ndim >= 2 else no_decay).append(param)
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
    fused=(device_type == "cuda" and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames),
)
if checkpoint is not None and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])

if getattr(cfg, "compile_model", False):
    print("compiling the model...")
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

@torch.no_grad()
def estimate_loss():
    model.eval()
    result = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(split)
            with ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
        result[split] = float(np.mean(losses))
    model.train()
    return result

def get_lr(step):
    if not getattr(cfg, "lr_decay", True):
        return cfg.learning_rate
    if step < cfg.warmup_iters:
        return cfg.learning_rate * (step + 1) / (cfg.warmup_iters + 1)
    if step >= cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (step - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (cfg.learning_rate - cfg.min_lr)

X, Y = get_batch("train")
t0 = time.time()
running_mfu = None
local_iter_num = 0
while True:
    lr = get_lr(iter_num)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if iter_num % cfg.eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.16g}, val loss {losses['val']:.16g}, lr {lr:.16g}")
        sample_text(raw_model, 5)
        should_save = getattr(cfg, "save_checkpoint", True) and (losses["val"] < best_val_loss or getattr(cfg, "always_save_checkpoint", False))
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
        if should_save and iter_num > 0:
            ckpt = {
                "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_args": raw_model.get_config(), "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
            }
            path = os.path.join(out_dir, f"{model_name}_ckpt_{iter_num}.pt")
            torch.save(ckpt, path)
            print(f"saved checkpoint: {path}")
    if iter_num == 0 and cfg.eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch("train")
        scaler.scale(loss).backward()
    if cfg.grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    now = time.time()
    dt = now - t0
    t0 = now
    if iter_num % cfg.log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        mfu_text = "n/a"
        if getattr(cfg, "peak_flops", None) and local_iter_num >= 5:
            tok_s = tokens_per_iter / max(dt, 1e-9)
            mfu = raw_model.estimate_mfu(tok_s, cfg.peak_flops)
            running_mfu = mfu if running_mfu is None else 0.9 * running_mfu + 0.1 * mfu
            mfu_text = f"{running_mfu:.8g}%"
        print(f"iter {iter_num}: loss {lossf:.16g}, time {dt * 1000:.8g}ms, mfu {mfu_text}")
    iter_num += 1
    local_iter_num += 1
    if iter_num > cfg.max_iters:
        break

if ddp:
    destroy_process_group()
