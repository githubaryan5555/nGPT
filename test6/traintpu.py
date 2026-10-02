"""PyTorch/XLA trainer for Model5555LM."""

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

import config as cfg
from model import Config, Model5555LM

import torch_xla
import torch_xla.core.xla_model as xm


# ============================================================
# CLI
# ============================================================

def apply_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    args, unknown = parser.parse_known_args()

    aliases = {
        "compile": "compile_model",
        "decay_lr": "lr_decay",
    }

    for item in unknown:
        if not item.startswith("--") or "=" not in item:
            continue

        name, value = item[2:].split("=", 1)
        name = aliases.get(name, name)

        if not hasattr(cfg, name):
            continue

        old = getattr(cfg, name)

        if isinstance(old, bool):
            value = value.lower() in {"1", "true", "yes", "on"}
        elif isinstance(old, int) and not isinstance(old, bool):
            value = int(value)
        elif isinstance(old, float):
            value = float(value)
        elif value.lower() == "none":
            value = None

        setattr(cfg, name, value)

    if args.resume:
        cfg.init_from = "resume"

    if args.eval_only:
        cfg.eval_only = True


apply_cli()


# ============================================================
# XLA DEVICE
# ============================================================

device = xm.xla_device()
device_type = "xla"

# XLA generally uses bfloat16 on TPU.
dtype_name = getattr(cfg, "dtype", "bfloat16")

if dtype_name == "float32":
    ptdtype = torch.float32
elif dtype_name == "bfloat16":
    ptdtype = torch.bfloat16
else:
    # TPU training should normally use BF16 rather than FP16.
    print(f"warning: XLA trainer replacing dtype={dtype_name} with bfloat16")
    dtype_name = "bfloat16"
    ptdtype = torch.bfloat16

ctx = (
    torch.autocast(device_type="xla", dtype=ptdtype)
    if dtype_name != "float32"
    else nullcontext()
)

master_process = xm.is_master_ordinal(local=False)

dataset_dtype = np.dtype(cfg.dataset_dtype)

if dataset_dtype.kind not in "iu":
    raise ValueError("dataset_dtype must be an integer NumPy dtype")


# ============================================================
# BASIC SETUP
# ============================================================

max_seq_len = cfg.max_seq_len
batch_size = cfg.batch_size

tokens_per_step = (
    cfg.gradient_accumulation_steps
    * batch_size
    * max_seq_len
)

if master_process:
    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"device={device}")
    print(f"dtype={dtype_name}")
    print(f"tokens per optimizer step={tokens_per_step:,}")


torch.manual_seed(cfg.seed)


# ============================================================
# DATA
# ============================================================

class TokenBatches:

    def __init__(self):
        self.maps = {}

        for split, path in (
            ("train", cfg.train_bin),
            ("val", cfg.val_bin),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

            data = np.memmap(
                path,
                dtype=dataset_dtype,
                mode="r",
            )

            if len(data) <= max_seq_len:
                raise ValueError(
                    f"{path} must contain more than max_seq_len tokens"
                )

            self.maps[split] = data

    def get(self, split):
        data = self.maps[split]

        starts = torch.randint(
            0,
            len(data) - max_seq_len,
            (batch_size,),
        ).numpy()

        offsets = np.arange(
            max_seq_len + 1,
            dtype=np.int64,
        )

        windows = np.asarray(
            data[
                starts[:, None]
                + offsets[None, :]
            ],
            dtype=np.int64,
        )

        x = torch.from_numpy(
            windows[:, :-1].copy()
        )

        y = torch.from_numpy(
            windows[:, 1:].copy()
        )

        x = x.to(device)
        y = y.to(device)

        return x, y


batches = TokenBatches()


# ============================================================
# MODEL
# ============================================================

def model_values():

    return {
        name: getattr(cfg, name)
        for name in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "attention_dropout",
            "hidden_dropout",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "initializer_range",
        )
    } | {
        "max_seq_len": max_seq_len
    }


def make_model(values=None):

    args = model_values()

    if values:
        args.update(values)

    return Model5555LM(
        Config(**args)
    )


# ============================================================
# CHECKPOINTS
# ============================================================

def checkpoint_files():

    pattern = os.path.join(
        cfg.out_dir,
        f"{cfg.model_name}_ckpt_*.pt",
    )

    files = []

    for path in glob.glob(pattern):

        match = re.search(
            r"_(\d+)\.pt$",
            path,
        )

        if match:
            files.append(
                (
                    int(match.group(1)),
                    path,
                )
            )

    files.sort()

    return files


def latest_checkpoint():

    files = checkpoint_files()

    if files:
        return files[-1][1]

    legacy = os.path.join(
        cfg.out_dir,
        cfg.checkpoint_name,
    )

    if os.path.isfile(legacy):
        return legacy

    return None


# ============================================================
# LOAD MODEL
# ============================================================

iter_num = 0
best_val_loss = float("inf")
checkpoint = None

if cfg.init_from == "resume":

    path = latest_checkpoint()

    if path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {cfg.out_dir}"
        )

    if master_process:
        print(f"resuming from {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model = make_model(
        checkpoint.get("model_args")
    )

    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model"].items()
    }

    model.load_state_dict(state)

    iter_num = int(
        checkpoint.get(
            "iter_num",
            checkpoint.get("step", 0),
        )
    )

    best_val_loss = float(
        checkpoint.get(
            "best_val_loss",
            float("inf"),
        )
    )

else:

    model = make_model()


model.to(device)


if master_process and cfg.print_model_info:

    print(
        f"parameters={model.get_num_params():,}, "
        f"model size={model.get_model_size_mb():.8g} MB"
    )


# ============================================================
# OPTIMIZER
# ============================================================

decay = []
no_decay = []

for parameter in model.parameters():

    if not parameter.requires_grad:
        continue

    if parameter.ndim >= 2:
        decay.append(parameter)
    else:
        no_decay.append(parameter)


optimizer = torch.optim.AdamW(
    [
        {
            "params": decay,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
    ],
    lr=cfg.learning_rate,
    betas=(
        cfg.beta1,
        cfg.beta2,
    ),
)


if checkpoint is not None and "optimizer" in checkpoint:
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )


# ============================================================
# XLA COMPILE
# ============================================================

# torch.compile is optional on XLA.
if getattr(cfg, "compile_model", False):

    if master_process:
        print("torch.compile enabled")

    model = torch.compile(
        model,
        backend="openxla",
    )


raw_model = model


# ============================================================
# LOSS
# ============================================================

def language_loss(logits, targets):

    return F.cross_entropy(
        logits.reshape(
            -1,
            logits.size(-1),
        ),
        targets.reshape(-1),
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    totals = {
        "train": 0.0,
        "val": 0.0,
    }

    for split in ("train", "val"):

        total = torch.zeros(
            (),
            device=device,
            dtype=torch.float32,
        )

        for _ in range(cfg.eval_iters):

            x, y = batches.get(split)

            with ctx:

                logits = model(x)

                value = language_loss(
                    logits,
                    y,
                )

            total += value.detach()

            xm.mark_step()

        total /= cfg.eval_iters

        # Get scalar back to CPU.
        total_value = total.item()

        totals[split] = total_value

    model.train()

    return totals


# ============================================================
# LR SCHEDULE
# ============================================================

def get_lr(step):

    if not cfg.lr_decay:
        return cfg.learning_rate

    if step < cfg.warmup_iters:

        return (
            cfg.learning_rate
            * (step + 1)
            / (cfg.warmup_iters + 1)
        )

    if step >= cfg.lr_decay_iters:
        return cfg.min_lr

    ratio = (
        (step - cfg.warmup_iters)
        / (cfg.lr_decay_iters - cfg.warmup_iters)
    )

    return (
        cfg.min_lr
        + 0.5
        * (1.0 + math.cos(math.pi * ratio))
        * (
            cfg.learning_rate
            - cfg.min_lr
        )
    )


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = None

if os.path.isfile(cfg.tokenizer_path):

    try:

        from tokenizers import Tokenizer

        backend_tokenizer = Tokenizer.from_file(
            cfg.tokenizer_path
        )

        class TokenizerAdapter:

            def encode(self, text):
                return backend_tokenizer.encode(text).ids

            def decode(self, ids):
                return backend_tokenizer.decode(
                    ids,
                    skip_special_tokens=True,
                )

        tokenizer = TokenizerAdapter()

    except Exception as exc:

        if master_process:
            print(
                f"warning: tokenizer unavailable ({exc})"
            )


# ============================================================
# SAMPLING
# ============================================================

def sample_text():

    if tokenizer is None or not master_process:
        return

    was_training = raw_model.training

    raw_model.eval()

    print("samples:")

    for number in range(cfg.sample_count):

        source, _ = batches.get("val")

        prompt_ids = source[
            0,
            :min(
                cfg.sample_prompt_tokens,
                max_seq_len,
            ),
        ].cpu().tolist()

        prompt = tokenizer.decode(
            prompt_ids
        )

        try:

            output = raw_model.generate(
                prompt,
                tokenizer,
                max_new_tokens=cfg.sample_new_tokens,
                temperature=cfg.sample_temperature,
                top_k=cfg.sample_top_k,
                top_p=cfg.sample_top_p,
                eos_token_id=cfg.eos_token_id,
            )

            print(
                f"  [{number + 1}] {output}"
            )

        except Exception as exc:

            print(
                f"  [{number + 1}] "
                f"generation failed: {exc}"
            )

    raw_model.train(was_training)

    xm.mark_step()


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(step, val_loss):

    if not master_process:
        return

    if not cfg.save_checkpoint:
        return

    # Ensure all XLA operations are complete.
    xm.mark_step()

    state_dict = {
        key.removeprefix("_orig_mod."): value.cpu()
        for key, value in raw_model.state_dict().items()
    }

    payload = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "model_args": raw_model.get_config(),
        "iter_num": step,
        "best_val_loss": val_loss,
        "config": cfg.as_dict(),
        "rng_state": torch.get_rng_state(),
    }

    path = os.path.join(
        cfg.out_dir,
        f"{cfg.model_name}_ckpt_{step}.pt",
    )

    temporary = (
        path
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        payload,
        temporary,
    )

    os.replace(
        temporary,
        path,
    )

    latest = os.path.join(
        cfg.out_dir,
        f"{cfg.model_name}_latest.pt",
    )

    latest_tmp = (
        latest
        + f".tmp.{os.getpid()}"
    )

    torch.save(
        payload,
        latest_tmp,
    )

    os.replace(
        latest_tmp,
        latest,
    )

    print(
        f"saved checkpoint: {path}"
    )


# ============================================================
# TRAINING
# ============================================================

X, Y = batches.get("train")

last_time = time.perf_counter()

while iter_num < cfg.max_iters:

    lr = get_lr(iter_num)

    for group in optimizer.param_groups:
        group["lr"] = lr

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if iter_num % cfg.eval_interval == 0:

        losses = estimate_loss()

        if master_process:

            print(
                f"step={iter_num:06d} "
                f"train_loss={losses['train']:.16g} "
                f"val_loss={losses['val']:.16g} "
                f"lr={lr:.16g}"
            )

            sample_text()

            improved = (
                losses["val"]
                < best_val_loss
            )

            if improved:
                best_val_loss = losses["val"]

            if (
                iter_num > 0
                and (
                    improved
                    or cfg.always_save_checkpoint
                )
            ):
                save_checkpoint(
                    iter_num,
                    best_val_loss,
                )

    if iter_num == 0 and cfg.eval_only:
        break

    # --------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )

    step_loss_sum = 0.0

    for micro_step in range(
        cfg.gradient_accumulation_steps
    ):

        with ctx:

            logits = model(X)

            micro_loss = language_loss(
                logits,
                Y,
            )

            loss = (
                micro_loss
                / cfg.gradient_accumulation_steps
            )

        step_loss_sum += (
            micro_loss.detach()
            .float()
            .item()
        )

        loss.backward()

        X, Y = batches.get("train")

        # Tell XLA that this section can be lowered.
        xm.mark_step()

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    if cfg.grad_clip:

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip,
        )

    # --------------------------------------------------------
    # XLA optimizer step
    # --------------------------------------------------------

    xm.optimizer_step(
        optimizer,
        barrier=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    average_step_loss = (
        step_loss_sum
        / cfg.gradient_accumulation_steps
    )

    now = time.perf_counter()

    elapsed = (
        now - last_time
    )

    last_time = now

    if (
        iter_num % cfg.log_interval == 0
        and master_process
    ):

        tokens_per_second = (
            tokens_per_step
            / max(elapsed, 1e-9)
        )

        try:
            flops_per_token = (
                raw_model.get_flops_per_token(
                    max_seq_len
                )
            )

            flops_per_second = (
                tokens_per_second
                * flops_per_token
            )

        except Exception:

            flops_per_token = 0.0
            flops_per_second = 0.0

        print(
            f"iter={iter_num:06d} "
            f"loss={average_step_loss:.16g} "
            f"time={elapsed:.8g}s "
            f"tokens/s={tokens_per_second:.8g} "
            f"FLOP/token={flops_per_token:.8g} "
            f"FLOP/s={flops_per_second:.8g}"
        )

    iter_num += 1


# ============================================================
# FINAL CHECKPOINT
# ============================================================

if (
    master_process
    and cfg.save_checkpoint
    and iter_num > 0
):
    save_checkpoint(
        iter_num,
        best_val_loss,
    )


xm.mark_step()
