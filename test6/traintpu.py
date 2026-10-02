"""PyTorch/XLA TPU trainer for Model5555LM."""

import argparse
import glob
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
            value = value.lower() in {
                "1", "true", "yes", "on"
            }
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

device = torch_xla.device()
device_type = "xla"

# TPU v5e: BF16 is preferred.
dtype_name = getattr(cfg, "dtype", "bfloat16")

if dtype_name == "float32":
    ptdtype = torch.float32
    ctx = nullcontext()
else:
    dtype_name = "bfloat16"
    ptdtype = torch.bfloat16
    ctx = torch.autocast(
        device_type="xla",
        dtype=torch.bfloat16,
    )

master_process = xm.is_master_ordinal(local=False)


# ============================================================
# MEMORY REPORT
# ============================================================

def report_memory(tag):
    xm.mark_step()

    m = xm.get_memory_info(device)

    used = m["bytes_used"] / (1024 ** 3)
    peak = m["peak_bytes_used"] / (1024 ** 3)
    limit = m["bytes_limit"] / (1024 ** 3)

    if master_process:
        print(
            f"[TPU] {tag} | "
            f"used={used:.3f} GiB | "
            f"peak={peak:.3f} GiB | "
            f"limit={limit:.3f} GiB"
        )


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

    print("=" * 60)
    print("TPU TRAINING")
    print("=" * 60)
    print(f"device={device}")
    print(f"dtype={dtype_name}")
    print(f"batch_size={batch_size}")
    print(
        f"gradient_accumulation_steps="
        f"{cfg.gradient_accumulation_steps}"
    )
    print(f"seq_len={max_seq_len}")
    print(f"tokens/optimizer_step={tokens_per_step:,}")
    print("=" * 60)


torch.manual_seed(cfg.seed)


# ============================================================
# DATA
# ============================================================

dataset_dtype = np.dtype(cfg.dataset_dtype)

if dataset_dtype.kind not in "iu":
    raise ValueError(
        "dataset_dtype must be an integer NumPy dtype"
    )


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
                    f"{path} must contain more "
                    f"than max_seq_len tokens"
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

        # CPU tensors.
        #
        # Do NOT move these to TPU here.
        x = torch.from_numpy(
            windows[:, :-1].copy()
        ).long()

        y = torch.from_numpy(
            windows[:, 1:].copy()
        ).long()

        return x, y


batches = TokenBatches()


# ============================================================
# MODEL CONFIG
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


# ------------------------------------------------------------
# Move model to TPU.
# ------------------------------------------------------------

model.to(device)

# Important:
# We let autocast handle BF16 computation rather than
# permanently converting every parameter to BF16.
#
# AdamW optimizer states remain FP32 where appropriate.


if master_process and cfg.print_model_info:

    print(
        f"parameters={model.get_num_params():,}"
    )

    print(
        f"model size BF16="
        f"{model.get_model_size_mb(2):.2f} MB"
    )

    print(
        f"model size FP32="
        f"{model.get_model_size_mb(4):.2f} MB"
    )


report_memory("after model")


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


if (
    checkpoint is not None
    and "optimizer" in checkpoint
):
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )


# ============================================================
# OPTIONAL COMPILE
# ============================================================

compile_enabled = getattr(
    cfg,
    "compile_model",
    False,
)

if compile_enabled:

    if master_process:
        print(
            "torch.compile/openxla enabled"
        )

    model = torch.compile(
        model,
        backend="openxla",
    )

else:

    if master_process:
        print(
            "torch.compile disabled"
        )


raw_model = model


# ============================================================
# LOSS
# ============================================================

def language_loss(
    logits,
    targets,
):

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

        total = 0.0

        for _ in range(cfg.eval_iters):

            x_cpu, y_cpu = (
                batches.get(split)
            )

            x = x_cpu.to(
                device=device,
                dtype=torch.long,
            )

            y = y_cpu.to(
                device=device,
                dtype=torch.long,
            )

            with ctx:

                logits = model(x)

                value = language_loss(
                    logits,
                    y,
                )

            # Bring only the scalar to CPU.
            total += float(
                value.float().item()
            )

            del logits
            del value
            del x
            del y
            del x_cpu
            del y_cpu

            xm.mark_step()

        totals[split] = (
            total / cfg.eval_iters
        )

    model.train()

    report_memory(
        f"after {split} evaluation"
    )

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
        / (
            cfg.lr_decay_iters
            - cfg.warmup_iters
        )
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

if os.path.isfile(
    cfg.tokenizer_path
):

    try:

        from tokenizers import Tokenizer

        backend_tokenizer = (
            Tokenizer.from_file(
                cfg.tokenizer_path
            )
        )

        class TokenizerAdapter:

            def encode(self, text):
                return (
                    backend_tokenizer
                    .encode(text)
                    .ids
                )

            def decode(self, ids):
                return (
                    backend_tokenizer
                    .decode(
                        ids,
                        skip_special_tokens=True,
                    )
                )

        tokenizer = TokenizerAdapter()

    except Exception as exc:

        if master_process:
            print(
                f"warning: tokenizer unavailable "
                f"({exc})"
            )


# ============================================================
# SAMPLING
# ============================================================

def sample_text():

    if (
        tokenizer is None
        or not master_process
    ):
        return

    was_training = raw_model.training

    raw_model.eval()

    print("samples:")

    for number in range(
        cfg.sample_count
    ):

        source, _ = batches.get(
            "val"
        )

        prompt_ids = source[
            0,
            :min(
                cfg.sample_prompt_tokens,
                max_seq_len,
            ),
        ].tolist()

        prompt = tokenizer.decode(
            prompt_ids
        )

        try:

            output = raw_model.generate(
                prompt,
                tokenizer,
                max_new_tokens=(
                    cfg.sample_new_tokens
                ),
                temperature=(
                    cfg.sample_temperature
                ),
                top_k=cfg.sample_top_k,
                top_p=cfg.sample_top_p,
                eos_token_id=cfg.eos_token_id,
            )

            print(
                f"  [{number + 1}] "
                f"{output}"
            )

        except Exception as exc:

            print(
                f"  [{number + 1}] "
                f"generation failed: {exc}"
            )

    raw_model.train(
        was_training
    )

    xm.mark_step()


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    step,
    val_loss,
):

    if not master_process:
        return

    if not cfg.save_checkpoint:
        return

    xm.mark_step()

    # Move state to CPU immediately.
    state_dict = {
        key.removeprefix(
            "_orig_mod."
        ): value.cpu()
        for key, value
        in raw_model.state_dict().items()
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

    del state_dict
    del payload

    print(
        f"saved checkpoint: {path}"
    )


# ============================================================
# TRAINING
# ============================================================

# CPU batch.
X, Y = batches.get("train")

last_time = time.perf_counter()


while iter_num < cfg.max_iters:

    lr = get_lr(iter_num)

    for group in optimizer.param_groups:
        group["lr"] = lr


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        iter_num % cfg.eval_interval
        == 0
    ):

        losses = estimate_loss()

        if master_process:

            print(
                f"step={iter_num:06d} "
                f"train_loss="
                f"{losses['train']:.8f} "
                f"val_loss="
                f"{losses['val']:.8f} "
                f"lr={lr:.8g}"
            )

            sample_text()

            improved = (
                losses["val"]
                < best_val_loss
            )

            if improved:
                best_val_loss = (
                    losses["val"]
                )

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


    if (
        iter_num == 0
        and cfg.eval_only
    ):
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

        # ----------------------------------------------------
        # Move ONLY current batch to TPU.
        # ----------------------------------------------------

        x = X.to(
            device=device,
            dtype=torch.long,
        )

        y = Y.to(
            device=device,
            dtype=torch.long,
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        with ctx:

            logits = model(x)

            micro_loss = (
                language_loss(
                    logits,
                    y,
                )
            )

            loss = (
                micro_loss
                / cfg.gradient_accumulation_steps
            )


        # ----------------------------------------------------
        # CPU scalar for logging.
        # ----------------------------------------------------

        step_loss_sum += float(
            micro_loss.detach()
            .float()
            .item()
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Release forward references.
        # ----------------------------------------------------

        del logits
        del micro_loss
        del loss
        del x
        del y


        # ----------------------------------------------------
        # Prepare next CPU batch.
        # ----------------------------------------------------

        if (
            micro_step + 1
            < cfg.gradient_accumulation_steps
        ):

            X, Y = batches.get(
                "train"
            )


        # ----------------------------------------------------
        # Execute XLA graph.
        # ----------------------------------------------------

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
    # Optimizer
    # --------------------------------------------------------

    xm.optimizer_step(
        optimizer,
        barrier=False,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    xm.mark_step()


    # --------------------------------------------------------
    # Next batch
    # --------------------------------------------------------

    X, Y = batches.get(
        "train"
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
        iter_num % cfg.log_interval
        == 0
        and master_process
    ):

        tokens_per_second = (
            tokens_per_step
            / max(
                elapsed,
                1e-9,
            )
        )

        try:

            flops_per_token = (
                raw_model
                .get_flops_per_token(
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
            f"loss={average_step_loss:.8f} "
            f"time={elapsed:.4f}s "
            f"tokens/s="
            f"{tokens_per_second:.2f} "
            f"FLOP/token="
            f"{flops_per_token:.4g} "
            f"FLOP/s="
            f"{flops_per_second:.4g}"
        )


    # --------------------------------------------------------
    # Memory reporting.
    # --------------------------------------------------------

    if (
        iter_num == 0
        or (
            iter_num % cfg.log_interval
            == 0
        )
    ):
        report_memory(
            f"after step {iter_num}"
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

report_memory("final")
