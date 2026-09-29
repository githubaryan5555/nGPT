"""DDP-capable trainer for Model5555LM.

The binary files are already tokenized. Configuration is loaded by config.py from
config.json, while command-line ``--name=value`` arguments can override it.

Optimizer:
    - Muon for 2-D hidden-layer matrices
    - AdamW for embeddings, LM head, norms, biases, and other parameters
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
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import config as cfg
from model import Config, Model5555LM


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

model_name = cfg.model_name
dataset_dtype = np.dtype(cfg.dataset_dtype)

if dataset_dtype.kind not in "iu":
    raise ValueError("dataset_dtype must be an integer NumPy dtype")


# ---------------------------------------------------------------------------
# DDP / DEVICE
# ---------------------------------------------------------------------------

ddp = int(os.environ.get("RANK", -1)) >= 0

if ddp:
    dist.init_process_group(backend=cfg.backend)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

else:
    rank = 0
    local_rank = 0
    world_size = 1
    device = cfg.device

master_process = rank == 0

if "cuda" in device and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is unavailable")

device_type = "cuda" if "cuda" in device else "cpu"

if device_type == "cpu" and cfg.dtype != "float32":
    dtype_name = "float32"
else:
    dtype_name = cfg.dtype

ptdtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[dtype_name]

ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(
        device_type="cuda",
        dtype=ptdtype,
    )
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=device_type == "cuda" and dtype_name == "float16",
)


# ---------------------------------------------------------------------------
# TRAINING SIZES
# ---------------------------------------------------------------------------

if cfg.gradient_accumulation_steps % world_size:
    raise ValueError(
        "gradient_accumulation_steps must be divisible by world size"
    )

local_grad_accum = cfg.gradient_accumulation_steps // world_size

max_seq_len = cfg.max_seq_len
batch_size = cfg.batch_size

tokens_per_step = (
    cfg.gradient_accumulation_steps
    * batch_size
    * max_seq_len
)

if master_process:
    os.makedirs(cfg.out_dir, exist_ok=True)

    print(f"device={device}, world_size={world_size}, dtype={dtype_name}")
    print(f"tokens per optimizer step={tokens_per_step:,}")


# ---------------------------------------------------------------------------
# RANDOM SEEDS / CUDA
# ---------------------------------------------------------------------------

torch.manual_seed(cfg.seed + rank)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

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
                starts[:, None] + offsets[None, :]
            ],
            dtype=np.int64,
        )

        x = torch.from_numpy(
            windows[:, :-1].copy()
        )

        y = torch.from_numpy(
            windows[:, 1:].copy()
        )

        if device_type == "cuda":
            x = (
                x.pin_memory()
                .to(device, non_blocking=True)
            )

            y = (
                y.pin_memory()
                .to(device, non_blocking=True)
            )

        else:
            x = x.to(device)
            y = y.to(device)

        return x, y


batches = TokenBatches()


# ---------------------------------------------------------------------------
# MODEL CONFIG
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CHECKPOINTS
# ---------------------------------------------------------------------------

def checkpoint_files():
    pattern = os.path.join(
        cfg.out_dir,
        f"{model_name}_ckpt_*.pt",
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

    return (
        legacy
        if os.path.isfile(legacy)
        else None
    )


# ---------------------------------------------------------------------------
# CREATE MODEL
# ---------------------------------------------------------------------------

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
        f"fp16 size={model.get_model_size_mb():.8g} MB"
    )


# ---------------------------------------------------------------------------
# HYBRID MUON + ADAMW
# ---------------------------------------------------------------------------
#
# Muon is intended for 2-D hidden-layer matrices.
#
# AdamW handles:
#   - token embeddings
#   - LM head
#   - RMSNorm parameters
#   - biases
#   - all other non-2-D parameters
#
# We identify embedding/output parameters by name so that tied embeddings
# are not accidentally sent through Muon.
# ---------------------------------------------------------------------------

muon_params = []
adamw_decay = []
adamw_no_decay = []

muon_names = []
adamw_decay_names = []
adamw_no_decay_names = []

for name, parameter in model.named_parameters():

    if not parameter.requires_grad:
        continue

    lower_name = name.lower()

    is_embedding_or_head = (
        "embedding" in lower_name
        or "embed" in lower_name
        or "wte" in lower_name
        or "wpe" in lower_name
        or "lm_head" in lower_name
        or lower_name.endswith(".head.weight")
        or lower_name.endswith("head.weight")
    )

    if parameter.ndim == 2 and not is_embedding_or_head:
        muon_params.append(parameter)
        muon_names.append(name)

    elif parameter.ndim >= 2:
        adamw_decay.append(parameter)
        adamw_decay_names.append(name)

    else:
        adamw_no_decay.append(parameter)
        adamw_no_decay_names.append(name)


if not hasattr(torch.optim, "Muon"):
    raise RuntimeError(
        "torch.optim.Muon is unavailable in this PyTorch build. "
        "Use a PyTorch version containing torch.optim.Muon."
    )


optimizer_muon = torch.optim.Muon(
    muon_params,
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
    momentum=0.95,
    nesterov=True,
)


fused = (
    device_type == "cuda"
    and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
)


optimizer_adamw = torch.optim.AdamW(
    [
        {
            "params": adamw_decay,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": adamw_no_decay,
            "weight_decay": 0.0,
        },
    ],
    lr=cfg.learning_rate,
    betas=(cfg.beta1, cfg.beta2),
    fused=fused,
)


if master_process:
    print(
        f"Muon parameters={sum(p.numel() for p in muon_params):,}"
    )

    print(
        f"AdamW parameters="
        f"{sum(p.numel() for p in adamw_decay) + sum(p.numel() for p in adamw_no_decay):,}"
    )

    print(
        f"Muon tensors={len(muon_params)}, "
        f"AdamW tensors={len(adamw_decay) + len(adamw_no_decay)}"
    )


# ---------------------------------------------------------------------------
# RESUME OPTIMIZER STATES
# ---------------------------------------------------------------------------

if checkpoint is not None:

    if "optimizer_muon" in checkpoint:
        optimizer_muon.load_state_dict(
            checkpoint["optimizer_muon"]
        )

    elif master_process:
        print(
            "warning: checkpoint has no Muon optimizer state; "
            "starting Muon state from scratch"
        )

    if "optimizer_adamw" in checkpoint:
        optimizer_adamw.load_state_dict(
            checkpoint["optimizer_adamw"]
        )

    elif master_process:
        print(
            "warning: checkpoint has no AdamW optimizer state; "
            "starting AdamW state from scratch"
        )


# ---------------------------------------------------------------------------
# COMPILE / DDP
# ---------------------------------------------------------------------------

if cfg.compile_model:
    model = torch.compile(model)

if ddp:
    model = DDP(
        model,
        device_ids=[local_rank],
    )

raw_model = model.module if ddp else model


# ---------------------------------------------------------------------------
# LOSS
# ---------------------------------------------------------------------------

def language_loss(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss():
    """Every rank evaluates, then all ranks receive the same global averages."""

    model.eval()

    totals = torch.zeros(
        2,
        device=device,
        dtype=torch.float64,
    )

    for index, split in enumerate(
        ("train", "val")
    ):
        for _ in range(cfg.eval_iters):

            x, y = batches.get(split)

            with ctx:
                value = language_loss(
                    model(x),
                    y,
                )

            totals[index] += (
                value.detach().double()
            )

    if ddp:
        dist.all_reduce(
            totals,
            op=dist.ReduceOp.SUM,
        )

    totals /= (
        cfg.eval_iters
        * world_size
    )

    model.train()

    return {
        "train": totals[0].item(),
        "val": totals[1].item(),
    }


# ---------------------------------------------------------------------------
# LEARNING RATE
# ---------------------------------------------------------------------------

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
        step - cfg.warmup_iters
    ) / (
        cfg.lr_decay_iters
        - cfg.warmup_iters
    )

    return (
        cfg.min_lr
        + 0.5
        * (1.0 + math.cos(math.pi * ratio))
        * (cfg.learning_rate - cfg.min_lr)
    )


# ---------------------------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------------------------

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

    except (ImportError, Exception) as exc:

        if master_process:
            print(
                f"warning: tokenizer unavailable ({exc})"
            )


# ---------------------------------------------------------------------------
# SAMPLING
# ---------------------------------------------------------------------------

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
                f"  [{number + 1}] generation failed: {exc}"
            )

    raw_model.train(was_training)


# ---------------------------------------------------------------------------
# SAVE CHECKPOINT
# ---------------------------------------------------------------------------

def save_checkpoint(step, val_loss):

    if not master_process or not cfg.save_checkpoint:
        return

    payload = {
        "model": raw_model.state_dict(),

        "optimizer_muon": (
            optimizer_muon.state_dict()
        ),

        "optimizer_adamw": (
            optimizer_adamw.state_dict()
        ),

        "model_args": raw_model.get_config(),

        "iter_num": step,

        "best_val_loss": val_loss,

        "config": cfg.as_dict(),

        "rng_state": torch.get_rng_state(),
    }

    path = os.path.join(
        cfg.out_dir,
        f"{model_name}_ckpt_{step}.pt",
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
        f"{model_name}_latest.pt",
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


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------

X, Y = batches.get("train")

last_time = time.perf_counter()

local_step = 0

running_mfu = None


while iter_num < cfg.max_iters:

    lr = get_lr(iter_num)

    # Same LR schedule applied to both optimizers.
    for group in optimizer_muon.param_groups:
        group["lr"] = lr

    for group in optimizer_adamw.param_groups:
        group["lr"] = lr


    # ---------------------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------------------

    if iter_num % cfg.eval_interval == 0:

        # This collective must be called by every rank.
        # Only rank zero prints/saves.

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

        if ddp:
            dist.barrier()


    if iter_num == 0 and cfg.eval_only:
        break


    # ---------------------------------------------------------------
    # FORWARD + BACKWARD
    # ---------------------------------------------------------------

    step_loss_sum = torch.zeros(
        (),
        device=device,
        dtype=torch.float64,
    )


    for micro_step in range(
        local_grad_accum
    ):

        if ddp:
            model.require_backward_grad_sync = (
                micro_step
                == local_grad_accum - 1
            )


        with ctx:

            logits = model(X)

            micro_loss = language_loss(
                logits,
                Y,
            )

            loss = (
                micro_loss
                / local_grad_accum
            )


        step_loss_sum += (
            micro_loss.detach().double()
        )


        X, Y = batches.get("train")


        scaler.scale(loss).backward()


    # ---------------------------------------------------------------
    # GRADIENT CLIPPING
    # ---------------------------------------------------------------

    if cfg.grad_clip:

        scaler.unscale_(
            optimizer_muon
        )

        scaler.unscale_(
            optimizer_adamw
        )

        torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(),
            cfg.grad_clip,
        )


    # ---------------------------------------------------------------
    # OPTIMIZER STEP
    # ---------------------------------------------------------------

    scaler.step(
        optimizer_muon
    )

    scaler.step(
        optimizer_adamw
    )

    scaler.update()


    # ---------------------------------------------------------------
    # ZERO GRAD
    # ---------------------------------------------------------------

    optimizer_muon.zero_grad(
        set_to_none=True
    )

    optimizer_adamw.zero_grad(
        set_to_none=True
    )


    # ---------------------------------------------------------------
    # DDP LOSS
    # ---------------------------------------------------------------

    if ddp:

        dist.all_reduce(
            step_loss_sum,
            op=dist.ReduceOp.SUM,
        )


    average_step_loss = (
        step_loss_sum
        / (
            local_grad_accum
            * world_size
        )
    ).item()


    # ---------------------------------------------------------------
    # SPEED / MFU
    # ---------------------------------------------------------------

    now = time.perf_counter()

    elapsed = now - last_time

    last_time = now


    if (
        iter_num % cfg.log_interval == 0
        and master_process
    ):

        tokens_per_second = (
            tokens_per_step
            / max(elapsed, 1e-9)
        )

        flops_per_token = (
            raw_model.get_flops_per_token(
                max_seq_len
            )
        )

        flops_per_second = (
            tokens_per_second
            * flops_per_token
        )

        mfu = None

        if cfg.peak_flops:

            mfu = raw_model.estimate_mfu(
                tokens_per_second,
                cfg.peak_flops,
            )

            running_mfu = (
                mfu
                if running_mfu is None
                else 0.9 * running_mfu
                + 0.1 * mfu
            )

        mfu_text = (
            "n/a"
            if running_mfu is None
            else f"{running_mfu:.8g}%"
        )


        print(
            f"iter={iter_num:06d} "
            f"loss={average_step_loss:.16g} "
            f"time={elapsed:.8g}s "
            f"tokens/s={tokens_per_second:.8g} "
            f"FLOP/token={flops_per_token:.8g} "
            f"FLOP/s={flops_per_second:.8g} "
            f"MFU={mfu_text}"
        )


    iter_num += 1
    local_step += 1


# ---------------------------------------------------------------------------
# FINAL CHECKPOINT
# ---------------------------------------------------------------------------

# max_iters is an exclusive update count:
# max_iters=1000 performs steps 0..999.

if (
    master_process
    and cfg.save_checkpoint
    and iter_num > 0
):
    save_checkpoint(
        iter_num,
        best_val_loss,
    )


# ---------------------------------------------------------------------------
# DDP CLEANUP
# ---------------------------------------------------------------------------

if ddp:
    dist.destroy_process_group()
