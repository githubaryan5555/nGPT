"""
Single-GPU model-agnostic training script.

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

MODEL CONTRACT
--------------

The trainer does not know what architecture is being trained.

The model only needs to satisfy:

    model(X, Y) -> (logits, loss)

where:

    X      : [batch, sequence]
    Y      : [batch, sequence]
    logits : [batch, sequence, vocab_size]
    loss   : scalar tensor

The model may be a Transformer, RWKV, MLP, SSM, CNN,
custom architecture, etc.

Model construction is completely controlled by config.py.
"""

import os
import time
import math
import json
import inspect
import importlib
from contextlib import nullcontext

import numpy as np
import torch


# =============================================================================
# Load config.py
# =============================================================================

import config as user_config


# Keep arbitrary config objects.

# The old trainer only accepted int/float/bool/str.
# That prevented things such as:
#
#     model_kwargs = {...}
#
# from being used.
#
# Private names are still ignored.
config = {
    k: v
    for k, v in vars(user_config).items()
    if not k.startswith("_")
}

globals().update(config)


# =============================================================================
# Configuration helpers
# =============================================================================

def fail(message):
    raise RuntimeError(
        f"\nCONFIGURATION ERROR:\n{message}\n"
    )


def require(condition, message):
    if not condition:
        fail(message)


# =============================================================================
# Required generic training configuration
# =============================================================================

required_config = [
    "vocab_size",
    "block_size",
    "batch_size",
    "gradient_accumulation_steps",
    "max_iters",
    "eval_iters",
    "eval_interval",
    "log_interval",
    "learning_rate",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
    "decay_lr",
    "warmup_iters",
    "lr_decay_iters",
    "min_lr",
    "init_from",
    "preload_data_to_gpu",
    "device",
    "dtype",
    "seed",
    "data_root",
    "dataset",
    "out_dir",
    "compile",
    "compile_mode",
    "eval_only",
    "always_save_checkpoint",
    "wandb_log",
    "wandb_project",
    "wandb_run_name",
]


for name in required_config:
    require(
        name in config,
        f"Missing required config variable: {name}"
    )


# =============================================================================
# Basic validation
# =============================================================================

require(
    vocab_size > 0,
    "vocab_size must be > 0."
)

require(
    block_size > 0,
    "block_size must be > 0."
)

require(
    batch_size > 0,
    "batch_size must be > 0."
)

require(
    gradient_accumulation_steps > 0,
    "gradient_accumulation_steps must be > 0."
)

require(
    max_iters >= 0,
    "max_iters must be >= 0."
)

require(
    eval_iters > 0,
    "eval_iters must be > 0."
)

require(
    eval_interval > 0,
    "eval_interval must be > 0."
)

require(
    log_interval > 0,
    "log_interval must be > 0."
)

require(
    learning_rate > 0,
    "learning_rate must be > 0."
)

require(
    weight_decay >= 0,
    "weight_decay must be >= 0."
)

require(
    0 <= beta1 < 1,
    "beta1 must be in [0, 1)."
)

require(
    0 <= beta2 < 1,
    "beta2 must be in [0, 1)."
)

require(
    grad_clip >= 0,
    "grad_clip must be >= 0."
)

require(
    init_from in {"scratch", "resume"},
    "init_from must be either 'scratch' or 'resume'."
)

require(
    isinstance(preload_data_to_gpu, bool),
    "preload_data_to_gpu must be True or False."
)

if decay_lr:

    require(
        warmup_iters >= 0,
        "warmup_iters must be >= 0."
    )

    require(
        lr_decay_iters > warmup_iters,
        "lr_decay_iters must be greater than warmup_iters."
    )

    require(
        min_lr >= 0,
        "min_lr must be >= 0."
    )

    require(
        min_lr <= learning_rate,
        "min_lr must be <= learning_rate."
    )


# =============================================================================
# Device
# =============================================================================

requested_device = str(device)


if requested_device.startswith("cuda"):

    require(
        torch.cuda.is_available(),
        (
            f"config requests device='{requested_device}', "
            "but CUDA is not available."
        )
    )

    device = torch.device("cuda:0")


elif requested_device == "cpu":

    device = torch.device("cpu")


elif requested_device.startswith("mps"):

    require(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available(),
        (
            f"config requests device='{requested_device}', "
            "but MPS is not available."
        )
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

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cudnn.benchmark = True


# =============================================================================
# Dtype / AMP
# =============================================================================

requested_dtype = str(dtype).lower()


require(
    requested_dtype in {
        "auto",
        "float32",
        "fp32",
        "float16",
        "fp16",
        "bfloat16",
        "bf16",
    },
    (
        f"Unsupported dtype='{dtype}'. "
        "Use auto, float32, float16, or bfloat16."
    )
)


def choose_dtype():

    if requested_dtype in {"float32", "fp32"}:
        return torch.float32


    if requested_dtype in {"float16", "fp16"}:

        require(
            using_cuda,
            "float16 training requires CUDA."
        )

        return torch.float16


    if requested_dtype in {"bfloat16", "bf16"}:

        if using_cuda:

            require(
                torch.cuda.is_bf16_supported(),
                (
                    "dtype='bfloat16' was requested, "
                    "but this CUDA device does not report BF16 support."
                )
            )

        return torch.bfloat16


    # auto

    if using_cuda:

        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


ptdtype = choose_dtype()


# =============================================================================
# Autocast
# =============================================================================

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

    autocast_enabled = True

    autocast_context = torch.autocast(
        device_type="cpu",
        dtype=torch.bfloat16,
    )


else:

    autocast_enabled = False
    autocast_context = nullcontext()


# =============================================================================
# GradScaler
# =============================================================================

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

train_path = os.path.join(
    data_dir,
    "train.bin",
)

val_path = os.path.join(
    data_dir,
    "val.bin",
)

dataset_config_path = os.path.join(
    data_dir,
    "dataset_config.json",
)

tokenizer_config_path = os.path.join(
    data_dir,
    "tokenizer_config.json",
)

tokenizer_path = os.path.join(
    data_dir,
    "tokenizer.json",
)


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

        with open(
            dataset_config_path,
            "r",
            encoding="utf-8",
        ) as f:

            dataset_config = json.load(f)

    except Exception as e:

        fail(
            f"Could not read dataset_config.json:\n{e}"
        )


# =============================================================================
# Dataset dtype
# =============================================================================

def normalize_numpy_dtype(value):

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

    storage_dtype = np.uint16

    print(
        "Dataset dtype metadata not found; "
        "using uint16 for .bin files."
    )


storage_dtype = np.dtype(storage_dtype)


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
        "Supported types are uint16, uint32, int16, int32, int64."
    )
)


print(
    f"Dataset storage dtype: {storage_dtype}"
)


# =============================================================================
# Dataset memmaps
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
# Dataset validation
# =============================================================================

def validate_token_range(data, name):

    sample_size = min(
        len(data),
        1_000_000,
    )


    if sample_size == len(data):

        sample = data

    else:

        indices = np.linspace(
            0,
            len(data) - 1,
            num=sample_size,
            dtype=np.int64,
        )

        sample = data[indices]


    require(
        sample.size > 0,
        f"{name} is empty."
    )


    if np.issubdtype(
        storage_dtype,
        np.signedinteger,
    ):

        min_token = int(sample.min())

        require(
            min_token >= 0,
            (
                f"{name} contains negative token ID "
                f"{min_token}."
            )
        )


    max_token = int(sample.max())


    require(
        max_token < vocab_size,
        (
            f"{name} contains token ID {max_token}, "
            f"but vocab_size={vocab_size}. "
            f"Valid IDs are 0..{vocab_size - 1}."
        )
    )


    print(
        f"{name}: "
        f"{len(data):,} tokens | "
        f"sample max ID={max_token:,} | "
        f"range OK"
    )


validate_token_range(
    train_data,
    "train.bin",
)

validate_token_range(
    val_data,
    "val.bin",
)


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
print("streaming       : enabled")
print("preload to GPU  : disabled")
print("=" * 70)


# =============================================================================
# Batch loader
# =============================================================================

cpu_x = torch.empty(
    (
        batch_size,
        block_size,
    ),
    dtype=torch.long,
)

cpu_y = torch.empty(
    (
        batch_size,
        block_size,
    ),
    dtype=torch.long,
)


# Pin CPU buffers once.

if using_cuda:

    cpu_x = cpu_x.pin_memory()
    cpu_y = cpu_y.pin_memory()


def get_batch(split):

    data = (
        train_data
        if split == "train"
        else val_data
    )


    max_start = (
        len(data)
        - block_size
        - 1
    )


    starts = torch.randint(
        0,
        max_start + 1,
        (
            batch_size,
        ),
        dtype=torch.int64,
    ).numpy()


    for row, start in enumerate(starts):

        x_np = np.asarray(
            data[
                start:
                start + block_size
            ],
            dtype=storage_dtype,
        )

        y_np = np.asarray(
            data[
                start + 1:
                start + 1 + block_size
            ],
            dtype=storage_dtype,
        )


        cpu_x[row].copy_(
            torch.from_numpy(
                x_np
            )
        )


        cpu_y[row].copy_(
            torch.from_numpy(
                y_np
            )
        )


    if using_cuda:

        return (
            cpu_x.to(
                device,
                non_blocking=True,
            ),
            cpu_y.to(
                device,
                non_blocking=True,
            ),
        )


    return (
        cpu_x.to(device),
        cpu_y.to(device),
    )


# =============================================================================
# Training information
# =============================================================================

tokens_per_iter = (
    batch_size
    * block_size
    * gradient_accumulation_steps
)


print()
print("=" * 70)
print("TRAINING")
print("=" * 70)
print(f"device                  : {device}")
print(f"dtype                   : {ptdtype}")
print(f"AMP                     : {autocast_enabled}")
print(f"GradScaler              : {use_grad_scaler}")
print(f"batch size              : {batch_size}")
print(f"gradient accumulation   : {gradient_accumulation_steps}")
print(f"effective tokens/step   : {tokens_per_iter:,}")
print(f"learning rate           : {learning_rate:g}")
print(f"max iterations          : {max_iters:,}")
print(f"compile                 : {compile}")
print("=" * 70)


# =============================================================================
# Generic model loading
# =============================================================================

require(
    "model_module" in config,
    (
        "config.py must define model_module."
    )
)

require(
    "model_class" in config,
    (
        "config.py must define model_class."
    )
)

require(
    "model_kwargs" in config,
    (
        "config.py must define model_kwargs."
    )
)

require(
    isinstance(model_kwargs, dict),
    "model_kwargs must be a dictionary."
)


def load_model_class():

    module = importlib.import_module(
        str(model_module)
    )

    require(
        hasattr(module, str(model_class)),
        (
            f"Model class '{model_class}' "
            f"was not found in module '{model_module}'."
        )
    )

    return module, getattr(
        module,
        str(model_class),
    )


def build_model():

    module, model_cls = load_model_class()


    config_class_name = config.get(
        "model_config_class",
        None,
    )


    # -------------------------------------------------------------------------
    # Model directly accepts kwargs
    # -------------------------------------------------------------------------

    if config_class_name is None:

        try:

            return model_cls(
                **model_kwargs
            )

        except TypeError as e:

            fail(
                "Could not construct the configured model.\n"
                f"model class : {model_class}\n"
                f"kwargs      : {model_kwargs}\n"
                f"error       : {e}"
            )


    # -------------------------------------------------------------------------
    # Model expects a configuration object
    # -------------------------------------------------------------------------

    require(
        hasattr(module, str(config_class_name)),
        (
            f"Model config class '{config_class_name}' "
            f"was not found in module '{model_module}'."
        )
    )


    config_cls = getattr(
        module,
        str(config_class_name),
    )


    try:

        model_config = config_cls(
            **model_kwargs
        )

    except TypeError as e:

        fail(
            "Could not construct model configuration.\n"
            f"config class : {config_class_name}\n"
            f"kwargs       : {model_kwargs}\n"
            f"error        : {e}"
        )


    try:

        return model_cls(
            model_config
        )

    except TypeError as e:

        fail(
            "Could not construct the model from its configuration.\n"
            f"model class  : {model_class}\n"
            f"config class : {config_class_name}\n"
            f"error        : {e}"
        )


# =============================================================================
# Generic model output handling
# =============================================================================

def unpack_model_output(output):

    """
    Convert different reasonable model return formats into:

        logits, loss

    Preferred contract:

        (logits, loss)

    Supported alternatives:

        logits
        {"logits": ..., "loss": ...}
        object.logits / object.loss
    """

    logits = None
    loss = None


    if isinstance(output, tuple):

        require(
            len(output) >= 1,
            "Model returned an empty tuple."
        )

        logits = output[0]

        if len(output) >= 2:
            loss = output[1]


    elif isinstance(output, dict):

        logits = output.get("logits")
        loss = output.get("loss")


    else:

        if hasattr(output, "logits"):
            logits = output.logits

        else:
            logits = output


        if hasattr(output, "loss"):
            loss = output.loss


    require(
        logits is not None,
        (
            "Model forward pass did not return logits."
        )
    )


    return logits, loss


def calculate_loss(logits, targets):

    """
    Generic next-token cross entropy.

    This is only used when the model does not provide its own loss.
    """

    require(
        torch.is_tensor(logits),
        "Model logits must be a torch.Tensor."
    )


    require(
        logits.ndim == 3,
        (
            "Generic loss calculation expects logits "
            "with shape [B, T, V]. "
            f"Received shape: {tuple(logits.shape)}"
        )
    )


    require(
        targets.ndim == 2,
        (
            "Targets must have shape [B, T]. "
            f"Received shape: {tuple(targets.shape)}"
        )
    )


    require(
        logits.shape[0] == targets.shape[0]
        and logits.shape[1] == targets.shape[1],
        (
            "Logits and targets have incompatible "
            f"sequence shapes: "
            f"logits={tuple(logits.shape)}, "
            f"targets={tuple(targets.shape)}"
        )
    )


    return torch.nn.functional.cross_entropy(
        logits.reshape(
            -1,
            logits.shape[-1],
        ),
        targets.reshape(-1),
    )


def forward_model(X, Y):

    """
    Universal model forward.

    Preferred:

        model(X, Y)

    Fallback:

        model(X)

    If the model does not provide a loss, the trainer computes
    standard next-token cross entropy from the returned logits.
    """

    try:

        output = model(
            X,
            Y,
        )

    except TypeError:

        output = model(
            X
        )


    logits, loss = unpack_model_output(
        output
    )


    if loss is None:

        loss = calculate_loss(
            logits,
            Y,
        )


    require(
        torch.is_tensor(loss),
        "Model loss must be a torch.Tensor."
    )


    require(
        loss.ndim == 0,
        (
            "Model loss must be scalar. "
            f"Received shape: {tuple(loss.shape)}"
        )
    )


    return logits, loss


# =============================================================================
# Build model
# =============================================================================

iter_num = 0
best_val_loss = float("inf")

checkpoint = None


if init_from == "scratch":

    print("\nInitializing model from scratch.")

    model = build_model()


elif init_from == "resume":

    ckpt_path = os.path.join(
        out_dir,
        "ckpt.pt",
    )


    require(
        os.path.isfile(ckpt_path),
        (
            "Cannot resume: checkpoint does not exist:\n"
            f"{ckpt_path}"
        )
    )


    print(
        f"\nResuming from: {ckpt_path}"
    )


    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )


    require(
        "model" in checkpoint,
        "Checkpoint does not contain model state."
    )


    model = build_model()


    state_dict = checkpoint["model"]


    # Remove torch.compile prefix if present.

    compiled_prefix = "_orig_mod."


    if any(
        key.startswith(compiled_prefix)
        for key in state_dict
    ):

        state_dict = {
            (
                key[len(compiled_prefix):]
                if key.startswith(compiled_prefix)
                else key
            ):
            value

            for key, value in state_dict.items()
        }


    try:

        model.load_state_dict(
            state_dict
        )

    except RuntimeError as e:

        fail(
            "Checkpoint/model state_dict mismatch.\n"
            f"{e}"
        )


    iter_num = int(
        checkpoint.get(
            "iter_num",
            0,
        )
    )


    best_val_loss = float(
        checkpoint.get(
            "best_val_loss",
            float("inf"),
        )
    )


# =============================================================================
# Model validation
# =============================================================================

require(
    isinstance(model, torch.nn.Module),
    (
        "Configured model must be an instance of "
        "torch.nn.Module."
    )
)


# =============================================================================
# Move model to device
# =============================================================================

model.to(device)


# =============================================================================
# Parameter count
# =============================================================================

num_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)


print(
    f"number of parameters: "
    f"{num_params:,} "
    f"({num_params / 1e6:.2f}M)"
)


# =============================================================================
# Optimizer
# =============================================================================

def build_optimizer():

    # If the model provides its own optimizer factory,
    # use it without knowing anything about the architecture.

    configure = getattr(
        model,
        "configure_optimizers",
        None,
    )


    if callable(configure):

        try:

            return configure(
                weight_decay,
                learning_rate,
                (beta1, beta2),
                device_type,
            )

        except TypeError:

            # Some models may expose a simpler signature.
            try:

                return configure(
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    betas=(beta1, beta2),
                )

            except TypeError as e:

                fail(
                    "Model.configure_optimizers() exists "
                    "but could not be called with the supported "
                    "generic signatures.\n"
                    f"error: {e}"
                )


    # Generic fallback.

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )


optimizer = build_optimizer()


if (
    checkpoint is not None
    and "optimizer" in checkpoint
):

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )


    for state in optimizer.state.values():

        for key, value in state.items():

            if torch.is_tensor(value):

                state[key] = value.to(
                    device
                )


checkpoint = None


# =============================================================================
# torch.compile
# =============================================================================

if compile:

    require(
        hasattr(torch, "compile"),
        (
            "compile=True but this PyTorch installation "
            "does not provide torch.compile."
        )
    )


    print(
        f"compiling model with mode='{compile_mode}'..."
    )


    model = torch.compile(
        model,
        mode=compile_mode,
    )


# =============================================================================
# Raw model helper
# =============================================================================

def get_raw_model():

    model_to_save = model


    if hasattr(
        model_to_save,
        "_orig_mod",
    ):

        model_to_save = (
            model_to_save._orig_mod
        )


    return model_to_save


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def estimate_loss():

    model.eval()


    results = {}


    for split in (
        "train",
        "val",
    ):

        losses = torch.empty(
            eval_iters,
            dtype=torch.float32,
        )


        for k in range(
            eval_iters
        ):

            X, Y = get_batch(
                split
            )


            with autocast_context:

                _, loss = forward_model(
                    X,
                    Y,
                )


            losses[k] = (
                loss
                .detach()
                .float()
                .cpu()
            )


        results[split] = (
            losses.mean().item()
        )


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
        1.0
        + math.cos(
            math.pi
            * decay_ratio
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
# Checkpoint
# =============================================================================

def save_checkpoint(val_loss):

    raw_model = get_raw_model()


    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": {
            "module": model_module,
            "class": model_class,
            "config_class": config.get(
                "model_config_class",
                None,
            ),
            "kwargs": model_kwargs,
        },
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config,
    }


    path = os.path.join(
        out_dir,
        "ckpt.pt",
    )


    print(
        f"saving checkpoint to {path}"
    )


    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# Output directory
# =============================================================================

os.makedirs(
    out_dir,
    exist_ok=True,
)


# =============================================================================
# Initial batch
# =============================================================================

X, Y = get_batch(
    "train"
)


# =============================================================================
# Training loop
# =============================================================================

t0 = time.time()

local_iter_num = 0


while True:

    # -------------------------------------------------------------------------
    # Learning rate
    # -------------------------------------------------------------------------

    lr = get_lr(
        iter_num
    )


    for param_group in optimizer.param_groups:

        param_group["lr"] = lr


    # -------------------------------------------------------------------------
    # Evaluation
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
            })


        improved = (
            val_loss
            < best_val_loss
        )


        if improved:

            best_val_loss = val_loss


        if always_save_checkpoint or improved:

            if iter_num > 0:

                save_checkpoint(
                    val_loss
                )


    # -------------------------------------------------------------------------
    # Evaluation-only mode
    # -------------------------------------------------------------------------

    if (
        iter_num == 0
        and eval_only
    ):

        break


    # -------------------------------------------------------------------------
    # Gradient accumulation
    # -------------------------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    last_loss = None


    for micro_step in range(
        gradient_accumulation_steps
    ):

        with autocast_context:

            _, loss = forward_model(
                X,
                Y,
            )


            loss_for_backward = (
                loss
                / gradient_accumulation_steps
            )


        last_loss = loss.detach()


        # Fetch next batch.

        X, Y = get_batch(
            "train"
        )


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

            scaler.unscale_(
                optimizer
            )


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )


    # -------------------------------------------------------------------------
    # Optimizer update
    # -------------------------------------------------------------------------

    if use_grad_scaler:

        scaler.step(
            optimizer
        )

        scaler.update()

    else:

        optimizer.step()


    # -------------------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------------------

    t1 = time.time()


    dt = t1 - t0

    t0 = t1


    if iter_num % log_interval == 0:

        loss_value = (
            last_loss
            .float()
            .item()
        )


        tok_per_sec = (
            tokens_per_iter / dt
            if dt > 0
            else 0.0
        )


        print(
            f"iter {iter_num}: "
            f"loss {loss_value:.4f}, "
            f"lr {lr:.6g}, "
            f"time {dt * 1000:.2f}ms, "
            f"tok/s {tok_per_sec:,.0f}"
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

print(
    "\nTraining finished."
)


if wandb_log:

    wandb.finish()
