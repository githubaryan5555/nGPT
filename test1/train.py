"""
Single-GPU optimized training script (nanoGPT style).
All settings live in config.py — edit that file, then:

$ python train.py
"""

import os
# must be set BEFORE torch import: reduces allocator fragmentation / OOMs
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

import config as C
from model import GPTConfig, GPT

# config snapshot for wandb + checkpoint metadata
config = {k: getattr(C, k) for k in dir(C)
          if not k.startswith('_') and isinstance(getattr(C, k), (int, float, bool, str))}

# -----------------------------------------------------------------------------
# dtype resolution ('auto' fixes the old dead-bf16-branch bug)
# -----------------------------------------------------------------------------
if C.dtype not in ('auto', 'float32', 'bfloat16', 'float16'):
    raise ValueError(f"unsupported dtype '{C.dtype}'")
dtype = C.dtype
if dtype == 'auto':
    dtype = 'bfloat16' if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else 'float16'
print(f"using dtype: {dtype}")

device = C.device
device_type = 'cuda' if 'cuda' in device else 'cpu'
if device_type == 'cuda':
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

tokens_per_iter = C.gradient_accumulation_steps * C.batch_size * C.block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(C.out_dir, exist_ok=True)
torch.manual_seed(C.seed)

# -----------------------------------------------------------------------------
# data: memmaps opened ONCE. Bins expected in datasets/<dataset>/ or datasets/.
# -----------------------------------------------------------------------------
candidates = [os.path.join(C.data_root, C.dataset), C.data_root]
data_dir = next((d for d in candidates if os.path.exists(os.path.join(d, 'train.bin'))), None)
assert data_dir is not None, f"could not find train.bin under '{C.data_root}'"
print(f"data dir: {data_dir}")

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# optional: preload all tokens into VRAM (fastest loader, zero H2D per batch)
use_gpu_data = False
if C.preload_data_to_gpu and device_type == 'cuda':
    need = int(train_data.nbytes + val_data.nbytes)
    free, _total = torch.cuda.mem_get_info(device)
    if need < 0.75 * free:  # leave headroom for activations/optimizer
        print(f"preloading dataset to GPU ({need / 1e9:.2f} GB) ...")
        train_gpu = torch.from_numpy(np.ascontiguousarray(train_data)).to(device)
        val_gpu = torch.from_numpy(np.ascontiguousarray(val_data)).to(device)
        use_gpu_data = True
    else:
        print(f"dataset needs {need / 1e9:.2f} GB; too large, using memmap loader")

_offsets = np.arange(C.block_size + 1, dtype=np.int64)
_rng = np.random.default_rng(C.seed)

def get_batch_cpu(split):
    data = train_data if split == 'train' else val_data
    ix = _rng.integers(0, len(data) - C.block_size - 1, size=C.batch_size)
    # single vectorized read of block_size+1 tokens; slice into x / y
    seq = torch.from_numpy(data[ix[:, None] + _offsets[None, :]].astype(np.int32))
    t = seq.pin_memory().to(device, non_blocking=True)
    x = t[:, :-1]        # int32 indices are valid for nn.Embedding
    y = t[:, 1:].long()  # cross_entropy targets MUST be int64
    return x, y

if use_gpu_data:
    _gpu_rng = torch.Generator(device=device).manual_seed(C.seed)
    _offsets_t = torch.arange(C.block_size + 1, device=device)
    def get_batch_gpu(split):
        data = train_gpu if split == 'train' else val_gpu
        ix = torch.randint(len(data) - C.block_size - 1, (C.batch_size,),
                           device=device, generator=_gpu_rng)
        seq = data[ix[:, None] + _offsets_t].int()
        return seq[:, :-1], seq[:, 1:].long()
    get_batch = get_batch_gpu
else:
    get_batch = get_batch_cpu

# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=C.n_layer, n_head=C.n_head, n_embd=C.n_embd,
                  block_size=C.block_size, bias=C.bias, vocab_size=None,
                  dropout=C.dropout)
if C.init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif C.init_from == 'resume':
    print(f"Resuming training from {C.out_dir}")
    ckpt_path = os.path.join(C.out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these to match the checkpoint, the rest (e.g. dropout) come from config
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif C.init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {C.init_from}")
    override_args = dict(dropout=C.dropout)
    model = GPT.from_pretrained(C.init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
else:
    raise ValueError(f"unknown init_from '{C.init_from}'")

# crop down the model block size if desired, using model surgery
if C.block_size < model.config.block_size:
    model.crop_block_size(C.block_size)
    model_args['block_size'] = C.block_size
model.to(device)

# GradScaler is a no-op unless dtype == 'float16'
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

# optimizer: reuse configure_optimizers for decay/no-decay grouping,
# then swap in fused AdamW (one fused CUDA kernel per step)
optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate,
                                       (C.beta1, C.beta2), device_type)
if device_type == 'cuda':
    fused_optimizer = torch.optim.AdamW(optimizer.param_groups, lr=C.learning_rate,
                                        betas=(C.beta1, C.beta2), eps=1e-8, fused=True)
    if C.init_from == 'resume':
        fused_optimizer.load_state_dict(checkpoint['optimizer'])
    optimizer = fused_optimizer
elif C.init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if C.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model, dynamic=False,
                          mode=(None if C.compile_mode == 'default' else C.compile_mode))

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        # accumulate on GPU: one .item() sync per split, not per batch
        losses = torch.zeros(C.eval_iters, device=device)
        for k in range(C.eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.detach()
        out[split] = losses.mean().item()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < C.warmup_iters:  # 1) linear warmup
        return C.learning_rate * (it + 1) / (C.warmup_iters + 1)
    if it > C.lr_decay_iters:  # 2) after decay, hold at min_lr
        return C.min_lr
    # 3) cosine decay from max to min
    decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return C.min_lr + coeff * (C.learning_rate - C.min_lr)

# logging
if C.wandb_log:
    import wandb
    wandb.init(project=C.wandb_project, name=C.wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')  # fetch the very first batch
t0 = time.time()
local_iter_num = 0
# unwrap torch.compile so checkpoints get clean keys and mfu works
raw_model = getattr(model, '_orig_mod', model)
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % C.eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if C.wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })
        if losses['val'] < best_val_loss or C.always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {C.out_dir}")
                torch.save(checkpoint, os.path.join(C.out_dir, 'ckpt.pt'))
    if iter_num == 0 and C.eval_only:
        break

    # forward backward update, with gradient accumulation to simulate larger batch size
    for micro_step in range(C.gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / C.gradient_accumulation_steps
        # async prefetch next batch; CPU gather + H2D overlap with backward
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    # clip the gradient
    if C.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
    # step the optimizer and scaler
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % C.log_interval == 0:
        lossf = loss.item() * C.gradient_accumulation_steps
        if local_iter_num >= 5:  # let the loop settle before measuring mfu
            mfu = raw_model.estimate_mfu(C.batch_size * C.gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > C.max_iters:
        break
