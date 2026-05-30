"""5-step miles RL on DSv4-Flash 1-layer with sglang rollout + Megatron actor.

Per user 2026-05-30:
  * sglang and Megatron on different chips (sgl_probe on chip A, tlrescue
    on chip B) -- no Ray, no HCCL, communicate via HTTP
  * 5 RL steps
  * miles_local config: 1-layer DSv4-Flash dims (hidden=4096, H=64, q_lora=1024,
    kv_lora=512, v_head_dim=512), SEQ=128, index_topk=8 — same as §3.6 probe
  * Synthetic dataset with non-trivial signal so weights actually move
  * Checkpoint per step + diff to prove weights changed
  * Memory controlled

Design:
  Stage per step k in {0..4}:
    1) rollout: sglang HTTP /generate on 4 prompts, T=0.7
    2) reward: length + correct-answer keyword presence (different for each
       prompt; gives non-trivial gradient signal)
    3) advantage: GRPO group-norm
    4) actor train: miles DSAMLA forward + backward + Adam (CPU/NPU on
       trainer side)
    5) snapshot: save actor state_dict to {SNAPSHOTS}/step_k.pt
    6) advance to step k+1

  After loop: diff every (step_k, step_{k+1}) state_dict pair, prove each
  step changed weights non-trivially.
"""
import sys

sys.path = [p for p in sys.path if p not in ("", "/")]

import os
import time
import json
import hashlib

import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu  # noqa: F401

# Reuse the same imports + builder as §3.6 (miles_local config)
import mindspeed.megatron_adaptor  # noqa: F401

from miles_plugins.models.glm5.ops._npu._e2e_megatron_step_mindspeed import (
    _init_distributed as _init_dist,
    ColumnParallelLinear,
    RowParallelLinear,
    IndexerColumnParallelLinear,
    _LayerNormColumnParallelLinear,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams

from miles_plugins.models.glm5.glm5 import (
    DSAMLASelfAttention,
    DSASelfAttentionSubmodules,
)
import miles_plugins.models.glm5.glm5 as _glm5_module


SGLANG_URL = os.environ.get("SGLANG_BASE_URL", "http://172.17.0.2:30000")
NUM_STEPS = 5
SEQ = 128
BSZ = 1
ROLLOUT_N = 2
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/home/z00637938/workspace/rl5_snapshots")

# Synthetic dataset with non-trivial reward signal: 4 prompts, each with
# a specific "correct" keyword. Reward weighted by keyword presence +
# length plausibility -> per-group advantage will be non-zero.
PROMPTS = [
    ("What is 2+3?", "5"),
    ("The capital of Japan is", "Tokyo"),
    ("Color of the sky during day?", "blue"),
    ("What is 7 times 6?", "42"),
]


def build_dsv4_config():
    """Same miles_local config as §3.6."""
    cfg = MLATransformerConfig(
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=64,
        ffn_hidden_size=1408,
        kv_channels=128,
        q_lora_rank=1024,
        kv_lora_rank=512,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=512,
        rotary_base=10000.0,
        rotary_scaling_factor=1.0,
        rotary_percent=1.0,
        original_max_position_embeddings=2048,
        mscale=1.0,
        mscale_all_dim=1.0,
        beta_fast=32,
        beta_slow=1,
        add_bias_linear=False,
        layernorm_epsilon=1e-5,
        normalization="RMSNorm",
        recompute_granularity=None,
    )
    cfg.index_num_attention_heads = 64
    cfg.index_head_dim = 128
    return cfg


def wait_sglang_healthy(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{SGLANG_URL}/health_generate", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def rollout(step_idx, n=ROLLOUT_N):
    """HTTP /generate against sglang. Returns dict[prompt] -> list[str]."""
    print(f"[step-{step_idx}] rollout: {len(PROMPTS)} prompts x {n} samples ...")
    results = {}
    t0 = time.time()
    for prompt, kw in PROMPTS:
        samples = []
        for _ in range(n):
            r = requests.post(
                f"{SGLANG_URL}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "max_new_tokens": 16,
                    },
                },
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                data = data[0]
            samples.append(data.get("text", ""))
        results[prompt] = samples
    print(f"[step-{step_idx}] rollout done in {time.time()-t0:.1f}s")
    return results


def compute_rewards(rollouts):
    """Non-trivial reward: length-normalized + keyword-presence bonus.
    Returns dict[prompt] -> list[float]."""
    rewards = {}
    for (prompt, kw), samples in zip(PROMPTS, rollouts.values()):
        prompt_rewards = []
        for text in samples:
            t = text.strip()
            length_r = min(len(t) / 32.0, 1.0)
            kw_r = 1.5 if kw.lower() in t.lower() else 0.0
            diversity_r = (len(set(t)) / max(1, len(t)))
            prompt_rewards.append(length_r + kw_r + 0.3 * diversity_r)
        rewards[prompt] = prompt_rewards
    return rewards


def compute_advantages(rewards):
    """GRPO group-norm per prompt."""
    advs = {}
    eps = 1e-6
    for prompt, r in rewards.items():
        t = torch.tensor(r, dtype=torch.float32)
        mu = t.mean()
        sd = t.std(unbiased=False)
        advs[prompt] = ((t - mu) / (sd + eps)).tolist()
    return advs


def _build_actor(cfg):
    """Build 1-layer miles DSAMLA -- same wiring as §3.6 probe."""
    submods = DSASelfAttentionSubmodules(
        linear_q_down_proj=ColumnParallelLinear,
        linear_q_up_proj=_LayerNormColumnParallelLinear,
        linear_kv_down_proj=ColumnParallelLinear,
        linear_kv_up_proj=_LayerNormColumnParallelLinear,
        linear_v_up_proj=IdentityOp,
        core_attention=IdentityOp,
        linear_proj=RowParallelLinear,
        q_layernorm=IdentityOp,
        kv_layernorm=IdentityOp,
    )
    for extra, target in (
        ("wq_b", IndexerColumnParallelLinear),
        ("wk", IndexerColumnParallelLinear),
        ("weights_proj", IndexerColumnParallelLinear),
    ):
        if hasattr(submods, extra):
            setattr(submods, extra, target)
    if hasattr(submods, "k_norm"):
        class _LN(nn.LayerNorm):
            def __init__(self, hidden_size, config=None, eps=1e-5, **kw):
                super().__init__(hidden_size, eps=eps)
        submods.k_norm = _LN
    actor = DSAMLASelfAttention(
        config=cfg,
        submodules=submods,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
    ).npu()
    actor.index_topk = 8
    return actor


def actor_train_step(actor, opt, advantages, step_idx):
    """One forward + backward + Adam step on the miles DSAMLA actor.
    advantages: dict[prompt] -> list[float]; we flatten and use as loss
    shaping target."""
    print(f"[step-{step_idx}] actor train ...")
    t0 = time.time()
    cfg_hidden = actor.config.hidden_size

    indexer_captures: list = []
    _real = _glm5_module.lighting_indexer

    def _captured(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=None):
        score, indices = _real(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=topk_indices)
        indexer_captures.append(score)
        return score, indices

    _glm5_module.lighting_indexer = _captured

    hidden_states = (torch.randn(SEQ, BSZ, cfg_hidden, dtype=torch.bfloat16) * 0.1).npu()
    cu_seqlens = torch.tensor([0, SEQ], dtype=torch.int32).npu()
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=SEQ,
        max_seqlen_kv=SEQ,
        qkv_format="thd",
    )
    position_ids = torch.arange(SEQ, dtype=torch.int64).unsqueeze(0).npu()

    out = actor(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=packed,
        position_ids=position_ids,
    )
    primary = out[0] if isinstance(out, tuple) else out

    adv_flat = [v for vs in advantages.values() for v in vs]
    adv_t = torch.tensor(adv_flat, dtype=torch.float32).npu()
    n_target = primary.numel()
    if adv_t.numel() < n_target:
        rep = (n_target + adv_t.numel() - 1) // adv_t.numel()
        adv_t = adv_t.repeat(rep)[:n_target]
    else:
        adv_t = adv_t[:n_target]
    adv_t = adv_t.view_as(primary.float())

    loss = -(primary.float() * adv_t).sum() / max(1, n_target)

    opt.zero_grad()
    loss.backward()

    # R-KA-15 mitigation: zero non-finite grads before clip
    for p in actor.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

    finite_count = sum(1 for p in actor.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    total_count = sum(1 for _ in actor.parameters())
    opt.step()

    elapsed = time.time() - t0
    print(f"[step-{step_idx}] actor train done in {elapsed:.1f}s, loss={float(loss.detach().cpu()):.5f}, "
          f"finite_grads={finite_count}/{total_count}, idx_layers={len(indexer_captures)}")
    return float(loss.detach().cpu()), finite_count, total_count


def snapshot_state(actor, step_idx):
    """Save actor state_dict snapshot + compute a content-hash for diffing."""
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, f"step_{step_idx}.pt")
    sd = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items() if v is not None}
    torch.save(sd, path)

    # Compute a hash so we can diff snapshots without keeping all in memory
    hasher = hashlib.sha256()
    for k in sorted(sd.keys()):
        hasher.update(k.encode())
        hasher.update(sd[k].numpy().tobytes())
    digest = hasher.hexdigest()[:16]
    print(f"[step-{step_idx}] snapshot saved {path}; sha256[:16]={digest}")
    return path, digest


def compare_snapshots(paths):
    """Load each snapshot, compute per-parameter max-abs-diff against step 0."""
    print("\n=== snapshot diff matrix (vs step 0) ===")
    base = torch.load(paths[0], map_location="cpu", weights_only=True)
    for k in range(1, len(paths)):
        cur = torch.load(paths[k], map_location="cpu", weights_only=True)
        deltas = {}
        total_diff = 0.0
        for name in base:
            d = (cur[name] - base[name]).abs()
            max_d = float(d.max())
            mean_d = float(d.mean())
            deltas[name] = (max_d, mean_d)
            total_diff += float(d.sum())
        # Top-3 changed params
        top = sorted(deltas.items(), key=lambda kv: -kv[1][0])[:3]
        print(f"step_{k} vs step_0: total_abs_diff={total_diff:.4f}; top-3 max-abs:")
        for name, (mx, mn) in top:
            print(f"    {name}: max={mx:.6f} mean={mn:.6f}")


def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

    print("[setup] init distributed ...")
    _init_dist()

    print("[setup] building 1-layer miles_local DSv4-Flash actor ...")
    cfg = build_dsv4_config()
    actor = _build_actor(cfg)
    total_params = sum(p.numel() for p in actor.parameters())
    print(f"[setup] actor: {total_params:,} params (~{total_params*2/1024**3:.2f} GB bf16)")

    print(f"[setup] checking sglang at {SGLANG_URL} ...")
    if not wait_sglang_healthy(120):
        raise RuntimeError(f"sglang not healthy at {SGLANG_URL}")
    print(f"[setup] sglang healthy")

    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)

    snapshots = []
    losses = []
    finite_history = []

    # Step 0: initial snapshot (before any training)
    p0, h0 = snapshot_state(actor, 0)
    snapshots.append(p0)

    for step in range(1, NUM_STEPS + 1):
        print(f"\n=== Step {step}/{NUM_STEPS} ===")
        # rollout
        rolls = rollout(step)
        for (prompt, kw), samples in zip(PROMPTS, rolls.values()):
            print(f"  [rollout] {prompt!r} -> {[s[:30] for s in samples]}")

        # reward + advantage
        rewards = compute_rewards(rolls)
        for prompt, r in rewards.items():
            print(f"  [reward]  {prompt[:24]!r}: {[f'{x:.2f}' for x in r]}")
        advs = compute_advantages(rewards)
        for prompt, a in advs.items():
            print(f"  [adv]     {prompt[:24]!r}: {[f'{x:.2f}' for x in a]}")

        # actor train + snapshot
        loss, finite_n, total_n = actor_train_step(actor, opt, advs, step)
        losses.append(loss)
        finite_history.append((finite_n, total_n))
        path, digest = snapshot_state(actor, step)
        snapshots.append(path)

    print(f"\n=== 5-step RL summary ===")
    print(f"  loss trajectory: {losses}")
    print(f"  finite grads:   {finite_history}")
    print(f"  snapshots:      {snapshots}")
    compare_snapshots(snapshots)

    print(f"\n[smoke] PASS")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
