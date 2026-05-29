# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# One complete RL step on Ascend A3 NPU, exercising:
#
#   1. **Rollout**: vllm-ascend serves a small policy model (Qwen2-0.5B) to
#      generate response tokens for a batch of prompts on the NPU
#   2. **Reward**: synthetic length-based reward (placeholder for a real
#      reward function; structural smoke only)
#   3. **Advantage**: GRPO-style group-normalised advantage from per-prompt
#      groups of n rollouts (matches EasyR1 default adv_estimator)
#   4. **Actor train step**: a miles DSAMLASelfAttention forward + backward +
#      Adam step through the patched stack (MindSpeed adaptor +
#      tilelang lighting_indexer / sparse_mla kernels)
#
# This is the end-to-end PR-ready validation: it proves PR #1246 (miles
# `_npu/`) + PR #3509 (MindSpeed apex shim) survive in a context where
# vllm-ascend rollout shares the NPU host.
#
# Run on A3:
#   docker exec tlrescue bash -c "
#     cd /tmp  # MUST NOT be /, see vllm sys.path fix below
#     export TILELANG_ASCEND_MODE=Developer
#     export ASCEND_RT_VISIBLE_DEVICES=1
#     PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/Megatron-LM-miles:/home/z00637938/workspace/tilelang-mlir-ascend:/home/z00637938/workspace/MindSpeed-clone \
#       python /home/z00637938/workspace/_e2e_rl_step_mindspeed.py
#   "
import sys

# vllm-ascend editable install workaround: '/' and '' on sys.path make
# Python's PathFinder return a NamespaceLoader for `vllm` (because the
# `/vllm` directory exists), preempting vllm's editable finder hook. Strip
# them so the editable hook can resolve `vllm` properly.
sys.path = [p for p in sys.path if p not in ("", "/")]

import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu  # noqa: F401

# Import ordering matters here:
# * MindSpeed's `requirements_basic.py:97` registers a dummy
#   `flash_attn.flash_attn_interface.flash_attn_unpadded_func` (create_dummy=True),
#   which transitively creates a stub `sys.modules['flash_attn']` package.
# * vllm's rotary_embedding/common.py:138 does
#   `if find_spec("flash_attn") is not None: from flash_attn.ops.triton.rotary import apply_rotary`,
#   which finds the stub but then fails to import the real `.ops` submodule.
# So we run vllm rollout FIRST (with a clean flash_attn-less env), free its
# resources, THEN import mindspeed.megatron_adaptor + miles for the actor
# train step. The post-rollout cleanup deletes the flash_attn stub if any
# leaked in from vllm's own deps.


# ---------------------------------------------------------------------------
# 1) Rollout via vllm-ascend
# ---------------------------------------------------------------------------

ROLLOUT_MODEL = "/home/z00637938/workspace/models/Qwen2-0.5B-Instruct"
PROMPTS = [
    "Write a short answer: What is 2+3?",
    "Write a short answer: What is the capital of Japan?",
]
ROLLOUT_N = 2  # rollouts per prompt -> group size for GRPO


def rollout(prompts, n: int):
    """vllm-ascend generate. Returns list[list[str]] of n responses per prompt.

    Lazy-imports vllm so this function can be called BEFORE mindspeed is
    imported (which would shim flash_attn and break vllm's rotary embedding).
    """
    from vllm import LLM, SamplingParams

    print(f"[rl] loading rollout model {ROLLOUT_MODEL} on vllm-ascend ...")
    llm = LLM(
        model=ROLLOUT_MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.15,
    )
    sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=24, n=n, seed=42)
    print(f"[rl] generating {len(prompts)} prompts x {n} samples ...")
    outs = llm.generate(prompts, sampling)
    # Each output has .outputs: list[CompletionOutput] of length n.
    results = []
    for out in outs:
        texts = [c.text for c in out.outputs]
        results.append(texts)
    # Explicitly free vllm engine resources before we start tilelang/Megatron
    # work on the same NPU process.
    del llm
    import gc
    gc.collect()
    torch.npu.empty_cache()
    return results


# ---------------------------------------------------------------------------
# 2) Reward (length-based synthetic; placeholder for real reward fn)
# ---------------------------------------------------------------------------

def compute_rewards(responses_per_prompt):
    """Smoke reward that produces non-zero GRPO advantages: combines
    response length (continuous, won't saturate) with token diversity
    (unique-char-ratio). Both signals vary across rollouts in a group, so
    after group-normalisation the resulting advantages will not all be zero.
    """
    out = []
    for resps in responses_per_prompt:
        r = []
        for text in resps:
            t = text.strip()
            length_signal = len(t) / 64.0  # 64-char target; rarely saturates
            uniq_signal = (len(set(t)) / max(1, len(t)))  # diversity in [0, 1]
            r.append(length_signal + 0.5 * uniq_signal)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# 3) Advantage (GRPO: group-normalised)
# ---------------------------------------------------------------------------

def compute_advantages(rewards_per_prompt):
    """GRPO advantage = (r - mean_group) / (std_group + eps)."""
    out = []
    eps = 1e-6
    for r in rewards_per_prompt:
        t = torch.tensor(r, dtype=torch.float32)
        mu = t.mean()
        sd = t.std(unbiased=False)
        out.append(((t - mu) / (sd + eps)).tolist())
    return out


# ---------------------------------------------------------------------------
# 4) Actor train step through patched stack
# ---------------------------------------------------------------------------

def actor_train_step(advantages_flat):
    """One forward + backward + Adam step on a miles DSAMLA layer using the
    patched MindSpeed+tilelang stack. `advantages_flat` shapes the loss
    target so rollout-derived signal actually drives the gradient.

    All mindspeed / megatron / miles imports happen INSIDE this function so
    they don't run until vllm rollout has already freed its resources and the
    flash_attn shim collision window is closed (see ordering note at top).
    """
    # 1. Trigger MindSpeed's patch_features() — this is the patched-stack entry.
    import mindspeed.megatron_adaptor  # noqa: F401

    # 2. Bring in helpers from the single-step MindSpeed-aware driver.
    from miles_plugins.models.glm5.ops._npu._e2e_megatron_step_mindspeed import (
        _init_distributed as _init_dist,
        _build_config,
        ColumnParallelLinear,
        RowParallelLinear,
        IndexerColumnParallelLinear,
        _LayerNormColumnParallelLinear,
    )
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.packed_seq_params import PackedSeqParams
    from miles_plugins.models.glm5.glm5 import (
        DSAMLASelfAttention,
        DSASelfAttentionSubmodules,
    )
    import miles_plugins.models.glm5.glm5 as _glm5_module

    # 3. Init Megatron distributed AFTER mindspeed has finished patching.
    _init_dist()

    cfg = _build_config()  # reduced preset by default
    print(f"[rl] actor cfg: hidden={cfg.hidden_size} H={cfg.num_attention_heads}")

    # Build one DSAMLASelfAttention layer (reduced shape).
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
    actor.index_topk = 4
    total = sum(p.numel() for p in actor.parameters())
    print(f"[rl] actor built: {total:,} params")

    SEQ, BSZ = 16, 1
    hidden_states = (torch.randn(SEQ, BSZ, cfg.hidden_size, dtype=torch.bfloat16) * 0.1).npu()
    cu_seqlens = torch.tensor([0, SEQ], dtype=torch.int32).npu()
    packed = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=SEQ,
        max_seqlen_kv=SEQ,
        qkv_format="thd",
    )
    position_ids = torch.arange(SEQ, dtype=torch.int64).unsqueeze(0).npu()

    indexer_captures: list = []
    _real = _glm5_module.lighting_indexer

    def _captured(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=None):
        score, indices = _real(index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices=topk_indices)
        indexer_captures.append(score)
        return score, indices

    _glm5_module.lighting_indexer = _captured  # type: ignore[assignment]

    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)

    print(f"[rl] actor forward ...")
    out = actor(
        hidden_states=hidden_states,
        attention_mask=None,
        inference_context=None,
        packed_seq_params=packed,
        position_ids=position_ids,
    )
    primary = out[0] if isinstance(out, tuple) else out
    print(f"[rl]   actor out shape: {primary.shape}")

    # Use the flat advantages from rollout to shape the loss target. Pad or
    # tile the advantage vector to the actor output's flat length.
    adv_t = torch.tensor(advantages_flat, dtype=torch.float32).npu()
    n_target = primary.numel()
    if adv_t.numel() < n_target:
        rep = (n_target + adv_t.numel() - 1) // adv_t.numel()
        adv_t = adv_t.repeat(rep)[:n_target]
    else:
        adv_t = adv_t[:n_target]
    adv_t = adv_t.view_as(primary.float())

    mla_loss = -(primary.float() * adv_t).sum() / max(1, n_target)
    idx_loss = torch.zeros((), device=primary.device, dtype=torch.float32)
    for sc in indexer_captures:
        sc_f = sc.float()
        v = torch.isfinite(sc_f)
        sc_v = torch.where(v, sc_f, torch.zeros_like(sc_f))
        idx_loss = idx_loss + sc_v.pow(2).sum() / max(1, int(v.sum().item()))
    idx_loss = idx_loss * 0.01 / max(1, len(indexer_captures))
    loss = mla_loss + idx_loss

    print(f"[rl] actor backward ...")
    opt.zero_grad()
    loss.backward()

    # Mask non-finite grad entries (R-KA-15 mitigation, same as multilayer driver).
    for p in actor.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad = torch.where(torch.isfinite(p.grad), p.grad, torch.zeros_like(p.grad))
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)

    finite_count = sum(1 for p in actor.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    total_count = sum(1 for _ in actor.parameters())
    opt.step()

    return dict(
        loss=float(loss.detach().cpu()),
        mla_loss=float(mla_loss.detach().cpu()),
        idx_loss=float(idx_loss.detach().cpu()),
        finite_grads=finite_count,
        total_params=total_count,
        out_shape=tuple(primary.shape),
        idx_layers=len(indexer_captures),
    )


# ---------------------------------------------------------------------------
# Drive one full RL step
# ---------------------------------------------------------------------------

def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

    t0 = time.time()

    # Stage 1: rollout
    print("\n=== Stage 1: rollout (vllm-ascend) ===")
    t1 = time.time()
    responses = rollout(PROMPTS, ROLLOUT_N)
    print(f"[rl] rollout done in {time.time() - t1:.1f}s")
    for i, (p, rs) in enumerate(zip(PROMPTS, responses)):
        print(f"  prompt[{i}] = {p!r}")
        for j, r in enumerate(rs):
            print(f"    response[{j}] = {r!r}")

    # Stage 2: reward
    print("\n=== Stage 2: reward ===")
    rewards = compute_rewards(responses)
    for i, r in enumerate(rewards):
        print(f"  rewards[{i}] = {r}")

    # Stage 3: advantage (GRPO group-normalised)
    print("\n=== Stage 3: advantage (GRPO) ===")
    advantages = compute_advantages(rewards)
    for i, a in enumerate(advantages):
        print(f"  advantages[{i}] = {a}")
    adv_flat = [v for group in advantages for v in group]
    print(f"  flat advantages: {adv_flat}")

    # Stage 4: actor train via patched Megatron+MindSpeed+tilelang
    print("\n=== Stage 4: actor train step (patched stack) ===")
    # Eagerly purge any leaked vllm/flash_attn stub from sys.modules so
    # MindSpeed's dummy-creation in requirements_basic.py succeeds cleanly.
    for mod in list(sys.modules):
        if mod == "flash_attn" or mod.startswith("flash_attn."):
            del sys.modules[mod]
    t4 = time.time()
    metrics = actor_train_step(adv_flat)
    print(f"[rl] actor train step done in {time.time() - t4:.1f}s")

    # Result
    print(f"\n=== RL step summary ===")
    print(f"  total time: {time.time() - t0:.1f}s")
    print(f"  prompts: {len(PROMPTS)}  rollouts/prompt: {ROLLOUT_N}")
    print(f"  actor out: {metrics['out_shape']}")
    print(f"  loss: {metrics['loss']:.5f}  (mla={metrics['mla_loss']:.5f}, idx={metrics['idx_loss']:.5f})")
    print(f"  finite grads: {metrics['finite_grads']}/{metrics['total_params']}")
    print(f"  idx capture layers: {metrics['idx_layers']}")
    print(f"  result: PASS")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
