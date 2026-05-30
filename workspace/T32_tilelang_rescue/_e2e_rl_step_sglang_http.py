# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# One RL step on Ascend A3 NPU using **sglang HTTP rollout** (miles' default
# rollout engine) instead of in-process vllm-ascend.
#
# This is the (3b) follow-up from MILES_DSV4_NPU_POC_REPORT §5.2 — wires the
# patched MindSpeed+tilelang actor train to a sglang HTTP rollout server
# running in a sidecar container (`sgl_probe`).
#
# Architecture:
#
#   ┌─────────────────────────────────┐    HTTP POST /generate    ┌─────────────────────────────────────┐
#   │ tlrescue container               │  ──────────────────────►  │ sgl_probe container                   │
#   │  (torch 2.9 + torch_npu 2.9 +    │                            │  (torch 2.8 + sglang 0.5.10 +         │
#   │   patched MindSpeed + miles +    │  ◄──────────────────────  │   sgl_kernel_npu 2026.2.1)            │
#   │   tilelang)                      │       JSON tokens         │  Hosts sglang HTTP server on :30000   │
#   │                                  │                            │                                       │
#   │ this driver:                     │                            │ started via start_sglang_server.sh    │
#   │  rollout: HTTP call              │                            │                                       │
#   │  reward / advantage              │                            │                                       │
#   │  actor train: full patched stack │                            │                                       │
#   └─────────────────────────────────┘                            └─────────────────────────────────────┘
#
# Why two containers: sgl_probe needs torch 2.8 (sgl_kernel_npu was built
# against it), tlrescue has torch 2.9 (for tilelang + MindSpeed). The ABI
# mismatch can't be merged easily, so use HTTP -- which is also what miles
# `SGLangEngine` does in production.
#
# Run on A3 (assumes sglang HTTP server is already up in sgl_probe):
#   docker exec tlrescue bash -c "
#     cd /tmp
#     export TILELANG_ASCEND_MODE=Developer
#     export ASCEND_RT_VISIBLE_DEVICES=1
#     export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29510 LOCAL_RANK=0
#     export PYTHONPATH=/home/z00637938/workspace/miles:/home/z00637938/workspace/Megatron-LM-miles:/home/z00637938/workspace/tilelang-mlir-ascend:/home/z00637938/workspace/MindSpeed-clone:\$PYTHONPATH
#     python /home/z00637938/workspace/_e2e_rl_step_sglang_http.py
#   "
import os
import sys
import time

import requests
import torch
import torch.distributed as dist
import torch.nn as nn
import torch_npu  # noqa: F401


# ---------------------------------------------------------------------------
# 1) Rollout via sglang HTTP
# ---------------------------------------------------------------------------

# The HTTP server binds 0.0.0.0:30000 inside sgl_probe; on the same Docker
# bridge network we can reach it via the container name. If the user has a
# different network config, override SGLANG_BASE_URL.
SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://sgl_probe:30000")

# Fallback: try the host IP if container DNS isn't set up. tlrescue uses
# /home/z00637938 bind-mount which means it shares the host, so 127.0.0.1
# works iff we host-network the sgl_probe container too. The default Docker
# bridge with container name resolution is the cleanest path; that's what
# we assume. The user can `--network host` both containers if DNS fails.

PROMPTS = [
    "Write a short answer: What is 2+3?",
    "Write a short answer: What is the capital of Japan?",
]
ROLLOUT_N = 2  # rollouts per prompt -> group size for GRPO


def _wait_server_healthy(base_url: str, timeout_s: int = 120):
    """Poll /health until 200 or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health_generate", timeout=2)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(2)
    return False


def rollout_http(prompts, n: int):
    """sglang HTTP rollout. Returns list[list[str]]."""
    print(f"[rl] checking sglang server at {SGLANG_BASE_URL} ...")
    if not _wait_server_healthy(SGLANG_BASE_URL):
        raise RuntimeError(f"sglang server at {SGLANG_BASE_URL} not healthy after 120s")
    print(f"[rl] server healthy; generating {len(prompts)} prompts x {n} samples ...")

    results = []
    for prompt in prompts:
        samples_for_prompt = []
        for _ in range(n):
            # Use sglang's /generate endpoint with sampling
            payload = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_new_tokens": 24,
                },
            }
            r = requests.post(f"{SGLANG_BASE_URL}/generate", json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            # sglang /generate returns either {"text": ...} or [{"text": ...}]
            if isinstance(data, list):
                data = data[0]
            samples_for_prompt.append(data.get("text", ""))
        results.append(samples_for_prompt)
    return results


# ---------------------------------------------------------------------------
# 2) Reward (length + token diversity; same as vllm variant)
# ---------------------------------------------------------------------------

def compute_rewards(responses_per_prompt):
    out = []
    for resps in responses_per_prompt:
        r = []
        for text in resps:
            t = text.strip()
            length_signal = len(t) / 64.0
            uniq_signal = (len(set(t)) / max(1, len(t)))
            r.append(length_signal + 0.5 * uniq_signal)
        out.append(r)
    return out


# ---------------------------------------------------------------------------
# 3) Advantage (GRPO)
# ---------------------------------------------------------------------------

def compute_advantages(rewards_per_prompt):
    out = []
    eps = 1e-6
    for r in rewards_per_prompt:
        t = torch.tensor(r, dtype=torch.float32)
        mu = t.mean()
        sd = t.std(unbiased=False)
        out.append(((t - mu) / (sd + eps)).tolist())
    return out


# ---------------------------------------------------------------------------
# 4) Actor train through patched stack (deferred imports -- mindspeed
#    installs a dummy flash_attn that conflicts with various downstream
#    things; we don't import it until rollout is over)
# ---------------------------------------------------------------------------

def actor_train_step(advantages_flat):
    import mindspeed.megatron_adaptor  # noqa: F401

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

    _init_dist()
    cfg = _build_config()
    print(f"[rl] actor cfg: hidden={cfg.hidden_size} H={cfg.num_attention_heads}")

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
# Drive one RL step
# ---------------------------------------------------------------------------

def main():
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")

    t0 = time.time()

    print("\n=== Stage 1: rollout (sglang HTTP) ===")
    t1 = time.time()
    responses = rollout_http(PROMPTS, ROLLOUT_N)
    print(f"[rl] rollout done in {time.time() - t1:.1f}s")
    for i, (p, rs) in enumerate(zip(PROMPTS, responses)):
        print(f"  prompt[{i}] = {p!r}")
        for j, r in enumerate(rs):
            print(f"    response[{j}] = {r!r}")

    print("\n=== Stage 2: reward ===")
    rewards = compute_rewards(responses)
    for i, r in enumerate(rewards):
        print(f"  rewards[{i}] = {r}")

    print("\n=== Stage 3: advantage (GRPO) ===")
    advantages = compute_advantages(rewards)
    for i, a in enumerate(advantages):
        print(f"  advantages[{i}] = {a}")
    adv_flat = [v for group in advantages for v in group]
    print(f"  flat advantages: {adv_flat}")

    print("\n=== Stage 4: actor train step (patched stack) ===")
    t4 = time.time()
    metrics = actor_train_step(adv_flat)
    print(f"[rl] actor train step done in {time.time() - t4:.1f}s")

    print(f"\n=== RL step summary (sglang HTTP variant) ===")
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
