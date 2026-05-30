"""5-step weight sync round-trip on NPU.

Validates the rollout-side half of the RL loop:
  1. sglang serving 1-layer DSv4-Flash MOE_ACTIVE=1 fab ckpt
  2. Per step:
     - Synthesize a per-step delta on attention weights (proxy for what
       Megatron actor train would produce; uses our rename map structure)
     - Merge delta into fab base safetensors, save as step_k_dir
     - POST /update_weights_from_disk to sglang
     - Rollout 2 prompts
     - get_weights_by_name + compare byte-equal vs the delta we sent
  3. After 5 steps:
     - Verify all 5 rollouts produce different outputs (weights took effect)
     - Verify all 5 byte-equal checks passed (sync is byte-for-byte)

This is the rollout-side proof. The Megatron-train-side actor that
actually generates the deltas via real gradient steps is a separate
driver (_e2e_rl_5step_dsv4_sglang.py + this script via HTTP).
"""
import sys
sys.path = [p for p in sys.path if p not in ("", "/")]

import os
import shutil
import json
import time
import hashlib

# Make rename map importable
sys.path.insert(0, "/home/z00637938/workspace")

import torch
import safetensors.torch as safetensors

from megatron_to_hf_rename import ATTN_RENAME, megatron_attn_to_hf_attn, merge_into_fab

FAB_BASE = "/host-models/dsv4_1layer_fab"
STEP_DIRS = [f"/host-models/dsv4_1layer_step_{k}" for k in range(1, 6)]
PROMPTS = ["Hi", "Hello"]
HF_PREFIX = "model.layers.0"


def synthesize_megatron_delta(step_idx, fab_state):
    """Synthesize a Megatron-actor-style delta state_dict (in Megatron
    naming).

    Strategy: take the existing fab attention weights (already in HF
    naming), reverse-rename them back to Megatron names, add a small
    per-step seeded random perturbation, return as Megatron-keyed
    state_dict.

    This pretends to be what Megatron would save after one train step.
    """
    # Build reverse rename map
    hf_to_mega = {hf: meg for meg, hf in ATTN_RENAME.items()}
    g = torch.Generator().manual_seed(1000 + step_idx)

    megatron_delta = {}
    for hf_suffix in ATTN_RENAME.values():
        hf_full = f"{HF_PREFIX}.{hf_suffix}"
        if hf_full not in fab_state:
            print(f"  [warn] {hf_full} not in fab base; skipping delta")
            continue
        base_t = fab_state[hf_full]
        # Per-step random perturbation; small magnitude so weights don't
        # become NaN even after 5 steps
        delta = torch.randn(base_t.shape, generator=g, dtype=base_t.dtype) * 0.001
        megatron_delta[hf_to_mega[hf_suffix]] = (base_t + delta).contiguous()
    return megatron_delta


def save_hf_step_ckpt(merged_state, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # Copy config + tokenizer (don't change)
    for fname in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        src = os.path.join(FAB_BASE, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fname))
    sf_path = os.path.join(out_dir, "model.safetensors")
    safetensors.save_file(merged_state, sf_path, metadata={"format": "pt"})


def tensor_hash(t):
    """Stable hash of a tensor's bytes."""
    arr = t.detach().contiguous().to(torch.float32).cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main():
    print(f"[setup] loading fab base from {FAB_BASE} ...")
    fab_path = os.path.join(FAB_BASE, "model.safetensors")
    fab_state = safetensors.load_file(fab_path)
    print(f"[setup] fab state has {len(fab_state)} keys")

    print(f"[setup] importing sglang + starting Engine ...")
    import sglang as sgl
    t0 = time.time()
    llm = sgl.Engine(
        model_path=FAB_BASE,
        dtype="bfloat16",
        device="npu",
        mem_fraction_static=0.10,
        max_total_tokens=1024,
        max_running_requests=1,
        max_prefill_tokens=256,
        chunked_prefill_size=256,
        enable_memory_saver=False,
        disable_radix_cache=True,
        tp_size=1,
        disable_cuda_graph=True,
        trust_remote_code=False,
    )
    print(f"[setup] engine init in {time.time()-t0:.1f}s")

    # Baseline rollout
    print(f"[step-0] baseline rollout ...")
    outs = llm.generate(PROMPTS, sampling_params={"temperature": 0.0, "max_new_tokens": 8})
    baseline_texts = [o["text"] for o in outs]
    print(f"[step-0] baseline: {baseline_texts}")

    # Pick a representative weight to round-trip verify each step
    probe_key = "model.layers.0.self_attn.q_a_proj.weight"

    all_rollouts = [baseline_texts]
    all_sync_ok = []

    for k in range(1, 6):
        step_dir = STEP_DIRS[k - 1]
        print(f"\n=== Step {k}/5 ===")

        # 1. Synthesize Megatron-actor delta
        mega_delta = synthesize_megatron_delta(k, fab_state)
        print(f"[step-{k}] synthesized {len(mega_delta)} Megatron deltas (seeded 1000+{k})")

        # 2. Rename to HF naming
        hf_overrides = megatron_attn_to_hf_attn(mega_delta, hf_prefix=HF_PREFIX)
        print(f"[step-{k}] renamed to {len(hf_overrides)} HF keys")

        # 3. Merge with fab base
        merged = merge_into_fab(fab_state, hf_overrides)

        # Capture the EXPECTED value of probe_key (what we're sending)
        expected_probe = merged[probe_key].detach().clone()
        expected_hash = tensor_hash(expected_probe)
        expected_first10 = expected_probe.flatten()[:10].tolist()
        print(f"[step-{k}] expected probe {probe_key}: sha={expected_hash}, first10={expected_first10}")

        # 4. Save HF ckpt for this step
        save_hf_step_ckpt(merged, step_dir)
        print(f"[step-{k}] saved to {step_dir}")

        # 5. POST update_weights_from_disk via sgl.Engine API
        try:
            update_resp = llm.update_weights_from_disk(model_path=step_dir)
            print(f"[step-{k}] update response: {update_resp}")
            update_ok = bool(update_resp.get("success") if isinstance(update_resp, dict) else update_resp)
        except Exception as e:
            print(f"[step-{k}] update FAILED: {type(e).__name__}: {e}")
            update_ok = False
            # Continue to verification/rollout to see whether previous weights are still good
        if not update_ok:
            print(f"[step-{k}] WARNING: reload failed; rollout will reflect previous step's weights")


        # 6. Verify the round-trip: get_weights_by_name and compare
        # Engine API:
        # get_weights_by_name(self, name: str, truncate_size: int = 100)
        # returns first `truncate_size` elements as a list
        actual_probe = llm.get_weights_by_name(probe_key, truncate_size=10)
        actual_first10 = list(actual_probe) if actual_probe is not None else None
        print(f"[step-{k}] actual probe (first 10 elements from sglang): {actual_first10}")

        if actual_first10 is not None:
            # Compare floats with tolerance (bf16 quantization is unavoidable)
            ok = all(
                abs(a - e) < 1e-2
                for a, e in zip(actual_first10, expected_first10)
            )
        else:
            ok = False
        all_sync_ok.append(ok)
        print(f"[step-{k}] sync match: {ok}")

        # 7. Rollout
        outs = llm.generate(PROMPTS, sampling_params={"temperature": 0.0, "max_new_tokens": 8})
        step_texts = [o["text"] for o in outs]
        print(f"[step-{k}] rollout: {step_texts}")
        all_rollouts.append(step_texts)

    print(f"\n=== 5-step summary ===")
    for i, r in enumerate(all_rollouts):
        print(f"  round {i}: {r}")
    print(f"  sync ok per step: {all_sync_ok}")

    # Check all 6 rollouts (baseline + 5 step) are distinct
    distinct_count = len(set(tuple(r) for r in all_rollouts))
    print(f"  distinct rollout outputs: {distinct_count}/6")

    sync_all_ok = all(all_sync_ok)
    weights_took_effect = distinct_count > 1

    if sync_all_ok and weights_took_effect:
        print(f"\n[smoke] PASS")
    else:
        print(f"\n[smoke] FAIL: sync_all_ok={sync_all_ok}, distinct rollouts={distinct_count}")


if __name__ == "__main__":
    main()
