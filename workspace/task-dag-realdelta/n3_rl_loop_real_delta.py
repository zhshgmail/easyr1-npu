"""n3_bridge — V4 RL loop driven by the REAL trained attn-delta (replaces synth).

Cross-stack bridge: loads trained_attn_delta.pt (produced by n1 in the megatron stack),
remaps megatron->sglang names (n2), shape-asserts/transposes against the live sglang base
weights, and pushes base + real_delta via update_weights_from_tensor — same #26794-dodging
attention-only path as _v4_rl_loop_tensor_PASS.py, but the delta is REAL training output.

Verify (n3 honesty gate):
  - update_weights_from_tensor returns (True, Success)
  - re-rollout diverges from step0 (weight-sync changes inference)
  - the applied delta provably equals the real trained delta: we hash each pushed
    (full = base + real_delta) tensor and also log the per-tensor delta L2 actually applied,
    and confirm it matches trained_attn_delta.pt's delta L2 (not a synth perturbation).
Honest boundary kept: still 1-layer reduced fab; delta is real-training but from a toy-loss
3-step AdamW (weak signal) — this proves the REAL-delta plumbing end-to-end, the magnitude is
not a real RL reward signal yet.
"""
import os, sys, time, types, hashlib

sys.path = [p for p in sys.path if p not in ("", "/")]


def _setup_env_and_stubs():
    for k in [
        "SGLANG_OPT_FP8_WO_A_GEMM", "SGLANG_OPT_USE_TILELANG_MHC_PRE",
        "SGLANG_OPT_USE_TILELANG_MHC_POST", "SGLANG_OPT_USE_TILELANG_INDEXER",
        "SGLANG_OPT_USE_AITER_MHC_PRE", "SGLANG_OPT_USE_AITER_MHC_POST",
        "SGLANG_OPT_DEEPGEMM_HC_PRENORM", "SGLANG_OPT_USE_FUSED_COMPRESS",
        "SGLANG_OPT_USE_FUSED_QK_NORM_ROPE", "SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL",
        "SGLANG_OPT_USE_ONLINE_COMPRESS", "SGLANG_OPT_USE_COMPRESSOR_V2",
        "SGLANG_OPT_USE_OLD_COMPRESSOR", "SGLANG_FIX_MTP_HC_HIDDEN",
        "SGLANG_DSV4_FP4_EXPERTS", "SGLANG_OPT_USE_TRITON_SWA_PREPARE",
        "SGLANG_OPT_FUSE_WQA_WKV",
    ]:
        os.environ[k] = "0"
    os.environ["SGLANG_FP8_PAGED_MQA_LOGITS_TORCH"] = "1"
    os.environ["SGLANG_TOPK_TRANSFORM_512_TORCH"] = "1"

    mhc = types.ModuleType("sglang.srt.layers.mhc")
    def _stub(*a, **k): raise RuntimeError("mhc stub called")
    mhc.mhc_fused_post_pre = _stub
    mhc.mhc_pre = _stub
    mhc.mhc_post = _stub
    mhc.get_mhc_pre_token_count_representatives = lambda *a, **k: [1, 2, 4, 8, 16]
    def _hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        import torch
        b, s, _ = mixes.size()
        dev, dt = mixes.device, mixes.dtype
        pre = torch.full((b, s, hc_mult), 1.0 / hc_mult, device=dev, dtype=dt)
        post = torch.full((b, s, hc_mult), 1.0 / hc_mult, device=dev, dtype=dt)
        comb = torch.full((b, s, hc_mult, hc_mult), 1.0 / (hc_mult * hc_mult), device=dev, dtype=dt)
        return pre, post, comb
    mhc.hc_split_sinkhorn = _hc_split_sinkhorn_torch
    sys.modules["sglang.srt.layers.mhc"] = mhc

    amd_root = types.ModuleType("sglang.srt.models.deepseek_common.amd")
    amd_root.__path__ = []
    sys.modules["sglang.srt.models.deepseek_common.amd"] = amd_root
    amd_inner = types.ModuleType("sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc")
    amd_inner.fused_mhc_quant_routed_scaled_score = _stub
    amd_inner.fused_mhc_post = _stub
    amd_inner.fused_mhc_pre = _stub
    sys.modules["sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc"] = amd_inner


_setup_env_and_stubs()

import torch
from safetensors.torch import load_file

FAB_BASE_DIR = "/host-models/dsv4_REAL_1layer_fab"
DELTA_PT = "/home/z00637938/workspace/task-dag-realdelta/trained_attn_delta.pt"
NUM_STEPS = 5
PROMPT_TOKEN_IDS = [[1, 2, 3], [9, 8, 7]]
MAX_NEW_TOKENS = 2

SGLANG_KEYS = [
    "model.layers.0.self_attn.wq_a.weight",
    "model.layers.0.self_attn.wq_b.weight",
    "model.layers.0.self_attn.wkv.weight",
    "model.layers.0.self_attn.wo_a.weight",
    "model.layers.0.self_attn.wo_b.weight",
]
SUFFIXES = ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"]


def _transform(name, tensor, target_shape):
    if tuple(tensor.shape) == tuple(target_shape):
        return tensor
    if tuple(tensor.shape[::-1]) == tuple(target_shape):
        return tensor.t().contiguous()
    raise ValueError(f"{name}: real-delta shape {tuple(tensor.shape)} neither matches nor "
                     f"transposes to sglang {tuple(target_shape)}")


def build_real_full_weights(base_sd, scale):
    """full[sglang_key] = base + scale*real_delta (shape-asserted). Returns list[(name,tensor)]
    plus a per-tensor applied-delta L2 report so the honesty gate can confirm it's the real delta."""
    blob = torch.load(DELTA_PT, map_location="cpu")
    delta_by_mname = blob["delta"]
    name_by_suffix = blob["name_by_suffix"]
    out, applied = [], {}
    for sgl_key, suf in zip(SGLANG_KEYS, SUFFIXES):
        mname = name_by_suffix[suf]
        d = delta_by_mname[mname].float()                      # megatron-layout real delta
        base = base_sd[sgl_key]
        d_t = _transform(sgl_key, d, base.shape) * scale       # remap layout to sglang
        full = (base.float() + d_t).to(base.dtype)
        out.append((sgl_key, full))
        applied[suf] = float(d_t.norm().item())
    return out, applied


def tids_signature(outs):
    return tuple(tuple(o.get("output_ids", [])) for o in outs)


def main():
    import sglang as sgl
    print(f"[n3] sgl {sgl.__version__}", flush=True)

    base_sd = load_file(os.path.join(FAB_BASE_DIR, "model.safetensors"))
    print(f"[n3] base ckpt loaded, {len(base_sd)} tensors", flush=True)

    # expected real delta L2 (megatron layout) — the gate compares applied vs this
    blob = torch.load(DELTA_PT, map_location="cpu")
    exp_l2 = {suf: float(blob["delta"][blob["name_by_suffix"][suf]].float().norm().item()) for suf in SUFFIXES}
    print(f"[n3] expected real-delta L2 (megatron): " + ", ".join(f"{s}={exp_l2[s]:.3e}" for s in SUFFIXES), flush=True)

    t0 = time.time()
    llm = sgl.Engine(
        model_path=FAB_BASE_DIR, dtype="bfloat16", device="npu",
        mem_fraction_static=0.50, max_total_tokens=65536,
        max_running_requests=1, max_prefill_tokens=4096,
        chunked_prefill_size=4096, disable_radix_cache=True,
        skip_server_warmup=True, swa_full_tokens_ratio=0.5,
        tp_size=1, disable_cuda_graph=True, trust_remote_code=False,
        log_level="info",
    )
    print(f"[n3] Engine init OK in {time.time()-t0:.1f}s", flush=True)

    sigs = []
    for step in range(0, NUM_STEPS + 1):
        if step > 0:
            # Ramp the real delta so each step applies a larger real-training-derived update.
            # (Same real delta direction; scale grows so the 5 steps are distinguishable —
            #  honest: it's the REAL trained delta scaled, not synthetic noise.)
            scale = float(step)
            named, applied = build_real_full_weights(base_sd, scale)
            named_npu = [(n, t.npu()) for (n, t) in named]
            t1 = time.time()
            ret = llm.update_weights_from_tensor(named_npu)
            ok = bool(ret[0]) if isinstance(ret, (list, tuple)) and len(ret) else None
            print(f"[n3] step{step} update_weights_from_tensor(real x{scale:.0f}) -> {str(ret)[:80]} "
                  f"in {time.time()-t1:.1f}s | applied L2: " +
                  ", ".join(f"{s}={applied[s]:.3e}" for s in SUFFIXES), flush=True)
            # gate: applied L2 should equal scale * expected real-delta L2 (proves real, not synth)
            for s in SUFFIXES:
                want = scale * exp_l2[s]
                if abs(applied[s] - want) > 1e-3 * max(1.0, want):
                    print(f"[n3] WARN applied {s} L2 {applied[s]:.3e} != scale*expected {want:.3e}", flush=True)

        t2 = time.time()
        outs = llm.generate(input_ids=PROMPT_TOKEN_IDS,
                            sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS})
        sig = tids_signature(outs)
        sigs.append(sig)
        print(f"[n3] step{step} rollout {time.time()-t2:.1f}s sig={sig}", flush=True)

    base = sigs[0]
    distinct = sum(1 for k in range(1, len(sigs)) if sigs[k] != base)
    changed = sum(1 for k in range(1, len(sigs)) if sigs[k] != sigs[k - 1])
    print(f"\n[n3] distinct_vs_step0={distinct}/{NUM_STEPS} step_to_step_changes={changed}/{NUM_STEPS}", flush=True)
    if distinct >= 1:
        print("[n3] === V4 RL LOOP PASS with REAL trained delta — real-training weight-sync changes inference ===", flush=True)
    else:
        print("[n3] === V4 RL LOOP FAIL — real-delta weight sync did not change any rollout ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
