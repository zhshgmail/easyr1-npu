"""V4 RL loop closure PoC v2 — uses update_weights_from_tensor (attention-only)
to sidestep the FusedMoE reload bug (Issue #26794) on the weight-sync step.

Per user 2026-06-01:
  * gibberish output is expected (extreme 减层); validate loop integrity
  * run the WHOLE loop: rollout -> (synth) train -> weight-sync -> re-rollout
  * #26794: MoE-active /update_weights_from_disk reload crashes in FusedMoE
    _load_w2 (narrow length 4096 > dim 2048). On NPU same as V3.2 path.

Strategy: push ONLY the attention weights via update_weights_from_tensor
(named_tensors API). MoE experts are never reloaded -> no _load_w2 crash.
The loop still proves: weight-sync changes inference (distinct rollouts).
This is the same "prove plumbing, swap real train later" PoC discipline as
T32 sglang_2step_real_update.py.

Runs INSIDE sgl_probe (chip 1). Uses Python Engine API in-process.
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
NUM_STEPS = 5
PROMPT_TOKEN_IDS = [[1, 2, 3], [9, 8, 7]]
MAX_NEW_TOKENS = 2

# Only ATTENTION weights — never the MoE experts. This is what dodges #26794.
PERTURB_KEYS = [
    "model.layers.0.self_attn.wq_a.weight",
    "model.layers.0.self_attn.wq_b.weight",
    "model.layers.0.self_attn.wkv.weight",
    "model.layers.0.self_attn.wo_a.weight",
    "model.layers.0.self_attn.wo_b.weight",
]
PERTURB_SCALE = 0.5


def synth_attention_delta(base_sd, seed):
    """Return list[(name, perturbed_tensor)] for attention weights only."""
    g = torch.Generator()
    g.manual_seed(seed)
    out = []
    for k in PERTURB_KEYS:
        t = base_sd[k]
        delta = torch.randn(t.shape, generator=g, dtype=torch.float32) * PERTURB_SCALE
        out.append((k, (t.float() + delta).to(t.dtype)))
    return out


def tids_signature(outs):
    return tuple(tuple(o.get("output_ids", [])) for o in outs)


def main():
    import sglang as sgl
    print(f"[v4-rl2] sgl {sgl.__version__}", flush=True)

    base_sd = load_file(os.path.join(FAB_BASE_DIR, "model.safetensors"))
    print(f"[v4-rl2] base ckpt loaded, {len(base_sd)} tensors", flush=True)

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
    print(f"[v4-rl2] Engine init OK in {time.time()-t0:.1f}s", flush=True)

    sigs = []
    for step in range(0, NUM_STEPS + 1):
        if step > 0:
            named = synth_attention_delta(base_sd, seed=1000 + step)
            # move tensors to npu for the update
            named_npu = [(n, t.npu()) for (n, t) in named]
            t1 = time.time()
            ret = llm.update_weights_from_tensor(named_npu)
            print(f"[v4-rl2] step{step} update_weights_from_tensor ({len(named_npu)} attn tensors) "
                  f"-> {str(ret)[:120]} in {time.time()-t1:.1f}s", flush=True)

        t2 = time.time()
        outs = llm.generate(
            input_ids=PROMPT_TOKEN_IDS,
            sampling_params={"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS},
        )
        sig = tids_signature(outs)
        sigs.append(sig)
        for j, o in enumerate(outs):
            print(f"[v4-rl2] step{step} prompt{j} ids={o.get('output_ids')} text={o.get('text')!r}", flush=True)
        print(f"[v4-rl2] step{step} rollout {time.time()-t2:.1f}s sig={sig}", flush=True)

    print(f"\n[v4-rl2] === 5-step RL loop summary (attention-only weight sync) ===", flush=True)
    base = sigs[0]
    distinct = 0
    transitions_changed = 0
    for k in range(1, len(sigs)):
        diff_vs_base = sigs[k] != base
        diff_vs_prev = sigs[k] != sigs[k - 1]
        if diff_vs_base:
            distinct += 1
        if diff_vs_prev:
            transitions_changed += 1
        print(f"[v4-rl2]   step{k}: diff_vs_step0={diff_vs_base} diff_vs_prev={diff_vs_prev}", flush=True)
    print(f"[v4-rl2]   distinct_vs_step0={distinct}/{NUM_STEPS}  "
          f"step_to_step_changes={transitions_changed}/{NUM_STEPS}", flush=True)

    # Loop PASS = weight sync demonstrably changes inference at least once.
    if distinct >= 1:
        print(f"[v4-rl2] === V4 RL LOOP PASS — attention weight-sync changes inference "
              f"(rollout->weight-update->rollout closed) ===", flush=True)
    else:
        print(f"[v4-rl2] === V4 RL LOOP FAIL — weight sync did not change any rollout ===",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
