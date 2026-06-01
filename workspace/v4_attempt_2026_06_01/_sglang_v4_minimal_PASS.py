import os, sys, types, time, traceback
sys.path = [p for p in sys.path if p not in ("", "/")]

# Disable ALL fp8/tilelang/triton/aiter optimizations to force the pure-torch
# fallback path of V4 — this is the path we want for NPU bf16 PoC
for k in [
    "SGLANG_OPT_FP8_WO_A_GEMM",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE",
    "SGLANG_OPT_USE_TILELANG_MHC_POST",
    "SGLANG_OPT_USE_TILELANG_INDEXER",
    "SGLANG_OPT_USE_AITER_MHC_PRE",
    "SGLANG_OPT_USE_AITER_MHC_POST",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
    "SGLANG_OPT_USE_FUSED_COMPRESS",
    "SGLANG_OPT_USE_FUSED_QK_NORM_ROPE",
    "SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL",
    "SGLANG_OPT_USE_ONLINE_COMPRESS",
    "SGLANG_OPT_USE_COMPRESSOR_V2",
    "SGLANG_OPT_USE_OLD_COMPRESSOR",  # keep new path off too
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH",  # already torch
    "SGLANG_FIX_MTP_HC_HIDDEN",
    "SGLANG_DSV4_FP4_EXPERTS",
    "SGLANG_OPT_USE_TRITON_SWA_PREPARE",
    "SGLANG_OPT_FUSE_WQA_WKV",
]:
    os.environ[k] = "0"

# Use torch fallback for these instead of fused triton
os.environ["SGLANG_FP8_PAGED_MQA_LOGITS_TORCH"] = "1"
os.environ["SGLANG_TOPK_TRANSFORM_512_TORCH"] = "1"

mhc = types.ModuleType("sglang.srt.layers.mhc")
def _stub(*a, **k): raise RuntimeError("mhc stub called — should NOT happen with all opts disabled")
mhc.mhc_fused_post_pre = _stub
mhc.mhc_pre = _stub
mhc.mhc_post = _stub
mhc.get_mhc_pre_token_count_representatives = lambda *a, **k: [1, 2, 4, 8, 16]

def _hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    # Torch fallback: produce shape-correct pre/post/comb without true sinkhorn.
    # Real sinkhorn is for top-k routing; for forward-flow PoC use uniform routing.
    import torch
    b, s, _ = mixes.size()
    dev, dt = mixes.device, mixes.dtype
    pre = torch.full((b, s, hc_mult), 1.0/hc_mult, device=dev, dtype=dt)
    post = torch.full((b, s, hc_mult), 1.0/hc_mult, device=dev, dtype=dt)
    comb = torch.full((b, s, hc_mult, hc_mult), 1.0/(hc_mult*hc_mult), device=dev, dtype=dt)
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



def _install_npu_jit_kernel_fallbacks():
    """V4 forward calls JIT-compiled CUDA kernels for fused norm+rope. On NPU
    we replace them with torch fallbacks so the algorithm executes."""
    import torch

    def _torch_fused_q_norm_rope(q_input, q_output, eps, freqs_cis, positions):
        # q_input: [N, n_heads, head_dim]; head_dim = nope_dim + rope_dim
        rope_dim = freqs_cis.shape[-1] * 2
        x = q_input.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + eps)
        x_norm = x_norm.to(q_input.dtype)
        nope_dim = q_input.shape[-1] - rope_dim
        if nope_dim > 0:
            q_output[..., :nope_dim].copy_(x_norm[..., :nope_dim])
        rope_part = x_norm[..., nope_dim:].float().contiguous()
        rope_complex = torch.view_as_complex(
            rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
        )
        f = freqs_cis[positions].unsqueeze(1)
        rotated = rope_complex * f
        rotated_real = torch.view_as_real(rotated).reshape(*rope_part.shape).to(q_input.dtype)
        q_output[..., nope_dim:].copy_(rotated_real)
        return None

    def _torch_fused_norm_rope_inplace(x, eps, freqs_cis, positions):
        # In-place rmsnorm over last dim + rope on rope_dim slice.
        # x: [N, n_heads, head_dim]
        rope_dim = freqs_cis.shape[-1] * 2
        nope_dim = x.shape[-1] - rope_dim
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        normed = (x_f * torch.rsqrt(var + eps)).to(x.dtype)
        if nope_dim > 0:
            x[..., :nope_dim].copy_(normed[..., :nope_dim])
        rope_part = normed[..., nope_dim:].float().contiguous()
        rope_complex = torch.view_as_complex(
            rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
        )
        f = freqs_cis[positions]
        # broadcast over head dim if present
        while f.dim() < rope_complex.dim():
            f = f.unsqueeze(-2)
        rotated = rope_complex * f
        rotated_real = torch.view_as_real(rotated).reshape(*rope_part.shape).to(x.dtype)
        x[..., nope_dim:].copy_(rotated_real)
        return None

    def _torch_fused_rope_inplace(x, freqs_cis, positions):
        rope_dim = freqs_cis.shape[-1] * 2
        nope_dim = x.shape[-1] - rope_dim
        rope_part = x[..., nope_dim:].float().contiguous()
        rope_complex = torch.view_as_complex(
            rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
        )
        f = freqs_cis[positions]
        while f.dim() < rope_complex.dim():
            f = f.unsqueeze(-2)
        rotated = rope_complex * f
        rotated_real = torch.view_as_real(rotated).reshape(*rope_part.shape).to(x.dtype)
        x[..., nope_dim:].copy_(rotated_real)
        return None

    # Patch the source module
    import sglang.jit_kernel.dsv4 as dsv4_pkg
    import sglang.jit_kernel.dsv4.elementwise as elw
    elw.fused_q_norm_rope = _torch_fused_q_norm_rope
    elw.fused_norm_rope_inplace = _torch_fused_norm_rope_inplace
    elw.fused_rope_inplace = _torch_fused_rope_inplace
    dsv4_pkg.fused_q_norm_rope = _torch_fused_q_norm_rope
    dsv4_pkg.fused_norm_rope_inplace = _torch_fused_norm_rope_inplace
    dsv4_pkg.fused_rope_inplace = _torch_fused_rope_inplace

    # Patch already-imported names inside deepseek_v4 module
    import sglang.srt.models.deepseek_v4 as dv4
    dv4.fused_q_norm_rope = _torch_fused_q_norm_rope
    dv4.fused_norm_rope_inplace = _torch_fused_norm_rope_inplace
    dv4.fused_rope_inplace = _torch_fused_rope_inplace
    print("[v4-min] dsv4 jit kernels monkey-patched to torch fallbacks (elw + dv4 namespaces)", flush=True)


def main():
    print("[v4-min] importing", flush=True)
    import sglang as sgl
    print(f"[v4-min] sgl {sgl.__version__}", flush=True)
    _install_npu_jit_kernel_fallbacks()
    print("[v4-min] Engine starting", flush=True)
    t0 = time.time()
    try:
        llm = sgl.Engine(
            model_path="/host-models/dsv4_REAL_1layer_fab",
            dtype="bfloat16",
            device="npu",
            mem_fraction_static=0.50,
            max_total_tokens=65536,
            max_running_requests=1,
            max_prefill_tokens=4096,
            chunked_prefill_size=4096,
            disable_radix_cache=True,
            skip_server_warmup=True,
            swa_full_tokens_ratio=0.5,
            tp_size=1,
            disable_cuda_graph=True,
            trust_remote_code=False,
            log_level="info",
        )
        print(f"[v4-min] Engine init OK in {time.time()-t0:.1f}s", flush=True)
        _install_npu_jit_kernel_fallbacks()
        print("[v4-min] CALLING generate (timeout 120s expected)", flush=True)
        t1=time.time()
        outs = llm.generate(input_ids=[[1, 2, 3]], sampling_params={"temperature":0.0, "max_new_tokens":2})
        print(f"[v4-min] generate done in {time.time()-t1:.1f}s", flush=True)
        print(f"[v4-min] output: {outs}", flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
