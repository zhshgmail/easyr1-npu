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
sys.modules["sglang.srt.layers.mhc"] = mhc

amd_root = types.ModuleType("sglang.srt.models.deepseek_common.amd")
amd_root.__path__ = []
sys.modules["sglang.srt.models.deepseek_common.amd"] = amd_root
amd_inner = types.ModuleType("sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc")
amd_inner.fused_mhc_quant_routed_scaled_score = _stub
amd_inner.fused_mhc_post = _stub
amd_inner.fused_mhc_pre = _stub
sys.modules["sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc"] = amd_inner


def main():
    print("[v4-min] importing", flush=True)
    import sglang as sgl
    print(f"[v4-min] sgl {sgl.__version__}", flush=True)
    print("[v4-min] Engine starting", flush=True)
    t0 = time.time()
    try:
        llm = sgl.Engine(
            model_path="/host-models/dsv4_REAL_1layer_fab",
            dtype="bfloat16",
            device="npu",
            mem_fraction_static=0.10,
            max_total_tokens=4096,
            max_running_requests=1,
            max_prefill_tokens=1024,
            chunked_prefill_size=1024,
            disable_radix_cache=True,
            skip_server_warmup=True,
            tp_size=1,
            disable_cuda_graph=True,
            trust_remote_code=False,
            log_level="info",
        )
        print(f"[v4-min] Engine init OK in {time.time()-t0:.1f}s", flush=True)
        print("[v4-min] CALLING generate (timeout 120s expected)", flush=True)
        t1=time.time()
        outs = llm.generate(["Hi"], sampling_params={"temperature":0.0, "max_new_tokens":2})
        print(f"[v4-min] generate done in {time.time()-t1:.1f}s", flush=True)
        print(f"[v4-min] output: {outs}", flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
