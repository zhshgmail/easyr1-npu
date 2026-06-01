"""Bypass sglang.Engine + scheduler / IPC / tokenizer. Build the V4 model
directly with sglang's loader, then call forward() with hand-rolled input
tokens. This isolates "does V4 forward run on NPU" from "does the sglang
server pipeline work for V4".
"""
import os, sys, types, time, traceback
sys.path = [p for p in sys.path if p not in ("", "/")]

# Force all V4 sub-opts off
for k in [
    "SGLANG_OPT_FP8_WO_A_GEMM",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE",
    "SGLANG_OPT_USE_TILELANG_MHC_POST",
    "SGLANG_OPT_USE_AITER_MHC_PRE",
    "SGLANG_OPT_USE_AITER_MHC_POST",
    "SGLANG_OPT_USE_FUSED_QK_NORM_ROPE",
    "SGLANG_OPT_FUSE_WQA_WKV",
    "SGLANG_OPT_USE_FUSED_COMPRESS",
    "SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM",
]:
    os.environ[k] = "0"

# stub mhc
mhc = types.ModuleType("sglang.srt.layers.mhc")
mhc.mhc_fused_post_pre = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
mhc.mhc_pre = mhc.mhc_fused_post_pre
mhc.mhc_post = mhc.mhc_fused_post_pre
mhc.get_mhc_pre_token_count_representatives = lambda *a, **k: [1, 2, 4, 8, 16]
sys.modules["sglang.srt.layers.mhc"] = mhc

amd_root = types.ModuleType("sglang.srt.models.deepseek_common.amd")
amd_root.__path__ = []
sys.modules["sglang.srt.models.deepseek_common.amd"] = amd_root
amd_inner = types.ModuleType(
    "sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc"
)
for name in (
    "fused_mhc_quant_routed_scaled_score",
    "fused_mhc_post",
    "fused_mhc_pre",
):
    setattr(
        amd_inner, name,
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"stub {name}")),
    )
sys.modules[
    "sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc"
] = amd_inner

print("[direct] importing sglang", flush=True)
import sglang
import torch
import torch_npu  # noqa

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs

print(f"[direct] sglang {sglang.__version__}", flush=True)
print("[direct] building ServerArgs", flush=True)
server_args = ServerArgs(
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
print("[direct] building ModelConfig", flush=True)
model_config = ModelConfig.from_server_args(server_args)
print(f"[direct] model_config arch={model_config.hf_config.architectures}", flush=True)

# Try to actually import the V4 model class and instantiate it
print("[direct] importing deepseek_v4 module", flush=True)
import sglang.srt.models.deepseek_v4 as v4mod
print(f"[direct] v4mod loaded, EntryClass={v4mod.EntryClass}", flush=True)

print("[direct] DONE — model class loadable + config parsed", flush=True)
print("[direct] V4 forward not invoked here; this script confirms only what's", flush=True)
print("[direct] reachable in the foreground process. Real forward needs ModelRunner.", flush=True)
