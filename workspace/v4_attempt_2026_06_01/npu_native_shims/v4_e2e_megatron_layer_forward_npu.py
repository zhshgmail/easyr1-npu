"""V4 e2e megatron-layer forward on NPU — adapts glm5 _e2e_megatron_step_mindspeed.py pattern.
mindspeed.megatron_adaptor + strip newer-megatron kwargs + MLA config + DeepSeekV4Attention(submodules=None)."""
import os, torch, torch_npu
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29531")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
os.environ.setdefault("TILELANG_ASCEND_MODE","Developer")
os.environ["MEGATRON_SPARSE_ATTN_IMPL"]="sparse"  # use V4 pure-torch sparse_attn_torch (runs on NPU via torch dispatch), avoids shim shape-mismatch
import mindspeed.megatron_adaptor  # noqa: F401  patch_features() — cuda->npu/RNG/TE/etc
import torch.distributed as dist
torch.npu.set_device(0)
if not dist.is_initialized(): dist.init_process_group(backend="hccl", rank=0, world_size=1)
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)

# --- strip newer-megatron kwargs the installed Mcore lacks (the skew fix, function-level) ---
import megatron.core.tensor_parallel.mappings as _maps
_orig_copy = _maps.copy_to_tensor_model_parallel_region
def _copy_strip(input_, group=None, **kw):   # drop all_reduce_grad_fp32 etc.
    return _orig_copy(input_, group=group)
_maps.copy_to_tensor_model_parallel_region = _copy_strip
# patch the symbol the V4 module already imported, too
import miles_plugins.models.deepseek_v4.deepseek_v4 as v4mod
if hasattr(v4mod, "copy_to_tensor_model_parallel_region"):
    v4mod.copy_to_tensor_model_parallel_region = _copy_strip
print("kwarg-strip patch applied")

from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.enums import AttnMaskType
cfg = MLATransformerConfig(num_layers=1, hidden_size=512, num_attention_heads=4, ffn_hidden_size=1024,
    kv_channels=128, q_lora_rank=1536, kv_lora_rank=512, qk_head_dim=128, qk_pos_emb_head_dim=64,
    v_head_dim=512, rotary_base=10000.0, rotary_scaling_factor=4, rotary_percent=1.0,
    original_max_position_embeddings=65536, mscale=1.0, mscale_all_dim=1.0, beta_fast=32, beta_slow=1,
    add_bias_linear=False, layernorm_epsilon=1e-6, normalization="RMSNorm",
    params_dtype=torch.bfloat16, bf16=True)
cfg.experimental_attention_variant="dsv4"; cfg.transformer_impl="transformer_engine"
for k,v in dict(dsv4_o_lora_rank=1024, dsv4_o_groups=4, dsv4_window_size=128, dsv4_compress_ratios=None,
    dsa_indexer_n_heads=4, dsa_indexer_head_dim=128, dsa_indexer_topk=16, dsv4_hc_mult=4,
    dsv4_hc_sinkhorn_iters=20, dsv4_hc_eps=1e-6, dsv4_compress_rope_theta=160000, dsv4_swiglu_limit=0.0,
    dsv4_n_hash_layers=3).items(): setattr(cfg,k,v)
print("cfg built")
from miles_plugins.models.deepseek_v4.deepseek_v4 import DeepSeekV4Attention
attn = DeepSeekV4Attention(cfg, submodules=None, layer_number=1, attn_mask_type=AttnMaskType.causal).npu().to(torch.bfloat16)
if hasattr(attn,"attn_sink"): attn.attn_sink.data = attn.attn_sink.data.float()
print(f"DeepSeekV4Attention built: {sum(p.numel() for p in attn.parameters())/1e6:.1f}M params")
SEQ, B = 64, 1
hs = (torch.randn(SEQ, B, cfg.hidden_size, dtype=torch.bfloat16)*0.1).npu()
try:
    out = attn(hs)
    o = out[0] if isinstance(out,(tuple,list)) else out
    torch.npu.synchronize()
    print(f"V4 ATTENTION LAYER FORWARD OK on NPU: out={tuple(o.shape)} finite={torch.isfinite(o.float()).all().item()}")
except Exception as e:
    import traceback; print("FORWARD FAIL:", type(e).__name__, str(e)[:150])
    for ln in traceback.format_exc().splitlines():
        if "deepseek_v4.py" in ln: print("  ", ln.strip()[:120])
