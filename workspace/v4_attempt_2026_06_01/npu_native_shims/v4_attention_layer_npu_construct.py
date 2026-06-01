import torch, torch_npu, os
torch.npu.set_device(0)
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29523")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1")
# CRITICAL: activate MindSpeed megatron patches (TELinear/TENorm -> NPU) BEFORE anything megatron
import mindspeed.megatron_adaptor   # noqa: F401  triggers patch_features()
print("mindspeed.megatron_adaptor imported (patch_features ran)")
import torch.distributed as dist
if not dist.is_initialized(): dist.init_process_group(backend="hccl", rank=0, world_size=1)
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(tensor_model_parallel_size=1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType
cfg = TransformerConfig(num_layers=1, hidden_size=512, num_attention_heads=4, kv_channels=128,
    tensor_model_parallel_size=1, add_bias_linear=False, gated_linear_unit=True,
    bias_activation_fusion=False, layernorm_epsilon=1e-6, params_dtype=torch.bfloat16, bf16=True)
cfg.experimental_attention_variant="dsv4"; cfg.transformer_impl="transformer_engine"
for k,v in dict(q_lora_rank=1536, kv_lora_rank=512, qk_pos_emb_head_dim=64, qk_head_dim=128,
    v_head_dim=128, dsv4_o_lora_rank=1024, dsv4_o_groups=4, dsv4_window_size=128,
    dsv4_compress_ratios=None, dsa_indexer_n_heads=4, dsa_indexer_head_dim=128, dsa_indexer_topk=16,
    dsv4_hc_mult=4, dsv4_hc_sinkhorn_iters=20, dsv4_hc_eps=1e-6, dsv4_compress_rope_theta=160000,
    dsv4_swiglu_limit=0.0, dsv4_n_hash_layers=3, rotary_base=10000, original_max_position_embeddings=65536, max_position_embeddings=4096, rotary_scaling_factor=4, mscale=1.0, mscale_all_dim=1.0, beta_fast=32, beta_slow=1).items(): setattr(cfg,k,v)
from miles_plugins.models.deepseek_v4.deepseek_v4 import DeepSeekV4Attention
try:
    attn = DeepSeekV4Attention(cfg, submodules=None, layer_number=1, attn_mask_type=AttnMaskType.causal).npu().to(torch.bfloat16)
    if hasattr(attn, "attn_sink"):
        attn.attn_sink.data = attn.attn_sink.data.float()  # V4 requires attn_sink fp32
    print(f"DeepSeekV4Attention CONSTRUCTED OK on NPU — params {sum(p.numel() for p in attn.parameters())/1e6:.1f}M")
except Exception as e:
    import traceback; print("CONSTRUCT FAIL:", type(e).__name__, str(e)[:140])
    for ln in traceback.format_exc().splitlines():
        if "deepseek_v4.py" in ln or "assert" in ln or "Error" in ln: print("  ", ln.strip()[:120])

# Forward pass through the constructed V4 attention layer on NPU
try:
    S, B = 128, 1
    hidden = torch.randn(S, B, 512, dtype=torch.bfloat16, device="npu:0")  # [seq, batch, hidden] megatron layout
    # V4 attention forward signature — try common megatron attention call forms
    import inspect
    sig = inspect.signature(attn.forward)
    print("forward params:", list(sig.parameters.keys())[:8])
    out = attn(hidden)
    o = out[0] if isinstance(out,(tuple,list)) else out
    torch.npu.synchronize()
    print(f"V4 ATTENTION FORWARD OK on NPU: out={tuple(o.shape)} dtype={o.dtype} finite={torch.isfinite(o.float()).all().item()}")
except Exception as e:
    import traceback; print("FORWARD FAIL:", type(e).__name__, str(e)[:160])
    for ln in traceback.format_exc().splitlines():
        if "deepseek_v4.py" in ln: print("  ", ln.strip()[:120])
