"""V4 reduced-to-1-layer FULL decoder block on NPU (proper TransformerBlock: attn+MLP+norm+residual)."""
import os, types, torch, torch_npu
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29535")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
os.environ.setdefault("TILELANG_ASCEND_MODE","Developer"); os.environ["MEGATRON_SPARSE_ATTN_IMPL"]="sparse"
import mindspeed.megatron_adaptor  # noqa
import torch.distributed as dist
torch.npu.set_device(0)
if not dist.is_initialized(): dist.init_process_group(backend="hccl", rank=0, world_size=1)
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)
from megatron.core.transformer.transformer_config import MLATransformerConfig
cfg = MLATransformerConfig(num_layers=1, hidden_size=512, num_attention_heads=4, ffn_hidden_size=1024,
    kv_channels=128, q_lora_rank=1536, kv_lora_rank=512, qk_head_dim=128, qk_pos_emb_head_dim=64,
    v_head_dim=512, rotary_base=10000.0, rotary_scaling_factor=4, rotary_percent=1.0,
    original_max_position_embeddings=65536, mscale=1.0, mscale_all_dim=1.0, beta_fast=32, beta_slow=1,
    add_bias_linear=False, layernorm_epsilon=1e-6, normalization="RMSNorm", params_dtype=torch.bfloat16, bf16=True,
    moe_layer_freq=0, num_moe_experts=None)
cfg.experimental_attention_variant="dsv4"; cfg.transformer_impl="transformer_engine"
for k,v in dict(dsv4_o_lora_rank=1024, dsv4_o_groups=4, dsv4_window_size=128, dsv4_compress_ratios=None,
    dsa_indexer_n_heads=4, dsa_indexer_head_dim=128, dsa_indexer_topk=16, dsv4_hc_mult=4,
    dsv4_hc_sinkhorn_iters=20, dsv4_hc_eps=1e-6, dsv4_compress_rope_theta=160000, dsv4_swiglu_limit=0.0,
    dsv4_n_hash_layers=3).items(): setattr(cfg,k,v)
# patch the all_reduce_grad_fp32 (already applied to installed megatron, but ensure)
from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec
args = types.SimpleNamespace(num_layers=1)
spec = get_dsv4_spec(args, cfg, vp_stage=None)
print("dsv4 block spec OK:", type(spec).__name__)
from megatron.core.transformer.transformer_block import TransformerBlock
block = TransformerBlock(cfg, spec, post_layer_norm=True, pre_process=True, post_process=True).npu().to(torch.bfloat16)
for n,p in block.named_parameters():
    if "attn_sink" in n: p.data = p.data.float()
print(f"V4 1-layer TransformerBlock built: {sum(p.numel() for p in block.parameters())/1e6:.1f}M params")
SEQ,B = 64,1
hs = (torch.randn(SEQ,B,cfg.hidden_size,dtype=torch.bfloat16)*0.1).npu()
try:
    out = block(hidden_states=hs, attention_mask=None)
    o = out[0] if isinstance(out,(tuple,list)) else out
    torch.npu.synchronize()
    print(f"V4 1-LAYER BLOCK FORWARD OK on NPU: out={tuple(o.shape)} finite={torch.isfinite(o.float()).all().item()}")
    loss=o.float().pow(2).mean(); loss.backward(); torch.npu.synchronize()
    gn=sum((p.grad.float().pow(2).sum() for p in block.parameters() if p.grad is not None)).sqrt().item()
    print(f"V4 1-LAYER BLOCK BACKWARD OK on NPU: loss={loss.item():.4f} grad_norm={gn:.3e}")
    print("=> V4 reduced-1-layer FULL decoder block (attn+MLP+norm+residual) training step runs on NPU")
except Exception as e:
    import traceback; print("FAIL:", type(e).__name__, str(e)[:150])
    for ln in traceback.format_exc().splitlines():
        if "deepseek_v4.py" in ln or "transformer_block" in ln or "transformer_layer" in ln: print("  ", ln.strip()[:110])
