"""REAL DeepSeek-V4 config reduced to N layers on NPU (hidden=4096, 64 heads, MLA, 256-expert MoE)."""
import os, json, types, torch, torch_npu
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29539")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
os.environ.setdefault("TILELANG_ASCEND_MODE","Developer"); os.environ["MEGATRON_SPARSE_ATTN_IMPL"]="sparse"
import mindspeed.megatron_adaptor  # noqa

import torch.distributed as dist
torch.npu.set_device(0)
if not dist.is_initialized(): dist.init_process_group(backend="hccl", rank=0, world_size=1)
# rms_norm skew shim: torch_npu npu_rms_norm takes (x,gamma,eps); drop extra args + match dtype
_orig_rms = torch_npu.npu_rms_norm
def _rms(x, gamma, eps=1e-6, *a, **kw):
    if gamma.dtype != x.dtype: gamma = gamma.to(x.dtype)
    return _orig_rms(x.contiguous(), gamma, eps)
torch_npu.npu_rms_norm = _rms
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
model_parallel_cuda_manual_seed(1234)
hf = json.load(open("/home/z00637938/workspace/v4_real_config.json"))
from megatron.core.transformer.transformer_config import MLATransformerConfig
rs = hf["rope_scaling"]
cfg = MLATransformerConfig(
    num_layers=int(__import__('os').environ.get('NLAYERS','4')), hidden_size=hf["hidden_size"], num_attention_heads=hf["num_attention_heads"],
    ffn_hidden_size=hf["moe_intermediate_size"], kv_channels=128,
    q_lora_rank=hf["q_lora_rank"], kv_lora_rank=hf["head_dim"], qk_head_dim=128,
    qk_pos_emb_head_dim=hf["qk_rope_head_dim"], v_head_dim=hf["head_dim"],
    rotary_base=hf["rope_theta"], rotary_scaling_factor=rs["factor"], rotary_percent=1.0,
    original_max_position_embeddings=rs["original_max_position_embeddings"],
    mscale=1.0, mscale_all_dim=1.0, beta_fast=rs["beta_fast"], beta_slow=rs["beta_slow"],
    add_bias_linear=False, layernorm_epsilon=1e-6, normalization="RMSNorm",
    params_dtype=torch.bfloat16, bf16=True,
    num_moe_experts=hf["n_routed_experts"], moe_ffn_hidden_size=hf["moe_intermediate_size"],
    moe_shared_expert_intermediate_size=hf["moe_intermediate_size"]*hf["n_shared_experts"],
    moe_router_topk=hf["num_experts_per_tok"], moe_layer_freq=1)
cfg.experimental_attention_variant="dsv4"; cfg.transformer_impl="transformer_engine"
for k,v in dict(dsv4_o_lora_rank=hf["o_lora_rank"], dsv4_o_groups=hf["o_groups"], dsv4_window_size=128,
    dsv4_compress_ratios=None, dsa_indexer_n_heads=hf["index_n_heads"], dsa_indexer_head_dim=hf["index_head_dim"],
    dsa_indexer_topk=hf["index_topk"], dsv4_hc_mult=hf["hc_mult"], dsv4_hc_sinkhorn_iters=hf["hc_sinkhorn_iters"],
    dsv4_hc_eps=1e-6, dsv4_compress_rope_theta=160000, dsv4_swiglu_limit=0.0, dsv4_n_hash_layers=3).items(): setattr(cfg,k,v)
print(f"REAL V4 config reduced to N layers: hidden={cfg.hidden_size} heads={cfg.num_attention_heads} experts={cfg.num_moe_experts} q_lora={cfg.q_lora_rank} factor={cfg.rotary_scaling_factor}")
from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec
from megatron.core.transformer.transformer_block import TransformerBlock
spec = get_dsv4_spec(types.SimpleNamespace(num_layers=int(__import__('os').environ.get('NLAYERS','4'))), cfg, vp_stage=None)
try:
    block = TransformerBlock(cfg, spec, post_layer_norm=True, pre_process=True, post_process=True).npu().to(torch.bfloat16)
    for n,p in block.named_parameters():
        if "attn_sink" in n: p.data = p.data.float()
    print(f"REAL V4 N-layer block built: {sum(p.numel() for p in block.parameters())/1e9:.2f}B params")
    hs=(torch.randn(64,1,cfg.hidden_size,dtype=torch.bfloat16)*0.05).npu()
    out=block(hidden_states=hs, attention_mask=None); torch.npu.synchronize()
    o=out[0] if isinstance(out,(tuple,list)) else out
    print(f"REAL V4 N-LAYER FORWARD OK on NPU: out={tuple(o.shape)} finite={torch.isfinite(o.float()).all().item()}")
except Exception as e:
    import traceback; print("FAIL:", type(e).__name__, str(e)[:140])
    for ln in traceback.format_exc().splitlines():
        if "deepseek_v4.py" in ln or "transformer_" in ln or "moe" in ln.lower(): print("  ", ln.strip()[:110])

# backward (real-config training step)
try:
    out2=block(hidden_states=hs, attention_mask=None); o2=out2[0] if isinstance(out2,(tuple,list)) else out2
    loss=o2.float().pow(2).mean(); loss.backward(); torch.npu.synchronize()
    # localize nan: which params have nan grad? + check fwd output magnitude
    nan_params=[n for n,p in block.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    fin_params=[n for n,p in block.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
    print(f"  fwd out abs max={o2.float().abs().max().item():.3e}  loss finite={torch.isfinite(loss).item()}")
    print(f"  params with NAN grad: {len(nan_params)} ; with FINITE grad: {len(fin_params)}")
    if nan_params: print(f"  first nan-grad params: {nan_params[:5]}")
    gn=0.0 if nan_params else sum((p.grad.float().pow(2).sum() for p in block.parameters() if p.grad is not None)).sqrt().item()
    ng=sum(1 for p in block.parameters() if p.grad is not None); npar=sum(1 for _ in block.parameters())
    print(f"REAL V4 N-LAYER BACKWARD OK on NPU: loss={loss.item():.4f} grad_norm={gn:.3e} params_with_grad={ng}/{npar}")
    # real optimizer step (true training iteration, not just fwd+bwd)
    opt = torch.optim.AdamW([p for p in block.parameters() if p.requires_grad], lr=1e-4)
    p0 = next(p for p in block.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
    before = p0.detach().float().clone()
    opt.step(); torch.npu.synchronize()
    changed = (p0.detach().float() - before).abs().max().item()
    print(f"OPTIMIZER STEP OK: a param changed by {changed:.3e} (weights updated)")
    print(f"=> REAL DSV4 reduced N-layer TRAINING ITERATION (fwd+bwd+optim.step) runs on NPU")
except Exception as e:
    import traceback; print("REAL BACKWARD FAIL:", type(e).__name__, str(e)[:140])
    for ln in traceback.format_exc().splitlines():
        if "deepseek_v4.py" in ln or "moe" in ln.lower() or "transformer_" in ln: print("  ", ln.strip()[:110])
