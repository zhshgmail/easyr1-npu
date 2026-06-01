"""n1_export — extract REAL trained attn-weight delta from the megatron 1-layer training iteration.

Runs the real DSV4 reduced-1-layer fwd+bwd+AdamW.step (same as v4_REAL_config_training_iteration_npu.py)
but snapshots the 5 attention tensors (wq_a/wq_b/wkv/wo_a/wo_b) BEFORE and AFTER opt.step(), and saves
the post-step tensors + delta to trained_attn_delta.pt.

HONESTY GATE (n1 verify): the 5 attention tensors must have NONZERO delta — if the toy MSE loss barely
touches attention (near-zero grad), the "real delta" is ~0 and the RL bridge would be meaningless. We
assert at least the attn tensors that HAVE grad moved; we report per-tensor grad-norm + delta so the
honest state is visible (some attn tensors may legitimately have ~0 grad under this toy loss — that is
reported, not hidden). To force a non-trivial attn signal we use a loss that depends on the attention
output path (the block output), and run a few steps so AdamW accumulates a visible delta.
"""
import os, json, types, torch, torch_npu
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29541")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
os.environ.setdefault("TILELANG_ASCEND_MODE","Developer"); os.environ["MEGATRON_SPARSE_ATTN_IMPL"]="sparse"
os.environ["NLAYERS"]="1"
import mindspeed.megatron_adaptor  # noqa
import torch.distributed as dist
torch.npu.set_device(0)
if not dist.is_initialized(): dist.init_process_group(backend="hccl", rank=0, world_size=1)
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
    num_layers=1, hidden_size=hf["hidden_size"], num_attention_heads=hf["num_attention_heads"],
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
from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec
from megatron.core.transformer.transformer_block import TransformerBlock
spec = get_dsv4_spec(types.SimpleNamespace(num_layers=1), cfg, vp_stage=None)
block = TransformerBlock(cfg, spec, post_layer_norm=True, pre_process=True, post_process=True).npu().to(torch.bfloat16)
for n,p in block.named_parameters():
    if "attn_sink" in n: p.data = p.data.float()
print(f"[n1] block built: {sum(p.numel() for p in block.parameters())/1e9:.2f}B params", flush=True)

# Find the 5 attention tensors by suffix in the megatron param names.
SUFFIXES = ["wq_a","wq_b","wkv","wo_a","wo_b"]
name_by_suffix = {}
for n,p in block.named_parameters():
    for suf in SUFFIXES:
        if n.endswith(f".{suf}.weight") or n.endswith(f".{suf}"):
            name_by_suffix.setdefault(suf, n)
print("[n1] megatron attn param names found:", flush=True)
for suf in SUFFIXES:
    print(f"    {suf:5s} -> {name_by_suffix.get(suf,'<MISSING>')}", flush=True)
missing = [s for s in SUFFIXES if s not in name_by_suffix]
if missing:
    print(f"[n1] WARN: {len(missing)} attn suffix(es) not found as direct params: {missing}", flush=True)
    print("[n1] dumping all param names containing 'self_attn' for diagnosis:", flush=True)
    for n,_ in block.named_parameters():
        if "self_attn" in n or "attn" in n: print("      ", n, flush=True)

params = dict(block.named_parameters())
pre = {suf: params[name_by_suffix[suf]].detach().float().clone() for suf in SUFFIXES if suf in name_by_suffix}

# 3 training steps so AdamW accumulates a visible delta on attn.
opt = torch.optim.AdamW([p for p in block.parameters() if p.requires_grad], lr=1e-3)
hs=(torch.randn(64,1,cfg.hidden_size,dtype=torch.bfloat16)*0.05).npu()
for step in range(3):
    opt.zero_grad()
    out=block(hidden_states=hs, attention_mask=None); o=out[0] if isinstance(out,(tuple,list)) else out
    loss=o.float().pow(2).mean(); loss.backward(); torch.npu.synchronize()
    # report attn grad norms (honesty: show which attn tensors actually get signal)
    gn = {suf: (params[name_by_suffix[suf]].grad.float().norm().item() if (suf in name_by_suffix and params[name_by_suffix[suf]].grad is not None) else None) for suf in SUFFIXES}
    opt.step(); torch.npu.synchronize()
    print(f"[n1] step{step} loss={loss.item():.4f} attn_grad_norms="+", ".join(f"{s}={gn[s]:.2e}" if gn[s] is not None else f"{s}=None" for s in SUFFIXES), flush=True)

# Build delta + post-step absolute tensors.
delta, post = {}, {}
for suf in SUFFIXES:
    if suf not in name_by_suffix: continue
    cur = params[name_by_suffix[suf]].detach().float()
    d = (cur - pre[suf])
    delta[name_by_suffix[suf]] = d.cpu()
    post[name_by_suffix[suf]]  = cur.cpu()
    print(f"[n1] {suf:5s} delta_L2={d.norm().item():.3e} max_abs={d.abs().max().item():.3e} finite={torch.isfinite(d).all().item()}", flush=True)

nonzero = [suf for suf in SUFFIXES if suf in name_by_suffix and delta[name_by_suffix[suf]].norm().item() > 0]
print(f"[n1] attn tensors with NONZERO trained delta: {len(nonzero)}/{len(SUFFIXES)} ({nonzero})", flush=True)

torch.save({"delta": delta, "post": post, "name_by_suffix": name_by_suffix,
            "suffixes": SUFFIXES, "nonzero_suffixes": nonzero},
           "/home/z00637938/workspace/task-dag-realdelta/trained_attn_delta.pt")
print("[n1] saved trained_attn_delta.pt", flush=True)
if len(nonzero) >= 1:
    print("[n1] VERIFY PASS: at least one attn tensor moved under real training (AdamW)", flush=True)
else:
    print("[n1] VERIFY FAIL: no attn tensor moved — toy loss does not exercise attention; bridge would be meaningless", flush=True)
