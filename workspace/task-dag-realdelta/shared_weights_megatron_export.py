"""SHARED-WEIGHTS megatron export — the honest version of the train->rollout flow.

Builds a 2-layer megatron DSV4 block, then exports its attention weights TWICE:
  - attn_INIT.pt  : the attn weights BEFORE training (this becomes sglang's shared init)
  - attn_TRAINED.pt: the attn weights AFTER 3 real AdamW steps

Key difference from the discredited n3: sglang's fab attn weights will be OVERWRITTEN with
attn_INIT (so both sides start from the SAME attention weights). Then pushing attn_TRAINED into
sglang is a meaningful update — it moves sglang's attn along the direction megatron actually trained,
because they started identical. Parameter flow megatron->sglang is real, not arbitrary.

Also dumps a synth control: attn_SYNTH.pt = attn_INIT + random delta of the SAME per-tensor L2 as
(attn_TRAINED - attn_INIT). The rollout verification requires: TRAINED rollout != INIT rollout AND
the change is NOT reproduced by the equal-magnitude synth control (else there's no training signal).
"""
import os, json, types, torch, torch_npu
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29543")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1"); os.environ.setdefault("LOCAL_RANK","0")
os.environ.setdefault("TILELANG_ASCEND_MODE","Developer"); os.environ["MEGATRON_SPARSE_ATTN_IMPL"]="sparse"
os.environ["NLAYERS"]="2"
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
    num_layers=2, hidden_size=hf["hidden_size"], num_attention_heads=hf["num_attention_heads"],
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
spec = get_dsv4_spec(types.SimpleNamespace(num_layers=2), cfg, vp_stage=None)
block = TransformerBlock(cfg, spec, post_layer_norm=True, pre_process=True, post_process=True).npu().to(torch.bfloat16)
for n,p in block.named_parameters():
    if "attn_sink" in n: p.data = p.data.float()
print(f"[shared] 2-layer block built: {sum(p.numel() for p in block.parameters())/1e9:.2f}B", flush=True)

# attention tensors per layer (both layers, since this is the 2-layer symmetric run)
SUFFIXES = ["wq_a","wq_b","wkv","wo_a","wo_b"]
def attn_names():
    names = {}
    for n,p in block.named_parameters():
        for suf in SUFFIXES:
            if n.endswith(f"self_attention.{suf}.weight"):
                names[n] = (p, suf)
    return names

params = dict(block.named_parameters())
init_names = attn_names()
print(f"[shared] attn tensors found: {len(init_names)} (expect 10 = 2 layers x 5)", flush=True)
attn_INIT = {n: params[n].detach().float().cpu().clone() for n in init_names}

# Freeze everything except the 10 attn tensors so AdamW optimizer states stay tiny
# (2-layer full-param AdamW OOMs at 8.84B on one 61GB chip — known memory wall).
# We only need the attn weights to train for the parameter-flow verification.
attn_param_set = set(init_names.keys())
for n, p in block.named_parameters():
    p.requires_grad_(n in attn_param_set)
trainable = [params[n] for n in init_names]
print(f"[shared] froze non-attn; training only {len(trainable)} attn tensors (AdamW states small)", flush=True)
opt = torch.optim.AdamW(trainable, lr=1e-3)
hs=(torch.randn(64,1,cfg.hidden_size,dtype=torch.bfloat16)*0.05).npu()
for step in range(3):
    opt.zero_grad()
    out=block(hidden_states=hs, attention_mask=None); o=out[0] if isinstance(out,(tuple,list)) else out
    loss=o.float().pow(2).mean(); loss.backward(); torch.npu.synchronize()
    opt.step(); torch.npu.synchronize()
    print(f"[shared] train step{step} loss={loss.item():.5f}", flush=True)

attn_TRAINED = {n: params[n].detach().float().cpu().clone() for n in init_names}
# per-tensor trained delta L2
for n in init_names:
    d = (attn_TRAINED[n]-attn_INIT[n]).norm().item()
    print(f"[shared] {n.split('.')[-2]:5s} L{n.split('.')[1]} trained_delta_L2={d:.3e}", flush=True)

# synth control: same per-tensor L2 as the trained delta, random direction
g = torch.Generator().manual_seed(777)
attn_SYNTH = {}
for n in init_names:
    real_d = attn_TRAINED[n]-attn_INIT[n]
    r = torch.randn(real_d.shape, generator=g)
    r = r / (r.norm()+1e-12) * real_d.norm()   # same L2 as the real delta
    attn_SYNTH[n] = (attn_INIT[n] + r)

meta = {"suffixes": SUFFIXES, "attn_names": list(init_names.keys())}
torch.save({"INIT": attn_INIT, "TRAINED": attn_TRAINED, "SYNTH": attn_SYNTH, "meta": meta},
           "/home/z00637938/workspace/task-dag-realdelta/shared_attn_2layer.pt")
print("[shared] saved shared_attn_2layer.pt (INIT / TRAINED / SYNTH control)", flush=True)
print("[shared] DONE — sglang side overwrites fab attn with INIT, then compares INIT vs TRAINED vs SYNTH rollouts", flush=True)
