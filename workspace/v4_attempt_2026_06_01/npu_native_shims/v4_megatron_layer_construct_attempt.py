import torch, torch_npu, os
torch.npu.set_device(0)
os.environ.setdefault("MASTER_ADDR","127.0.0.1"); os.environ.setdefault("MASTER_PORT","29515")
os.environ.setdefault("RANK","0"); os.environ.setdefault("WORLD_SIZE","1")
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend="hccl", rank=0, world_size=1)
from megatron.core import parallel_state as ps
ps.initialize_model_parallel(tensor_model_parallel_size=1)
from megatron.core.transformer.transformer_config import TransformerConfig

# Reduced DeepSeekV4 config (mirrors mbridge _build_config dsv4 fields)
cfg = TransformerConfig(
    num_layers=1, hidden_size=512, num_attention_heads=4, kv_channels=128,
    tensor_model_parallel_size=1, sequence_parallel=False,
    add_bias_linear=False, gated_linear_unit=True, bias_activation_fusion=False,
    layernorm_epsilon=1e-6,
)
# dsv4 experimental-attention-variant fields
cfg.experimental_attention_variant = "dsv4"
for k, v in dict(
    q_lora_rank=256, kv_lora_rank=128, qk_pos_emb_head_dim=64, qk_head_dim=128, v_head_dim=128,
    dsv4_o_groups=8, dsv4_o_lora_rank=256, dsv4_window_size=128, dsv4_compress_ratios=None,
    dsa_indexer_n_heads=4, dsa_indexer_head_dim=128, dsa_indexer_topk=16,
    dsv4_hc_mult=4, dsv4_hc_sinkhorn_iters=20, dsv4_hc_eps=1e-6, dsv4_compress_rope_theta=160000,
    dsv4_swiglu_limit=0.0, dsv4_n_hash_layers=3,
).items():
    setattr(cfg, k, v)
print("config built OK")
try:
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import get_experimental_attention_variant_module_spec as getspec
    spec = getspec(cfg)
    print("dsv4 attention spec OK:", type(spec).__name__)
except Exception as e:
    import traceback; print("spec FAIL:", type(e).__name__, str(e)[:160])
