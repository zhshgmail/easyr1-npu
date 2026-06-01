# Integration demo: V4 attention_core with NPU-native dispatch (no tilelang import).
# Mirrors miles ops/attention_core.py::sparse_attn_tilelang but routes to CANN NSA on NPU.
import torch, torch_npu

def sparse_attn_tilelang_OR_npu(q, kv, attn_sink, topk_idxs, sm_scale=None):
    if q.is_npu:
        # NPU path: CANN-native NSA (bypasses the tilelang CUDA kernel entirely)
        T1, N1, Dqk = q.shape; Dv = 128
        if sm_scale is None: sm_scale = 1.0/(Dqk**0.5)
        sbc = topk_idxs.shape[-1]; sbs = 64
        T2 = kv.shape[0]; N2 = kv.shape[1] if kv.dim()==3 else 1
        k = kv[..., :Dqk].contiguous().view(T2,N2,Dqk)
        v = kv[..., Dqk:Dqk+Dv].contiguous().view(T2,N2,Dv)
        out = torch_npu.npu_nsa_select_attention(q.contiguous(),k,v,topk_idxs.int().contiguous(),
                sm_scale,N1,sbs,sbc,actual_seq_qlen=[T1],actual_seq_kvlen=[T2])
        return out[0]
    else:
        from .kernel import tilelang_sparse_mla_fwd as f  # lazy: only on CUDA
        return f.sparse_mqa_fwd_interface(q,kv,attn_sink,topk_idxs,sm_scale=sm_scale)[0]

if __name__ == "__main__":
    torch.npu.set_device(0); torch.manual_seed(0)
    T1,N1,Dqk,Dv,T2,N2,sbc = 128,4,192,128,1024,1,16
    q=torch.randn(T1,N1,Dqk,dtype=torch.float16,device="npu:0")
    kv=torch.randn(T2,N2,Dqk+Dv,dtype=torch.float16,device="npu:0")
    topk=torch.arange(sbc,dtype=torch.int32,device="npu:0").view(1,1,sbc).expand(T1,N2,sbc).contiguous()
    o=sparse_attn_tilelang_OR_npu(q,kv,None,topk,None); torch.npu.synchronize()
    print(f"[V4 attn_core NPU-integrated] o={tuple(o.shape)} finite={torch.isfinite(o.float()).all().item()} — V4 sparse_attn runs on NPU, no tilelang")
