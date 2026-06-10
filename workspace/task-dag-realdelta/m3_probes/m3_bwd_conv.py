import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
fwd=torch_npu.npu_nsa_select_attention; grad=torch_npu.npu_nsa_select_attention_grad
# Config: select ALL blocks => dense attention over full KV (so torch ref is exact & simple)
Nq=4;Nkv=1;D_qk=192;D_v=128;sbs=64;sbc=16;S_kv=1024  # sbc*sbs = 1024 = S_kv => all blocks selected
Tq=64;Tkv=S_kv
q=torch.randn(Tq,Nq,D_qk,dtype=torch.bfloat16,device=dev)
k=torch.randn(Tkv,Nkv,D_qk,dtype=torch.bfloat16,device=dev)
v=torch.randn(Tkv,Nkv,D_v,dtype=torch.bfloat16,device=dev)
# topk = all 16 blocks (0..15) for every query => dense
topk=torch.arange(sbc,dtype=torch.int32,device=dev).view(1,1,sbc).expand(Tq,Nq,sbc).contiguous()
scale=1.0/(D_qk**0.5)
o,smax,ssum=fwd(q,k,v,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])

# --- torch reference: dense MQA (Nkv=1 broadcast to Nq heads), autograd ---
qf=q.float().detach().requires_grad_(True)
kf=k.float().detach().requires_grad_(True)
vf=v.float().detach().requires_grad_(True)
# scores [Tq,Nq,Tkv] = scale * q . k  (k broadcast over heads: k is [Tkv,1,D])
sc = scale * torch.einsum('tnd,sd->tns', qf, kf[:,0,:])  # [Tq,Nq,Tkv]
P = torch.softmax(sc, dim=-1)
oref = torch.einsum('tns,sd->tnd', P, vf[:,0,:])  # [Tq,Nq,D_v]
print("fwd o vs torch-dense-ref maxabs:", (o.float()-oref).abs().max().item())
g = torch.randn_like(oref)
oref.backward(g)
dq_ref, dk_ref, dv_ref = qf.grad.clone(), kf.grad.clone(), vf.grad.clone()

def cmp(name, native, ref):
    native=native.float(); ref=ref.float()
    rel=( (native-ref).norm()/(ref.norm()+1e-9) ).item()
    cos=torch.nn.functional.cosine_similarity(native.flatten(), ref.flatten(), dim=0).item()
    print(f"  {name}: rel={rel:.4f} cos={cos:.4f} native_norm={native.norm():.3f} ref_norm={ref.norm():.3f}")

# Try variant 1: feed smax/ssum AS-IS (8-wide)
print("=== variant A: smax/ssum as-is (8-wide) ===")
try:
    dq,dk,dv=grad(g.to(torch.bfloat16),q,k,v,o,smax,ssum,topk,scale,Nq,sbs,sbc,actual_seq_qlen=[Tq],actual_seq_kvlen=[Tkv])
    cmp("dq",dq,dq_ref); cmp("dk",dk,dk_ref); cmp("dv",dv,dv_ref)
except Exception as e: print("  ERR",str(e)[:200])
