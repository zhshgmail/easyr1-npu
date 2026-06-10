import torch, torch_npu
torch.manual_seed(0); dev="npu:0"
fwd=torch_npu.npu_lightning_indexer; grad=torch_npu.npu_lightning_indexer_grad
B=1;S=128;Nq=4;D=128
q=torch.randn(B,S,Nq,D,dtype=torch.bfloat16,device=dev)
k=torch.randn(B,S,1,D,dtype=torch.bfloat16,device=dev)
w=torch.randn(B,S,Nq,dtype=torch.bfloat16,device=dev)
# fwd: returns (sparse_indices, values?) ; use return_value=True to get values for grad
idx,val=fwd(q,k,w,layout_query="BSND",layout_key="BSND",sparse_count=S,sparse_mode=3,return_value=True)
print("fwd: idx",tuple(idx.shape),idx.dtype,"val",tuple(val.shape),val.dtype)
# the indexer 'value' = weighted indexer logits. dy = grad wrt val
dy=torch.randn_like(val)
print("=== run indexer_grad ===")
try:
    res=grad(q,k,dy,idx,w,layout="BSND",sparse_mode=3)
    print("returns",len(res),"tensors:",[tuple(t.shape) if torch.is_tensor(t) else t for t in res])
    for i,t in enumerate(res):
        if torch.is_tensor(t): print(f"  d[{i}] finite={torch.isfinite(t.float()).all().item()} norm={t.float().norm():.3f}")
except Exception as e: print("ERR",str(e)[:250])
# Build torch ref for the indexer to check if native grad matches.
# lightning indexer (from sglang/DSv4): score[b,s,n,j] = w[b,s,n] * (q[b,s,n] . k[b,j,0]); per-row over j (causal mode3)
qf=q.float().detach().requires_grad_(True); kf=k.float().detach().requires_grad_(True); wf=w.float().detach().requires_grad_(True)
sc=torch.einsum('bsnd,bjod->bsnj', qf, kf)  # [B,S,Nq,S] (o=1 broadcast)
sc=(wf.unsqueeze(-1)*sc)  # weight per (b,s,n)
# causal mask (mode3 rightdown): j<=s
mask=torch.tril(torch.ones(S,S,device=dev,dtype=torch.bool))
scm=sc.masked_fill(~mask.view(1,1,1,S).expand(B,1,Nq,S).bool() if False else ~mask[None,:,None,:], float('-inf'))
# the 'value' output is these scores (gathered by topk). Hard to match exactly without selection.
print("(torch-ref exact match needs the indexer's precise value semantics; gradcheck-style finite-diff next if native ran)")
