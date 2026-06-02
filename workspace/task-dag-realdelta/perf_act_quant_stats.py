"""act_quant AscendC-vs-torch perf in the SAME stats format as sinkhorn: mean/median/min/max/n.
Symmetric same-wrapper, both on NPU, ratio = torch-fp32-sim / AscendC. Sweep representative shapes
so the distribution (incl. the large-shape regression) is captured honestly in min/max."""
import sys, time, statistics
sys.path.insert(0, "/home/z00637938/workspace/opgen_call_test/act_quant/kernel/build")
import torch, torch_npu
import _act_quant_ext as _ext

FP8_MAX = 448.0; AMAX_FLOOR = 1e-4; BLOCK = 128
torch.manual_seed(0)

def _fp8_round(t):
    a = t.abs().clamp(min=1e-30); e = torch.floor(torch.log2(a)).clamp(min=-6); step = torch.pow(2.0, e-3)
    return torch.sign(t)*torch.round(t/step)*step
def torch_aq(x, b):
    N=x.shape[-1]; nb=N//b; xb=x.float().reshape(*x.shape[:-1],nb,b)
    am=torch.clamp(xb.abs().amax(-1,keepdim=True),min=AMAX_FLOOR); s=am/FP8_MAX
    return _fp8_round(torch.clamp(xb/s,-FP8_MAX,FP8_MAX)).reshape(*x.shape), s.squeeze(-1)
def asc_aq(x, b): return _ext.run_act_quant(x, b)
def meas(fn,x,b,w=5,r=20):
    for _ in range(w): fn(x,b)
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(r): fn(x,b)
    torch.npu.synchronize(); return (time.perf_counter()-t0)/r*1e6

# 6 representative cases (match sinkhorn's n=6 cardinality), spanning small->large like a real sweep
CASES = [(4,256),(8,512),(32,1024),(32,2048),(64,4096),(128,4096)]
ratios=[]
for M,N in CASES:
    x=torch.randn(M,N,dtype=torch.bfloat16).npu()
    rf=meas(torch_aq,x,BLOCK); rc=meas(asc_aq,x,BLOCK); ra=rf/rc
    ratios.append(ra); print(f"[stat] ({M},{N}): torch={rf:.1f}us AscendC={rc:.1f}us ratio={ra:.2f}x", flush=True)
ratios.sort()
mean=statistics.mean(ratios); med=statistics.median(ratios)
print(f"[stat] === act_quant: {mean:.2f}x mean (median {med:.2f}, min {min(ratios):.2f}, max {max(ratios):.2f}, n={len(ratios)}) ===", flush=True)
print(f"[stat] min<1 means SLOWER than torch at that shape; distribution spans both regimes — mean alone hides it.", flush=True)
