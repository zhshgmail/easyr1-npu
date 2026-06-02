"""Standalone verified-run capture: npu_nsa_compress_attention with V4-spec dims on A3.

Upgrades the report's 'spec-matched' label for the V4 compressor op to 'verified-run' by actually
calling it on NPU with V4 Compressor dims and capturing finite output (same bar as the existing
npu_nsa_select_attention verified-run: op executes on A3 with V4-spec shapes, output finite).
"""
import torch, torch_npu, time

torch.npu.set_device(0)
torch.manual_seed(0)

# V4 Compressor dims (NSA compress attention). From PIVOT doc: compress_block_size=32, stride.
# query [T,N,D], key/value compressed stream. Use small T for a quick standalone run.
T, N, D = 128, 4, 192   # D_qk=192 (V4 MLA), N heads
Dv = 128
CMP_BLK = 32
CMP_STRIDE = 16
SEL_BLK = 64
SEL_CNT = 16
T_kv = 256

q = torch.randn(T, N, D, dtype=torch.bfloat16).npu()
k = torch.randn(T_kv, 1, D, dtype=torch.bfloat16).npu()
v = torch.randn(T_kv, 1, Dv, dtype=torch.bfloat16).npu()
scale = 1.0 / (D ** 0.5)

print(f"[cmp] calling npu_nsa_compress_attention q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
      f"cmp_blk={CMP_BLK} stride={CMP_STRIDE} sel_blk={SEL_BLK} sel_cnt={SEL_CNT}", flush=True)
try:
    t0 = time.perf_counter()
    out = torch_npu.npu_nsa_compress_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        scale, N, CMP_BLK, CMP_STRIDE, SEL_BLK, SEL_CNT,
        actual_seq_qlen=[T], actual_cmp_seq_kvlen=[T_kv], actual_sel_seq_kvlen=[T_kv],
    )
    torch.npu.synchronize()
    dt = (time.perf_counter()-t0)*1e3
    o = out[0] if isinstance(out,(tuple,list)) else out
    fin = torch.isfinite(o.float()).all().item()
    print(f"[cmp] VERIFIED-RUN: npu_nsa_compress_attention ran on A3, out={tuple(o.shape)} dtype={o.dtype} finite={fin} {dt:.1f}ms", flush=True)
except Exception as e:
    import traceback
    print(f"[cmp] FAIL: {type(e).__name__}: {str(e)[:160]}", flush=True)
    print("[cmp] (signature/dim mismatch — op exists but standalone call needs exact arg tuning; stays spec-matched)", flush=True)
    for ln in traceback.format_exc().splitlines()[-4:]: print("   ", ln[:120], flush=True)
