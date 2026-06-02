"""Symmetric perf: op-gen AscendC act_quant kernel vs the torch fp32-sim it would replace, on A3 NPU.

The earlier N/A was a harness-env failure (phase_o5 ran where torch_npu was absent). Here we measure
on A3 (torch_npu present), SAME-WRAPPER both sides (the discipline that corrected sinkhorn 51x->5.34x):
each side is the identical timed callable — warmup then N repeats, time.perf_counter bracketed by
torch.npu.synchronize. Reference = the torch fp32 fp8-grid sim `_act_quant_npu` (what the running
layer actually uses, and what the AscendC kernel would REPLACE — its .to(float8) ref can't run on NPU
anyway). Candidate = the AscendC `_ext.run_act_quant`. Reports ratio = ref_us / cand_us.
"""
import os, sys, time
sys.path.insert(0, "/home/z00637938/workspace/opgen_call_test/act_quant/kernel/build")
import torch, torch_npu
import _act_quant_ext as _ext

FP8_MAX = 448.0; AMAX_FLOOR = 1e-4
BLOCK = 128
torch.manual_seed(0)

def _fp8_e4m3_round(t):
    a = t.abs().clamp(min=1e-30)
    e = torch.floor(torch.log2(a)).clamp(min=-6)
    step = torch.pow(2.0, e - 3)
    return torch.sign(t) * torch.round(t / step) * step

def torch_act_quant_npu(x, block):          # the pytorch version the layer uses (fp32 sim, NPU-runnable)
    N = x.shape[-1]; nblk = N // block
    xb = x.float().reshape(*x.shape[:-1], nblk, block)
    amax = torch.clamp(xb.abs().amax(dim=-1, keepdim=True), min=AMAX_FLOOR)
    scale = amax / FP8_MAX
    yq = _fp8_e4m3_round(torch.clamp(xb / scale, -FP8_MAX, FP8_MAX))
    return yq.reshape(*x.shape), scale.squeeze(-1)

def ascendc_act_quant(x, block):            # the op-gen AscendC kernel (candidate replacement)
    return _ext.run_act_quant(x, block)

def measure(fn, x, block, warmup=5, repeats=20):
    for _ in range(warmup): fn(x, block)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats): fn(x, block)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / repeats * 1e6   # us/call

for (M, N) in [(4, 256), (32, 2048), (128, 4096)]:
    x = torch.randn(M, N, dtype=torch.bfloat16).npu()
    ref_us = measure(torch_act_quant_npu, x, BLOCK)
    cand_us = measure(ascendc_act_quant, x, BLOCK)
    ratio = ref_us / cand_us if cand_us > 0 else float('nan')
    print(f"[perf] shape=({M},{N}): torch-sim(ref)={ref_us:.1f}us  AscendC(cand)={cand_us:.1f}us  ratio={ratio:.2f}x", flush=True)

print("[perf] method: SYMMETRIC same-wrapper (identical measure() callable, warmup=5/repeats=20, "
      "perf_counter + npu.synchronize, both on NPU). ratio = torch-fp32-sim / AscendC-kernel.", flush=True)
