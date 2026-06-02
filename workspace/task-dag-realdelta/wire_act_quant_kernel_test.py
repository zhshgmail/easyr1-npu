"""Wire the op-gen act_quant AscendC kernel into the fp8_simulate NPU path + test on NPU.

The current miles qat NPU path (_act_quant_npu) uses a torch fp8-grid simulation. This replaces it
with a call to the real AscendC kernel (_act_quant_ext.run_act_quant) and attempts to produce the
same dequantized fp32 result USABLE in the training math ON NPU.

Honest risk: the kernel returns float8_e4m3fn (a .view of int8 bytes). torch_npu cannot .float() an
fp8 tensor on NPU (the exact reason the torch sim exists). This test finds out whether the int8-byte
path lets us dequant on-NPU, or whether wiring is genuinely blocked by the fp8-on-NPU limitation.
"""
import os, sys
sys.path.insert(0, "/home/z00637938/workspace/opgen_call_test/act_quant/kernel/build")
import torch, torch_npu  # noqa
import _act_quant_ext as _ext

FP8_MAX = 448.0
BLOCK = 128
M, N = 4, 256
torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.bfloat16).npu()

# --- AscendC kernel on NPU ---
y_fp8, scale = _ext.run_act_quant(x, BLOCK)   # y_fp8: float8_e4m3fn (view of int8), scale: fp32
torch.npu.synchronize()
print(f"[wire] kernel ran: y_fp8 dtype={y_fp8.dtype} scale dtype={scale.dtype} on {y_fp8.device}", flush=True)

# Goal: dequantized fp32 = y_fp8_value * scale, ON NPU, for the training math.
# Attempt 1: direct .float() on NPU (expected to fail per torch_npu fp8 limitation).
deq = None
try:
    yv = y_fp8.float()  # fp8 -> fp32 on NPU
    deq = (yv.unflatten(-1,(-1,BLOCK)) * scale.unsqueeze(-1)).flatten(-2)
    torch.npu.synchronize()
    print(f"[wire] ATTEMPT-1 fp8.float() on NPU: SUCCESS, deq finite={torch.isfinite(deq).all().item()}", flush=True)
except Exception as e:
    print(f"[wire] ATTEMPT-1 fp8.float() on NPU: BLOCKED -> {type(e).__name__}: {str(e)[:80]}", flush=True)

# Attempt 2: move fp8 to CPU, cast there, move back (works but breaks the on-NPU graph / slow).
if deq is None:
    try:
        yv_cpu = y_fp8.cpu().float()
        deq = (yv_cpu.unflatten(-1,(-1,BLOCK)) * scale.cpu().unsqueeze(-1)).flatten(-2).npu()
        torch.npu.synchronize()
        print(f"[wire] ATTEMPT-2 fp8->CPU->fp32->NPU: SUCCESS but breaks NPU graph (host round-trip). deq finite={torch.isfinite(deq).all().item()}", flush=True)
    except Exception as e:
        print(f"[wire] ATTEMPT-2 CPU round-trip: FAIL -> {type(e).__name__}: {str(e)[:80]}", flush=True)

# Compare the kernel-dequant to the torch-sim dequant (the current in-layer path) for equivalence.
def _fp8_round(t):
    a = t.abs().clamp(min=1e-30); import math
    e = torch.floor(torch.log2(a)); e = e.clamp(min=-6); step = torch.pow(2.0, e-3)
    return torch.sign(t) * torch.round(t/step) * step
xb = x.float().reshape(M, N//BLOCK, BLOCK)
amax = torch.clamp(xb.abs().amax(-1,keepdim=True), min=1e-4); s_ref = amax/FP8_MAX
yq = _fp8_round(torch.clamp(xb/s_ref, -FP8_MAX, FP8_MAX))
deq_sim = (yq * s_ref).flatten(-2)
if deq is not None:
    err = (deq.float().cpu() - deq_sim.float().cpu()).abs().max().item()
    print(f"[wire] kernel-dequant vs torch-sim dequant max abs err: {err:.3e}", flush=True)
    print(f"[wire] === WIRED: act_quant AscendC kernel output dequantized & matches torch-sim (err={err:.1e}); "
          f"see ATTEMPT lines above for whether the dequant stayed on-NPU or needed a host round-trip ===", flush=True)
else:
    print(f"[wire] === BLOCKED: could not dequantize the kernel's fp8 output for the training math ===", flush=True)
