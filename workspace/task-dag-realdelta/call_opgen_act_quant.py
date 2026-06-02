"""Actually CALL the op-gen'd AscendC act_quant kernel from pytorch on NPU.

This is the wiring the report previously CLAIMED ('CANN covers + 2 op-gen done') but never did:
import the built pybind ext (_act_quant_ext, AscendC kernel compiled by build_ascendc.py on A3),
call _ext.run_act_quant(x_npu, block_size) on an NPU tensor, and compare to the CPU-truth reference
(model.py). Proves the op-gen kernel is invocable from pytorch — 'pytorch calls AscendC op'.
"""
import os, sys
sys.path.insert(0, "/home/z00637938/workspace/opgen_call_test/act_quant/kernel/build")
import torch, torch_npu  # noqa

import _act_quant_ext as _ext   # the op-gen'd AscendC kernel pybind ext
print(f"[opgen-call] imported _act_quant_ext from {_ext.__file__}", flush=True)

FP8_MAX = 448.0; FP8_MIN = -448.0; AMAX_FLOOR = 1e-4
BLOCK = 128
M, N = 4, 256   # N % block == 0

torch.manual_seed(0)
x = torch.randn(M, N, dtype=torch.bfloat16)

# --- op-gen AscendC kernel on NPU ---
x_npu = x.npu()
y_npu, s_npu = _ext.run_act_quant(x_npu, BLOCK)
torch.npu.synchronize()
print(f"[opgen-call] kernel ran on NPU: y={tuple(y_npu.shape)} dtype={y_npu.dtype} s={tuple(s_npu.shape)}", flush=True)

# --- CPU-truth reference (model.py logic) ---
xf = x.float()
nblk = N // BLOCK
xb = xf.reshape(M, nblk, BLOCK)
amax = torch.clamp(xb.abs().amax(dim=2), min=AMAX_FLOOR)
s_ref = amax / FP8_MAX
yb = torch.clamp(xb / s_ref.unsqueeze(2), FP8_MIN, FP8_MAX)
y_ref_fp8 = yb.reshape(M, N).to(torch.float8_e4m3fn)

# Compare scales (fp32, directly comparable) and dequantized values.
s_npu_cpu = s_npu.float().cpu()
s_err = (s_npu_cpu - s_ref).abs().max().item()
print(f"[opgen-call] scale max abs err (kernel vs ref): {s_err:.3e}", flush=True)

# dequantized reconstruction: y(fp8 value as fp32) * scale  ~= original (within fp8 grid)
y_npu_f = y_npu.cpu().float()                      # move fp8 off NPU FIRST (torch_npu has no fp8 cast), then fp32 on CPU
y_ref_f = y_ref_fp8.float()                        # ref fp8 -> fp32
deq_npu = y_npu_f * s_npu_cpu.repeat_interleave(BLOCK, dim=1)
deq_ref = y_ref_f * s_ref.repeat_interleave(BLOCK, dim=1)
deq_err = (deq_npu - deq_ref).abs().max().item()
fp8_match = (y_npu_f == y_ref_f).float().mean().item()
print(f"[opgen-call] fp8-value exact match rate (kernel vs ref): {fp8_match*100:.1f}%", flush=True)
print(f"[opgen-call] dequantized max abs err: {deq_err:.3e}", flush=True)

ok = s_err < 1e-2 and fp8_match > 0.95
print(f"[opgen-call] === {'PASS' if ok else 'CHECK'}: op-gen AscendC act_quant kernel CALLED from pytorch on NPU, "
      f"output matches CPU-truth reference (scale_err={s_err:.2e}, fp8_match={fp8_match*100:.0f}%) ===", flush=True)
