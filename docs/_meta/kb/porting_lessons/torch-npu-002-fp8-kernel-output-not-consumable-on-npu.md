---
id: torch-npu-002
date: 2026-06-02
layer: torch-npu
title: An op-gen fp8 (Float8_e4m3fn) kernel can be built+called on NPU but its output cannot be dequantized in the NPU training math — torch_npu lacks fp8 cast
trigger:
  - "wiring an AscendC fp8-output kernel (act_quant / fp8 quant) into a pytorch NPU training/forward path"
  - "kernel returns torch.float8_e4m3fn and the next op needs it as fp32 on NPU"
  - "RuntimeError: Float8_e4m3fn has not been supported (on .float()/.to(fp32)/arithmetic on NPU)"
symptom_in_wild:
  - "AscendC kernel run_act_quant(x, block) returns (y_fp8, scale); y_fp8.float() on NPU raises 'Float8_e4m3fn has not been supported'"
  - "a torch fp8-grid SIMULATION exists in the codebase specifically instead of calling the real kernel — and you don't know why until you try to wire the kernel"
  - "kernel verified bit-exact vs CPU truth in isolation, but cannot be slotted into the on-NPU numeric path"
root_cause: >
  torch_npu (as of CANN 8.5.2) has no fp8 (Float8_e4m3fn) compute/cast support on-device. An
  AscendC fp8 kernel produces e4m3 bytes (allocated int8, .view'd to float8_e4m3fn — metadata only).
  Calling the kernel works; reinterpreting works; but the FIRST real arithmetic/cast on the fp8
  tensor on NPU (e.g. dequant y_fp8 * scale, which needs y_fp8 as fp32) hits the unsupported-dtype
  wall. So a callable, accuracy-verified fp8 kernel is still not consumable inside an NPU forward.
mistake_pattern: "assuming 'kernel callable + accuracy-verified' implies 'kernel wireable into the NPU compute path'; the consume-side dtype support is a separate gate"
correction:
  - "Before claiming an fp8 op-gen kernel is 'wired' / 'integrated', test the CONSUME side on NPU: can the next op actually use its output? .float() the fp8 result on NPU — if it raises 'Float8_e4m3fn has not been supported', wiring is blocked regardless of how clean the kernel is."
  - "Three unblock paths: (a) torch_npu adds fp8->fp32 support; (b) the kernel returns a DEQUANTIZED fp32 value (or fp8-grid value in fp32, like the torch sim does) instead of fp8 bytes; (c) decode the e4m3 bytes to fp32 on NPU with integer/bit ops (the kernel itself forbids compute, but a separate small NPU decode kernel could)."
  - "Until then, a torch fp8-grid SIMULATION (per-block amax -> scale -> grid-round, all in fp32) is the correct in-layer path on NPU — NOT a shortcut, a necessity. Don't 'fix' it by calling the fp8 kernel; that re-hits the wall."
  - "CPU round-trip (y_fp8.cpu().float()...npu()) works but breaks the NPU graph + autograd + perf — not viable in a training forward."
evidence:
  - "workspace/task-dag-realdelta/wire_act_quant_kernel_test.py: ATTEMPT-1 fp8.float() on NPU -> RuntimeError 'Float8_e4m3fn has not been supported'; ATTEMPT-2 CPU round-trip works but breaks graph"
  - "kernel pybind (a5_ops .../act_quant/kernel/pybind11.cpp): writes int8 bytes, .view(at::kFloat8_e4m3fn), returns {y_fp8, s_out}"
  - "miles_qat_npu_patched.py: the torch fp8-grid sim _act_quant_npu exists precisely to avoid the NPU fp8 cast"
  - "call_opgen_act_quant.py: kernel IS bit-exact vs CPU truth (scale_err=0, fp8 match 100%) — callability+accuracy are fine; only the NPU consume-side is blocked"
applies_to:
  - "torch_npu @ CANN 8.5.2 on A3; any AscendC kernel whose output dtype is Float8_e4m3fn consumed in an on-NPU compute path"
verified_on:
  - "Ascend A3 NPU (Ascend910_9382), tlrescue container, CANN 8.5.2, 2026-06-02"
unverified_on:
  - "newer torch_npu / CANN that may add fp8 support — re-test the consume side when bumping"
deprecated_after: ""
---

# torch-npu-002 — fp8 op-gen kernel output is not consumable in the NPU compute path

## Why this matters

It is easy to (over)claim an op-gen fp8 kernel is "integrated" because the hard-looking parts pass:
it builds with bisheng, it's callable from pytorch (`_ext.run_act_quant(x_npu, block)`), and it's
bit-exact vs CPU truth. All true — and all insufficient. The kernel returns `float8_e4m3fn`, and the
moment the training forward tries to USE that output on NPU (dequant = `y_fp8 * scale`, which needs
`y_fp8` as fp32), torch_npu raises `Float8_e4m3fn has not been supported`. Callable + accurate ≠
wireable.

## The discriminator (run before claiming "wired")

```python
y_fp8, scale = _ext.run_act_quant(x_npu, block)   # works
y_fp8.float()                                      # <-- this is the real gate; raises on NPU
```

If `.float()` (or any arithmetic) on the fp8 tensor raises on NPU, the kernel is not wireable into
the NPU numeric path, no matter how clean it is.

## Why the torch simulation is correct, not a shortcut

The in-layer `_act_quant_npu` does the fp8-grid math entirely in fp32 (per-block amax → scale →
clamp → grid-round) and never materializes an fp8 tensor on NPU. That is the right design given the
torch_npu limitation — replacing it with a call to the fp8 kernel re-introduces the exact wall.

## Unblock paths (for whoever wires it later)

1. torch_npu adds fp8↔fp32 support on-device (upstream Ascend/pytorch).
2. The kernel returns a dequantized fp32 value (or fp8-grid value in fp32) instead of fp8 bytes —
   a kernel-output-contract change.
3. A separate small NPU kernel decodes the e4m3 int8 bytes → fp32 with integer/bit ops (the
   act_quant kernel itself forbids compute, but a companion decode kernel could).

Related: the same torch_npu fp8 gap is why miles MoE routing uses `fp8_simulate` (report §四.4).
