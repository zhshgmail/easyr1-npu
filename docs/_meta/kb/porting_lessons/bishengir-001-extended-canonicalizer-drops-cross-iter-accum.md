---
id: bishengir-001
date: 2026-05-23
layer: bishengir
title: ExtendedCanonicalizer drops cross-iter softmax accumulators on DPS in-place vmul (R-KA-16)
trigger:
  - "sparse_mla_fwd NS≥2 NaN on Ascend A3"
  - "tilelang multi-stage softmax with online accumulator on NPU"
  - "scf.for iter_args silently disappearing across MLIR pass pipeline"
  - "bishengir-compile produces correct IR for NS=1 but NaN at NS≥2"
symptom_in_wild:
  - "miles `sparse_mla_fwd_interface` output全 NaN at NS=2/4/8 (NS=1 PASS)"
  - "tilelang code uses `num_stages>=2` + online softmax; rewriting to `num_stages=1` makes it PASS"
  - "`bishengir-opt --mlir-print-ir-after-all` shows iter_args count crashing through a single pass boundary"
  - "GPU reference (H100 / triton) on same shape PASS; only NPU bishengir path NaN"
root_cause: >
  bishengir's `ExtendedCanonicalizer` pass (93-line Python shell) re-uses upstream
  MLIR SCF's `RemoveUnusedIterArgs` canonicalization. On a DPS-style in-place
  `vmul(acc_l, correction, acc_l)`, the canonicalization treats `acc_l` as a sink
  whose RHS is not consumed downstream, then deletes the iter_arg that carries
  `acc_l`/`acc_m` across loop iterations. The next iteration starts with
  uninitialized accumulator -> propagates NaN.

  This is structurally a class of bug where DPS write-in-place semantics fool a
  liveness-based canonicalization that was designed for SSA. The 93-line
  ExtendedCanonicalizer wrapper is not the culprit; the upstream pass it calls
  is.
mistake_pattern: "DPS in-place RHS marked dead by SSA-liveness canonicalization, dropped before lowering"
correction:
  - "Reproducer: `workspace/T32_tilelang_rescue/repro_rka16.py` (self-contained, NS=1/2/4/8)"
  - "Bisect recipe: `bishengir-opt --mlir-print-ir-after-all rka16_ns4.npuir 2>&1 > passes.txt`, then `grep -nE 'scf.for.*iter_args' passes.txt | awk -F'iter_args' '{print NR, NF-1}'` to count iter_args across passes; the line where count drops marks the culprit. See memory `bishengir_iter_args_bisect_recipe.md`."
  - "Workaround on consumer side: pin `num_stages=1` in tilelang so no cross-iter accumulator is needed. miles fork `npu-tilelang-dispatch` does this with a `# R-KA-16 mitigation` comment block."
  - "Upstream fix direction (3 patches proposed in issue #251 body): (a) conservative-keep rule — when DPS in-place and RHS contains loop-induction-var, retain iter_arg; (b) opt out of the canonicalization for `scf.for` carrying yields with DPS writes; (c) widen the liveness analyzer to recognize DPS sinks."
  - "Do NOT chase this through tilelang or miles fixes — it's a compiler bug. Open at `Ascend/AscendNPU-IR`."
evidence:
  - "Issue: https://gitcode.com/Ascend/AscendNPU-IR/issues/251 (107-line Chinese diagnostic report by zhshgmail, 2026-05-23)"
  - "Pass identified: `ExtendedCanonicalizer` at line 10801 of the 311-pass dump"
  - "Full IR dump: `workspace/T32_tilelang_rescue/rka16_ns4_passes.txt` (2.9 MB) + `rka16_ns4_pass_index.txt` (311 lines)"
  - "miles workaround commit: `sparse_mla.py:71-87` sets `num_stages=1`; `_sparse_mla_fwd_kernel.py:137-145` adds `correction_expanded` cleanup"
  - "GPU H100 triton on identical shape NS=8: PASS, max abs err 3e-4 vs reference"
---

# bishengir-001 — ExtendedCanonicalizer drops cross-iter softmax accumulators

## Why this matters

This single compiler bug is the only thing blocking miles `sparse_mla_fwd` real-shape numerical correctness on Ascend A3. Without it fixed:
- miles `_real_shape_smoke.py` reports 3/4 ops PASS, 1/4 fails with NaN
- miles `_e2e_megatron_step.py MILES_E2E_SHAPE=real` produces NaN gradients
- multi-step RL training on real shape will diverge after step 1
With `num_stages=1` workaround:
- 4/4 ops compile and run
- output values are bit-clean enough for forward-pass demo
- BUT: throughput penalty unmeasured; production needs `num_stages>=2`

## Why I almost wasted weeks before finding it

Initial debugging attempts assumed the bug was in:
1. tilelang accumulator init (it's not — init code is correct)
2. miles sparse_mla shape derivation (it's not — same code path NS=1 works)
3. CANN op library (it's not — same MLIR text produces NaN at any CANN version)

The 5-min bisect recipe (`bishengir-opt --mlir-print-ir-after-all` + grep iter_args count) collapsed the search from "any of 311 passes" to "this one pass". The recipe is now memorized; every future "compiler IR mutates incorrectly" bug should start with it.

## Concrete repro

```bash
cd workspace/T32_tilelang_rescue
python repro_rka16.py --ns 1   # PASS
python repro_rka16.py --ns 2   # NaN
python repro_rka16.py --ns 4   # NaN
python repro_rka16.py --ns 8   # NaN
```

The Python repro dumps a `.npuir`; `dump_rka16_ir.py` is the monkey-patch driver that re-ran bishengir-opt with `--mlir-print-ir-after-all` to produce the 311-pass dump.
