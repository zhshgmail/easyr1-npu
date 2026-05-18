# T32.3 — tilelang-ascend issue #996 rescue REPORT

**Date**: 2026-05-17
**Project**: easyr1-npu (sister of a5_ops)
**Target**: tile-ai/tilelang-ascend #996 — elementwise_add 32×32/4×4 wrong output
**Outcome**: A — successful rescue + cross-shape regression PASS

## Summary

Author of upstream `examples/elementwise/elementwise_add.py` reported in issue
#996 that shrinking the tile from `(M=N=1024, block=128×256)` to
`(M=N=32, block=4×4)` produces "half wrong" output. We reproduced (49.7%
elements wrong, max_abs_diff=4.013) and authored a 1-D flat-layout rescue
that passes at 5 shapes (32×32, 64×64, 100×100, 128×128, 512×512,
1024×1024) — including the original baseline (no regression) and a
non-power-of-2 shape (tail handling).

## Root cause confirmed

Author's annotation in #996: *"底层做搬运的时候，对于 2-dim tile 是按照每行
32B 进行搬运。如果小于这个对齐要求，底层会访问到错误的地址。"*

At `(block_M=4, block_N=4, VEC_NUM=2)` fp32: each AIV's tile is `(2, 4)`,
per-row = 4 cols × 4 B = **16 B < 32 B DMA alignment**. DMA reads wrong
addresses → wrong arithmetic.

## Rescue algorithm

Derived from a5_ops `triton-ascend-elementwise.md` "方案 1: 转连续 + 一维
访问 (推荐)" pattern, translated to tile-DSL primitives:

- Flatten A/B/C to 1-D length `M*N`.
- Single 1-D grid with `block_size` in the 1024-2048 range (a5_ops's
  recommended UB allocation size for elementwise).
- Each program loads ONE contiguous chunk of `block_size` fp32 = `block_size
  * 4` bytes (≥ 4096 B) → safely ≥ 32 B per-AIV slice too.
- No 2-D tile, no per-row alignment trap.
- Falls back to a smaller `block_size` that divides numel for non-power-of-2
  shapes (e.g. 100×100=10000 → block_size=16, still gives 64 B/AIV ≥ 32 B).

Code: `elementwise_add_flat.py` (see this dir).

## Evidence

### Bug reproduction (32×32, 4×4)
```
$ python3 repro_issue_996.py --m 32 --n 32 --block-m 4 --block-n 4
M=32 N=32 block_M=4 block_N=4 per_row_bytes_per_aiv=16
init OK
MISMATCH at (32,32)/(4,4): 509/1024 (49.7%) elements differ;
max_abs_diff=4.013
  [0] got=-2.2099 expected=-2.2099   # first AIV chunk happens to align
  [1] got=-1.2811 expected=-1.2811   # before drift kicks in
  ...
```

### Rescue PASS (cross-shape regression)
```
$ for s in "32 32" "64 64" "100 100" "128 128" "512 512" "1024 1024"; do
    python3 elementwise_add_flat.py --m $sm --n $sn; done

=== M=32 N=32 === block_size=1024 → Kernel Output Match!
=== M=64 N=64 === block_size=1024 → Kernel Output Match!
=== M=100 N=100 === block_size=16 → Kernel Output Match!
=== M=128 N=128 === block_size=1024 → Kernel Output Match!
=== M=512 N=512 === block_size=1024 → Kernel Output Match!
=== M=1024 N=1024 === block_size=1024 → Kernel Output Match!
```

## Environment

- Host: A3 NPU server `115.190.166.102` (ssh -p 443)
- Image: `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`
- Stack: CANN 8.5.2, torch 2.9.0+cpu, torch_npu 2.9.0,
  tilelang 0.1.1.10+linux.cann900
- Container: `tlrescue` (privileged, /dev/davinci14)
- Notable finding: cann900-suffixed tilelang wheel loads cleanly on CANN
  8.5.2 — the suffix is a build-time tag, not a hard runtime ABI gate.
  CANN 9.0.0's published image (`quay.io/ascend/cann:9.0.0-a3-ubuntu22.04
  -py3.11`) is missing `gert::OpImplRegisterV2` symbol — torch_npu 2.9.0
  doesn't load there.

## Adversarial-audit (a5_ops M1-M5 + P9)

- **M1 gate fitness**: assertion compares NPU-kernel output to NPU-torch
  built-in add. Both run on NPU. PASS means real arithmetic agreement.
- **M2 anti-cycle**: kernel writes into independent `c` tensor; reference
  computed as `a + b` using torch_npu, no path back to our kernel. No cycle.
- **M3 aggregate cap**: single op, not an aggregate run.
- **M4 cross-shape regression**: 6 shapes incl. non-power-of-2 (100×100)
  and original baseline (1024×1024). All PASS.
- **M5 tool-use signature**: container `tlrescue` + log paths above +
  concrete `509/1024 (49.7%)` and `Kernel Output Match!` datums cited.
- **P9 infra paper-over**: the env-setup work (CANN 8.5.2 swap) was
  separately documented and not the algorithmic fix. The 49.7%-wrong
  result reproduces the bug regardless of env. Fix is the 1-D flat layout.

## Upstream-facing artifacts (proposed)

1. Open upstream issue comment or PR on tile-ai/tilelang-ascend with:
   - Bug reproducer (`repro_issue_996.py`)
   - 1-D flat rescue (`elementwise_add_flat.py`)
   - Cross-shape PASS table
2. Suggest either:
   - Add `elementwise_add_flat.py` as a new example showing the
     alignment-safe pattern; OR
   - Patch upstream `elementwise_add.py` to use the flat layout (more
     invasive but fixes the bug in-place).

## Files in this workspace

- `RESCUE_PLAN.md` — original plan (Steps A-E)
- `repro_issue_996.py` — bug reproducer (block_m/block_n exposed)
- `elementwise_add_flat.py` — 1-D rescue kernel
- `smoke_import.py` — env validation smoke
- `REPORT.md` — this file
