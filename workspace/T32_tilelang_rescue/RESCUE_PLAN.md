# T32.3 — tilelang-ascend elementwise rescue (issue #996)

## Target bug

tile-ai/tilelang-ascend issue **#996**: `examples/elementwise/elementwise_add.py`
with `M=N=32`, `block_M=block_N=4` gives "half wrong" output vs `M=N=1024,
block_M=128, block_N=256`.

## Root cause (from author's own annotation in #996)

> "底层做搬运的时候，对于 2-dim tile 是按照每行 32B 进行搬运。如果小于这个对齐要求，
>  底层会访问到错误的地址。"

- Each AIV (`VEC_NUM=2`) gets `(block_M // 2, block_N) = (2, 4)` fp32 tile
- Per-row bytes = `4 cols * 4 byte fp32 = 16 B` < 32 B DMA alignment requirement
- DMA reads wrong addresses → "half wrong" result

## Rescue strategy (a5_ops-pattern-derived)

a5_ops `triton-ascend-elementwise.md` "方案 1 (推荐)": **contiguous + 1-D flat
access**. Translate to tilelang-ascend tile-DSL:

1. **Flatten**: view A/B/C as 1-D length-`M*N`. Now the smallest tile unit is
   a contiguous slice — no per-row alignment trap.
2. **1-D grid**: `T.Kernel(grid_size, is_npu=True)` where
   `grid_size = ceil(M*N / BLOCK_SIZE)`.
3. **BLOCK_SIZE per a5_ops 推荐**: 1024-2048 fp32 elements = 4096-8192 B,
   safely ≥32 B per DMA chunk.
4. **vid (per-AIV) split**: each AIV processes `BLOCK_SIZE // VEC_NUM`
   contiguous elements — still ≥32 B because 512 fp32 = 2048 B.
5. **Mask for tail**: last grid program may have `< BLOCK_SIZE` valid elements;
   guard with `if start + i < n_elements` style or pad to multiple of BLOCK_SIZE.

## Plan

### Step A — establish baseline (must work before any rescue)
- Run upstream `elementwise_add.py --m 1024 --n 1024` → expect "Kernel Output Match!"
- This validates the env is sane

### Step B — reproduce #996
- Run upstream variant with `--m 32 --n 32 --block_m 4 --block_n 4`
  (need parser-extended copy; original example doesn't expose block_m/n)
- Expect assertion failure at `torch.testing.assert_close`

### Step C — author rescue kernel
- Write `elementwise_add_flat.py` in our workspace
- Use 1-D flat layout per "方案 1"
- Validate at the same 32×32 / 4×4 shape — must PASS
- Also validate at 1024×1024 — must still PASS (regression)
- Validate at non-power-of-2 shape (e.g. 100×100) — must PASS (tail handling)

### Step D — package as upstream PR
- Diff our `elementwise_add_flat.py` against original
- Open issue/PR on tile-ai/tilelang-ascend referencing #996
- Or, if patch is non-invasive, propose as `examples/elementwise/elementwise_add_v2.py`
  with comment explaining 32B-alignment trap

### Step E — adversarial-audit our own claim (a5_ops M1-M5)
- M1 gate-fitness: does "Match!" output actually mean rescue worked, or are
  we comparing wrong shape?
- M2 anti-cycle: did we sneak the bug into the test (e.g. comparing to NPU
  ref instead of CPU ref)?
- M4 cross-shape regression: PASS only at 32×32 → suspect overfitting
- M5 tool-use signature: every claim must cite container + log path + datums

## Container

`tilelang-install` on A3 host, base `quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11`,
NPU bind set, --network=host. Wheel `/wheels/tilelang-0.1.1.10+linux.cann900-*.whl`.

## Pinning

- tilelang-ascend HEAD `b925cbe` (cloned 2026-05-17)
- a5_ops HEAD `ef38ffde` (sync 2026-05-16, per sister-projects.md)
- CANN 9.0.0, torch 2.9.0+cpu, torch_npu 2.9.0, tilelang 0.1.1.10
