# RETRACTED 2026-06-02 — NOT a real bug. Do NOT file.

My earlier "fp8_lighting_indexer fails at h<32" finding was a **TEST-HARNESS error, not a tilelang bug**.

Root cause of the false finding: `examples/fp8_lighting_indexer.py` `__main__` HARDCODES
`B=2, M=2048, H=32, K=64, N=4096, BLOCK_SIZE_N=64` as literals, **ignoring `--h`**. So `--h 4`
compiled the kernel for h=4 but built the test data + reference with H=32 → the comparison was between
a heads≠32 kernel and heads=32 data = meaningless mismatch (24.7% / 99%, shape-dependent — the tell).

When the harness is fixed to honor `--h` (use args.h for the test tensors + reference too), the
indexer **PASSES at h=4/8/16/32** (exit 0, no assert failure). So there is no indexer head-count
kernel bug.

Lesson: I swept `--h` without verifying the example's `__main__` actually propagates `--h` to BOTH
the kernel AND the data/reference. sparse_mla_fwd does (its fix is real); fp8_lighting_indexer does
NOT — and I should have checked that before reporting a "bug". The only real artifact here is the
example harness inconsistency (hardcoded shapes), which is a minor example-quality nit, not a kernel
correctness bug.
