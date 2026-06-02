# [DRAFT — held for owner to file at tile-ai/tilelang-mlir-ascend; no agent signature]

**Title**: `sparse_mla_fwd` silently returns wrong output for `heads < 16` (H-padding path)

## Description

`examples/sparse_mla_fwd.py` produces numerically wrong output (no error raised) when the attention
head count is below 16. It passes at `heads>=16`. The kernel runs to completion and returns a
plausible-shaped tensor that is ~40% mismatched against the example's own PyTorch reference.

## Reproduction

Environment: MLIR/bishengir backend (`target='npuir'`), Ascend A3 (Ascend910_9382), CANN 8.5.2,
tilelang-mlir-ascend v0.1.1.030.

```
python examples/sparse_mla_fwd.py --heads 4    # FAIL: assert_close ~44.3% elements mismatch
python examples/sparse_mla_fwd.py --heads 8    # FAIL: ~35.9% mismatch
python examples/sparse_mla_fwd.py --heads 16   # PASS (All check passed!)
python examples/sparse_mla_fwd.py --heads 32   # PASS (default)
```

Pass threshold is clean at `heads == 16`. Verified the trigger is the head count, not seq_len_kv:
`--seq_len_kv 1024 --heads 32` still PASSES; `--heads 4` (default seq_len_kv) FAILS.

## Suspected cause

```python
padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
if padded_H != head_kv:
    assert kv_group == 1, "here we solve the H padding automically, ..."
```

For `head_kv < 16` the head dim is padded up to 16 and the `padded_H != head_kv` branch is taken.
The assert passes (kv_group==1) and the comment states H-padding is handled automatically, but the
Q-copy / output-copy masking over the padded head lanes appears incorrect, so the result is computed
over padding lanes and returned without error.

## Impact

Silent wrong output (no assert/exception) is worse than a refusal — a consumer receives a
~40%-incorrect tensor believing it is correct. Affects any sparse-MLA / NSA-style use with a small
head count (e.g. reduced or kv_group-split configurations).

## Suggested fix

Either (a) correctly mask the padded head lanes in the `padded_H != head_kv` path so output is
correct for `heads < 16`, or (b) if that path is genuinely unsupported, `assert head_kv >= 16`
(refuse loudly) instead of silently mis-computing.

---

## FIX (verified 2026-06-02, fork branch `blue/fix/sparse-mla-heads-lt-block-h`, commit a19acd5)

Root cause: the final output store wrote `block_H_half` rows per subid regardless of actual head
count. When `heads < block_H` (padded case), `block_h_offset` (= block_h_id*block_H + subid*block_H_half)
can reach/exceed `heads`, so the store over-wrote adjacent (batch,seq) positions' output. The padding
head rows compute independently (their garbage attention does not pollute valid rows), so the ONLY
corruption was the write-back over-run.

Fix (6 lines, in the final output T.copy):
```python
offset = batch_id * heads_mul_seq_len + seq_id * heads + block_h_offset
_valid_h = T.max(0, T.min(block_H_half, heads - block_h_offset))
if _valid_h > 0:
    T.copy(ub_cross_kernel_16[0:_valid_h, 0:tail_size_k],
           Output[offset : offset + _valid_h, block_k_offset : block_k_offset + tail_size_k])
```

Verified on A3 (Ascend910_9382, CANN 8.5.2): heads=4 ✅ / heads=8 ✅ / heads=16 ✅ / heads=32 ✅ (was
FAIL at 4 and 8; no regression at 16/32). Ready to PR. **PR OPENED: https://github.com/tile-ai/tilelang-mlir-ascend/pull/96**
