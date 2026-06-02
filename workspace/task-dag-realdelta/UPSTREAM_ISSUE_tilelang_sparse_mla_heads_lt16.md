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
