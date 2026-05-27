# Miles tilelang ops → NPU A3 drop-in dispatch plan

## Goal

End-state: `miles.train` (or any miles entry-point that imports
`miles_plugins.models.glm5.ops.{indexer, sparse_mla}`) runs on Ascend A3
NPU without manual code surgery, using our mlir-ascend kernels in place
of the GPU tilelang kernels.

## Contract surfaces (miles top-level interfaces)

`miles_plugins/models/glm5/ops/`:

| File | Function | Signature |
|---|---|---|
| `tilelang_indexer_fwd.py` | `indexer_fwd_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True)` | returns `logits [seq, seq_kv] fp32` |
| `tilelang_indexer_bwd.py` | `indexer_bwd_interface(index_q, weights, index_k, topk_indices, grad_scores)` | returns `(grad_q, grad_w, grad_k)` |
| `tilelang_sparse_mla_fwd.py` | `sparse_mla_fwd_interface(...)` | returns sparse MLA fwd outputs |
| `tilelang_sparse_mla_bwd.py` | `sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale, is_casual=True, return_kernel=False, delta=None)` | returns `(dq, dkv)` |

These are consumed by `glm5.py`:

* `indexer.IndexerFunction.{forward,backward}` → call `indexer_*_interface`
* `sparse_mla.SparseMLA` → calls `sparse_mla_fwd_interface` / `sparse_mla_bwd`

## NPU implementations (already in fork)

All in `examples/deepseek_v4/`:

* `example_sparse_mla_fwd_kernel.py` → `sparse_mla_fwd(...)` — production (5e-4 err)
* `example_sparse_mla_bwd_kernel.py` → `sparse_mla_bwd_main/preprocess/postprocess` — R-KA-13 E5 (cosine 0.93)
* `example_lighting_indexer_fwd_kernel.py` → `lighting_indexer_fwd(...)` — production bf16 (0 err)
* `example_lighting_indexer_bwd_kernel.py` → `lighting_indexer_bwd(...)` — SEQ=1 production (1e-5/0/4e-5)

## Dispatch mechanism

Two options:

### Option A: `miles_npu_dispatch/` add-on package

Drop a new package alongside miles that monkey-patches the 4 `_interface`
functions on `import` when `torch.npu.is_available()`:

```python
# miles_npu_dispatch/__init__.py
import torch
if hasattr(torch, "npu") and torch.npu.is_available():
    from . import indexer, sparse_mla
    import miles_plugins.models.glm5.ops.indexer as miles_indexer
    import miles_plugins.models.glm5.ops.sparse_mla as miles_sparse_mla
    miles_indexer.indexer_fwd_interface = indexer.npu_indexer_fwd_interface
    miles_indexer.indexer_bwd_interface = indexer.npu_indexer_bwd_interface
    miles_sparse_mla.sparse_mla_fwd_interface = sparse_mla.npu_sparse_mla_fwd_interface
    miles_sparse_mla.sparse_mla_bwd = sparse_mla.npu_sparse_mla_bwd
```

Pros: zero miles edits; `import miles_npu_dispatch` in miles startup.
Cons: fragile if miles ever uses `from .tilelang_indexer_fwd import indexer_fwd_interface` and binds the symbol locally before the patch runs.

### Option B: fork miles, conditional inside its `_interface`

Edit miles' 4 `tilelang_*.py` to detect NPU and dispatch:

```python
def indexer_fwd_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    if q.is_npu:
        from miles_npu_kernels import npu_indexer_fwd
        return npu_indexer_fwd(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits)
    # original GPU path follows
```

Pros: no monkey-patch fragility; symbol binding is irrelevant.
Cons: miles fork drift; need to upstream the change or maintain a patch.

**Recommended: Option B**, with the NPU kernels imported lazily so non-NPU envs aren't affected.

## Shape adapter responsibilities

Miles uses 3D `[seq, heads, dim]` for index_q and 2D `[seq_kv, dim]` for index_k.
Our `lighting_indexer_fwd` takes `[seq_len, heads, index_dim]` and `[seq_len_kv, index_dim]` — same layout.

Things to wire:
- `cu_seqlen_ks` / `cu_seqlen_ke` → our kernel currently doesn't honor causal mask; the integration shim in `example_miles_indexer_integration.py` already implements masked post-processing externally. Reuse that pattern.
- `clean_logits=True` (mask invalid positions to `-inf`) → wrap kernel output with the same mask logic.
- `topk_indices` on bwd → already a kernel arg.
- `grad_scores` shape `[seq, topk]` fp32 → matches kernel.

For `sparse_mla`:
- Kernel uses `[batch=1, seq, heads, dim+tail_dim]` layout. Miles' interface uses `[B, S, H, DQK]` — same.
- Indices are `[B, S, kv_group=1, topk]` int32 in our kernel; miles uses `[B, S, kv_group, topk]`.
- `lse` is `[B, S, H, 1]` in our kernel; miles squeezes the trailing 1.

## Validation plan

1. **Per-op smoke at miles' contract level**: synthesize miles-style inputs, call `npu_*_interface`, compare vs CPU autograd ref (same as `P2.miles_integration_smoke`).
2. **GLM-5 single-layer forward**: instantiate `miles_plugins.models.glm5.glm5.GLM5Layer` with NPU device, run `forward`, observe finite outputs.
3. **GLM-5 single-layer backward**: same + `loss.backward()`, check gradient finiteness + shape.
4. **GLM-5 1-iter optim step**: tiny optimizer.step(), watch for explosion.

## Open blockers / dependencies

- **R-KA-13** workaround in P1.4 is in fork branch `t33-sparse-mla-fwd-port-and-tdynamic`, NOT in PR #59. Once #59 merges, file P1.4 as a follow-up PR (skip while #59 still pending to avoid stacked PRs).
- **R-KA-14** (multi-block scatter NaN) — work around by calling `npu_indexer_bwd_interface` per-seq-position. Matches miles' `batched_indexer_bwd` natural loop.
- **R-KA-15** (atomic_addx4 all-zero src 6e37) — wrapper-side short-circuit when `grad_scores.abs().max() < 1e-30`.
- **No sglang on NPU yet** (image triton ABI skew, filed as #277). End-to-end `miles.train` driver requires sglang for rollout; that path is currently unavailable. For NPU testing we substitute a synthetic-prompt driver (replaces rollout side), feeds training loop directly. Pure GRPO training math doesn't need sglang inference if we mock the rollout output.

## Status

* All 4 NPU kernels in fork, 3 of them in upstream PR #59 (`MERGEABLE`, CI green, awaiting reviewer merge).
* P1.4 (bwd) workaround held back from PR #59 per maintainer scope guidance.
* Drop-in dispatch package: TODO — execute when PR #59 merges OR if user wants to ship the dispatch as a standalone artifact in `easyr1-npu` repo.
