---
id: sglang-004
date: 2026-06-01
layer: sglang
title: DeepseekV4ForCausalLM on Ascend NPU exposes ~8 missing AscendAttnBackend V4 hooks + V4 KV-pool NotImplementedError — no-op stubs only suffice for 1-layer PoC
trigger:
  - "running sglang trunk DeepseekV4ForCausalLM with attention_backend=ascend on NPU"
  - "AttributeError: 'AscendAttnBackend' has no attribute forward_c4_indexer / forward_core_compressor / store_cache / init_forward_metadata_indexer / forward_compress / _maybe_upgrade_forward_metadata"
  - "DeepSeekV4TokenToKVPool.get_key_buffer raises NotImplementedError on NPU"
  - "V4 forward passes compress_ratio kwarg that NPU forward_extend/forward_decode does not accept"
  - "porting DeepSeek-V4-Flash inference to Ascend A3 via sglang.Engine"
symptom_in_wild:
  - "generate() on V4 + device=npu crashes with AttributeError on one of the 6 AscendAttnBackend V4 hooks"
  - "TypeError: forward_extend() got an unexpected keyword argument 'compress_ratio'"
  - "NotImplementedError from DeepSeekV4TokenToKVPool.get_key_buffer during forward_extend cache read"
  - "Someone declares 'V4 runs on NPU' after stubbing the hooks — but it is a 1-layer max_tokens=2 PoC, not production"
  - "Claim that 'all V4 ops need hand-written AscendC kernels' (this is the early misjudgment; core ops are CANN-native)"
root_cause: >
  sglang trunk ships the V4 model class (deepseek_v4.py, EntryClass=[DeepseekV4ForCausalLM])
  but the NPU backend (sgl-kernel-npu / AscendAttnBackend) has not implemented the
  V4-specific attention hooks and KV-pool methods that the V4 forward path calls.
  V4 adds c4/c128 multi-level compressed KV, a lightning sparse indexer, and an NSA-style
  compressor — none wired on NPU. The CUDA backend has them; the Ascend backend does not.
mistake_pattern: "in-tree model class exists ⇒ assume backend supports it; stub-to-green a forward path and call it production"
correction:
  - "Treat the ~8 stub sites as an UPSTREAM GAP INVENTORY, not a deliverable. Each maps to a concrete sgl-kernel-npu / sglang-NPU-adapter hole."
  - "Honestly classify every fix: 4 are production-ready native-op swaps; 1 is a correctly-kept fp32 path; ~8 are PoC-only no-op-stub/short-circuit walkarounds."
  - "Production needs sgl-kernel-npu to REALLY implement forward_c4_indexer, forward_core_compressor, store_cache, init_forward_metadata_indexer, forward_compress, _maybe_upgrade_forward_metadata, and DeepSeekV4TokenToKVPool.get_key_buffer for the c4/c128 paged paths."
  - "Do NOT generalize to 'all V4 ops need hand-written kernels'. The training-side core ops are CANN-native (npu_nsa_select_attention / npu_lightning_indexer / npu_nsa_compress_attention / npu_mla_prolog_v3 / npu_rms_norm), verified on A3. Only hash-coding sinkhorn + act_quant needed AscendC op-gen."
  - "When PRing the safe parts: ship the 4 native swaps + the 6 no-op stubs (default-safe) + **kwargs back-compat, and file ONE clean issue for the real c4/compressor/indexer/KV-pool implementation."
evidence:
  - "workspace/v4_attempt_2026_06_01/README.md — 14-gap table + 2026-06-01 PASS log + RL LOOP PASS"
  - "docs/_meta/DSV4_NPU_PORTING_REPORT.md §2 — walkaround-vs-production classification (authoritative)"
  - "PASS log: sgl 0.5.12.post2.dev434+gb13d3d18c, Engine init 27.7s, generate 0.9s, output '醺报废' (gibberish, shape-correct)"
  - "Frozen patched sources: _ascend_backend_PASS.py (7 hook stubs + V4 dense short-circuits), _elementwise_PASS.py, _moe_PASS.py, _gemm_PASS.py, _sglang_v4_minimal_PASS.py"
  - "Native swaps verified: npu_rms_norm 0.000e+00 bit-exact; npu_clipped_swiglu ≤0.77% rel (bf16 noise, clamp bit-exact); native_op_snapshots/NATIVE_OP_SWAP_FINDINGS.md"
  - "RoPE NOT swapped: npu_apply_rotary_pos_emb is rotate-half ≠ V4 interleaved-complex, err≈4.22 — fp32 torch complex-mul kept (more accurate, not a compromise)"
  - "Engine ctor unblock: max_total_tokens=65536 + swa_full_tokens_ratio=0.5 (PrefillAdder SWA-budget NO_TOKEN before this)"
applies_to: "sglang trunk DeepseekV4ForCausalLM @ 2026-06-01 (sgl 0.5.12.post2.dev434+gb13d3d18c), attention_backend=ascend, sgl-kernel-npu @ 2026-06-01"
verified_on: ["Ascend A3 NPU, tlrescue container, bf16, 1-layer reduced DSv4-Flash fab, 2026-06-01"]
unverified_on:
  - "full 43-layer DeepSeek-V4-Flash (only 1-layer reduced fab verified)"
  - "numerical correctness (PoC output is shape-correct gibberish: reduced layers + random init + 8 stubs)"
  - "real DSv4-Flash trained weights"
deprecated_after: ""
---

# sglang-004 — V4 AscendAttnBackend hook gaps + V4 KV-pool NotImplementedError

## Why this matters

`DeepseekV4ForCausalLM` exists in sglang trunk (deepseek_v4.py, ~2259 lines, `EntryClass=[DeepseekV4ForCausalLM]`),
so it is tempting to read "model class is in-tree" as "V4 runs on NPU". It does not. The model class is
fine; the **NPU backend** (`AscendAttnBackend` in sgl-kernel-npu) has not implemented the V4-specific
attention hooks or the V4 paged KV-pool reads. V4 is the first DeepSeek arch to add c4/c128 multi-level
compressed KV, a lightning sparse indexer, and an NSA-style compressor — and none of those are wired on
Ascend. Getting `generate()` to return on NPU required stubbing them out; that only suffices for a
1-layer, `max_tokens=2`, short-sequence PoC. **Shipping these stubs as "V4 on NPU" would be exactly the
overclaim this KB exists to prevent.**

## What almost went wrong

Two near-misses worth recording:

1. **"Model class exists ⇒ backend supports it."** The 1-layer fab loaded with 0 missing weights, KV pool
   (full=4096 swa=256 c4=1024 c128=32 c4_state=8 c128_state=128) allocated, Engine init OK in ~22-32s.
   Everything looked ready — but `generate()` then crashed on a chain of `AttributeError`s and a
   `NotImplementedError`, one per missing V4 backend hook. The "it loaded, therefore it works" inference
   is wrong here.

2. **"All V4 ops need hand-written AscendC kernels."** This was the early misjudgment the owner explicitly
   corrected. The training-side core ops are CANN-native and verified on A3 (see KEY CORRECTION below).
   The inference-side gaps are missing *backend wiring/hooks*, not missing *algorithms* — sgl-kernel-npu
   needs to implement the hooks, not invent kernels from zero.

## Clear separation: walkaround vs production

### ⚠️ WALKAROUND (PoC-only — must NOT ship; needs real sgl-kernel-npu impl)

These ~8 are no-op-stub / short-circuit hacks. They let a 1-layer short-sequence forward execute (shape-correct),
nothing more. Production requires sgl-kernel-npu to genuinely implement them for the c4/c128/sparse paths.

| Site | Current PoC walkaround | Real upstream fix |
|---|---|---|
| `AscendAttnBackend._maybe_upgrade_forward_metadata` missing | no-op stub | sgl-kernel-npu: V4 metadata upgrade |
| `AscendAttnBackend.forward_c4_indexer` missing | no-op stub | sgl-kernel-npu: c4 indexer |
| `AscendAttnBackend.forward_core_compressor` missing | no-op stub | sgl-kernel-npu: NSA-style compressor |
| `AscendAttnBackend.store_cache` missing | no-op stub | sgl-kernel-npu: c4/c128 cache store |
| `AscendAttnBackend.init_forward_metadata_indexer` missing | no-op stub | sgl-kernel-npu: indexer metadata init |
| `AscendAttnBackend.forward_compress` missing | no-op stub | sgl-kernel-npu: compress path |
| `DeepSeekV4TokenToKVPool.get_key_buffer` raises `NotImplementedError` | V4 dense short-circuit in `forward_extend`/`forward_decode` | sgl-kernel-npu: V4 paged KV-pool `get_key_buffer` for c4/c128 |
| `fused_k_norm_rope_flashmla` writes FP8-packed bytes to paged kvcache | PoC skips scatter (1-layer max_tokens=2 short enough) | sgl-kernel-npu: FP8 packed kvcache scatter |
| NPU `aclnnIndex` doesn't support complex64 | index real domain, view-as-complex after | file NPU complex64 aclnnIndex issue |

Frozen in `_ascend_backend_PASS.py` (the 6 hook stubs + the two V4 dense short-circuits in
`forward_extend`/`forward_decode`). These are the body of the single clean upstream issue:
"V4 (`DeepseekV4ForCausalLM`) on `device=npu` requires N missing AscendAttnBackend V4 hooks", each gap a
concrete line-number + PoC-workaround evidence link.

### ✅ PRODUCTION-READY native-op swaps (measured-equivalent, e2e-gated, PR-shaped)

These four are not walkarounds — they replace a CUDA-JIT path with a measured-equivalent native `torch_npu`
op (or a safe back-compat shim), and the patch form is native-first + torch-fallback guard, suitable for a
direct upstream PR.

| Swap | native op | measured vs torch ref |
|---|---|---|
| RMSNorm part of `fused_q_norm_rope` / `fused_k_norm_rope_flashmla` | `npu_rms_norm(x, ones, eps)` | **0.000e+00 bit-exact** |
| `silu_and_mul_clamp` (`moe.silu_and_mul_clamp` JIT CUDA) | `npu_clipped_swiglu(alpha=1.0, bias=0.0, interleaved=False, limit=lim)` | ≤0.77% rel (bf16 noise); clamp bit-exact |
| `gemm.linear_bf16_fp32` uses `torch.mm(out_dtype=fp32)` (NPU unsupported) | drop the kwarg, `.float()` cast | equivalent |
| V4 dispatch passes `compress_ratio` kwarg NPU `forward_extend`/`forward_decode` rejects | absorb via `**kwargs` | back-compat, safe |

e2e gate: after swapping in the two native ops, the whole V4 RL loop re-ran PASS (EXIT=0,
`distinct_vs_step0=5/5 step_to_step_changes=5/5`) and the native patches were confirmed to execute on the
forward path (deepseek_v4.py:423 / deepseek_v2.py:364 call-sites hit unconditionally, not bypassed by an
OPT flag). Snapshots: `native_op_snapshots/_elementwise_NATIVE.py` / `_moe_NATIVE.py`.

### ✅ CORRECTLY-KEPT fp32 RoPE (a deliberate keep, not a walkaround)

The RoPE complex-multiply part is **kept in fp32 torch on purpose**. `npu_apply_rotary_pos_emb` /
`npu_rotary_mul` implement the **rotate-half** convention, which is **not** V4's **interleaved-complex**
RoPE — swapping it in produces err≈4.22 (or 561002 unsupported). fp32 torch complex-mul is *more* accurate
than the bf16 native kernel would be, so keeping it is the better choice, not a compromise. A prior note
claiming `npu_apply_rotary_pos_emb` "verified runs" was wrong: it runs but is numerically incorrect for V4.

## KEY CORRECTION to honor (do not regress)

The inference-side gaps above are missing **backend hooks**, not missing kernels. On the **training side**,
the V4 core ops are **CANN-native and verified on Ascend A3**:

- sparse-MLA fwd/bwd → `npu_nsa_select_attention` (D_qk=192 / D_v=128, select_block=64, count=16)
- C4 indexer → `npu_lightning_indexer` / `npu_sparse_lightning_indexer_grad_kl_loss`
- compressor → `npu_nsa_compress_attention`
- MLA prep → `npu_mla_prolog_v3`
- rms_norm → `npu_rms_norm` (bit-exact)

Only **hash-coding sinkhorn** and **act_quant (fp8)** have no CANN-native equivalent and required AscendC
op-gen (both precision-verified). So: do **not** write "all V4 ops need hand-written kernels" — that was
the early misjudgment. Verify `dir(torch_npu)` for nsa/mla/sparse/indexer before assuming a port is needed.

## How to know it's this gap (vs your own bug)

1. Does the 1-layer fab load with 0 missing weights and allocate the KV pool (full/swa/c4/c128)? If yes,
   it's not a weight-format problem.
2. Does `generate()` crash with `AttributeError` on `forward_c4_indexer` / `forward_core_compressor` /
   `store_cache` / `init_forward_metadata_indexer` / `forward_compress` / `_maybe_upgrade_forward_metadata`?
   That's a missing AscendAttnBackend V4 hook.
3. `NotImplementedError` from `DeepSeekV4TokenToKVPool.get_key_buffer`? That's the V4 KV-pool gap.
4. `TypeError ... unexpected keyword argument 'compress_ratio'`? That's the back-compat kwarg gap (#10).
5. If you hit a `PrefillAdder` `NO_TOKEN` admission stall before any forward at all, that's the SWA-budget
   ctor issue, not a V4 hook — set `max_total_tokens=65536` + `swa_full_tokens_ratio=0.5` to unblock.

## Honesty boundary (the reduced-layer basis)

The verified basis is a **1-layer reduced fab** with `max_tokens=2`, bf16, on A3. Output is **shape-correct
gibberish** (reduced layers + random init + ~8 stubs + 2 dense short-circuits replacing the sparse
compressor/indexer). This is **not** full-43-layer, **not** numerically correct, **not** real DSv4-Flash
weights. The RL loop PASS (rollout→weight-sync→re-rollout, 5/5 steps change output) used a **synthetic**
weight delta (plumbing proven), not a real miles training gradient. State all of this when reporting.

## Verification matrix (before claiming any V4-on-NPU inference milestone)

- [ ] 1-layer fab Engine init + 0 missing weights + KV pool (full/swa/c4/c128) allocated
- [ ] `generate()` returns non-empty (shape-correct) — record it is gibberish from reduced fab, not correctness
- [ ] every walkaround tagged in the source with the upstream owner (sgl-kernel-npu hook / NPU aclnn issue)
- [ ] the 4 native swaps each measured against torch ref (bit-exact / bf16-noise band), e2e-gated on RL loop
- [ ] RoPE stays fp32 torch complex-mul (do NOT regress to npu_apply_rotary_pos_emb rotate-half)
- [ ] single clean sgl-project issue drafted with line-number citations for the ~8 real-impl gaps
- [ ] NOT claimed as production / numerically-correct / full-model
