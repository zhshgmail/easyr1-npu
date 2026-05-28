# Miles-DSv4-RL-on-NPU — Upstream Patch Map

> Goal: enumerate every upstream project that needs to update or be patched so miles (DeepSeek-V4-Flash RL post-training) runs on Ascend A3 NPU end-to-end. Maintained as the picture sharpens; do not let this drift from current state.

## 全景图 — Architectural layers

```
┌───────────────────────────────────────────────────────────────┐
│ miles (radixark/miles)                                        │  Application: RL post-training framework
│  • DSAMLASelfAttention (glm5.py)                              │  Forked from slime, vendored Megatron-LM
│  • 4 ops: lighting_indexer_fwd/bwd, sparse_mla_fwd/bwd        │
│  • _e2e_megatron_step.py (manual cuda→npu monkey-patches)     │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ uses
┌───────────────────────────────────────────────────────────────┐
│ Megatron-LM-miles (radixark/Megatron-LM miles-main)           │  Vendored Megatron at Mcore 0.16.0rc0
│  • MultiLatentAttention, AbsorbedMLASelfAttention             │  Forked from NVIDIA/Megatron-LM
│  • experimental_attention_variant framework                   │
│  • te_general_gemm references in moe_utils.py (under wraps)   │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ monkey-patched by
┌───────────────────────────────────────────────────────────────┐
│ MindSpeed (Ascend/MindSpeed core_r0.16.0)                     │  Megatron→NPU adaptor
│  • megatron_adaptor.py runs ~430 register_patch() calls       │  ACTIVE BRANCH: core_r0.16.0
│  • TransformerEngineBasicFeature, DSA feature, ~65 features   │
│  • Ships npu_lightning_indexer.cpp + triton_indexer_bf16.py   │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ uses
┌───────────────────────────────────────────────────────────────┐
│ triton-ascend (Ascend/triton-ascend v3.2.0)                   │  Triton DSL on Ascend
│  • Tracks Triton v3.2.0 submodule, custom backend             │
│  • Ships triton/backends/compiler.py and ascend/ backend      │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ uses
┌───────────────────────────────────────────────────────────────┐
│ tilelang-mlir-ascend (tile-ai/tilelang-mlir-ascend) +         │  Tilelang DSL + IR compiler
│ AscendNPU-IR (Ascend/AscendNPU-IR)                            │
│  • bishengir-compile (open) → bisheng clang (Huawei-only)     │
└───────────────────────────────────────────────────────────────┘
                            ▲
                            │ runs on
┌───────────────────────────────────────────────────────────────┐
│ Ascend A3 NPU (dav-c220 / 910C)                               │
│  • CANN 8.5.0, npu-smi 26.0.rc1                               │
└───────────────────────────────────────────────────────────────┘
```

Adjacent (not in critical path for compile/training but in deployment):
* **vllm-ascend** (`Ascend/vllm-ascend`) — inference; pulls `xgrammar` which transitively pins mainline `triton`.
* **MindSpeed-LLM** (`Ascend/MindSpeed-LLM`) — LLM-specific pretraining; depends on MindSpeed Core; 47 model families. Not needed if miles + Mcore-0.16 covers the model.
* **MindSpeed-RL** (`Ascend/MindSpeed-RL`) — RL on top; miles replaces it.

---

## Required upstream changes (by repo, what / why / status)

### 1. `tile-ai/tilelang-mlir-ascend` — CheckUBBudget diagnostic

| | |
|---|---|
| **Why** | Bishengir produces opaque "ub overflow" errors after 30s; need early-fail at 192 KB UB budget exhaustion to save bisect time |
| **What** | PR #80: new `CheckUBBudget` pass after `LowerOpaqueBlock`, scope-restricted to `{local, local.fragment}`, raises only at ≥2× UB cap; emits per-allocation breakdown + suggested `block_M` |
| **Status** | PR #80 MERGEABLE, REVIEW_REQUIRED; CI all green (incl. test PASS 24m15s). Awaiting tile-ai maintainer. |
| **Branch** | `zhshgmail/tilelang-mlir-ascend npuir-check-ub-budget` |
| **Commit** | `df7431e` (3-commit chain `daea72f` → `d2d1871` → `df7431e`) |

### 2. `Ascend/AscendNPU-IR` — R-KA-16 ExtendedCanonicalizer drop

| | |
|---|---|
| **Why** | bishengir HIVM ExtendedCanonicalizer (delegating to upstream MLIR `RemoveUnusedIterArgs`) misreads DPS in-place `vmul(acc_l, correction, acc_l)` / `vadd(arg, new_max, acc_m)` after OneShotBufferize, dropping cross-iter softmax accumulators → sparse_mla_fwd NaN at NS≥2 |
| **What** | Issue #251 with 311-pass dump bisect (mutation point line 10801), before/after IR diff, 3 patch direction suggestions. Huawei compiler team to write the C++ fix; the fix is outside bishengir's repo boundary (upstream MLIR SCF canonicalizer) so we don't PR it directly. |
| **Status** | Issue #251 update posted (comment `1.73358592e+08`); Huawei compiler team owns the C++ patch. |
| **PR** | n/a (issue path) |

### 3. `radixark/miles` + `radixark/Megatron-LM` — `_npu/` subpackage + dispatch

| | |
|---|---|
| **Why** | miles glm5 ops on NPU need tilelang implementations; manual monkey-patch in `_e2e_megatron_step.py` is a temporary bridge |
| **What** | Two coupled PRs (currently bundled in one branch on the miles fork): |
| | • miles `_npu/` subpackage with 4 tilelang kernels + dispatcher (commit `d03db2c`, 13 files, 1767 LOC) |
| | • Megatron-LM-miles 8-line guard in `moe_utils.py except ImportError` to bind `te_general_gemm = None` (commit `6f3209b`) |
| **Status** | Audit-clean, NOT yet opened. Gated on T13.B/C validation through MindSpeed core_r0.16.0 (per user direction "走 MindSpeed 适配路线" 2026-05-28 20:56). |
| **Branch** | `zhshgmail/miles npu-tilelang-ops` |
| **PR body** | `/tmp/miles_pr_body.md` |

### 4. `Ascend/triton-ascend` — packaging conflict with mainline `triton`

| | |
|---|---|
| **Why** | triton-ascend 3.2.0 and mainline triton 3.6.0 share `triton/backends/compiler.py` path but their forks have diverged (mainline has `Language`, triton-ascend has `AttrsDescriptor`). If both are installed, the later one wins and the former's amd/nvidia backends crash with `ImportError: cannot import name 'Language'`. |
| **What** | Issue at `Ascend/triton-ascend` to: (a) declare `Provides-Dist: triton` so pip treats triton-ascend as the triton implementation, OR (b) declare `Conflicts: triton`, OR (c) document "uninstall triton first". Minimal repro included. |
| **Status** | NOT yet filed (T13.D candidate). Workaround for our work: `pip uninstall triton + pip install --force-reinstall triton-ascend`. Memory `feedback_triton_vs_triton_ascend_packaging_conflict.md` records the recipe. |
| **Triton 3.6.0 needed?** | **No** for miles training. miles's triton kernels use the stable `@triton.jit / tl.*` DSL that triton-ascend 3.2.0 implements. Mainline triton 3.6 is pulled in only by xgrammar (via vllm) for inference-time grammar paths, dead code in training. |

### 5. `Ascend/MindSpeed` core_r0.16.0 — gaps surfaced during T13

| | |
|---|---|
| **Why** | T13.A confirmed adaptor boots cleanly. T13.B/C may surface additional Mcore 0.16-specific patches that Huawei MindSpeed team hasn't landed yet. |
| **What** | TBD — pending T13.B miles e2e PASS log. If gaps found, contribute precise commits to `core_r0.16.0` branch (e.g. cherry-pick our R-KA-16 diagnostic, or add a missing `register_patch` for a Megatron-LM-miles symbol). |
| **Status** | TODO after T13.B / T13.C. |
| **Branch** | `Ascend/MindSpeed core_r0.16.0` is the upstream; fork on demand |

### 6. Container-level: tlrescue image triton install hygiene

| | |
|---|---|
| **Why** | Pre-built tlrescue image shipped both `triton 3.6.0` and `triton-ascend 3.2.0`, breaking import. User direction: NPU-only branches should have only triton-ascend. |
| **What** | (a) Patch the image recipe to drop mainline triton install (or to pin xgrammar without its triton dep); OR (b) provide a post-image `setup_env.sh` that does the uninstall step. |
| **Status** | TODO; not blocking T13.B if workaround applied. |

---

## Cardinality of patches needed for end-to-end

If everything in the picture lands:

| # | Upstream | Patch type | Outstanding work |
|---|---|---|---|
| 1 | tile-ai/tilelang-mlir-ascend | open PR (small Python pass) | Reviewer merge |
| 2 | Ascend/AscendNPU-IR | filed issue, Huawei writes C++ | External |
| 3 | radixark/miles | PR open after T13.B/C | Open after MindSpeed validation |
| 4 | radixark/Megatron-LM (vendored in miles) | 8-line guard PR bundled with #3 | Same |
| 5 | Ascend/triton-ascend | new packaging issue + maybe PR | TODO |
| 6 | Ascend/MindSpeed (core_r0.16.0) | TBD by T13.B/C | TODO |
| 7 | tlrescue container | image recipe | TODO |

**Total at present**: 4 strictly required upstream changes (#1 in flight, #2 with Huawei, #3+#4 prepared, #5 simple), plus 2-3 likely arising from T13 validation. None of them is blocking us today; we have a working manual-monkey-patch fallback that gets us through real-shape e2e.

---

## Maintenance protocol

* Every time T13.B / T13.C / T14 yields a new upstream gap, add a new row in §"Required upstream changes" with the same fields.
* When a PR opens, lands, or closes, update the corresponding row's Status and link to the PR URL.
* If a discovery invalidates an earlier row (e.g. "MindSpeed already does this"), strikethrough the row but leave it for history.
* Mirror status into the project ROADMAP T12/T13/T14 cells.
