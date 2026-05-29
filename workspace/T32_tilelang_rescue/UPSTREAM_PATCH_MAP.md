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

### 3. `radixark/miles` — `_npu/` subpackage + dispatch

| | |
|---|---|
| **Why** | miles glm5 ops on NPU need tilelang implementations; dispatch hook makes them activate when `q.is_npu`. |
| **What** | miles `_npu/` subpackage with 4 tilelang kernels + dispatcher (commit `d03db2c`, 13 files, 1767 LOC). |
| **Status** | **PR open**: `radixark/miles#1246` (2026-05-28). MERGEABLE, REVIEW_REQUIRED. |
| **PR URL** | https://github.com/radixark/miles/pull/1246 |
| **Branch on fork** | `zhshgmail/miles npu-tilelang-ops` commit `d03db2c` |
| **te_general_gemm sub-patch (`6f3209b` on `Megatron-LM-miles`)** | **WITHDRAWN as redundant** — T13.A confirmed MindSpeed core_r0.16.0 already binds `te_general_gemm = None` when TE is absent. The 8-line guard is no longer needed; the local branch stays as a fallback for the no-MindSpeed path. |

### 4. `triton-lang/triton-ascend` — packaging conflict with mainline `triton`

| | |
|---|---|
| **Why** | triton-ascend 3.2.0/3.2.1 and mainline triton 3.6.0 share `triton/backends/compiler.py` path but their forks have diverged (mainline has `Language`, triton-ascend has `AttrsDescriptor`). If both are installed, the later one wins and the former's amd/nvidia backends crash with `ImportError: cannot import name 'Language'`. |
| **What** | Issue with full repro + 4 fix-direction options (Provides-Dist, Conflicts, namespace packaging, install-guide). |
| **Status** | **Issue open** (2026-05-29). |
| **Issue URL** | https://github.com/triton-lang/triton-ascend/issues/306 |
| **Repo move** | The project moved from `gitcode.com/Ascend/triton-ascend` (now archived) to **`github.com/triton-lang/triton-ascend`** (canonical, under upstream Triton org). All future contributions land at the github home. |
| **gc CLI dead-end (memo)** | Filing the issue via `gc CLI`/raw API was blocked: both returned `400 Bad Request` with `body不能为空` regardless of which body field name was tried. The repo's `Bug-Report` YAML template enforced 5 required textareas at the server level; only the gitcode web UI could satisfy them. Since the canonical home is now github, this dead-end is no longer relevant for new issues. |
| **Triton 3.6.0 needed?** | **No** for miles training. miles's triton kernels use the stable `@triton.jit / tl.*` DSL that triton-ascend 3.2.0+ implements. Mainline triton 3.6 is pulled in only by xgrammar (via vllm) for inference-time grammar paths, dead code in training. |
| **Adjacent context** | The README announces a planned Triton 3.5 upgrade for 2026 (current 3.2.1 tracks Triton 3.2). The packaging conflict is orthogonal to which mainline version is tracked — even at 3.5, the two `compiler.py` files will continue to diverge on per-vendor symbols. |

### 5. `Ascend/MindSpeed` core_r0.16.0 — `apex.transformer.functional.fused_apply_rotary_pos_emb_thd` shim

| | |
|---|---|
| **Why** | T13.B (2026-05-28) confirmed MindSpeed `core_r0.16.0`'s `patch_features()` covers cuda→npu, RNG, Stream, TE — but does NOT install a fallback for `apex.transformer.functional.fused_apply_rotary_pos_emb_thd`. miles `glm5.py:fuse_rope` imports it directly. |
| **What** | T13.C: New module-level `_fused_apply_rotary_pos_emb_thd_fallback` (38-line self-contained pure-torch implementation) + 1 `pm.register_patch` line in `apex_adaptation`. Total +41 / -1 lines, only `mindspeed/features_manager/megatron_basic/requirements_basic.py` touched. Intentionally avoids importing from `mindspeed.core.transformer.flash_attention.reset_attention_mask.adaptor` (which pulls `from megatron.training import get_args` and breaks minimal Megatron checkouts). |
| **Status** | **PR open** (2026-05-29 06:20 Beijing). |
| **PR URL** | https://gitcode.com/Ascend/MindSpeed/merge_requests/3509 |
| **Branch on fork** | `zhengshencn_hwca/MindSpeed apex-rope-thd-shim` commit `1220468b` |
| **Empirical evidence** | A3 tlrescue with patched mindspeed installed → driver `_e2e_megatron_step_mindspeed.py` whose only NPU adaptation is `import mindspeed.megatron_adaptor` (no manual cuda→npu / RNG / Stream / apex / te shims) runs `MILES_E2E_SHAPE=real` to completion: 52M-param DSAMLASelfAttention forward + backward + Adam step, 4 algos compile at H=64 SEQ=2048 real DSv4-Flash shape. Without the patch, same driver raises `ModuleNotFoundError: No module named 'apex.transformer'`. |

### 6. Container-level: tlrescue image triton install hygiene

| | |
|---|---|
| **Why** | Pre-built tlrescue image (actual base: `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`) ships both `triton 3.6.0` and `triton-ascend 3.2.0`, breaking import. The verl image is published by Huawei, not by us, so the right fix is either to escalate to Huawei or to document a post-install recipe. |
| **What** | The post-install recipe is already documented in two durable locations: (a) memory `feedback_triton_vs_triton_ascend_packaging_conflict.md` (mechanical fix steps) and (b) §4 of this file (which describes how triton-ascend's packaging causes the conflict in the first place). The miles PR #1246 body also points users at the workaround. |
| **Status** | **DONE via documentation** — the underlying packaging bug belongs to #4 (Ascend/triton-ascend), and once #4 lands the image-level fix is automatic. The interim recipe is recorded in memory and referenced from the miles PR. |
| **Open follow-up if anyone asks** | If Huawei verl image authors want a single-line image fix, the recipe is: append `RUN pip uninstall -y triton && pip install --force-reinstall --no-deps triton-ascend` to the relevant Dockerfile after the bulk pip install layer. This is what we apply at runtime today. |

---

## Cardinality of patches needed for end-to-end

If everything in the picture lands:

| # | Upstream | Patch type | Outstanding work |
|---|---|---|---|
| 1 | tile-ai/tilelang-mlir-ascend | Python pass + UT | Reviewer merge (PR #80 open) |
| 2 | Ascend/AscendNPU-IR | C++ compiler pass | External (Huawei team owns the patch on issue #251) |
| 3 | radixark/miles | Python: tilelang kernels + dispatch | **PR #1246 open, MERGEABLE, REVIEW_REQUIRED** |
| 4 | triton-lang/triton-ascend | metadata or docs | **Issue #306 open** at github canonical home (gitcode home is archived) |
| 5 | Ascend/MindSpeed (core_r0.16.0) | New `MindSpeedFeature` for apex rope shim | **PR #3509 open** |
| 6 | tlrescue container | image recipe (Huawei-owned base) | DONE via documentation; underlying fix tracked under #4 |

**Status as of 2026-05-29 02:30 Beijing**: 5 of 6 actively in flight (#1 awaits review, #2 with Huawei, #3 PR #1246 open, #4 issue #306 open at github canonical home, #5 PR #3509 open). #6 closed via documentation. **Nothing is blocking us today** — manual-monkey-patch driver + triton workaround keep miles compiling and running at real shape. Numerical correctness is gated only on #2.

---

## Maintenance protocol

* Every time T13.B / T13.C / T14 yields a new upstream gap, add a new row in §"Required upstream changes" with the same fields.
* When a PR opens, lands, or closes, update the corresponding row's Status and link to the PR URL.
* If a discovery invalidates an earlier row (e.g. "MindSpeed already does this"), strikethrough the row but leave it for history.
* Mirror status into the project ROADMAP T12/T13/T14 cells.
