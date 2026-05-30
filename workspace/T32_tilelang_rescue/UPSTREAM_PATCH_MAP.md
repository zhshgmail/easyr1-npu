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

Adjacent (rollout / deployment side, in critical path for full RL):
* **sglang on NPU** (`sgl-project/sglang` + `sgl-project/sgl-kernel-npu`) — **miles 默认 rollout engine**(`miles/ray/rollout.py:16` 顶层 `from sglang.srt.constants import …`)。NPU 支持 in-tree(`docs/platforms/ascend/`),不是 Ascend fork。**已发现的版本错位**:Huawei 发布的 `quay.io/ascend/verl:verl-sglang-8.5.0` 和 `swr.../lmsysorg/sglang:cann8.5.0-a3-glm5` 都比上游落后(我们手上 sgl-kernel-npu 2026.02.01 vs 上游 2026.05.01.post3,差 3 个 release wave;sglang 0.5.10 vs 上游 v0.5.12.post1)。**推荐路径**:`lmsysorg/sglang:main-cann8.5.0-a3`(每天更新)或 `lmsysorg/sglang:v0.5.12.post1-cann8.5.0-a3`。**Tracker**:sglang issue **#23598 "DeepSeek-V4 Day 0 Support on NPUs" 仍 open**;PR #23882「Deepseek V4」merged 2026-05-08 进 v0.5.12,PR #18521「GlmMoeDsaForCausalLM」merged 2026-02-10,PR #11061「DeepSeek V3.2 Exp」merged 2025-10-06。
* **vllm-ascend** (`Ascend/vllm-ascend`) — 备用 inference engine;pulls `xgrammar` which transitively pins mainline `triton`。
* **MindSpeed-LLM** (`Ascend/MindSpeed-LLM`) — LLM-specific pretraining;depends on MindSpeed Core;47 model families。Not needed if miles + Mcore-0.16 covers the model.
* **MindSpeed-RL** (`Ascend/MindSpeed-RL`) — RL on top;miles replaces it.

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

### 4. Packaging conflict between `triton-ascend` and mainline `triton` (REFRAMED)

| | |
|---|---|
| **Symptom on tlrescue** | `import triton` raises `ImportError: cannot import name 'Language' from 'triton.backends.compiler'`. Both `triton-ascend 3.2.0/3.2.1` (NPU DSL) and mainline `triton 3.6.0` (GPU DSL) are present in site-packages; their forks of `triton/backends/compiler.py` have diverged enough that the last-installed one breaks the other's amd/nvidia backends. |
| **Reframing 2026-05-29** | I initially filed this at `triton-lang/triton-ascend` (issue #306) asking for `Provides-Dist: triton` / namespace packaging / conflict declaration. User correctly pointed out: **`triton-ascend` shouldn't need to defend against mainline triton being present** — they are alternatives that aren't meant to coexist. The real source of the conflict is upstream: **`xgrammar` declares `Requires-Dist: triton; platform_system == "Linux" and platform_machine == "x86_64"`**, which fires on NPU hosts (also Linux x86_64) and transitively pulls in mainline triton via `vllm` → `vllm-ascend`. Then `pip install triton-ascend` collides. So the responsible layers are `xgrammar` (dep declaration) or the `quay.io/ascend/verl` image recipe (install order / mutual exclusion). |
| **What I did** | Closed the `triton-lang/triton-ascend` issue as "not planned" with a comment explaining the reframing; the close comment preserves the underlying repro and the workaround so future folks landing via web search aren't lost. |
| **Closed issue** | https://github.com/triton-lang/triton-ascend/issues/306 (closed 2026-05-29 with reframing comment) |
| **Real upstream targets if escalated** | (A) `mlc-ai/xgrammar` — request platform-aware triton dep that doesn't fire on NPU hosts; (B) Huawei `quay.io/ascend/verl` image authors — request install-order fix so triton-ascend and xgrammar's mainline triton dep aren't both materialized; (C) downstream NPU integrators (vllm-ascend, miles users) — internal workaround documented + applied as part of `setup_env.sh` recipes. |
| **Status** | **Closed via reframing** (not a triton-ascend issue). The workaround (`pip uninstall triton && pip install --force-reinstall --no-deps triton-ascend`) remains documented in memory and in this map; not blocking miles training. |
| **gc CLI dead-end (memo)** | The earlier attempt to file at `gitcode.com/Ascend/triton-ascend` via `gc CLI`/raw API was blocked with `400 Bad Request body不能为空`. The gitcode repo is now archived (project moved to github canonical home) and this dead-end is moot for future work. |
| **Triton 3.6.0 needed?** | **No** for miles training. miles's triton kernels use the stable `@triton.jit / tl.*` DSL that triton-ascend 3.2.0+ implements. Mainline triton 3.6 only enters via xgrammar's inference-time grammar path (dead code in training). |

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
| 4 | triton-lang/triton-ascend | (reframed) | **Issue #306 closed as "not planned"** — see §4 for full reframing |
| 5 | Ascend/MindSpeed (core_r0.16.0) | New `MindSpeedFeature` for apex rope shim | **PR #3509 open** |
| 6 | tlrescue container | image recipe (Huawei-owned base) | DONE via documentation; underlying fix tracked under #4 |

**Status as of 2026-05-29 10:30 Beijing**: 3 actively in flight (#1 awaits review, #2 with Huawei) + 2 PRs intentionally held as drafts pending fuller multi-step / multi-layer validation (#3 PR #1246 draft, #5 PR #3509 draft). #4 closed via reframing. #6 closed via documentation.

**Component validation matrix as of 2026-05-29 10:25 Beijing**:

| Test | Patched-MindSpeed stack | Manual-monkey-patch stack | Notes |
|---|---|---|---|
| `MILES_E2E_SHAPE=reduced` single step | ✅ 11/12 finite grads, PASS | ✅ same | wq_b.weight non-finite is R-KA-15 indexer_bwd known |
| `MILES_E2E_SHAPE=reduced` 2-layer x 3-iter | ✅ 25/25 finite per iter, PASS | ✅ baseline `4e12a91` | no NaN drift across iters |
| `MILES_E2E_SHAPE=real` single step | ✅ compile + flow PASS | ✅ same | R-KA-16 numerical NaN (gated on #2) |
| `MILES_E2E_SHAPE=real` multi-iter | not run | not run | gated on R-KA-16 |
| RL full step (rollout → reward → actor train) | not run | not run | requires vllm-ascend rollout path — TBD |

**Nothing is blocking miles real-shape compile-and-flow-through today**. Numerical correctness on real shape is gated only on #2 (R-KA-16). The PR drafts will move to ready after full RL-step or further reviewer feedback. miles fork `npu-tilelang-dispatch` commit `aef6e2d` carries the MindSpeed-aware drivers.

---

## Maintenance protocol

* Every time T13.B / T13.C / T14 yields a new upstream gap, add a new row in §"Required upstream changes" with the same fields.
* When a PR opens, lands, or closes, update the corresponding row's Status and link to the PR URL.
* If a discovery invalidates an earlier row (e.g. "MindSpeed already does this"), strikethrough the row but leave it for history.
* Mirror status into the project ROADMAP T12/T13/T14 cells.
