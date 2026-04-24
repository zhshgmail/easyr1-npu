# vllm-ascend/day0-expert

**Product**: patch vllm-ascend to catch up with an upstream delta (torch,
transformers, community vllm, CANN) that vllm-ascend's shipped release
hasn't adapted to yet. Invoked **after** torch-day0 / transformers-day0
produces a working base image; this expert's job is to make
vllm-ascend run on top.

Outcomes: **A** (already-works), **B** (env-var workaround), **C-patch**
(source change with PR material), **C-report** (fix belongs to community
upstream we don't touch).

## When to use

- An upstream expert (torch-day0 / transformers-day0) shipped a deploy
  image and the caller wants vllm-ascend to work on it
- A consumer reports "vllm-ascend segfaults / fails to import on new
  torch" and the root cause appears to be in vllm-ascend's own code
- Before a new vllm-ascend release, quickly validate whether it works
  against pending community upstream moves

## When NOT to use

- The fix belongs to community vllm / community torch — use
  `vllm-day0-expert` or file upstream report
- vllm-ascend release already exists that handles the delta — check
  `pip index versions vllm-ascend`
- The segfault is in triton-ascend / CANN / torch_npu itself — route
  to that upstream's Day-0 expert

## Ground truth snapshot (2026-04-23)

Session `vllm-ascend-day0-analysis-20260423-0636` + `deploy-20260423-0655`:
- Target delta: torch 2.11.0+cpu ABI breaks pre-2.11-built
  `vllm_ascend_C.cpython-311-*.so`
- Call sites affected: `AscendRMSNorm.forward_oot` (RMSNorm →
  `_C_ascend.npu_add_rms_norm_bias`), `AscendTopKTopPSampler._apply_top_k_top_p_ascendc`
- Fix B+ patch shipped on `ascend-day0-torch211-20260423` (personal
  fork): auto-set `VLLM_BATCH_INVARIANT=1` in `vllm_ascend/__init__.py`
  when torch version doesn't match what `vllm_ascend_C` was built for
- V1.3 Qwen2-0.5B rollout smoke PASS after patch, without manual env
  var, no `VLLM_BATCH_INVARIANT=1` in the docker run command

## Scope / boundaries

**In scope**:
- Analyze vllm-ascend segfaults / import errors under a new upstream
- Locate affected call sites in vllm-ascend
- Choose minimum-invasive fix level (env-var vs python patch vs C++
  change)
- Apply patch on `ascend-day0-<delta>-<SESSION>` branch of
  `upstream/vllm-ascend/`
- Rebuild overlay image COPY-patching the modified files
- Smoke test to PASS; write PR material

**Out of scope**:
- Editing community upstreams (`upstream/vllm/`, `upstream/pytorch/`)
- Rebuilding the C++ extension against new torch headers (the "proper"
  long-term fix; tracked as tech debt in the companion PR material)
- CANN driver issues (separate team)

## Inputs

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `vllm-ascend-day0-20260501-0900` |
| `TARGET_DELTA` | e.g. `torch-2.11`, `vllm-0.20.0` |
| `BASE_IMAGE` | Deployed image from upstream Day-0 session (torch-day0 output) |

## Deliverable

```json
{
  "session_tag": "vllm-ascend-day0-...",
  "target_delta": "torch-2.11",
  "outcome": "A|B|C-patch|C-report",
  "base_image": "<torch-day0 deployed tag>",
  "patched_branch": "ascend-day0-torch211-<SESSION>" | null,
  "patched_image_tag": "...-vllmascend-fixb:..." | null,
  "smoke_results": {"V1.3": "PASS (marker matched)"},
  "affected_call_sites": ["vllm_ascend/ops/layernorm.py:73", "..."],
  "fix_level": "env-var | python-patch | cpp-rebuild",
  "pr_material_path": "...",
  "provenance": {"produced_by": "vllm-ascend-day0-worker"},
  "cleanup": "partial (patched overlay preserved; validation images rmi'd)"
}
```
