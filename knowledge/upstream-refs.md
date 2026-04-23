# Upstream git references matching the verl-A3 images

When reviewing or porting against an NPU dependency, **do not default to master/main**. Each NPU project maintains version-specific branches (CANN↔torch pairs, vLLM release trains, Triton release lines), and the A3 images ship specific combinations of these. Reviewing master when the image pins a branch will produce misleading findings — API surface, bundled ops, and even CANN requirements can differ.

This doc maps each upstream repo to the ref that matches each verl-A3 image, so future reviews check the right code.

Derivation: read from each repo's README or compat table, cross-checked against the pip-freeze extracted from the image.

## 8.5.0 image — `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest`

| Repo | Image version | Matching ref | Evidence |
|---|---|---|---|
| torch-npu | torch_npu 2.8.0 (paired with CANN 8.5.0) | `origin/v2.8.0-7.3.0` | torch-npu README CANN-compat table: "CANN 8.5.0 \| 2.8.0 \| 2.8.0.post2 \| v2.8.0-7.3.0". Image has 2.8.0 (pre-post2 dev build; this branch is the right base). |
| transformers | 4.57.6 | `v4.57.6` tag | exact tag exists in upstream/transformers. |
| vllm-ascend | 0.13.1.dev18+g2e5f72f92 | `origin/releases/v0.13.0` as the nearest stable branch; actual image uses an unpublished `.dev18` build off the branch head | releases/v0.13.0 README targets CANN 8.3.rc2 + torch 2.8.0 (not exactly CANN 8.5.0, so the image is a CI rebuild). Commit `2e5f72f92` is not in the public branch — it's internal. |
| triton-ascend | 3.2.0 | `origin/release/3.2.x` | veRL NPU requirements pins `triton-ascend==3.2.0`; `v3.2.0` tag also exists. |
| CANN | 8.5.0 | — | not in `upstream/`; lives at `gitcode.com/cann`. Pull on demand. |

## 8.5.2 image — `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`

Note: the image tag says `8.5.2` but CANN is **8.5.1**. `8.5.2` is the verl image revision.

| Repo | Image version | Matching ref | Evidence |
|---|---|---|---|
| torch-npu | torch_npu 2.9.0 (paired with CANN 8.5.1) | `origin/v2.9.0-7.3.0` (closest; note table only documents CANN 8.5.0, so this is the nearest supported combo) | README table: "CANN 8.5.0 \| 2.9.0 \| 2.9.0 \| v2.9.0-7.3.0". CANN 8.5.1 is a minor revision without a separate torch-npu branch. |
| transformers | 5.3.0.dev0 | `main` or a 5.x pre-release branch | No 5.x tag yet (highest 5.x tag does not exist at time of extraction). Image is built off a transformers dev commit. |
| vllm-ascend | 0.17.0rc2.dev109+g54879467c | `origin/main` as of ≈2026-03/04, or a pre-`releases/v0.18.0` snapshot | main README describes CANN 8.5.1 + torch 2.9.0 + vllm-ascend matching vllm. `releases/v0.18.0` branch exists but image is `0.17.0rc2.dev109`, i.e. between 0.17 and 0.18. |
| triton-ascend | 3.2.0 (plus upstream `triton 3.6.0` alongside) | `origin/release/3.2.x` for triton-ascend; upstream triton from its own 3.6 release | upstream triton is pulled in separately, not via triton-ascend. |
| CANN | 8.5.1 | — | gitcode.com/cann. |

## Day-0 overlay combinations (2026-04-23)

These are produced by Day-0 skills layered on top of a base image. The
refs listed are what the overlay's changed components correspond to,
not what the base image ships.

### `easyr1-npu-torch211:torch-day0-manual-20260423-0537`

**Base**: `easyr1-npu-852:trans-upg-e2e-20260422-2200` (= v2 image pattern
above)

| Component | Overlay version | Ref | Evidence |
|---|---|---|---|
| torch | 2.11.0+cpu | upstream `v2.11.0` tag | download.pytorch.org/whl/cpu wheel, manylinux_2_28_x86_64 cp311 |
| torch_npu | 2.11.0rc1 | Ascend/pytorch `v2.11.0` source tag (no GitHub release yet) | PyPI pre-release wheel 2026-03-24 |
| torchvision | 0.26.0+cpu | torch ecosystem v0.26.0 | download.pytorch.org |
| torchaudio | 2.11.0+cpu | torch ecosystem v2.11.0 | download.pytorch.org |
| CANN | 8.5.1 (inherited from base) | — | README pairs 2.11.0rc1 with CANN 8.5.0; 8.5.1 is one patch ahead, validated functional |

### `easyr1-npu-torch211-vllmascend-fixb:ascend-day0-torch211-20260423`

**Base**: `easyr1-npu-torch211:torch-day0-manual-20260423-0537` (above)

| Component | Overlay version | Ref | Evidence |
|---|---|---|---|
| vllm-ascend | patched 0.17.0rc2.dev109 | session-local trace branch `ascend-day0-torch211-20260423` on mirror fork `zhshgmail/vllm-ascend`, branched from upstream `54879467` (the image's shipped commit); authoritative handoff = `workspace/vllm-ascend-day0-deploy-20260423-0655/PR_MATERIAL.md` for vllm-ascend maintainer | 2 file-level edits: `utils.py` torch-ABI-safe guard + `__init__.py` early VLLM_BATCH_INVARIANT set |

### `easyr1-npu-trans56:trans-day0-wetrun-20260423-0109`

**Base**: v2 image

| Component | Overlay version | Ref | Evidence |
|---|---|---|---|
| transformers | 5.6.0 | `v5.6.0` tag | community release 2026-04-22 |

## How to use this doc

Before running a code review, static analysis, or patch against any of these repos, `git checkout` the matching ref first:

```bash
cd upstream/torch-npu && git checkout origin/v2.8.0-7.3.0
cd upstream/transformers && git checkout v4.57.6
cd upstream/vllm-ascend && git checkout origin/releases/v0.13.0
cd upstream/triton-ascend && git checkout origin/release/3.2.x
```

When porting EasyR1 or extending the harness:
- API references should be confirmed against the **ref matching the target image**, not master.
- Bug reports / features on master may already be fixed on the release branch, or vice versa.
- Memory: always note which ref a finding was made against, so later readers can re-verify.

## Update policy

- Bump this doc each time we change the target image or a new image revision ships.
- When probing a new NPU repo, check for:
  - `releases/*` or `v*-*` branches (vllm-ascend, torch-npu pattern)
  - `release/*` branches (triton-ascend pattern)
  - A README compatibility table (torch-npu, vllm-ascend both have one)
- Never assume the default branch is the right one.
