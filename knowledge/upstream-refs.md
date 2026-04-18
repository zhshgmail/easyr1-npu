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
