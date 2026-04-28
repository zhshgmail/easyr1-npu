# EasyR1 → Ascend 910C (A3) NPU — Design

Status: living document. Populated sections below are Requirements, Background, Restrictions, Task Decomposition, and High-Level Design. Detailed Design remains TBD until the dependency matrix (`dep-matrix.md`) is built.

Last updated: 2026-04-17. Section 4.1 (dep-tree analysis) complete; see `dep-matrix.md` and `porting-journal.md` for outputs.

---

## 1. Requirements

### 1.1 Functional

- **R1. EasyR1 rollout on A3.** EasyR1 master (April 2026) must be able to run its rollout/inference path (vLLM-based) on Ascend 910C (A3) hardware, using `vllm-ascend` in place of GPU vLLM.
- **R2. EasyR1 RL training on A3.** End-to-end RL training (GRPO/PPO-family algorithms as shipped by EasyR1 master) must run on A3, including policy update, advantage computation, and reference/reward model forward passes.
- **R3. Parity with EasyR1 master feature set, within v1 scope.** No feature regressions vs. upstream EasyR1 master for the recipes in v1 scope (see 1.4). If a feature is unavoidably NPU-incompatible, document it in `porting-journal.md` with the reason and any fallback.

### 1.2 Deliverables

The project has **two** deliverables, not one:

- **D1. Working EasyR1 on A3.** A reproducible, documented path from source + docker image to a successful rollout + short training run on A3.
- **D2. Reusable porting harness.** A set of assets that automate the same porting work for future EasyR1 versions and adjacent RL stacks:
  - **Skills** (`repo/skills/`) — Claude Code skills for recurring porting sub-tasks (image inspection, dep diffing, gap classification, upstream-branch hygiene, etc.).
  - **Scripts** (`repo/scripts/`) — image introspection, dep extraction, dep diff tooling.
  - **Knowledge** (`repo/knowledge/`) — extracted facts per image (pip freeze, site-packages layout, apt packages, env setup), organized for reuse.
  - **Workflows / docs** — `dep-matrix.md`, `porting-journal.md`, and any process docs that capture decisions.

D2 is first-class: harness assets are produced **continuously during** the porting work, not assembled at the end.

### 1.3 Non-requirements (for v1)

- Performance tuning / throughput parity with GPU. We target *functional* correctness on A3 first.
- Support for Ascend hardware other than 910C A3.
- Support for EasyR1 branches/forks other than master tip at the time of this design.
- Multi-node scale-out validation beyond what EasyR1's default recipes exercise.

### 1.4 v1 scope (narrowed after dep + source review)

The first working milestone is deliberately narrow to reduce surface area:

- **In**: text-only PPO / GRPO recipes on the Qwen2 / Qwen3 model families (same architectures EasyR1 exercises by default).
- **In**: default loggers — `console`, `file`, `wandb`. `mlflow`, `swanlab`, `tensorboard` are out-of-scope optional extensions.
- **In**: 8.5.0 A3 image as the base.
- **Out (v2+)**: VLM recipes (Qwen2-VL, Qwen2.5-VL, Qwen3-VL). `qwen_vl_utils` video ingestion pulls `decord`/`av` backends that need separate validation.
- **Out (v2+)**: video recipes.
- **Out (v2+)**: 8.5.2 image migration (torch_npu 2.9 / vllm_ascend 0.17 / transformers 5.x / huggingface_hub 1.x — a separate port task set).
- **Out (v2+)**: `liger-kernel` fused ops via `triton-ascend`.
- **Out (v2+)**: `flash-attn` *kernel* replacement beyond SDPA — v1 uses SDPA or torch_npu FA op; perf tuning is deferred.

---

## 2. Background

- **EasyR1** (github.com/hiyouga/EasyR1) is a slimmed-down fork of **veRL** focused on RL post-training recipes. It inherits veRL's architecture (rollout workers + trainer + parameter sync) but carries a narrower dependency surface and its own recipe set.
- **veRL** (github.com/verl-project/verl) has already been ported to Ascend NPU. The port is distributed as docker images under `quay.io/ascend/verl:verl-8.5.*-a3-*`. These images encode the *currently working* combination of `torch_npu`, `vllm-ascend`, `triton-ascend`, `transformers`, CANN, and related Ascend stack components on A3.
- Because EasyR1 is a veRL derivative, the veRL NPU port is a near-complete starting point. What remains is to **derive EasyR1's specific NPU dependency set** and close any gaps EasyR1 introduces.
- Typical sources of gap:
  - EasyR1 tracks a newer `transformers` than the veRL-A3 image ships.
  - EasyR1 uses libraries veRL doesn't (or vice versa).
  - Model-architecture-specific kernels (e.g. flash-attn) may need replacement with CANN-provided equivalents.
- CANN (the Ascend compute library) source is hosted at `gitcode.com/cann`, not in `upstream/`. Pull on demand when a CANN-level question arises.

---

## 3. Restrictions & Current State

### 3.1 Hardware

- **No A3 hardware access as of 2026-04-17.** Host is x86 (H100 GPU box, `115.190.166.102`), Ubuntu.
- The verl-A3 docker images are built for Ascend and **cannot execute** on this host. They are, however, **inspectable**: `docker create` + `docker cp` / `docker export` exposes the filesystem, which is sufficient for pip freeze, site-packages, apt packages, and environment setup.
- Any task that requires running code *on NPU* — validation, micro-benchmarks, kernel checks — is **BLOCKED** until A3 access is provisioned. The task decomposition below marks these explicitly.

### 3.2 Upstream edit discipline

- Each `upstream/<repo>/` is its own git clone. Edits go on a dedicated `ascend-port` branch in the relevant clone — **not** into patch files maintained out-of-tree.
- Patch files may be exported as an end-of-work artifact if the port ships as a patchset rather than a merged contribution.

### 3.3 Dependency gap profile (expected, to be confirmed)

- `transformers`: EasyR1 master likely requires a newer version than `torch_npu` currently supports. Upgrade path will need investigation in the `torch-npu` and `transformers` repos.
- `flash-attn`: GPU-only; on NPU must be replaced with CANN-provided attention ops, surfaced via `torch_npu` or model-level shims.
- `vllm`: replaced wholesale by `vllm-ascend`.
- `triton`: replaced by `triton-ascend` where used.
- Minor Python-only libraries usually port cleanly (no C++/CUDA code), so most gaps will be version-resolution, not true porting work.

### 3.4 Working preferences

- Mirror milestone updates and "waiting on user input" responses to Discord (chat_id `1494825170399924366`).
- No Claude-related text in commit messages.
- Design-doc format is formal (this document).

---

## 4. Task Decomposition

Status tags: `TODO`, `IN PROGRESS`, `BLOCKED` (requires A3 hardware), `DONE`.

Work that does not require hardware is prioritized. Hardware-gated items are tracked but parked.

### 4.1 Dependency-tree analysis

- **4.1.1** [DONE] Extract EasyR1 master's declared deps. Output: `repo/knowledge/easyr1-master-deps.md`.
- **4.1.2** [DONE] Extract veRL master's declared deps (GPU + NPU). Output: `repo/knowledge/verl-master-deps.md`.
- **4.1.3** [DONE] Inspect `verl-8.5.0-a3`. Output: `repo/knowledge/images/verl-8.5.0-a3.md` (CANN 8.5.0, torch 2.8.0, transformers 4.57.6).
- **4.1.4** [DONE] Inspect `verl-8.5.0-a3`. Output: `repo/knowledge/images/verl-8.5.0-a3.md` (CANN 8.5.1, torch 2.9.0, transformers 5.3.0.dev0).
- **4.1.5** [DONE] Build `repo/docs/easyr1/dep-matrix.md`.
- **4.1.6** [DONE] Gap classification applied in-matrix. Three real gaps: flash-attn (R), liger-kernel (R/D), pillow (A). One image-choice gap: transformers `<5.0.0` excludes 8.5.2 image.
- **4.1.7** [DONE] Prioritized sub-task list landed in `porting-journal.md` (2026-04-17 entry).

### 4.2 Upstream porting tasks (from 4.1.7 + source-scan findings)

Strategy: target the **8.5.0 image first** (EasyR1's `transformers<5.0.0` excludes 8.5.2). All 4.2 tasks land on branch `ascend-port` in `upstream/EasyR1/`. See `dep-matrix.md` §"Code-path blockers" for the call-site catalogue driving these.

**Dependency-level**

- **4.2.1** [TODO] Declare hidden direct deps (`jinja2`, `psutil`, `pyyaml`) in EasyR1's `requirements.txt`; tighten `tensordict` pin to match veRL NPU (`>=0.8.0,<=0.10.0,!=0.9.0`); make `flash-attn` and `liger-kernel` requirements conditional/optional.

**Code-path level (CUDA → NPU abstraction)**

- **4.2.2** [TODO] Device-module accessor util: `get_device_module()` returns `torch.npu` when `torch_npu` is importable, else `torch.cuda`. Replace all `torch.cuda.*` call sites (≈35, catalogued in dep-matrix.md).
- **4.2.3** [TODO] Distributed backend selection: make `init_process_group(backend=...)` configurable; default `hccl` on Ascend, `nccl` elsewhere. Affects `fsdp_workers.py`.
- **4.2.4** [TODO] Device-mesh device type + `device_map`: replace hardcoded `"cuda"` with a resolver. Affects `fsdp_workers.py` (4 mesh call-sites + 1 `device_map`).
- **4.2.5** [TODO] Attention backend selection: make `attn_implementation` configurable in `from_pretrained` calls; re-register `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` with an NPU-aware forward via `models/monkey_patch.py`. v1 uses SDPA or torch_npu FA op.
- **4.2.6** [TODO] Vendor flash-attn `bert_padding` helpers (`index_first_axis`, `pad_input`, `unpad_input`, `rearrange`) as a local module — these are pure-torch logic, not a kernel. Fall back to `torch.nn.functional.cross_entropy` when flash-attn's triton CE kernel is absent.

**Deferred / conditional**

- **4.2.7** [TODO — deferred] 8.5.2 migration: transformers 4.x → 5.x compatibility sweep; huggingface_hub 1.x API audit.
- **4.2.8** [TODO — conditional] `triton-ascend` work to bring `liger-kernel` fused ops to NPU. Only if perf follow-up justifies it.
- **4.2.9** [TODO — deferred] VLM / video recipe enablement (v2+ scope).

### 4.3 Harness buildout (runs continuously, in parallel with 4.1 and 4.2)

- **4.3.1** [TODO] Skill: `inspect-docker-image` — given an image ref, produce a `repo/knowledge/images/<tag>.md` with pip freeze, key site-packages dirs, apt packages, env, entrypoint.
- **4.3.2** [TODO] Skill: `diff-dep-sets` — given two dep sets, produce a matrix row-per-package with version deltas.
- **4.3.3** [TODO] Skill: `classify-dep-gap` — given a package name and the version/kind delta, propose a classification (version-bump / port / CANN-replace / drop) and an investigation plan.
- **4.3.4** [TODO] Skill: `upstream-branch-hygiene` — enforces the "branch, don't patch-file" rule across the six `upstream/` clones (branch naming, rebasing, export format).
- **4.3.5** [TODO] Script: `scripts/extract-image.sh` — wraps `docker create` + filesystem copy + pip freeze extraction.
- **4.3.6** [TODO] Script: `scripts/dep-diff.py` — consumes two JSON dep sets, emits the matrix.
- **4.3.7** [TODO] `porting-journal.md` — dated entries as findings accumulate; not end-of-project.
- **4.3.8** [TODO] Harvest each completed porting sub-task for any reusable pattern → land as a skill or knowledge doc before closing the sub-task.

### 4.4 Integration + runtime validation

- **4.4.1** [DONE, build-only] `upstream/EasyR1/Dockerfile.npu` builds an `easyr1-npu:ascend-port` image on top of `verl-8.5.0-a3`. Image layering + all Python imports smoke-tested on x86. Execution requires A3.
- **4.4.2** [BLOCKED — needs A3] Rollout-only smoke test: one EasyR1 recipe, rollout path only.
- **4.4.3** [BLOCKED — needs A3] End-to-end short training run (a few steps) on a small model.
- **4.4.4** [BLOCKED — needs A3] Document runtime findings in `porting-journal.md` and feed any recurring issues back into the harness (4.3).

---

## 5. High-Level Design

### 5.1 Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. Inspect both verl-A3 images                                        │
│    docker create / cp  →  pip freeze, site-packages, apt, env         │
│    output: repo/knowledge/images/verl-8.5.{0,2}-a3.md                 │
├──────────────────────────────────────────────────────────────────────┤
│ 2. Extract dep sets from source                                       │
│    EasyR1 master declared + resolved                                  │
│    veRL master declared + resolved (GPU reference)                    │
├──────────────────────────────────────────────────────────────────────┤
│ 3. Build dep matrix                                                   │
│    rows: packages  cols: EasyR1 / veRL-GPU / verl-A3-8.5.0 / 8.5.2    │
│    output: repo/docs/easyr1/dep-matrix.md                                    │
├──────────────────────────────────────────────────────────────────────┤
│ 4. Classify each gap                                                  │
│    - version-bump only                                                │
│    - port needed                                                      │
│    - replace with CANN-provided alternative (e.g. flash-attn → CANN)  │
│    - drop on NPU (not applicable)                                     │
│    output: prioritized sub-task list in porting-journal.md            │
├──────────────────────────────────────────────────────────────────────┤
│ 5. Execute ports                                                      │
│    branch = ascend-port in the relevant upstream/<repo>/              │
│    one branch per repo; many commits per branch                       │
├──────────────────────────────────────────────────────────────────────┤
│ 6. Harness extraction (continuous, not end-of-project)                │
│    every recurring step in 1–5 becomes a skill / script / knowledge   │
│    doc as soon as it's done twice                                     │
├──────────────────────────────────────────────────────────────────────┤
│ 7. Integration + A3-gated validation (BLOCKED until hardware)         │
│    build image, rollout smoke test, short training run                │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Why this order

- Dep matrix before porting: avoids premature port work on packages that turn out to be already-handled by the verl-A3 image.
- Harness extraction in parallel with porting (not after): the cost of capturing a workflow is lowest immediately after performing it. Deferring to the end loses detail.
- Runtime validation last, because it is the only hardware-gated step. Everything up to validation is doable on x86 today.

### 5.3 Gap classification — decision rules

| Signal | Classification | Action |
|---|---|---|
| Same package present in verl-A3 image, older version than EasyR1 needs | version-bump only | try upgrade in a test venv; if it loads, likely fine; runtime-verify when A3 is available |
| Package absent from verl-A3 image, pure Python | likely ports cleanly | add to image, verify import |
| Package absent, has compiled CUDA/C++ extensions | port needed | investigate NPU equivalent; may require work in `torch-npu`, `vllm-ascend`, or `triton-ascend` |
| Package is GPU-specific kernel (e.g. flash-attn) | replace with CANN-provided alternative | route through `torch_npu` ops or `vllm-ascend` attention backend |
| Package is dev/test only and unused at runtime | drop on NPU | skip |

### 5.4 Harness — what counts as reusable

A skill, script, or knowledge doc earns a place in `repo/` when either:
- it would be used again for the **next** EasyR1 version bump, or
- it would be used again for porting an **adjacent** RL stack (e.g. OpenRLHF) to NPU.

One-shot investigations stay in `porting-journal.md` and don't graduate to the harness.

---

## 6. Detailed Design

**Still TBD, but the blocking inputs now exist.** `dep-matrix.md` (matrix + code-path blockers table) and the sub-task list in §4.2 above are the inputs. Next pass will cover:

- Shim module layout in EasyR1: where `get_device_module()`, distributed backend resolver, device-type resolver, and attention backend registration live. Probably a new `verl/utils/device.py` module plus targeted edits in `workers/fsdp_workers.py`, `workers/sharding_manager/fsdp_vllm.py`, `models/monkey_patch.py`.
- The vendored flash-attn `bert_padding` module and the `cross_entropy` fallback: where to land them, how to keep tests green when flash-attn is absent.
- Docker image build plan layered on `verl-8.5.0-a3`: Dockerfile, EasyR1 install step, env overrides for NPU.
- Test strategy: import-level and unit tests runnable on x86 (e.g. validate device-module accessor returns `torch.cuda` in our dev env and resolves correctly when `torch_npu` shim is mocked); rollout + short-training integration tests gated to A3.
- Rollback plan per sub-task: each 4.2.x ships as an independent commit on `ascend-port`; revert is per-commit.
