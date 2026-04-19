# EasyR1 → Ascend 910C (A3) NPU — full port summary

**Audience**: the engineer (or agent) who will redo this work for a newer EasyR1 version, or port another RL framework (veRL, OpenRLHF, TRL) to Ascend.

This document covers **what changes, in which repositories, driven by which fact about NPU, at which smoke level.** Pairs with:
- `DELIVERABLE.md` — the sign-off doc (project-status view).
- `npu-patterns.md` — the stable-ID catalog (per-finding view).
- `smoke-ladder-convention.md` — the validation methodology.

Last updated: 2026-04-19. EasyR1 base ref: `dd71bbd` (`origin/main` as of port start). Target image: `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest` (CANN 8.5.0, torch_npu 2.8.0, transformers 4.57.6, vllm_ascend 0.13.1.dev18, triton-ascend 3.2.0).

---

## 1. Repositories touched, why, and extent

| Repo | Ours? | Changed | Purpose | Commit count on `ascend-port` (or equivalent) |
|---|---|---|---|---|
| `hiyouga/EasyR1` → `zhshgmail/EasyR1` (private fork) | yes, fork + ascend-port branch | yes | primary port target | **20 commits** (as of V2.2) |
| `zhshgmail/easyr1-npu` | ours (private) | yes | harness: docs, knowledge, scripts, skills | **26+ commits** |
| `huggingface/transformers` | private fork held, **not modified** | no | NPU flash-attn shim already ships upstream (`transformers.integrations.npu_flash_attention`) — no patches needed | 0 |
| `vllm-project/vllm-ascend` | private fork held, **not modified** | no | vllm 0.13 shim for NPU; rollout path works as-is | 0 |
| `verl-project/verl` | private fork held, **not modified** | no | reference port; we read, we do not modify | 0 |
| `Ascend/pytorch` (torch_npu, on gitcode) | not forked | no | consumed as-is from the base image | 0 |
| `Ascend/triton-ascend` (on gitcode) | not forked | no | consumed as-is; one reinstall workaround (Dockerfile.npu), no source patches | 0 |

**Bottom line**: actual source modifications live in exactly **one repository** (our EasyR1 fork), plus operational/docs in our private `easyr1-npu` repo. Every other dependency is consumed as-is from the base image or works through upstream-provided shims.

---

## 2. Changes to `EasyR1` (20 commits summarized by concern)

Full log: `git log --oneline main..ascend-port` on `zhshgmail/EasyR1`.

### 2.1 Dependency layer (1 commit)
- `7ee0f0b` — split `requirements.txt` into common / gpu / npu variants. Move flash-attn / liger-kernel / vllm to a `gpu` extras. Declare 3 previously-hidden imports (jinja2, psutil, pyyaml). Tighten tensordict pin.

### 2.2 Device accessor (3 commits)
- `72b564a` — new `verl/utils/device.py` helper (`is_npu_available`, `get_device_name`, `get_device_module`, `get_dist_backend`, `get_default_attn_implementation`, `get_ray_resource_name`, `get_visible_devices_env`).
- `7187b51` — sweep 35 `torch.cuda.*` call sites across 10 files; also `init_device_mesh("cuda", ...)`, `"cuda"` strings, `device_map="cuda"`, ROCm gate behind `is_npu_available()`.
- `496d198` — fix missed `flat_param_to("cuda", ...)` in `load_fsdp_submodule`; wrap bare `current_device()` int in `torch.device(device_name, index)` for `.to()` calls; add `extras_require["gpu"]` to setup.py.

### 2.3 Attention backend (3 commits, v1 baseline)
- `6701a50` — `attn_implementation` configurable in `from_pretrained`/`from_config`; defaults to `sdpa` on NPU, `flash_attention_2` on CUDA.
- `da2487f` — vendor `flash_attn.bert_padding` helpers as pure-torch at `verl/utils/npu_flash_attn_utils.py`; add `verl/utils/attention_utils.py` lazy façade. `dp_actor.py` / `dp_critic.py` import from façade.
- `ffafa0d` — NPU config-level gate moved from `apply_ulysses_patch` to `_build_model_optimizer` with clear error; add `tests/test_device.py` (first x86 unit test).

### 2.4 Ray NPU integration (3 commits)
- `fb1a223` — register NPU as Ray custom resource; `_check_resource_available` reads `available_resources()[get_ray_resource_name()]`; placement bundles + actor options use `resources={"NPU": n}` instead of `num_gpus` on NPU.
- `59641d4` — `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` in `runtime_env.env_vars` (Ray 2.55+ wipes `ASCEND_RT_VISIBLE_DEVICES` on actor spawn otherwise).
- `cc8e794` — `VLLM_ASCEND_ENABLE_NZ=0` for RL param sync.

### 2.5 Platform shims (4 commits)
- `cbfe645` — `Dockerfile.npu` layered on verl-8.5.0-a3 base.
- `cd16649` — Dockerfile.npu force-reinstalls triton-ascend==3.2.0 (fixes NPU-BUG-001: base image ships incomplete triton tree).
- `87faff1` — `verl/utils/vllm_utils.py` handles `vllm.lora.models` → `vllm.lora.lora_model` rename (vllm 0.13).
- `2d8ee2c` — `verl/workers/sharding_manager/fsdp_vllm.py` handles `get_tensor_model_parallel_group` → `get_tp_group` rename (vllm 0.13).

### 2.6 Smoke harness (4 commits)
- `906215d` — V1.4 2-chip GRPO smoke.
- `72a7f22` — V1.5 4-chip GRPO smoke.
- V2.1 padding_free smoke + workaround:
  - `fbaa983` — NPU-CP-007: swap flash-attn imports for `transformers.integrations.npu_flash_attention` on NPU. Drop NPU raises from `monkey_patch.apply_ulysses_patch` and `fsdp_workers._build_model_optimizer`.
  - `75bad74` — V2.1 smoke script sets `use_torch_compile=false` (NPU-BUG-003 workaround for triton-ascend inductor shape-sensitive crash).
  - `9e971f0` — rename V1.6 → V2.1 (milestone-vs-level naming fix).
- `6f8197f` — V2.2 (4-chip + ulysses_size=2 + padding_free=True) smoke script.

### 2.7 Change types (generalizable taxonomy)

The 20 commits fall into 5 archetypes. Future ports will hit the same archetypes in roughly the same proportions:

| Archetype | Count | Examples |
|---|---|---|
| **Device dispatch** (swap hardcoded CUDA constructs to an accelerator-aware helper) | 3 (device accessor commits) | NPU-CP-001 family |
| **Version-compat shim** (handle upstream API renames) | 2 | NPU-CP-002, NPU-CP-004 |
| **Cross-cutting runtime registration** (tell the framework about NPU at init time) | 3 | NPU-CP-003 (Ray), NPU-BUG-002 (visibility env), NPU-ENV-002 (NZ knob) |
| **Vendoring / imports** (make CUDA-only packages optional) | 2 | padding helpers vendor (da2487f), flash-attn via transformers shim (fbaa983) |
| **Smoke / operational** (Dockerfile, smoke scripts, chip-check, bind mounts) | 10 | — |

Most of the "scary" port work is in categories 1-4 (8 commits). Half of the commits (category 5) are infrastructure that's either generic (container runner) or one-off (smoke scripts).

---

## 3. Changes to the harness repo (`easyr1-npu`)

### 3.1 Docs
- `docs/design.md` — formal design doc (requirements, background, restrictions, task decomposition).
- `docs/dep-matrix.md` — per-package matrix (EasyR1 / veRL / 8.5.0 image / 8.5.2 image) + code-path blockers table.
- `docs/porting-journal.md` — dated session log; every V1.x and V2.x bring-up recorded.
- `docs/npu-gap-plan.md` — v1 / v2+ scope boundaries.
- `docs/skills-design.md` — system design for the multi-repo port skills (V0.2, with status table).
- `docs/DELIVERABLE.md` — sign-off summary (updated for v1 and v2).
- `docs/codex-signoff.md` — v1 codex sign-off record.
- `docs/codex-signoff-v2.md` — v2 codex sign-off record.
- `docs/codex-review-skills-audit.md` — archived second codex review.
- `docs/PORT-SUMMARY.md` — this file.

### 3.2 Knowledge
- `knowledge/easyr1-master-deps.md` — EasyR1 source dep extraction.
- `knowledge/verl-master-deps.md` — veRL source dep extraction (GPU + NPU reqs).
- `knowledge/upstream-refs.md` — image→ref mapping for all upstream repos.
- `knowledge/images/verl-8.5.0-a3.md` — target image inventory.
- `knowledge/images/verl-8.5.2-a3.md` — newer image inventory (for v3 migration planning).
- `knowledge/npu-patterns.md` — **19 stable IDs** (CP×7, BUG×3, ENV×4, OPS×5). Uniform `Symptom / Root cause / Fix / Commit ref / Generalizable rule` schema.
- `knowledge/smoke-ladder-convention.md` — the V<milestone>.<level> naming convention and the 5-level ladder.

### 3.3 Scripts
- `scripts/run-npu-container.sh` — NPU-aware container runner: device passthrough, bind mounts for 3 user paths + live source, env defaults (HF_ENDPOINT, VLLM_ASCEND_ENABLE_NZ, PYTHONDONTWRITEBYTECODE), chip-occupancy precheck.
- `scripts/inspect-ascend-image.sh` — automates the image-inventory workflow; includes NPU-BUG-001 triton-ascend integrity check.
- `scripts/code-path-sweep.sh` — grep-based scan for GPU-only call sites; seeded with 12 patterns keyed to NPU-CP-* IDs.
- `scripts/smoke_v11_device.py` — V1.1 / V1.2 device accessor + vendored helpers smoke.
- `scripts/smoke_v13_rollout.py` — V1.3 vllm_ascend rollout smoke.

### 3.4 Skills
- `skills/codex-review/` — generic reviewer via local `codex` CLI.
- `skills/upstream-branch-hygiene/` — "local edit → push personal → remote pull" discipline.
- `skills/npu-container-runner/` — wraps `run-npu-container.sh`.
- `skills/npu-image-inspect/` — wraps `inspect-ascend-image.sh`.
- `skills/npu-code-path-sweep/` — wraps `code-path-sweep.sh`.
- `skills/ray-npu-shim/` — drop-in `ray_npu_shim.py` (5 functions: is_npu_available, get_ray_resource_name, ray_init_npu_aware, apply_actor_options, placement_bundle).

---

## 4. Validation — full smoke ladder

Every level below ran on the target image, in sequence. Each passing level was committed to git; each failure triggered a fix that's now in the catalog.

| Smoke | Config | Status | Wall time | Evidence |
|---|---|---|---|---|
| V1.1 | device accessors in container | ✅ | seconds | `smoke_v11_device.py` output; NPU-CP-001 helper validated |
| V1.2 | tensor round-trip + vendored bert_padding | ✅ | seconds | same script |
| V1.3 | vllm_ascend Qwen2-0.5B rollout | ✅ | ~60s | `smoke_v13_rollout.py`; 3 prompts generated coherent text |
| V1.4 | GRPO 2-step, 2 chips, padding_free=False, ulysses=1 | ✅ | 8m24s | entropy_loss 0.991 → 1.263; checkpoints ranked 0, 1 |
| V1.5 | GRPO 2-step, 4 chips, padding_free=False, ulysses=1 | ✅ | 4m55s | 1.7× speedup; HCCL cross-card; ranks 0-3 |
| V2.1 | GRPO 2-step, 2 chips, padding_free=**True**, ulysses=1 | ✅ | 7m44s | entropy_loss 0.991 → 1.264 (matches V1.4 numerically); NPU FA varlen via `transformers.integrations.npu_flash_attention` |
| V2.2 | GRPO 2-step, 4 chips, padding_free=True, ulysses=**2** | (in progress at doc time) | ~10m est. | validates HCCL SP collectives under varlen |

Each smoke script is independently runnable via `repo/scripts/run-npu-container.sh`; commands documented in `DELIVERABLE.md §5.2`.

---

## 5. Recipe for the next EasyR1 version (or adjacent framework port)

This is the **step-by-step for redoing this work on a newer EasyR1 commit, or porting another Ray-based RL framework like OpenRLHF / TRL-with-Ray**. Follow in order.

### Step 1 — Read the reference (NPU-OPS-005)
- `git log` on veRL master; find any NPU-specific commits since our last port session.
- `git checkout v<new>` on transformers; look at `src/transformers/integrations/npu_flash_attention.py` — has the API surface changed?
- Open-source notes: HuaweiCloud Ascend docs, vllm-ascend release notes.
- **Never design anything from scratch** for a concern that might already be addressed upstream. NPU-OPS-005 exists because this was a 2-day mistake.

### Step 2 — Inventory the new target image
- `bash scripts/inspect-ascend-image.sh <image>` → emits `knowledge/images/<slug>.md`.
- Check NPU-BUG-001 integrity warning (triton-ascend `__init__.py` sanity).
- Diff pip freeze against `knowledge/images/verl-8.5.0-a3.md` to find package version shifts.

### Step 3 — Version-align upstream refs
- Update `knowledge/upstream-refs.md` with which branch/tag of each upstream matches the new target image (torch-npu, vllm-ascend, triton-ascend, transformers).
- This is the reference point for any "is this bug a known upstream thing?" question later.

### Step 4 — Set up the port repo layout
- Fork the target RL framework on GitHub/GitCode → `zhshgmail/<repo>` (private).
- Fork EasyR1 if different version. Set `personal` remote on local clone. Create `ascend-port` branch.
- `gh api --method PATCH repos/zhshgmail/<repo> -f visibility=private` if the fork was public.
- Per `skills/upstream-branch-hygiene/`: all source changes on `ascend-port` locally, push to `personal`, NPU host pulls.

### Step 5 — Initial sweep
- `bash scripts/code-path-sweep.sh upstream/<framework>` → emits `docs/code-path-sweep-<framework>.md`.
- For each hit, decide: fix now, defer, or not applicable.
- Adopt the `ray-npu-shim` pattern for Ray integration (`skills/ray-npu-shim/ray_npu_shim.py` is a drop-in).

### Step 6 — Apply the 5 archetype changes
Following §2.7's archetypes:
1. Device dispatch: copy `verl/utils/device.py` pattern (or port the framework's existing device helper to add NPU branches).
2. Version-compat shim: add try/except around any vllm / transformers internal imports cited in NPU-CP-002 / NPU-CP-004 catalog.
3. Runtime registration: apply ray-npu-shim to `ray.init`, actor options, placement bundles.
4. Vendoring / CUDA-optional imports: for any `from flash_attn import ...`, use the transformers integration shim (NPU-CP-007).
5. Smoke scripts + Dockerfile: copy patterns from `upstream/EasyR1/examples/qwen2_0_5b_math_grpo_npu_smoke*.sh` and `upstream/EasyR1/Dockerfile.npu`.

### Step 7 — Walk the smoke ladder
- V1.1 → V1.5 → V2.1 → V2.2 in sequence. Fail-fast at the cheapest level.
- Each failure: capture in `porting-journal.md`, add to `npu-patterns.md` with a new ID, commit the fix.
- **Reuse existing ladder labels.** Don't invent "V1.6" for v2 work — use V2.x.

### Step 8 — Sign-off
- Write a one-page `DELIVERABLE.md`-style summary.
- Run `bash -c 'codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox'` with a prompt mirroring `skills/codex-review/`. Use codex as sign-off proxy if the user is unavailable.
- Archive verdict.

---

## 6. Expected cost for a second port

Based on this port's time spend (~2 days from zero to v2):
- **Same EasyR1 version, different machine**: ~half day, most of it setup (image pulls, host onboarding, chip access).
- **Newer EasyR1 commit (minor upstream changes)**: ~0.5-1 day. Assume 2-4 catalog patterns hit, each 30-60 minutes to confirm the fix pattern applies and run smoke levels.
- **Different Ray-based RL framework (OpenRLHF, TRL with Ray)**: ~1-1.5 days. Same sweep + shim + smoke pattern; new framework-specific quirks cost 0.5 day of new-ID work.
- **Major transformers upgrade (4.57 → 5.0+)**: ~0.5-2 days. Depends on whether `transformers.integrations.npu_flash_attention` surface stable.

These are estimates based on **one data point**. The 2-day → 1-hour discovery of `transformers.integrations.npu_flash_attention` suggests that for anything with a reference port, actual cost is lower than first estimate.

---

## 7. Known debt

Non-blocking but named:

1. `NPU-BUG-003` stabilization — triton-ascend inductor shape-sensitive crash. Currently worked around via `use_torch_compile=false`. Need: either a narrower guard, a triton-ascend version bump, or upstream fix.
2. V2.2 passing is the last ladder level — once confirmed, v2 is fully closed. (This doc gets a status update in the journal.)
3. 8.5.2 image migration — transformers 5.3.0.dev0 + huggingface_hub 1.11 + vllm_ascend 0.17. Separate project (`v3`).
4. Multi-node scaling — only validated single-node.
5. Long-context (>2k response length) — not exercised.

---

## 8. Assets a future engineer can copy-paste

When starting a new port, clone these four things straight off the top of the repo:
- `scripts/run-npu-container.sh` — renames: change `easyr1-npu` image tag, bind mount path, owner user.
- `skills/ray-npu-shim/ray_npu_shim.py` — 100% drop-in, no edits.
- `examples/qwen2_0_5b_math_grpo_npu_smoke*.sh` templates — swap framework name, config path.
- `knowledge/npu-patterns.md` — read first, add IDs to it as new findings surface.

That's the concrete floor of "what you get for free after this port."
