# EasyR1 → Ascend 910C (A3) — deliverable summary

Status: **v1 functional milestone reached.** Date: 2026-04-18. Work done in 1 day.

---

## TL;DR

- `easyr1-npu:ascend-port` docker image runs GRPO training on Qwen2-family models on Ascend 910C A3 hardware.
- V1.4 smoke: 2 GRPO steps on Qwen2-0.5B + math12k dataset, 2 chips, completed end-to-end (8m24s).
- V1.5 smoke: same recipe scaled to 4 chips (2 A3 cards) — in progress / finished depending on when you read this, see §4.
- 7 NPU-specific issues identified, all fixed on `zhshgmail/EasyR1` branch `ascend-port` (13 commits).
- 6 reusable skills and 3 reusable scripts in `zhshgmail/easyr1-npu` (this repo) for the next port.
- v2 work (padding-free via `npu_fusion_attention`, ulysses SP on NPU, 8.5.2 migration) explicitly deferred; see §7.

---

## 1. Scope delivered

Functional target met per `design.md §1.1-1.4`:

- **R1** EasyR1 rollout on A3: ✅ verified in V1.3 (Qwen2-0.5B via `vllm_ascend`).
- **R2** EasyR1 RL training on A3: ✅ verified in V1.4 (2 steps GRPO, FSDP across 2 chips, HCCL, checkpoint saved).
- **R3** Parity with EasyR1 master feature set (within v1 scope — text-only, no Ulysses SP, no padding_free, default loggers).

---

## 2. What's in the port

### 2.1 `zhshgmail/EasyR1` (private fork of hiyouga/EasyR1), branch `ascend-port`

13 commits. The port is **additive** — no CUDA path regressed. Grouped by concern:

| Concern | Commits | What |
|---|---|---|
| Dep discipline | 1 (`7ee0f0b`) | Split `requirements.txt` into common / gpu / npu variants; declare 3 previously-hidden direct imports (jinja2, psutil, pyyaml); tighten `tensordict` pin. |
| Device accessor | 2 (`72b564a`, `7187b51`, `496d198`) | `verl/utils/device.py` helper (`is_npu_available`, `get_device_name`, `get_device_module`, `get_dist_backend`, `get_default_attn_implementation`, `get_ray_resource_name`, `get_visible_devices_env`). Sweep of 35 `torch.cuda.*` call sites; device-mesh / device_map / ROCm gate. |
| Attention backend | 2 (`6701a50`, `da2487f`, `ffafa0d`) | `attn_implementation` configurable, defaults to `sdpa` on NPU. Vendor flash-attn `bert_padding` helpers as pure-torch. NPU-aware config-level gate for `padding_free=True`. x86 unit tests. |
| Ray NPU | 3 (`fb1a223`, `59641d4`, `cc8e794`) | Register NPU as Ray custom resource. `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` (NPU-BUG-002). `VLLM_ASCEND_ENABLE_NZ=0` for RL param-sync. |
| Platform shims | 2 (`cbfe645`, `87faff1`, `2d8ee2c`) | `Dockerfile.npu` layered on verl-8.5.0-a3 with triton-ascend reinstall (NPU-BUG-001). `vllm.lora.models` → `vllm.lora.lora_model` compat. `get_tensor_model_parallel_group` → `get_tp_group` compat (vllm 0.13 rename). |
| Smoke harness | 1 (`906215d`, `72a7f22`) | `examples/qwen2_0_5b_math_grpo_npu_smoke.sh` (2 chip) and `..._4chip.sh` (4 chip). |

### 2.2 `zhshgmail/easyr1-npu` (this repo, private)

| Asset | Purpose |
|---|---|
| `docs/design.md` | Formal design doc (requirements, background, restrictions, tasks). |
| `docs/dep-matrix.md` | Per-package matrix across EasyR1 source / veRL source / 8.5.0 image / 8.5.2 image, plus Code-path blockers table. |
| `docs/porting-journal.md` | Dated log of every session. Full traceability. |
| `docs/npu-gap-plan.md` | v1 / v2+ / deferred work and rationale. |
| `docs/skills-design.md` | System design for the multi-repo port skills. |
| `docs/DELIVERABLE.md` | This file. |
| `knowledge/easyr1-master-deps.md` | EasyR1 source deps. |
| `knowledge/verl-master-deps.md` | veRL source deps (GPU + NPU reqs). |
| `knowledge/upstream-refs.md` | Maps each image to the matching upstream refs (avoid reviewing master). |
| `knowledge/npu-patterns.md` | **Stable-ID catalog** of code patterns (`NPU-CP-*`), platform bugs (`NPU-BUG-*`), env knobs (`NPU-ENV-*`), operational notes (`NPU-OPS-*`). |
| `knowledge/images/verl-8.5.0-a3.md` | Target image inventory. |
| `knowledge/images/verl-8.5.2-a3.md` | Newer image inventory (for v2 migration). |
| `scripts/run-npu-container.sh` | Launches a container with device passthrough + bind mounts + env defaults + chip-occupancy precheck. |
| `scripts/inspect-ascend-image.sh` | Automates the image inventory process including NPU-BUG-001 integrity check. |
| `scripts/code-path-sweep.sh` | Scans a python tree for GPU-only call sites, groups by NPU-CP-* ID. |
| `scripts/smoke_v11_device.py` | V1.1/V1.2 smoke (device accessors + tensor round-trip). |
| `scripts/smoke_v13_rollout.py` | V1.3 smoke (vllm_ascend rollout). |
| `skills/codex-review/` | Review any code/doc/plan via the local `codex` CLI. |
| `skills/upstream-branch-hygiene/` | "Local edit → push personal → remote pull" discipline. |
| `skills/npu-container-runner/` | Wraps `run-npu-container.sh` with a skill contract. |
| `skills/npu-image-inspect/` | Wraps `inspect-ascend-image.sh`. |
| `skills/npu-code-path-sweep/` | Wraps `code-path-sweep.sh`. |
| `skills/ray-npu-shim/` | Drop-in Python module `ray_npu_shim.py` for any Ray-based trainer → NPU. |

---

## 3. The 7 NPU-specific findings (all fixed + catalogued)

| ID | Class | What |
|---|---|---|
| `NPU-BUG-001` | Platform bug | `verl-8.5.0-a3` triton-ascend 3.2.0 is partially installed — `triton/__init__.py` missing → `import torch_npu` fails. Fix: force-reinstall in our Dockerfile layer. |
| `NPU-CP-001` | Code pattern | 35 `torch.cuda.*` / `"cuda"` / `nccl` call sites need routing through a device-accessor helper. |
| `NPU-CP-002` | Code pattern | `vllm.lora.models` renamed to `vllm.lora.lora_model` in vllm 0.13. Try/fallback import. |
| `NPU-CP-003` | Code pattern | Ray doesn't auto-detect NPU as `"GPU"` resource. 4-file fix: register NPU as custom resource, swap `num_gpus`/`"GPU"` lookups across helper + main + trainer + placement groups. |
| `NPU-CP-004` | Code pattern | `vllm.distributed.parallel_state.get_tensor_model_parallel_group` → `get_tp_group` in vllm 0.13. hasattr gate. |
| `NPU-BUG-002` | Platform bug | Ray 2.55+ clears `ASCEND_RT_VISIBLE_DEVICES` inside actors when `num_gpus=0/None`. Fix: `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` in runtime_env. |
| `NPU-ENV-001` / `NPU-ENV-002` | Env vars | `HF_ENDPOINT=https://hf-mirror.com` (CN network). `VLLM_ASCEND_ENABLE_NZ=0` (FRACTAL_NZ layout drift during RL param sync). Both auto-injected by `run-npu-container.sh`. |
| `NPU-OPS-001` / `NPU-OPS-002` | Operational | Editable install target shadows bind-mounted source. Stale `__pycache__` after live source swap. Both fixed in `run-npu-container.sh`. |

Each entry in `knowledge/npu-patterns.md` has: symptom, root cause, fix, commit ref, generalizable rule.

---

## 4. Smoke results

| Test | Status | Notes |
|---|---|---|
| **V1.1** device accessors inside container | ✅ | `torch.npu.is_available()` True; all 6 accessors resolve correctly. |
| **V1.2** tensor + `attention_utils.py` round-trip on NPU | ✅ | Vendored `unpad_input` / `pad_input` correctness verified against real torch_npu. |
| **V1.3** vllm_ascend rollout | ✅ | Qwen2-0.5B generated coherent text, ~42 tok/s single chip. |
| **V1.4** GRPO training 2 steps, 2 chips | ✅ | 8m24s, FSDP world_size=2, HCCL, checkpoint saved. entropy_loss 0.991 → 1.263. |
| **V1.5** GRPO training 2 steps, 4 chips (2 A3 cards) | pending | Launched 2026-04-18; result not captured in this doc yet. |

---

## 5. How to reproduce (on a fresh A3 host)

1. Clone: `git clone git@github.com:zhshgmail/easyr1-npu.git repo` and `git clone -b ascend-port git@github.com:zhshgmail/EasyR1.git upstream/EasyR1`. On a CN host set up an askpass helper for private repos — see `a3_server.md`.
2. Pull the base image: `docker pull quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest` (14 GB, ≈2 min from quay.io).
3. Build the layered image: `cd upstream/EasyR1 && docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .` (triton-ascend reinstall is baked in).
4. Download a small model: `HF_ENDPOINT=https://hf-mirror.com python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-0.5B-Instruct', local_dir='/data/$USER/models/Qwen2-0.5B-Instruct')"`.
5. Smoke up: `bash repo/scripts/run-npu-container.sh --chips 0,1 -- python3 /home/$USER/workspace/easyr1-npu/repo/scripts/smoke_v11_device.py` (V1.1). Then V1.3 rollout, then V1.4 training.
6. V1.4: `bash repo/scripts/run-npu-container.sh --chips 0,1 -- bash -c 'cd /home/$USER/workspace/easyr1-npu/upstream/EasyR1 && bash examples/qwen2_0_5b_math_grpo_npu_smoke.sh'`.

---

## 6. Known limitations (v1 scope)

- `padding_free=True` raises `NotImplementedError` on NPU. Uses SDPA which doesn't support the flat-packed varlen layout. Covered by v2 (§7.1).
- `ulysses_size > 1` raises `NotImplementedError` on NPU. Requires NPU-native `flash_attn_varlen_func` equivalent. Covered by v2 (§7.2).
- Perf is unoptimized: `enforce_eager=true`, no inductor graph. See v2 (§7.4).
- 8.5.2 image migration not attempted. Transformers 5.x + huggingface_hub 1.x compatibility sweep needed. Covered by v2 (§7.3).
- LoRA rollout path touched via `vllm_utils.py` compat but not exercised end-to-end. If someone uses LoRA first thing, there may be further issues.
- Single-node only. Multi-node scaling unverified (out of v1 per `design.md §1.3`).

---

## 7. v2 / follow-up work (prioritized)

1. **NPU varlen attention → enables `padding_free=True`**. Write a `_custom_flash_attention_forward` variant backed by `torch_npu.npu_fusion_attention` with `cu_seqlens` mapped to `actual_seq_{q,kv}len`. Un-gate in `monkey_patch.apply_ulysses_patch`.
2. **NPU-aware Ulysses SP**. Prerequisite: (1). Then the existing Ulysses collectives (`gather_seq_scatter_heads`) just need HCCL variants.
3. **8.5.2 image migration**. Transformers 4.57.6 → 5.3.0.dev0, huggingface_hub 0.36 → 1.11, vllm_ascend 0.13 → 0.17. Likely re-expands the NPU-CP catalog.
4. **Perf pass**. Turn off `enforce_eager`, enable inductor graph compile, benchmark. Compare against veRL NPU baseline if available.
5. **Liger on NPU**. Only if perf pass identifies fused kernel gaps big enough to matter. Likely a triton-ascend project.

---

## 8. How the knowledge carries over to the next port

The scaffold is built so the **next framework port** (OpenRLHF, TRL, custom) gets a head start:

- `skills/npu-image-inspect/` — run it on any new Ascend image, get an inventory doc.
- `skills/npu-code-path-sweep/` — run it on any new framework's source, get the CP-* hit list.
- `skills/ray-npu-shim/` — copy `ray_npu_shim.py` into a Ray-based trainer; 4-5 call-site swaps and you're NPU-compatible.
- `skills/npu-container-runner/` — reuse `run-npu-container.sh` as-is; only the `--image` and bind-mount paths change.
- `skills/upstream-branch-hygiene/` — the operational rule for any multi-repo port.
- `knowledge/npu-patterns.md` — add new NPU-CP/NPU-BUG/NPU-ENV IDs as they surface.

The expectation is that a second port takes **much less than 1 day**, because 80%+ of the surprises in this port are in the catalog.

---

## 9. Sign-off checklist

- [ ] User reviewed this document.
- [ ] V1.5 finished and results appended to §4 and `porting-journal.md`.
- [ ] User decides on v2 priority (§7).
