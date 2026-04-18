# NPU port patterns and platform bugs — catalog

> Stable-ID catalog of recurring code patterns and platform bugs encountered during NPU ports. Modeled after a5_ops's `PATTERN_INDEX.md` / `ERROR_CORRECTIONS.md`.
>
> **ID scheme**:
> - `NPU-CP-NNN` — **code patterns**: places in source that need a port-specific change. Each entry is a class of call site plus the standard fix.
> - `NPU-BUG-NNN` — **platform bugs**: defects in the NPU stack (images, vendor builds, upstream renames) that are not our code but we have to work around.
> - `NPU-ENV-NNN` — **environment/configuration**: things that need to be set/unset in the runtime environment, not source.
> - `NPU-OPS-NNN` — **operational**: host-level / workflow rules (shared-host discipline, disk pressure, editable-install bind mounts).
>
> **Uniform schema**: every entry has `Symptom`, `Root cause`, `Fix`, `Commit ref`, `Generalizable rule`. If a field is not applicable (e.g. operational rules rarely have a commit ref), write `—` explicitly so readers can tell it was considered.
>
> **When to cite**: in commit messages, journal entries, skill reports. A fresh session should be able to read this file, grep a pattern ID, and land the same fix in a new repo without re-deriving it.

---

## Code patterns (NPU-CP-NNN)

### NPU-CP-001 — `torch.cuda.*` / `"cuda"` device strings / `.cuda()` tensor method

**Pattern**: any CUDA-named API. Covers several facets that all share the same fix family:
- namespace calls: `torch.cuda.empty_cache()`, `torch.cuda.current_device()`, `torch.cuda.get_rng_state()`, `torch.cuda.mem_get_info()`, `torch.cuda.max_memory_allocated()`, `torch.cuda.set_device()`, `torch.cuda.get_device_name()`, `torch.cuda.manual_seed()`, `torch.cuda.set_rng_state()`.
- tensor method: `.cuda()`.
- string device specs: `torch.device("cuda")`, `init_device_mesh("cuda", ...)`, `device="cuda"`, `device_map="cuda"`.
- visibility env: `CUDA_VISIBLE_DEVICES` literal.
- distributed backend literal: `backend="nccl"` in `init_process_group`.

**Symptom**: `AttributeError: module 'torch' has no attribute 'cuda'.<op>` when `torch_npu` is active, or no-op behavior because no CUDA device exists. For `backend="nccl"`: `init_process_group` fails because HCCL must be used on NPU.

**Root cause**: `torch.cuda` and the `"cuda"` device string both hardwire the CUDA backend. NPU uses `torch.npu` (registered as PrivateUse1 by `torch_npu`) and the `"npu"` device string.

**Fix**: introduce a helper module (`verl/utils/device.py` in EasyR1) exposing:
- `is_npu_available()` — lru-cached.
- `get_device_module()` — returns `torch.npu` on NPU, `torch.cuda` otherwise.
- `get_device_name()` — returns `"npu"` or `"cuda"`.
- `get_dist_backend()` — returns `"hccl"` or `"nccl"`.
- `get_visible_devices_env()` — returns `"ASCEND_RT_VISIBLE_DEVICES"` or `"CUDA_VISIBLE_DEVICES"`.
- `get_default_attn_implementation()` — returns `"sdpa"` on NPU, `"flash_attention_2"` on CUDA.

Then replace every `torch.cuda.<op>(...)` with `get_device_module().<op>(...)`, every `"cuda"` device spec with `get_device_name()`, every `"nccl"` with `get_dist_backend()`.

**Commit ref**: `72b564a` (helper introduction) + `7187b51` (sweep of 35 sites across 10 files) + `496d198` (fix missed `flat_param_to("cuda")` + bare-int `current_device()` wrap).

**Generalizable rule**: run `repo/scripts/code-path-sweep.sh <source>` — this pattern is sub-divided into ~5 sub-facets there, all with the same fix family. 80%+ of the callsites in a Ray-based RL trainer hit this.

---

### NPU-CP-002 — `vllm.lora.models.LoRAModel` moved in vllm 0.13

**Pattern**: `from vllm.lora.models import LoRAModel` (and similar `vllm.lora.*` top-level submodule imports).

**Symptom**: `ModuleNotFoundError: No module named 'vllm.lora.models'` at module load, which cascades through every import of the affected module.

**Root cause**: vllm 0.13 upstream rename — class moved to `vllm.lora.lora_model` as part of a `LoRAModel` / `LoRAModelManager` refactor. Not NPU-specific but shows up because NPU images ship vllm 0.13+.

**Fix**: try-new-path / fallback-old-path import:
```python
try:
    from vllm.lora.lora_model import LoRAModel  # vllm >= 0.13
except ImportError:
    from vllm.lora.models import LoRAModel  # vllm <= 0.12
```
Also: move sibling `vllm.lora.worker_manager` / `vllm.lora.utils` imports into the function body that uses them, so module load doesn't force eager attribute lookups on a vllm build whose internals have moved.

**Commit ref**: `87faff1` (in `verl/utils/vllm_utils.py`).

**Generalizable rule**: **vllm 0.12 → 0.13 is a deep internal rename**. For any code integrating vllm via non-public APIs (lora, distributed, executor, model_executor internals), audit every `from vllm.<submod> import ...` against the installed version. See also NPU-CP-004.

---

### NPU-CP-003 — Ray doesn't auto-detect Ascend NPU

**Pattern**: Ray-based frameworks using any of: `{"GPU": 1}` placement-group bundles, `ray.available_resources().get("GPU", 0)`, `options["num_gpus"] = n` actor option.

**Symptom**: `ValueError: Total available GPUs 0 is less than total desired GPUs N` during worker/actor init, despite `torch.npu.device_count()` returning N > 0.

**Root cause**: Ray auto-detects CUDA devices as the builtin `"GPU"` resource with `num_gpus` sugar. Ascend chips don't get that sugar — Ray must be told explicitly via `ray.init(resources={"NPU": n})` and claimed via `options["resources"]={"NPU": n}`.

**Fix**: four coordinated edits (canonical implementation in `repo/skills/ray-npu-shim/ray_npu_shim.py`):
1. `get_ray_resource_name()` → `"NPU"` on NPU, `"GPU"` otherwise.
2. `ray.init(..., resources={"NPU": torch.npu.device_count()})` on NPU.
3. `available_resources().get(get_ray_resource_name(), 0)` for count lookups.
4. Actor spawn: `options["resources"]={"NPU": n}` on NPU (NOT `options["num_gpus"]`); placement-group bundles use `{resource_name: 1}`.

**Commit ref**: `fb1a223` in EasyR1 (across `device.py` / `trainer/main.py` / `trainer/ray_trainer.py` / `single_controller/ray/base.py`).

**Generalizable rule**: Ray on non-CUDA accelerators is opt-in. Any Ray-based training framework (veRL, EasyR1, OpenRLHF) needs this 4-site fix. Use `skills/ray-npu-shim/` as a drop-in to avoid re-deriving.

---

### NPU-CP-004 — `vllm.distributed.parallel_state.get_tensor_model_parallel_group` renamed

**Pattern**: `vllm.distributed.parallel_state.get_tensor_model_parallel_group()` (and parallel renames for other groups like `get_pipeline_model_parallel_group`).

**Symptom**: `AttributeError: module 'vllm.distributed.parallel_state' has no attribute 'get_tensor_model_parallel_group'`.

**Root cause**: vllm 0.13 upstream rename. Replacement is `get_tp_group()`. The returned `GroupCoordinator` object's `.device_group` attribute is unchanged.

**Fix**:
```python
if hasattr(vllm_ps, "get_tp_group"):
    self.tp_group = vllm_ps.get_tp_group().device_group
else:
    self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group
```

**Commit ref**: `2d8ee2c` in `verl/workers/sharding_manager/fsdp_vllm.py`.

**Generalizable rule**: sibling of NPU-CP-002. Any vllm internal API used by a framework needs a version-aware shim.

---

### NPU-CP-005 — `flash_attn` / `liger_kernel` imports (CUDA-only kernels)

**Pattern**: `from flash_attn import ...` / `from liger_kernel import ...` / `import flash_attn` / `import liger_kernel`.

**Symptom**: `ImportError: No module named 'flash_attn'` (or `liger_kernel`) at module load — these packages are CUDA-only and NPU images don't ship them.

**Root cause**: flash-attn's kernels are CUDA-specific (Triton + CUDA C++); liger-kernel's Triton kernels compile only for CUDA backend as of writing. Both packages are intentionally absent from NPU images.

**Fix**: vary by sub-case:
- **Import-time guard**: wrap in `try/except ImportError: pass` if the symbols are only used on a code path that also checks `is_npu_available()`. EasyR1 did this for `flash_attn.bert_padding` until we went further.
- **Vendor the pure-torch parts**: many "flash-attn" utilities (the `bert_padding` helpers — `index_first_axis`, `pad_input`, `unpad_input`) are actually just pure-torch autograd functions that happen to ship inside flash-attn. Vendor them directly as a local module. See `verl/utils/npu_flash_attn_utils.py` + `verl/utils/attention_utils.py` façade.
- **Drop the feature on NPU**: for truly kernel-bound optimizations (liger's fused kernels), default to the torch implementation on NPU; defer a triton-ascend port to v2+.

**Commit ref**: `da2487f` (vendor `bert_padding` helpers + façade) + `6701a50` (SDPA default instead of flash_attention_2 on NPU).

**Generalizable rule**: any framework that imports flash_attn / liger_kernel / xformers / deepspeed CUDA ops at module scope needs an NPU-specific import or vendor path. `skills/npu-code-path-sweep/` detects these imports.

---

### NPU-CP-006 — `torch.backends.cuda.*` knobs are no-ops on NPU

**Pattern**: `torch.backends.cuda.matmul.allow_tf32 = False` / `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` / other `torch.backends.cuda.*` numerical-precision toggles.

**Symptom**: the knobs silently do nothing on NPU; the author's intent (precision stability) is not preserved. On some torch builds an `AttributeError` may be raised if the namespace is lazily constructed and CUDA isn't available.

**Root cause**: `torch.backends.cuda` configures CUDA-kernel behavior. NPU matmul uses its own precision modes (set via `torch.npu.set_compile_mode` or via CANN env vars), and the CUDA knobs don't influence it.

**Fix**: wrap the knob behind `if not is_npu_available():`. On NPU, document the equivalent precision control if known; otherwise note that TF32-off has no NPU counterpart (the default NPU bf16 path is already reduced-precision).

**Commit ref**: `7187b51` (in `verl/workers/fsdp_workers.py`).

**Generalizable rule**: never leave a `torch.backends.cuda.*` knob unguarded in a multi-accelerator codebase; also don't assume the numerical-stability intent transfers.

---

## Platform bugs (NPU-BUG-NNN)

### NPU-BUG-001 — `verl-8.5.0-a3` base image: triton-ascend install partial

**Symptom**: `import torch_npu` fails with `ImportError: cannot import name 'Config' from 'triton'`. The filesystem has `/usr/local/python3.11.14/.../site-packages/triton/` directory but no `__init__.py` (even though its dist-info `RECORD` lists it). Also missing `triton/compiler` submodules.

**Root cause**: the vllm-ascend `Dockerfile.a3` build sequence installs vllm first (pulling upstream triton), then `pip uninstall -y triton`, then reinstalls triton-ascend. In this image revision, the reinstall silently didn't populate all top-level files. Reproduced in both `verl-8.5.0-a3` and `verl-sglang-8.3.rc1-a3` images — class of bug, not one-off.

**Fix**: in our Dockerfile, force-reinstall triton-ascend as an early layer:
```dockerfile
RUN pip install --no-cache-dir --force-reinstall --no-deps triton-ascend==3.2.0 || \
    pip install --no-cache-dir --force-reinstall --no-deps \
      --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi triton-ascend==3.2.0
```

**Commit ref**: `cd16649` in `upstream/EasyR1/Dockerfile.npu`.

**Generalizable rule**: before trusting any new Ascend base image, run `skills/npu-image-inspect/` — its integrity check detects this class of bug and emits the exact `pip install --force-reinstall` line.

---

### NPU-BUG-002 — Ray 2.55 clears `ASCEND_RT_VISIBLE_DEVICES` on actor spawn

**Symptom**: driver-side `torch.npu.is_available() == True, device_count == 2`, but inside a Ray actor `False, 0` with `os.environ["ASCEND_RT_VISIBLE_DEVICES"] == ""`. Downstream code relying on `is_npu_available()` falls back to the CUDA path and emits misleading CUDA-named error messages.

**Root cause**: Ray 2.55+ defensively clears `{CUDA,ASCEND_RT,HABANA,NEURON_RT}_VISIBLE_*` env vars on actor spawn when `num_gpus=0/None`. Our NPU actors claim chips via `options["resources"]={"NPU": n}`, not via `num_gpus`, so Ray wipes the visibility list. Ray itself warns about the upcoming default flip and surfaces the override knob.

**Fix**: set `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` in `ray.init(runtime_env={"env_vars": ...})`.

**Commit ref**: `59641d4` in `verl/trainer/main.py`.

**Generalizable rule**: always set this env var when using Ray 2.55+ on NPU (or HPU/Neuron/TPU). Encoded as a default in `skills/ray-npu-shim/ray_npu_shim.py::_npu_runtime_env_defaults`.

---

## Environment / configuration (NPU-ENV-NNN)

### NPU-ENV-001 — container needs `HF_ENDPOINT=https://hf-mirror.com`

**Symptom**: dataset/model download retries 5× then fails with `HTTPSConnectionPool(host='huggingface.co', port=443): ... Network is unreachable`.

**Root cause**: A3 host is in China; `huggingface.co` is not reachable from the host.

**Fix**: `run-npu-container.sh` injects `HF_ENDPOINT=https://hf-mirror.com` and `HF_HOME=/data/<user>/hf-cache` (so cache survives container teardown) by default. Overridable via caller env.

**Commit ref**: `5820f5f` in `repo/scripts/run-npu-container.sh`.

**Generalizable rule**: any CN-side training run needs the HF mirror env var. If `HF_HUB_ENABLE_HF_TRANSFER=1` is set but the `hf_transfer` package isn't installed, flip the env to 0 or install it.

---

### NPU-ENV-002 — RL training needs `VLLM_ASCEND_ENABLE_NZ=0`

**Symptom**: `ValueError: FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios. Please set VLLM_ASCEND_ENABLE_NZ=0.` raised by vllm-ascend during rollout engine prepare.

**Root cause**: vllm-ascend's default FRACTAL_NZ weight layout re-packs parameters when the rollout engine loads them. In RL, the actor writes fresh params into the rollout engine every step; NZ re-packing changes numerical values slightly, so importance ratios and KL/rewards drift. vllm-ascend detects this and bails with a prescriptive error.

**Fix**: export `VLLM_ASCEND_ENABLE_NZ=0` in the training runtime env. For Ray-based trainers, add it to `ray.init(runtime_env={"env_vars": ...})`.

**Commit ref**: `cc8e794` in `verl/trainer/main.py`; also default-on in `repo/scripts/run-npu-container.sh`.

**Generalizable rule**: any RL scenario using vllm-ascend **must** set this. Inference-only workloads (no in-training weight updates) can keep default NZ for perf.

---

### NPU-ENV-003 — HCCL determinism flags default off

**Symptom**: two otherwise-identical runs on the same hardware produce different reward/loss trajectories; bit-for-bit reproduction of a prior run is impossible.

**Root cause**: the verl-A3 base image sets `LCCL_DETERMINISTIC=0 LCCL_PARALLEL=0` — LCCL (a component under HCCL) runs in its faster non-deterministic mode.

**Fix**: set both to 1 in `runtime_env.env_vars` for reproducibility-sensitive runs:
```python
"LCCL_DETERMINISTIC": "1",
"LCCL_PARALLEL": "1",
```
Expect measurably slower collectives. Not on by default in our harness.

**Commit ref**: —

**Generalizable rule**: for published-result or debugging-regressions runs, flip these flags. Document throughput delta when measured.

---

### NPU-ENV-004 — RNG state portability across accelerators

**Symptom**: training resumed from a CUDA-saved checkpoint on NPU (or vice versa) produces different numerics from step 0 onward.

**Root cause**: the RNG state saved under the `"accelerator"` key is a device-specific byte blob. `torch.npu.set_rng_state` and `torch.cuda.set_rng_state` use incompatible formats.

**Fix**: our checkpoint manager stores under `"accelerator"` and falls back to the old `"cuda"` key on load. This keeps the code path forward-compatible but does NOT give bit-for-bit cross-accelerator reproducibility. Accept that cross-accelerator resumes are best-effort.

**Commit ref**: `496d198` (checkpoint RNG key migration).

**Generalizable rule**: do not claim reproducibility across CUDA/NPU. Document resume behavior as "same accelerator family only." Cross-family resumption is a per-framework design question; out of v1 scope.

---

## Operational (NPU-OPS-NNN)

### NPU-OPS-001 — editable install target shadows bind-mounted source

**Symptom**: code change pushed to personal fork, pulled on A3, but container still runs old code. `python3 -c "import verl; print(verl.__file__)"` shows `/opt/easyr1/verl/__init__.py` (build-time COPY snapshot), not the live `upstream/EasyR1/`.

**Root cause**: `Dockerfile.npu` does `COPY . /opt/easyr1 && pip install -e .`. The editable install writes a `.pth` file pointing at `/opt/easyr1`, which is the image's COPY layer. A bind mount over `/opt/easyr1` is required to make host-side `git pull` take effect without rebuild.

**Fix**: `run-npu-container.sh` bind-mounts `upstream/EasyR1/` over `/opt/easyr1`. `git pull` on the host = next container spawn picks up the change. Only Dockerfile changes need a `docker build`.

**Commit ref**: `1595af4` in `repo/scripts/run-npu-container.sh`.

**Generalizable rule**: whenever a Dockerfile does `COPY . <path> && pip install -e <path>`, add a bind-mount over `<path>` in the runner script. Encoded in `skills/upstream-branch-hygiene/` + `skills/npu-container-runner/`.

---

### NPU-OPS-002 — stale `__pycache__` after source swap

**Symptom**: source visible inside container is updated, but Python picks up an old `.pyc`.

**Root cause**: `__pycache__/*.pyc` persists in the bind-mounted source dir. If the source file's mtime doesn't advance (or filesystem resolution is coarser than the .pyc compile timestamp), Python's import-cache validation fails open.

**Fix**: set `PYTHONDONTWRITEBYTECODE=1` in the container runner so `.pyc` files never get written to the bind-mount. Default-on in `run-npu-container.sh`. Alternative: clear pycache dirs before each run.

**Commit ref**: `0da8db2` in `repo/scripts/run-npu-container.sh` (added `PYTHONDONTWRITEBYTECODE=1` env default).

**Generalizable rule**: pair every live-bind-mounted development setup with either `PYTHONDONTWRITEBYTECODE=1` or a pre-run pycache clear.

---

### NPU-OPS-003 — shared-host chip contention

**Symptom**: our training job dies with HBM-allocation errors, or quietly performs worse than expected; `npu-smi info` shows another user's process on a chip we claimed.

**Root cause**: A3 host at `115.190.166.102` is shared with other users (`/home/baymax`, `/home/wjq`, `/home/lynn`, `/home/chj`, ...). Nothing in the docker runtime or driver enforces exclusive chip allocation.

**Fix**: `run-npu-container.sh` prechecks each requested chip via `npu-smi info -t proc-mem -i <card>`. If any chip has a non-self process holding HBM, abort with a clear message pointing at the `npu-smi` output. Override with `--skip-chip-check` only when the caller knows the holding process is their own.

**Commit ref**: `0da8db2` in `repo/scripts/run-npu-container.sh`.

**Generalizable rule**: on any shared NPU host, always check occupancy before claiming chips. Never kill another user's process. If your experiment **must** have specific chips (e.g. for HCCS topology reasons), coordinate with the user; otherwise pick different chips.

---

### NPU-OPS-004 — disk pressure on a shared host

**Symptom**: `docker pull` fails with `no space left on device`, or a training job dies during checkpoint save.

**Root cause**: A3 root fs was 93% used at onboarding, with ~258 GB free. Each docker image (14-18 GB), each weight download (0.5-14 GB for Qwen models), and each checkpoint (model size × FSDP shard count) competes for the same `/` partition. `/var/lib/docker` (images) and `/data` (weights/checkpoints per user convention) both live on `/`.

**Fix**: prune docker images aggressively (`docker image prune -a`), keep only 1-2 weight sets at a time, move long-running checkpoint directories off-host when possible. Use `/data/<user>/` for anything large (convention, not enforcement).

**Commit ref**: —

**Generalizable rule**: on shared hosts, monitor `df -h /` before any large artifact operation. Don't delete other users' files to make room — ask instead.

---

## Referencing from commit messages

Commit messages should cite IDs when the change implements a known pattern. Example:

```
NPU-CP-003: register Ascend NPU as custom Ray resource

Ray auto-detects CUDA as "GPU" but not NPU. Apply the 4-site fix
from repo/knowledge/npu-patterns.md (helper, ray.init, resource
check, placement-group bundles + actor spawn options).
```

This lets `git log --grep=NPU-CP-003` find every application of the pattern across repos and forks, and lets future sessions `grep -r NPU-CP-` to see where patterns land.

---

## IDs defined (summary)

- Code patterns: `NPU-CP-001` ... `NPU-CP-006` (6 entries)
- Platform bugs: `NPU-BUG-001` ... `NPU-BUG-002` (2 entries)
- Environment/config: `NPU-ENV-001` ... `NPU-ENV-004` (4 entries)
- Operational: `NPU-OPS-001` ... `NPU-OPS-004` (4 entries)

**Total: 16 stable IDs** across 4 categories, each with uniform `Symptom / Root cause / Fix / Commit ref / Generalizable rule` schema.
