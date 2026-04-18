# NPU port patterns and platform bugs — catalog

> Stable-ID catalog of recurring code patterns and platform bugs encountered during NPU ports. Modeled after a5_ops's `PATTERN_INDEX.md` / `ERROR_CORRECTIONS.md`.
>
> **ID scheme**:
> - `NPU-CP-NNN` — **code patterns**: places in source that need a port-specific change. Each entry is a class of call site plus the standard fix.
> - `NPU-BUG-NNN` — **platform bugs**: defects in the NPU stack (images, vendor builds, upstream renames) that are not our code but we have to work around.
> - `NPU-ENV-NNN` — **environment/configuration**: things that need to be set/unset in the runtime environment, not source.
>
> **When to cite**: in commit messages, journal entries, skill reports. A fresh session should be able to read this file, grep a pattern ID, and land the same fix in a new repo without re-deriving it.

---

## Code patterns (NPU-CP-NNN)

### NPU-CP-001 — `torch.cuda.*` → device-module accessor

**Pattern**: calls like `torch.cuda.empty_cache()`, `torch.cuda.current_device()`, `torch.cuda.get_rng_state()`, `torch.cuda.mem_get_info()`, `torch.cuda.max_memory_allocated()`, `torch.cuda.set_device()`, `torch.cuda.get_device_name()`, `torch.cuda.manual_seed()`, `torch.cuda.set_rng_state()`.

**Symptom on NPU**: `AttributeError` or no-op; on newer torch the call doesn't error but does nothing useful because the runtime has no CUDA device.

**Fix**: introduce a tiny helper module (`verl/utils/device.py` in EasyR1) exposing `get_device_module()` that returns `torch.npu` on NPU, `torch.cuda` otherwise. Replace every `torch.cuda.<op>(...)` with `get_device_module().<op>(...)`. Use `get_device_name()` (returns `"npu"`/`"cuda"`) for calls that take a device string (`torch.device(...)`, `init_device_mesh("cuda", ...)`, `.to("cuda")`).

**Where applied in EasyR1**: commits `72b564a` (helper) + `7187b51` (sweep of 35 sites) + `496d198` (missed `flat_param_to("cuda")` + bare-int `current_device()` wrap).

**Rule**: search the codebase for `torch.cuda.`, `init_device_mesh("cuda"`, `device_map="cuda"`, `"cuda"`, `.cuda(` and classify each hit. Add to the sweep tool (`scripts/dep-diff.py` or a new `scripts/code-path-sweep.py`).

---

### NPU-CP-002 — `vllm.lora.models.LoRAModel` moved

**Pattern**: `from vllm.lora.models import LoRAModel`.

**Symptom on NPU**: `ModuleNotFoundError: No module named 'vllm.lora.models'` at module load time, which breaks every downstream import transitively.

**Root cause**: not NPU-specific — this is a **vllm 0.13 upstream rename**. The class moved to `vllm.lora.lora_model` as part of a `LoRAModel` / `LoRAModelManager` split. Still a port issue because the NPU images ship vllm 0.13+.

**Fix**: try-new-path / fallback-old-path import. Also move sibling vllm.lora imports (`worker_manager`, `utils`) inside the function body that uses them, so module load doesn't trigger eager attribute lookups on a vllm build whose internals have moved.

**Where applied**: commit `87faff1` in `verl/utils/vllm_utils.py`.

**Rule**: **vllm 0.12 → 0.13 is a deep internal rename**. Grep the target codebase for `from vllm.` / `import vllm.` and for each, verify the symbol exists on the actual installed vllm version. The same class of breakage will hit `vllm.distributed.parallel_state`, `vllm.lora.*`, `vllm.executor.*`. See also NPU-CP-004.

---

### NPU-CP-003 — Ray doesn't auto-detect NPU

**Pattern**: Ray-based frameworks assume placement groups can use `{"GPU": n}` bundles and `ray.available_resources().get("GPU", 0)` to count accelerators; Ray's `num_gpus` actor option is CUDA-only.

**Symptom on NPU**: `ValueError: Total available GPUs 0 is less than total desired GPUs N` during `init_workers`, even though NPU is available and `torch.npu.device_count()` returns N.

**Root cause**: Ray auto-detects CUDA devices as builtin `"GPU"` resource. Ascend chips need to be registered explicitly via `ray.init(resources={"NPU": N})` and requested via `options["resources"]={"NPU": n}` in actor spawn (NOT `options["num_gpus"]`).

**Fix across 4 sites** (EasyR1 example):
1. Add `get_ray_resource_name()` → `"NPU"`/`"GPU"` helper.
2. `ray.init(...)` on NPU passes `resources={"NPU": torch.npu.device_count()}`.
3. `_check_resource_available` reads `available_resources()[resource_name]`.
4. Placement-group bundles use `{resource_name: 1}`; actor spawn uses `options["resources"]={"NPU": n}` on NPU, `options["num_gpus"]=n` on CUDA.

**Where applied**: commit `fb1a223` across `device.py` / `trainer/main.py` / `trainer/ray_trainer.py` / `single_controller/ray/base.py`.

**Rule**: Ray on non-CUDA accelerators is **opt-in**. Any RL/training framework that uses Ray (veRL, EasyR1, OpenRLHF) will need this 4-site fix. Candidate for a reusable helper that we ship alongside our device module.

---

### NPU-CP-004 — `vllm.distributed.parallel_state.get_tensor_model_parallel_group` renamed

**Pattern**: `vllm.distributed.parallel_state.get_tensor_model_parallel_group()` (and similar `get_X_model_parallel_group` functions).

**Symptom on NPU**: `AttributeError: module 'vllm.distributed.parallel_state' has no attribute 'get_tensor_model_parallel_group'`.

**Root cause**: **vllm 0.13 upstream rename**. Replacement is `get_tp_group()` (and parallel renames for other groups). The returned `GroupCoordinator` object's `.device_group` attribute is unchanged.

**Fix**: hasattr gate + fallback to old name.

**Where applied**: commit `2d8ee2c` in `verl/workers/sharding_manager/fsdp_vllm.py`.

**Rule**: any code integrating vllm via low-level distributed hooks needs a vllm-version-aware import shim. See also NPU-CP-002 which is the same pattern family.

---

## Platform bugs (NPU-BUG-NNN)

### NPU-BUG-001 — `verl-8.5.0-a3` base image: triton-ascend install partial

**Symptom**: `import torch_npu` fails with `ImportError: cannot import name 'Config' from 'triton'`. The filesystem has `/usr/local/python3.11.14/.../site-packages/triton/` directory but no `__init__.py` (even though its dist-info `RECORD` lists it).

**Root cause**: the vllm-ascend `Dockerfile.a3` build sequence installs vllm first (which pulls upstream triton), then `pip uninstall -y triton` to clear GPU triton, then reinstalls triton-ascend. In that image revision the triton-ascend reinstall silently didn't replace all top-level files. Also present in the adjacent `verl-sglang-8.3.rc1-a3` image — class of bug, not one-off.

**Fix**: in our Dockerfile, force-reinstall triton-ascend as an early layer:
```dockerfile
RUN pip install --no-cache-dir --force-reinstall --no-deps triton-ascend==3.2.0 || \
    pip install --no-cache-dir --force-reinstall --no-deps \
      --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi triton-ascend==3.2.0
```

**Where applied**: commit (in EasyR1's Dockerfile.npu).

**Rule**: before trusting a new Ascend base image, sanity-check `import torch_npu` actually works inside it. If not, the triton-ascend reinstall is the most common repair. Candidate skill: `smoke-ascend-image` to run this check automatically.

---

### NPU-BUG-002 — Ray 2.55 clears `ASCEND_RT_VISIBLE_DEVICES` on actor spawn

**Symptom**: driver sees `torch.npu.is_available() == True`, actor sees `False`, `device_count == 0`. `os.environ["ASCEND_RT_VISIBLE_DEVICES"] == ""` inside the actor. Downstream code relying on `is_npu_available()` falls back to the CUDA path and emits misleading CUDA error messages.

**Root cause**: Ray 2.55+ defensively clears `{CUDA,ASCEND_RT,HABANA,NEURON_RT}_VISIBLE_*` env vars when an actor is spawned with `num_gpus=0` or `None`. Our Runner actors claim NPU via `options["resources"]={"NPU": n}`, NOT via `num_gpus`, so Ray wipes the visibility list.

Ray itself surfaces a warning: *"In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0"*.

**Fix**: set `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` in `ray.init(runtime_env={"env_vars": ...})`.

**Where applied**: commit `59641d4` in `verl/trainer/main.py`.

**Rule**: always set this env var when using Ray + a non-CUDA accelerator (NPU, HPU, Neuron, TPU). Candidate for ray-runtime-env template in skills.

---

## Environment / configuration (NPU-ENV-NNN)

### NPU-ENV-002 — RL training needs `VLLM_ASCEND_ENABLE_NZ=0`

**Symptom**: `ValueError: FRACTAL_NZ mode is enabled. This may cause model parameter precision issues in the RL scenarios. Please set VLLM_ASCEND_ENABLE_NZ=0.` raised by vllm-ascend during rollout engine prepare.

**Root cause**: vllm-ascend's default FRACTAL_NZ weight layout re-packs parameters when the rollout engine loads them. In RL, the actor writes fresh parameters into the rollout engine every step; NZ re-packing changes numerical values slightly, so importance ratios and KL/rewards drift from what the actor produced. vllm-ascend itself detects this and bails with a prescriptive error.

**Fix**: export `VLLM_ASCEND_ENABLE_NZ=0` in the training runtime env. For Ray-based trainers, add it to `ray.init(runtime_env={"env_vars": ...})`.

**Where applied**: commit `cc8e794` in `verl/trainer/main.py`.

**Rule**: **any RL scenario using vllm-ascend must set this**. Inference-only (single pass, no in-training weight updates) can keep default NZ for perf.

---

### NPU-ENV-001 — container needs `HF_ENDPOINT=https://hf-mirror.com`

**Symptom**: dataset/model download retries 5x then fails with `HTTPSConnectionPool(host='huggingface.co', port=443) ... Network is unreachable`.

**Root cause**: A3 host is in China, `huggingface.co` is not reachable. Our container runner didn't inherit any `HF_ENDPOINT` override.

**Fix**: `run-npu-container.sh` injects `HF_ENDPOINT=https://hf-mirror.com` and `HF_HOME=/data/z00637938/hf-cache` (so cache survives container teardown) by default. Overridable via caller env.

**Where applied**: commit `5820f5f` in `repo/scripts/run-npu-container.sh`.

**Rule**: any CN-side training run needs the HF mirror env var. Also note the host needs `hf_transfer` python package if `HF_HUB_ENABLE_HF_TRANSFER=1` is set — ours defaults to 1 but if the mirror user doesn't have that package installed, export `HF_HUB_ENABLE_HF_TRANSFER=0` or install the package.

---

## Operational (how the port work is organized)

### NPU-OPS-001 — editable install target shadows bind-mounted source

**Symptom**: code change pushed to personal fork, pulled on A3, but container still runs old code. Python's `import verl` resolves to `/opt/easyr1/verl/__init__.py` (the build-time COPY snapshot), not the live upstream/EasyR1 on disk.

**Root cause**: `Dockerfile.npu` does `COPY . /opt/easyr1 && pip install -e .`. The editable install writes a `.pth` file pointing at `/opt/easyr1`, which is the image's COPY layer, not anything bind-mountable. So `git pull` on the host is invisible to the container unless we `docker build` again.

**Fix**: `run-npu-container.sh` bind-mounts the live `upstream/EasyR1/` over `/opt/easyr1`, so the editable install's target is now the bind-mount (live). Source changes take effect on next container spawn; only Dockerfile changes need a rebuild.

**Where applied**: commit `1595af4`.

**Rule**: whenever a Dockerfile does `COPY . <path> && pip install -e <path>`, add a bind-mount over `<path>` in the runner script so development iterates fast. Encode in `upstream-branch-hygiene` SKILL.md.

---

### NPU-OPS-002 — stale `__pycache__` after source swap

**Symptom**: source visible inside container is updated, but Python picks up the old `.pyc`.

**Root cause**: `__pycache__/*.pyc` files persist in the bind-mounted source dir. If the source file's mtime doesn't advance (or the filesystem's resolution is coarser than the .pyc compile time), Python's import cache validation fails open and keeps the old byte-code.

**Fix**: clear `__pycache__` dirs after a `git pull` that touches Python source; or set `PYTHONDONTWRITEBYTECODE=1` in the container runner to skip writing `.pyc` in the first place.

**Rule**: pair every live-bind-mounted development setup with either `PYTHONDONTWRITEBYTECODE=1` or a pre-run pycache clear. To add to `run-npu-container.sh` in a follow-up.

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
