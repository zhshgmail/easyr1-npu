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

### NPU-CP-007 — swap flash-attn imports for transformers NPU integrations

**Pattern**: framework code does `from flash_attn import flash_attn_func, flash_attn_varlen_func` (or similar) to drive FA2-style variable-length attention.

**Symptom**: on NPU the flash-attn package isn't installed; `ImportError` at module load. Worse: if the framework guards this import behind `try/except ImportError`, the exported functions are undefined but the module imports cleanly → `NameError` is deferred until runtime.

**Root cause**: flash-attn's CUDA kernels are absent on NPU. However, `transformers >= 4.45` ships its own NPU adapters at `transformers.integrations.npu_flash_attention` — `npu_flash_attn_func` and `npu_flash_attn_varlen_func` — that wrap `torch_npu.npu_fusion_attention` behind flash-attn-style signatures. The adapters are available on transformers 4.57.6 (our v1 target) and on main.

**Fix**: conditional-import swap before the flash-attn import:
```python
from verl.utils.device import is_npu_available
# ... or your own is_npu_available helper ...

if is_npu_available():
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
    _flash_use_top_left_mask = False  # NPU FA uses bottom-right causal
elif is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    # ... existing CUDA introspection ...
```

After the swap, code downstream (`_flash_attention_forward` from `transformers.modeling_flash_attention_utils`, `fa_peft_integration_check`, and custom varlen paths) stays backend-agnostic — transformers' `_flash_attention_forward` itself does the same dispatch (see `src/transformers/modeling_flash_attention_utils.py` lines 161-163 in master).

**Commit ref**: `fbaa983` in EasyR1's `verl/models/transformers/flash_attention_utils.py`.

**Generalizable rule**: **transformers is the NPU adapter layer for flash-attention**, not torch_npu directly. Any framework with a conditional `if is_npu_available(): from torch_npu ...` hand-written adapter should be refactored to use the transformers integration. This is also the pattern veRL uses in `verl/models/transformers/qwen2_vl.py:52-55` and `glm4v.py:52-55`. See also NPU-OPS-005 — reading the reference port saved ~2 days vs writing a custom adapter.

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

### NPU-BUG-003 — triton-ascend inductor kernel crash on shape-sensitive log-probs op

**Symptom**: `[Error]: The vector core execution is abnormal. Kernel task happen error, retCode=0x31, [vector core exception].` raised during training, async traceback points at `aclnnNonzero` or another op downstream of the failing one. With `ASCEND_LAUNCH_BLOCKING=1` the real op is pinned to a triton-ascend inductor-compiled kernel called from `verl/utils/torch_functional.py::log_probs_from_logits` (wrapped by `torch.compile`).

Full blocking-mode stack (EasyR1 repro 2026-04-19):
```
torch_functional.log_probs_from_logits  (wrapped by torch.compile)
  → torch._inductor → torch_npu._inductor.npu_triton_heuristics.run
  → triton.backends.ascend.driver.__call__
  → [vector core exception]
```

**Root cause**: `torch.compile(..., dynamic=True)` on NPU compiles through `torch._inductor` which generates a triton-ascend kernel. On certain input shapes (observed: flat `(1, total_nnz)` packing under `padding_free=True`, vs padded `(batch, seqlen)` under `padding_free=False`), the generated kernel hits a vector-core error. V1.4 happened to avoid it because `padding_free=false` paths give a different shape that compiles and runs.

**Fix (workaround)**: set `worker.actor.use_torch_compile=false` in any EasyR1 config running on NPU. Falls back to eager-mode `log_probs_from_logits` (slight perf cost, no correctness impact).

**Commit ref**: `75bad74` in EasyR1 (V2.1 smoke script sets `use_torch_compile=false`).

**Generalizable rule**: **on NPU, default-disable `torch.compile` for RL / varlen workloads** until triton-ascend is proven stable for the shapes in use. Before enabling `torch.compile`, run the target workload with `ASCEND_LAUNCH_BLOCKING=1` and watch for `vector core exception` traces — if any, keep compile disabled. Candidate future `NPU-ENV-NNN`: bake `TORCHINDUCTOR_DISABLE=1` into the container runner env defaults.

**Status on 8.5.2 image (probed 2026-04-19, `bug003_probe` script on drill branch)**: could not confirm or refute. The probe hit NPU-BUG-004 (triton 3.6 + triton-ascend 3.2 coexistence crash) at `torch.compile` registration, **before** the inductor could execute a kernel. Until NPU-BUG-004's fix lands (remove upstream triton's amd/nvidia backends in the drill Dockerfile), BUG-003's state on CANN 8.5.1 remains unknown.

---

### NPU-BUG-004 — upstream `triton==3.6.0` + `triton-ascend==3.2.0` in same site-packages

**Symptom**: any `torch.compile` code path on the 8.5.2-based image fails at inductor registration with `ImportError: cannot import name 'Language' from 'triton.backends.compiler'` during `torch_npu._inductor` import, raised from `triton/backends/amd/compiler.py:1`. Followed by `torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised: ImportError ...` and a secondary `TypeError: cannot pickle 'frame' object` as Ray tries to serialize the error frames back to the driver.

**Root cause**: the `verl-8.5.2-a3-...-qwen3-5` base image installs **both** `triton==3.6.0` (upstream CUDA/AMD triton) and `triton-ascend==3.2.0` into the single `site-packages/triton/` tree. Their backend trees merge: `backends/amd/` + `backends/nvidia/` come from upstream 3.6, while `backends/ascend/` + the top-level `backends/compiler.py` come from triton-ascend 3.2. Upstream 3.6's `backends/amd/compiler.py` starts with `from triton.backends.compiler import BaseBackend, GPUTarget, Language` — but the `compiler.py` currently on disk (from triton-ascend 3.2) doesn't export `Language`. Any `torch.compile` path eventually imports `torch_npu._inductor`, which imports `triton.runtime`, which triggers `triton.backends._discover_backends()`, which walks **every** `backends/*/compiler.py` including `amd` — boom.

**Fix**: delete the `amd/` and `nvidia/` backend dirs after installing triton-ascend. We only need `ascend/` on an NPU host, and keeping the other two was already broken. One-line fix in the Dockerfile:

```
RUN python3 -c "import triton.backends, os, shutil; \
  root=os.path.dirname(triton.backends.__file__); \
  [shutil.rmtree(os.path.join(root,d), ignore_errors=True) for d in ('amd','nvidia')]"
```

Alternative (heavier) fixes: (a) downgrade upstream triton to 3.2 to match triton-ascend; (b) rebuild triton-ascend against upstream 3.6 tip; (c) pass `TRITON_DISABLE_BACKENDS=amd,nvidia` env if upstream adds such a knob (none exists today in 3.6).

**Commit ref**: `Dockerfile.npu-852` on the `ascend-port-transformers-upgrade` drill branch (2026-04-19). Not needed on the 8.5.0-a3 image (ships only triton-ascend 3.2, no upstream triton).

**Generalizable rule**: **whenever two triton distributions share a site-packages**, their `backends/` subdirs will auto-discover together, and any ABI drift between the version of `backends/compiler.py` on disk and the version imported by a sibling backend's `compiler.py` crashes all compile paths. On a new base image, inspect `pip show triton triton-ascend` and `ls site-packages/triton/backends/` early; if both are present and the backend dirs don't match the on-disk `compiler.py`, prune the non-ascend backends as part of the Dockerfile.

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

### NPU-OPS-005 — always inspect the reference implementation before designing

**Symptom**: time estimate for a piece of work is off by an order of magnitude. Typically: "this is going to take days because we need to write an adapter from scratch" — then it turns out the upstream library already ships that adapter, the reference port uses it in 4 lines, and the real work is 1 hour of import swap.

**Root cause**: skipping the "what does the reference port do for this?" check before designing. The combination that makes this trap fire:
1. There is an existing ported system solving the same problem (in our case, **veRL is the ported version of what EasyR1 is a slim fork of**).
2. There is an upstream library already supporting the target platform (in our case, **`transformers.integrations.npu_flash_attention`** ships `npu_flash_attn_varlen_func` baked into the library, not in the framework layer).
3. The agent reasons from first principles about what would be needed to solve the problem, without first checking whether it's already solved in the adjacent project.

Concrete 2026-04-18 incident: I estimated v2 (NPU varlen attention for `padding_free=True`) at 2 days of work writing a `torch_npu.npu_fusion_attention` adapter. After the user asked "how does veRL do it?" I checked `upstream/verl/verl/models/transformers/qwen2_vl.py:52-55` and found:
```python
if is_npu_available:
    from transformers.integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
    from transformers.integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
```
4 lines. Total work dropped from ~2 days to ~1-2 hours. I had violated my own version-aware-reviews rule — had skipped reading the reference before proposing an implementation.

**Fix**:
- Before proposing any non-trivial work on a problem that has a known adjacent port (veRL ↔ EasyR1 relation; OpenRLHF ↔ TRL; any fork ↔ upstream; any frontend ↔ backend), **spend 5-10 minutes grep-reading the adjacent system** for the same sub-problem. Look specifically in `models/transformers/`, `utils/`, integration modules — anywhere `if is_npu_available()` / `if is_cuda_available()` branching happens.
- Separately, check whether the **upstream library** (in our case transformers, in other cases pytorch, vllm, ray) has a ready-made integration. The integration usually lives under `*integrations*`, `*backends*`, or `*_utils.py`.
- Only after both of those come up dry, design from scratch.

**Where applied**: —

**Generalizable rule**: **any time you're about to write an adapter, shim, or "from scratch" implementation of a cross-platform concern (device routing, kernel replacement, distributed backend, IO)**, ask first: has this already been solved one layer up? The pattern holds across ports — the adjacent ported project AND the upstream library are both cheaper to read than to rewrite. Log the answer (even if it's "nobody did this") in the relevant design doc so the next session doesn't re-explore.

---

### NPU-OPS-004 — disk pressure on a shared host

**Symptom**: `docker pull` fails with `no space left on device`, or a training job dies during checkpoint save.

**Root cause**: A3 root fs was 93% used at onboarding, with ~258 GB free. Each docker image (14-18 GB), each weight download (0.5-14 GB for Qwen models), and each checkpoint (model size × FSDP shard count) competes for the same `/` partition. `/var/lib/docker` (images) and `/data` (weights/checkpoints per user convention) both live on `/`.

**Fix**: prune docker images aggressively (`docker image prune -a`), keep only 1-2 weight sets at a time, move long-running checkpoint directories off-host when possible. Use `/data/<user>/` for anything large (convention, not enforcement).

**Commit ref**: —

**Generalizable rule**: on shared hosts, monitor `df -h /` before any large artifact operation. Don't delete other users' files to make room — ask instead.

---

### NPU-OPS-007 — base image missing `/etc/pip.conf` causes build-time pypi.org hang

**Symptom**: `docker build` hangs indefinitely inside a `RUN pip install ...` step. `ps` shows pip in `S` (sleeping) state on TLS sockets. No retry messages, no error — just silence. Kills only via signal.

**Root cause**: the base image shipped no baked-in pip configuration (`/etc/pip.conf`, `/root/.pip/pip.conf`, or `PIP_INDEX_URL` env), so pip defaults to `https://pypi.org/simple/`. On a firewalled host, pypi.org is either unreachable (GFW) or only reachable via the docker daemon's HTTP proxy — which the build container does *not* inherit. Unlike `docker run` (where `-e http_proxy=...` can be plumbed), `docker build` does not automatically pass the daemon's proxy into the build context.

**Fix**: set pip's index URL in the Dockerfile itself so it survives image rebuilds and never touches pypi.org:

```
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/
ENV PIP_TRUSTED_HOST="mirrors.aliyun.com mirrors.huaweicloud.com"
ENV PIP_DEFAULT_TIMEOUT=120
```

This is per-image, not per-invocation — covers the whole build context plus any later `pip install` inside running containers.

**Commit ref**: `Dockerfile.npu-852` — transformers upgrade drill (2026-04-19).

**Generalizable rule**: whenever you change the base image, re-inspect it for baked-in pip config: `docker run --rm --entrypoint cat <image> /etc/pip.conf` and check `PIP_INDEX_URL` in `docker inspect <image>`. If absent, patch the layer **before** adding `RUN pip install ...` steps. Don't rely on the daemon's `HTTP_PROXY` — it doesn't plumb through to build.

---

### NPU-OPS-008 — huaweicloud's ascend pypi mirror is not a reliable source

**Symptom**: `pip install triton-ascend==3.2.0 --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi` returns `ERROR: Could not find a version that satisfies the requirement triton-ascend==3.2.0 (from versions: none)` even though that version is a GA release on aliyun. Browsing the huaweicloud index URL returns an empty listing for the package.

**Root cause**: the huaweicloud ascend pypi mirror's per-package HTML pages are empty/incomplete for several Ascend wheels (observed for triton-ascend on 2026-04-19; other packages may be affected intermittently). It is maintained out-of-band by Huawei and stale or blank listings happen.

**Fix**: use aliyun's general-purpose pypi simple index as the primary source for triton-ascend and other Ascend wheels:

```
pip install --no-cache-dir --force-reinstall --no-deps \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com \
    triton-ascend==3.2.0
```

Keep huaweicloud as a documented fallback in a bash `||` chain, but put aliyun first. Aliyun carries the aarch64+x86_64 cp39/cp310/cp311 wheels we need.

**Commit ref**: `Dockerfile.npu-852` (drill branch, 2026-04-19).

**Generalizable rule**: don't treat huaweicloud's ascend pypi index as authoritative. Probe with `curl -sL '<index-url>/<package>/' | grep -oE '<package>[^"<>]*\.whl' | sort -u` before wiring a new package into a Dockerfile; if the listing is empty, try aliyun, then the official vendor release page. Record which indexes worked on which date in the commit message so future triage can tell whether the mirror or the network is the problem.

---

### NPU-OPS-006 — docker daemon stuck behind dead/slow HTTP proxy on A3

**Symptom**: `docker pull <any-registry>` on A3 times out with `Get "https://<registry>/v2/": net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)`, even though `curl` from the same shell reaches the registry in <1s (HTTP 200/401). Retrying the pull just hits the same wall. Layers may get stuck in `Verifying Checksum` or `Retrying in N seconds` loops and eventually die with `unexpected EOF`.

**Root cause**: A3 is behind the GFW and the docker daemon is configured with an upstream HTTP proxy (e.g. `HTTP_PROXY=http://100.66.1.4:7897`) in `/etc/systemd/system/docker.service.d/http-proxy.conf`. The proxy is shared infra and periodically saturates/dies. Even if the registry is a CN-local mirror that the host can reach directly, the daemon still forces traffic through the proxy because the mirror hostname isn't listed in `NO_PROXY`. Note: the `registry-mirrors` field in `/etc/docker/daemon.json` only applies to Docker Hub (`docker.io`); it does **not** rewrite requests for third-party registries like `quay.io` — you must pull by the mirror hostname directly and/or retag.

**Fix**: (1) verify host-level reachability with `curl -I https://<mirror>/v2/` (expect HTTP 200 quickly); (2) add the mirror hostname to `NO_PROXY` in `/etc/systemd/system/docker.service.d/http-proxy.conf` so docker bypasses the dead proxy for that host; (3) `systemctl daemon-reload && systemctl restart docker`; (4) pull by the mirror hostname — e.g. `docker pull quay.nju.edu.cn/ascend/verl:...` instead of `quay.io/ascend/verl:...`. For CN-hosted quay mirrors that work today: `quay.nju.edu.cn` (NJU) responds HTTP 200; `quay.mirrors.ustc.edu.cn` and `hub-mirror.c.163.com` were unreachable in our tests. Docker caches partial layer blobs keyed by digest, so retagging a mirror that serves the same digests will reuse any bytes already pulled against quay.io.

**Commit ref**: —

**Generalizable rule**: on firewalled hosts behind a shared HTTP proxy, never assume `docker pull` timeouts mean the registry is down — probe with `curl` first. If the registry is reachable but docker isn't, the proxy is the problem. Keep a short list of working CN mirror hostnames in the knowledge base per registry (quay, ghcr, gcr, docker.io) and prefer mirror-hostname pulls over proxy-dependent pulls. Never change `HTTP_PROXY` itself without checking what else depends on it; extend `NO_PROXY` instead.

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

- Code patterns: `NPU-CP-001` ... `NPU-CP-007` (7 entries)
- Platform bugs: `NPU-BUG-001` ... `NPU-BUG-004` (4 entries)
- Environment/config: `NPU-ENV-001` ... `NPU-ENV-004` (4 entries)
- Operational: `NPU-OPS-001` ... `NPU-OPS-008` (8 entries)

**Total: 23 stable IDs** across 4 categories, each with uniform `Symptom / Root cause / Fix / Commit ref / Generalizable rule` schema.
