# EasyR1-port Error Corrections Reference

> Structured error→repair mappings for compile / import / smoke failures.
> Load when: Phase C build fails, Phase D smoke fails, static_check flags an issue.
> Format: Error pattern → Root cause → Fix → Verify → Related pattern ID

All entries derived from actual incidents on this project
(porting-journal entries 2026-04-17..22). Each has a concrete commit or dated
observation as proof the fix works.

---

## EC-01: SyntaxError — import inserted inside multi-line import block

**Error pattern**:
```
SyntaxError: invalid syntax
  File "verl/workers/fsdp_workers.py", line 62
    from ..utils.device import is_npu_available, get_device_module, ...
    ^^^^
```

Happens when agent / auto-editor inserts a new `from X import Y` statement inside an existing unclosed `from Z import (` multi-line block:

```python
from ..utils.torch_functional import (
from ..utils.device import is_npu_available, ...  # ← inserted here, breaks parent import
    AnyPrecisionAdamW,
    ...
)
```

**Root cause**: code-path-sweep / any auto-edit tool suggests "add this import near other imports" without checking if the nearest imports are inside a `( ... )` block. Insertion succeeds (text) but breaks parent import's syntax.

**Fix**:
1. Close the parent multi-line import first
2. Add the new import as its own statement AFTER the closing `)`

```python
from ..utils.torch_functional import (
    AnyPrecisionAdamW,
    get_constant_schedule_with_warmup,
    ...
)
from ..utils.device import is_npu_available, get_device_module  # ← now a top-level statement
```

**Verify**: `python -m py_compile verl/workers/fsdp_workers.py` exits 0.

**Related**: OL-01 (static_check must run before claiming Phase C done).

**First observed**: 2026-04-22 round 2 cold-drive agent; see porting-journal 2026-04-22 entry.

---

## EC-02: ImportError no_init_weights from transformers.modeling_utils

**Error pattern**:
```
ImportError: cannot import name 'no_init_weights' from 'transformers.modeling_utils'
```

Surfaces when running on transformers >=5.0 (drill image 8.5.2 has transformers 5.3.0.dev0).

**Root cause**: transformers 5.0 moved `no_init_weights` from `transformers.modeling_utils` to `transformers.initialization`. Our code that was written against transformers 4.x fails.

**Fix** (backward-compatible with 4.x AND 5.x):

```python
# verl/workers/fsdp_workers.py
try:
    from transformers.modeling_utils import no_init_weights  # transformers < 5
except ImportError:
    from transformers.initialization import no_init_weights  # transformers >= 5
```

**Verify**: `python -c "from verl.workers.fsdp_workers import *; print('OK')"` on both transformers 4.x and 5.x.

**Related**: patterns/domains/transformers_compat.md; first backward-compat fix (commit `1f716ea` on ascend-port).

---

## EC-03: AttributeError read-only property SamplingParams.eos_token_id

**Error pattern**:
```
AttributeError: property 'eos_token_id' of 'SamplingParams' object has no setter
```

Surfaces on vllm 0.18+. Our `update_sampling_params` contextmanager tries to setattr it.

**Root cause**: vllm 0.18 turned `SamplingParams.eos_token_id` into a read-only `@property` (backed by private `_eos_token_id`). Our contextmanager generically setattr'd every kwarg matching a class attribute.

**Fix**: filter out read-only descriptors before setattr:

```python
# verl/workers/rollout/vllm_rollout_spmd.py
sp_cls = type(self.sampling_params)
for key, value in kwargs.items():
    if not hasattr(self.sampling_params, key):
        continue
    # Skip read-only @property (vllm 0.18+ eos_token_id etc.)
    cls_attr = getattr(sp_cls, key, None)
    if isinstance(cls_attr, property) and cls_attr.fset is None:
        continue
    # ... old setattr logic
    old_sampling_params_args[key] = getattr(self.sampling_params, key)
    setattr(self.sampling_params, key, value)
```

**Verify**: on vllm 0.13 (contextmanager still applies non-property kwargs normally) and vllm 0.18+ (eos_token_id skipped, no crash).

**Related**: patterns/domains/vllm_compat.md; backward-compat fix commit `ecce71d` on ascend-port.

---

## EC-04: ImportError Language from triton.backends.compiler

**Error pattern**:
```
ImportError: cannot import name 'Language' from 'triton.backends.compiler'
  File "triton/backends/amd/compiler.py", line 1, in <module>
    from triton.backends.compiler import BaseBackend, GPUTarget, Language
```

Surfaces in drill image (8.5.2 base) or any image with both upstream triton and triton-ascend installed.

**Root cause**: NPU-BUG-004. Base image ships `triton==3.6.0` (upstream CUDA/AMD) and `triton-ascend==3.2.0` into the SAME `site-packages/triton/` tree. Their `backends/` subdirs merge: `backends/amd/` + `backends/nvidia/` from upstream 3.6, `backends/ascend/` + top-level `backends/compiler.py` from triton-ascend 3.2. When any torch.compile path triggers `triton.backends._discover_backends()`, it walks every `backends/*/compiler.py` including `amd` — which imports `Language` from a `compiler.py` that doesn't export it.

**Fix**: remove `amd/` and `nvidia/` backend dirs in Dockerfile (we only need `ascend/` on NPU):

```dockerfile
RUN python3 -c "import triton.backends, os, shutil; \
  root=os.path.dirname(triton.backends.__file__); \
  [shutil.rmtree(os.path.join(root,d), ignore_errors=True) for d in ('amd','nvidia')]"
```

**Verify**: `python3 -c "import torch_npu"` no longer fails with the Language import error.

**Related**: PLATFORM_BUGS.md NPU-BUG-004; drill commits 15f9450 + a18d1f8.

---

## EC-05: ValueError Total available GPUs 0 is less than total desired N

**Error pattern**:
```
ValueError: Total available GPUs 0 is less than total desired GPUs 2
  File "verl/trainer/ray_trainer.py", line 117, in _check_resource_available
```

Surfaces when Ray launches but can't find NPU as a resource.

**Root cause 1**: Ray doesn't auto-detect NPU as a resource. It needs explicit `resources={"NPU": n}` registration.

**Root cause 2**: Even when `ASCEND_RT_VISIBLE_DEVICES` is set, Ray 2.55+ wipes it when `num_gpus=0` is passed (default). Need `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`.

**Root cause 3 (IF container NPU enum also returns 0)**: container missing bind mounts `/usr/local/dcmi`, `/usr/local/Ascend/driver/lib64` (subdir, not whole tree), `/etc/ascend_install.info`. Then dcmi_init fails inside container → torch_npu.npu.device_count() returns 0 → Ray sees 0 NPU → this error. See NPU-OPS-009.

**Fix**: apply `patterns/domains/ray_integration.md` Ray NPU shim, AND confirm container bind mounts:

```python
# verl/trainer/ray_trainer.py (or ray init site)
runtime_env = {
    "env_vars": {
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        "VLLM_ASCEND_ENABLE_NZ": "0",
    }
}
ray.init(runtime_env=runtime_env, resources={"NPU": num_chips})
# Actor options:
options = {"resources": {"NPU": per_actor_chips}}  # NOT {"num_gpus": per_actor_chips}
# Placement bundles:
bundles = [{"CPU": c, "NPU": 1} for _ in range(world_size)]
```

**Verify**: `ray.available_resources()["NPU"] == num_chips`; smoke V1.4 advances past Ray resource check.

**Related**: patterns/domains/ray_integration.md; commits fb1a223, 59641d4, cc8e794 on ascend-port. Container-level: see EC-07.

---

## EC-06: HFValidationError Repo id must be in the form 'repo_name'

**Error pattern**:
```
huggingface_hub.errors.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/data/z00637938/models/Qwen2-0.5B-Instruct'
```

**Root cause**: Local filesystem path passed where HuggingFace expects a repo id. **But the underlying error** is usually: model directory doesn't exist inside the container → HF falls back to treating the path as a repo id → rejects because it has `/`.

In practice caused by `run-npu-container.sh` not bind-mounting `/data/$USER` because `NPU_USER=$USER` defaults to `root` when ssh as root, and `/data/root` doesn't exist.

**Fix**:

1. Immediate: pass `--user <actual-user>` to container runner OR set `NPU_USER` env explicitly:
   ```bash
   bash scripts/deploy_to_a3.sh --user z00637938 ...
   ```
2. Better: runner script should detect "ssh-as-root-on-behalf-of-$owner" and pick the right owner from the bind source path.

**Verify**: inside container, `ls /data/$USER/models/Qwen2-0.5B-Instruct` shows config.json + model.safetensors.

**Related**: Stage 0 T1-005 (follow-up UX fix); first observed 2026-04-22 round 2.

---

## EC-07: dcmi model initialized failed, because the device is used (ret -8020)

**Error pattern** (inside container):
```
DrvMngGetConsoleLogLevel failed. (ret=4)
dcmi model initialized failed, because the device is used. ret is -8020
```

Also `torch_npu.npu.device_count()` returns 0.

**Root cause**: NOT "another process is using the NPU". Misleading message. Actually: container is missing critical bind mounts for DCMI userspace. Specifically:

- Missing `/usr/local/dcmi` mount
- Binding ENTIRE `/usr/local/Ascend/driver` tree instead of just `lib64` subdir
- Missing `/etc/ascend_install.info`

Without these, DCMI init fails → device enumeration reports 0 → downstream errors.

**Fix**: use this bind set in `docker run` (see `patterns/domains/dockerfile.md` §container-runner):

```bash
docker run \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --device /dev/davinci0 ...  # per chip
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \      # subdir only
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /usr/local/dcmi:/usr/local/dcmi \                                     # critical
  -v /etc/ascend_install.info:/etc/ascend_install.info \                   # critical
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  ...
```

Do **NOT** `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver` (whole tree) — that triggers this error.

**Verify**:
```bash
docker run <image> python3 -c "import torch_npu; print(torch_npu.npu.device_count())"
# should print number of chips passed via --device
```

**Related**: PLATFORM_BUGS.md NPU-OPS-009 (originally misdiagnosed as UDA ns leak); fix commit b3f7a0f on easyr1-npu repo; first root-caused 2026-04-21 by diffing against `roll_npu_new` working sibling container.

---

## EC-08: npu get board type failed. ret is -9005

**Error pattern**:
```
npu get board type failed. ret is -9005
```

Usually appears alongside or instead of EC-07 depending on exactly what's missing.

**Root cause**: Same class as EC-07 — incomplete container bind set for DCMI userspace. Often seen when `/usr/local/dcmi` is bound but other pieces of the Ascend install metadata are missing.

**Fix**: apply EC-07 full bind set. If EC-07 fix doesn't clear EC-08, check:
- `/etc/ascend_install.info` bound and readable inside container
- `/usr/local/Ascend/driver/version.info` bound

**Verify**: same as EC-07.

**Related**: EC-07.

---

## EC-09: dmesg uda_occupy_dev_by_ns Conflict open udevid

**Error pattern** (in host `dmesg`, not container log):
```
[ascend] [uda] [ERROR] [uda_occupy_dev_by_ns 932] <npu-smi:PID:PID:N> Conflict open udevid.
(udevid=0; access_ns=00000000XXXXXXXX; ns=00000000YYYYYYYY)
```

**Important**: This dmesg line **does NOT** automatically mean "another namespace has NPU locked out". On our project it misled us 2 full days (2026-04-20..21).

**Root cause**: Usually a **downstream side effect** of EC-07 (dcmi init failure mid-way). When dcmi partially initializes and fails, it can leave a ns refcount in an inconsistent state. The dmesg line is emitted during the failed init, not because a different ns is holding the device.

**Diagnostic first**: before assuming ns conflict, test a **working sibling container** on the same host:

```bash
docker run --rm --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  <some-known-good-image> \
  python3 -c "import torch_npu; print(torch_npu.npu.device_count())"
```

If this works → your issue is YOUR container config (EC-07).
If this also fails → then yes, it might be a real ns conflict; check for long-running processes (`ps -ef | grep raylet`).

**Fix**: apply EC-07 first. Only escalate to process-kill if sibling container also fails.

**Verify**: see EC-07.

**Related**: EC-07; PLATFORM_BUGS.md NPU-OPS-009 (retraction note — historical misdiagnosis recorded as warning).

**Never run `device_hot_reset.sh` on a suspected ns conflict**. Observed 2026-04-21: it killed PCI enumeration and required a full host reboot to recover.

---

## EC-10: pip install triton-ascend hangs for minutes

**Error pattern**: `docker build` pipeline sits on:
```
RUN pip install --no-cache-dir --force-reinstall --no-deps triton-ascend==3.2.0
```

for 50+ minutes, no progress.

**Root cause**: NPU-OPS-008. The huaweicloud `mirrors.huaweicloud.com/ascend/repos/pypi/triton-ascend/` mirror is intermittently empty / slow. Without a timeout, pip hangs on it forever.

**Fix**: Dockerfile should use aliyun mirror first (reliable) OR add a timeout:

```dockerfile
RUN pip install --no-cache-dir --default-timeout=60 --force-reinstall --no-deps \
    --index-url https://mirrors.aliyun.com/pypi/simple/ triton-ascend==3.2.0 || \
    pip install --no-cache-dir --default-timeout=60 --force-reinstall --no-deps \
    --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi triton-ascend==3.2.0
```

The key is `--default-timeout=60` so pip will fail fast instead of hanging.

**Verify**: `docker build` either completes or fails within ~2 min on the triton-ascend step, not 50 min.

**Related**: OL-07; PLATFORM_BUGS.md NPU-OPS-008; first observed 2026-04-22 round 2 (build killed manually after 50 min).

---

## EC-11: V1.4 smoke_validate reports "no entropy_loss marker" even though training completed

**Error pattern**:
```
❌ V1.4 FAIL: no 'entropy_loss:' marker in log — smoke probably errored before step 1
```
But the log shows step 1 + step 2 completed and a checkpoint was saved.

**Root cause**: EasyR1 master writes per-step metrics ONLY to
`<save_checkpoint_path>/experiment_log.jsonl` via the `FileLogger` in
`verl/utils/logger/logger.py`. It does NOT print `entropy_loss:` to stdout.
The historical v1 baseline of 0.991 was established against the `ascend-port`
branch which had stdout printing patched in. Master `dd71bbd` does not.

**Fix**:
1. Smoke script's `trainer.logger` MUST include `'file'` (and `'console'` for
   human-readability). Minimum: `trainer.logger=['console','file']`.
2. `smoke_validate.sh` (post-2026-04-22 round 3 patch) reads jsonl by
   default. On an older copy, pass `--metrics-jsonl <A3-side-path>`.

**Verify**: rerun smoke. The assertion output should say
`source: jsonl (...)` instead of `source: stdout grep (...)`.

**Related**: Round 3 finding 2026-04-22; SMOKE_BASELINE.md §canonical config.

---

## EC-12: V1.4 entropy_loss ~1.27 (out of v1 band 0.94-1.04) even though port is correct

**Error pattern**: Step-1 entropy_loss around 1.27 (close to v2-image 1.275
baseline). All device / Ray / attention / vllm fixes applied. No Python
errors. Training completes 2 steps cleanly.

**Root cause**: Smoke config deviates from canonical
`ascend-port/examples/qwen2_0_5b_math_grpo_npu_smoke.sh`. **Most likely**:
KL term left on in master's defaults. Canonical smoke sets
`algorithm.disable_kl=true` and `algorithm.use_kl_loss=false`; without
those, the KL contribution raises entropy_loss substantially. Other
sensitivity points (max_prompt_length, gradient_checkpointing, ref offload)
also shift numerics.

**Fix**: Copy the canonical smoke config verbatim. Do not omit:
- `algorithm.disable_kl=true`
- `algorithm.use_kl_loss=false`
- `worker.actor.model.enable_gradient_checkpointing=true`
- `worker.ref.fsdp.enable_cpu_offload=true`
- `worker.ref.offload.offload_params=true`

**Verify**: step-1 entropy_loss falls back to [0.94, 1.04] band.

**Related**: Round 3 finding 2026-04-22; SMOKE_BASELINE.md §critical parameters.

---

## EC-13: V1.4 "no entropy_loss" on a rerun after a prior session left global_step_N

**Error pattern**:
```
❌ V1.4 FAIL: no 'entropy_loss:' marker in log ...
```
But the smoke's own stdout/jsonl shows it ran only validation, no training
step. Looking at the checkpoints dir you notice `global_step_2/` from an
earlier session. Training was silently skipped.

**Root cause**: EasyR1 master's trainer defaults `find_last_checkpoint=true`.
When a prior session left `<save_checkpoint_path>/global_step_N/` and
`checkpoint_tracker.json`, the trainer auto-resumes at `global_step=N`. If
`max_steps` is also set to `N` (typical smoke), the training loop
short-circuits — no step 1, no step 2, only the final validation is written
to `experiment_log.jsonl`. Smoke assertion then sees "no entropy_loss" (EC-11
symptom), but the underlying cause is NOT the stdout/jsonl contract — it's
that training actually didn't run.

**Fix**: two lines, both in the smoke script:

```bash
# 1. Clean the checkpoint dir at the start of every smoke run (belt).
rm -rf /opt/easyr1/checkpoints/easy_r1/v14_smoke

# 2. And disable auto-resume in case another dir is lingering (suspenders).
python3 -m verl.trainer.main \
    ... \
    trainer.find_last_checkpoint=false
```

**Verify**: jsonl now has step 1 + step 2 rows with `actor.entropy_loss`.
If training really ran, log shows `Running step: 100%|██████████| 2.00/2.00`
and per-step metrics show up.

**Why this isn't covered by EC-11**: EC-11's fix (switch assertion to jsonl
reader) is necessary but insufficient here. Even a jsonl-aware assertion
returns "no entropy_loss" because training literally didn't run — the jsonl
only has a val row. Don't confuse the two: EC-11 is contract bug, EC-13 is
trainer behavior + stale state.

**Related**: Round 4 finding 2026-04-22 iter1 (commit `aa07f21`); EC-11
(adjacent symptom, different root cause).

---

## EC-14: V1.3 rollout smoke fails with HFValidationError or file-not-found on `/data/nobody/models/...`

**Error pattern**: V1.3 smoke (e.g. `smoke_v13_rollout.py`) errors early with:
- `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'...`
- or a plain `FileNotFoundError` on `/data/nobody/models/Qwen2-0.5B-Instruct`
- or a `path is not a local folder and is not a valid model identifier` error

**Root cause**: The smoke script computes `/data/${USER}/models/...` but inside
the NPU container `$USER` resolves to `nobody` (container's default non-root
user). The correct NPU-owned path is `/data/z00637938/models/...` (owner is
the host user `$NPU_USER`, not the container's in-process `$USER`).

**Fix**: smoke scripts must auto-discover `NPU_USER` from the host, not rely
on container `$USER`. Either:
1. Take it from an env var (orchestrator / `deploy_to_a3.sh` / `run-npu-container.sh`
   already set `NPU_USER=z00637938`). The smoke script should use `${NPU_USER}`
   rather than `${USER}`.
2. Or walk `/data/` and pick the non-root owned dir: `NPU_USER=$(ls /data 2>/dev/null | head -1)`.

Canonical smoke scripts (round 4, wet-run) use:
```bash
NPU_USER="${NPU_USER:-$(ls /data 2>/dev/null | grep -v '^$' | head -1)}"
MODEL_PATH="${MODEL_PATH:-/data/${NPU_USER}/models/Qwen2-0.5B-Instruct}"
```

**Verify**: V1.3 log shows model loaded from `/data/z00637938/...` not
`/data/nobody/...`.

**Observed**: round 3 iter1, round 4 iter (commit `8c8dc40`), wet-run iter1
(commit `d6acbc8`). Chronic — every cold-drive agent has to rediscover this
unless the smoke template is pre-written with the fallback.

**Related**: OL-06 (A3/GFW unrelated); NPU-CP-003 (Ray env; unrelated).
This is purely a smoke-script input-plumbing issue.

---

## Unclassified failure protocol

If a Phase D failure doesn't match any EC:

1. Document the full traceback in `workspace/easyr1-port-{ts}/PROGRESS.md` §"unclassified failures"
2. Check PLATFORM_BUGS.md for a matching NPU-BUG-NNN
3. Check patterns/domains/<area>.md for a known pitfall in that area
4. If still stuck after ≥3 iterations on same signature → exit @smoke-probe (Stage 1+) or @review-fail (Stage 0)
5. New reproducible failures → add new EC entry here (for future agents)
