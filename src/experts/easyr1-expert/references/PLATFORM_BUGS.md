# Platform bugs — NPU-BUG-001..004

> Defer-load at Phase D on failures. Each entry: symptom → root cause →
> workaround → related EC/OL. These are bugs in NPU platform packages
> (triton-ascend, vllm-ascend, torch_npu) and the host/container stack,
> not in EasyR1 itself.

---

## NPU-BUG-001: triton-ascend 3.2.0 broken install in verl-8.5.0-a3 base image

**Symptom**:
```
ImportError: cannot import name 'Config' from 'triton' (/usr/local/python3.11.14/lib/python3.11/site-packages/triton/__init__.py)
```
or `ImportError` from `torch._inductor.runtime.triton_compat` when doing
`import torch_npu`.

**Root cause**: The `verl-8.5.0-a3` base image ships `triton-ascend 3.2.0`
but the installed `/usr/local/python3.11.14/lib/python3.11/site-packages/triton/`
tree is incomplete — `__init__.py` and a bunch of top-level modules
(`triton.compiler.CompiledKernel`, `triton.runtime.autotuner.OutOfResources`,
etc.) are missing. Likely a packaging step error upstream. The wheel's own
`RECORD` lists all the files; they just didn't get placed.

**Workaround**: In `Dockerfile.npu`, re-install the exact same version with
`--force-reinstall --no-deps` to rewrite the file tree from the wheel.
See `patterns/domains/dockerfile.md` for the exact line.

**Related**: OL-07 (pip mirror), EC-10 (pip hang on huaweicloud).

---

## NPU-BUG-002: Ray clears `ASCEND_RT_VISIBLE_DEVICES` on actor boot

**Symptom**:
```
ValueError: Total available GPUs 0 is less than total desired GPUs N
```
from EasyR1 when it computes `world_size` from NPU count inside a Ray actor.
Host-side `npu-smi info` shows chips idle; the container can `ls /dev/davinci*`;
only Ray actors see 0 NPUs.

**Root cause**: Ray 2.x treats "accelerator-like" env vars as managed and
unsets them on actor boot if Ray itself didn't set them. Ray has an
override: `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` (ask Ray to leave them
alone if accelerator count is 0, which from Ray's CUDA-centric view
NPU is).

**Workaround**: Set `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` in the actor's
`runtime_env` or at top of `verl/single_controller/base/worker.py`. See
`patterns/domains/ray_integration.md` piece 3 (NPU-BUG-002 workaround).

**Related**: NPU-CP-003, EC-05.

---

## NPU-BUG-003: torch._inductor crashes on `log_probs` path with triton-ascend

**Symptom**: Running V1.6-style smoke with `torch.compile` enabled crashes
inside `torch._inductor` at the `log_probs` tensor op. Stack trace points
at a triton-generated kernel with an "unsupported operand" or "NaN" error.

**Root cause**: `triton-ascend 3.2.0` doesn't fully support the codegen
path torch inductor takes for `log_softmax`-derived tensors at bf16/fp16.
It's an upstream gap, not an EasyR1 bug.

**Workaround**: Disable `torch.compile` in smoke scripts for NPU (V1.6/V2.1+
use `model.compile=false` or comment the `.compile()` call).

**Related**: Port-branch commit `75bad74` "V1.6 smoke: disable torch.compile
(triton-ascend inductor crashes)". EC-NN if recurs → add here.

---

## NPU-BUG-004: triton 3.6 + triton-ascend 3.2 coexistence

**Symptom**:
```
ImportError: cannot import name 'Language' from 'triton.backends.compiler'
```

**Root cause**: If the base image or a requirements.txt line pulls in
upstream `triton` (any version), it shadows `triton-ascend`'s integration
patches. `triton-ascend 3.2` patches the importable `triton` package; a
separate install of upstream `triton` overlays those patches with stock
files that don't have NPU backend hooks.

**Workaround**: NEVER let upstream `triton` install alongside `triton-ascend`.
In `Dockerfile.npu` the repair line must use `--no-deps` to avoid pulling
upstream triton. In `requirements.txt`/`requirements-npu.txt`, do NOT
pin `triton==...`.

Check after build:
```bash
pip show triton | grep -E '^Name|^Version|^Location'
# expect: triton-ascend 3.2.0 at /usr/local/python3.11.14/…/triton_ascend-3.2.0.*
# NOT: triton 3.6.0 from pypi
```

**Related**: NPU-BUG-001 (same package, different failure mode). EC-04.

---

## Unclassified — what to do when you hit a new platform bug

1. Grep this file first (`NPU-BUG-001..004`).
2. If no match, grep `ERROR_CORRECTIONS.md` (`EC-01..10`).
3. If still no match: in PROGRESS.md, add a "## Unclassified failures" section
   with the full traceback, minimal repro, and what patterns/domains/ file
   seemed closest. Report and exit `stuck` (OL-10).
4. Do NOT guess a fix and retry silently. New platform bugs deserve review
   before next round.
