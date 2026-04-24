# Platform bugs that recur on NPU base images (shared with easyr1-expert)

> Load at Phase C build / Phase D smoke failure. These bugs are in NPU platform
> packages (triton-ascend, upstream triton, torch_npu), not in EasyR1 or in
> anything this expert touches.

---

## NPU-BUG-001: triton-ascend broken install in base images

**Symptom**:
```
ImportError: cannot import name 'Config' from 'triton'
```
or a torch._inductor import chain failure on `import torch_npu`.

**Root cause**: the `verl-*-a3` base images ship `triton-ascend 3.2.0` but
with an incomplete `/usr/local/python3.11.14/.../site-packages/triton/` tree
— `__init__.py` and several top-level modules are missing. Wheel's own
RECORD lists them; they just didn't get placed.

**Workaround**: force-reinstall the exact version with `--no-deps` in the
Dockerfile to rewrite the file tree from the wheel. See
`patterns/domains/dockerfile-target.md` Stage 0 template.

**Recurs on**: both v1 (verl-8.5.0-a3) and v2 (verl-8.5.2-a3) drill images.
Almost certainly on any future ascend/verl image until packaging upstream
is fixed.

**Related**: OL-07 (pip mirror); EC-10 (pip hang on huaweicloud); drill
commits `cbfe645` / `cd16649` for v1; `318925f` / `16051e2` for v2.

---

## NPU-BUG-004: upstream triton backends clash with triton-ascend

**Symptom**:
```
ImportError: cannot import name 'Language' from 'triton.backends.compiler'
```

**Root cause**: the v2 base image's installed `triton` wheel ships
`triton/backends/{amd,nvidia}/` subdirs that reference CUDA / ROCm headers
not present on A3. When triton-ascend walks the `backends/` dir to
register `npu`, it blows up on the first broken backend's
`import_backends()`.

**Workaround**: prune the amd/nvidia backend subdirs in Dockerfile
(see `patterns/domains/dockerfile-target.md` §NPU-BUG-004 prune block).
Do NOT `pip uninstall triton` — triton-ascend shares the same `triton/`
dir and relies on it being present.

**Recurs on**: v2. v1 did NOT ship amd/nvidia subdirs (different upstream
triton wheel). Check every new base image:
```bash
docker run --rm $TARGET_IMAGE python3 -c "
import importlib.util, os
spec = importlib.util.find_spec('triton')
print('amd:',
      os.path.isdir(os.path.join(os.path.dirname(spec.origin),'backends/amd')))
print('nvidia:',
      os.path.isdir(os.path.join(os.path.dirname(spec.origin),'backends/nvidia')))
"
```

**Related**: EC-04; drill commits `15f9450` / `a18d1f8`.

---

## Other NPU-BUGs that affect easyr1-expert but not this expert

NPU-BUG-002 (Ray `ASCEND_RT_VISIBLE_DEVICES` clearing) and NPU-BUG-003
(inductor crash on `log_probs` with torch.compile) live in
`easyr1/port-expert/references/PLATFORM_BUGS.md`. They're consumer-side port
concerns, not base-image concerns. This expert doesn't need to address
them directly — if a target image somehow "fixes" either by accident, the
consumer's existing workaround keeps working.

---

## Unclassified platform failures

If a target-image platform failure doesn't match NPU-BUG-001/004 here:

1. Check easyr1-expert's PLATFORM_BUGS.md (it has NPU-BUG-002/003 and may
   have newer entries)
2. Check ERROR_CORRECTIONS.md for EC match
3. If still novel: record as new finding in PROGRESS.md, exit stuck,
   DO NOT guess a workaround
