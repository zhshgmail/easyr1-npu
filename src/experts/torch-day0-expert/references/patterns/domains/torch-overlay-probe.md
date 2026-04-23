# Domain — torch-overlay probe protocol

## Phase A pre-probe (static, no A3 needed)

```bash
# 1. Does torch-npu already handle this?
cd upstream/torch-npu && git fetch origin --tags
git log origin/main -S '<target-version>' -- setup.py requirements.txt
git tag --list '*<target-version>*'

# 2. Is there an rc wheel on PyPI?
pip index versions torch-npu  # or
curl -s https://pypi.org/pypi/torch-npu/json | jq '.releases | keys[]' | tail -10

# 3. What CANN does the rc target? (check README compat table)
grep -A5 'CANN' upstream/torch-npu/README.md | head -20

# 4. Does the base image's CANN match?
grep -i cann repo/knowledge/images/<target-image>.md
```

**Stop conditions**:
- torch-npu main has already adapted → switch to newer target
- rc wheel for target torch version not on PyPI → session blocked on
  Ascend/pytorch release, emit advisory to user, mark as deferred

## Phase B overlay build

Template: `overlay-image.md` (sibling file).

Key rule: **no `import torch` at build time**. PyTorch 2.11's
`_import_device_backends()` forces eager torch_npu load, which needs
CANN libs that aren't mounted during `docker build`. Use:

```dockerfile
RUN python3 -c "from importlib.metadata import version as v; \
    print('torch', v('torch'), 'torch_npu', v('torch_npu'))"
```

for version verification. For any syntactic validation of patched
files, use `python3 -m py_compile` or `ast.parse` which don't import.

## Phase C runtime smoke

Inside an NPU-enabled container (required mounts + devices):

```bash
docker run --rm --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    --device=/dev/davinci0 --device=/dev/davinci_manager \
    --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    <overlay-image> \
    bash /opt/<target>/smoke.sh
```

6 steps (every Day-0 torch smoke, don't skip):

1. pip metadata: `torch` + `torch_npu` versions exact
2. `import torch` — proves `_import_device_backends()` path works
   (torch 2.11 only)
3. `import torch_npu` — proves the .so loads
4. `torch.npu.is_available()` + `device_count()` + `get_device_name(0)`
5. Minimal NPU op: `x = torch.randn(64,64).npu(); z = x @ x.t(); z.cpu().mean()`
6. API-presence checks for any surface the target introduces or
   consumers depend on (e.g. `torch.npu.Stream.native_handle`)

All 6 must PASS for outcome A.

## Phase D API-drift classification

Diff torch N → torch N+1 for breaking changes in:

1. `native_functions.yaml` — new ops with `CompositeExplicitAutograd`
   cover NPU automatically; CUDA-only variants are safe to ignore
2. `c10/core/DispatchKey.h` — dispatcher enum changes (ABI-breaking
   for prebuilt C++ extensions)
3. `torch/csrc/distributed/` — HCCL integration surfaces
4. `torch/_ops.py` / custom op dispatch internals — any change here
   breaks pre-built `TORCH_LIBRARY_IMPL` for PrivateUse1

Gap classification table format:

| torch-N+1 item | torch_npu-rc? | python_fallback | native_required | notes |
|---|---|---|---|---|

## Phase E deploy artifacts

Per `_shared/references/patterns/domains/day0-deploy-artifacts.md`.
5 deliverables mandatory:

1. `Dockerfile.overlay-torch<M><m>`
2. `smoke_torch<M><m>.sh`
3. `deploy_torch<M><m>.sh`
4. `ONBOARDING.md`
5. `PR_MATERIAL.md` (only if C-patch)

Cold-drive validation of `deploy_*.sh` before writing ONBOARDING.

## Cross-references

- `overlay-image.md` — Dockerfile template
- `../KB_INDEX.md` — per-version symptom → outcome table
- `../../../_shared/references/patterns/domains/day0-deploy-artifacts.md`
- Concrete instance: `workspace/torch-day0-{analysis,manual,deploy}-20260423-*/`
