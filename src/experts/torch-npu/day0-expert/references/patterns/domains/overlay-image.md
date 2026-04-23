# Domain — Dockerfile.overlay-torch<M><m> template

**Load when**: Phase B building the Day-0 torch overlay image.

## Why overlay (not full rebuild)

Adding `pip install torch + torch_npu rc` on top of BASE_IMAGE is ~55
min (dominated by wheel download over A3 proxy, ~2 GB). Rebuilding the
v2 base image from scratch is several hours (includes CANN install).
Overlay wins decisively.

## Template

```dockerfile
# Overlay image: BASE_IMAGE + target community PyTorch + torch_npu rc.
# CANN (from BASE_IMAGE) and most other deps are inherited unchanged.
# Worker (or team) may apply a second layer later for downstream
# ecosystem patches (vllm-ascend Fix B+, etc.).

ARG BASE_IMAGE=<BASE_IMAGE tag>
FROM ${BASE_IMAGE}

# Do NOT rely on `import torch` at build time. PyTorch 2.11 introduced
# _import_device_backends() which eagerly imports torch_npu on `import
# torch`. torch_npu in turn dlopens libascend_hal.so from CANN at
# import-time — NOT available inside a plain `docker build` container
# (no /dev/davinci, no CANN device mount). Build-time smoke must stay
# at pip-metadata / py_compile level.

RUN pip install --no-cache-dir --no-deps \
        torch==<TARGET_TORCH_VERSION>+cpu \
        torchvision==<MATCHING_TORCHVISION> \
        torchaudio==<MATCHING_TORCHAUDIO> \
        torch_npu==<TARGET_TORCH_NPU_VERSION> \
        --extra-index-url https://download.pytorch.org/whl/cpu/

# Build-time version check — metadata-only, no imports
RUN python3 -c "from importlib.metadata import version as v; \
    print('torch', v('torch')); \
    print('torch_npu', v('torch_npu')); \
    print('torchvision', v('torchvision')); \
    print('torchaudio', v('torchaudio'))"

# Drop runtime smoke into /opt/torch<M><m>/smoke.sh
# Operator invokes it via `docker run ... bash /opt/torch<M><m>/smoke.sh`
# with NPU devices + CANN libs mounted.
COPY smoke_torch<M><m>.sh /opt/torch<M><m>/smoke.sh
RUN chmod +x /opt/torch<M><m>/smoke.sh
```

Image tag convention:
`easyr1-npu-torch<major><minor>:<SESSION_TAG>` (e.g.
`easyr1-npu-torch211:torch-day0-20260423-0537`).

## Why `--no-deps`?

- The base image has numpy / safetensors / protobuf etc. at known-good
  versions. `pip install torch` without `--no-deps` may pull
  conflicting minor bumps that aren't tested on NPU
- Phase 1 analysis §2.4 verified torch 2.11 doesn't require new transitive
  versions beyond what torch 2.9 had; same for torch_npu 2.11.0rc1.
- If the analysis finds a required transitive bump, add it explicitly
  to this RUN line, don't drop `--no-deps`

## Runtime smoke file

Lives in the deploy dir (not generated inline in Dockerfile); the COPY
above references it. See `smoke_torch<M><m>.sh` for the 6-step battery
in `_shared/references/patterns/domains/day0-deploy-artifacts.md` and
sibling sessions.

## Post-build verification (pre-smoke sanity)

```bash
docker run --rm <overlay-image> python3 -c "
from importlib.metadata import version as v
print('torch:', v('torch'))
print('torch_npu:', v('torch_npu'))
print('torchvision:', v('torchvision'))
print('torchaudio:', v('torchaudio'))
"
```

This doesn't trigger the `import torch` trap. Full runtime validation
is the smoke container with NPU devices mounted.

## Cleanup

`cleanup_session.sh --session-tag <TAG> --preserve-image` keeps the
overlay for downstream experts (e.g. vllm-ascend-day0 will
`FROM easyr1-npu-torch<M><m>:<TAG>`).

## Deploy-time companion

For downstream-facing deploy, also prepare the 4 other deliverables
per `_shared/references/patterns/domains/day0-deploy-artifacts.md`:
`smoke_torch<M><m>.sh`, `deploy_torch<M><m>.sh`, `ONBOARDING.md`, and
`PR_MATERIAL.md` (if C-patch).
