---
name: sglang-npu-day0
description: >
  Day-0 NPU validation for SGLang against a new SGLang main / sgl-kernel-npu RC /
  CANN bump. Unlike vllm-ascend (plugin) or triton-ascend (vendored fork),
  SGLang's NPU support is in-tree (`pyproject_npu.toml` + NPU code paths in
  main repo) coordinated with a separate kernel library
  (`sgl-project/sgl-kernel-npu`). This skill detects when those 3 axes
  (sglang main / sgl-kernel-npu / CANN+torch_npu) drift apart and routes
  the user to the right hand-off.

  Usage: /sglang-npu-day0 --target-sglang-tag <tag>
                          --target-kernel-npu-tag <tag>
                          --target-cann-version <ver>
                          [--device-type a3|a2]
argument-hint: >
  target-sglang-tag: e.g. v0.5.10.post1 or main
  target-kernel-npu-tag: e.g. 2026.04.15.rc4
  target-cann-version: e.g. 8.5.0
  device-type: a3 (default) or a2
context: inline
---

# /sglang-npu-day0 — SGLang version validation on Ascend NPU

## What it does

SGLang's Ascend NPU support is **upstream first-class** as of 2026-04
(see `docs.sglang.io/platforms/ascend/`). This skill is **NOT** for
porting SGLang to NPU (it's already done). It's for **catching the
3-axis drift** that breaks the working stack:

| Axis | Repo / artifact | Release cadence |
|---|---|---|
| `sglang` main | [`sgl-project/sglang`](https://github.com/sgl-project/sglang) | ~monthly (`v0.5.10.post1` 2026-04-09) |
| `sgl-kernel-npu` | [`sgl-project/sgl-kernel-npu`](https://github.com/sgl-project/sgl-kernel-npu) | weekly RC (`2026.04.15.rc4` 2026-04-22) |
| CANN + `torch_npu` | Ascend toolkit + PTA | quarterly (CANN 8.5.0 + PTA 7.3.0 + torch_npu 2.8.0.post2) |

When any axis advances, the other two need re-validation. This skill
runs that validation and emits an outcome (A/B/C).

## When to use

- **`pyproject_npu.toml` changes**: SGLang main bumps `transformers==X` or
  `torchao==Y` pin → run probe to see if integrated overlay still loads
- **`sgl-kernel-npu` new RC**: pair-test against current sglang main
- **CANN / PTA bump**: re-test sgl-kernel-npu wheels (they're built per
  CANN+PTA ABI)
- **Customer wants SGLang on A3**: cold-drive the whole prereq chain to
  produce a customer ONBOARDING note

## When not to use

- New SGLang **feature** support (e.g. add a model architecture) — that's
  upstream sglang work, not day-0
- vllm-ascend port-expert territory: SGLang ≠ vllm; don't conflate
- Per-model perf tuning: separate benchmark task
- EasyR1 / verl integration: SGLang is a serving framework, not
  trainer; if EasyR1 wants SGLang as rollout backend, that's a separate
  consumer-side port (different from this skill)

## Prerequisites

- A3 host with CANN driver ≥ 25.5.x; HDK 25.5.2 (per upstream docs)
- Docker, privileged + `--ipc=host`
- ≥ 50 GB free for the SGLang NPU image
- Probe target image existence on `quay.io/ascend/sglang:<tag>-cann<ver>-<device>`
  before running

## Workflow (P0..P6)

### P0 — parse args + sanity check upstream existence

```bash
SGLANG_TAG=${1:-main}        # or v0.5.10.post1
KERNEL_NPU_TAG=${2:-2026.04.15.rc4}
CANN_VERSION=${3:-8.5.0}
DEVICE=${4:-a3}              # a3 or a2

# Probe official image existence:
docker manifest inspect quay.io/ascend/sglang:${SGLANG_TAG}-cann${CANN_VERSION}-${DEVICE} \
  || { echo "image not published yet"; exit 1; }

# Probe sgl-kernel-npu release zip:
curl -s -I "https://github.com/sgl-project/sgl-kernel-npu/releases/download/${KERNEL_NPU_TAG}/sgl-kernel-npu-${KERNEL_NPU_TAG}-torch2.8.0-py311-cann${CANN_VERSION}-${DEVICE}-x86_64.zip" \
  | head -1
```

If either FAIL, abort: customer's target version isn't released yet.

### P1 — pull image + import smoke

```bash
docker pull quay.nju.edu.cn/ascend/sglang:${SGLANG_TAG}-cann${CANN_VERSION}-${DEVICE} \
  || docker pull quay.io/ascend/sglang:${SGLANG_TAG}-cann${CANN_VERSION}-${DEVICE}
```

Inside the image (with NPU bind set, see
[`_shared/integrated-overlay-build/SKILL.md`](../../_shared/integrated-overlay-build/SKILL.md)
P4 for the bind block):

```python
import torch, torch_npu
print("torch", torch.__version__, "torch_npu", torch_npu.__version__,
      "device_count", torch.npu.device_count())

import sglang
import sglang.srt.entrypoints.engine
from sglang.srt.utils.common import is_npu, is_cuda
print("is_npu:", is_npu(), "Engine:", sglang.srt.entrypoints.engine.Engine)

import sgl_kernel_npu
print("sgl_kernel_npu:", sgl_kernel_npu.__file__)

import deep_ep
print("deep_ep:", deep_ep.__file__)
```

PASS criteria:
- `is_npu()` returns `True`
- `device_count()` ≥ 1
- All 4 imports succeed without `ImportError`

### P2 — pyproject_npu.toml drift scan

Compare `pyproject.toml` (GPU default) vs `pyproject_npu.toml` (NPU
override) at the target sglang tag:

```bash
git -C /sgl-workspace/sglang diff main:python/pyproject.toml \
                               main:python/pyproject_npu.toml \
  -- :^*.lock | grep -E '^[+-]\s*"' | head -30
```

Look for:
- `transformers==X` pin (NPU likely on a slightly older transformers than GPU)
- `torchao==X` pin (NPU's `0.9.0+cpu` vs GPU's newer)
- `flashinfer-python` (GPU only, NPU should remove)
- `triton-ascend` vs `triton` (NPU uses `triton-ascend`)

If a NPU-pinned dep moved beyond what the integrated overlay currently
ships → flag for `/transformers-day0` follow-up.

### P3 — server start smoke

```bash
docker run -d --name sglang-smoke \
  --privileged --ipc=host --shm-size=64g \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --device /dev/davinci2 \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  ...   # full NPU bind set per NPU-OPS-009
  -p 30000:30000 \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  quay.io/ascend/sglang:${SGLANG_TAG}-cann${CANN_VERSION}-${DEVICE} \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 0.0.0.0 --port 30000 \
    --trust-remote-code \
    --mem-fraction-static 0.7

# Wait for "ready" log:
docker logs -f sglang-smoke 2>&1 | grep -m1 'The server is fired up'
```

Then curl test:

```bash
curl http://localhost:30000/generate -H 'Content-Type: application/json' \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}' \
  | python3 -m json.tool
```

PASS criteria:
- Server prints "The server is fired up" within 5 minutes
- Curl returns 200 with non-empty `text` field

### P4 — multi-card / TP smoke (optional but recommended)

For 2-card TP=2 with a model that exercises kernel-npu ops:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --tp 2 --dp 1 \
  --quantization modelslim \
  --host 0.0.0.0 --port 30000
```

PASS criteria:
- Server starts within 15 minutes
- Curl returns coherent output

If FAIL with `ModuleNotFoundError: sglang.srt.layers.attention.npu.X` →
sgl-kernel-npu vs sglang version mismatch (axis 1 ↔ axis 2 drift).

### P5 — record outcome

| Outcome | Meaning | Action |
|---|---|---|
| **A** | All 4 imports + server start + curl + TP=2 PASS | Ship: image tag becomes a known-good combo |
| **A-with-note** | Server PASS but specific feature unsupported (e.g. `--quantization` flag not ready in this kernel-npu tag) | Ship + ONBOARDING note about the limitation |
| **B** | Single env var / sglang flag workaround needed (e.g. `--attention-backend torch_native` if NPU backend missing) | Document workaround in customer ONBOARDING |
| **C-patch** | Real NPU-side bug requires fix in sgl-kernel-npu | File issue at `sgl-project/sgl-kernel-npu`; we don't fork (upstream is responsive) |
| **C-report** | Bug lives in community sglang main | File issue at `sgl-project/sglang` |

Unlike vllm-ascend / torch-npu / transformers, **we do NOT fork
SGLang or sgl-kernel-npu**. Both upstream repos have active NPU
maintainers (per `pyproject_npu.toml` + `2026 Q2 NPU roadmap` in
sglang issue #22949), so day-0 hand-off is via GitHub issue, not
fork branch.

### P6 — handoff

Write to `docs/sglang/PR_MATERIAL_<sglang-tag>_<kernel-tag>_<cann>_<outcome>.md`:

- Image tag tested
- 3-axis version triple
- P1..P4 PASS/FAIL per check
- Smoke logs + curl response samples
- Outcome classification + workaround / blocker

## What is different from the other 4 day-0 skills

| Aspect | vllm-ascend / torch-npu / transformers / triton-ascend | sglang |
|---|---|---|
| NPU port location | Plugin / vendored fork in NPU-side repo | **In-tree in `sgl-project/sglang` main** |
| Drift surface | F1..F8 community→NPU API drift | 3-axis version coordination |
| Our deliverable | `ascend-port/<target>` fork branch + PR_MATERIAL | Issue at upstream + customer ONBOARDING (no fork) |
| Validation level | Import + V1.4 GRPO | Import + server start + inference curl |
| Scanner | `sweep.sh` over commit range | `pyproject_npu.toml` diff + image probe |

## Common gotchas

1. **Image not published yet**: SGLang RCs on GitHub may not have a
   matching `quay.io/ascend/sglang:<tag>-cann<ver>-a3` published. Either
   wait for image, or build from source via `docker/npu.Dockerfile`.

2. **sgl-kernel-npu wheel pinned to specific torch+CANN**: the wheel name
   embeds `torch2.8.0`, `cann8.5.0`. Mixing wheels = ABI mismatch silent
   FAIL similar to `vllm-ascend-002` (fix-c-image-name-is-not-proof).
   Always re-`pip show sgl_kernel_npu` after every image change.

3. **`pyproject_npu.toml` vs `pyproject.toml`**: only **one** is active
   (the build moves `pyproject_npu.toml` over `pyproject.toml`). When
   debugging "why is this dep wrong version", check which one was active
   at install time:
   ```bash
   docker run --rm <image> head -3 /sgl-workspace/sglang/python/pyproject.toml
   ```

4. **Q1 2026 roadmap features may be partial**: `--grpc-mode`, SWA
   memory, certain prefill-decode disagg modes are listed "Planned" in
   `ascend_npu_support_features.md`. If customer hits one, that's
   roadmap blocker, not a bug.

## Provenance

- **T27** (2026-04-28): first-pass design + on-A3 import smoke via
  `quay.io/ascend/verl:verl-sglang-8.3.rc1-a3` image (same SGLang +
  sgl_kernel_npu + memfabric stack, older snapshot). 4-import smoke PASS.
- **T28** (2026-04-28): cold-drive on real
  `quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3` image (17 hours old).
  Findings:
  - Skill prereq script `probe_versions.sh` correctly verifies all 3
    axes have published artifacts (sglang main / sgl-kernel-npu
    2026.04.15.rc4 / cann 8.5.0).
  - Image pull, P1 import smoke all PASS.
  - P3 server start PASS (Uvicorn running on 30000).
  - **Inference call FAILS** with `RuntimeError: 0 active drivers` —
    `sglang main 0.5.10.post2.dev742` ↑ sees community triton driver
    init regression vs the bundled `sgl_kernel_npu 2026.3.1` wheel.
  - **Outcome C-report** — bug in upstream sglang↔sgl-kernel-npu
    pairing. See [`references/KB_INDEX.md` § "Concrete case registry"](references/KB_INDEX.md#concrete-case-registry)
    for the full reproducer + workaround.
  - **The skill itself worked as designed**: caught a real 3-axis
    version-mismatch bug on first cold-drive of the latest official image.

## See also

- [Upstream NPU docs](https://docs.sglang.io/platforms/ascend/) — official
- [`docker/npu.Dockerfile`](https://github.com/sgl-project/sglang/blob/main/docker/npu.Dockerfile) — canonical build
- [`sgl-kernel-npu` releases](https://github.com/sgl-project/sgl-kernel-npu/releases) — kernel RC list
- [`_shared/upstream-day0-workflow.md`](../../_shared/upstream-day0-workflow.md) — shared day-0 invariants
- [`_shared/integrated-overlay-build/SKILL.md`](../../_shared/integrated-overlay-build/SKILL.md) — NPU bind set + container helper
- [`docs/_meta/NPU_ADAPTATION_GAP.md`](../../../docs/_meta/NPU_ADAPTATION_GAP.md) — sglang previously listed as
  "档 C 不需要"; T27/T28 reclassification: **upstream-supported, day-0
  needed for version-drift coordination only**
