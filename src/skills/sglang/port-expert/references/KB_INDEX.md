# sglang-day0-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL rules | Phase A |
| [../../../_shared/upstream-day0-workflow.md](../../../_shared/upstream-day0-workflow.md) | G1/G2/G3 invariants | Phase A |

## Quick symptoms → classification

| Symptom | Likely outcome | Route |
|---|---|---|
| `quay.io/ascend/sglang:<tag>-cann<ver>-a3` 404 | **Skip** | Image not published yet; wait or build from `docker/npu.Dockerfile` |
| `import sglang` PASS, `import sgl_kernel_npu` FAIL with `_C` ABI error | **C-patch** | sgl-kernel-npu wheel was built against different torch/CANN; pin matching wheel via `--target-kernel-npu-tag` |
| Server starts, model loads, curl returns coherent text | **A** | Ship image tag as known-good combo |
| Server starts but inference returns garbage / NaN | **C-report** | Numeric regression in sgl-kernel-npu kernel — file at `sgl-project/sgl-kernel-npu` |
| Server starts but throws `NotImplementedError` on specific feature flag (`--grpc-mode`, `--enable-multimodal`, etc.) | **A-with-note** | Feature is "Planned" per upstream `ascend_npu_support_features.md` table; document as known limitation |
| `pyproject_npu.toml` pins `transformers==X` and X conflicts with EasyR1 cap | **A-with-note** | Loosen consumer cap; route to `/transformers-day0` for byte-compat verification |

## SGLang NPU 3-axis version triple

The 3 axes that need to be coordinated. Mismatched triples is the #1
cause of "image worked yesterday, broken today":

| Axis | Source of truth | Cadence | Latest as of 2026-04-28 |
|---|---|---|---|
| `sglang` main / tag | [`sgl-project/sglang/releases`](https://github.com/sgl-project/sglang/releases) | ~monthly major + post1 | `v0.5.10.post1` (2026-04-09) |
| `sgl-kernel-npu` | [`sgl-project/sgl-kernel-npu/releases`](https://github.com/sgl-project/sgl-kernel-npu/releases) | weekly RC | `2026.04.15.rc4` (2026-04-22) |
| CANN + torch_npu | hiascend.com / Ascend/pytorch | quarterly | CANN 8.5.0 + torch_npu 2.8.0.post2 (PTA 7.3.0) |

The `quay.io/ascend/sglang:<tag>-cann<ver>-<device>` image tag encodes
two of three axes; the third (`sgl-kernel-npu`) is the wheel inside.
**`docker inspect <image>` doesn't tell you which sgl-kernel-npu RC
went in** — only `pip show sgl_kernel_npu` does.

## SGLang NPU-supported features (snapshot — re-check per release)

Per upstream `ascend_npu_support_features.md`. When `Server supported`
column says:

- **A2, A3** → both Atlas series support
- **Planned** → roadmap, not in current release
- **Special For GPU** → never supported on NPU

Categories with "Planned" entries to watch (2026-04 snapshot):
- `--grpc-mode` (HTTP server section)
- `--swa-full-tokens-ratio`, `--disable-hybrid-swa-memory` (memory section)
- Specific quantization params (`--modelopt-*`, `--quantize-and-serve`, `--rl-quant-profile`)

## Supported model families (snapshot)

Per `ascend_npu_support_models.md`. Key model families validated on
A3 + A2 (2026-04 snapshot):

- **DeepSeek**: V3 / V3.1 / V3.2-W8A8 / R1-0528-W8A8 / V2-Lite-W8A8
- **Qwen**: Qwen3-30B-A3B-Instruct, Qwen3-32B, Qwen3-Next-80B-A3B,
  Qwen3-Coder-480B-A35B-w8a8-QuaRot, Qwen2.5-7B, Qwen3-0.6B, QWQ-32B-W8A8
- **Llama**: Llama-4-Scout-17B-16E-Instruct, Llama-3.1-8B, Llama-3.2-1B
- **GLM**: GLM-4-9B-Chat, GLM-5 (separate dedicated example doc)
- **Multimodal**: Phi-4-multimodal, qwen3-VL family (per separate doc)
- **MoE**: GLM MoE, Granite MoE, OLMoE, ERNIE-4.5 MoE, XVERSE-MoE-A36B,
  DBRX, Mistral, Llama-4 MoE
- **Other**: Gemma-3, Baichuan2, Kimi-K2-Thinking, MiMo-7B-RL,
  Command-R, Grok, ChatGLM, InternLM 2, MiniCPM v3, Persimmon, Ling,
  StableLM, SmolLM, Arcee AFM, Phi, EXAONE 3

If customer requests a model **NOT in this list** → A-with-note (unknown,
needs new validation) or C-report (architecture not yet wired).

## Reproducer for cold-drive

```bash
# 0) Probe upstream image existence
docker manifest inspect quay.io/ascend/sglang:main-cann8.5.0-a3 \
  || docker manifest inspect quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3

# 1) Pull
docker pull quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3

# 2) Import smoke (with NPU bind set)
bash repo/src/scripts/run-npu-container.sh \
  --chips 0 \
  --image quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3 \
  -- python3 -c '
import torch, torch_npu, sglang, sgl_kernel_npu, deep_ep
from sglang.srt.utils.common import is_npu
from sglang.srt.entrypoints.engine import Engine
print("imports OK; is_npu=", is_npu())
'

# 3) Server smoke (Qwen3-0.6B is the smallest validated model)
bash repo/src/scripts/run-npu-container.sh \
  --chips 0 \
  --image quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3 \
  -- python3 -m sglang.launch_server \
       --model-path Qwen/Qwen3-0.6B \
       --host 0.0.0.0 --port 30000 \
       --trust-remote-code \
       --mem-fraction-static 0.7

# 4) From host (other terminal):
curl http://A3_HOST:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}'
```

## What is NOT this skill's job

- **Not** porting SGLang to NPU from scratch (already upstream supported)
- **Not** writing NPU kernels — that's `sgl-project/sgl-kernel-npu` upstream
- **Not** verl-side rollout integration with SGLang (different path; if
  needed becomes a `/easyr1-port-sglang-rollout` skill)
- **Not** model architecture support — file at upstream

## Concrete case registry

### Case 1 — 2026-04-28 T28 cold-drive of official image catches version-mismatch bug

**Setup**:
- Image: `quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3` (15.2 GB, 17h-old)
- Installed triple: `sglang 0.5.10.post2.dev742+g47b8eadbc` / `sgl_kernel_npu 2026.3.1` / `torch_npu 2.8.0.post2`
- Reproducer: per § "Reproducer for cold-drive" above, `python3 -m sglang.launch_server --model-path /models/Qwen3-0.6B`

**Result**:
- P1 import smoke **PASS** (all 4 imports clean, `is_npu()=True`)
- P3 server start: `Application startup complete` + `Uvicorn running on http://0.0.0.0:30000`
- **First inference request triggers**: `RuntimeError: 0 active drivers ([]). There should only be one.` in `triton/runtime/driver.py:9` `_create_driver`
- Same symptom with `--disable-cuda-graph --disable-piecewise-cuda-graph --disable-radix-cache --attention-backend ascend`

**Root cause** (preliminary): `sglang main` (0.5.10.post2.dev742, late-April mainline tip) imports community triton's `runtime.driver` module which expects exactly one registered active driver. `sgl_kernel_npu 2026.3.1` (March-built) was packaged before the relevant driver-registration code path stabilized. Mainline sglang has moved past what the bundled kernel-npu wheel wires up.

**Outcome classification**: **C-report**
- Not a fork-able fix in our repo: bug is in `sglang main` ↔ `sgl-kernel-npu` coordination
- Related issues in upstream:
  - [`sgl-project/sglang#13648`](https://github.com/sgl-project/sglang/issues/13648) — "image worked yesterday, broken today" pattern
  - [`sgl-project/sglang#16360`](https://github.com/sgl-project/sglang/issues/16360) — Triton compilation error on Ascend
- **Workaround for customers today**: pin to the older verl-sglang image (`quay.io/ascend/verl:verl-sglang-8.3.rc1-a3-...`) where `sglang` and `sgl_kernel_npu` were jointly tagged; or wait for the next image rebuild that pairs `sgl_kernel_npu >= 2026.04.15.rc4` with sglang main

**Validation of skill itself**: this is the **first time** the `/sglang-npu-day0` skill ran cold on a fresh agent against the official current image, and it correctly identified a real 3-axis drift bug that customer would otherwise hit silently. The skill workflow worked as designed.

## Provenance

- T27 (2026-04-28): design + image probe via existing `verl-sglang-8.3.rc1-a3` image; 4-import smoke PASS
- T28 (2026-04-28): real `quay.nju.edu.cn/ascend/sglang:main-cann8.5.0-a3` cold-drive; identified Case 1 (C-report)
