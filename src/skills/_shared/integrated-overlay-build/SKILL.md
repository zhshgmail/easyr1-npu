---
name: integrated-overlay-build
description: Build a single A3-ready overlay docker image that stacks the 4 NPU-upstream ascend-port branches (vllm-ascend / torch-npu / transformers / triton-ascend) on a chosen vllm-era base image, install EasyR1 master with the transformers cap loosened, and validate end-to-end with a V1.4 GSM8K-style GRPO smoke. Output is `easyr1-npu:integrated-<DATE>` + a smoke log + an updated PR_MATERIAL doc. Use when a new vllm-era base ships, when one of the 4 ascend-port branches advances, or when EasyR1 master picks up a dep change.
---

# integrated-overlay-build

## What it does

End-to-end RL integration: takes the 4 already-validated ascend-port
branches (per `docs/_meta/UPSTREAM_FORKS.md`) + EasyR1 master + a
chosen base image, produces ONE customer-runnable image that
reproduces the 2026-04-27 V1.4 GRPO smoke. Each invocation is a
fresh proof that the per-skill ports compose correctly.

## When to use

- Base image bump (e.g. `easyr1-npu-vllm0200:iter21` ships with
  newer torch / vllm / transformers).
- One of the 4 ascend-port branches advances (vllm-ascend adds a
  new shim, torch-npu picks up a new compat module, etc).
- EasyR1 master changes `requirements.txt` or `verl/integrations/*`.
- Before a customer demo (re-validate the integrated stack).

## When not to use

- Per-skill validation. Each ascend-port branch has its own
  PR_MATERIAL.md; this skill is the **integration** layer above
  those.
- Single-component bug-fix that doesn't change the integration
  surface — just patch the per-skill branch.
- vllm-ascend / torch-npu / transformers / triton-ascend version
  upgrades — those are the per-skill day-0 tasks first; only feed
  output here once they pass.

## Prerequisites

- All 4 ascend-port branches at known-good tips per
  [`docs/_meta/UPSTREAM_FORKS.md`](../../../docs/_meta/UPSTREAM_FORKS.md):
  - `github.com/zhshgmail/vllm-ascend` `ascend-port/vllm-main`
  - `gitcode.com/zhengshencn_hwca/pytorch` `ascend-port/torch-2.12-rc3`
  - `github.com/zhshgmail/transformers` `ascend-port/transformers-v5.4`
    (marker only, no overlay needed if image ships compatible
    transformers)
  - `gitcode.com/zhengshencn_hwca/triton-ascend` `ascend-port/triton-v3.6.0`
    (no overlay; image vendor 3.2.0 already ships a working CANN-bundled
    bishengir; the source-build branch ships only as code-side
    artifact for the maintainer)
- EasyR1 master with `transformers<5.0.0` cap loosened — branch
  `ascend-port-integrated-<DATE>` on `github.com/zhshgmail/EasyR1`.
- A base image with a real (non-stub) vllm matching the version
  axis the shims target (currently vllm 0.20.0 — confirmed via
  `_shared/patterns/F-family-taxonomy.md` per-row analysis). Do NOT
  use `verl-8.5.2-a3` (vllm 0.18.0+empty stub) or
  `vllm-ascend:releases-v0.13.0-a3` (vllm 0.13, pre-symbol-introduction).
- A3 host access (`ssh -p 443 root@115.190.166.102`) with at least
  2 free chips on a single NPU card.
- Per-skill validation already PASSED on the chosen base image. If
  not, run those skills first; do not paper over with this skill.

## Workflow (P0..P7)

### P0 — choose base image + freeze inputs

Inputs to record:

| Input | How to choose |
|---|---|
| Base image | latest local `easyr1-npu-vllm0200:*` whose `pip show vllm` reports a real version (not `+empty`); current default `easyr1-npu-vllm0200:iter20-abi-both`. |
| 4 ascend-port branch tips | per UPSTREAM_FORKS.md ledger; record commit SHAs at start. |
| EasyR1 ascend-port-integrated-<DATE> branch | based on `ascend-port` + transformers-cap-lift commit. |
| Output image tag | `easyr1-npu:integrated-<YYYYMMDD>`; date is invocation date. |

### P1 — pre-flight: re-validate per-skill smokes still PASS on base

For each of the 4 ascend-port branches, re-run its on-A3 import
smoke (NOT walk-through — see
[`docs/_meta/kb/porting_lessons/cross-layer-007`](../../../docs/_meta/kb/porting_lessons/cross-layer-007-walk-through-is-not-real-run.md)).
PASS criteria from per-skill PR_MATERIAL.md:

- vllm-ascend: 3/3 shim modules import + lazy resolve PASS on the
  base image. Pattern in
  [`vllm-ascend-003`](../../../docs/_meta/kb/porting_lessons/vllm-ascend-003-shim-plugin-init-order.md).
- torch-npu: shim resolves OLD path on base image's torch (no-op
  when torch <= 2.11; only takes NEW path when torch >= 2.12).
- transformers: byte-compare image's transformers version against
  v5.4 baseline; outcome A or A-with-note.
- triton-ascend: vendor 6/6 NPU smoke PASS — see
  `repo/src/scripts/smoke_triton_vector_add.py`.

If any FAIL, abort: fix the per-skill branch first.

### P2 — assemble overlay Dockerfile

Single Dockerfile, copy-into-existing-image style:

```dockerfile
FROM <base-image>

# Layer 1: vllm-ascend compat shims + call-site swaps
COPY vllm-ascend-overlay/ /vllm-ascend/

# Layer 2: torch_npu compat (no-op at torch <= 2.11)
COPY torch-npu-overlay/ /usr/local/python3.11.14/lib/python3.11/site-packages/torch_npu/

# Layer 3: EasyR1 master with cap loosened
ARG EASYR1_BRANCH=ascend-port-integrated-<DATE>
RUN git clone --depth 1 -b ${EASYR1_BRANCH} https://github.com/zhshgmail/EasyR1.git /opt/easyr1 && \
    cd /opt/easyr1 && \
    pip install --no-deps -e .

# (transformers, triton-ascend: no overlay — image-default versions are sufficient.)
```

Build context lives at `/tmp/integrated-overlay/` on the developer
host, with subdirs populated by `scp`-from-fork-branches.

### P3 — build + tag

```bash
docker build -t easyr1-npu:integrated-$(date +%Y%m%d) /tmp/integrated-overlay/
```

Verify build success + record image SHA + size.

### P4 — sanity verify image

```bash
docker run --rm -e ASCEND_RT_VISIBLE_DEVICES=0 \
  --privileged --ipc=host --shm-size=64g \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --device /dev/davinci2 \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  ...   # full bind set per NPU-OPS-009 / NPU-OPS-011
  easyr1-npu:integrated-<DATE> \
  bash -c '
    pip show vllm vllm_ascend torch torch_npu transformers
    python3 -c "import vllm_ascend.compat.shared_fused_moe; print(\"shim 1 OK\")"
    python3 -c "import vllm_ascend.compat.default_moe_runner; print(\"shim 2 OK\")"
    python3 -c "import vllm_ascend.compat.spec_decode_base_proposer; print(\"shim 3 OK\")"
    python3 -c "from torch_npu.compat.inductor_codecache import Union; print(\"torch shim OK\")"
    python3 -c "import easyr1_or_verl_root_module"  # adapt
  '
```

All print statements must succeed.

### P5 — V1.4 GSM8K GRPO smoke

```bash
NPU_USER=<workspace-owner> \
  bash repo/scripts/run-npu-container.sh \
    --chips 0,1 \
    --image easyr1-npu:integrated-<DATE> \
    --live-source ~/workspace/easyr1-npu/upstream/EasyR1 \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

PASS criteria:
- 0 exceptions in `experiment_log.jsonl`
- checkpoint saved at `/tmp/<owner>/easyr1_smoke_ckpt/global_step_2/`
- val metrics in band (`accuracy_reward >= 0.01` for fresh baseline,
  or `entropy_loss within ±10%` of recorded baseline if baseline
  exists)

### P6 — record outputs

Update:

- `docs/_meta/UPSTREAM_FORKS.md` if any branch tip advanced.
- `docs/_meta/RL_INTEGRATION_PLAN.md` §T22.7 with new image SHA + smoke metrics.
- `docs/easyr1/PORT-GUIDE-v2-integrated.md` if image tag changes.

### P7 — handoff

Provide back:
- Image tag + SHA + size
- Smoke log path
- Checkpoint path
- 4 fork branch commit SHAs (immutable references)
- Status: ship / blocked / re-run

## Common gotchas (record from T22 sessions)

1. **`ASCEND_RT_VISIBLE_DEVICES`** is in-container chip index, NOT
   host phy-id. See
   [`NPU-OPS-012`](../../../knowledge/npu-patterns.md#npu-ops-012).
2. **ssh-as-root** requires `NPU_USER=<owner>` env when calling
   the container helper. See
   [`NPU-OPS-013`](../../../knowledge/npu-patterns.md#npu-ops-013).
3. **Base image version axis** matters. Shims target a specific
   vllm version range; mismatched base = silent FAIL. T22.4 row 3
   first attempt on vllm-ascend release-13 (vllm 0.13) failed
   because the symbols the shims target didn't exist yet in that
   vllm version.
4. **Don't conflate walk-through with on-A3 smoke**.
   [`cross-layer-007`](../../../docs/_meta/kb/porting_lessons/cross-layer-007-walk-through-is-not-real-run.md).
   This skill always runs P4+P5 with real on-A3 imports.
5. **Image cleanup**. Per memory `a3_cleanup_and_reuse.md`, delete
   intermediate dangling images and large tarballs at the end of
   the session if the new image is > 5GB and you've validated it.

## See also

- [`_shared/upstream-day0-workflow.md`](../upstream-day0-workflow.md)
  — per-skill workflow this builds on top of.
- [`_shared/patterns/F-family-taxonomy.md`](../patterns/F-family-taxonomy.md)
  — drift family vocabulary.
- [`docs/_meta/UPSTREAM_FORKS.md`](../../../docs/_meta/UPSTREAM_FORKS.md)
  — authoritative branch ledger.
- [`docs/_meta/RL_INTEGRATION_PLAN.md`](../../../docs/_meta/RL_INTEGRATION_PLAN.md)
  — T22's first-pass execution log; this skill encodes that into
  reusable form.
- [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](../../../docs/easyr1/PORT-GUIDE-v2-integrated.md)
  — customer-facing detail of the v2 integrated path.
- [`ONBOARDING.md`](../../../ONBOARDING.md) — one-page customer quickstart.

## Provenance

First-pass manual execution: T22 session 2026-04-27. This skill
captures the workflow that produced
`easyr1-npu:integrated-20260427` (SHA `044ba0b7618338`) + V1.4 GRPO
smoke PASS. See `RL_INTEGRATION_PLAN.md` §T22.7 for the original
run-log.

## T25 cold-drive replay (2026-04-28)

A second fresh agent re-ran P0..P7 against the existing image.
Results:

- P0..P4 — re-validated: all 4 ascend-port branch tips on
  `UPSTREAM_FORKS.md`; image `easyr1-npu:integrated-20260427`
  present on A3 with SHA `044ba0b76183`.
- **P5 V1.4 GRPO smoke** — re-PASSED on chips 2,3 (in-container
  index 0,1) after fixing two real bugs caught only by this replay:

  1. **NPU-OPS-014**: A3's `/home/z00637938/workspace/easyr1-npu/repo/`
     was a stale non-git copy of an early v0 layout (no `src/` dir).
     Documented commands `bash repo/src/scripts/run-npu-container.sh`
     fail with `No such file or directory`. Fix: re-`git clone` to
     A3 from origin/main.
  2. **NPU-OPS-012 helper bug**: `run-npu-container.sh` was setting
     `ASCEND_RT_VISIBLE_DEVICES=$CHIPS` — i.e. the host phy-id, not
     the in-container index. T22.7 worked only because `--chips 0,1`
     coincides numerically. With `--chips 2,3` or `--chips 4,5`, Ray
     reports `Total available GPUs 0 is less than total desired GPUs 2`.
     Fix: helper now derives `IN_CONTAINER_CSV=0,1,...,N-1` and emits
     that.

- **Smoke result on chips 2,3 with fixed helper**: 2 GRPO step + post-train val PASS, ~10 min.
  - `accuracy_reward: 0.014` (matches T22.7 baseline 0.014)
  - `reward_score: 0.013` (within ±10% of T22.7 0.0126)
  - `val_response_length mean: 184.3` (matches T22.7 184.3)
  - Checkpoint saved: `/tmp/<owner>/easyr1_smoke_ckpt/global_step_2/actor/`

The **value of cold-drive replay** is exactly catching bugs that
happy-path testing hides — both fixed bugs would have broken any
new user picking a non-zero chip range. KB and helper updated; both
documented for future replays.
