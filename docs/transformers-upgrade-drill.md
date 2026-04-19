# Transformers upgrade drill — report

**Branch**: `ascend-port-transformers-upgrade` (forked off V2.2 head `6f8197f`)
**Date**: 2026-04-19
**Drill target**: swap base image `verl-8.5.0-a3` → `verl-8.5.2-a3` (transformers `4.57.6` → `5.3.0.dev0`, vllm `0.13.1.dev18` → `0.18.0+empty`, vllm_ascend `0.13` → `0.17.0rc2`, torch_npu `2.7.1` → `2.9.0`, CANN `8.5.0` → `8.5.1`) on the same V2.2 smoke workload (4-chip + padding_free=True + ulysses=2) and measure actual cost vs the estimates in `PORT-SUMMARY.md`.

---

## Result: PASS

Two full GRPO steps completed end-to-end on the upgraded stack:

| | Step 1 | Step 2 |
| --- | --- | --- |
| `entropy_loss` | **1.434** | 1.58 |
| `grad_norm` | 1.493 | 0.0 (all rewards tied) |
| `pg_loss` | 0.0 | 0.0 |
| `ppo_kl` | 0.0 (KL disabled) | 0.0 |
| reward mean / max | 0.056 / 0.9 | 0.002 / — |
| validation `accuracy_reward` / `overall_reward` | — | 0.016 / 0.014 |

Step-1 `entropy_loss=1.434` sits inside the V1.4 / V2.2 baseline band (1.43-1.45), so the transformers 5.3.0.dev0 forward path is **numerically equivalent** to 4.57.6 for Qwen2-0.5B on NPU within the noise of a single 4-way FSDP run.

Checkpoint saved: 4-rank shards (model + optim + extra_state) under `/tmp/z00637938/easyr1_smoke_ckpt_v2_2/global_step_2/actor/`, including `huggingface/` model-config directory.

---

## Actual vs predicted cost

`PORT-SUMMARY.md` §5 ("If you have to upgrade X, do this") predicted the transformers-upgrade playbook cost and listed failure-mode classes. Here's how the drill's measured cost compares.

### Time: ~45 min wall-clock total, ~3 productive dev hours worth of activity

| Phase | Predicted | Actual | Δ |
| --- | --- | --- | --- |
| Pull new image | "several min" | **~18 min** (quay.io flapping from A3, switched to `quay.nju.edu.cn`) | worse |
| Build drill image | ~5 min | **~7 min** (3 rebuilds due to mirror/pipefail issues, ~2 min for the clean build) | same order |
| Smoke dev-loop (fix, scp, rerun) | 1-2 iterations | **3 smoke runs** — ImportError → eos_token_id → clean | slightly more |
| First green step | "~30 min from starting" | **~40 min** | +30% |

Overall: upgrade drill cost was within 30% of the estimate. The *code* diff to adapt EasyR1 was cheaper than predicted (2 commits, 4 changed LOC — see below). The *infrastructure* cost (image pull, build mirror, proxy) was higher than predicted because 3 new operational pits surfaced that we hadn't met on the 8.5.0 path.

### Code changes to EasyR1: 2 commits, 4 lines changed

| Commit | File | Change |
| --- | --- | --- |
| `55bb730` | `verl/workers/fsdp_workers.py` | `from transformers.modeling_utils import no_init_weights` → try/except with fallback to `transformers.initialization` |
| `d213f01` | `verl/workers/rollout/vllm_rollout_spmd.py` | `update_sampling_params` context-manager now detects read-only `property` descriptors (via `cls_attr.fset is None`) and skips them |

Both fixes are **backward-compatible** — old import path and settable fields still work on 8.5.0 (transformers 4.57 / vllm 0.13).

Dockerfile changes (separate from EasyR1 source): 3 commits tuning the 8.5.2-specific build recipe (`df84212`, `318925f`, + the follow-up Dockerfile edits that scpd in place on A3 — mirror-pinning, triton-ascend index, comments).

### What broke vs what we predicted

`PORT-SUMMARY.md` predicted these risk classes would need attention when transformers ≥5 lands:
1. ✅ "attention backend registration may move" — **did not hit**; `ALL_ATTENTION_FUNCTIONS` still at `transformers.modeling_utils` in 5.3.
2. ✅ "npu_flash_attention signatures may drift" — **did not hit**; signatures verified zero-diff between 4.57 and 5.3 in pre-drill local inspection.
3. ⚠️ "some private/utility imports may move" — **did hit**; `no_init_weights` moved `transformers.modeling_utils` → `transformers.initialization`. One import, simple fix.
4. ⚠️ "vllm SamplingParams API drift" — **did hit**; `eos_token_id` became a read-only `@property` in vllm 0.18. Simple fix.
5. ❌ "NPU-BUG-001 (triton-ascend partial install) may recur" — **confirmed recur**; `import triton_ascend` fails in the 8.5.2 base image just like 8.5.0. The force-reinstall from `Dockerfile.npu` had to stay.
6. ❌ "NPU-BUG-003 (inductor crash on log_probs) may reappear" — **could not evaluate** because V2.2 smoke already sets `use_torch_compile=false` as the V2.2 workaround, so the inductor path wasn't exercised. Verifying this bug's status in CANN 8.5.1 is a separate follow-up.
7. ❌ "vllm_ascend rename/move that was fixed in NPU-CP-002 / NPU-CP-004 may re-break" — **did not hit**; both fixes still work on vllm 0.18 without code change (the hasattr/fallback gates trip through correctly).

### New operational pits surfaced (to capture as NPU-OPS)

Three new reusable findings — all infrastructure, not code:

- **NPU-OPS-006** (already written): **docker proxy trap on A3**. The host-level docker daemon's `HTTP_PROXY=http://100.66.1.4:7897` was dead/slow today; `docker pull` timed out on TLS handshake even though `curl` from the same host reached the registry instantly. Fix: add the CN mirror hostname to docker's `NO_PROXY` and pull by mirror hostname. See `repo/knowledge/npu-patterns.md#npu-ops-006` and `memory/a3_docker_proxy.md`.
- **NPU-OPS-007** (to add): **8.5.2 base image has no `/etc/pip.conf`** (unlike 8.5.0 which had huaweicloud baked in). A naked `pip install` inside the build container hangs on pypi.org. Fix: set `ENV PIP_INDEX_URL=...` in the Dockerfile so the build container has a working CN mirror. Generalizable: whenever the base image changes, re-inspect `/etc/pip.conf` and `/root/.pip/pip.conf`.
- **NPU-OPS-008** (to add): **huaweicloud ascend pypi mirror is empty for `triton-ascend` today** (the index file lists 0 wheels). aliyun's general-purpose pypi simple index has the 3.2.0 wheels. Generalizable: don't hard-code huaweicloud as the triton-ascend source; prefer aliyun and keep huaweicloud as a fallback, not the default.

Plus one light operational lesson:

- **Launcher scripts must `set -o pipefail`**. Our `drill_launch.sh` had `set -eux` but `docker build ... 2>&1 | tee build.log` masked build failures (tee's 0 exit code swallowed pip's nonzero). A failed build proceeded to launch the smoke, which then tried `docker run` against a nonexistent image and hit docker.io. One-line fix in the harness.

---

## What this drill proves

1. **The ported EasyR1 is resilient to a major image swap.** A transformers 4 → 5 jump and a vllm 0.13 → 0.18 jump cost 2 commits / 4 LOC of code changes — the helper abstractions (`verl/utils/device.py`, the hasattr-gated vllm imports, the contextmanager-based sampling-params override) absorb most of the drift.
2. **NPU-CP-00* patterns are stable across the transformers upgrade.** None of the CP-001/002/003/004 fixes regressed. That's what makes them "patterns" rather than one-off patches.
3. **NPU-BUG-001 (triton-ascend broken install) is a base-image-level bug that recurs across image revisions.** Keep the repair step unconditional, not conditional on image version.
4. **The main cost of a transformers upgrade on NPU is operational, not code.** Image pull, mirror discovery, pip proxy, build-time index — these dominate the dev loop unless the base image inherits well-known defaults.
5. **Numerical equivalence holds across transformers 4.57 → 5.3.0.dev0** for Qwen2-0.5B GRPO on 4-chip ulysses + padding_free within this short smoke. Two steps isn't a full convergence proof; a V2.3-level 20-step smoke on 8.5.2 is the natural next drill.

---

## Recommendations for `PORT-SUMMARY.md`

Fold these in on the next edit:

- Revise the "estimated cost" for a transformers/vllm image upgrade from "~30 min end-to-end" to **"~45 min if you hit a new image-infra pit, ~20 min if not."** Add: "image-infra pits are worth 10-15 min each; expect 1-2 on a new base image."
- Add an upgrade pre-flight check: *before* running the smoke, inspect the new base image for `/etc/pip.conf`, `/root/.pip/pip.conf`, `PIP_INDEX_URL` env, and `HTTP_PROXY` daemon config. If any are missing/changed, patch the Dockerfile proactively instead of reactively.
- Add: "triton-ascend force-reinstall is mandatory, not conditional — `import triton_ascend` fails in every Ascend verl base image we've inspected (8.5.0 and 8.5.2 both broken)."

---

## Artifacts

- Drill branch: `personal/ascend-port-transformers-upgrade` (5 commits on top of `6f8197f`).
- Drill image: `easyr1-npu-852:drill` (layered on `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`).
- Smoke logs: `/tmp/z00637938/easyr1-logs/drill_20260419_135556.log` (the successful run); earlier failed runs in the same directory.
- Checkpoint: `/tmp/z00637938/easyr1_smoke_ckpt_v2_2/global_step_2/`.
- New memory: `a3_docker_proxy.md` (NPU-OPS-006 quick reference).

---

## Open follow-ups

1. Add NPU-OPS-007 (pip.conf absence) and NPU-OPS-008 (huaweicloud mirror empty) to `repo/knowledge/npu-patterns.md` with full Symptom/Root cause/Fix schema.
2. Add `set -o pipefail` to `repo/scripts/run-npu-container.sh` and any other harness scripts that use `| tee` or `|`. A one-liner but important.
3. Evaluate NPU-BUG-003 (inductor crash on log_probs) on CANN 8.5.1 by flipping `use_torch_compile=true` on the drill branch and rerunning. If CANN 8.5.1 fixes it, update the bug entry in `npu-patterns.md`.
4. Run a longer smoke on the drill branch (e.g. 20 steps, `use_torch_compile=false`) to confirm numerical equivalence over a non-trivial trajectory — step-1 match isn't a guarantee.
5. Decide whether to merge the drill branch's two code commits (`55bb730`, `d213f01`) back to `ascend-port` **now** (they're backward-compatible), or keep them on the drill branch until the team commits to the 8.5.2 image. The code is safe to take either way.
