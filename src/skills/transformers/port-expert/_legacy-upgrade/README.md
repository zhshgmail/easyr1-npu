# transformers-upgrade-expert

**Product**: a single-dep NPU upgrade expert. Given a target base-image swap
(e.g. `verl-8.5.0-a3` → `verl-8.5.2-a3`, which carries transformers 4.57→5.3,
vllm 0.13→0.18, vllm_ascend 0.13→0.17, torch_npu 2.7→2.9, CANN 8.5.0→8.5.1),
produce a **validated working image** on A3 that can host an EasyR1 training
workload without regression beyond the expected v2 baseline band.

## When to use

- `easyr1-expert`'s Phase P1 `dep-gap-detect` reports D≥1 for a dep that this
  expert's base-image swap covers.
- User explicitly asks to upgrade the NPU stack (e.g. new transformers
  version lands upstream and EasyR1 master requires it).

## When NOT to use

- EasyR1 itself needs code porting (NPU-CP-001..007) — that belongs to
  `easyr1-expert`.
- The target image already exists and validates (use `--reuse-image` path
  in easyr1-expert directly).

## Scope / boundaries

**In scope**:
- Write `Dockerfile.npu-<version>` layered on new base image
- Repair platform bugs specific to the new stack (NPU-BUG-001/004 variants)
- Cherry-pick or write backcompat shims for transformers/vllm API renames
  (EC-02 `no_init_weights`, EC-03 SamplingParams read-only, etc.) —
  backward-compat with v1 is MANDATORY, so easyr1-expert sees no regression
- Validate the stack with a single V1.4-equivalent smoke on the new image
  (2-chip, padding_free=False, 2-step) — assert step-1 entropy_loss in the
  image's own baseline band (v2: [1.21, 1.34] per SMOKE_BASELINE.md)

**Out of scope**:
- EasyR1 source code changes beyond the backcompat shims (NPU-CP-001..007 is
  easyr1-expert's domain; this expert only touches EasyR1 if an API rename
  forces a shim)
- Multi-dep-upgrade in one session (each dep gets its own expert session)

## Inputs (orchestrator passes via env)

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `trans-upg-20260430-1200` |
| `SOURCE_IMAGE` | Current working base image (v1 by convention) |
| `TARGET_IMAGE` | Upgrade target base image |
| `UPSTREAM_CONSUMER` | Which consumer repo this is for (default `EasyR1`) |
| `UPSTREAM_REF` | Consumer branch/commit at which to run validation smoke |
| `A3_HOST/PORT/USER`, `NPU_USER` | Standard |

## Deliverable

1. Dockerfile (`Dockerfile.npu-<target-short>`) committed to a branch named
   `ascend-upg-<SESSION_TAG>` in the consumer repo (branches off the
   baseline-working ref, not master)
2. Docker image `easyr1-npu:<SESSION_TAG>` built on A3
3. V1.4-equivalent smoke PASS on that image with entropy_loss in v2 band
4. `workspace/transformers-upgrade-{SESSION_TAG}/PROGRESS.md` + `RESULTS.md`
   with full provenance
5. Handoff payload back to orchestrator: `{image_tag, branch, v2_baseline_delta}`
6. OL-04b cleanup executed OR handoff image explicitly preserved via
   `--preserve-image` when the orchestrator plans to pass it to easyr1-expert
   via `--reuse-image`

## How it relates to the broader harness

- `_shared/references/ALWAYS_LOADED_UNIVERSAL.md` — universal OL rules
- Its own `references/ALWAYS_LOADED_RULES.md` adds OL-03 denylist (different
  from easyr1-expert's — this expert CAN read the `ascend-port-transformers-upgrade`
  drill branch since that's *its* prior art; it CANNOT read ascend-port* branches
  or round3/4 cold-drive workspaces since those are easyr1-expert's domain)
  and OL-08 edit-scope (Dockerfile.npu-* + 1-2 EasyR1 backcompat-shim files)
- Pinned to `_shared/` version via `SHARED_VERSION.txt`
