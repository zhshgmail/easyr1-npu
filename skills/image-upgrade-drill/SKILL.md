---
name: image-upgrade-drill
description: Rehearse a major base-image / framework upgrade (e.g. CANN 8.5.0→8.5.1, transformers 4→5, vllm 0.13→0.18) on a dedicated drill branch before touching the production stack. Produces a cost report (predicted vs actual), surfaces any new NPU-* pits as catalog entries, and leaves the team with a decision artifact — backward-compat code commits ready to cherry-pick, or evidence that the upgrade is blocked. Use whenever a new Ascend base image ships or an upstream framework tag is a candidate for the next port cycle.
---

# image-upgrade-drill

## What it does

Takes a candidate image/framework upgrade target and runs it through a structured rehearsal on a throwaway branch. Output is a drill report (`docs/<target>-upgrade-drill.md`) that either clears the upgrade to ship or documents why it's blocked. Along the way any new recurring pits land as stable IDs in `npu-patterns.md`.

The first concrete instance of this playbook is the transformers-upgrade drill on 2026-04-19 — `docs/transformers/transformers-upgrade-drill.md`. Read that before touching this skill; it's the ground truth for how the pieces fit.

## When to use

- A new `quay.io/ascend/verl:*` image ships and you want to know the cost of switching to it.
- An upstream framework (transformers, vllm, torch, torch_npu) cuts a new major version and you need to learn whether your port survives.
- A dependency you pin (e.g. `triton-ascend==3.2.0`) has a candidate upgrade and you want to de-risk it.
- Before bumping CI or the team's default image, to generate evidence a reviewer will trust.

## When not to use

- For a **minor** upstream bump where pip semver guarantees compatibility (just run V1.4 and V2.2 smokes directly on the new deps; no branch needed).
- For a bug-fix on an unchanged stack — that's a regular PR, not a drill.
- When you don't have a short-list of **candidate target versions** yet. Do `inspect-ascend-image` on a few options first; this skill consumes that output.

## Prerequisites

- Working V1.4 + V2.2 smokes on the current (pre-upgrade) image as your **baseline reference** — you cannot evaluate drift without one.
- Access to the target image (pulled or available via a CN mirror per NPU-OPS-006).
- A3 host reachable with the user's chip quota.
- `docs/easyr1/PORT-SUMMARY.md` up to date — step 6 of this skill feeds back into it.

## How to invoke

This is a six-step playbook, not a single command. Each step has a decision point.

### Step 1 — Pick the target + establish the baseline reference

- Run `npu-image-inspect` against the new image to lock in the actual installed versions (image tags drift from what the release notes claim; 8.5.2's "qwen3-5" tag shipped transformers 5.3.0.dev0, not the 5.0 the repo nominally targets).
- Identify **every** dependency that's moving, not just the one you care about. Our transformers drill accidentally upgraded vllm 0.13→0.18, torch_npu 2.7→2.9, and CANN 8.5.0→8.5.1 simultaneously because they all ship in the same base image.
- Record the baseline on the old image: exact `entropy_loss` and `grad_norm` from V1.4 step 1 and V2.2 step 1. You need numeric-equivalence comparisons later; approximate bands aren't enough.

### Step 2 — Infra pre-flight

Before building anything, check the new base image for the three pits that **will** slow you down if you hit them reactively (15 min each) but cost 30 seconds to audit:

```bash
# NPU-OPS-007: does the image ship a pip index?
docker run --rm --entrypoint cat <image> /etc/pip.conf 2>/dev/null || echo 'NO_PIP_CONF'
docker run --rm --entrypoint env <image> | grep PIP_INDEX_URL || echo 'NO_PIP_INDEX_URL_ENV'

# NPU-OPS-008: does huaweicloud still have the wheels we pin?
for pkg in triton-ascend torch-npu vllm-ascend; do
  count=$(curl -sL --max-time 30 "https://mirrors.huaweicloud.com/ascend/repos/pypi/$pkg/" | \
    grep -coE "$pkg[._-][^\"<>]*\\.whl")
  echo "$pkg: $count wheels on huaweicloud"
done

# NPU-OPS-006: does the host proxy need an override?
docker info | grep -A2 'Registry Mirrors'
systemctl cat docker | grep -E 'HTTP_PROXY|NO_PROXY'
```

If `/etc/pip.conf` is empty, bake `ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` into your drill Dockerfile up front. If huaweicloud returns 0 wheels for a package you pin, wire the drill Dockerfile to aliyun's simple index instead. If the daemon has a proxy, add your chosen mirror host to `NO_PROXY` before the first pull.

### Step 3 — Stand up the drill branch

```bash
# From the upstream repo (e.g. EasyR1)
git checkout -b ascend-port-<target>-upgrade ascend-port   # branch name is a contract
cp Dockerfile.npu Dockerfile.npu-<short-target>             # never modify the prod Dockerfile
# Relax any version pins that the new image enforces as system packages
# (e.g. transformers>=4.54,<5.0.0 -> transformers>=4.54.0 if image ships 5.x)
$EDITOR requirements.txt
git add Dockerfile.npu-<short-target> requirements.txt
git commit -m "[drill] Dockerfile.npu-<short> + relax <dep> upper pin"
```

The `[drill]` commit prefix is part of the contract — it's how the cherry-pick step knows which commits belong only on the drill branch vs which should flow to mainline.

### Step 4 — Build + smoke, iterate on code breaks

Run your V2.2 smoke against the new image inside the drill branch. **Expect 1-3 real API breaks** per major upgrade. Each one: read the traceback, find the moved symbol via `grep` in the new upstream source (which you should have checked out locally to match the new image's version — see NPU-version-aware-reviews memory), write a backward-compatible fix (`try/except` on imports, `hasattr` gates on attributes, `isinstance(descriptor, property) and descriptor.fset is None` for read-only properties), commit without the `[drill]` prefix.

For the **method** — how to identify, grep, and fix breaks — follow this pattern:
1. Read the traceback; note the module + symbol that failed to import or attribute that failed to set
2. Grep the new upstream source (at the ref matching the new image, per `knowledge/upstream-refs.md`) for where that symbol now lives
3. Write the backward-compat fix. Common patterns:
   - **Import moved**: `try/except ImportError` with both old and new paths
   - **Attribute removed / gated**: `hasattr(obj, "attr")` before reading
   - **Field became read-only `@property`**: `isinstance(cls_attr, property) and cls_attr.fset is None` before setattr
   - **API renamed**: `try` new name, `except AttributeError/ImportError` fall back to old
4. Commit without the `[drill]` prefix

Worked example: **`repo/docs/transformers/transformers-upgrade-drill.md`** is the 2026-04 drill's full report — includes both API breaks, both diffs, and the reasoning. **Do not open it during a dry-run skill test** — it's the answer key. Open it only after you've attempted the drill yourself (or for a post-mortem).

Expected size: **1-3 commits, 2-20 LOC** is typical for a major bump when the helper-layer is doing its job; if you find yourself writing 100+ LOC you're probably also refactoring, stop and narrow scope.

### Step 5 — Probe any carried-forward bugs (NPU-BUG-00*)

The V2.2 smoke has workarounds baked in (e.g. `use_torch_compile=false` for BUG-003, `VLLM_ASCEND_ENABLE_NZ=0` for ENV-002). A drill that inherits the workarounds does **not** tell you whether the underlying bug is fixed on the new stack. Write a sibling probe script (identical to V2.2 but with one workaround flipped to its "risky" value) and run it explicitly, once per carried-forward bug you want to de-risk.

Record the outcome in the BUG entry's "Status on <image>" line. **Don't trust a clean run; compare the metrics against the eager baseline**. The transformers drill's BUG-003 probe found that the kernel ran without crashing but produced `grad_norm=88973` (vs baseline 1.49) at step 1 and then tripped the same vector-core exception at step 2 — silent-corruption plus delayed crash is the 8.5.1 failure mode, worse than 8.5.0's immediate crash.

New pits surfaced by the probe get stable IDs (`NPU-BUG-<next>`, `NPU-OPS-<next>`) with full Symptom/Root cause/Fix/Commit ref/Generalizable rule schema, even if they're not on the critical path. The drill's job is to build the catalog, not just to pass.

### Step 6 — Trajectory smoke

A 2-step smoke proves only that forward + first gradient match baseline. Run a 20-step smoke (V2.2 config with `max_steps=20`) to confirm the stack is stable over a non-trivial trajectory: `entropy_loss` stays in a sensible band, `grad_norm` doesn't blow up, no HCCL/OOM/vector-core errors, checkpoint saves correctly.

Criteria for pass:
- 20/20 training steps finish, validation + checkpoint both clean.
- `entropy_loss` trajectory inside `[0.5, 2.5]` for Qwen2-0.5B (widen for other models).
- `grad_norm` max inside `[0, 20]` excluding reward-tied-zero steps.
- No new error signatures vs the baseline run's log.

If any criterion fails, **the upgrade is blocked** regardless of how clean step-1 looked. Capture the failure in a new BUG entry and decide: fix and re-drill, or shelve the upgrade.

### Step 7 — Report + close-out

Write `docs/<target>-upgrade-drill.md` following the transformers drill's structure:
- Result (PASS/BLOCKED) + the numeric evidence (step-1 entropy_loss vs baseline; 20-step band).
- Actual vs predicted cost table (wall-clock phases + LOC changed + iteration count).
- What broke vs what the playbook predicted, with ✅/⚠️/❌ per risk class.
- New operational pits surfaced (each with its stable ID cross-ref).
- Recommendations for PORT-SUMMARY.md.
- Open follow-ups (things the drill couldn't answer — e.g. convergence over 500 steps, other model families).

Then:
- Cherry-pick backward-compat commits from the drill branch onto `ascend-port` mainline (not the `[drill]`-prefixed commits; those stay on the branch).
- Push both branches to `personal/*` so the team can see them.
- Update `PORT-SUMMARY.md` Known-debt with the drill outcome + new cost data point (§6 is now "two data points" not "one").

## Rules / gotchas

- **Never modify `Dockerfile.npu`** on the drill branch. Always work on a separate `Dockerfile.npu-<target>`. The prod Dockerfile's only drill-branch change should be none.
- **Don't skip the baseline.** "It probably still matches V1.4" is the number one way drills go wrong. Record the actual baseline numbers before you start.
- **One flag at a time in the probes.** If you flip `use_torch_compile=true` AND switch the image simultaneously, you can't tell which change caused the failure. Probe scripts should vary exactly one knob from V2.2.
- **Silent corruption is the failure mode you miss.** A crash is a feature — it tells you something broke. A clean run with bad metrics is what you have to defend against; always compare numerics, not just exit codes.
- **The `[drill]` prefix is a contract.** Commits with `[drill]` are branch-only (Dockerfile changes, probe scripts, smoke scripts with bumped `max_steps`). Commits without it are candidates for cherry-pick to mainline. Keep the boundary clean so step-7 cherry-pick is mechanical.
- **Infra pits compound.** Hitting NPU-OPS-006, 007, 008 reactively costs ~45 min of iteration each. Doing the Step-2 pre-flight costs 2 min total. The compounding factor is real; respect it.
- **Don't rebuild unnecessarily.** `/opt/easyr1` is bind-mounted by `npu-container-runner`; code fixes don't need a rebuild, only Dockerfile changes do. The transformers drill rebuilt 3 times for Dockerfile fixes and 0 times for code fixes (correct ratio).

## What a drill is NOT

- A substitute for production convergence testing. A 20-step smoke is a necessary gate, not a sufficient one.
- A one-shot operation. Expect to iterate Step 4 two or three times.
- A code-refactor opportunity. If you want to clean up the rollout loop, do it in a separate PR after the drill. Drills find breakage; they don't invent it.
- A reason to delete the baseline image. Keep the pre-upgrade image around until the drill branch merges — you may need to re-run the baseline.

## Related

- `docs/transformers/transformers-upgrade-drill.md` — the first instance; treat as the reference implementation.
- `docs/easyr1/PORT-SUMMARY.md` §5-6 — the upstream playbook this skill operationalizes.
- `knowledge/npu-patterns.md` — where new findings from a drill land.
- `knowledge/smoke-ladder-convention.md` — baseline smoke levels (V1.1 → V2.2) referenced by Step 1.
- `skills/npu-image-inspect/` — Step 1's input.
- `skills/npu-container-runner/` — Step 4's harness (`run-npu-container.sh`).
- `skills/upstream-branch-hygiene/` — how the drill branch integrates with git remotes.
- `skills/codex-review/` — optional sign-off for Step 7's report.
- memory `a3_docker_proxy.md` — recovery if NPU-OPS-006 bites in Step 2.
