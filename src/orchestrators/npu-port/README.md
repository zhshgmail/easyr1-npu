# npu-port — cross-expert orchestrator

**Product**: single-entry orchestration of the full NPU-port pipeline.
Given a consumer repo + target ref + candidate image, drives:

```
  /dep-analysis → (if P2) /<dep>-upgrade  → /<consumer>-port
                  (if P1) skip upgrade    ↘
                                           → final image + port branch + smoke PASS
```

No human interaction between stages. Orchestrator reads each stage's
handoff payload, decides the next expert to invoke (or surfaces to user
if a stage returns `stuck`), and emits a single final RESULTS.md.

## When to use

- User wants to port a consumer repo to NPU and doesn't want to manage
  each expert separately.
- Stage-0-like scenarios (D=0 → straight to port) AND Stage-1 scenarios
  (D≥1 → chain upgrade-expert → port-expert) both covered.

## When NOT to use

- Already-known-exact scenario: prefer invoking a single expert directly
  (e.g. `/easyr1-port reproduce v1` if you already know v1 works)
- When manual review is desired between stages (the orchestrator is
  autonomous; it exits only at "done", "stuck", or "@review-fail")

## Inputs

| Arg | Meaning | Default |
|---|---|---|
| `--consumer-repo` | consumer repo dir (upstream/<name>) | `upstream/EasyR1` |
| `--consumer-ref` | target commit/branch | consumer's `main` tip |
| `--candidate-image` | starting NPU base image | v1 (verl-8.5.0-a3) |
| `--session-tag` | e.g. `npu-port-20260501-0830` | auto-generated |
| `--dry-run` | run dep-analysis only, skip upgrade + port stages | false |

## Outputs

- Single `workspace/npu-port-<SESSION_TAG>/RESULTS.md` with:
  - dep-analysis result (scenario P1/P2, classification)
  - upgrade-expert run(s) result (if any)
  - port-expert run result (port branch, image tag, smoke numerics)
  - overall closure classification (CLOSED / PARTIAL / FAILED)
  - provenance table per artifact (every row must be worker or orchestrator;
    human-intervention in any row → not CLOSED)

## Invariants (orchestrator-level G1/G2/G3)

- **G1**: orchestrator (this skill's prompt) MUST NOT directly Edit any
  consumer-tree file or Dockerfile. All such edits go through experts.
  Each sub-skill's own PreToolUse hook enforces its slice of this.
- **G2**: orchestrator relies on each expert's G2 for static-check;
  orchestrator adds its own sanity check that each sub-expert's Handoff
  JSON parsed clean with required fields.
- **G3**: final RESULTS.md may say "closed" ONLY if the port-expert's
  Handoff cited a log + entropy_loss in the correct band. Orchestrator
  re-reads the port-expert's PROGRESS.md to verify (doesn't just trust
  the JSON).

## When to surface to user

- `dep-analysis` returns `stuck` (unknown D with no upgrade-expert) →
  surface with the unsupported dep list
- `<dep>-upgrade` returns `stuck` (platform bug with no known fix) →
  surface with the bug signature
- `<consumer>-port` returns `stuck` (port-side issue not in EC) →
  surface
- Any OL violation self-reported by any worker → surface + exit
  `@review-fail`

Surface = write to the user channel (Discord), stop the pipeline, don't
silently retry.

## How it relates to the broader harness

- Orchestrator lives in `src/orchestrators/npu-port/`, separate from
  `src/experts/` (it's not an expert — it's a chain-of-experts).
- Calls each expert via `Agent(subagent_type=Explore, brief=<expert>/agent.md)`
  passing SESSION_TAG + inputs via env. Experts' Stop hooks enforce
  their own invariants; orchestrator consumes their Handoff.
- Pins `_shared/` version via `SHARED_VERSION.txt` (for the universal
  OL set the orchestrator itself respects).
