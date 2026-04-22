---
name: dep-analysis
description: >
  Classify a consumer repo's runtime deps against an NPU base image into
  A/B/C/D/E (A=npu-native, B=npu-ported-fork, C=cuda-only-bypassable via
  shim, D=cuda-only BLOCKER, E=pure Python/CPU). Emit a task plan the
  orchestrator can act on: P1 (all A/B/C/E) → proceed to port-expert;
  P2 (any D) → spawn upgrade-expert(s) first. Read-only; no docker build,
  no A3 NPU work, no code writes outside the session workspace.

  Usage: /dep-analysis --consumer-ref <sha> --candidate-image <tag>
                       [--consumer-repo <path>] [--reqs-file <name>]
argument-hint: >
  consumer-ref: required, the target commit/branch of the consumer repo
  candidate-image: required, the NPU base image to evaluate against
  consumer-repo: optional, default upstream/EasyR1
  reqs-file: optional, default requirements.txt
context: inline
---

# /dep-analysis — dep classification + task planning (Stage 0)

**Your role**: spawn `dep-analysis-worker`, get its task plan JSON back,
hand it to the caller (typically the /npu-port orchestrator).

## Canonical workflow spec

Authoritative invariants + phase definition: `state_machine.yaml`.

## Workflow overview

```
P0: Parse args + load env
P1: Inspect candidate image (pip freeze; cached if available)
P2: Pull consumer requirements.txt at CONSUMER_REF
P3: Spawn dep-analysis-worker → runs scripts/dep-gap-detect.sh, emits
    classification + task plan
P4: Return task plan JSON to caller
```

## Stage 0 constraints

- **Read-only**. The only thing this skill writes is its own workspace.
  No repo edits, no image pulls, no Dockerfile touches, no A3 actions
  that bind NPUs.
- **No answer-key docs**. OL-03 denylist covers
  `docs/easyr1-dep-chain-audit.md` specifically — that's the reference
  classification history, not a cheat sheet; the agent must reproduce
  the classification from rules not from the doc.

## Invariants the skill enforces

- **G1** (universal): orchestrator (you) must NOT write outside
  session workspace. G1 is lenient here because the agent is pure
  read-only.
- **G2** (universal, trivial): no .py edits → py_compile N/A; dry-import
  of `verl` still happens at Stop hook as a sanity check that consumer
  repo is healthy.
- **G3** (expert-specific): any "no blockers" / "P1" / "D=0" claim must
  cite the exact `dep-gap-report.md` path + grep'd line evidence per
  dep classification.

## Return payload to caller

See README.md §"Deliverable" for the JSON shape.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/ALWAYS_LOADED_RULES.md` — OL-03/08 (this expert's locals)
- `../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` — cross-expert OLs
- `references/NPU_ECOSYSTEM_MAP.md` — the classification rule table
