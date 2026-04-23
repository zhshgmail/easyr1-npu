---
name: npu-port
description: >
  End-to-end autonomous NPU port orchestrator. Given a consumer repo +
  target ref + candidate NPU base image, runs:
  1) /dep-analysis → scenario P1 or P2 + task_plan
  2) If P2: run each upgrade-expert in task_plan order to produce a
     working upgraded image
  3) Run consumer port-expert (currently easyr1-expert) with --reuse-image
     of the final working image
  4) Read all experts' handoffs, verify claims against their workspaces,
     emit single RESULTS.md and closure classification. No human
     interaction between stages; exits only on done / stuck / @review-fail.

  Usage: /npu-port --consumer-ref <sha> --candidate-image <tag>
                   [--consumer-repo <path>] [--session-tag <tag>] [--dry-run]
argument-hint: >
  consumer-ref: target commit/branch of the consumer repo (required)
  candidate-image: starting NPU base image to evaluate (required)
  consumer-repo: default upstream/EasyR1
  session-tag: default auto-generated
  dry-run: if set, only runs /dep-analysis and stops
context: inline
---

# /npu-port — cross-expert port orchestrator

## Your role (as this skill's runtime)

You are the orchestrator. You do NOT touch code directly. You:

1. Parse args + derive SESSION_TAG (e.g. `npu-port-$(date -u +%Y%m%d-%H%M)`).
2. Create `workspace/npu-port-<SESSION_TAG>/` and record inputs in a
   top-level PROGRESS.md.
3. Spawn `dep-analysis-worker` via Agent(subagent_type=Explore), brief =
   `src/experts/dep-analysis-expert/agent.md`, env:
   - `DEP_ANALYSIS_EXPERT_ROOT=<repo>/src/experts/dep-analysis-expert`
   - `SESSION_TAG=depan-<SESSION_TAG>` (namespace isolation)
   - `CONSUMER_REPO`, `CONSUMER_REF`, `CANDIDATE_IMAGE`
   (re-use the same env vars that `/dep-analysis` uses, minus the
   `dep-analysis-` prefix; the expert's agent.md documents them).
4. Read returned `task-plan.json`. Based on `scenario`:
   - `P1` → skip to step 6 (consumer port)
   - `P2` → for each upgrade-expert in `task_plan`, spawn its worker in
     sequence, wait for its Handoff payload with `image_tag`, feed that
     as `CANDIDATE_IMAGE` to the next step
   - `unsupported` (D with no expert) → write RESULTS.md `@review-fail`
     with the unsupported dep list, exit
5. (If P2) spawn each `<dep>-upgrade-worker` via Agent. Read its Handoff;
   verify the image tag exists on A3 and the claimed numerics are cited;
   if it exited `stuck` → RESULTS.md `@review-fail`, surface, exit.
6. Spawn the consumer port-expert (today: `easyr1-port-worker` at
   `src/experts/easyr1-expert/agent.md`) with env:
   - `EASYR1_EXPERT_ROOT=<repo>/src/experts/easyr1-expert`
   - `SESSION_TAG=easyr1-<SESSION_TAG>`
   - `EASYR1_REF=<consumer-ref>`
   - `REUSE_IMAGE_TAG=<upgraded-image-tag-or-candidate>` (this triggers
     easyr1-expert's `--reuse-image` Phase C path, skipping rebuild)
7. Read port-expert's Handoff. Verify:
   - V1.1 + V1.3 + V1.4 all PASS
   - V1.4 step-1 entropy_loss in band
   - Cleanup field = clean OR partial with reason
   - Every artifact's produced_by ∈ {\*-worker, orchestrator, user-provided}
8. Write top-level RESULTS.md with:
   - Consolidated provenance table (each artifact across experts)
   - Classification: CLOSED / PARTIAL / FAILED
   - Cleanup summary (per-expert cleanup states)
   - Return tuple: `{consumer_branch, final_image_tag, v14_entropy_loss, session_tag}`
9. Send final status to user channel.

## Invariants (as the orchestrator itself)

- **Orchestrator is stateless**: every decision flows from read sub-expert
  Handoff payloads + re-verify against their workspace files. Do NOT
  trust an expert's "PASS" string without re-reading the claimed log /
  jsonl / report (NPU-OPS-010 anti-pattern applies at orchestrator level
  too).
- **Orchestrator never edits consumer code or Dockerfiles.** If a stage
  seems to need a manual tweak, that's a skill bug — exit `@review-fail`,
  surface, never patch.
- **Session isolation**: each expert gets its own SESSION_TAG prefix
  (e.g. `depan-<tag>`, `trans-upg-<tag>`, `easyr1-<tag>`) so workspaces
  and cleanup boundaries are clear.

## Dry-run semantics

`--dry-run`: only run step 3 (dep-analysis). Report task plan back to
user and exit. Useful for "what would this take?" without consuming any
A3 time.

## Failure modes

| Sub-result | Orchestrator action |
|---|---|
| dep-analysis `stuck` (unknown D) | RESULTS.md `@review-fail` + surface unsupported deps |
| upgrade-expert `stuck` | RESULTS.md `@review-fail` + surface platform-bug signature |
| port-expert `stuck` | RESULTS.md `@review-fail` + surface EC signature |
| any worker `@review-fail` (OL violation self-report) | RESULTS.md `@review-fail` + surface which OL |
| G3 re-verify fails (claimed entropy_loss not in band, or log missing) | RESULTS.md FAILED + surface discrepancy |
| all green | RESULTS.md CLOSED + return final tuple |

## See also

- `README.md` — product definition
- `../../experts/dep-analysis-expert/` — step 1 expert
- `../../experts/transformers/upgrade-expert/` — step 2 candidate
- `../../experts/easyr1-expert/` — step 3 consumer port
- `../../experts/_shared/` — universal OL rules + reference hooks/scripts
