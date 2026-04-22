---
name: easyr1-port
description: >
  Port EasyR1 (hiyouga/EasyR1) to Ascend 910C (A3) NPU end-to-end. One-shot
  invocation spawns easyr1-port-worker agent that: reads KB, runs code-path-
  sweep, applies the 5 archetype fixes (device dispatch / Ray shim / flash
  attention swap / vllm compat / Dockerfile), builds image, deploys to A3,
  runs smoke ladder V1.1 → V2.2, asserts numeric baseline per rung, and
  reports PASS/FAIL with log evidence. Stage 0 scope: D=0 (no new NPU
  adaptation dep), targets CANN 8.5.0 image.
  Usage: /easyr1-port reproduce v1  (re-run canonical port on v1 image)
         /easyr1-port new-commit <hash> v1  (port a specific EasyR1 commit)
         /easyr1-port --diag (verbose diag mode for debugging)
argument-hint: >
  mode: "reproduce" | "new-commit <easyr1-commit>"
  target: "v1" (= verl-8.5.0-a3) | "v2" (= verl-8.5.2-a3 drill) | full image tag
  optional: "--diag" for extra log output
context: inline
---

# /easyr1-port — EasyR1 → Ascend NPU orchestrator (Stage 0)

**Your role**: spawn `easyr1-port-worker` agent, read its artifacts, decide
done / retry. You do NOT edit port code yourself — that's the worker's job.

## Canonical workflow spec

**The authoritative definition of phases, invariants, and prohibited actions
lives in `src/experts/easyr1-expert/state_machine.yaml`** (machine-readable,
consumed by workflow critic at runtime). This SKILL.md is a prose companion.

## Workflow overview

```
Phase P0: Parse args + load .easyr1_env (ssh config, A3 host, chip defaults)
Phase P1: dep-gap-detect (must return D=0 for Stage 0)
Phase P2: image inventory (inspect target docker image)
Phase P3: upstream fetch (if not already present)
Phase P4: code-path sweep (scan EasyR1 for CUDA-only callsites)
Phase P5: spawn easyr1-port-worker → Phase A/B/C/D internally
          * fix-loop on smoke fail, up to N iters per rung
          * exits: done | handoff for future smoke-probe (Stage 1+)
Phase P6: smoke ladder verification (V1.1 → V2.2, log evidence per rung)
Phase P7: report + commit + append porting-journal
```

Full sequence + invariants G1-G3: see `state_machine.yaml`.

## Stage 0 constraints

This skill is specifically for **D=0 scenarios**. If P1 dep-gap-detect returns
D≥1, halt and escalate per `docs/P2-WORKFLOW.md` (future work: that scenario
spawns a dedicated dep expert, not handled by this skill at Stage 0).

## Invariants the skill enforces

- **G1**: Orchestrator (you) must NOT directly Edit files under
  `upstream/EasyR1/verl/` or `Dockerfile.npu*`. All port edits via agent.
- **G2**: Any code claimed "ready" must pass `static_check.py` (py_compile +
  dry-import). Enforced by agent's Stop hook.
- **G3**: Any "PASS" / "smoke PASS" / "V1.x green" claim must be backed by
  concrete log file path + entropy_loss numeric evidence. Enforced by
  Stop hook + this orchestrator's post-verify.

## See also

- `README.md` — product definition
- `state_machine.yaml` — authoritative workflow spec
- `agent.md` — easyr1-port-worker definition
- `references/ALWAYS_LOADED_RULES.md` — agent's mandatory first read
- `docs/design/SKILLS_ARCH_TARGET.md` — target architecture, Stage 0 scope
