# dep-analysis-expert

**Product**: a read-only analyzer. Given a consumer's target commit and a
candidate NPU base image, classify every runtime dep into A/B/C/D/E and
emit a task DAG the orchestrator can act on:

- All A/B/C/E → P1 (no new NPU work) → hand off to the consumer's
  port-expert (e.g. easyr1-expert) with `--reuse-image` of the current
  NPU stack
- Any D → P2 → spawn the relevant upgrade-expert (e.g.
  transformers-upgrade-expert) to lift the stack, then P1 against the
  new image

## When to use

- `/npu-port` orchestrator's first step: "given consumer commit X, what
  does the task decomposition look like?"
- Manually: before starting a port, "is there any blocker on this image?"

## When NOT to use

- Already-known-D-zero scenarios (e.g. well-exercised consumer commits
  on a well-exercised image). The expert still runs cheap (no A3
  operations, no docker build), but skipping is fine when the orchestrator
  has a recent cached result.
- Deciding HOW to port. This expert only emits the task list — the
  port itself is other experts' domain.

## Scope / boundaries

**In scope**:
- Pull consumer's `requirements.txt` at the target ref
- Inspect the candidate NPU image (`pip freeze` via `docker run` or
  cached `knowledge/images/<slug>.md`)
- Run `scripts/dep-gap-detect.sh` → produce a structured dep-gap-report.md
- Classify each dep A/B/C/D/E; cite reasoning from `references/NPU_ECOSYSTEM_MAP.md`
- Emit task plan JSON describing what the orchestrator should spawn next

**Out of scope**:
- Any file write outside `workspace/dep-analysis-<SESSION_TAG>/`
- Any `docker pull` / `docker build` / A3 network action for real (we
  only read pip-freeze; image pulls belong to upgrade-expert's P1)
- Any code editing anywhere (pure analysis)
- Numerics / smoke validation (upgrade-expert + port-expert's territory)

## Inputs (orchestrator passes via env)

| Var | Meaning |
|---|---|
| `SESSION_TAG` | e.g. `depan-20260501-0830` |
| `CONSUMER_REPO` | path to the consumer checkout (default: `upstream/EasyR1`) |
| `CONSUMER_REF` | target commit / branch (required) |
| `CANDIDATE_IMAGE` | NPU base image to evaluate against (default: v1 verl-8.5.0-a3) |
| `REQS_FILE` | path to requirements.txt inside consumer (default: `requirements.txt`) |

## Deliverable (structured payload returned to caller)

```json
{
  "consumer_ref": "<sha>",
  "candidate_image": "<tag>",
  "classification": {
    "A": ["peft", ...],
    "B": ["vllm -> vllm-ascend", ...],
    "C": ["flash-attn (shim)", ...],
    "D": [],
    "E": ["numpy", "pandas", ...]
  },
  "scenario": "P1" | "P2",
  "task_plan": [
    {"step": 1, "expert": "transformers-upgrade", "input": {...}, "if_needed": "P2"},
    {"step": 2, "expert": "easyr1", "input": {...}, "depends_on": [1]}
  ],
  "artifacts": {
    "dep_gap_report": "workspace/dep-analysis-<SESSION_TAG>/dep-gap-report.md",
    "progress": "workspace/dep-analysis-<SESSION_TAG>/PROGRESS.md"
  },
  "provenance": {"produced_by": "dep-analysis-worker"}
}
```

`scenario=P1` → orchestrator proceeds directly to consumer port-expert.
`scenario=P2` → orchestrator spawns each upgrade-expert in `task_plan`
order, then consumer port-expert with `--reuse-image` of the final
upgraded image.

## How it relates to the broader harness

- Uses `_shared/references/ALWAYS_LOADED_UNIVERSAL.md` (universal OLs).
  Most OLs are trivially satisfied: this expert doesn't write .py, doesn't
  build images, doesn't touch A3 chips. OL-01 (py_compile), OL-04 (unique
  image tag), OL-05 (chip precheck) all N/A. OL-02 (PASS with evidence),
  OL-09 (provenance) still apply.
- Its own ALWAYS_LOADED_RULES.md has OL-03 denylist (no classification
  answer-keys, specifically `docs/easyr1/easyr1-dep-chain-audit.md`) and OL-08
  (read-only; the ONLY writable path is `workspace/dep-analysis-.../`).
- Pinned to `_shared/` via `SHARED_VERSION.txt`.
