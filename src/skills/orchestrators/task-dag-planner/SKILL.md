---
name: task-dag-planner
description: >
  Generic autonomous orchestrator for the loop the owner keeps asking for:
  goal analysis → subtask decomposition → dependency-DAG planning → staged
  topological execution. Given a free-text goal it:
  1) Phase A — analyzes the goal against ROADMAP.md + porting-lesson KB +
     auto-memory for prior art, and loads ANTI_PRESSURE_PROTOCOLS.
  2) Phase B — decomposes into subtasks, classifying each
     (reversible/destructive · needs-NPU/local · KB-lookup-first).
  3) Phase C — builds a dependency DAG, emits `task_dag.json`, and registers
     it as CC Task nodes with blocks/blockedBy edges.
  4) Phase D — executes in topological order: dep-free siblings fan out via
     the CC Workflow tool (companion `dag.workflow.js`), serial nodes run
     inline, every node's claimed output is verified on disk before it is
     marked complete. No human interaction between stages; exits only on
     all-done / blocked-on-destructive / blocked-on-external.

  Usage: /task-dag-planner <free-text goal> [--dry-run]
argument-hint: >
  goal: free-text objective in plain language (required), e.g.
        "port the V4 training-side sparse-MLA op to A3 and validate fwd+bwd"
  --dry-run: stop after Phase C — emit task_dag.json + the CC Task graph,
             execute nothing. Use for "what would this take / is the DAG
             right?" without consuming A3 time or doing destructive work.
context: inline
---

# /task-dag-planner — generic task→DAG→staged-execution orchestrator

## Your role (as this skill's runtime)

You are the orchestrator. You own decomposition, dependency planning, and
staged dispatch — you do NOT do the leaf work directly. Leaf work runs in
spawned workers (via the Workflow tool) or, for cheap serial nodes, inline.
Your job is to keep the DAG honest: every node's claimed artifact is
re-verified on disk before the node flips to `completed`.

Operating rules that bind this skill (project-level, non-negotiable):

- **Questions go to Discord, never console.** Any decision you must surface
  (a destructive node, an ambiguous goal, an external blocker) is sent to
  Discord chat_id `1494825170399924366` as a numbered list. The CC console
  transcript never reaches the owner — no AskUserQuestion, no ExitPlanMode
  picker. (Memory: `feedback_no_console_only_questions.md`.)
- **Project docs default to Chinese**, code / commit messages / SKILL.md
  frontmatter stay English. (Memory: `project_docs_language.md`.)
- **Commit messages carry no agent signature.**
- **Reduced-layer (1–2 layer) is the verified basis for V4**, never claim
  full 43-layer. Any V4-touching node's success criterion is stated at the
  layer it was actually run. (See `docs/_meta/DSV4_NPU_PORTING_REPORT.md`.)

## Required reading (Phase A, load once)

- `src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md` — load in full;
  re-cite the relevant `Px` at every high-leverage moment below.
- `docs/_meta/ROADMAP.md` — single source of truth for open work; the goal
  almost always maps onto an existing row (don't invent parallel backlog).
- `docs/_meta/kb/porting_lessons/index.md` — grep the keyword table for
  prior art before decomposing; a subtask that matches a cookbook ID
  inherits that cookbook's known pits + correction.
- Relevant `~/.claude/projects/.../memory/*.md` named by the goal — at
  minimum `end_to_end_vs_described.md` (binds Phase D honesty gate).

High-leverage moments where you MUST re-cite a `Px` (per the protocols'
load contract):

| Moment | Cite |
|---|---|
| Spawning a worker / fanning out a DAG layer | P3 + P8 |
| Marking any node `completed` | P1 + P5 + P7 |
| About to skip a node's on-disk verify | P1 |
| About to `nohup` / `&` / bypass Workflow with a raw Agent | P8 |
| About to inline a workaround instead of the planned fix | P6 + P9 |
| Declaring a node "blocked, upstream limitation" | P5 |

## Phase A — analyze the goal

1. Parse the free-text goal and `--dry-run`. Derive
   `SESSION_TAG = tdp-$(date -u +%Y%m%d-%H%M)`.
2. Create `workspace/task-dag-<SESSION_TAG>/` and write `PROGRESS.md` with the
   verbatim goal + parsed flags.
3. Read ROADMAP.md, the porting-lessons index, and the named memory entries.
   Record in PROGRESS.md: (a) which ROADMAP row(s) the goal maps to, (b) any
   cookbook IDs that match (these become per-node "KB-lookup-first" tags),
   (c) prior-art artifacts already on disk that a node can reuse instead of
   rebuilding.
4. If the goal is ambiguous enough that decomposition would be guesswork,
   surface 2–3 numbered interpretations to Discord and stop. Do NOT guess a
   decomposition and silently proceed.

## Phase B — decompose into subtasks

Break the goal into the smallest subtasks that each produce a single
verifiable artifact (a file, a tag, a passing smoke, a committed patch).
Classify every subtask on three axes — the classification drives Phase D
dispatch and the exit conditions:

| Axis | Values | Consequence |
|---|---|---|
| Reversibility | `reversible` / `destructive` | destructive nodes never auto-run — they pause for Discord OK (see Exit conditions) |
| Locale | `local` / `needs-NPU` | `needs-NPU` nodes precheck chip occupancy first (Memory: `a3_chip_economy.md`) and use the minimum chip count that validates the target |
| KB | `kb-lookup-first` (+ cookbook ID) / `novel` | `kb-lookup-first` nodes must Read the cited cookbook before doing work; inherit its known pits |

Destructive examples (require Discord OK): `docker rm` / `container prune`,
force-push, deleting forensic artifacts, overwriting a protected/tagged
result (e.g. `PROTECTED_flash_attention_npu_RESULT.md`), pushing a PR.

Honesty rule carried from the V4 work: when a node is "swap CUDA/TileLang op
for an NPU equivalent", default the success criterion to **CANN-native
first** (`npu_nsa_select_attention`, `npu_sparse_lightning_indexer_grad_kl_loss`,
`npu_nsa_compress_attention`, `npu_mla_prolog_v3`, `npu_rms_norm` were all
mapped on A3 — don't decompose them into "write an AscendC kernel"). Only
classify a node as op-gen when CANN / aclnn / torch-dispatch coverage has
been checked and is genuinely absent (the V4 cases were just hash-coding
sinkhorn + act_quant). (Memory:
`feedback_cann_has_basic_ops_dont_hand_gen.md`,
`project_v4_ops_cann_native_mapping.md`.)

## Phase C — build the dependency DAG

1. For each ordered pair of subtasks decide whether one's artifact is an
   input to the other → directed edge `A → B` (A blocks B). Siblings with no
   path between them are parallel-eligible.
2. Emit `workspace/task-dag-<SESSION_TAG>/task_dag.json`:

   ```json
   {
     "session_tag": "tdp-<...>",
     "goal": "<verbatim>",
     "nodes": [
       {
         "id": "n1",
         "title": "<imperative>",
         "reversibility": "reversible|destructive",
         "locale": "local|needs-NPU",
         "kb": "kb-lookup-first:<cookbook-id>|novel",
         "artifact": "<abs path / tag / smoke-name this node must produce>",
         "verify": "<exact on-disk check that proves the artifact>",
         "blockedBy": ["n0", "..."]
       }
     ]
   }
   ```

3. Register the graph as CC Tasks: one `TaskCreate` per node, then
   `TaskUpdate` to set `blocks` / `blockedBy` edges so the live task graph
   mirrors `task_dag.json`. The JSON is the durable artifact (survives
   compaction); the Task graph is the runtime view.
4. **`--dry-run` exits here.** Report the DAG (node count, the topological
   layers, the destructive/needs-NPU node list) to Discord and stop. Nothing
   is executed.

## Phase D — staged topological execution

Walk the DAG in topological order, one layer at a time. A "layer" = all
nodes whose `blockedBy` are already `completed`.

For each layer:

1. **Destructive nodes in the layer never auto-run.** Surface them to
   Discord as a numbered list with their `verify` criterion and wait. Skip
   the rest of the layer's destructive nodes; run the reversible ones.
2. **Parallel-eligible reversible nodes** (≥2 in the layer, no edge between
   them) fan out via the CC **Workflow tool**, referencing the companion
   script `src/skills/orchestrators/task-dag-planner/dag.workflow.js`. Pass
   the layer's nodes as the workflow's `args.task_dag.nodes` (each with its
   `id`, `prompt` built from `title`+`artifact`+`verify`+`kb`, and `deps`).
   (Cite P3 + P8 before spawning — use the Workflow, do not hand-roll raw
   `Agent` / `nohup`.)
3. **Serial / singleton nodes** run inline in this runtime.
4. **`needs-NPU` nodes** precheck chip occupancy before doing anything and
   claim the minimum chips that validate the target.

### Honesty gate (mandatory before any node → `completed`)

A worker returning "PASS" is NOT sufficient. Before flipping a node to
`completed` and unblocking its dependents you MUST:

- Read the node's claimed `artifact` from disk and run its `verify` check;
  cite the concrete evidence (a `log_path` + a numeric/diff value, a `ls` of
  the tag, a re-run smoke result). A "PASS" string with no on-disk evidence
  → node stays `in_progress`, re-dispatch. (Cite P1 + P7.) The companion
  `dag.workflow.js` already runs a per-node adversarial verifier and demotes
  any node whose claim fails review — trust that demotion, do not override it.
- **Never mark a node `completed` on a cheaper intermediate.** If the node's
  artifact is "1-layer full training iteration (fwd+bwd+optimizer)" do not
  accept "1-layer forward only" or a synth-delta stand-in. End-to-end means
  end-to-end. (Memory: `end_to_end_vs_described.md`; this is the exact sin
  the protocol exists to prevent.)
- For V4 nodes, the `verify` criterion is stated at the layer actually run
  (1–2 layer reduced basis), and the node's report says so — it must not
  read as a full-43-layer claim. (Memory:
  `poc_gibberish_is_expected_for_reduced_fab.md`,
  `v4_npu_path_exists_workaround_table.md`.)

After each node completes, update `task_dag.json` (node status + evidence
ref) and the CC Task, append PROGRESS.md, and push a coarse Discord update
at layer boundaries (no >15-minute silence; Memory:
`discord_reporting_cadence.md`).

## Exit conditions

| Condition | Action |
|---|---|
| **all-done** — every node `completed` with verified evidence | Write `RESULTS.md` (DAG outcome + per-node evidence table + reduced-layer caveats where they apply); send final Discord reply (new message, so the owner's device pings). |
| **blocked-on-destructive** — a destructive node is the next required step | Surface the node + its `verify` to Discord as a numbered list; pause that branch; keep executing any independent reversible branches; do not auto-run, do not invent a non-destructive substitute. |
| **blocked-on-external** — a node depends on an external party (upstream PR merge, maintainer review, a chip freed by someone else, a CANN release) | Mark the node `blocked`, record the precise external dependency in `task_dag.json` + ROADMAP, surface to Discord, continue independent branches. |

Do NOT stop to ask "should I do (a)/(b)/(c)?" when the DAG already orders the
work — drive the next eligible node. The only legitimate stops are the three
exit conditions above. (Memory: `feedback_run_the_whole_loop_no_asking.md`,
`no_premature_stop_asking.md`.)

## Invariants

- **Planner is stateless across compaction**: every decision is recoverable
  from `task_dag.json` + PROGRESS.md, never from chat memory. On wake-up,
  re-read `task_dag.json` before resuming Phase D.
- **The orchestrator never edits leaf artifacts directly** — if a layer
  seems to need a manual tweak the orchestrator itself makes, that's a node
  the DAG is missing; add the node, don't paper over it.
- **One node = one verifiable artifact.** A node with no `verify` check is
  not a node — it's a wish; refine it in Phase B.

## See also

- `dag.workflow.js` — companion CC Workflow script that fans out a DAG layer
- `../npu-port/SKILL.md` — sibling cross-expert orchestrator (house style)
- `../../../../docs/_meta/kb/porting_lessons/index.md` — cookbook keyword table
- `../../_shared/references/ANTI_PRESSURE_PROTOCOLS.md` — P1..P9
- `../../_shared/references/OPERATIONAL_KNOWLEDGE.md` — OL-* keyword index
