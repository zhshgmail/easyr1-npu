---
name: vllm-ascend-port
description: >
  Day-0 NPU probe for vllm-ascend against a deeper upstream move (e.g.
  new torch that vllm-ascend's C++ extension was not rebuilt for, or
  new community vllm that dropped a symbol vllm-ascend imports). Runs
  ON TOP of a deployed torch-day0 / transformers-day0 base image.
  Produces validated patches to vllm-ascend on an ascend-day0 branch.

  Usage: /vllm-ascend-day0 --target-delta <symbol-or-desc>
                           --base-image <torch-day0-deployed-tag>
argument-hint: >
  target-delta: what moved (e.g. "torch==2.11", "vllm==0.20.0", "C++ ABI drift")
  base-image: torch-day0 or transformers-day0 deployed image
context: inline
---

# /vllm-ascend-day0 — patch vllm-ascend for new upstream ABI/API

## Your role (orchestrator)

Spawn `vllm-ascend-day0-worker`, wait, read Handoff, propagate {outcome,
patched branch, overlay image tag, PR material} back to caller. This
skill is specifically for **patching vllm-ascend itself** on a
Huawei-owned upstream; it's invoked AFTER torch-day0 / transformers-day0
has produced a usable base image.

## Two modes — decide at P0

**Mode Single-SHA** (original): user brings a specific failure + target
SHA. Reproduce failure, classify family, patch. Steps P0..P6 below
describe this mode.

**Mode Sweep** (added 2026-04-24 after cold-drive caught the gap):
user wants "adapt vllm-ascend to vllm main" with no specific failure
in hand. Iterate over the commit range from the latest-adapted vllm
version (say `v0.20.0`) to the target (say `origin/main`), running
`kb_drive_test.py` on each commit. Collect the union of drifts
impacting vllm-ascend. Each discovered drift then flows into P3
(design fix) onward. See "Sweep driver" below.

## Workflow

```
P0  parse args (target delta, base image from previous Day-0 session);
    pick Mode Single-SHA or Mode Sweep
P0.5 (Mode Sweep only) iterate kb_drive_test.py over commit range,
     aggregate impactful drifts, de-duplicate, feed into P1
P1  analysis: reproduce failure minimally, classify root cause
    (API-level drift vs C++ ABI drift vs schema mismatch vs other);
    identify which call sites in vllm-ascend are affected. For F1/F2
    drifts, kb_drive_test already did most of this — its `summary.json`
    is the P1 output.
P2  probe upstream vllm-ascend main: has the fix already landed?
    If yes, switch target or reproduce on an explicit commit
P3  design fix at minimum-invasive level: python-side workaround if
    possible, env-var guard, source patch if needed. For F1/F2/F2-path-move
    drifts, write a compat shim at `vllm_ascend/compat/<symbol>.py`
    per the KB template; for C++ ABI or other categories, read
    `patterns/domains/vllm-api-drift.md §F{N}` for the relevant family's
    fix shape.
P4  apply patch on `<target-version>_auto_porting` branch of
    upstream/vllm-ascend/ (e.g. `vllm-main_auto_porting`); smoke test
    via `/drift-port-validate` (F1/F2) + V1.3 rollout (runtime gate)
P5  Phase 2.5 deploy artifacts per shared pattern
P6  handoff: patched branch + overlay image + ONBOARDING + PR material
```

## Sweep driver

The cold-drive LLM should use this protocol when no specific failure
is given:

1. Identify the commit range: `git log --format="%h" <last-adapted-ref>..<target-ref>`
   in the community vllm repo. Example: `v0.20.0..origin/main` yields
   ~150 commits; scan is under 5 minutes.
2. For each SHA:
   ```bash
   python3 scripts/kb_drive_test.py \
     --vllm-ref $SHA \
     --vllm-path /path/to/vllm \
     --vllm-ascend-path /path/to/vllm-ascend \
     --kb-dir references \
     --out /tmp/sweep/$SHA
   ```
   Discard scans whose `summary.json` has `impact_ascend == 0`.
3. Collect the impactful scans; dedupe by `(kind, symbol)` so a single
   symbol rediscovered across commits isn't counted twice.
4. Cross-check each finding against `references/KB_INDEX.md §"Concrete
   case registry"` — if the symbol already has a row, the fix is
   already landed (or at least codified); skip.
5. Novel findings go into P3. Each novel finding is one commit on the
   fork branch.

A ready-made sweep wrapper is a TODO; today you run the loop in bash.

## Scanner coverage (2026-04-24 late)

Five scanner tools, each targeting a different drift family:

| Tool | Mode | Covers |
|---|---|---|
| `scripts/kb_drive_test.py` | per-commit | F1, F2-rename (limited), F3, F5-suspect |
| `scripts/sweep.sh` | tag-range wrapper over `kb_drive_test` | same as above, deduped |
| `scripts/check_f4.py` | tag-range | F4 return-type migration |
| `scripts/check_f7_f8.py` | tag-range, AST-based | F7 new attr, F8 new method |
| — | — | F6 (kv_cache runtime contract) still manual |

A complete Mode Sweep run looks like:

```bash
# F1 / F2-rename / F3 / F5-suspect across 156 commits
scripts/sweep.sh --commit-range v0.20.0..origin/main \
  --vllm-path /path/to/vllm \
  --vllm-ascend-path /path/to/vllm-ascend

# F4 (return-type) end-to-end tag diff
scripts/check_f4.py --vllm-path ... --vllm-ascend-path ... \
  --baseline-tag v0.20.0 --target-tag origin/main

# F7 / F8 (class-API additions) end-to-end tag diff
scripts/check_f7_f8.py --vllm-path ... --vllm-ascend-path ... \
  --baseline-tag v0.20.0 --target-tag origin/main
```

`kb_drive_test.py` emits a F1 `new_home_candidates` field: when a
symbol is removed but still exists at another path in target, the
scanner git-greps for `class <Sym>` / `def <Sym>` and lists up to 5
candidate new homes — so the fix author can pick the correct new
import. Empty = F1 real-removal (write local class body in compat);
non-empty = F2-path-move (try/except import).

## Stage 0 constraints

- **Edit scope strictly vllm-ascend only** (Huawei open source). Never
  edit community vllm, community torch, community transformers. Those
  are C-report territory.
- Env-var / python-level workarounds preferred over C++ changes. C++
  changes typically trigger rebuilds of the .so which is a slow
  development loop. Many "Day-0 on new torch" bugs can be fixed by
  detecting the ABI mismatch in Python and routing around it.
- **The `vllm_is_batch_invariant()` gate** (vllm's own mechanism) is
  an invaluable escape hatch because vllm-ascend already gates many
  call sites on it for OTHER reasons. Setting `VLLM_BATCH_INVARIANT=1`
  before vllm's module-import-time cache can bypass every gated call
  site at once.
- Patches land on a branch in `upstream/vllm-ascend/`, not on image
  files. Overlay image consumes the patch via git clone or COPY.

## Invariants

- G1: orchestrator never edits vllm-ascend or any upstream. Worker only
  on `ascend-day0-<delta>-<SESSION>` branch.
- G2: patch must be validated end-to-end via V1.3 rollout smoke marker
  match AFTER applying the patch + rebuilding overlay.
- G3: PR material must include a minimal reproducer + before/after
  behavior table.

## Outcome classification

| Outcome | Meaning | Action |
|---|---|---|
| A | No patch needed — existing vllm-ascend handles the delta already | Ship notes + overlay image; update KB |
| B | Workaround in consumer config (env var, CLI flag) without code patch | Document in ONBOARDING + add to KB |
| C-patch | vllm-ascend source change needed; we can make it | Patch on ascend-day0-<delta>-<SESSION> branch, smoke PASS, PR material |
| C-report | Fix belongs to community vllm / community torch / etc. | Blocker report |

**Goal is C-patch + smoke PASS**. Our value is producing validated,
PR-ready patches for the vllm-ascend team.

## See also

- `README.md`, `agent.md`, `state_machine.yaml`
- `references/ALWAYS_LOADED_RULES.md` — OL-03 / OL-08 specific to this expert
- `references/KB_INDEX.md` — known vllm-ascend Day-0 patterns + 2026-04-23 wet-run findings
- `references/patterns/domains/vllm-ascend-probe.md` — reproducer minimization, call-site location, fix-level selection
- `../torch-npu/port-expert/` — sibling whose deploy output is this expert's base image
- `../../_shared/references/patterns/domains/day0-deploy-artifacts.md` — Phase E deliverables
