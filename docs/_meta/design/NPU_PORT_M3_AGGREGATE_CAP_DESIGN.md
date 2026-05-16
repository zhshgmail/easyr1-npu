# `/npu-port` orchestrator — M3 aggregate-cap design (P0r)

> **Status**: design only (P0r). Implementation deferred until `/npu-port` enters real cold-drive use with sub-skill spawn. Currently the orchestrator is documented but never produced a real spawn chain in production.
>
> **Trigger**: a5_ops adversarial-audit rule **M3 — per-X cap must have aggregate-X cap**. Single per-agent-type iter_cap is bypassed by stacking multiple agent types (a5_ops `10_LayerNorm` had 41 spawn events vs per-type cap=9). Cold-start episode reset wipes counts entirely.
>
> Our `/npu-port` will eventually spawn:
> - 1 `dep-analysis-worker`
> - N `<upstream>-upgrade-worker` (one per upstream in task_plan, N up to 5)
> - 1 `easyr1-port-worker`
>
> If each has its own per-type retry budget, attacker / runaway worker stacks them.

---

## 1. Threat model

Two attack/failure shapes:

**M3.a (intra-session aggregate)**: a `/npu-port` invocation legitimately spawns 6-7 workers. If retries-per-agent-type allows e.g. 3 retries each, total retries can balloon to 18-21 in one session. A reward-hacking worker that wraps the orchestrator could exploit this.

**M3.b (cross-session aggregate)**: cold-start re-invocation of `/npu-port` resets all counts; same op can be re-attempted indefinitely via cold-start churn. a5_ops `10_LayerNorm` hit this — 41 spawn events because each cold-start reset the counter.

---

## 2. Design

### 2.1 Aggregate cap definition

```python
# src/scripts/safety/npu_port_caps.py — single source of truth

NPU_PORT_AGGREGATE_CAPS = {
    # Per-invocation caps (this orchestrator session)
    "max_total_workers_per_invocation": 8,    # dep + 5 upstream + 1 consumer + 1 retry headroom
    "max_total_retries_per_invocation": 4,    # across ALL workers combined
    "max_runtime_seconds_per_invocation": 7200,  # 2h wall clock

    # Cross-invocation caps (this consumer-ref + candidate-image triple)
    # Survives cold-start by persisting to state.
    "max_invocations_per_target": 3,          # 1 initial + 2 retries; beyond → user gate
    "max_total_workers_per_target": 16,       # aggregate across all invocations
}
```

### 2.2 State persistence

```
workspace/npu-port-<consumer-ref>-<candidate-image>/
├── state.json                 # invocation count + worker totals across all calls
├── invocation-<N>/            # per-invocation
│   ├── PROGRESS.md
│   ├── worker_log.jsonl       # one row per worker spawn (type, started, ended, retries)
│   └── final_status.json
```

`state.json` schema:

```json
{
  "consumer_ref": "<sha>",
  "candidate_image": "<tag>",
  "first_seen": "<ISO>",
  "invocations": [
    {"n": 1, "started": "<ISO>", "ended": "<ISO>",
     "total_workers_spawned": 6, "total_retries": 2, "status": "stuck"},
    {"n": 2, "started": "<ISO>", ...}
  ],
  "aggregate_workers_to_date": 12,
  "aggregate_retries_to_date": 4
}
```

State is **persisted at `workspace/npu-port-<target>/` — NOT inside the per-invocation dir** — so cold-start re-invocation reads existing count and refuses if cap exceeded.

### 2.3 Cap enforcement points

| Cap | Where checked | Action when exceeded |
|---|---|---|
| `max_total_workers_per_invocation` | before each `Agent(subagent_type=...)` spawn | abort invocation with `INFRA_TRANSIENT_RETRY_EXHAUSTED`-equivalent terminal |
| `max_total_retries_per_invocation` | before each retry of any worker | same |
| `max_runtime_seconds_per_invocation` | wall-clock monitor in orchestrator loop | same |
| `max_invocations_per_target` | at orchestrator entry, after reading state.json | refuse to proceed; emit "user-gate required: target already attempted N times" |
| `max_total_workers_per_target` | at orchestrator entry + before each spawn | same |

### 2.4 User-gate semantics

When a per-target cap is hit, orchestrator does NOT auto-retry. It writes `state.json.user_gate_required: true` and emits:

```
USER GATE REQUIRED: <consumer-ref>+<candidate-image> already attempted
3 times (cap=max_invocations_per_target=3), 16 total workers (cap=16).
This target appears to be unrecoverable under current orchestrator
logic. Manual escalation needed:
  - Did you mean to retry yet again? Set
    EASYR1_NPU_PORT_FORCE_RETRY=1 and rerun.
  - Or escalate: pick a different candidate-image, file an upstream
    issue, or split the target into separate /<upstream>-day0 calls.
```

This forces explicit operator decision rather than runaway loop.

### 2.5 worker_log.jsonl schema

Each worker spawn writes one line:

```json
{
  "ts": "<ISO>",
  "worker_type": "<dep-analysis | vllm-ascend-day0-worker | ...>",
  "spawn_id": "<uuid>",
  "retry_n": 0,
  "started": "<ISO>",
  "ended": "<ISO>",
  "status": "<done | stuck | review-fail>",
  "handoff_path": "<workspace>/handoff.json",
  "next_action_in_orchestrator": "<continue | retry | abort>"
}
```

worker_log.jsonl is **append-only** (audit trail). Orchestrator on retry reads back the last entries for that worker_type to honor per-worker retry sub-budget.

### 2.6 a5_ops anti-pattern explicitly forbidden

> "Per-agent-type retry caps sum to more than max_total_retries_per_invocation" is a configuration bug.

Sanity-suite check (when this design lands as code) will lint:

```python
def test_npu_port_cap_consistency():
    """Sum of per-worker-type caps must be <= max_total_retries_per_invocation."""
    # ...
```

---

## 3. What this does NOT solve

- **`WORKER-GATE-ENUMERATION`** still open at a5_ops too (DEBT-7). Caps reduce blast radius but don't prevent the worker from reading orchestrator source.
- **Bypass via direct Agent(subagent_type=...) call** (P8) — orchestrator cap doesn't apply if caller invokes Agent directly. Full enforcement needs P0e workflow critic at PreToolUse hook (DEBT-5).

---

## 4. Implementation order

1. Build `state.json` reader + writer (idempotent, tolerates partial files)
2. Add cap checks inside `/npu-port` orchestrator before each `Agent(...)`
3. Add `EASYR1_NPU_PORT_FORCE_RETRY` override + user-gate emit logic
4. Add `worker_log.jsonl` append on each spawn
5. Sanity-suite test: simulate workspace with 17 worker entries → orchestrator refuses to start invocation 4

**NOT in scope of P0r**: actual implementation. P0r is the design doc; code lands when `/npu-port` enters real cold-drive use and the cap matters.

---

## 5. Triggers to implement (when this design becomes code)

- First real `/npu-port` invocation that genuinely spawns sub-skills (not a mock)
- First incident where worker retry chain runs unchecked
- a5_ops M3 fix lands on their side (sibling project will demonstrate failure mode if not adopted)

When any of these fires, prioritize P0r → real impl (estimated ~4-6h).

---

## 见也

- ROADMAP §2 P0r
- a5_ops adversarial-audit M3: [ADVERSARIAL_REWARD_HACKING_AUDIT.md](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/design/ADVERSARIAL_REWARD_HACKING_AUDIT.md)
- TEST_SAFETY_FEEDBACK_DESIGN.md v4 §11 (M3 deferral rationale)
- src/skills/orchestrators/npu-port/SKILL.md (orchestrator definition)
