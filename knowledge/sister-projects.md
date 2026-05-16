# Sister projects we track in `upstream/`

Sister harness projects whose **design + skills + KB** we borrow from
(or coordinate with). Living next to NPU community / Huawei upstreams
in the same `upstream/` tree so cross-references are stable.

> **Pull cadence**: explicit `git -C upstream/<name> pull` is required.
> Pinned-by-commit so we know when we last synced (avoids "their main
> moved while we were testing"). User direction 2026-05-16: don't auto-
> pull; we control sync timing.

## a5_ops (sister project — AscendC kernel-gen harness)

| Field | Value |
|---|---|
| Path | `upstream/a5_ops/` |
| Remote | `https://gitcode.com/zhengshencn_hwca/a5_ops` |
| Default branch | `main` |
| Last synced HEAD | `ef38ffde78b4575be8d6f60bdfbfb1bc38f63c36` (2026-05-16) |
| Last sync subject | `apply_adam_w_v2 ModelNew — resolve a5_capture.pt in .harness/ fallback` |

### Purpose (why we track them)

a5_ops is the **AscendC kernel-generation** harness (Ascend910C V220 / Ascend950PR V100). They've shipped:

- 30+ slash-command skills (`/ascendc-op-gen`, 16+ `aog-*` agents)
- `vendor/AscendOpGenAgent` submodule (proprietary AscendC gen pipeline)
- 50+ pytest test suite + workflow_critic.py (1971 LOC)
- 200+ stable IDs in KB (OL/EC/PB/P-P)
- 11-pattern porting-self-challenge (we forked the structure)
- Adversarial-audit framework (M1-M5 + P9 — we adopted)

We borrow:
- **Design patterns**: test net / safety net / feedback loop architecture (T29-T31)
- **Anti-pressure protocols**: P1-P9 (with their incident anchors)
- **Skill organization**: single user-entry + `aog-*` prefix for internals
- **KB structure**: cross-skill `_shared/references/` + per-skill `references/KB_INDEX.md`
- **Postmortem format**: their `SAFETY_NET_NAME_COUPLING_2026_05_14.md` structure

We do NOT use a5_ops for runtime — their skills generate kernels, not ports. The two projects share the philosophical layer (LLM-as-engineer discipline + mechanical safety nets), not the operational layer.

### How to sync

```bash
# Explicit sync (only when we want to absorb recent a5_ops work)
git -C ~/workspace/easyr1-npu/upstream/a5_ops pull --rebase

# Update this file's "Last synced HEAD" + "Last sync subject" lines
# Commit the updated knowledge/sister-projects.md
```

### What NOT to do

- ❌ Don't add a5_ops as a git submodule of our repo — it would pin commits
  into our git history; we want loose coupling.
- ❌ Don't auto-pull on every session — user direction is "control sync timing".
- ❌ Don't push to a5_ops; we're consumers, not contributors (their team
  owns the design direction).

## See also

- [`docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md`](../docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md) — borrowed test/safety/feedback architecture
- [`src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md`](../src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md) — P1-P9 porting
- [`docs/_meta/postmortem/2026-05-16_T30_design_v1_to_v4_mode_coupling.md`](../docs/_meta/postmortem/2026-05-16_T30_design_v1_to_v4_mode_coupling.md) — incident where we cargo-culted a5_ops literals
