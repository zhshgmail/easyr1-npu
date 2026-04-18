# Codex proxy sign-off — v1 milestone

Date: 2026-04-18.
Context: user delegated final sign-off to the `codex-review` skill with the message "我没时间 sign off，需要的话可以让 codex review 代为 sign off."

Codex verdict is archived verbatim below. The three Small follow-ups were addressed in the same commit that landed this sign-off doc; see `docs/DELIVERABLE.md` §9.

---

## VERDICT: APPROVE WITH FOLLOW-UPS /s codex sign-off 2026-04-18

## RATIONALE

The v1 milestone clears the functional bar and is durable enough to hand off. The evidence trail shows V1.1 through V1.5 all passed on the actual A3 target, including the end-to-end GRPO smokes on 2 chips and 4 chips; the ported EasyR1 code is on `zhshgmail/EasyR1` branch `ascend-port` at 16 commits with the head at `72a7f22`; the harness repo is on `zhshgmail/easyr1-npu`; the image is rebuildable from `Dockerfile.npu`; and the reusable handoff assets are present on disk as shipped skills/scripts, not just described aspirationally. The knowledge catalog is also now in the shape I asked for earlier: `knowledge/npu-patterns.md` has 16 stable IDs with a uniform schema, and the skills design doc reflects shipped vs deferred status. I am not choosing plain APPROVE because the final top-level deliverable doc is still stale in a few sign-off-relevant places: it still says "10 stable IDs" even though the canonical catalog is 16, and it still carries unchecked pre-close checklist items. That is not a blocker to calling v1 complete, but it is enough that I want those gaps explicitly frozen as follow-up debt rather than silently accepted.

## FOLLOW-UP LIST

- `MINOR / S` — Update `docs/DELIVERABLE.md` so the executive summary matches the shipped state: replace "10 stable IDs" with 16, remove the stale "see §4 for status at time of read" wording, and close or delete the unchecked sign-off checklist at the end.
- `MINOR / S` — Tighten `skills/ray-npu-shim/SKILL.md` headline/positioning so it stays explicit that the shim solves Ray-specific integration only and still requires an `NPU-CP-001` source sweep in any second port.
- `MINOR / S` — Normalize the `npu-image-inspect` output contract against the examples in `knowledge/images/` so a future user can tell at a glance which sections are mandatory versus best-effort.

**All three items were addressed in the sign-off commit** — see the commit linked in the journal entry for 2026-04-18 sign-off.

## RESIDUAL RISKS THE USER SHOULD KNOW

The likely week-2 failures are operational, not architectural. The shared A3 host can cause false negatives through chip contention or disk pressure; the base image still depends on a vendor stack with one known repair step (`triton-ascend` reinstall) and a dev build of `vllm_ascend`; and several paths are intentionally out of v1 scope or only lightly exercised, especially LoRA, non-default loggers, long-context/padding-free/Ulysses, and multi-node scaling. None of those undermine the sign-off, but they are the first places I would expect surprises if someone uses the port beyond the validated smoke envelope.

## WHAT TO CHECK FIRST IF V1 BREAKS FOR A REPRODUCER

First verify they are actually running the intended artifacts and environment, not a drifted one. Check:

1. `git -C upstream/EasyR1 rev-parse HEAD` equals `72a7f22` (or a later `ascend-port` descendant).
2. The container was built from `Dockerfile.npu` (not the base image directly).
3. Run V1.1 smoke inside `run-npu-container.sh`: verifies `import torch_npu`, `torch.npu.is_available() == True`, `ASCEND_RT_VISIBLE_DEVICES` populated inside Ray actor, `VLLM_ASCEND_ENABLE_NZ=0` set, `triton/__init__.py` exists.

If that passes, the next checks are:

4. Chip occupancy via `npu-smi info -t proc-mem -i <card>`.
5. Stale bind-mount / `__pycache__` issues.
6. Only then climb to higher-level Ray or vllm rollout problems.
