# Smoke-ladder convention for NPU (and other accelerator) ports

Reusable naming + structure convention for stepwise bring-up smoke testing. Evolved through the EasyR1-on-A3 port (2026-04-17 → 2026-04-19). Writing it down so the next port starts with the same ladder rather than inventing new labels.

---

## Two orthogonal axes

**Milestone** (`v<N>`): a feature level of the port. Grows with scope.
- `v1` — "functional minimum" (it runs, it produces correct output on the most conservative settings).
- `v2` — "performance/feature unlock" (enable the features v1 explicitly deferred, e.g. `padding_free=True`, ulysses SP).
- `v3+` — further features / perf tuning (compile-mode, larger-scale multi-node, new model families, etc.).

**Smoke level** (`.<M>`): a progression of tests within a milestone. Each level is independently runnable and has a clear PASS/FAIL assertion, so a failure signal points at a specific layer.

Combined label: **`V<milestone>.<level>`**, e.g. `V1.3` = "v1 milestone, level 3 smoke". `V2.1` = "v2 milestone, level 1 smoke".

---

## The standard 5-level smoke ladder (per milestone)

Applicable to any accelerator-port project that ships training + inference code. Tailor the assertions for the project; the ladder shape is reusable.

| Level | Name | Assertion | Failure localizes to |
|---|---|---|---|
| `.1` | device accessor | Helper module (e.g. `verl/utils/device.py`) returns the correct backend string/module inside the target container. `is_npu_available() == True`, `get_device_name() == "npu"`, etc. | Device-probe layer — are we on the right hardware? Is the helper wired correctly? |
| `.2` | tensor round-trip + vendored helpers | A small tensor allocated on the accelerator computes correctly; any vendored helpers (e.g. `bert_padding`) produce the expected output vs reference. | Basic tensor ops + vendored pure-torch reimpls. |
| `.3` | rollout-only | Inference engine (e.g. `vllm_ascend`) loads a small model and generates coherent text. No training loop. | Inference engine compatibility with the framework. |
| `.4` | single-card training | One training step end-to-end on the minimum configuration (single accelerator or single-card + gradient accumulation). | Full training path: actor forward + backward + optimizer + data + reward. |
| `.5` | multi-card training | Same as `.4` but multi-card, testing the collective library (HCCL/NCCL). | Collective init, FSDP topology, cross-card sync. |

For the EasyR1-on-A3 port:
- `V1.1` = device accessor smoke.
- `V1.2` = tensor + vendored `bert_padding` helpers.
- `V1.3` = vllm_ascend rollout (Qwen2-0.5B).
- `V1.4` = GRPO 2-step, 2 chips, `padding_free=False`, `ulysses=1`.
- `V1.5` = GRPO 2-step, 4 chips (2 A3 cards).
- `V2.1` = GRPO 2-step, 2 chips, `padding_free=True` (v2 feature validation).
- `V2.2` (planned) = GRPO 2-step, 4 chips, `padding_free=True` + `ulysses=2` (v2 full feature).

---

## Why ladder, not "one big test"

- **Failure localization**: if V1.4 fails you know the issue is in train-path code. If V1.3 fails it's vllm-ascend. A monolithic "run real GRPO for 100 steps" test gives you a crash but not a layer to look at.
- **Independently runnable**: each level's script stands alone. You can re-run just V1.2 after changing the vendored helpers, without waiting for vllm to load a model.
- **Regression detection**: a new bug that shows up at V1.5 but not V1.4 points at multi-card / HCCL specifically.
- **Progressive time budget**: V1.1 is seconds, V1.2-V1.3 is minutes, V1.4-V1.5 is ~10 minutes each. Fail-fast at cheap levels before spending expensive cycles.
- **Shared v1 baseline, milestone-specific upper levels**: v2 doesn't re-run V1.1 (device accessor hasn't changed). V2.1 assumes V1.1 through V1.4 already passed under v1; v2 only needs to test what v2 changed.

---

## Script / experiment naming

- **Smoke script**: `examples/<model>_<task>_<algorithm>_<backend>_smoke_v<M>_<N>_<feature>.sh`
  - e.g. `qwen2_0_5b_math_grpo_npu_smoke_v2_1_padfree.sh` — Qwen2-0.5B, math dataset, GRPO algorithm, NPU backend, v2 milestone, level 1 of v2 smoke ladder, tests padding_free feature.
- **Trainer experiment name**: `v<M>_<N>_<model>_<task>_<feature>` — e.g. `v2_1_qwen2_0_5b_math_padfree`. Underscores because YAML/shell.
- **Log file**: `/tmp/$USER/logs/v<M>_<N>_<timestamp>.log`.
- **Checkpoint dir**: `/tmp/$USER/<project>_smoke_ckpt_v<M>_<N>/`.

The underscored form is a file-system-safe encoding of the dotted label. Both refer to the same thing; the dotted label appears in human-facing docs, the underscored form in file/experiment names.

---

## When to add a new smoke level

- Adding a capability that wasn't validated before (new chip count, new attention backend, new optimizer, new dataset modality) → add a new `.N`.
- Adding a feature that only works under specific conditions (`padding_free=True`, `ulysses_size>1`, LoRA, BF16→FP8 etc.) → new milestone (`V<N>.1` baseline, `V<N>.2+` extending).
- Debugging an observed regression → typically a one-off probe script, NOT a new smoke level (avoid ladder bloat).

Default ceiling: 5 levels per milestone. Past that, the feature probably deserves its own milestone.

---

## Pitfall: milestone-label overload

The first version of this catalog conflated milestone with level: V1.6, V1.7 were invented to describe v2 work while still on the v1 numbering. That produced the awkward claim "V1.6 validates v2 feature." Fixed in this revision: **v2 features get V2.x labels, full stop**.

Generalization: **when you introduce a new feature-milestone, reset the smoke level counter**. Keep milestone and level strictly separate. Historical references in dated journal entries can stay (they were correct for their timestamp).

---

## Related knowledge

- `repo/knowledge/npu-patterns.md` — the stable-ID catalog for NPU-specific findings. Smoke-ladder levels surface specific patterns; e.g. V1.4 typically surfaces `NPU-CP-001` (torch.cuda.* sweep), V2.1 surfaced `NPU-BUG-003` (triton-ascend inductor shape-sensitive crash).
- `repo/docs/skills-design.md` — the overall harness plan; this convention is a sub-element of how we organize the `npu-smoke-test` skill area.
- `repo/docs/porting-journal.md` — dated log of each smoke level being run and its outcome. Use the smoke label as a grep anchor (`grep -n '^V1\.4\|^V2\.1' journal`).
