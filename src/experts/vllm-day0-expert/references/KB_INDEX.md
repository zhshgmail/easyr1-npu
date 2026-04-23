# vllm-day0-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL (incl. OL-11 build-time import trap + OL-12 VLLM_BATCH_INVARIANT cache) | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | OL-03 + OL-08 + outcome matrix (A / A-with-note / B / C-patch / C-report) | Phase A |
| [patterns/domains/vllm-overlay-probe.md](patterns/domains/vllm-overlay-probe.md) | vllm-specific API probe protocol + known drift surfaces | Phase A + B |
| [patterns/domains/overlay-image.md](patterns/domains/overlay-image.md) | Dockerfile.overlay-vllm<MM> template | Phase C |
| [../../_shared/references/patterns/domains/day0-deploy-artifacts.md](../../_shared/references/patterns/domains/day0-deploy-artifacts.md) | 5 deploy deliverables mandatory for A / C-patch | Phase E |

## Outcome decision — C-patch vs C-report split (2026-04-23)

Previously "outcome C" was a single catch-all. After 2026-04-23 dusk
work, C splits cleanly:

| Outcome | Meaning | Our action |
|---|---|---|
| **A** | vllm-ascend plugin loads + all APIs stable + no consumer impact → V1.3/V1.4 PASS | Ship overlay + notes |
| **A-with-note** | PASS but an unused code path has a visible gap (e.g. gpt_oss MoE kernel missing) | Ship + note in ONBOARDING |
| **B** | Consumer-side shim needed (SamplingParams new RO property, LLM.generate sig change) | Commit shim on fixture + smoke PASS |
| **C-patch** | vllm-ascend itself (Huawei-owned) needs fix; we can ship it | Branch `ascend-day0-vllm<MM>-<SESSION>` on upstream/vllm-ascend/ + PR material. Escalate to `/vllm-ascend-day0` skill if the patch is deep |
| **C-report** | Fix lands in community vllm or other community upstream | blocker-report with reproducer + suggested fix; session ends |

**Goal is A or C-patch PASS**. C-report is last resort.

## Quick symptoms → classification

| Symptom after pip-overlay target vllm | Likely outcome |
|---|---|
| `vllm.platform_plugins` still finds `ascend` plugin + all probed APIs stable + consumer source clean | **A** |
| New kernel fails to import but ONLY for model types consumer doesn't use (e.g. gpt_oss MoE when consumer runs Qwen2-0.5B) | **A-with-note** |
| SamplingParams grew new RO property / LLM.generate sig shifted AND consumer uses it | **B** — introspection shim (EC-03 variant) |
| vllm-ascend's plugin fails to register with new vllm (registry API changed) | **C-patch** — route to `/vllm-ascend-day0` skill |
| New vllm imports a kernel/symbol from an older NPU dep (e.g. `triton.runtime.jit.constexpr_function` missing from triton-ascend) AND consumer path uses it | **C-patch** on triton-ascend or **C-report** to community |
| C++ extension in vllm-ascend SIGSEGV through dispatcher after new torch installed | **C-patch** on vllm-ascend (Fix B+ pattern); route to `/vllm-ascend-day0` skill |
| `ImportError: cannot import name 'X' from vllm.model_executor.*` where vllm dropped X | **C-patch** on vllm-ascend (forward-compat helper). **Check main tip first** — often already fixed upstream, then it's just a "base image stale" issue, outcome changes to A on refreshed image |

## 2026-04-23 baseline snapshot

- v1 NPU image: vllm 0.13.0+empty / vllm_ascend 0.13.1.dev18
- v2 NPU image: vllm 0.18.0+empty / vllm_ascend 0.17.0rc2.dev109
- community vllm latest: **0.19.1** (2026-04-18)
- vllm-ascend latest release: **0.18.0rc1** (2026-04-01; tracks vllm 0.18.x)

Orchestrator pre-probe of `pip install vllm==0.19.1 --no-deps` overlay on
v2 image (2026-04-23):
- vllm-ascend 0.17 plugin **still registers** against vllm 0.19.1 via
  `vllm.platform_plugins` group — plugin architecture is version-tolerant
- `lora.lora_model.LoRAModel` present; `parallel_state.get_tp_group`
  present
- `SamplingParams` RO properties: `{all_stop_token_ids, bad_words_token_ids,
  eos_token_id}` — same set as 0.18 (no drift)
- **NEW DAY-0 FINDING (pre-probe)**: `gpt_oss_triton_kernels_moe.py` in 0.19 imports
  `triton.runtime.jit.constexpr_function` — triton-ascend 3.2.0 doesn't
  export that. BUT only fires for gpt_oss MoE model type; EasyR1's
  Qwen2-0.5B path doesn't hit it.
- **WET-RUN FINDING (2026-04-23 follow-up, outcome C)**: V1.3 LLM(...)
  construction raises `ImportError: cannot import name 'vllm_is_batch_invariant'`
  from `vllm.model_executor.layers.batch_invariant` (vllm 0.19.1 removed
  the public getter). vllm-ascend 0.17 imports it in 5 call sites. Session
  reclassified pre-probe "A with note" → real outcome **C**. Fix must ship
  in vllm-ascend (forward-compat helper reading `_batch_invariant_MODE`).
  Takeaway: **pre-probe of plugin registration is necessary but insufficient
  — a wet-run is still required**, because some removals only surface at
  runtime plugin init.

## Related KB (sibling experts)

- `vllm-upgrade-expert/references/patterns/domains/vllm-rename-catalog.md`
  — per-version API ledger (CP-002/CP-004/EC-03). Load for shim patterns
  if outcome B.
- `transformers-day0-expert/references/patterns/domains/overlay-image.md`
  — Dockerfile.overlay template + build-time import trap warning (applies
  here too: don't `import vllm` at Dockerfile build time — it triggers
  torch_npu dlopen)
- **`../../vllm-ascend-day0-expert/`** — sibling Day-0 expert for when
  the fix belongs IN vllm-ascend rather than consumer. Route to this
  skill when detected C-patch on vllm-ascend.
- **`../../torch-day0-expert/`** — parent Day-0 expert when new vllm
  also requires new torch. Usually chain: torch-day0 first → then
  vllm-ascend-day0 if vllm-ascend needs rebuilding for new torch → then
  vllm-day0 if still needed.
- **`../../_shared/references/patterns/domains/day0-deploy-artifacts.md`**
  — 5 deploy deliverables (mandatory for A / A-with-note / C-patch).

## vllm 0.20.0 drift surface (2026-04-23 late session finding)

Day-0 probe on vllm v0.20.0 (commit 579602aa4, 2026-04-22; 156 commits
after vllm-ascend's last main2main cursor) exposes a deeper drift
surface than vllm 0.19.1. V1.3 smoke on overlay requires **6+ point
patches** to reach profile_run, and a 7th structural refactor beyond
that. Concrete drift ledger:

| # | Symptom at LLM() construction | vllm upstream PR | Fix scope |
|---|---|---|---|
| 1 | `ImportError: vllm_is_batch_invariant` | #35007 | 4-file env-var migration (same as vllm-ascend main PR #7787) |
| 2 | `ImportError: Qwen3NextGatedDeltaNet` | #37975 | patch_qwen3_next.py import fallback to `layers.mamba.gdn_linear_attn.GatedDeltaNetAttention` |
| 3 | `ImportError: Qwen3_5GatedDeltaNet` | #37975 | patch_qwen3_5.py same fallback |
| 4 | `ImportError: create_vllm_config_for_draft_model` | #37880 | draft_proposer.py import optional + stub on call |
| 5 | `NotImplementedError manual_seed_all` | #38468 | NPUPlatform.manual_seed_all classmethod |
| 6 | `AttributeError: logprob_token_ids` | vllm 0.20 feat | NPUInputBatch.__init__ add field |
| 7 | `AttributeError: 'Tensor' has no 'gpu'` (self.positions.gpu[]) | #32951 | CpuGpuBuffer wrapper reverted (broke parent's `[]` slicing) — partial, needs per-site rewrite |
| 8 | `AttributeError: 'float' has no 'language_model'` | vllm 0.20 `CompilationTimes` NamedTuple | worker.py compile_or_warm_up_model returns CompilationTimes NamedTuple |
| 9 | `TypeError: _get_cumsum_and_arange() missing 'arange_out'` | #32951 | _prepare_inputs: provide arange_out buffer + old-sig fallback |
| 10 | `TypeError: _prepare_input_ids() missing 'cu_num_tokens'` | #32951 | Call site: add num_reqs positional arg |
| 11 (stopped here) | `TypeError: CpuGpuBuffer object is not subscriptable` | Self-inflicted from iter 7 wrapper | Revert wrapper; 11 per-site `.gpu/.cpu/.np/.copy_to_gpu` rewrites in model_runner_v1.py = **structural main2main work** |

**Conclusion after 11 iterations**: beyond layer 7 (CpuGpuBuffer→tensor
refactor, vllm PR #32951), the drift becomes structural and each fix
unlocks the next. vllm-ascend main's own fix for #32951 (PR #7787 at
commit 811271d1) is ~100 lines of model_runner_v1.py rewrite —
introduces `_positions_cpu_buf` / `_positions_np_buf` / `self.query_pos`
fields and rewrites 11+ call sites. Replicating their work by hand
point-by-point duplicates their main2main skill. **Day-0 skill
boundary hit at layer 7**; beyond that → vllm-ascend team's main2main
skill is the right tool.

Concrete artifacts: `workspace/vllm-day0-vllm0200-20260423-1623/findings.md`
+ personal fork branch `ascend-day0-torch211-20260423` commits
`149393a0..335d99eb` (6 drift-patch commits on top of torch 2.11 Fix B+/C).

**Takeaway**: Day-0 probe past 1-2 minor versions of vllm-ascend cursor
can expose 6+ drift layers. First 6 are each 1-file point patches the
skill can apply iteratively. Beyond that, sync work belongs to the
vllm-ascend team's main2main skill (they have one per PR #6983), not
our iterative Day-0 loop.

## Pre-probe discipline (2026-04-23 lesson)

**Before committing a target vllm version**, check if vllm-ascend main
tip has already adapted:

```bash
cd upstream/vllm-ascend && git fetch origin --tags
git log origin/main -S '<key_symbol_that_new_vllm_removed>' -- vllm_ascend/
```

- If **main has the fix**: the session's real target is not `vllm==X`,
  it's **"base image is older than vllm-ascend main"**. Switch to a
  newer vllm that main also hasn't adapted to → re-run pre-probe.
- If main doesn't have the fix: legitimate Day-0 target.

Concrete 2026-04-23: initially thought vllm 0.19.1 was the target. After
pre-probe on main, discovered commit `811271d1` (PR #7787, 2026-04-03)
already handled `vllm_is_batch_invariant` removal via `vllm.envs.VLLM_BATCH_INVARIANT`.
Real Day-0 target: vllm **v0.20.0** (2026-04-22 release, 156 commits
beyond main's current cursor). See memory `day0_real_target.md`.
