# Codex Review — T30 Test/Safety/Feedback Design (2026-05-15)

> **Source**: codex-cli 0.121.0 adversarial review of `docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md` (commit `7c03dd9`).
> **Run command**: `codex exec --dangerously-bypass-approvals-and-sandbox`
> **Token usage**: 55,306

---

## 1. Design Closure

**Verdict:** partial closure only. The five sub-components do cover the four incidents in §0.1 at a high level, but they miss one important defect class and under-specify two others.

**Covers the four cited classes**
- T22 "walk-through PASS masquerading as real PASS" is targeted by provenance scanning and gates (§0.1, §3, §4).
- T25.5 helper-script bug is addressed by the sanity suite conceptually (§2).
- T26 stale-doc drift is addressed by scanner plus roadmap/doc tests (§2).

**Missed 5th class: workflow bypass**
- The design assumes the scanner/gate path is invoked, but does not defend against "raw path" bypass. P8 in ANTI_PRESSURE_PROTOCOLS explicitly says agents bypass skills/hooks when pressured — which bypasses exactly the protections this design is adding.
- The design also explicitly excludes the stronger workflow critic/hook layer (§0.3). That is a real structural gap, not just "future nice-to-have".

**Two under-covered classes**
- Environment authenticity / stale workspace is a structural source of false evidence in this repo (OL-15). None of P0h-P0l directly validates "am I checking the right clone / branch / environment?"
- Mode heterogeneity is under-modeled. The repo has forked day-0 flows, issue-only flows (sglang), and integrated overlay flows. The design reads as if one artifact model covers all of them. It does not.

## 2. Over-Engineering

**Verdict:** six GateIDs is not the main problem; the problem is that two are redundant and one is not mechanically well-defined.

**Redundant**
- `OUTCOME_PROVENANCE` duplicates P0i almost exactly. If scanner PASS is already a hard precondition, keeping it as a second gate adds ceremony, not independence.
- `PR_MATERIAL_EXISTS` and `KB_CASE_REGISTRY` are both "documentation artifact completeness" checks. They can be one gate with per-mode required artifacts.

**Weak / likely non-mechanical**
- `OL_REGRESSION_FREE` is underspecified and not obviously mechanical. "no new DEBT-N conflicts with existing OL" sounds semantic. That violates your own "gate 优先做 mechanical check" rule.

**Keep separate**
- `FORK_BRANCH_PUSHED` and `SMOKE_LOG_PRESENT` are independent enough to justify separate checks if the mode actually needs them.

**My cut**
- Reduce to 4 gates:
  - `CLAIM_EVIDENCE_PRESENT`
  - `REQUIRED_ARTIFACTS_PRESENT`
  - `EXTERNAL_PUBLICATION_VERIFIED`
  - `VALIDATION_ARTIFACT_VERIFIED`
- Drop `OL_REGRESSION_FREE` from day-0.
- Fold `OUTCOME_PROVENANCE` into scanner or into `CLAIM_EVIDENCE_PRESENT`, not both.

## 3. Dependency Correctness

**Verdict:** runtime layering should stay sequential, but implementation order does not need strict P0h → P0i → P0j → P0k → P0l.

**Runtime**
- Scanner before finalizer is defensible. The flow in §1 is reasonable.
- But scanner and gate are both reading the same claim/evidence surface. If you keep both, define a strict contract between them. Right now they are two ad hoc scrapers.

**Implementation order**
- P0i and P0j can be developed in parallel once the claim schema is fixed.
- P0l is already admitted as parallel in §7.
- ROADMAP says strict dependency for every step, including P0l depending on P0k. That conflicts with §7 of the design. Pick one.

**Best sequencing**
1. Define one structured claim/artifact schema first.
2. Build P0i and P0j against that schema in parallel.
3. Add P0h tests around both before enforcement.
4. P0k after artifact schema stabilizes.
5. P0l whenever you want; it is not a hard dependency.

## 4. a5_ops Self-Critic Layer

**Verdict:** do not import a 41-item self-check layer wholesale. This repo already has that category of defense; the problem is integration, not absence.

**Why I'm saying no**
- The repo already has `/porting-self-challenge` plus 11 challenge patterns and explicit mandatory invocation points.
- The anti-pressure doc already codifies pre-claim behavior.
- Adding 41 more checklist items without integrating them into the actual claim path is classic checklist theater.

**What is actually missing**
- The design never wires `/porting-self-challenge` into the new flow, even though outcome claims are exactly where it is supposed to run.
- So the gap is not "need C1-C41"; it is "make self-challenge a required precondition before scanner/finalize or before user-visible completion claims."

**Recommendation**
- Reuse the existing self-challenge layer.
- Add at most 3-5 missing questions if you find concrete gaps.
- Do not create a second parallel self-critic framework.

## 5. Name-Coupling Protection

**Verdict:** no, the design does not yet avoid the name-coupling/mode-coupling failure mode. P0i and especially P0j are not truly mode-invariant.

**P0i scanner**
- The doc claims scanner is path-parameterized and therefore safe.
- That is only path-level decoupling. Semantic coupling remains:
  - it assumes evidence is expressed as free-text markdown around words like `PASS`, `works`, `imports clean`.
  - it assumes the authoritative sources are `PR_MATERIAL.md` and `KB_INDEX.md`.
- That will be brittle across A/A-with-note vs C-patch vs C-report, and across fork vs no-fork modes.

**P0j gate path parameters**
- The design says `finalize_day0_check.py --workspace upstream/vllm-ascend`.
- The sample gate hard-codes `workspace / "PR_MATERIAL.md"`.
- That directly contradicts the repo architecture for sglang, which explicitly has no fork and no ascend-port branch.
- `FORK_BRANCH_PUSHED` is also not mode-invariant for sglang.

**Bottom line**
- You avoided hard-coding one filename in one CLI flag, but you did not avoid hard-coding one workflow model.
- This is a mode-coupling bug waiting to happen.

## RECOMMENDATIONS

1. **Replace markdown scraping with one structured claim manifest.** Use a single `claim_manifest.yaml` or JSON emitted by the skill, with fields like `mode`, `outcome`, `evidence[]`, `artifacts[]`, `external_refs[]`, `validation_level`. Then let scanner/finalizer validate that structure. Right now both P0i and P0j are fragile regex systems over prose.

2. **Model workflow modes explicitly.** At minimum: `fork_patch`, `issue_only`, `integrated_overlay`. Gate requirements should derive from mode, not from filenames. This fixes the sglang contradiction.

3. **Shrink the gate set.** Merge doc-structure gates, remove or defer `OL_REGRESSION_FREE`, and avoid duplicating P0i inside P0j.

4. **Wire in the existing self-challenge instead of cloning a5_ops's 41-item layer.** Make "run `/porting-self-challenge` before completion claim" part of the formal flow. That is higher leverage than another checklist.

5. **Add one bypass-control item to the design.** If you refuse to do P0e now, at least specify where invocation is enforced: wrapper script, hook, or skill contract. Otherwise the whole net is optional.

6. **Watch the implementation traps.**
   - pytest tests that parse markdown via `split("## §6")` are brittle against harmless heading edits.
   - PyYAML in hooks/pre-commit paths is a dependency risk; use `safe_load`, and keep commit-time checks stdlib-only if possible.
   - `mtime < 24h` for smoke logs is weak evidence and easy to game accidentally.
   - Network/API-backed gates (`FORK_BRANCH_PUSHED`) should not sit on a path that must work offline or during local commit unless you are explicit about that operational requirement.
