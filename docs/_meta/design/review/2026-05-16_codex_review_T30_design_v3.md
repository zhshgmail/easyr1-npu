# Codex Review — T30 Test/Safety/Feedback Design v3 (2026-05-16)

> **Source**: codex-cli adversarial review of `docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md` v3 (commit `73b28c2`).
> **Run command**: `codex exec --dangerously-bypass-approvals-and-sandbox`
> **Token usage**: 68,747
> **Verdict**: v3 has multiple regressions vs v2 + new bugs from cargo-culting a5_ops literals. v4 required.

---

I’ve confirmed the M3 claim against the repo: `/sglang-npu-day0` and `/integrated-overlay-build` don’t spawn sub-skills, but `/npu-port` already does. I’m tightening the review now around where the v3 text drifted away from the actual shared-schema and mode-dispatch constraints.
codex
## 1. Incorporation correctness

**Verdict:** partial only. v3 names the right lessons, but two of the three incorporations are still under-mechanized, and one is mostly policy prose.

- **M2 anti-cycle is not actually unambiguous or robust.** The doc promises a general “data-flow-independent” rule and even says the scanner rejects a verifier that reads “the path-derived file the verifier is supposed to verify” ([design §4.2](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:401), [§4.2 impl](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:417)). The implementation shown only does raw substring checks for `"claim_manifest.yaml"` and `"verification.json"` ([lines 425-435](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:425)). That is easy to evade with variable-built paths, helper functions, symlinks, renamed copies, or reading the manifest indirectly through another file. Also, `verification.json` is an a5_ops literal, not a repo-native object here; that is cargo-culting, not threat modeling.
- **M2 has a path-resolution bug in the design itself.** The manifest location is `workspace/<skill-tag>/claim_manifest.yaml` ([lines 131-133](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:131)), but the sample verifier paths are repo-root style like `src/scripts/smoke/replay_v14.sh` ([lines 206-212](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:206)). `_check_anti_cycle()` resolves them as `manifest_dir / v["path"]` ([lines 417-421](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:417)), which points at `workspace/<skill-tag>/src/...`, not repo root. That will falsely report missing scripts unless every manifest rewrites paths relative to its own directory.
- **M2 claims an executable-bit check but does not implement it.** The error list says “doesn't exist OR isn't executable” ([line 405](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:405)); the code only checks `exists()` ([lines 421-423](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:421)).
- **M1 gate-fitness is miswired in the example.** §3.6 says fixtures must prove each gate catches crafted fraud ([lines 333-363](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:333)), but the example test only calls `validate_manifest()` ([lines 348-359](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:348)). Several listed frauds are assigned to P0j gates, not schema validation, so this harness would never prove the finalizer works.
- **P9 is only policy text today, not a mechanical protocol.** §11.2 gives good instincts, but it does not define a classifier that can mechanically distinguish “baseline violated” from “fixable helper/script bug” or “recoverable transient” ([lines 622-639](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:622)). Without that boundary, workers will still improvise.

## 2. M5 short-term degradation soundness

**Verdict:** security theater. I would not count it as a real defense.

- The degraded check scans `self_challenge.notes` for strings like `Read.*finalize_day0_check` or `Bash.*test.*gate` ([lines 407-410](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:407)). That is not tool telemetry. It is self-reported prose.
- `self_challenge` is manually merged back into the manifest by the day-0 skill ([lines 489-494](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:489)). An agent intent on gate-enumeration can simply omit the incriminating sentence.
- The wording is internally inconsistent: the text says “contains substring `Read.*finalize_day0_check`” ([line 408](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:408)). If implemented literally as substring search, `.*` is not regex, so it matches almost nothing useful.

Recommendation: move this entirely to **telemetry-only / warning-only** status and treat the real control as **DEBT-6 / P0e**. Right now it creates false confidence.

## 3. Bad fixture coverage

**Verdict:** the six fixtures are not the right six.

- **You are missing a bad fixture for Gate 3 (`EXTERNAL_PUBLICATION_VERIFIED`).** The four gates are defined in §4.3 ([lines 443-448](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:443)), and §3.6 says every gate needs crafted fraud coverage ([line 362](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:362)). None of the six fixtures exercises “branch not pushed / image tag not real / issue URL unreachable.”
- **`bad_issue_only_with_fork` overlaps with an existing v2 bad fixture.** §3.1 already has `bad_claim_wrong_mode_artifact_combo.yaml` ([lines 257-258](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:257)); §3.6 adds `bad_issue_only_with_fork` for the same contradiction ([line 344](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:344)).
- **`bad_self_citing_verifier` and `bad_circular_evidence` are mostly the same family, and the latter is poorly modeled.** `evidence[smoke_log].path == verifier_scripts[0].path` ([line 343](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:343)) is not a convincing cycle; it is usually just “log path equals script path,” which is type confusion, not the real self-citation attack.
- **`bad_mirror_md5_trivial` is not grounded in any defined mechanic.** The fixture claims integrated-overlay should be forced from `A` to `A-with-note` if artifacts are byte-identical baseline files ([line 340](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:340)), but the schema and gates define no baseline-hash source or comparison rule ([schema](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:145), [gates](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:443)).
- **You are missing a fixture for the new verifier-path contract.** Given the repo-root vs manifest-relative ambiguity, there should be a bad fixture for a verifier path that is valid in one interpretation and invalid in the other. Right now the design can fail on honest manifests.

The six I would want are:
1. self-citing verifier
2. verifier path resolves outside allowed root / wrong root
3. issue_only mode with fork artifact
4. published artifact missing/unreachable
5. numeric metric null under PASS
6. log assertion is construction-marker / synthetic

## 4. M3 deferral

**Verdict:** the deferral rationale is factually wrong.

- v3 says M3 is inapplicable because there is “no multi-agent orchestrator yet” and `/npu-port` does not really spawn sub-agents ([line 613](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:613)).
- The repo already contains a real orchestrator skill at [`src/skills/orchestrators/npu-port/SKILL.md`](/home/z00637938/workspace/easyr1-npu/repo/src/skills/orchestrators/npu-port/SKILL.md:26). It explicitly spawns `dep-analysis-worker`, sequential upgrade workers, and then a consumer port expert ([lines 35-57](/home/z00637938/workspace/easyr1-npu/repo/src/skills/orchestrators/npu-port/SKILL.md:35)).
- `/sglang-npu-day0` does **not** orchestrate sub-skills today ([SKILL](/home/z00637938/workspace/easyr1-npu/repo/src/skills/sglang/port-expert/SKILL.md:24)), and `/integrated-overlay-build` also does not spawn sub-agents ([SKILL](/home/z00637938/workspace/easyr1-npu/repo/src/skills/_shared/integrated-overlay-build/SKILL.md:61)). So if your question is strictly about those two: no, they are not the reason M3 must apply.
- But the broader claim “we have no orchestrator yet” is still false because `/npu-port` exists right now.

So M3 deferral is not defensible as written. At minimum, scope it down to “not required for standalone day-0 skills, but required for `/npu-port`.”

## 5. Schema field addition

**Verdict:** `verifier_scripts[]` is the right direction, but the field is incomplete.

- I would **not** merge it into `evidence[]`. Evidence is what supports the claim; verifier scripts are mechanisms whose independence must be audited. Those are different invariants.
- The real problem is that `verifier_scripts[]` is currently **disconnected** from the evidence items it is supposed to validate or produce ([lines 202-213](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:202)). That is why §4.2 can talk vaguely about “the path-derived file the verifier is supposed to verify” without any actual linkage ([lines 403-405](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:403)).
- Clean fix: keep `verifier_scripts[]`, but add an explicit relation such as `evidence[].verified_by_role` or `verifier_scripts[].produces_evidence_types`. Without that, the anti-cycle rule is underspecified.

## 6. Handling of a5_ops postmortem learning

**Verdict:** too much of v3 treats a5_ops as instruction text, not as attack data to be re-modeled for this repo.

- The strongest tell is the hard-coded `verification.json` check ([lines 430-435](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:430)). That file belongs to the foreign incident. Importing its literal name into this repo’s detector means the lesson was copied, not translated.
- The fixture names `DS-PERF-FRAUD-1` and `WORKER-PRECISION-T1-BY-CONSTRUCTION` are also imported labels ([lines 341-342](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:341)) without a strong local model of how those frauds arise against `claim_manifest.yaml`.
- P9 is directionally correct, but still not adapted into a repo-specific triage rule for “host baseline broken” vs “our helper script is broken.” That distinction matters a lot here.
- The M3 error is another sign: the design imported the audit taxonomy without re-checking the current repo’s actual orchestrator surface.

## 7. New regressions vs v2

**Verdict:** yes, v3 regressed the v2 “single shared schema + mode-derived dispatch” story.

- **Schema/gate contract drifted apart.** In §2.3, `integrated_overlay` requires `evidence[commit_ref]` “for image SHA” and says `artifacts[pr_material]` is optional ([lines 221-233](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:221)). In §4.4, the gate requires `ArtifactRole.IMAGE_TAG`, `SMOKE_LOG`, and `CHECKPOINT_PATH` for `integrated_overlay`, and `WORKAROUND_DOC` for `issue_only` ([lines 458-480](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:458)). Those are different models. That is a direct regression against the v2 principle that scanner and gate share one strict contract.
- **Outcome naming is no longer canonical.** The architecture block uses `A-w-note` ([lines 62-63](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:62)), the schema example uses `A-with-note` ([line 142](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:142)), and the code enum uses `A_WITH_NOTE` ([lines 474-479](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:474)). v2 was trying to kill this kind of mode-coupled string drift.
- **Mode-aware dispatch is weakened by artifact-role drift.** §4.4 says the gate code should not contain literals like `PR_MATERIAL.md` ([line 483](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:483)), but §4.3 still describes explicit section-header checks for `PR_MATERIAL.md`-style content ([line 446](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:446)), while the schema example still hard-codes a `PR_MATERIAL.md` path ([lines 177-183](/home/z00637938/workspace/easyr1-npu/repo/docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md:177)). That is not fatal by itself, but it shows the abstraction is already leaking.
- **Verifier path handling introduces a new cross-mode failure path.** The manifest-root vs repo-root ambiguity affects every mode, not just one.

## RECOMMENDATIONS

1. Rewrite M2 as a real provenance rule, not a grep rule. Add explicit verifier-to-evidence linkage, define path roots unambiguously, and reject repo-external or helper-mediated reads of the manifest.
2. Downgrade M5 degraded scan to “telemetry only” or delete it from the control story. The real implementation belongs in DEBT-6/P0e.
3. Fix the M1 harness so bad fixtures hit the correct layer: schema fixtures test `validate_manifest()`, gate fixtures test `finalize_day0_check.py`, and every one of the four gates has at least one fraud case.
4. Remove or amend the M3 deferral now. `/npu-port` already makes aggregate-cap style controls relevant.
5. Reconcile §2.3 and §4.4 into one canonical mode/evidence/artifact table. Right now v3 reintroduced exactly the contract drift v2 was trying to eliminate.
6. Treat the a5_ops audit as attack data. Every borrowed rule should be restated in local terms, with local artifacts and local bypasses, not foreign filenames and foreign labels.
tokens used
