---
name: dep-gap-detect
description: Automated classification of a framework's declared Python dependencies into A/B/C/D/E tiers against a target NPU base image, to judge whether a version bump / new EasyR1 commit requires new NPU adaptation work (scenario P2) or just routine image upgrade (scenario P1). Consumes a requirements.txt-style file plus an image inventory markdown produced by the npu-image-inspect skill.
---

# dep-gap-detect

## What it does

Given:
- A `requirements.txt` (or any pip-style req file) from EasyR1 / target framework
- An image-inventory markdown (produced by `scripts/inspect-ascend-image.sh`)

Emits a classification report in markdown:
- **A** — NPU-native (upstream supports NPU; use as-is)
- **B** — NPU-ported version exists (e.g. `vllm` → `vllm-ascend`)
- **C** — CUDA-only but bypassable via framework shim (no blocker)
- **D** — **CUDA-only BLOCKER** (needs new NPU adaptation work → scenario P2)
- **E** — pure Python / CPU (accelerator-agnostic)

Exit code:
- `0` if all deps fall in A/B/C/E (scenario P1 — proceed to image-upgrade-drill)
- `1` if any D found (scenario P2 — stop, file a task in `docs/easyr1/npu-adaptation-tasks.md`, complete adaptation first)
- `2` on usage / input errors

## When to use

- **Before** running `image-upgrade-drill` on a new EasyR1 commit or new target image
- As the **automated judgment** for the "scenario P1 vs P2" decision
- When hiyouga/EasyR1 ships a new commit and you want to know in 30 seconds whether the port survives the new deps
- During initial assessment of a brand-new framework to port (OpenRLHF, TRL)

## When NOT to use

- To classify a **single upgraded package** — just grep `PACKAGE_RULES` in the script directly
- When you don't yet have an image-inventory markdown for the target — run `npu-image-inspect` first
- For deps that change at runtime (dynamic imports) — this skill only covers declared deps

## Prerequisites

- Bash, awk, sed (portable POSIX, no gawk required)
- A target-image inventory markdown — produced by the [`npu-image-inspect`](../npu-image-inspect/SKILL.md) skill
- A `requirements.txt`-style deps file. If the framework has its deps split across `requirements.txt` + `pyproject.toml [tool.setuptools.dynamic]` etc., cat them into one file before passing

## How to invoke

```bash
# Usual case: check EasyR1 master against v1 image
git -C upstream/EasyR1 show main:requirements.txt > /tmp/easyr1-reqs.txt
bash scripts/dep-gap-detect.sh \
  --reqs /tmp/easyr1-reqs.txt \
  --image-inventory knowledge/images/verl-8.5.0-a3.md \
  --out /tmp/gap-report.md
```

Exit code tells you the verdict; the markdown output has the per-package detail.

## The knowledge base inside

The script embeds a `PACKAGE_RULES` table (inside the script, a heredoc). Each
row: `name=TIER:comment`. This is the **NPU ecosystem knowledge base** —
update it when:

- A new package is encountered and classified during a port/drill session
- An NPU-ported version appears for something previously C or D
- A CUDA-only package that was D becomes bypassable (C)

**Do not silently edit the rules without also**:
1. Recording the rationale in `knowledge/npu-patterns.md` as a stable ID (if the rule reflects a recurring pattern)
2. Noting the change in `docs/easyr1/porting-journal.md` dated
3. Re-running the detection on EasyR1 master + all target images to confirm the new rule doesn't flip prior verdicts

## Interpretation

### D = 0 → scenario P1

All deps are handled. Proceed with `image-upgrade-drill`:
1. Build drill image
2. Run V1.4/V2.2 smoke
3. Iterate on any API breaks (tier-1 work: try/except imports, hasattr gates)

No new adaptation work needed from sister projects or Ascend team.

### D ≥ 1 → scenario P2

Stop. Before running drill:
1. For each D-tier dep, manually verify — is there really no NPU version under a different name? Is it actually required or can the feature be disabled?
2. If genuinely required, file a task in `docs/easyr1/npu-adaptation-tasks.md` at the correct tier:
   - **Tier 1** (this repo): if a Python shim in EasyR1 suffices (moves dep to C)
   - **Tier 2** (delegate): if kernel / Python-library port needed → sister projects (`ascend-fused-accuracy-probe` for A3 kernel verification, `a5_ops` for A5 kernel gen, A3 kernel repo for A3)
   - **Tier 3** (escalate): if CANN runtime C layer bug → Ascend team
3. Track the task to completion. When done, update `PACKAGE_RULES` in this skill's script to reflect the new status. Re-run detection; dep should now be A/B/C.
4. Only then proceed with drill

## False-positive / false-negative risks

**False positive (D reported but actually fine)**:
- A dep got renamed upstream and image installs under the new name (e.g. `flash-attn-2` vs `flash-attn`). Fix: add an entry to `PACKAGE_RULES` with `:` comment pointing at the real name.
- A dep is pulled in transitively but not declared; script reads only `requirements.txt` so it's silent. Usually harmless.

**False negative (D missed but should have been caught)**:
- A dep is listed in `PACKAGE_RULES` with an old-now-incorrect tier. Audit `PACKAGE_RULES` against `knowledge/npu-patterns.md` periodically.
- Image inventory missed a package that's actually installed (inspector script has stale parsing). Re-run `inspect-ascend-image.sh` and diff.

## Output schema

The report has 3 sections:
1. **Summary** — per-tier counts + PASS/FAIL verdict
2. **Per-package classification** — every dep with its tier and reasoning
3. **Next steps** — concrete action list depending on verdict

Stable-format enough that future tooling (e.g. CI regression) can parse the Summary table's D count.

## Gotchas

- The script uses portable POSIX awk, not gawk. Don't switch to `match(... , array)` syntax — it won't work on minimal Alpine-style shells.
- `PACKAGE_RULES` is embedded in the script (not a separate YAML). This is intentional: the rules ARE the knowledge; co-locating ensures they don't drift. Update in-place when adding rules.
- Image-inventory parser expects markdown tables with `| pkg | version |` rows. If `inspect-ascend-image.sh` changes its output format, re-sync the awk section.

## Related

- [`npu-image-inspect`](../npu-image-inspect/SKILL.md) — produces the image inventory input
- [`image-upgrade-drill`](../image-upgrade-drill/SKILL.md) — the next step after P1 verdict
- [`docs/easyr1/easyr1-dep-chain-audit.md`](../../docs/easyr1/easyr1-dep-chain-audit.md) — the manual baseline audit that this skill's rules are seeded from
- [`docs/easyr1/npu-adaptation-tasks.md`](../../docs/easyr1/npu-adaptation-tasks.md) — where D-tier findings become tracked tasks
- [`knowledge/npu-patterns.md`](../../knowledge/npu-patterns.md) — stable-ID NPU patterns some rules reference
