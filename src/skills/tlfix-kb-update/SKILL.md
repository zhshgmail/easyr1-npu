---
name: tlfix-kb-update
description: Phase 5 of /tilelang-fix. Sediment a fix into KB_TILELANG_ASCEND.md so future sessions can triage and patch similar bugs without re-discovery. Updates §8.1 verification matrix row, appends a §10.x deep-dive, adds a §11 taxonomy row, and (when applicable) a §12 preventive rule.
---

# /tlfix-kb-update — Phase 5 (write back to KB)

Standalone runnable; called by `/tilelang-fix` orchestrator. Only runs
when `/tlfix-verify` returned verdict `PASS`.

## Inputs

- `kb_path`: path to `KB_TILELANG_ASCEND.md` (default:
  `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md`)
- `triage_json`: from `/tlfix-triage` — gives hypothesis + bug class
- `patch_meta`: from `/tlfix-patch` — gives recipe name + files changed
- `verify_json`: from `/tlfix-verify` — gives PASS verdict + sample data
- `--new-bug-class`: set when triage was `UNKNOWN` and the user manually
  classified — adds a fresh §11 row and (optionally) a new §10.x section

## What gets written

For each `(recipe, bug_class)` pair, append entries in this order:

### §8.1 verification matrix
Add one row:
```markdown
| <target_op>         | <pre_status> → PASS | <recipe_name> | <date> |
```

### §10.x deep-dive (only if `--new-bug-class`)
New numbered subsection following the existing §10.3.1 / §10.3.2 template:
- Symptom (signature regex)
- Root cause (which lowering pass / file:line)
- Fix outline (which lines edited and why)
- Recipe pointer (`workspace/T32_tilelang_rescue/apply_<name>.py`)

### §11 taxonomy
New row in the bug-class table:
```markdown
| <id> | <signature_regex> | <stage> | <recipe> | <KB §10.x ref> |
```

### §12 preventive rule (optional)
If the bug class implies a class of mistakes a future contributor could
re-introduce, add `R-<area>-<n>: <one-line rule>` with KB §10.x backref.

### §13.1 7-check fix checklist
Tick the row for this recipe (recipe-applied / format / rebuild / target-op /
regression-sample / both-mode / KB-updated). All seven must be green
before the recipe is considered "shipped".

## Idempotence

Skill is rerunnable — checks for existing rows by `(target_op, recipe)`
key and updates in place rather than appending duplicates. Diff is
written to `out_dir/kb_diff.patch` so the user can review before
committing the KB edit.

## Output schema

`out_dir/kb_update.json`:
```jsonc
{
  "kb_path": "workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md",
  "sections_touched": ["§8.1", "§11", "§13.1"],
  "rows_added": 1,
  "rows_updated": 0,
  "diff_path": "out_dir/kb_diff.patch",
  "committed": false
}
```

The skill never commits on its own — leaves the working tree dirty for
the user / orchestrator to review and commit explicitly.

## How to invoke

```bash
bash run.sh <triage_json> <patch_meta> <verify_json> <out_dir> \
  [--kb <path>] [--new-bug-class]
```

## See also

- `tlfix-triage/SKILL.md` — supplies bug class
- `tlfix-patch/SKILL.md` — supplies recipe identity
- `tlfix-verify/SKILL.md` — gates whether we get to run at all
- `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md` §8.1 / §10 /
  §11 / §12 / §13 — the sections this skill mutates
