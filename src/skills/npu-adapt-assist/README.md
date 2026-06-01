# /npu-adapt-assist — quick reference

NPU adaptation KB retrieval. Paste an error trace, get the matching cookbook entry from `docs/_meta/kb/porting_lessons/`.

## Usage

```
/npu-adapt-assist <paste error or stack trace>
```

Or with a file:

```
/npu-adapt-assist --trace-file /tmp/run.log
```

## What it does

0. **Preflight** (automatic) — `scripts/preflight.sh` verifies prerequisites
   (KB dir reachable, retrieve.py present, python3 OK, ≥1 schema-conformant
   KB entry). ABORT verdict → retrieve.py refuses to proceed (exit 2).
1. Loads all 25 schema-conformant KB entries (`docs/_meta/kb/porting_lessons/*.md`)
2. Ranks by frontmatter `trigger` and `symptom_in_wild` match against input
3. Returns top match + correction recipe quoted verbatim

## What it does NOT do

- Open PRs (always human review)
- Apply patches (Phase B, future)
- Search beyond local KB (no web, no other repos)
- Invent matches when nothing fits (returns `UNKNOWN` + suggests adding a KB entry)

## Cold-drive validated cases

| Case | Input | Expected match |
|---|---|---|
| A | RMSNorm bias AttributeError in `fused_split_qk_norm.py:118` | `sglang-002` |
| B | ImportError from vllm/sglang on `/` cwd | `cross-layer-008` |
| C | `sparse_mla_fwd` NaN at NS≥2, NS=1 OK | `bishengir-001` |

Run validator: `./scripts/cold_drive_validate.sh`.

## Preflight standalone

```
./scripts/preflight.sh           # human verdict
./scripts/preflight.sh --json    # machine verdict (CI)
./scripts/preflight.sh --quiet   # output only on WARN/ABORT
```

Verdicts: `CLEAN` (exit 0) | `WARN` (exit 1) | `ABORT` (exit 2).

## When to add a new KB entry

When `/npu-adapt-assist` returns UNKNOWN for a real failure mode you encountered:

1. Diagnose root cause yourself
2. Copy `docs/_meta/kb/porting_lessons/_schema.md` template
3. Fill in `trigger`, `symptom_in_wild`, `root_cause`, `correction`, `evidence`
4. Save as `<layer>-<NNN>-<slug>.md`
5. Add link to `docs/_meta/kb/porting_lessons/index.md`
6. If a new bug class: add `P-<CLASS>-<N>` row in `output/miles-dsv4-flash-poc/docs/REPORT.md` §4.0

Future calls to `/npu-adapt-assist` will then match this case.

## See also

- KB schema: `docs/_meta/kb/porting_lessons/_schema.md`
- KB index: `docs/_meta/kb/porting_lessons/index.md`
- PoC report problem catalog: `output/miles-dsv4-flash-poc/docs/REPORT.md` §4
