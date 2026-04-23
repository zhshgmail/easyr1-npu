# _shared/ — cross-expert templates

Reusable material for any NPU-port expert. Each expert directory (e.g.
`easyr1-expert/`, `transformers/upgrade-expert/`) includes its own
SKILL.md/agent.md/state_machine.yaml, but can pull in these templates
instead of writing them from scratch.

## What lives here

| Path | Who uses it | How |
|---|---|---|
| `references/ALWAYS_LOADED_UNIVERSAL.md` | Every expert | Each expert's own `ALWAYS_LOADED_RULES.md` links to this file; only adds expert-specific items on top (OL-03 denylist, OL-08 allowed edit paths) |
| `scripts/static_check.py` | Every expert | Copy or symlink into expert's `scripts/` — py_compile + dry-import of a caller-specified package |
| `scripts/cleanup_session.sh` | Every expert | Same. Takes `--session-tag` and optionally `--keep-user-provided` / `--preserve-image` |
| `scripts/deploy_to_a3.sh` | Every expert producing a docker image | Same. Takes `--branch`, `--image-tag` or `--reuse-image` |
| `hooks/check_stop_worker.sh` | Every expert spawning a worker agent | Generic Stop-hook body: enforces G2 (static_check), G3 (PASS-claim-has-evidence), OL-04b (Cleanup field), OL-09 (PROGRESS provenance fields). Takes expert-specific config via env |
| `hooks/check_edit_scope.sh` | Every expert defining a PreToolUse restriction | G1 enforcement. Takes allowed-path glob via env |
| `templates/state_machine_skeleton.yaml` | Every expert | Reference skeleton with required phase/invariant/artifact structure; fill in phases + G-invariants specific to the expert |
| `templates/agent_brief_skeleton.md` | Every expert | Reference structure for Phase A→…→D-end brief (load rules, code-path sweep, codegen, build, smoke, cleanup, Handoff) |

## What does NOT go here

Anything dep-specific: the pattern catalog (`CODE_PATH_PATTERNS.md`), error
corrections (`ERROR_CORRECTIONS.md`), platform bugs with version-specific
workarounds (`PLATFORM_BUGS.md`), smoke baseline (`SMOKE_BASELINE.md`),
the `patterns/domains/*.md` deep templates. Each expert owns its own
versions of those.

## Versioning

Experts pin a shared-layer version in their own `SHARED_VERSION.txt`
(a short sha or date) so a later shared-layer change doesn't silently
shift expert behavior. Bump the pin intentionally when adopting a new
shared-layer revision.

## How new experts are instantiated

1. Copy `src/experts/_shared/templates/state_machine_skeleton.yaml` →
   `src/experts/<expert-name>/state_machine.yaml`, fill in phases + invariants.
2. Copy `src/experts/_shared/templates/agent_brief_skeleton.md` →
   `src/experts/<expert-name>/agent.md`, fill in Phase A/B/C/D specifics.
3. Reference (symlink or copy) the shared scripts/hooks into the expert's
   `scripts/` and `hooks/` directories.
4. Write expert-specific references (`ALWAYS_LOADED_RULES.md` extending
   `ALWAYS_LOADED_UNIVERSAL.md`; per-dep `CODE_PATH_PATTERNS.md` etc).
5. Pin `SHARED_VERSION.txt` to the current git sha of this directory.
