# easyr1-npu — project context for Claude Code

## What this project is

Port **EasyR1 (master tip, April 2026)** to **Ascend 910C (A3)** NPU, and ship a reusable harness (skills + KB + scripts) that automates the same porting work for future EasyR1 versions or 4 NPU upstreams (vllm-ascend / torch-npu / transformers / triton-ascend).

EasyR1 is a slimmed-down fork of veRL. veRL has already been ported to NPU (see `quay.io/ascend/verl:verl-8.5.*-a3-*` images). This project derives the EasyR1-on-NPU dependency set, lands forward-compat shims at the 4 NPU upstreams when their community parents drift forward, and stacks them into a single integrated overlay image that runs V1.4 GRPO end-to-end.

**Two deliverables**:

- A working EasyR1 on A3 (rollout + RL training).
- A reusable harness — slash-command skills, knowledge docs, scripts, workflows.

See [`docs/_meta/ARCHITECTURE.md`](docs/_meta/ARCHITECTURE.md) for the full architecture (with mermaid diagrams) and end-to-end flow.

## Current state (2026-04-28)

- `easyr1-npu:integrated-20260427` (28.2 GB, on A3 host) — V1.4 GRPO PASS twice (T22.7 fresh baseline + T25.5 cold-drive replay).
- All 4 NPU upstream `ascend-port/<target>` branches per [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md).
- A3 host: `ssh -p 443 root@115.190.166.102`. Workspace: `/home/z00637938/workspace/easyr1-npu/`.

## Repository layout

```
~/workspace/easyr1-npu/
├── repo/                              # this git repo (github.com/zhshgmail/easyr1-npu)
│   ├── README.md                      # entry point
│   ├── ONBOARDING.md                  # one-page customer quickstart (v1 + v2)
│   ├── CLAUDE.md                      # this file
│   ├── docs/
│   │   ├── easyr1/                    # EasyR1 customer-facing
│   │   │   ├── PORT-GUIDE.md              # v1 path
│   │   │   └── PORT-GUIDE-v2-integrated.md# v2 integrated overlay
│   │   ├── _meta/                     # project-level authoritative docs
│   │   │   ├── ARCHITECTURE.md            # overall architecture + flow (mermaid)
│   │   │   ├── UPSTREAM_FORKS.md          # fork+branch ledger
│   │   │   ├── SKILLS-USAGE.md            # slash-command usage
│   │   │   ├── GLOSSARY.md                # term glossary
│   │   │   ├── DOCS-CONVENTION.md         # where each doc kind lives
│   │   │   └── kb/                        # cross-layer lessons + self-critic patterns
│   │   ├── vllm-ascend/PORTING-GUIDE.md   # for vllm-ascend maintainers
│   │   ├── torch-npu/PORTING-GUIDE.md     # for torch-npu maintainers
│   │   ├── transformers/PR_MATERIAL_v5.4_outcome_A.md  # for transformers maintainers
│   │   └── _archive/                  # archived / superseded docs
│   ├── src/
│   │   ├── skills/                    # CC-format skills
│   │   │   ├── _shared/               # shared workflows + patterns
│   │   │   ├── vllm-ascend/port-expert/   # /vllm-ascend-day0
│   │   │   ├── torch-npu/port-expert/     # /torch-npu-day0
│   │   │   ├── transformers/port-expert/  # /transformers-day0
│   │   │   ├── triton-ascend/port-expert/ # /triton-ascend-port
│   │   │   ├── easyr1/port-expert/        # /easyr1-port
│   │   │   ├── dep-analysis/expert/       # /dep-analysis
│   │   │   └── orchestrators/npu-port/    # /npu-port
│   │   └── scripts/                   # install-skills.sh, run-npu-container.sh, smoke
│   └── knowledge/
│       ├── npu-patterns.md            # 29 stable NPU pattern IDs
│       ├── upstream-refs.md           # NPU image → upstream tag mapping
│       └── images/                    # pip freeze etc. extracted from verl-A3 images
└── upstream/                          # each subdir is an independent git clone
    ├── EasyR1/                        # github.com/hiyouga/EasyR1
    ├── verl/                          # github.com/verl-project/verl  (GPU reference)
    ├── torch-npu/ , transformers/ , vllm-ascend/ , triton-ascend/   # 4 NPU upstreams
    └── pytorch/ , vllm/                                              # community repos for drift scans
```

**Ground rule on upstream edits**: work on `ascend-port/<target-version-slug>` branches in the personal forks. No patch files as the dev format — real git only. Export PRs as the end-of-work artifact for the upstream maintainer.

## Docker images (on A3 host)

- `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5` (24 GB, newer reference)
- `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest` (14 GB, v1 base)
- `easyr1-npu:integrated-20260427` (28.2 GB, v2 integrated, validated 2x)
- `easyr1-npu-vllm0200:iter20-abi-both` (v2 base — vllm 0.20.0 / torch 2.11)

CANN source is **not** in `upstream/` — it lives at `gitcode.com/cann`. Pull on demand.

## Working preferences

- **ROADMAP is single source of truth for open work**: any "next step / TODO / pending / 技术债 / DEBT-N / P0xxx" request → **first action is Read [`docs/_meta/ROADMAP.md`](docs/_meta/ROADMAP.md)**. Don't search handovers / SKILL.md / TaskList first. Ad-hoc backlog files (TODO.md / BACKLOG.md / FOLLOW_UPS.md) are forbidden; collapse them into ROADMAP §6.
- **Anti-pressure protocols**: every day-0 / port skill loads [`src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md`](src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md) at Phase A; cite P1..P8 at high-leverage decision points (spawn agent / emit outcome / skip verify / nohup / inline workaround).
- **Cross-skill OL catalog**: [`src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md`](src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md) is the grep-keyword index for OL-01..OL-27. Search this file first, then load the detail file.
- **Session handover**: every session ends with a `docs/_meta/handovers/SESSION_HANDOVER_<date>_<slug>.md` filled from [`SESSION_HANDOVER_TEMPLATE.md`](docs/_meta/handovers/SESSION_HANDOVER_TEMPLATE.md); next agent reads it before anything else.
- **Auto-compact context-preservation**: see the dedicated **Compact Instructions** section below; on resume, follow that protocol before any new action.
- **Docs organization** is governed by [`docs/_meta/DOCS-CONVENTION.md`](docs/_meta/DOCS-CONVENTION.md). Every kind of information has a single authoritative file. README is index only, never content sink.
- **Customer-facing docs must not contain stale or internal info.** When writing for customers (README / ONBOARDING / PORT-GUIDE / SKILLS-USAGE / PORTING-GUIDE), drop session tags (T22.x), worklog phrasing, and dated dev process artifacts. Move that to KB or archive.
- **Mirror milestones to Discord.** User is primarily on Discord, not terminal. `mcp__plugin_discord_discord__reply` with chat_id `1494825170399924366`.
- **Commit messages**: no Claude-related text.
- **Project docs default to Chinese**: README / ARCHITECTURE / port guides 默认中文；代码注释、commit message、SKILL.md frontmatter 仍英文。

## Tools available

- `gh` (GitHub, logged in) — for GitHub repo actions
- `gc` (GitCode, logged in) — for Ascend/* NPU repos (primary NPU repos live on gitcode; GitHub mirrors may lag)
- `docker` — image inspection on dev box; full container ops on A3
- A3 host: `ssh -p 443 root@115.190.166.102` (workspace under `/home/z00637938/`)

---

## Compact Instructions

Claude Code auto-compacts the conversation around 75% context utilization. Compaction RE-INJECTS this CLAUDE.md file fresh from disk, but DISCARDS:
- Inline conversation history (only a summary survives)
- Specific file paths and line numbers I mentioned in prior turns
- Error messages, stack traces, debugging hypotheses
- Detailed instructions / clarifications the user gave mid-session
- Architectural decision reasoning expressed in chat

To stay coherent across compactions, **two contracts**: (a) the writer side — anything important goes into a durable file indexed below; (b) the reader side — on every wake-up after compaction, before any new action, follow the recovery protocol.

### Writer side: what MUST be persisted (every time)

If any of these happens mid-session, write it to a versioned file BEFORE moving on. Don't trust chat memory.

| Event | Where it goes |
|---|---|
| User correction / preference / "don't do X" / "always do Y" | New `memory/feedback_*.md` + add link to `memory/MEMORY.md` |
| Important project fact (deadline, who-owns-what, why-we-chose-X) | New `memory/project_*.md` + link in `MEMORY.md` |
| External system reference (Linear/Slack/dashboard URL) | New `memory/reference_*.md` + link in `MEMORY.md` |
| In-flight task state at session end / context near full | New `docs/_meta/handovers/SESSION_HANDOVER_<date>_<slug>.md` from `SESSION_HANDOVER_TEMPLATE.md` |
| Long-term open work / backlog item / tech debt | New row in `docs/_meta/ROADMAP.md` §2/§6 |
| Recurring code-level lesson / KB-worthy bug pattern | New row in `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md` §10/§11/§12 (or equivalent KB) |
| Customer-facing fact about deliverable | `docs/easyr1/PORT-GUIDE*.md` / `ONBOARDING.md` (not chat) |

### Reader side: post-compaction wake-up protocol (mandatory)

When a fresh context loads (system reminder shows compaction or you can't recall what was just discussed), execute these reads BEFORE any new tool call:

1. **Index check** (this CLAUDE.md is already loaded — that's how you're reading this)
2. **Read auto-memory index**: `Read ~/.claude/projects/-home-z00637938-workspace-easyr1-npu/memory/MEMORY.md` (the system reminder may or may not include it)
3. **Read latest handover**: `ls -t docs/_meta/handovers/SESSION_HANDOVER_*.md | head -1` then Read it in full
4. **Read ROADMAP if the user mentioned "next step / TODO / pending / DEBT-N / P0xxx"**: `Read docs/_meta/ROADMAP.md`
5. **Read relevant memory entries** based on the user's first prompt and what the handover dispatches you to do. Don't read all of them — just what's named in the dispatch.
6. **Verify durable claims before recommending action**. Memory says a file/function/flag exists? Grep-check it — file might have moved or been deleted since the memory was written.

Only then call tools to do new work.

### Compact-survival index: critical paths (always durable)

Memorize this index — these are the entry points that always live in tree (not chat), so even after worst-case compaction the next agent can find current state.

**Project state**:
- `CLAUDE.md` (this file) — coordination + entry points
- `docs/_meta/ROADMAP.md` — single source of truth for open work
- `docs/_meta/ARCHITECTURE.md` — system architecture + mermaid diagrams
- `docs/_meta/handovers/` — per-session inline tactical state (most-recent file = current state)
- `docs/_meta/UPSTREAM_FORKS.md` — which forks / branches own what
- `docs/_meta/SKILLS-USAGE.md` — how to invoke project skills
- `docs/_meta/kb/` — porting lessons + challenge patterns
- `docs/_meta/SESSION_HANDOVER_TEMPLATE.md` — fill at session end

**Behavioral memory (cross-session)**:
- `~/.claude/projects/-home-z00637938-workspace-easyr1-npu/memory/MEMORY.md` — auto-memory index (also auto-loaded into system reminder)
- `memory/*.md` — individual feedback / user-preference / project-context / reference entries

**Knowledge base**:
- `workspace/T32_tilelang_rescue/KB_TILELANG_ASCEND.md` — tilelang on Ascend: env, bug taxonomy (§11), prevention rules (§12), runbook (§13), verification matrix (§8.1)

**Tooling state**:
- `src/skills/<skill-name>/SKILL.md` — every CC skill in this repo
- `src/skills/_shared/references/OPERATIONAL_KNOWLEDGE.md` — OL-* keyword index
- `src/skills/_shared/references/ANTI_PRESSURE_PROTOCOLS.md` — P1..P8 anti-shortcut rules

**External infrastructure**:
- A3 host: `ssh -p 443 root@115.190.166.102`; workspace `/home/z00637938/workspace`; containers must have `/home/z00637938 → /home/z00637938` bind-mount; never delete containers (stopped or running) without explicit user OK; conda + cache cleanups are allowed but `docker rm` and `docker container prune` are NOT
- Discord: `mcp__plugin_discord_discord__reply` chat_id `1494825170399924366`
- Fork: `github.com/zhshgmail/tilelang-mlir-ascend`; mainline upstream: `github.com/tile-ai/tilelang-mlir-ascend`

### Hard rules that bind across compactions

1. **Never plan around session boundaries.** Don't write "this session vs next session" or invoke a 5-minute rule. Keep working until done; record decisions to versioned files. (Memory: `no_session_planning.md`.)
2. **Never reference user's timezone / sleep / availability.** The agent has no view into user's rhythm. (Memory: `no_user_schedule_references.md`.)
3. **End-to-end is end-to-end.** Don't mark Phase tasks "completed" when only a cheaper intermediate result exists; cold-drive the skill chain on a clean reproducer. (Memory: `end_to_end_vs_described.md`.)
4. **NPU containers on A3 are sacred** — never `docker rm` / `docker container prune` even of stopped containers without explicit user OK. (Memory: `a3_cleanup_and_reuse.md`.)
5. **Discord cadence** — push coarse-grained updates at milestones; no > 15-minute silences. (Memory: `discord_cadence.md`, `discord_reporting_cadence.md`.)
6. **Project docs default to Chinese** (READMEs / port-guides); code + commit messages stay English. (Memory: `project_docs_language.md`.)
7. **Commit messages**: no Claude-related text.
8. **A3 firewalled**: do web/probe locally, git-sync to A3. (Memory: `a3_is_firewalled.md`.)
9. **Before recommending a function/file/flag from memory**, verify it still exists (grep / ls). Memory snapshots can rot.

### How to extend this

When the user introduces a new "always do X" / "never do Y" rule:
1. Write a new `memory/feedback_<short_name>.md` with frontmatter `type: feedback`, body following the **Rule → Why → How to apply** structure
2. Add a one-line entry to `memory/MEMORY.md` linking to it
3. If the rule should bind permanently across all sessions, also add a numbered bullet to "Hard rules that bind across compactions" above
