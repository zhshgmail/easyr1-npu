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

**Guidance to the auto-compact SUMMARIZER (preserve these info types verbatim — losing them caused the 2026-06-02 "is inference fake" trust crisis):** when summarizing this conversation, you MUST carry forward, not drop:
1. **Working recipes** — for anything that was made to RUN/PASS, keep the EXACT path that worked: which script, which env vars / flags / monkey-patches, which container, which model. (Crisis cause: the summary kept the recent "grinding on tilelang" thread but dropped that V4 inference works via **torch-fallback (stub tilelang), NOT by compiling tilelang** → next reasoning re-derived the wrong path and declared real work "blocked/fake".)
2. **Key architecture decisions + rationale** — which approach works and which was rejected and WHY (the discriminator), not just the most-recent tactical activity.
3. **Verified-vs-unverified status with evidence pointers** — for each claimed result, whether it was verified-this-session (with log path) or relayed from a prior artifact (unverified). Never collapse "ran once per an old log" into "done/works".
4. **Open trust/honesty corrections the user made** — verbatim, so they aren't re-violated.
5. **Workspace/artifact OWNERSHIP & PROVENANCE** — caused the 2026-06-03 "T32 codegen is someone else's active fix" catastrophic-forgetting (same class as the 06-02 "is inference fake" false alarm). MUST carry forward, verbatim: **this is a SOLO workspace — the user (single operator) and I are the only ones who touch `/home/z00637938/workspace/*`. ALL uncommitted/dirty changes, half-finished branches, `.orig` backups, and prior-"T32-session" edits are MY OWN past work, NOT a teammate's.** Never infer a third party ("someone else is editing / a teammate's in-flight work / can't touch it") from a dirty tree, a branch name, or an unfamiliar diff — when I don't recognize my own prior work, the correct conclusion is "I forgot I did this," NOT "someone else did this." Before ever claiming work belongs to another party or is "actively being worked on": verify with `git reflog` (whose branch), file mtime (stale vs live), and live-process check (`ps` for ninja/cmake) — and even then, default to "mine, forgotten" because the workspace is solo.
If summary space is tight, drop tactical play-by-play BEFORE dropping any of these five.

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
| **A result was made to RUN/PASS** (the single most-compaction-fragile info type — caused the 2026-06-02 "is inference fake" crisis) | **New `workspace/<area>/RESULT_<slug>_<date>.md`** capturing: (1) **the exact reproduction RECIPE** — runnable script path + the env vars / flags / monkey-patches that made it work + container + model path; (2) **the captured raw output** (the proof line, e.g. `output_ids=[...]` / `grad_norm=...`); (3) **why THIS path and not the alternative** (e.g. "torch-fallback, NOT tilelang-compile, because sglang V4 has torch fallbacks for every tilelang kernel"); (4) verified-vs-unverified + limits. Also drop a timestamped run-log in `workspace/<area>/_runlogs/`. |
| **A key architecture/path DECISION + its rationale** (which approach works, which was rejected & why) | New `memory/project_*.md` — state the decision AND the discriminator ("path A works because X; path B fails/rejected because Y"). Compaction keeps recent tactical threads but DROPS these decisions → next session re-derives wrong. |

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

**🚨 ALWAYS READ FIRST IF IN A "DO NOT STOP" SESSION**:
- `docs/_meta/handovers/SESSION_HANDOVER_V4_attempt_in_progress.md` — V4 真路径 attempt in-progress, user-stated "never stop while tokens remain", strict no-timezone-assumption / no-V3.2-substitution rules
- `memory/deception_under_closure_pressure_2026_06_01.md` — V3.2 substitution catastrophic incident, permanent anti-pattern
- `memory/verify_architecture_class_against_huggingface_truth.md` — every arch-class decision MUST verify HF truth + user OK
- `output/miles-dsv4-flash-poc/docs/REPORT.md` §0 Disclosure — what is the V3.2 vs V4 distinction in shipped artifacts
- `workspace/v4_attempt_2026_06_01/README.md` — V4 attempt log, contains stepwise narrowing of where forward hangs

**Project state**:
- `CLAUDE.md` (this file) — coordination + entry points
- `docs/_meta/ROADMAP.md` — single source of truth for open work
- `docs/_meta/ARCHITECTURE.md` — system architecture + mermaid diagrams
- `docs/_meta/handovers/` — per-session inline tactical state (most-recent file = current state)
- `docs/_meta/UPSTREAM_FORKS.md` — which forks / branches own what
- `docs/_meta/SKILLS-USAGE.md` — how to invoke project skills
- `docs/_meta/kb/` — porting lessons + challenge patterns
- `docs/_meta/SESSION_HANDOVER_TEMPLATE.md` — fill at session end

**T32 / T33 forensic artifacts (NEVER delete or move without ROADMAP update)**:
- `workspace/T32_tilelang_rescue/ROADMAP.md` — T32 effort task table + DAG (the inner ROADMAP, not the project one)
- `workspace/T32_tilelang_rescue/rka16_ns4_passes.txt` (~2.9 MB) — full bishengir 311-pass IR dump for R-KA-16 reproducer; cited from AscendNPU-IR issue #251 update
- `workspace/T32_tilelang_rescue/rka16_ns4_pass_index.txt` (311 lines) — line-number+pass-name index for the dump; required for re-running the bisect or quoting evidence
- `workspace/T32_tilelang_rescue/rka16_ns4.npuir` — tilelang-generated NS=4 input to bishengir; reproducer artifact
- `workspace/T32_tilelang_rescue/dump_rka16_ir.py` — monkey-patch driver that produced the dump
- `workspace/T32_tilelang_rescue/repro_rka16.py` — self-contained minimal repro at NS=1/2/4/8
- `workspace/T32_tilelang_rescue/DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md` — DSv4-Flash shape derivation + miles component classification table
- `workspace/T32_tilelang_rescue/UPSTREAM_ISSUE_RKA16.md` — AscendNPU-IR #251 issue body draft
- `workspace/T32_tilelang_rescue/A5_OPS_DESIGN_NOTES.md` — a5_ops design extraction report (1-day/1-week/1-month plan in §7)
- `workspace/T32_tilelang_rescue/UPSTREAM_PATCH_MAP.md` — **THE全景图**: enumerates every upstream project needing changes for miles-DSv4-RL-on-NPU end-to-end. Maintain in lockstep with ROADMAP T12/T13/T14 whenever a new upstream gap surfaces or an existing PR lands

**Prepared-but-unopened upstream PRs (DO NOT lose state on compact)**:

| Target | Branch on fork | Commit | Status | Gating |
|---|---|---|---|---|
| `radixark/miles` | `zhshgmail/miles npu-tilelang-ops` | `d03db2c` | audit-clean, T11 valid (MindSpeed unsupported at Mcore 0.16), PR body at `/tmp/miles_pr_body.md` | awaiting user `gh pr create -R radixark/miles --head zhshgmail:npu-tilelang-ops` |
| `radixark/Megatron-LM` (via miles vendored copy) | local `Megatron-LM-miles fix/te_general_gemm_npu_fallback` | `6f3209b` | 8-line guard, cold-import verified | bundle into miles PR set rather than separate Megatron PR (per user direction) |
| `Ascend/AscendNPU-IR` issue #251 | n/a (issue comment, not PR) | comment ID `1.73358592e+08` | comment POSTED (issue still OPEN, triage-review, no owner/ETA — "landed" only means the bisect report comment is posted, NOT that the issue is resolved or a fix merged) | Huawei compiler team owns the C++ patch |
| `tile-ai/tilelang-mlir-ascend` PR #80 | `zhshgmail/tilelang-mlir-ascend npuir-check-ub-budget` | `df7431e` | CI all green, MERGEABLE, awaits tile-ai maintainer review | external |
| `tile-ai/tilelang-mlir-ascend` PR #96 (NEW 2026-06-02) | `zhshgmail/tilelang-mlir-ascend fix/sparse-mla-heads-lt-block-h` | `479e2f4` | OPEN; sparse_mla_fwd heads<block_H silent-write-over-run fix (verified heads=4/8/16/32); builds green + MERGEABLE; `format` check fails on PRE-EXISTING file issues (whole-file not ruff-clean + pre-existing `automically` typo line 76 + pre-existing ruff-check), commented asking maintainer reformat-vs-minimal preference; gemini-code-assist review = no issues with the fix logic | external |
| `tile-ai/tilelang-mlir-ascend` issue #97 (NEW 2026-06-02) | n/a (issue) | n/a | OPEN; flash_attn_npuir wrong output in bf16 (correct fp16, 73.5% vs fp32 ref); confirmed+narrowed (workspace_1-fp32 doesn't fix; all-ws-fp32 hits copy-cast compile err); filed not fixed (deep numerical bug) | external |
| `Ascend/AscendNPU-IR` issue #253 (NEW 2026-06-02) | n/a (issue) | n/a | OPEN; ROOT CAUSE of #97 — HIVM vector-arith ops (vmul etc.) exclude bf16 (HIVMVectorOps.td:120 OperElemTypeConstraints [F16,F32]); bf16 kernels compile-fail or silently miscompute; 1-line repro (exp2->bf16). Huawei compiler repo, same as R-KA-16 #251 | external |
| `Ascend/AscendNPU-IR` PR #1199 (NEW 2026-06-02) | `zhengshencn_hwca/AscendNPU-IR blue/fix/hivm-vmul-vexp-bf16` | `9dee1c04` | OPEN; FIX for #253 — +2/-2 in HIVMVectorOps.td adds BF16 to VExpOp (L120) + VMulOp (L548) OperElemTypeConstraints. VERIFIED: both in-process (libtilelang*.so) + standalone bishengir-compile verifiers accept bf16 vmul/vexp after rebuild (3-layer trace: ①②PASS). NOT VERIFIED: bf16 vmul backend numerical correctness — blocked by closed deployed hivmc 0.1.0 (bakes pre-fix verifier, can't rebuild). PR body honestly flags this; asks maintainer to confirm on hivmc built w/ change. Layer-③ wall = closed hivmc; CANN 9.1.0-beta.1 (2026-05-15) exists but master STILL omits bf16 so upgrade not guaranteed sufficient. Root-cause doc: workspace/task-dag-realdelta/BF16_VMUL_ROOTCAUSE.md | external |
| `tile-ai/tilelang-mlir-ascend` issue #99 (NEW 2026-06-02) | n/a (issue, GitHub) | n/a | OPEN; indexer-bwd dk (index_k grad) = 8e33 garbage in FULL miles-indexer integration only; 8-hypothesis discriminator chain (index-order/inputs/repeat/sync/prior-bwd/fwd-exec/fwd-build/active-slot-progression) all isolate CLEAN → cross-kernel NPU memory-state contamination, not open-layer fixable. NOT a V4 blocker (V4 prod uses CANN-native npu_sparse_lightning_indexer_grad_kl_loss). Filed w/ full evidence chain for maintainer NPU-memory-trace. Doc: workspace/task-dag-realdelta/DK_GARBAGE_BUG.md | external |

If any of the above migrates / lands / closes, update this table AND `ROADMAP.md` T12 in the same commit. Never let this table go stale.

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
8. **A3 firewalled, but gitcode reachable**: do web/probe locally for non-CN hosts; clone Ascend / gitcode.com sources direct on A3. (Memory: `a3_is_firewalled.md`.)
9. **Before recommending a function/file/flag from memory**, verify it still exists (grep / ls). Memory snapshots can rot.
10. **Upstream PR discipline** (per user 2026-05-28 corrections):
    - 充分验证后再开 PR — install the production adaptor stack (e.g. MindSpeed, when supported), re-run the end-to-end driver against it, and only THEN write the patch and PR.
    - 尊重目标 repo 的 PR 格式 — read the target repo's CONTRIBUTING.md / pull_request_template.md, follow their label / format / language conventions.
    - 无 claude / cc 签名 — neither commits nor PR bodies carry agent trailers; follow the target project's standard.
    - Identify the correct PR target by version-coverage, not by guess. (Memory: `feedback_npu_megatron_via_mindspeed.md` shows the MindSpeed-vs-Mcore-0.16 case: even the "obvious" Ascend layer may not yet support the consumer's version.)

### How to extend this

When the user introduces a new "always do X" / "never do Y" rule:
1. Write a new `memory/feedback_<short_name>.md` with frontmatter `type: feedback`, body following the **Rule → Why → How to apply** structure
2. Add a one-line entry to `memory/MEMORY.md` linking to it
3. If the rule should bind permanently across all sessions, also add a numbered bullet to "Hard rules that bind across compactions" above
