# easyr1-npu — project context for Claude Code

## What this project is

Port **EasyR1 (master tip, April 2026)** to **Ascend 910C (A3)** NPU.

EasyR1 is a slimmed-down fork of veRL. veRL has already been ported to NPU (see `quay.io/ascend/verl:verl-8.5.*-a3-*` images). Our job is to derive the NPU dependency set for EasyR1 master by:

1. Comparing EasyR1 source vs. veRL source (what does EasyR1 need that veRL didn't?).
2. Inspecting the verl-A3 docker images to see which versions of `torch_npu`, `vllm-ascend`, `triton-ascend`, `transformers`, CANN etc. actually work on NPU today.
3. Identifying gaps — e.g. EasyR1 may need a newer `transformers` than `torch_npu` currently supports. Those become porting tasks.

**Deliverables are two things**, not just one:
- A working EasyR1 on A3 (rollout + RL training).
- A reusable harness — skills, knowledge docs, scripts, workflows — that automates the same porting work for future EasyR1 versions or adjacent RL stacks.

## Current state (2026-04-17)

- No A3 hardware access yet; we're on an H100 Ubuntu box (`115.190.166.102`, x86 CPU for these images — NPU images can't execute here, only be inspected).
- Deferred A3-hardware work goes on a blocked list in `docs/_meta/design.md`.
- See `docs/_meta/design.md` for requirements, task decomposition, and status.

## Repository layout

```
~/workspace/easyr1-npu/
├── repo/                              # our git-tracked deliverable
│                                      # (github.com/zhshgmail/easyr1-npu)
│   ├── README.md                      # entry point — start here
│   ├── ONBOARDING.md                  # one-page customer quickstart (v1 + v2 paths)
│   ├── CLAUDE.md                      # this file
│   ├── docs/
│   │   ├── easyr1/
│   │   │   ├── PORT-GUIDE.md              # v1 path — "EasyR1 on verl-8.5.0 + ascend-port"
│   │   │   ├── PORT-GUIDE-v2-integrated.md# v2 integrated overlay path
│   │   │   ├── PORT-SUMMARY.md            # high-level port story
│   │   │   ├── DELIVERABLE.md             # deliverable spec
│   │   │   ├── dep-matrix.md              # GPU↔NPU dep mapping
│   │   │   └── porting-journal.md         # dated log of findings
│   │   ├── _meta/
│   │   │   ├── UPSTREAM_FORKS.md          # authoritative fork+branch ledger
│   │   │   ├── HANDOVER.md                # current state + open work
│   │   │   ├── SKILLS-USAGE.md            # slash-command usage for upstream maintainers
│   │   │   ├── SKILLS-GUIDE.md            # redo-the-port-from-zero guide
│   │   │   ├── design.md                  # requirements + task decomp + status
│   │   │   ├── DOCS-CONVENTION.md         # where each kind of info lives
│   │   │   ├── RL_INTEGRATION_PLAN.md     # T22 integration log
│   │   │   └── kb/porting_lessons/        # cross-layer lessons
│   │   ├── vllm-ascend/PORTING-GUIDE.md
│   │   ├── torch-npu/PORTING-GUIDE.md
│   │   ├── transformers/                  # PR_MATERIAL + drill status
│   │   └── triton-ascend/                 # work plans
│   ├── src/
│   │   ├── skills/                    # reusable porting skills (CC skill format)
│   │   │   ├── _shared/               # shared workflows + patterns + small helpers
│   │   │   ├── vllm-ascend/port-expert/      # /vllm-ascend-day0
│   │   │   ├── torch-npu/port-expert/        # /torch-npu-day0
│   │   │   ├── transformers/port-expert/     # /transformers-day0
│   │   │   ├── triton-ascend/port-expert/    # /triton-ascend-port
│   │   │   ├── easyr1/port-expert/           # /easyr1-port
│   │   │   ├── dep-analysis/expert/          # /dep-analysis
│   │   │   └── orchestrators/npu-port/       # /npu-port
│   │   └── scripts/                   # install-skills.sh, run-npu-container.sh, smoke harness
│   └── knowledge/
│       ├── npu-patterns.md            # NPU-CP/BUG/ENV/OPS pattern catalogue
│       ├── upstream-refs.md           # which upstream tag matches which NPU image
│       └── images/                    # extracted facts from verl-A3 images
└── upstream/                          # each subdir is its own git clone, own branch
    ├── EasyR1/                        # github.com/hiyouga/EasyR1           (April tip)
    ├── verl/                          # github.com/verl-project/verl        (April tip, GPU ref)
    ├── torch-npu/                     # gitcode.com/Ascend/pytorch          (torch_npu source)
    ├── vllm-ascend/                   # github.com/vllm-project/vllm-ascend
    ├── triton-ascend/                 # gitcode.com/Ascend/triton-ascend
    └── transformers/                  # github.com/huggingface/transformers (reference for NPU port status)
```

**Ground rule on upstream edits:** work on a branch (`ascend-port`) in the relevant upstream repo. Don't maintain patch files as the dev format — use real git. Export patches only as an end-of-work artifact if needed.

## Docker images (on this host)

- `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5` (24GB, newer, primary reference)
- `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest` (14GB, base reference)

These are A3 images — they won't **execute** on x86, but their filesystem is inspectable via `docker create` + `docker cp` / `docker export`, which gives us pip freeze, site-packages contents, installed apt packages, environment setup. That's what drives the dep-matrix.

CANN source is **not** in `upstream/` — it lives at `gitcode.com/cann`. Pull on demand.

## Working preferences

- **Docs organization is governed by `docs/_meta/DOCS-CONVENTION.md`.** Every kind of information (bugs, status, decisions, plans, kb updates, skill changes, version bumps) has a single authoritative file. README is **index only, never content sink**. Read DOCS-CONVENTION at session start; don't reinvent the place-of-record each time.
- **README must be 2-hop-reachable to every important doc.** If you add a doc, wire it into README (either directly or via an index doc like HANDOVER / DOCS-CONVENTION).
- **Mirror milestones and waiting-for-input responses to Discord.** User is primarily reachable via Discord, not the terminal. Use `mcp__plugin_discord_discord__reply` with chat_id `1494825170399924366`.
- **Commit messages:** no Claude-related text.
- **Design doc format:** formal (requirements, background, restrictions, high/detailed design). For now, requirements + task decomposition + status are the populated sections; detailed design stays TBD until dep-matrix is done.

## Tools available

- `gh` (GitHub, logged in) — use for any GitHub repo actions.
- `gc` (GitCode, logged in) — use for Ascend/* NPU repos. **Primary NPU repos live on gitcode, not github. GitHub mirrors may lag.**
- `docker` — for image inspection.
- No A3 hardware.
