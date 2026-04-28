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
