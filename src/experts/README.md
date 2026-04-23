# easyr1-npu expert skills — directory

Each subdirectory is a Claude Code skill (an "expert") for a specific
NPU-porting scenario. This README maps scenarios → experts so a new
session can pick the right entry point.

## Decision tree

```
"I want to run X (a community release) on NPU."
│
├─ X is transformers ─ consumers commonly bump it
│   ├─ NPU ecosystem already ships X in a base image?
│   │   ├─ yes → /transformers-upgrade
│   │   └─ no  → /transformers-day0
│   │
├─ X is vllm ─ consumers' rollout path
│   ├─ vllm-ascend already released matching version?
│   │   ├─ yes → /vllm-upgrade
│   │   └─ no  → /vllm-day0
│   │        │
│   │        (if vllm-day0 discovers the problem is actually in vllm-ascend
│   │         C++/python source vs new torch, route next to:)
│   │        → /vllm-ascend-day0
│
├─ X is torch / torch_npu ─ stack foundation
│   ├─ stable torch_npu already ships X?
│   │   ├─ yes → /torch-npu-upgrade
│   │   └─ no  → /torch-day0
│   │        │
│   │        (after torch-day0 deploys a base image, downstream C++
│   │         extensions (vllm-ascend, triton-ascend) may need their own
│   │         Day-0 patch:)
│   │        → /vllm-ascend-day0 (+ future /triton-ascend-day0)
│
└─ X is the consumer framework itself (EasyR1 master, custom RL stack)
    → /npu-port  (orchestrates dep-analysis + routes to sub-experts)
```

## Experts (by stage)

### Stage 0 — base porting

- **[easyr1-expert/](easyr1-expert/)** — reference porting expert for
  EasyR1 itself on NPU (original Stage 0 skill)

### Stage 1 — dependency analysis

- **[dep-analysis-expert/](dep-analysis-expert/)** — analyzes consumer's
  `requirements.txt` against NPU base image + shipped NPU-ecosystem
  versions, classifies each dep (A/B/C/D/E), and routes blockers to
  the right upgrade/day0 expert

### Stage 2 — shim-adapt (NPU has, but different version)

When the NPU ecosystem **has shipped** the target version in an image:

- **[transformers/upgrade-expert/](transformers/upgrade-expert/)** —
  transformers version in NPU image but consumer needs shim work
- **[vllm/upgrade-expert/](vllm/upgrade-expert/)** — vllm-ascend version
  bumped but consumer needs shim work
- **[torch-npu/upgrade-expert/](torch-npu/upgrade-expert/)** — torch_npu
  bumped within same transformers/vllm window

### Stage 3 — Day-0 (community has, NPU hasn't caught up)

When the NPU ecosystem **has NOT shipped** the target version:

- **[transformers/day0-expert/](transformers/day0-expert/)** — community
  transformers version with NPU shim needed
- **[vllm/day0-expert/](vllm/day0-expert/)** — community vllm whose
  matching vllm-ascend isn't released
- **[torch-npu/day0-expert/](torch-npu/day0-expert/)** — community PyTorch whose
  stable torch_npu isn't released (torch_npu rc on PyPI used)
- **[vllm-ascend/day0-expert/](vllm-ascend/day0-expert/)** — vllm-ascend
  itself needs patching for a deeper upstream move (new torch ABI drift,
  new vllm symbol removal, etc.). Usually chained downstream of
  torch-day0 or transformers-day0.

### Orchestration

- **[../commands/npu-port.md](../commands/npu-port.md)** — orchestrator
  command that runs dep-analysis then routes to the right upgrade or
  day0 expert per dep

## Shared infrastructure

- **[_shared/](_shared/)** — common references, scripts, hooks reused
  across experts:
  - `references/ALWAYS_LOADED_UNIVERSAL.md` — OL rules every worker
    reads in Phase A (OL-01 thru OL-12)
  - `references/patterns/domains/day0-deploy-artifacts.md` — 5
    deploy-artifact deliverables mandatory for Day-0 sessions with
    outcome A / C-patch
  - `scripts/` — reference copies of `static_check.py` etc. that
    experts fork and specialize

## Outcome C-patch — handoff to upstream maintainer

**Audience reminder**: these skills are authored for the
Huawei-owned upstream repo's maintainers (vllm-ascend / torch_npu /
triton-ascend / transformers NPU integrations), not for us to push
our own PR. When a Day-0 expert produces outcome **C-patch**:

1. Worker opens local branch `ascend-day0-<delta>-<SESSION>` on the
   relevant `upstream/<huawei-repo>/` checkout — **session-local trace
   only**, so the fix is reproducible in the session workspace
2. Worker commits minimal patch with `[BugFix]` / `[Feature]` prefix
   to that trace branch
3. Optional: worker pushes the trace branch to a mirror fork the
   session operator has push rights on (e.g. `zhshgmail/<repo>`),
   purely for traceability; this is **not the authoritative handoff**
4. **Authoritative deliverable**: `PR_MATERIAL.md` + reference
   `.py.patched` files in the session workspace, written so the
   actual upstream maintainer can cherry-pick the diff into their
   own tree (where they own the upstream push rights and CI)
5. Worker emits handoff JSON pointing at `pr_material_path`; delivery
   to the maintainer is off-session (email / issue / existing
   relationship channel — outside skill scope)

Enabled Huawei-owned targets for C-patch:
- `github.com/vllm-project/vllm-ascend`
- `gitcode.com/Ascend/pytorch` (torch_npu)
- `gitcode.com/Ascend/triton-ascend`
- `github.com/huggingface/transformers` (only files under
  `src/transformers/integrations/npu_*`; community transformers core
  stays off-limits)

**NOT in scope** (C-report only — we file issues, don't patch):
- `github.com/vllm-project/vllm` (community vllm core)
- `github.com/pytorch/pytorch` (community PyTorch)
- `github.com/huggingface/transformers` (non-NPU files)

## 2026-04-23 validated combinations

| Stack | Transformers | vllm | vllm-ascend | torch | torch_npu | CANN | V1.3 rollout | V1.4 training |
|---|---|---|---|---|---|---|---|---|
| v1 | 4.57 | 0.13.0 | 0.13.1.dev18 | 2.7.x | 2.7.x | 8.4.0 | (historic) | (historic) |
| v2 | 5.3.0.dev0 | 0.18.0 | 0.17.0rc2.dev109 | 2.9.0 | 2.9.0 | 8.5.1 | PASS | PASS |
| v2 + torch 2.11 overlay (Fix B+ only) | 5.3.0.dev0 | 0.18.0 | 0.17.0rc2.dev109 + Fix B+ (3 commits) | **2.11.0+cpu** | **2.11.0rc1** | 8.5.1 | **PASS** | progresses to `Update policy 25%`, then **FAIL** on `aten::linear_backward` CPU fallback |
| v2 + torch 2.11 overlay (Fix B+ + Fix C) | 5.3.0.dev0 | 0.18.0 | 0.17.0rc2.dev109 + Fix B+ + Fix C rebuild (4 commits) | **2.11.0+cpu** | **2.11.0rc1** | 8.5.1 | **PASS** | **PASS** entropy_loss=1.275 (exact v2 baseline match) |
| v2 + transformers 5.6.0 overlay | **5.6.0** | 0.18.0 | 0.17.0rc2.dev109 | 2.9.0 | 2.9.0 | 8.5.1 | PASS | PASS (step1 entropy_loss 1.310 in band) |

**Note on torch 2.11 overlay V1.4 FAIL**: Fix B+ forces batch-invariant
mode to bypass broken `_C_ascend` custom ops; batch-invariant's Triton
`linear_batch_invariant` kernel asserts 2D input. Training passes 3D
`[batch, seq, hidden]` to `F.linear` → AssertionError.

Long-term fixes (tracked):
- Fix C (task #77): rebuild `vllm_ascend_C.so` against torch 2.11 —
  eliminates need for batch-invariant fallback
- vllm-ascend `linear_batch_invariant` patch to reshape 3D → 2D →
  restore (see vllm-ascend/day0-expert KB_INDEX Option 1)

## See also

- Project root: [../../../README.md](../../../README.md)
- HANDOVER: [../../../docs/HANDOVER.md](../../../docs/HANDOVER.md)
- User-facing examples:
  - [../../../docs/examples/transformers-5.6.0-day0.md](../../../docs/examples/transformers-5.6.0-day0.md)
  - [../../../docs/examples/torch-2.11-day0.md](../../../docs/examples/torch-2.11-day0.md)
- Design doc: [../../../docs/design/SKILLS_ARCH_TARGET.md](../../../docs/design/SKILLS_ARCH_TARGET.md)
