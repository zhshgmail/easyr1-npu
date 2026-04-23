# NPU ecosystem package map — classification rules (A/B/C/D/E)

> **This is the authoritative rule table** for dep-analysis-worker. It is
> NOT an answer key — classification still runs mechanically through
> `scripts/dep-gap-detect.sh` + this table. The table encodes **rules**
> (why each package is classified), not per-commit verdicts.

## How the agent uses this

1. For each dep in consumer's `requirements.txt`, check if it appears in
   `PACKAGE_RULES` below. If yes → use the rule's classification + its
   rationale.
2. If not in rules but present in `image-freeze.txt` (candidate image)
   → provisionally classify **E** (pure Python) unless the package name
   matches a known CUDA-coupled pattern.
3. If not in rules and missing from image → provisionally **D**
   (blocker). Worker must grep consumer's source for the package import
   to confirm it's actually used; a declared-but-unused dep is a false D.
4. Any D must name the upgrade-expert that could fix it (or mark
   `unsupported`).

---

## PACKAGE_RULES table

### Deep-learning stack

| Package | Class | Rule / rationale | Upgrade-expert when blocker |
|---|---|---|---|
| `torch` | A | torch_npu is a drop-in adjoint; import-compat maintained by Ascend | base-image-upgrade |
| `torch_npu` | A | NPU-native; usually comes from base image, not user-installed | base-image-upgrade |
| `transformers` | A (≤4.57) / shim-needed (≥5.0, C) | transformers ≥5 moved `no_init_weights`; small shim in consumer per EC-02 | transformers-upgrade (for ≥5 bump) |
| `vllm` | B | Use `vllm-ascend` fork; don't install upstream vllm on NPU | base-image-upgrade |
| `vllm-ascend` | A | NPU-native by design; from base image | base-image-upgrade |
| `accelerate` | A | Backend-agnostic; walks `torch_npu` path if present | — |
| `peft` | A | transformers-adjacent; no CUDA hardcoded | — |
| `datasets` | E | CPU/disk only | — |
| `tokenizers` | E | CPU only | — |
| `safetensors` | E | CPU/disk only | — |

### Attention / kernel

| Package | Class | Rule / rationale | Upgrade-expert when blocker |
|---|---|---|---|
| `flash-attn` | C | CUDA-only. Bypass via `transformers.integrations.npu_flash_attention` (NPU-CP-005). No install on NPU. | — (always C) |
| `liger-kernel` | C | CUDA-only Triton kernels. Bypass: don't enable kernel fusion path on NPU. | — (always C) |
| `triton` | N/A | Don't install upstream triton on NPU — it conflicts with triton-ascend (NPU-BUG-004). If seen in reqs → strip and use triton-ascend from base image | — |
| `triton-ascend` | A | NPU-native; comes from base image. Repair via Dockerfile force-reinstall (NPU-BUG-001) | base-image-upgrade |

### Ray / training infra

| Package | Class | Rule / rationale | Upgrade-expert when blocker |
|---|---|---|---|
| `ray` | A (with Python shim) | NPU needs `resources={"NPU":N}` scheme registration; a few Python-layer shims in consumer (NPU-CP-003). Classifies A because ray itself runs; C for the shim work. | — |
| `tensordict` | E | CPU; torch-adjacent but no CUDA specifics | — |
| `torchdata` | E | CPU | — |

### Data / numeric

| Package | Class | Rule / rationale |
|---|---|---|
| `numpy` | E | CPU. Pin numpy<2 per NPU stack convention |
| `pandas` | E | CPU |
| `pyarrow` | E | CPU |
| `pillow` | E | CPU image proc |
| `codetiming` | E | Pure Python utility |
| `mathruler` | E | Consumer-owned util, pure Python |
| `omegaconf` | E | Pure Python config |

### Metric/observability

| Package | Class | Rule / rationale |
|---|---|---|
| `tensorboard` | E | CPU-side logger |
| `wandb` | E | Network; CPU |
| `swanlab` | E | Same |
| `mlflow` | E | Same |

---

## Classification decision tree

```
Is <pkg> in PACKAGE_RULES?
├── yes → use the table's (Class, rationale). Done.
└── no →
     Is <pkg> in image-freeze.txt?
     ├── yes →
     │   Does <pkg> name match a CUDA-coupled pattern
     │   (contains 'cuda'|'cudnn'|'flash'|'nvidia')?
     │   ├── yes → C (needs shim) with note "new CUDA-coupled dep; inspect usage"
     │   └── no  → E (provisional; note "not in rules, presumed Python/CPU")
     └── no →
         Is <pkg> imported in consumer source?
         (grep 'import <pkg>' or 'from <pkg>' in consumer's *.py)
         ├── yes → D (provisional blocker, not on NPU today)
         │         → name the upgrade-expert candidate in task plan
         └── no  → E-unused (declared-but-not-imported; orchestrator
                             can log a warning but doesn't block)
```

## Upgrade-expert routing

If D is encountered, check which expert owns it. **First decide
shim-adapt vs day-0** (see "Shim-adapt vs Day-0 decision" below), then
among shim-adapt or day-0, pick the narrowest dep-specific expert.

### Shim-adapt vs Day-0 decision (FIRST BRANCH)

- **Shim-adapt path**: the consumer wants a transformers / vllm /
  torch_npu version that IS SHIPPED in a known NPU base image (e.g.
  `verl-8.5.0-a3` = transformers 4.57 / `verl-8.5.2-a3` = transformers
  5.3.0.dev0 / etc.). Route to the Stage-2 upgrade-experts.
- **Day-0 path**: the consumer wants a transformers version that is
  NOT SHIPPED in any known NPU image. Typically when the community has
  released a newer version than the NPU ecosystem currently ships.
  Route to a Stage-3 day0-expert.

Probe this by grep'ing the version against `knowledge/images/*.md`
pip-freeze sections. If present in any image → shim-adapt. If not →
day-0.

### Shim-adapt routing (Stage 2)

1. **Whole-stack swap** (base-image family change — moves transformers +
   vllm + torch_npu + CANN together, like v1→v2 historical drill) →
   `transformers-upgrade-expert`. Triggers: consumer wants a new NPU
   base image family; OR two+ of {transformers, vllm, torch_npu} need
   to move together.
2. **vllm-only bump** within a fixed base-image family (e.g. vllm 0.13→0.14
   on same CANN+transformers+torch_npu) → `vllm-upgrade-expert`.
   Triggers: consumer req has a conflict ONLY on `vllm` or `vllm-ascend`
   while other deps stay put; or user asks to pick up a vllm-ascend
   release for CVE/bugfix.
3. **torch-stack-only bump** within a fixed transformers/vllm world
   (e.g. torch_npu 2.8→2.9 on same transformers/vllm) →
   `torch-npu-upgrade-expert`. Triggers: consumer req has a conflict
   ONLY on `torch` / `torch_npu` / `triton-ascend` / `torchdata`;
   new base image shipping upgraded torch but keeping transformers.
4. **New Ascend-ecosystem package** that no existing expert covers
   (e.g. future `ascend-flash-attn-standalone`, `cann-training-runtime`,
   etc.) → mark `unsupported: needs new expert`. Orchestrator surfaces
   to user for decision (build new expert or drop the dep).
5. **CUDA-only library with no NPU analog** → mark
   `unsupported: cuda-only, no NPU port available`; orchestrator
   surfaces to user (typical resolution: make the dep optional via
   [gpu] extras).

### Day-0 routing (Stage 3)

For versions not in any NPU image:

1. **transformers target not in any image** → `transformers-day0-expert`.
   Probe the new version via pip overlay on v2 image; emit A/B/C
   outcome (works-as-is / forward-port / blocked). Hardest scenario
   per user 2026-04-23; try this first.
2. **vllm / vllm-ascend target not shipped** → `vllm-day0-expert`
   (Stage 3, built 2026-04-23). Typical case: community vllm ahead of
   vllm-ascend by 1-2 minor. Expert probes whether existing vllm-ascend
   plugin still loads against the new vllm (often YES due to
   plugin-architecture version tolerance) + emits A/B/C outcome.
3. **torch / torch_npu target not shipped** → `torch-day0-expert`
   (Stage 3, not yet built). Usually easiest case (NPU tracks torch
   closely) but still day-0 if genuinely unshipped.

### Rule for "which single expert" on ambiguous multi-dep D

If D contains e.g. `{transformers>=5, vllm>=0.18}` — both part of the
historical v1→v2 family — prefer **`transformers-upgrade-expert`**
(atomic swap). Don't chain two single-dep experts when one atomic
expert covers both; the atomic expert already carries the cross-dep
wiring (e.g. vllm-ascend matching transformers version).

If D contains e.g. `{vllm>=0.20}` alone but transformers stays put,
prefer `vllm-upgrade-expert` — shorter, doesn't disturb other deps.

## Maintenance

Update this table when:
- A new package joins NPU ecosystem (move from D/C → A/B).
- A new upgrade-expert is added (add a row to "Upgrade-expert when blocker").
- A version bump changes the classification (e.g. transformers 6.x introduces
  a real API incompatibility → C or D row for that version range).

Always cite evidence (drill report, commit, test run) in the rule's
rationale column. Rules without evidence are assumptions.
