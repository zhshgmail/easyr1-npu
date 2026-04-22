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

If D is encountered:

1. Is it a `transformers` / `vllm` / `torch_npu` / CANN version gap? →
   route to `transformers-upgrade-expert` (its drill covered this family).
2. Is it a new Ascend-ecosystem package? → route to a future
   `ascend-ecosystem-upgrade-expert` (doesn't exist yet; mark
   `unsupported: needs new expert`).
3. Is it a CUDA-only library with no NPU analog? → mark
   `unsupported: cuda-only, no NPU port available`; orchestrator surfaces
   to user (they may choose to remove the dep from consumer).

## Maintenance

Update this table when:
- A new package joins NPU ecosystem (move from D/C → A/B).
- A new upgrade-expert is added (add a row to "Upgrade-expert when blocker").
- A version bump changes the classification (e.g. transformers 6.x introduces
  a real API incompatibility → C or D row for that version range).

Always cite evidence (drill report, commit, test run) in the rule's
rationale column. Rules without evidence are assumptions.
