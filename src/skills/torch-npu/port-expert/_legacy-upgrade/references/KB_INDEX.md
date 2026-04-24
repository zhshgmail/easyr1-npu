# torch-npu-upgrade-expert KB — Search Index

## Decision frameworks (load unconditionally at Phase A)

| File | What | When |
|---|---|---|
| [../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md](../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md) | Cross-expert OL rules | Phase A |
| [ALWAYS_LOADED_RULES.md](ALWAYS_LOADED_RULES.md) | This expert's OL-03 + OL-08 | Phase A |
| [patterns/domains/torch-stack-migration.md](patterns/domains/torch-stack-migration.md) | Canonical Dockerfile + NPU-BUG-001/003/004 workaround templates + per-version history | Phase B |

## Quick-lookup errors (deferred load at Phase D failure)

- **`ImportError: cannot import name 'Config' from 'triton'`** → NPU-BUG-001 (triton-ascend partial install; force-reinstall in Dockerfile)
- **`ImportError: cannot import name 'Language' from 'triton.backends.compiler'`** → NPU-BUG-004 (upstream triton amd/nvidia backend dirs present; prune in Dockerfile)
- **`TypeError: cannot pickle 'frame' object` inside inductor at log_probs** → NPU-BUG-003 (torch.compile regression); set `worker.actor.use_torch_compile=false` in smoke config — already in canonical V1.4 smoke template; if still hit, verify setting propagated
- **`RuntimeError: CANN version mismatch`** → torch_npu wheel doesn't match image's CANN; target image is misconfigured, escalate to user (not fixable in this expert)
- **V1.1 fails with chip 0 busy** → OL-05 violation (someone else on chip); abort, don't retry

## Related KB from sibling experts

- `easyr1/port-expert/references/PLATFORM_BUGS.md` — NPU-BUG-001..004 body (identical content; this expert's patterns/domains/ references the same)
- `easyr1/port-expert/references/SMOKE_BASELINE.md` — per-image V1.1/V1.3/V1.4 baselines; this expert uses the target-image row for V1.4 assertion
- `transformers/port-expert/_legacy-upgrade/references/patterns/domains/dockerfile-target.md` — sibling Dockerfile template; very similar structure but targets the full stack swap, not just torch

## Verification protocol (mechanical)

1. Phase A probes (all inside target image, via one-shot `docker run`):
   ```bash
   docker run --rm $TARGET_IMAGE python3 -c 'import torch, torch_npu, triton; print(torch.__version__, torch_npu.__version__, triton.__version__)'
   # Expected: all three resolve. If ImportError → NPU-BUG-001 or 004; see catalog.
   ```
2. Probe backend pollution:
   ```bash
   docker run --rm $TARGET_IMAGE python3 -c '
   import importlib.util, os
   spec = importlib.util.find_spec("triton")
   print("amd:", os.path.isdir(os.path.join(os.path.dirname(spec.origin),"backends/amd")))
   print("nvidia:", os.path.isdir(os.path.join(os.path.dirname(spec.origin),"backends/nvidia")))
   '
   # If either prints True → need NPU-BUG-004 prune block in Dockerfile
   ```
3. Phase B: only write the Dockerfile blocks for NPU-BUGs that probes said
   are needed. Don't add boilerplate speculatively.
4. Phase C: build, then run the verification in the catalog.
5. Phase D: V1.1 → V1.3 → V1.4 in that order. V1.1 is the kill-switch — if
   it fails, stop and debug (NPU-BUG-001/004 recur).
