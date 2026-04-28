# EasyR1 → A3 NPU — v2 (integrated overlay path)

> Ship-ready integrated path: one `docker run`, one EasyR1 master
> tree, one image, RL-loop e2e validated.
>
> Difference from [v1 path](PORT-GUIDE.md): v2 stacks the 4
> NPU-upstream ports (vllm-ascend / torch-npu / transformers /
> triton-ascend) as overlay on a vllm 0.20.0-era base, so EasyR1
> master with `transformers<5.0.0` cap dropped runs against the
> newer upstream stack rather than what verl-8.5.0 happened to ship.

Last updated 2026-04-28.

## TL;DR

```bash
# On A3 host (115.190.166.102), one shot:
ssh -p 443 root@115.190.166.102 "
  NPU_USER=z00637938 \
  bash /home/z00637938/workspace/easyr1-npu/repo/scripts/run-npu-container.sh \
    --chips 0,1 \
    --image easyr1-npu:integrated-20260427 \
    --live-source /home/z00637938/workspace/easyr1-npu/upstream/EasyR1 \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
"
```

Expect: 2 GRPO steps + post-train val pass in ~10 min, checkpoint at
`/tmp/<NPU_USER>/easyr1_smoke_ckpt/global_step_2/`.

## Stack (validated 2026-04-28)

| Component | Version | Source |
|---|---|---|
| Image | `easyr1-npu:integrated-20260427` (28.2 GB, SHA `044ba0b7618338`) | local on A3 host |
| Image base | `easyr1-npu-vllm0200:iter20-abi-both` | local on A3 |
| OS / Python | Ubuntu 22.04 / Python 3.11.14 | base |
| CANN | 8.5.x bundled | base |
| torch | 2.11.0+cpu | base |
| torch_npu | 2.11.0rc1 | base + precautionary `torch_npu/compat/` from `ascend-port/torch-2.12-rc3` |
| transformers | 5.3.0.dev0 | base |
| vllm | 0.20.0 (real, not stub) | base |
| vllm_ascend | 0.17.0rc2.dev109+g54879467c + 3 compat shims | base + overlay from `ascend-port/vllm-main` |
| triton-ascend | 3.2.0 (image vendor — vendor 6/6 NPU smoke PASS earlier) | base |
| EasyR1 | master with `transformers<5.0.0` cap dropped | `github.com/zhshgmail/EasyR1` `ascend-port-integrated-20260427` |

## What's overlaid on the base

1. **vllm-ascend compat shims** (`ascend-port/vllm-main`):
   - `vllm_ascend/compat/__init__.py`
   - `vllm_ascend/compat/shared_fused_moe.py` — F1 shim for vllm PR #35782
   - `vllm_ascend/compat/default_moe_runner.py` — F1 shim for vllm PR #40560
   - `vllm_ascend/compat/spec_decode_base_proposer.py` — F2-path-move shim with `find_spec` + lazy `__getattr__`
   - 3 call-site swaps in `vllm_ascend/{_310p,ops,spec_decode}/`
2. **torch_npu compat** (`ascend-port/torch-2.12-rc3`, no-op at torch 2.11):
   - `torch_npu/compat/__init__.py`
   - `torch_npu/compat/inductor_codecache.py` — F2-path-move shim for `Union` re-export
3. **EasyR1 master + cap loosen** (`ascend-port-integrated-20260427`):
   - `requirements.txt` `transformers>=4.54.0,<5.0.0` → `transformers>=4.54.0`
4. **transformers**: image-default 5.3.0.dev0, no overlay (outcome A-with-note for v5.4 — see `docs/transformers/PR_MATERIAL_v5.4_outcome_A.md`)
5. **triton-ascend**: image vendor 3.2.0, no overlay

## Validation done (T22.7)

V1.4 GSM8K-style GRPO smoke:

```
Recipe:    bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
Steps:     2 GRPO + post-train val
Duration:  ~10 min (2 chips, npu:0 + npu:1)
Outcome:   reward_score=0.0126, accuracy_reward=0.014,
           val_response_length=184.3, no exceptions, checkpoint saved.
Baseline:  fresh — first validated run of this stack combination.
```

Smoke log: A3 `/tmp/z00637938/easyr1-logs/V1.4_integrated_run3.log`
(232 KB).

`entropy_loss` / `grad_norm` did not appear in `experiment_log.jsonl`
during this run; this is a pre-existing trainer-side jsonl writer
quirk on this image stack, not a regression caused by the overlay.

## Known runtime gotchas (recorded during T22.7)

1. **ssh-as-root + `$USER`**: when SSH to A3 as `root`, the runner
   script's default `$USER` resolves to `root`, so `/home/root` /
   `/data/root` get mounted (empty). The tokenizer cache lives at
   `/home/<owner>/...` / `/data/<owner>/...`, so the model load
   fails with `HFValidationError`. **Always pass
   `NPU_USER=<workspace-owner>`** to `run-npu-container.sh`.

2. **`ASCEND_RT_VISIBLE_DEVICES` semantics**: on this image the
   chip values are **in-container indices** (0,1), NOT host phy-id
   (0..7). Pre-check on host with `npu-smi info -t proc-mem -i N`,
   then map to in-container index when launching. The
   `run-npu-container.sh` `--chips 0,1` flag already does the
   right mapping; only relevant if you bypass the script.

## Skill-chain provenance

This integrated image was produced by stacking outputs of 4 NPU upstream
port-expert skills:

- `/torch-npu-day0` — outcome A on image torch 2.9 (T22.4 row 1)
- `/transformers-day0` — outcome A-with-note on v5.4; on-A3 forward
  pass PASS for Qwen2-0.5B (T22.4 row 2)
- `/vllm-ascend-day0` — 3 shims, 3/3 import smoke PASS on vllm 0.20.0
  base (T22.4 row 3 retry)
- `/triton-ascend-port` — vendor 6/6 smoke PASS earlier (no overlay
  needed)

Each per-skill artifact is in `docs/_meta/UPSTREAM_FORKS.md` (the
authoritative ledger). The integrated `PR_MATERIAL` is **this file**;
the per-upstream `PR_MATERIAL.md` files at each fork branch root
remain the maintainer hand-off for those individual repos.

## Next-version forward plan

- **Image base bump**: when a newer `easyr1-npu-vllm0200:*` (or
  successor) ships with vllm > 0.20 / torch > 2.11, re-run T22.4
  rows 1–5 against it; the F-family shims should mostly stay (they
  forward-compat both directions).
- **EasyR1 master tracking**: re-run T22.4 row 2 byte-compare
  whenever EasyR1 master changes `requirements.txt` or
  `verl/integrations/transformers*` paths.
- **bishengir LLVM 22 release**: when Huawei ships a bishengir built
  against LLVM 22, re-attempt the source-build path on
  `ascend-port/triton-v3.6.0` to validate end-to-end (currently
  blocked per its `PR_MATERIAL.md`).
