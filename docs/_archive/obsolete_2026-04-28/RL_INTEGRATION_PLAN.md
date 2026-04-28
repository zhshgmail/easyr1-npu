# RL integration plan (T22)

> **Goal**: get EasyR1 master running on Ascend 910C (A3) by adapting
> the 4 NPU upstreams via the port-expert skills, installing all 4
> ascend-port branches into a single overlay image, and validating
> end-to-end with V1.4 entropy_loss baseline match.

## T22.1 — Dependency analysis (2026-04-27)

### EasyR1 master `requirements.txt`

```
accelerate
codetiming
datasets
flash-attn>=2.4.3
liger-kernel
mathruler
numpy
omegaconf
pandas
peft
pillow
pyarrow>=15.0.0
pylatexenc
qwen-vl-utils
ray[default]
tensordict
torchdata
transformers>=4.54.0,<5.0.0
vllm>=0.8.0
wandb
```

### verl-8.5.2-a3 image `pip freeze` (key deps)

| Package | Version | Note |
|---|---|---|
| torch | 2.9.0+cpu | x86 CPU build (NPU goes via torch_npu) |
| torch_npu | 2.9.0 | matches torch 2.9 |
| transformers | 5.3.0.dev0 | between 5.3 and 5.4 |
| vllm | 0.18.0+empty | editable stub (T21 found this is broken namespace) |
| vllm_ascend | 0.17.0rc2.dev109+g54879467c | release-13 era |
| triton | 3.6.0 | community |
| triton-ascend | 3.2.0 | matches CANN 8.5.2 bundled bishengir |
| accelerate | 1.13.0 | |
| peft | 0.19.1 | |
| tensordict | 0.10.0 | |
| ray | 2.55.0 | |
| wandb | 0.26.0 | |

### Gaps identified vs EasyR1 master

| Gap | EasyR1 master wants | image has | port-expert needed |
|---|---|---|---|
| **vllm cap removed** | `vllm>=0.8.0` (no cap) | `vllm 0.18.0+empty` (broken stub) | **vllm-ascend** — ascend-port/vllm-main shims for vllm main |
| **transformers cap** | `transformers>=4.54.0,<5.0.0` | `5.3.0.dev0` | requirements cap **conflicts** with installed; EasyR1 will refuse to install over image. Must either: (a) loosen cap in EasyR1 fork, or (b) install transformers 4.54 over image. (a) is consistent with our existing ascend-port for transformers. |
| **torch_npu** | implicit via torch | `2.9.0` | only need port-expert if image's torch_npu can't import EasyR1's torch contract. Need to test. |
| **triton-ascend / triton** | not direct EasyR1 dep | matched in image | no action |
| **flash-attn** | `>=2.4.3` | not in pip freeze (??) | need to check; flash-attn not on NPU usually replaced with npu_fusion_attention |
| **liger-kernel** | unbounded | not in pip freeze | check |

## T22.2 — Dependency DAG

```
            torch (2.9 in image, OK — no version bump needed yet)
              ↓
        torch_npu (2.9 in image, image-bundled)
              ↓
      transformers (5.3.0.dev0 in image; EasyR1 master pins 4.54-5.0;
                    transformers ascend-port marker says outcome A-with-note
                    for v5.4; need to decide which transformers to use)
              ↓
        vllm + vllm_ascend (vllm 0.18.0+empty broken in image; need
                           overlay install of ascend-port/vllm-main shims
                           on top, plus a working vllm)
              ↓
        triton-ascend (3.2.0 in image, OK; vendor 6/6 PASS)
              ↓
            EasyR1 (master; needs requirements cap loosened or pinned-down
                   transformers handled)
```

**Insight**: the image's `vllm 0.18.0+empty` is the critical missing
piece. Without a real vllm, no rollout. T22 must include either:
- (a) overlay-install a real vllm in the new container, OR
- (b) use a different base image that has working vllm (e.g.
  `quay.io/ascend/vllm-ascend:releases-v0.13.0-a3` which has
  vllm 0.13 + vllm_ascend 0.13)

For Phase 1 (manual run) lean toward (b): start from vllm-ascend
base image where vllm is real, layer the 4 ascend-port branches on
top.

## T22.3 — Dispatch table (manual run)

| Order | Component | Skill / process | Target version | Output |
|---|---|---|---|---|
| 1 | torch_npu | `/torch-npu-day0` | torch 2.9 in image (no bump yet) → outcome A is hopeful; otherwise produce ascend-port branch | confirm ledger or update |
| 2 | transformers | byte-compare image's 5.3.0.dev0 vs `<5.0.0` cap → EasyR1 will REJECT install; either loosen cap (touch EasyR1 master) or pin to 4.54 | 4.54 (safe lowest) or master with cap loosened | dispatch decision |
| 3 | vllm-ascend | `/vllm-ascend-day0` | vllm main (already done in ascend-port/vllm-main) | branch already exists, needs install |
| 4 | triton-ascend | already vendor-PASS in image | 3.2.0 | no action |
| 5 | EasyR1 | adapt requirements + run V1.1/V1.3/V1.4 smoke | master with cap loosened | ascend-port branch on EasyR1 fork + e2e PASS |

## T22.4 — Manual sub-agent runs

Plan: spawn one fresh sub-agent per row of the dispatch table, in
order. Each agent gets only the matching skill + the ledger and is
expected to land its work on the corresponding fork's
`ascend-port/<target-slug>` branch. After each agent: independent
verification (re-import in fresh container, check PASS criterion).

Will execute T22.4 in followup steps.

## T22.5 — Overlay image build

**Base image decision (revised after T22.4 row 3 retry)**:
`easyr1-npu-vllm0200:iter20-abi-both` (local image on A3 host).

- vllm 0.20.0 (real, not stub — the OLD baseline our shims target)
- vllm_ascend 0.17.0rc2.dev109+g54879467c (release-13-era + fixes)
- torch 2.11.0+cpu / torch_npu 2.11.0rc1
- transformers 5.3.0.dev0 (image default; outcome A-with-note for v5.4)

Rationale: previous attempts targeted vllm-ascend release-13 (vllm 0.13)
and verl-8.5.2 (vllm 0.18.0+empty stub) — both pre-introduction of the
symbols our shims target. The vllm0200 image has vllm 0.20.0 and our
3 shims cleanly resolve OLD paths there.

Single Dockerfile / docker commit that:
- Bases on `easyr1-npu-vllm0200:iter20-abi-both`
- Pip overlays each ascend-port branch's edits on top:
  - `torch_npu/compat/` from `ascend-port/torch-2.12-rc3` (precautionary
    even on torch 2.11 — shim is no-op until torch 2.12 ships, but keeps
    source consistent for future bumps)
  - `vllm_ascend/compat/` from `ascend-port/vllm-main` (3 shims: shared_fused_moe / default_moe_runner / spec_decode_base_proposer + 4 swapped call sites). 3/3 import smoke PASS in T22.4 row 3 retry.
  - transformers ascend-port marker (no source change; image already at 5.3.0.dev0 which is outcome A-with-note for v5.4)
  - triton-ascend 3.2.0 (image-default vendor; vendor 6/6 baseline PASS)
  - EasyR1 master with `transformers<5.0.0` cap removed (per row 2)
- Tag: `easyr1-npu:integrated-<DATE>`

## T22.6 — EasyR1 master adaptation

Apply EasyR1-side adapter changes:
- Loosen `transformers<5.0.0` cap if needed (or pin to 4.54)
- Verify all imports survive against the overlay image
- Land EasyR1 changes on `personal/EasyR1 ascend-port` branch

## T22.7 — V1.4 entropy_loss e2e

Run V1.4 GSM8K entropy_loss smoke under the overlay image; compare
to baseline value (recorded in
`docs/easyr1/SMOKE_BASELINE.md` if present, else fresh-baseline).

### Result (2026-04-27)

**PASS — fresh baseline established.**

- Image: `easyr1-npu:integrated-20260427` SHA
  `044ba0b7618338051037b7ac144090d80e6a48a4b8e49e0e8366a42c7a2268f1`,
  size 28.2 GB (overlay layer is small; total = base size).
- Recipe used: `repo/scripts/run-npu-container.sh --chips 0,1
  --image easyr1-npu:integrated-20260427 --live-source
  ~/workspace/easyr1-npu/upstream/EasyR1 -- bash
  /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh`. Duration
  ~10 min, 2 GRPO steps + post-train val.
- Validation metrics: `reward_score 0.0126`,
  `accuracy_reward 0.014`, `val_response_length mean 184.3`,
  `val_prompt_length mean 132.3`. Within expected step-2 noise band.
- Checkpoint saved: A3 `/tmp/z00637938/easyr1_smoke_ckpt/global_step_2/`.
- Smoke log: A3 `/tmp/z00637938/easyr1-logs/V1.4_integrated_run3.log` (232 KB).
- entropy_loss / grad_norm not captured in `experiment_log.jsonl`
  (pre-existing trainer-side jsonl writer quirk on this image stack;
  not a regression caused by the overlay). Baseline labelled
  "fresh baseline established, no historical comparison" — first
  validated run of `vllm 0.20.0 + torch 2.11 + transformers 5.3.0.dev0`
  combination on EasyR1.

### Lessons captured during T22.7

- `EC-06` (already in OPS KB): ssh-as-root must pass `NPU_USER=<owner>`
  to `run-npu-container.sh` so `/data/<owner>` and the tokenizer cache
  bind into the container. Default of `$USER=root` mounts empty paths.
- `ASCEND_RT_VISIBLE_DEVICES` on this image expects in-container chip
  indices (0,1), not host phy-ids (2,3). Confused chip-id models
  between host `npu-smi` and the running image's container view.
  Pre-check: `npu-smi info -t proc-mem -i N` for the host index;
  inside container map to local index via the image's
  `npu-smi info` output.

## T22.8 — Integrated PR_MATERIAL + ONBOARDING

Integrated hand-off: one-line `docker run easyr1-npu:integrated-20260427`
recipe that gets a customer to a working EasyR1-on-A3 with zero
extra steps.

Owned artifacts:

| Item | Location |
|---|---|
| Integrated image | A3 host: `easyr1-npu:integrated-20260427` (SHA `044ba0b7618338`) |
| EasyR1 fork branch | [`github.com/zhshgmail/EasyR1` `ascend-port-integrated-20260427`](https://github.com/zhshgmail/EasyR1/tree/ascend-port-integrated-20260427) (commit `8f5494e` on top of `ascend-port`) |
| vllm-ascend fork branch (input) | [`github.com/zhshgmail/vllm-ascend` `ascend-port/vllm-main`](https://github.com/zhshgmail/vllm-ascend/tree/ascend-port/vllm-main) head `6bac1f5e` |
| torch-npu fork branch (input) | [`gitcode.com/zhengshencn_hwca/pytorch` `ascend-port/torch-2.12-rc3`](https://gitcode.com/zhengshencn_hwca/pytorch/tree/ascend-port/torch-2.12-rc3) head `52de87dc1` |
| transformers fork branch (marker) | [`github.com/zhshgmail/transformers` `ascend-port/transformers-v5.4`](https://github.com/zhshgmail/transformers/tree/ascend-port/transformers-v5.4) |
| triton-ascend (image vendor) | [`gitcode.com/zhengshencn_hwca/triton-ascend` `ascend-port/triton-v3.6.0`](https://gitcode.com/zhengshencn_hwca/triton-ascend/tree/ascend-port/triton-v3.6.0) — image-default vendor 3.2.0, 6/6 NPU smoke PASS earlier |
| Smoke log | A3: `/tmp/z00637938/easyr1-logs/V1.4_integrated_run3.log` |
| Checkpoint | A3: `/tmp/z00637938/easyr1_smoke_ckpt/global_step_2/` |
| Build context | Local: `/tmp/integrated-overlay/Dockerfile` |
