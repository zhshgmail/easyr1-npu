# NPU gap plan — what's left to make EasyR1 really work on A3

Living document. Owner: this port. Date: 2026-04-18.

Purpose: collect everything that still stands between the current `ascend-port` branch (x86-validated, image-built) and a production-quality EasyR1-on-A3 deployment. This doc feeds task scheduling — each section is either "done", "in v1", "v2+", or "perf follow-up", with explicit rationale and next action.

Scope anchor: the dependency matrix at `dep-matrix.md` and the design doc at `design.md`. Read those first.

Target image: `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest` (CANN 8.5.0, torch_npu 2.8.0, transformers 4.57.6, vllm_ascend 0.13.1.dev18, triton_ascend 3.2.0). Verified against its matching upstream refs at `upstream-refs.md`.

---

## 0. Snapshot: what ships in the image vs what EasyR1 needs

Out of 20 declared EasyR1 runtime deps, **every one** is already present in `verl-8.5.0-a3`, at a version satisfying EasyR1's pins — verified by rebuilding the layered Dockerfile.npu on both hosts and seeing "Requirement already satisfied" for every line of `requirements.txt`.

Three hidden direct imports (not in EasyR1's `requirements.txt` but present in source) also ship in the image: `jinja2 3.1.6`, `psutil 7.2.2`, `pyyaml 6.0.3`.

The port is not mainly a dependency problem. It is a **code-path problem** — EasyR1 has ~35 CUDA-only call sites plus a handful of attention-backend assumptions that don't fit NPU. The `ascend-port` branch has landed all the easy lifts; what remains is the last mile.

---

## 1. Done on `ascend-port` (8 commits)

| Commit | What it does | Status on A3 |
|---|---|---|
| `7ee0f0b` | Split `requirements.txt` into common/gpu/npu variants; declare hidden deps; tighten `tensordict` pin | **in image** |
| `72b564a` | Add `verl/utils/device.py` with `is_npu_available`, `get_device_name`, `get_device_module`, `get_dist_backend`, `get_default_attn_implementation`, `get_visible_devices_env` | **in image** |
| `7187b51` | Sweep ~35 `torch.cuda.*` call sites to `get_device_module()`; device-mesh device type; dist backend; `device_map`; ROCm gate | **in image** |
| `496d198` | Wrap bare `current_device()` int in `torch.device(device_name, index)` for `.to()` calls; fix missed `flat_param_to("cuda", ...)` in `load_fsdp_submodule`; add `extras_require["gpu"]`; raise `KeyError` loudly on malformed RNG state | **in image** |
| `6701a50` | Attention backend NPU-aware: `sdpa` default on NPU, `flash_attention_2` on CUDA; `ModelConfig.attn_implementation` optional; `apply_ulysses_patch` gated on NPU | **in image** |
| `da2487f` | Vendor `flash_attn.bert_padding` helpers (`index_first_axis`, `pad_input`, `unpad_input`) at `verl/utils/npu_flash_attn_utils.py`; `verl/utils/attention_utils.py` lazy dispatch façade; `dp_actor.py` / `dp_critic.py` use the façade | **in image** |
| `ffafa0d` | NPU gate moved from `apply_ulysses_patch` (where message blamed ulysses) to `_build_model_optimizer` (config-level, right diagnosis); `tests/test_device.py` added — first x86-runnable unit tests | **in image** |
| `cbfe645` | `Dockerfile.npu` layered on `verl-8.5.0-a3` base, `.dockerignore` | **built** as `easyr1-npu:ascend-port` |

---

## 2. v1 scope — what MUST work for the port to count as "working"

### 2.1 Functional contract

From `design.md §1.4`:

- Text-only PPO / GRPO recipes on Qwen2 / Qwen3 family models.
- 8.5.0 base image.
- Default loggers only (console, file, wandb).
- `padding_free=False` (required — see 3.1).
- `ulysses_size=1` (required — see 3.2).

### 2.2 Still-to-validate on A3 (BLOCKED-until-hardware)

| # | Task | Expected outcome | If it fails |
|---|---|---|---|
| V1.1 | `docker run --device /dev/davinci0 -e ASCEND_RT_VISIBLE_DEVICES=0 easyr1-npu:ascend-port python -c 'import verl; from verl.utils.device import is_npu_available, get_device_name; print(is_npu_available(), get_device_name())'` | `True, "npu"` | root cause is probably device-passthrough or an env var the base image expects |
| V1.2 | same but load a Qwen2-0.5B config with `attn_implementation` unset | model loads with SDPA, no error | likely transformers 4.57.6 still thinks it can't find SDPA for our config; narrow by running with `attn_implementation="eager"` |
| V1.3 | rollout-only smoke test: one recipe, rollout path via vllm_ascend, no training | completes, outputs generated text | common breaks: HCCL init on single card, vllm_ascend model coverage |
| V1.4 | short training run: 2-4 steps of GRPO on Qwen2-0.5B, `padding_free=False`, `ulysses_size=1`, `batch_size=1` | 4 gradient steps complete without OOM | common breaks: attention mask shape mismatch, RNG state restore, optimizer offload/load path |
| V1.5 | multi-card short training: same as V1.4 but world_size=4 (one A3 card, both chips, plus one more A3 card) | 4 gradient steps with HCCL all-reduce | HCCL init, FSDP topology, ulysses=1 sanity |

These all need hardware. We have 8 A3 cards idle on the host (confirmed 2026-04-18). Can proceed immediately.

### 2.3 v1 known-unknowns

- Does `transformers 4.57.6`'s SDPA path on NPU return correct log-probs for Qwen2/Qwen3 on short sequences? Codex found NPU-specific handling in `sdpa_attention.py` but we haven't numerically verified on hardware.
- Does `vllm_ascend 0.13.1.dev18` cover the Qwen2/Qwen3 rollout code paths EasyR1 exercises (LoRA? speculative? specific attention masks)?
- HBM headroom for a 0.5B model + FSDP shards + vllm engine in 64GB per chip. Expected fine, untested.

---

## 3. Deferred to v2+ (deliberate, with rationale)

### 3.1 Padding-free training on NPU

**What it is:** EasyR1's `padding_free=True` packs variable-length sequences into a flat token stream and relies on `flash_attn_varlen_func` to attend across the pack.

**Why deferred:** the varlen attention kernel is flash-attn-only on GPU; on NPU it would have to be rewritten on top of `torch_npu.npu_fusion_attention` with `input_layout` / `actual_seq_qlen` / `actual_seq_kvlen` args. That's a non-trivial forward.

**What's already in place:** the padding helpers (`index_first_axis`, `pad_input`, `unpad_input`) are vendored at `verl/utils/npu_flash_attn_utils.py` and accessed via `verl/utils/attention_utils.py` façade. So **when someone writes the NPU varlen forward, the helper surface is ready**.

**Next action when we pick this up:**
1. Wire a new `_custom_flash_attention_forward` variant calling `torch_npu.npu_fusion_attention` with `input_layout="TND"`, cu_seqlens mapped to `actual_seq_{q,kv}len`.
2. Branch `apply_ulysses_patch` to register the NPU variant instead of raising.
3. Remove the `padding_free=True` gate in `_build_model_optimizer`.

### 3.2 Ulysses sequence parallelism on NPU

**What it is:** optional sequence-dim parallelism that splits long sequences across ranks. Used for long-context training.

**Why deferred:** the existing Ulysses forward at `verl/models/transformers/flash_attention_utils.py` is a tight wrapper around `flash_attn_varlen_func` with `gather_seq_scatter_heads` / `gather_heads_scatter_seq` collectives. NPU would need the same forward rewritten on `npu_fusion_attention` + HCCL variants of the collectives.

**Prereq:** 3.1 (the NPU varlen forward). Once 3.1 exists, Ulysses is mostly a collective-substitution exercise.

### 3.3 8.5.2 image migration

**What it is:** switch target image from `verl-8.5.0-a3` (CANN 8.5.0, torch_npu 2.8.0, transformers 4.57.6, vllm_ascend 0.13.1.dev18) to `verl-8.5.2-a3` (CANN 8.5.1, torch_npu 2.9.0, transformers 5.3.0.dev0, vllm_ascend 0.17.0rc2.dev109, + upstream triton 3.6.0).

**Why deferred:** EasyR1's `transformers>=4.54.0,<5.0.0` excludes the 8.5.2 image. Migration adds:
- a transformers 4.x → 5.x compatibility sweep across EasyR1 (API churn in `ALL_ATTENTION_FUNCTIONS`, model-class relocations, etc.);
- a `huggingface_hub 0.36` → `1.11` compatibility audit;
- vllm_ascend 0.17 vs 0.13 API/config shifts.

**When we do it:** after v1 is stable, and only if 8.5.2 delivers a concrete perf or feature win. Most common motivation is a newer `npu_fusion_attention` revision that makes 3.1 easier.

### 3.4 liger-kernel on NPU

**What it is:** fused Triton kernels for RMSNorm, RoPE, SwiGLU, cross-entropy.

**Why deferred:** GPU-only (flash-attn–adjacent Triton). NPU doesn't have a direct equivalent; some subset could be ported via `triton-ascend`, but the upstream Liger project doesn't support Ascend.

**When we do it:** only if we see a clear perf gap in v1 that Liger would close. First candidate: the `F.cross_entropy` upcast-to-fp32 fallback at `torch_functional.py:68`. If that costs real throughput, write a `triton-ascend` CE kernel as a focused drop-in.

### 3.5 `torch_npu.npu_fusion_attention` as the EasyR1 default on NPU

**What it is:** instead of using transformers' built-in SDPA dispatcher, register `npu_fusion_attention` directly as the `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` handler on NPU (analogous to what `apply_ulysses_patch` does on GPU with flash-attn).

**Why deferred:** SDPA is a correct drop-in for v1, perf-competitive for short/medium sequences. `npu_fusion_attention` wins at longer sequences and varlen. Not needed until we have a perf target to hit.

### 3.6 Multi-node scale-out

Out of v1 by design (`design.md §1.3`). Single-node A3 is the target.

---

## 4. Hidden / latent issues worth tracking

These are things that aren't blocking but could surface as runtime surprises:

1. **RNG state portability** (commit `496d198`) — checkpoints saved on CUDA contain `"cuda"` key; we read as fallback for `"accelerator"`. A CUDA checkpoint restored on NPU (or vice versa) copies an RNG byte string that's device-specific. Might desynchronize nothing-reproducibility claims across accelerators. OK for v1, note for users.

2. **`torch.backends.cuda.matmul.allow_tf32` semantics on NPU** — we guarded behind `not is_npu_available()` so the knobs are inert on NPU. But their intent ("improve numerical stability by disabling TF32") has NO equivalent knob on NPU; npu matmul has its own precision modes. Someone reading the code might assume the guard was enough — it's not a numerical-equivalence guarantee.

3. **`compressed_tensors` version skew (0.12.2 in image, vllm pulls newer on main)** — harmless at v1 but will matter when we try 8.5.2.

4. **HCCL deterministic flags** — image sets `LCCL_DETERMINISTIC=0 LCCL_PARALLEL=0`. If we need deterministic RL runs for paper-style results, we'd flip these. Document implication: slower but reproducible.

5. **Disk pressure on A3** — host is 93% used, 258 GB free. Every docker image (18 GB each), weight download (0.5–14 GB each for Qwen), checkpoint save (~model size × fsdp shards) eats into that. Partition: images under `/var/lib/docker` (root fs, tight); weights under `/data/z00637938/` (root fs, same partition — **also tight**). Plan: prune docker images aggressively, keep only 1-2 weight sets at a time, move checkpoints off-host if runs go long.

6. **Shared-host discipline** — we're not the only user. Other users' `safe.directory` entries show up in global git config. Our rule: chip 0-7 are usually ours (chip 8-15 were 52 GB/card when first checked, idle later), but confirm with `npu-smi info -t proc-mem` before each run. Don't kill processes we didn't start.

---

## 5. Harness buildout still to do (repo/skills/, repo/scripts/)

From `design.md §4.3`. What's landed:
- `repo/skills/codex-review/SKILL.md` — general reviewer via the local codex CLI.
- `repo/knowledge/` templates for image inventory, source-dep extraction, upstream-refs.

Still to formalize (and earn their place when the pattern gets used twice):
- `scripts/extract-image.sh` — `docker create` + `docker cp` site-packages + derive pip freeze + produce `knowledge/images/<tag>.md`.
- `scripts/dep-diff.py` — two dep sets → matrix markdown.
- `skills/inspect-ascend-image` — wraps the above into an agent-invocable flow.
- `skills/classify-dep-gap` — given a package + version delta, propose V/P/R/A/D and an investigation plan.
- `skills/upstream-branch-hygiene` — enforce "work on `personal/ascend-port` branches, never patch files" across all deps.

---

## 6. Sequencing recommendation

Given hardware is now available and 8 A3 cards are idle:

1. **Now → next session**: V1.1 → V1.4 on hardware. Every failure feeds `porting-journal.md`. Expected duration: 1 session if no major surprise, 2-3 if SDPA / vllm_ascend coverage has gaps.
2. **After V1.4 passes**: V1.5 (multi-card), then declare v1 done.
3. **After v1**: decide perf vs feature priority. If the v1 runs are fast enough for research iteration, defer 3.1 (NPU varlen). If not, 3.1 first.
4. **8.5.2 migration (3.3)**: only when we have a concrete driver (e.g. transformers 5.x feature EasyR1 picks up upstream).
