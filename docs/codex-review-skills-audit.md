
Given hardware is now available and 8 A3 cards are idle:

1. **Now → next session**: V1.1 → V1.4 on hardware. Every failure feeds `porting-journal.md`. Expected duration: 1 session if no major surprise, 2-3 if SDPA / vllm_ascend coverage has gaps.
2. **After V1.4 passes**: V1.5 (multi-card), then declare v1 done.
3. **After v1**: decide perf vs feature priority. If the v1 runs are fast enough for research iteration, defer 3.1 (NPU varlen). If not, 3.1 first.
4. **8.5.2 migration (3.3)**: only when we have a concrete driver (e.g. transformers 5.x feature EasyR1 picks up upstream).

exec
/bin/bash -lc 'rg -n "NPU-|Root cause|Symptom|Fix|Where applied|Rule|hidden|issue|bug|drift|NotImplementedError|hccl|nccl|padding_free|ulysses|LoRA|pycache|HF_ENDPOINT|ASCEND_RT_VISIBLE_DEVICES|get_tensor_model_parallel_group|flash_attn|liger|torch.backends.cuda|deterministic|RNG|validation|checkpoint" docs/porting-journal.md docs/npu-gap-plan.md' in /home/z00637938/workspace/easyr1-npu/repo
 succeeded in 0ms:
docs/npu-gap-plan.md:17:Three hidden direct imports (not in EasyR1's `requirements.txt` but present in source) also ship in the image: `jinja2 3.1.6`, `psutil 7.2.2`, `pyyaml 6.0.3`.
docs/npu-gap-plan.md:27:| `7ee0f0b` | Split `requirements.txt` into common/gpu/npu variants; declare hidden deps; tighten `tensordict` pin | **in image** |
docs/npu-gap-plan.md:30:| `496d198` | Wrap bare `current_device()` int in `torch.device(device_name, index)` for `.to()` calls; fix missed `flat_param_to("cuda", ...)` in `load_fsdp_submodule`; add `extras_require["gpu"]`; raise `KeyError` loudly on malformed RNG state | **in image** |
docs/npu-gap-plan.md:31:| `6701a50` | Attention backend NPU-aware: `sdpa` default on NPU, `flash_attention_2` on CUDA; `ModelConfig.attn_implementation` optional; `apply_ulysses_patch` gated on NPU | **in image** |
docs/npu-gap-plan.md:32:| `da2487f` | Vendor `flash_attn.bert_padding` helpers (`index_first_axis`, `pad_input`, `unpad_input`) at `verl/utils/npu_flash_attn_utils.py`; `verl/utils/attention_utils.py` lazy dispatch façade; `dp_actor.py` / `dp_critic.py` use the façade | **in image** |
docs/npu-gap-plan.md:33:| `ffafa0d` | NPU gate moved from `apply_ulysses_patch` (where message blamed ulysses) to `_build_model_optimizer` (config-level, right diagnosis); `tests/test_device.py` added — first x86-runnable unit tests | **in image** |
docs/npu-gap-plan.md:47:- `padding_free=False` (required — see 3.1).
docs/npu-gap-plan.md:48:- `ulysses_size=1` (required — see 3.2).
docs/npu-gap-plan.md:54:| V1.1 | `docker run --device /dev/davinci0 -e ASCEND_RT_VISIBLE_DEVICES=0 easyr1-npu:ascend-port python -c 'import verl; from verl.utils.device import is_npu_available, get_device_name; print(is_npu_available(), get_device_name())'` | `True, "npu"` | root cause is probably device-passthrough or an env var the base image expects |
docs/npu-gap-plan.md:57:| V1.4 | short training run: 2-4 steps of GRPO on Qwen2-0.5B, `padding_free=False`, `ulysses_size=1`, `batch_size=1` | 4 gradient steps complete without OOM | common breaks: attention mask shape mismatch, RNG state restore, optimizer offload/load path |
docs/npu-gap-plan.md:58:| V1.5 | multi-card short training: same as V1.4 but world_size=4 (one A3 card, both chips, plus one more A3 card) | 4 gradient steps with HCCL all-reduce | HCCL init, FSDP topology, ulysses=1 sanity |
docs/npu-gap-plan.md:64:- Does `transformers 4.57.6`'s SDPA path on NPU return correct log-probs for Qwen2/Qwen3 on short sequences? Codex found NPU-specific handling in `sdpa_attention.py` but we haven't numerically verified on hardware.
docs/npu-gap-plan.md:65:- Does `vllm_ascend 0.13.1.dev18` cover the Qwen2/Qwen3 rollout code paths EasyR1 exercises (LoRA? speculative? specific attention masks)?
docs/npu-gap-plan.md:74:**What it is:** EasyR1's `padding_free=True` packs variable-length sequences into a flat token stream and relies on `flash_attn_varlen_func` to attend across the pack.
docs/npu-gap-plan.md:78:**What's already in place:** the padding helpers (`index_first_axis`, `pad_input`, `unpad_input`) are vendored at `verl/utils/npu_flash_attn_utils.py` and accessed via `verl/utils/attention_utils.py` façade. So **when someone writes the NPU varlen forward, the helper surface is ready**.
docs/npu-gap-plan.md:82:2. Branch `apply_ulysses_patch` to register the NPU variant instead of raising.
docs/npu-gap-plan.md:83:3. Remove the `padding_free=True` gate in `_build_model_optimizer`.
docs/npu-gap-plan.md:89:**Why deferred:** the existing Ulysses forward at `verl/models/transformers/flash_attention_utils.py` is a tight wrapper around `flash_attn_varlen_func` with `gather_seq_scatter_heads` / `gather_heads_scatter_seq` collectives. NPU would need the same forward rewritten on `npu_fusion_attention` + HCCL variants of the collectives.
docs/npu-gap-plan.md:104:### 3.4 liger-kernel on NPU
docs/npu-gap-plan.md:114:**What it is:** instead of using transformers' built-in SDPA dispatcher, register `npu_fusion_attention` directly as the `ALL_ATTENTION_FUNCTIONS["flash_attention_2"]` handler on NPU (analogous to what `apply_ulysses_patch` does on GPU with flash-attn).
docs/npu-gap-plan.md:124:## 4. Hidden / latent issues worth tracking
docs/npu-gap-plan.md:128:1. **RNG state portability** (commit `496d198`) — checkpoints saved on CUDA contain `"cuda"` key; we read as fallback for `"accelerator"`. A CUDA checkpoint restored on NPU (or vice versa) copies an RNG byte string that's device-specific. Might desynchronize nothing-reproducibility claims across accelerators. OK for v1, note for users.
docs/npu-gap-plan.md:130:2. **`torch.backends.cuda.matmul.allow_tf32` semantics on NPU** — we guarded behind `not is_npu_available()` so the knobs are inert on NPU. But their intent ("improve numerical stability by disabling TF32") has NO equivalent knob on NPU; npu matmul has its own precision modes. Someone reading the code might assume the guard was enough — it's not a numerical-equivalence guarantee.
docs/npu-gap-plan.md:134:4. **HCCL deterministic flags** — image sets `LCCL_DETERMINISTIC=0 LCCL_PARALLEL=0`. If we need deterministic RL runs for paper-style results, we'd flip these. Document implication: slower but reproducible.
docs/npu-gap-plan.md:136:5. **Disk pressure on A3** — host is 93% used, 258 GB free. Every docker image (18 GB each), weight download (0.5–14 GB each for Qwen), checkpoint save (~model size × fsdp shards) eats into that. Partition: images under `/var/lib/docker` (root fs, tight); weights under `/data/z00637938/` (root fs, same partition — **also tight**). Plan: prune docker images aggressively, keep only 1-2 weight sets at a time, move checkpoints off-host if runs go long.
docs/porting-journal.md:30:- **3 real gaps**: `flash-attn` (NPU replacement), `liger-kernel` (drop or triton-ascend port), `pillow` (add to install list).
docs/porting-journal.md:35:**Why:** EasyR1's transformers ceiling is compatible with 8.5.0 out of the box. This cuts the initial port to two concrete changes (flash-attn + liger-kernel) instead of also fighting a transformers 5.x migration.
docs/porting-journal.md:41:1. **EasyR1-side shim — attention + kernels** (`upstream/EasyR1/ascend-port`): make `flash-attn` and `liger-kernel` imports optional, gate on device type, route attention through `torch_npu` / `vllm_ascend` backend. Size: small–medium, x86-doable at import level.
docs/porting-journal.md:45:5. **A3 runtime validation (BLOCKED — needs hardware)**: build image on top of 8.5.0 adding the EasyR1 ascend-port branch; rollout smoke test; short training run.
docs/porting-journal.md:65:- **pillow was NOT missing from the images.** Both 8.5.0 and 8.5.2 ship `pillow-12.2.0`. My original filter grep in `grep -E '^(torch|transformers|…)[_-]'` did not include `pillow`, so I assumed absent. Corrected in `dep-matrix.md`. "Three real gaps" → "two dep-level gaps (flash-attn, liger-kernel) + the 8.5.2-only transformers ceiling."
docs/porting-journal.md:71:- Distributed backend hardcoded `nccl` (`fsdp_workers.py:83`) — must become `hccl` on NPU.
docs/porting-journal.md:74:- `torch.cuda.*` calls pervasive across `workers/`, `utils/`, `protocol.py`, `checkpoint/`.
docs/porting-journal.md:75:- Direct `from flash_attn ...` imports: kernel func + `bert_padding` helpers + triton cross-entropy — some of these (padding helpers) aren't attention kernels at all, they're pure-torch utilities that just happen to live in flash-attn's package.
docs/porting-journal.md:83:Design doc §1.2 said "port EasyR1" broadly. Narrowed §1.4 to v1 = text-only PPO/GRPO on Qwen2/Qwen3, 8.5.0 base image, default loggers only. VLM/video, 8.5.2 migration, liger-kernel via triton-ascend, and deeper flash-attn perf work all go to v2+.
docs/porting-journal.md:87:§4.4.1 (docker image build plan) previously said "on top of `verl-8.5.2-a3`" while §4.2 strategy says "target 8.5.0 first." Fixed to 8.5.0 throughout.
docs/porting-journal.md:110:1. `7ee0f0b` — split `requirements.txt` into common / gpu / npu variants. Moved flash-attn / liger-kernel / vllm to `requirements-gpu.txt`. Declared hidden direct imports (`jinja2`, `psutil`, `pyyaml`). Tightened `tensordict` pin. `requirements-npu.txt` is a placeholder pointing at the verl-A3 base image.
docs/porting-journal.md:112:3. `7187b51` — routed ~35 `torch.cuda.*` call sites across 10 files through the device-module accessor. Covers RNG, mem_get_info, empty_cache, current_device, max_memory_*, manual_seed, device-mesh device type, dist backend, device_map, ROCm detection (guarded on NPU). Also `torch.backends.cuda.matmul.*` knobs guarded behind `not is_npu_available()`.
docs/porting-journal.md:140:- `torch_npu.__init__.py:258` registers "hccl" as a torch.distributed backend. That's what makes `init_process_group(backend="hccl")` work.
docs/porting-journal.md:151:5. `6701a50` — attention backend NPU-aware. New `get_default_attn_implementation()` in `verl/utils/device.py` (sdpa on NPU, flash_attention_2 on CUDA). Made `ModelConfig.attn_implementation` optional — resolves at runtime if None. Threaded through the two `AutoClass.from_pretrained` / `from_config` calls in `fsdp_workers.py`. Made `apply_ulysses_patch` raise on NPU (codex later caught that the error text blamed the wrong thing).
docs/porting-journal.md:152:6. `da2487f` — decoupled padding helpers from flash-attn. Added `verl/utils/npu_flash_attn_utils.py` (pure-torch reimpl of index_first_axis / pad_input / unpad_input, lifted from flash-attn bert_padding.py per veRL's own NPU port). Added `verl/utils/attention_utils.py` façade — lazy dispatch between flash_attn.bert_padding (CUDA) and the vendored copy (NPU). Updated `dp_actor.py` and `dp_critic.py` to import from the façade.
docs/porting-journal.md:153:7. `ffafa0d` — corrective pass from codex review. Moved the NPU `padding_free=True` rejection from `apply_ulysses_patch` to `_build_model_optimizer` in `fsdp_workers.py` (config-level, with clearer error text). Kept `apply_ulysses_patch`'s NPU branch as a defensive backstop. Added `tests/test_device.py` — first x86 unit tests in the port — mocking `is_npu_available()` to verify every accessor's cuda/npu resolution.
docs/porting-journal.md:158:- **Padding-free is NPU-blocked in v1**: SDPA doesn't handle the unpadded/flattened token path that `flash_attn_varlen_func` does. Wiring NPU varlen would require writing an equivalent forward on top of `npu_fusion_attention`. Out of v1.
docs/porting-journal.md:159:- **Ulysses SP is NPU-blocked in v1**: same reason; the ulysses-SP forward at `flash_attention_utils.py` calls `flash_attn_varlen_func` directly.
docs/porting-journal.md:166:- Caught a real messaging bug: my NPU gate in `apply_ulysses_patch` blamed ulysses when the real issue was padding_free. Moving the check upstream to the config-level validation made the error traceable.
docs/porting-journal.md:171:- Replace `F.cross_entropy` upcast with an NPU-optimized CE kernel (triton-ascend?).
docs/porting-journal.md:172:- Write NPU varlen attention forward on top of `npu_fusion_attention` so padding_free=True works on NPU.
docs/porting-journal.md:173:- Write NPU-aware ulysses SP forward.
docs/porting-journal.md:178:Phase 4.2 code changes (2.1–2.6) are complete on `ascend-port`. 7 commits, +339/-63. Everything AST-parses. Device tests pass locally (under stubbed torch). Moving to phase 4.4 (build a Dockerfile layered on verl-8.5.0-a3 + install EasyR1 ascend-port) — which is **as far as x86 can take us**. Then the rest is BLOCKED on A3 hardware: rollout smoke test, short training run, runtime validation.
docs/porting-journal.md:186:### Build output observations (dep-matrix validation)
docs/porting-journal.md:188:Every common-requirements pin came back as "Requirement already satisfied." That confirms the dep-matrix analysis: the 20 declared EasyR1 runtime deps plus the 3 hidden imports (jinja2/psutil/pyyaml) are fully covered by the base image. Zero downloads during the install step.
docs/porting-journal.md:194:Ran three kinds of smoke test inside the image on this x86 host, with `TORCH_DEVICE_BACKEND_AUTOLOAD=0` to bypass the torch_npu autoload (which fails here because `libascend_hal.so` is NPU-only):
docs/porting-journal.md:196:1. **Default accessors** — `verl.utils.device` imports and returns `cuda` / `flash_attention_2` / `nccl` / `CUDA_VISIBLE_DEVICES`. Would flip to the NPU variants on A3.
docs/porting-journal.md:197:2. **attention_utils façade** — `index_first_axis`, `pad_input`, `unpad_input`, `rearrange` import via the façade without triggering flash_attn or torch_npu. Consumers (dp_actor, dp_critic) get clean imports.
docs/porting-journal.md:198:3. **Vendored NPU padding-helpers correctness** — ran a real round-trip: `unpad_input(hidden, attention_mask)` then `pad_input(unpadded, indices, batch, seqlen)` on a small tensor with a sample attention mask. Output shape `(2, 4, 3)`, non-masked positions match the original input. The vendored implementation is correct at the tensor level (the same torch ops run on NPU via torch_npu).
docs/porting-journal.md:203:- **4.4.3** [BLOCKED] End-to-end short training run (few steps) on a small Qwen2/Qwen3 model with `padding_free=False`, `ulysses_size=1` (v1 scope).
docs/porting-journal.md:204:- **4.4.4** [BLOCKED] Document runtime findings. Any recurring issue that surfaces there feeds back into the harness (new skill or script).
docs/porting-journal.md:233:### First real NPU issue: broken triton-ascend install in the base image
docs/porting-journal.md:243:Confirmed the same error in the adjacent `verl-sglang-8.3.rc1-a3` image someone else had running — so this is a base-image-build issue, not specific to our layer.
docs/porting-journal.md:252:Fix in our `Dockerfile.npu`: force-reinstall triton-ascend immediately after `FROM`, before anything else. One extra RUN layer, ~1.6 GB delta.
docs/porting-journal.md:276:get_dist_backend(): hccl
docs/porting-journal.md:277:get_visible_devices_env(): ASCEND_RT_VISIBLE_DEVICES
docs/porting-journal.md:284:All accessors resolve to their NPU variants. The vendored `npu_flash_attn_utils.py` helpers run correctly against `torch_npu` — first hardware validation of that file beyond x86 CPU.
docs/porting-journal.md:288:Need to download a small Qwen2 checkpoint (Qwen2-0.5B) via `hf-mirror.com` into `/data/z00637938/models/`, then run a rollout through `vllm_ascend` inside the container. This is the first real test of the vllm_ascend attention path on this image.
docs/porting-journal.md:296:- `device_config=npu`, `backend=hccl` (confirms our device + collective routing)
docs/porting-journal.md:310:Requires wiring an EasyR1 trainer config with `padding_free=False`, `ulysses_size=1`, the local Qwen2-0.5B-Instruct path, a tiny dataset, a trivial reward. Then 2-4 GRPO steps.
docs/porting-journal.md:314:## 2026-04-18 (late) — V1.4 bring-up bugs
docs/porting-journal.md:316:Wrote `examples/qwen2_0_5b_math_grpo_npu_smoke.sh` (tiny batch, `padding_free=false`, `ulysses_size=1`, `max_steps=2`). Launched via `repo/scripts/run-npu-container.sh`. Each attempt uncovered one more NPU-specific issue, fixed on the local workstation and git-synced. Cumulative NPU-findings ledger:
docs/porting-journal.md:318:### NPU-BUG-001: base image's triton-ascend install is partial
docs/porting-journal.md:322:**Fix**: `Dockerfile.npu` adds a `pip install --force-reinstall --no-deps triton-ascend==3.2.0` layer before any other installs. Falls back to huaweicloud Ascend PyPI mirror if PyPI is unreachable.
docs/porting-journal.md:324:### NPU-CP-002: EasyR1 imports a vllm module that was renamed in vllm 0.13
docs/porting-journal.md:326:`verl/utils/vllm_utils.py` does `from vllm.lora.models import LoRAModel`. vllm 0.13 (what our target image ships) renamed that file to `vllm.lora.lora_model`. Every `verl.trainer.main` import path transitively pulls vllm_utils in, so the module-level import breaks the whole trainer before anything runs.
docs/porting-journal.md:328:**Fix**: try new-path first, fall back to old-path. Also moved `worker_manager` and `utils` imports into `hijack()` so module load doesn't force eager attribute lookups.
docs/porting-journal.md:330:### NPU-CP-003: Ray doesn't auto-detect Ascend NPU as a builtin resource
docs/porting-journal.md:334:**Fix across 4 files**:
docs/porting-journal.md:340:### NPU-ENV-001: container didn't have HF_ENDPOINT
docs/porting-journal.md:344:**Fix**: `run-npu-container.sh` injects `HF_ENDPOINT=https://hf-mirror.com` and `HF_HOME=/data/z00637938/hf-cache` by default.
docs/porting-journal.md:346:### NPU-BUG-002: Ray 2.55 clears visibility env vars inside actors
docs/porting-journal.md:348:Most confusing one. After the previous 3 fixes, Runner actor still failed with the pre-fix error message ("Total available GPUs 0"). Probed with a fresh Ray actor that imports `torch` + `torch_npu`: **driver saw `torch.npu.is_available() == True, device_count == 2`, actor saw `False, 0` and `ASCEND_RT_VISIBLE_DEVICES=""`**.
docs/porting-journal.md:354:**Fix**: `verl/trainer/main.py` adds `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0` to `runtime_env.env_vars`.
docs/porting-journal.md:356:### Non-bug but operational issue: build-time COPY shadowed our edits
docs/porting-journal.md:360:**Fix**: `run-npu-container.sh` now bind-mounts `/home/z00637938/workspace/easyr1-npu/upstream/EasyR1` over `/opt/easyr1`, so `git pull` is enough (no rebuild) for pure source changes. Dockerfile changes still need a rebuild.
docs/porting-journal.md:364:### Also cleared `/opt/easyr1`'s stale `__pycache__`
docs/porting-journal.md:366:After changing imports in edited modules while `.pyc` files existed, Python sometimes picked up the stale cache. `find .../upstream/EasyR1 -name __pycache__ -type d -exec rm -rf {} +` once — and for future sessions, the runner script should probably clear caches on startup or set `PYTHONDONTWRITEBYTECODE=1`.
docs/porting-journal.md:370:Final bug before pass: `NPU-ENV-002` — vllm-ascend's FRACTAL_NZ layout causes param-sync precision drift in RL. Set `VLLM_ASCEND_ENABLE_NZ=0` in `ray.init` runtime_env. vllm-ascend itself raises a prescriptive `ValueError` pointing at exactly this knob.
docs/porting-journal.md:374:- 2/2 training steps finished in 8m24s (Qwen2-0.5B, chips 0+1, bf16, eager, `padding_free=false`, `n=2` rollouts, `global_batch_size=4`)
docs/porting-journal.md:381:This is the v1 functional milestone per `design.md §1.1-1.4`. Everything from this point is either V1.5 (multi-card scale-up, same code), perf work, or the deferred v2 items (NPU varlen attention, ulysses, 8.5.2 migration).
docs/porting-journal.md:383:### Summary of NPU-specific findings from the V1.4 bring-up
docs/porting-journal.md:387:- Code patterns (`NPU-CP-*`): 4 found
docs/porting-journal.md:388:- Platform bugs (`NPU-BUG-*`): 2 found
docs/porting-journal.md:389:- Env/config (`NPU-ENV-*`): 2 found
docs/porting-journal.md:390:- Operational (`NPU-OPS-*`): 2 found (bind-mount shadowing, stale pycache)

codex
**STRENGTHS**
- `docs/DELIVERABLE.md` and `docs/porting-journal.md` tell a coherent story about the actual EasyR1 port: the same 10 stable IDs recur, the same commit set shows up, and the same v1/v2 boundary is preserved.
- `knowledge/npu-patterns.md` is already useful as a reusable memory artifact: most entries have concrete symptoms, fixes, commit anchors, and transfer rules, not just anecdotes.
- The shipped skill set is directionally right for a second Ray-based RL port: image inspect, code-path sweep, container runner, branch hygiene, Ray shim, and external review cover the main early workflow.
- `skills/ray-npu-shim/ray_npu_shim.py` is small and plausible as a drop-in. The exported API matches the core Ray/NPU integration points the EasyR1 port actually needed.

**ISSUES**
- **MAJOR — `scripts/code-path-sweep.sh`, `skills/npu-code-path-sweep/SKILL.md`, `knowledge/npu-patterns.md`: stable-ID drift.** The catalog defines `NPU-CP-001`..`004`, but the script emits invented IDs like `NPU-CP-001a`, `NPU-CP-003b`, plus `GPU-ONLY` and `CFG`. Those do not exist in `npu-patterns.md`, so review focus #1 fails for the sweep artifact.
- **MAJOR — `scripts/code-path-sweep.sh`: pattern tuple parsing is internally broken.** Entries contain unescaped `|` inside the regex and suggestion fields, but the parser does `IFS='|' read -r id title regex suggest`. That means several patterns are split incorrectly, so the implemented regex/suggestion is not the one documented.
- **MAJOR — `docs/skills-design.md` §4 vs shipped skills: taxonomy mismatch is only partially explained.** The doc predicts 10 skills, but only 6 shipped. §7 says “pick 2 high-value skills first,” which explains a phased rollout, but the document never marks which of the 10 are intentionally deferred versus superseded by scripts/manual docs. It also predicts `npu-smoke-test`, `dep-diff`, `version-align`, and `gap-plan` as skills, but those are not shipped; meanwhile `ray-npu-shim` is shipped and not present in §4 at all.
- **MAJOR — `knowledge/npu-patterns.md` vs `docs/DELIVERABLE.md` §3: “each entry has symptom / root cause / fix / commit ref / rule” is overstated.** The catalog is not field-uniform. `NPU-CP-001` has no explicit “Root cause” field; “commit ref” is sometimes “Where applied”; operational/env entries are structured differently. It is readable, but not schema-consistent.
- **MAJOR — `docs/skills-design.md` §5 is stale against the delivered catalog.** It still says stable IDs are a TODO and gives `NPU-CP-001 = nccl -> hccl backend swap` as an example, while the actual catalog uses `NPU-CP-001` as the broader device-accessor pattern family.
- **MINOR — `docs/DELIVERABLE.md` §3 / `docs/npu-gap-plan.md` §4: some reusable findings remain uncatalogued.** RNG-state portability, TF32-knob semantics on NPU, HCCL deterministic flags, and disk-pressure/shared-host constraints are treated as recurring risks but have no `NPU-ENV-*` / `NPU-OPS-*` IDs.
- **MINOR — `skills/ray-npu-shim/SKILL.md` overpromises slightly on “4-5 swaps.”** The recipe covers Ray registration, actor options, and placement bundles, but a real second port still needs a parallel `NPU-CP-001` pass for device strings, backend names, and `torch.cuda.*` usage. The doc mentions this in step 5, but the headline reads more turnkey than it is.
- **MINOR — `ray_npu_shim.py`: one edge-case gotcha.** `apply_actor_options()` adds `resources={"NPU": n}` on NPU but does not clear a preexisting `num_gpus` field. In a framework that pre-populates `num_gpus`, you could end up with mixed signals unless the caller removes it first.
- **MINOR — `skills/npu-image-inspect/SKILL.md` and script are internally consistent, but `docs/skills-design.md`’s image-skill gate is not implemented.** The design doc says image outputs should include sections like “Matching upstream refs” and “Open questions”; the actual script emits a smaller schema.

**SUGGESTIONS**
- Normalize the ID model now: either keep only canonical IDs (`NPU-CP-001`) in sweep output, or add sub-IDs to `npu-patterns.md`. Right now it is neither stable nor grep-friendly.
- Fix `scripts/code-path-sweep.sh`’s pattern table format before handoff. A TSV/JSON array would remove the broken `|` parsing.
- Add a short status table to `docs/skills-design.md`: `planned`, `shipped`, `deferred`, `superseded`. That resolves the 10-vs-6 mismatch cleanly and makes `ray-npu-shim` an explicit addition.
- Make `npu-patterns.md` schema-uniform. Every entry should use the same headers: `Symptom`, `Root cause`, `Fix`, `Commit ref`, `Generalizable rule`.
- Promote the recurring latent risks to IDs where appropriate: at least RNG portability, HCCL determinism, and shared-host/disk-pressure ops issues.
- In `ray_npu_shim.py`, defensively delete `num_gpus` on the NPU path inside `apply_actor_options()`.

**VERDICT**
The harness is close, but it needs one more iteration before handoff if the goal is “someone else can port the next Ray-based RL repo using just this system.” The core pieces are good and mostly reusable, but the stable-ID story is not yet trustworthy end-to-end because the sweep tool invents IDs and its pattern parser is flawed, and the design doc still describes a planned system more than the shipped one. Fix those two areas and the handoff becomes credible.
tokens used
51,057
**STRENGTHS**
- `docs/DELIVERABLE.md` and `docs/porting-journal.md` tell a coherent story about the actual EasyR1 port: the same 10 stable IDs recur, the same commit set shows up, and the same v1/v2 boundary is preserved.
- `knowledge/npu-patterns.md` is already useful as a reusable memory artifact: most entries have concrete symptoms, fixes, commit anchors, and transfer rules, not just anecdotes.
- The shipped skill set is directionally right for a second Ray-based RL port: image inspect, code-path sweep, container runner, branch hygiene, Ray shim, and external review cover the main early workflow.
- `skills/ray-npu-shim/ray_npu_shim.py` is small and plausible as a drop-in. The exported API matches the core Ray/NPU integration points the EasyR1 port actually needed.

**ISSUES**
- **MAJOR — `scripts/code-path-sweep.sh`, `skills/npu-code-path-sweep/SKILL.md`, `knowledge/npu-patterns.md`: stable-ID drift.** The catalog defines `NPU-CP-001`..`004`, but the script emits invented IDs like `NPU-CP-001a`, `NPU-CP-003b`, plus `GPU-ONLY` and `CFG`. Those do not exist in `npu-patterns.md`, so review focus #1 fails for the sweep artifact.
- **MAJOR — `scripts/code-path-sweep.sh`: pattern tuple parsing is internally broken.** Entries contain unescaped `|` inside the regex and suggestion fields, but the parser does `IFS='|' read -r id title regex suggest`. That means several patterns are split incorrectly, so the implemented regex/suggestion is not the one documented.
- **MAJOR — `docs/skills-design.md` §4 vs shipped skills: taxonomy mismatch is only partially explained.** The doc predicts 10 skills, but only 6 shipped. §7 says “pick 2 high-value skills first,” which explains a phased rollout, but the document never marks which of the 10 are intentionally deferred versus superseded by scripts/manual docs. It also predicts `npu-smoke-test`, `dep-diff`, `version-align`, and `gap-plan` as skills, but those are not shipped; meanwhile `ray-npu-shim` is shipped and not present in §4 at all.
- **MAJOR — `knowledge/npu-patterns.md` vs `docs/DELIVERABLE.md` §3: “each entry has symptom / root cause / fix / commit ref / rule” is overstated.** The catalog is not field-uniform. `NPU-CP-001` has no explicit “Root cause” field; “commit ref” is sometimes “Where applied”; operational/env entries are structured differently. It is readable, but not schema-consistent.
- **MAJOR — `docs/skills-design.md` §5 is stale against the delivered catalog.** It still says stable IDs are a TODO and gives `NPU-CP-001 = nccl -> hccl backend swap` as an example, while the actual catalog uses `NPU-CP-001` as the broader device-accessor pattern family.
- **MINOR — `docs/DELIVERABLE.md` §3 / `docs/npu-gap-plan.md` §4: some reusable findings remain uncatalogued.** RNG-state portability, TF32-knob semantics on NPU, HCCL deterministic flags, and disk-pressure/shared-host constraints are treated as recurring risks but have no `NPU-ENV-*` / `NPU-OPS-*` IDs.
- **MINOR — `skills/ray-npu-shim/SKILL.md` overpromises slightly on “4-5 swaps.”** The recipe covers Ray registration, actor options, and placement bundles, but a real second port still needs a parallel `NPU-CP-001` pass for device strings, backend names, and `torch.cuda.*` usage. The doc mentions this in step 5, but the headline reads more turnkey than it is.
- **MINOR — `ray_npu_shim.py`: one edge-case gotcha.** `apply_actor_options()` adds `resources={"NPU": n}` on NPU but does not clear a preexisting `num_gpus` field. In a framework that pre-populates `num_gpus`, you could end up with mixed signals unless the caller removes it first.
- **MINOR — `skills/npu-image-inspect/SKILL.md` and script are internally consistent, but `docs/skills-design.md`’s image-skill gate is not implemented.** The design doc says image outputs should include sections like “Matching upstream refs” and “Open questions”; the actual script emits a smaller schema.

**SUGGESTIONS**
- Normalize the ID model now: either keep only canonical IDs (`NPU-CP-001`) in sweep output, or add sub-IDs to `npu-patterns.md`. Right now it is neither stable nor grep-friendly.
- Fix `scripts/code-path-sweep.sh`’s pattern table format before handoff. A TSV/JSON array would remove the broken `|` parsing.
- Add a short status table to `docs/skills-design.md`: `planned`, `shipped`, `deferred`, `superseded`. That resolves the 10-vs-6 mismatch cleanly and makes `ray-npu-shim` an explicit addition.
- Make `npu-patterns.md` schema-uniform. Every entry should use the same headers: `Symptom`, `Root cause`, `Fix`, `Commit ref`, `Generalizable rule`.
- Promote the recurring latent risks to IDs where appropriate: at least RNG portability, HCCL determinism, and shared-host/disk-pressure ops issues.
- In `ray_npu_shim.py`, defensively delete `num_gpus` on the NPU path inside `apply_actor_options()`.

**VERDICT**
The harness is close, but it needs one more iteration before handoff if the goal is “someone else can port the next Ray-based RL repo using just this system.” The core pieces are good and mostly reusable, but the stable-ID story is not yet trustworthy end-to-end because the sweep tool invents IDs and its pattern parser is flawed, and the design doc still describes a planned system more than the shipped one. Fix those two areas and the handoff becomes credible.
