# DSV4 NPU Porting Report — Independent Verification (2026-06-04 update)

Synthesis of 5 adversarial verifier groups (A–E). All verification was read-only:
A3 saved run-logs, on-disk source, evidence docs, `gh` API, `docker inspect`. No NPU
compute was run (minirl is live on chips 8–15).

---

## 1. Headline verdict

**25 claims total.**

| Status | Count |
|---|---|
| CONFIRMED | 20 |
| REFUTED | 2 |
| UNVERIFIABLE_WITHOUT_NPU | 3 |
| UNVERIFIABLE_NO_EVIDENCE | 0 |

Breakdown by group:

| Group | Claims | CONFIRMED | REFUTED | UNVERIFIABLE_WITHOUT_NPU |
|---|---|---|---|---|
| A — tilelang-on-A3 capability + perf | 6 | 5 | 0 | 1 |
| B — fp8 = A3 hardware wall | 4 | 2 | 1 | 1 |
| C — miles DSv4 plugin quantization | 7 | 6 | 1 | 0 |
| D — CANN 9.1.0-beta.1 reuse-base | 5 | 4 | 0 | 1 |
| E — upstream PR/issue status | 3 | 3 | 0 | 0 |

The core theses survive: fp8 is a real A3 hardware wall (B1/B4 confirmed), the miles DSv4
quantization is QAT-only/no-PTQ with a real-fp8-tensor coupling flaw (C confirmed), tilelang
vector (sinkhorn) numeric PASS and perf are genuine (A1/A3/A4 confirmed), and the upstream
PR/issue ledger is honest with zero merged-overclaims (E confirmed). **But two specific
error-attribution / dead-code framings are overclaimed and must be corrected** — see §2.

---

## 2. REFUTED claims — MUST be corrected in the report

### REFUTED #1 (Group B, claim 2) — 8.5.0 fp8 error attribution is wrong

> **Report claim:** "8.5.0 bishengir error: `'hivm.hir.vcast' op don't support cast bfloat16->fp8`"

**REFUTED.** The author's own contemporaneous notes consistently describe the 8.5.0 fp8
failure as a **`hivm.hir.store` allow-list omission**, not a `vcast` cast error:
- `CODEGEN_FIX_DEBUG_2026-06-03.md:124`: `8.5.0 bishengir 0.1.0 error: 'hivm.hir.store' op ... should have element type [list without float8] (verifier allow-list omission)`
- `KB_TILELANG_ASCEND.md:1179`: `fp8 tilelang kernel fails at hivm.hir.store verifier — "element type [list WITHOUT float8]"`
- `CODEGEN_FIX_DEBUG:91-93` frames it as a HIVM `hir.store` type allow-list omission, parallel to the bf16 #1199 store fix.

The string `vcast ... cast bfloat16->fp8` appears **nowhere** on local repo or A3 except in
the report itself (`DSV4_NPU_PORTING_REPORT.md:125`). The compiler's real unsupported-cast
message template (`AscendNPU-IR/.../HIVMVectorOps.cpp:266`) is
`currently don't support cast <castNameWithMode>` — different wording, and never shown to be
the actual fp8 failure. The report conflates a real CastOp message template with the actual
documented store-path failure. **Wrong op (vcast vs store) and wrong error class
(unsupported-cast vs allow-list omission).**

*Note: this does NOT weaken the fp8-hardware-wall thesis — claims B1/B3/B4 (the load-bearing
ones) hold. Only the 8.5.0 error-string attribution is inaccurate.*

### REFUTED #2 (Group C, claim 6) — fp8_gemm/fp4_gemm "dead code in training plugin" mislocates the kernels

> **Report claim:** "fp8_gemm/fp4_gemm GEMM kernels are dead code in the miles training plugin (0 callers, only kernel defs); but act_quant (fp8) and fp4 activation-quant ARE called by QAT"

**REFUTED (two scope errors):**

1. **The GEMM kernels are not in the training plugin at all.** `grep -rn 'fp8_gemm|fp4_gemm'`
   returns exit 1 across **all five** miles trees on A3 (`_miles_dsv4_preserved`, `_miles_real`,
   `miles`, `_miles_stub`, `Megatron-LM-miles`). They live in
   `tilelang-mlir-ascend/examples/deepseek_v4/inference/kernel.py` (`fp8_gemm_kernel:236`,
   `fp4_gemm_kernel:505`). The "0 callers" is **vacuously true only because the defs are absent**
   from the plugin. Where they actually exist (the inference example), **fp8_gemm HAS a caller and
   is NOT dead**: `inference/model.py:12 from kernel import act_quant, fp8_gemm` and
   `model.py:121 return fp8_gemm(...)`. The genuinely uncalled one is **fp4_gemm** (commented out at
   `model.py:118`).

2. **"fp4 activation-quant ARE called by QAT" is unsupported.** In the preserved training plugin,
   FP4 appears **only in a comment** (`deepseek_v4.py:41`); `compressor.py`/`v4_indexer.py` call
   **only `fp8_simulate_qat`**, never fp4. No fp4 function is invoked anywhere in the plugin.

*The `act_quant(fp8)`-called-by-QAT sub-claim IS correct (qat.py→act_quant, gated by
MEGATRON_USE_KV_QAT). The report's own line 397 already hedges ("dead code 仅对训练插件成立,
rollout 侧用途未 grep"), so the framing is acknowledged-incomplete — but as written it mislocates
the kernels and overstates fp4 usage.*

---

## 3. UNVERIFIABLE_WITHOUT_NPU claims (not disproven — deferred until A3 chips free)

These rest on an NPU/compile run with **no persisted proof line**. The version preconditions,
source mechanisms, and surrounding facts are confirmed; only the run-event artifact is missing.
Re-verify when minirl releases chips 8–15.

### A-claim2 — sparse_mla_fwd "All check passed!" on API codegen path
- matmul's API-path success **IS** logged (`RESULT_matmul_perf_20260603T164539Z.log`, numerically
  correct fp16). But **sparse_mla_fwd's "All check passed!" has NO saved log** —
  `grep -rln 'All check passed' _v4_runlogs/` returns nothing. The ONLY saved sparse_mla logs
  (`RESULT_mla_perf_094732`, `mla_example_asis2`) show sparse_mla **FAILING** in Developer-mode
  device-codegen (`getBroadcastDim` CHECK-fail at `codegen_npuir_dev.cc:175`). The success exists
  solely as prose in `CODEGEN_FIX_DEBUG:54`. Author already self-flagged (debug-doc:56) they never
  isolated whether sparse_mla needed the fix or passed pre-fix.

### B-claim3 — 9.1.0 fp8-store "Current hardware doesn't support fp8 type" run
- The error string + its SoC-gate semantics ARE confirmed in AscendNPU-IR source
  (`HIVMDMAOps.cpp:248`, also :347/:541) and the 2026-01-20 allow-list merge date is corroborated.
  But **no saved log captures the actual 9.1.0 bishengir-compile run on fp8.npuir emitting it** —
  the string lives only in binaries, source, and narrative docs. Mechanism real; run-event
  unpersisted.

### D-claim1 — torch_npu 2.8.0.post2 NPU matmul ABI-pass on 9.1.0 runtime
- Version preconditions all CONFIRMED (py3.11.14, torch_npu 2.8.0.post2, 9.1.0 toolkit co-resident
  in `cann910_test`). But the "NPU matmul passes, no ABI break" line has **no saved artifact** — the
  intended `/tmp/mm_910.log` failed to start (`run_matmul_api.py` deleted post-run). `plog-487` shows
  a clean short python3 NPU session but no allclose/PASS line.

---

## 4. CONFIRMED claims (20) — by group

**Group A (tilelang-on-A3):**
- A1 — sinkhorn numeric PASS `max_abs_diff 8.941e-08`, all shape bands ≤1.490e-07 (N=1024).
  `RESULT_sinkhorn_numeric_20260603T074741Z.log` + `RESULT_sinkhorn_shapesweep_20260603T083042Z.log`.
- A3 — batched sinkhorn perf: 2.42×-vs-1iter; vs-torch 1.15×(N256)/0.96×(N1024)/0.17×(N4096).
  `RESULT_sink_batched_*` logs, verbatim match.
- A4 — matmul 1024×512×2048 fp16 = 0.1123ms vs 0.0142ms = 0.13×.
  `RESULT_matmul_perf_20260603T164539Z.log`, exact match.
- A5 — nd2nz codegen fix is NOT load-bearing (honesty self-correction deflating own contribution);
  internally consistent with A6 loader logic. *Caveat: underlying nm/maps evidence is author-reported,
  not read-only re-verifiable; .so files rebuilt 2026-06-04 after all logs.*
- A6 — runtime loads `libtilelang_module.so` (not `libtilelang.so`).
  Verified in `tilelang/__init__.py:74` loader source.

**Group B (fp8 = A3 hardware wall):**
- B1 — open-stack patches present: `codegen_npuir_api.cc:503-511` maps fp8→getFloat8E4M3FNType/
  getFloat8E5M2Type; `data_type.h:434-460` parses float8_e4m3fn/float8_e5m2.
  *Nuance: vector-fp8 returns NoneType — scalar-only mapping (report doesn't mention, doesn't contradict).*
- B4 — fp8 cannot be unblocked by software on A3; A5/arch35 target. Source gate
  `HIVMDMAOps.cpp:248 if(!isAscend910_95 && fp8) emitOpError("Current hardware doesn't support fp8 type")`
  sits on the unavoidable DMA verify path upstream of author's codegen.

**Group C (miles DSv4 plugin quant):**
- C1 — sparse_mla_bwd hard-asserts bf16 at 3 sites (`tilelang_sparse_mla_bwd.py:24,61,103`);
  fwd+indexer bf16 hardcoded. Report correctly scopes the hard-assert to bwd.
- C2 — `deepseek_v4.py:142 assert self.wo_a.weight.dtype == torch.bfloat16` (exact line).
- C3 — `deepseek_v4.py:241-242` env-gated `MEGATRON_USE_KV_QAT`, default OFF via `.get(...,"0")`.
- C4 — modelopt/post_training ABSENT from dsv4 (grep exit 1), PRESENT at `glm5.py:12`.
- C5 — qat.py→act_quant materializes a real `float8_e4m3fn` tensor (`act_quant.py:90`); NOT pure-16-bit.
  *Nuance: no literal `T.Cast(bf16->float8_e4m3fn)` token — author's paraphrase of implicit cast; substance correct.*
- C7 — `qat.py:18` backward returns `grad_kv, None` = STE (exact).

**Group D (CANN 9.1.0-beta.1 reuse-base):**
- D2 — bishengir 1.1.0 / 2026-05-09 / AscendNPU-IR 7058cef3. Exact match from read-only `--version`.
- D3 — 9.1.0 gemm REGRESSION / LLVM crash. `/tmp/mm.log` captures cann-9.1.0-beta.1 bishengir-compile
  crashing with 19-frame LLVM stack dump on the gemm npuir. *8.5.0 "o85.mix_aic.o" half not on disk,
  but report discloses the dialect-version scope caveat (§三.3:184) — no overclaim.*
- D4 — #100 still segfaults on 9.1.0 (sinkhorn iters=4 → core dump), CANN-independent (crash is in
  tilelang in-process PassManager FFI before any bishengir binary). `/tmp/s4.log`.
  *Nuance: 8.5.0 threshold was iters≥5; 9.1.0 crashes at iters=4 (lower/structure-sensitive) — doesn't weaken core point.*
- D5 — `cann910_test` container on `/dev/davinci5`. `docker inspect` confirms (Up 28h, device mount).

**Group E (upstream PR/issue ledger):**
- E1 — tilelang-mlir-ascend issue #100 FILED + OPEN; title/body match report §六 item-12.
- E2 — item 11 = PR #96 (sparse_mla heads<block_H, OPEN), item 12 = issue #100 (OPEN); both match gh.
- E3 — report claims ONLY item 10 (a5_ops internal repo) as MERGED; every public item state matches gh:
  PR#80/PR#96/miles#1246/sgl-kernel-npu#531/sglang#26794 OPEN, triton-ascend#306 CLOSED(not-planned).
  Zero public unmerged items presented as merged. *Item 10 MERGED state is internal-repo, not gh-reachable,
  but report keeps it scoped as internal toolchain fix.*

---

## 5. Recommended report edits

| # | Report location | Action | Fix |
|---|---|---|---|
| 1 | `DSV4_NPU_PORTING_REPORT.md:125` (B-claim2) | **FIX (REFUTED)** | Replace `'hivm.hir.vcast' op don't support cast bfloat16->fp8` with the actual documented 8.5.0 error: `'hivm.hir.store' op ... should have element type [list WITHOUT float8]` (verifier allow-list omission, parallel to bf16 #1199). Source of truth: CODEGEN_FIX_DEBUG:124 / KB:1179. |
| 2 | §六 / line ~397 (C-claim6) | **FIX (REFUTED)** | Reword the "dead code in training plugin" claim. State precisely: (a) fp8_gemm/fp4_gemm are NOT in any miles training plugin — they live in `tilelang-mlir-ascend/examples/deepseek_v4/inference/kernel.py`; (b) in that inference example **fp8_gemm IS called** (`model.py:121`) — only **fp4_gemm** is commented-out/uncalled; (c) the training plugin's only quant call is fp8 `act_quant` via `fp8_simulate_qat`; **fp4 appears only in a comment** (`deepseek_v4.py:41`), not invoked. Drop "fp4 activation-quant ARE called by QAT". |
| 3 | sparse_mla "All check passed!" line (A-claim2) | **QUALIFY** | Split the bundled matmul+sparse_mla claim. matmul API-path success is logged; **sparse_mla_fwd success is NOT logged** and the only saved sparse_mla logs are Developer-mode FAILURES. Mark sparse_mla as "expert-mode pass observed in session, not captured to run-log — pending re-run when A3 free." |
| 4 | B-claim3 / §三 fp8-store narrative | **QUALIFY** | Note the 9.1.0 fp8-store hardware-reject is source-confirmed (mechanism) but the actual run output was not persisted; mark the run-event as pending-recapture. |
| 5 | D-claim1 ABI-pass line | **QUALIFY** | The torch_npu 2.8.0.post2 NPU-matmul-on-9.1.0 ABI-pass has no saved log (script deleted). Soften to "version stack co-resident + clean short NPU runtime session observed; allclose PASS line not persisted — pending recapture." |
| 6 | C-claim5 `T.Cast(...)` wording | **MINOR** | Optional: note `T.Cast(bf16->float8_e4m3fn)` is a paraphrase of an implicit cast (assignment into fp8-typed fragment), not a verbatim source token. Substance is correct. |
| 7 | B-claim1 fp8 mapping | **MINOR** | Optional: note the fp8 codegen mapping is **scalar-only** (vector/non-scalar fp8 returns NoneType). Doesn't change the thesis. |

**Bottom line:** the report's central conclusions are sound and the upstream ledger is honest
(no merged-overclaims). Two error-attribution/dead-code framings (edits #1 and #2) are genuine
overclaims that contradict the author's own evidence and must be corrected. Three NPU-run claims
lack persisted proof lines (edits #3–#5) and should be softened to "pending recapture" rather than
stated as confirmed passes — they are not disproven, just not independently re-confirmable while
minirl holds the chips.
