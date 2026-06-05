# Codegen fix debug state — gemm/Cube nd2nz + getBroadcastDim (2026-06-03)

> **Owner directive (16:08 UTC)**: 深挖这个 codegen 生成问题，看能否修复。
> **OWNERSHIP (corrected, owner-confirmed)**: this workspace is SOLO — the 154-line dirty `codegen_npuir_dev.cc` diff is **MY OWN T32 work**, on **my own PR #96 branch** `blue/fix/sparse-mla-heads-lt-block-h`. Safe to edit + rebuild. NOT a teammate's. See [[feedback_solo_workspace_dirty_is_mine]].
> Host: `ssh -p 443 root@115.190.166.102`, container `sgl_probe` (has NPU davinci1). tilelang at `/home/z00637938/workspace/tilelang-mlir-ascend`.

## State of my T32 codegen edits (already compiled into current .so, built 06-02 10:19 > src 06-02 04:20)
My dirty diff to `src/target/codegen_npuir_dev.cc` (~154 lines) — theme = **dynamic-shape + scalar codegen**:
1. `ReshapeTensorImpl` (~1279): dynamic-dim CollapseShapeOp.
2. `NeedGenInsertSlice` (~2144): `tensor::EmptyOp` with dynamicSizes via tensor/memref DimOp.
3. `CreateHIVMBinaryVectorOp` (~2709): BufferLoadNode-as-scalar; `buffer_shape.clear()` to short-circuit getBroadcastDim; commutative operand swap; scalar-scalar arith.{Mul,Add,Sub,Div}F fallback.
4. `VisitStmt_(AllocateNode)` (~3872/3889): `tensor::EmptyOp` dynamicSizes for symbolic extents.

## TWO compile errors still failing (with my fixes already in the .so)
### (A) matmul.py → `nd2nz` result-semantics mismatch  ← ROOT-CAUSED, fix identified
- Error: `'hivm.hir.nd2nz' op expected the number of tensor results (0) to be equal to the number of output tensors (1)`
- Failing IR: `"hivm.hir.nd2nz"(%28, %18) ... : (memref<128x16xf16,...>, tensor<128x16xf16>) -> ()` — dst `%18` is a **tensor**, but op emitted with **0 results**.
- ROOT CAUSE: `Nd2NzCodegen` (codegen_npuir_dev.cc:2581) creates `ND2NZOp` with `mlir::TypeRange res = {}` (0 results, memref/DPS style). But my EmptyOp changes made the L1 buffers **tensors** → nd2nz now needs **value semantics**: when dst is a tensor, op must return 1 tensor result, and the dst SSA binding must be updated to that result (so downstream `mmadL1` consumes the filled tensor). My EmptyOp fix was INCOMPLETE — made buffers tensors but didn't update nd2nz (and likely nz2nd, vflip, others that use `res={}` with a now-tensor dst).
- FIX PLAN: in `Nd2NzCodegen`, if `dst.getType().isa<TensorType>()`: `res = {dst.getType()}`, capture `auto r = builder.create<ND2NZOp>(...).getResult(...)`, and rebind dst's source SSA var to `r` (SetVarValue / update the region's buffer value). If dst is memref, keep `res={}`. Check ND2NZOp's builder signature for the result-returning form.
- SAME PATTERN likely needed for `Nz2NdCodegen` (2596+), `VFlipOp` (2575), and any other op created with `TypeRange{}` + tensor dst.

### (B) sparse_mla_fwd → `getBroadcastDim` CHECK(shape0[i]==shape1[i])  ← second bug, codegen_npuir_dev.cc:175
- A vcast/broadcast op with two non-1 mismatched dims. My `buffer_shape.clear()` short-circuit (in CreateHIVMBinaryVectorOp scalar path) handles the scalar case but not this one. Investigate after (A).

## PROGRESS LOG (live)
- **2026-06-03 ~16:20**: patched `Nd2NzCodegen` in `codegen_npuir_dev.cc:2599` (tensor-dst → value-semantics + SetVarValue rebind). Rebuilt libtilelang.so (make -j64; had to symlink `/usr/bin/cmake` → the missing pip cmake path first; `[100%] Built target tilelang`, .o+.so fresh). **matmul STILL fails same nd2nz error.**
- **Root of why fix didn't take**: matmul uses **`CodeGenTileLangNPUIRAPI`** (codegen_npuir_**api**.cc), NOT `...DEV`. The nd2nz emit for matmul is in **`codegen_npuir_api.cc`** — TWO sites: line **1371** (expert-mode `T.copy(GM,L1)→nd2nz`, `TypeRange{}`) and line **1786** (`Nd2NzCodegen`, `res={}`). Both emit 0-result; both need the same tensor-dst fix as I did in _dev.cc. I patched the wrong file's copy.
- **OPEN QUESTION before blindly patching api.cc**: the api path shows NO `SetVarValue` calls → its memory model may be memref-based (not tensor-rebind). Need to confirm: is the dst at api.cc:1786/1371 a tensor or memref? The verifier error shows `tensor<128x16xf16>` dst → tensor. **But the deeper question: did my EmptyOp→tensor change (in _dev.cc) overreach into the Cube/L1/cbuf (shared.dyn) path that should have stayed memref?** The `.orig.t32_emptyop` backup suggests I was mid-investigation on exactly this. Two candidate fixes:
  - (a) Apply the same tensor-dst value-semantics fix to api.cc:1371 + 1786 (mirror _dev.cc). Risk: api path may not support tensor rebind cleanly.
  - (b) **Revert** the EmptyOp-as-tensor change for **shared.dyn / L1 / cbuf** buffers (keep them memref), so nd2nz(memref,memref)→() stays valid. This may be the more correct fix — gemm L1 operands are inherently memref/DPS, not value-semantic tensors. The vector path (sinkhorn) is what needed tensors.
- **NEXT**: determine dst type at api.cc:1371; decide (a) vs (b). (b) is likely more correct + lower-risk if the EmptyOp-tensor change was only meant for the vector/UB path.

## PROGRESS 2 (2026-06-03 ~16:30) — fixed 3 emit sites but matmul nd2nz comes from an UNFOUND path
- Patched ND2NZOp tensor-dst value-semantics at: `_dev.cc:2599` (Nd2NzCodegen) + `_api.cc:1786` (Nd2NzCodegen) + `_api.cc:1371` (expert T.copy GM→L1). Rebuilt each time (make -j64 tilelang, cmake symlinked). 
- **matmul STILL fails identical nd2nz 0-result error.** Header says `CodeGenTileLangNPUIRDEV` (so DEV codegen + its Finish() verification, per rt_mod_npuir.cc:75).
- **Added a `llvm::errs() << "[T32.10-DIAG]"` print inside `_dev.cc:Nd2NzCodegen`, rebuilt, ran matmul → the DIAG line NEVER prints.** ⇒ **matmul's nd2nz is NOT emitted via `Nd2NzCodegen` at all.** It comes from some other path in the DEV codegen that I haven't located (grep shows only _dev.cc:2601/2606 create ND2NZOp in dev, both inside the now-instrumented Nd2NzCodegen — yet it's not called). 
- HYPOTHESES for where matmul's nd2nz really comes from: (1) an MLIR **transform pass** inserts nd2nz on the module AFTER codegen, before Finish()'s verification (search src/transform for nd2nz insertion / a bufferization/L1-staging pass); (2) the `T.gemm`/`tl.npuir_dot` or a `T.copy(GM,L1)` lowers to a different op-handler that builds ND2NZOp inline (check VisitExpr_/VisitStmt_ dispatch for the gemm-operand-load, and `src/op/gemm.cc` / `bulk_copy.cc`); (3) Developer-mode (lower.py:170) routes a distinct lowering.
- **NEXT (didn't finish — owner-time / deep)**: find the real nd2nz emit for matmul L1 operand loads. Check `grep -rn nd2nz src/transform src/op`; check how matmul.py's `T.gemm`/copy lowers (it uses `T.alloc_L1` + `T.gemm(...)`); instrument the actual emit once found. The fix pattern (tensor-dst → value-result + rebind) is correct; just applied to the wrong/incomplete site.
- BACKUPS: `_dev.cc` → `/tmp/codegen_npuir_dev.cc.bak_nd2nz`; `_api.cc` → `/tmp/codegen_npuir_api.cc.bak`. (The DIAG print is still in _dev.cc — harmless, remove before any commit.)

## PROGRESS 3 (2026-06-03 ~16:33) — confirmed NOT a cache artifact; nd2nz emit path for matmul is genuinely unlocated
- Cleared tilelang cache (`tilelang.cache.clear_cache()` + rm cache dirs) and re-ran matmul → DIAG STILL doesn't fire, same error. So it's not stale cache, not stale .so (DIAG string IS in the rebuilt .so per `strings`).
- **CONFIRMED**: matmul's `T.load_nd2nz` → `hivm.hir.nd2nz` is emitted by a path that is NOT `CodeGenTileLangNPUIRDEV::Nd2NzCodegen` (instrumented, never called) and NOT the 2 _api.cc sites. The 3 grep-able `create<ND2NZOp>` sites are all in functions that don't fire for matmul. 
- ⇒ The nd2nz MLIR op for the gemm L1-operand load must be materialized either (a) in a TIR→MLIR **lowering/transform pass** (not the VisitExpr codegen), or (b) via a dispatch/op-builder I haven't grepped (the op name `tl.npuir_load_nd2nz` dispatches at _dev.cc:3713 → SHOULD call Nd2NzCodegen, but doesn't reach it for matmul — suggests the TIR op is rewritten/lowered to the MLIR op BEFORE the VisitExpr dispatch runs, i.e. inside an MLIR pass on the module).
- **HONEST STATUS**: bug CLASS root-caused (tensor-dst nd2nz needs value-semantics result+rebind; my EmptyOp-tensor design requires it). 3 emit sites fixed. But matmul's ACTUAL nd2nz emit path UNLOCATED → **matmul still does NOT compile. NOT fixed.** Did not achieve the goal this session-segment.
- **NEXT to find the real path**: dump the MLIR module BEFORE the failing verification (Finish at _dev.cc:365) to see at which pass nd2nz appears with 0-results; or grep the AscendNPU-IR / tilelangir transform passes (not src/transform which is the TIR side — the MLIR-side passes live in 3rdparty/AscendNPU-IR or tilelangir) for nd2nz creation; or trace how `tl.npuir_load_nd2nz` TIR op is converted to `hivm.hir.nd2nz` MLIR op (there may be a TIR-op → MLIR-op table/conversion separate from VisitExpr).
- Cleanup before any commit: remove the `[T32.10-DIAG]` llvm::errs print from _dev.cc.

## RESOLVED (2026-06-03 ~16:45) — I was testing on the WRONG codegen path the whole time
- **lower.py:170 routing**: `TILELANG_ASCEND_MODE` None or in {expert,exp,e} → **API codegen** (`tilelang_npuir_apis`, the DEFAULT). Anything else (e.g. **"Developer"**) → **DEV codegen** (`tilelang_npuir_dev`).
- I had been setting `TILELANG_ASCEND_MODE=Developer` for ALL my matmul/sparse_mla tests → DEV path. That's why my `_api.cc` fix "didn't fire" and the DEV `Nd2NzCodegen` diagnostic was the only thing I checked. **The gemm/attention kernels are meant to run on the API/expert path, not Developer.**
- **Re-tested with `TILELANG_ASCEND_MODE=expert` (API path), cache cleared:**
  - `examples/gemm/matmul.py` → **`All check passed!`** ✅ (compiles + runs + numerically correct on NPU)
  - `examples/sparse_mla_fwd.py` → **`All check passed!`** ✅
- → **gemm/Cube tilelang kernels DO compile + run + verify on NPU (API/expert path). My `_api.cc` nd2nz tensor-dst fix is in and the path works.** The earlier "gemm/Cube can't compile" was a **wrong-mode artifact** (I tested in Developer mode), NOT a real fork limitation.
- HONESTY caveat: I have NOT isolated whether sparse_mla_fwd needed my fix or already passed on API pre-fix (all earlier fails were Developer mode). What's certain: BOTH pass now on API path with my fix in. Matmul: I only ever got it to pass AFTER the _api.cc fix.
- **NEXT (the original owner goal)**: now that gemm/attention compile on API path → measure their tilelang-vs-torch PERF (this was infra-blocked before). Use `TILELANG_ASCEND_MODE=expert`.

## GEMM PERF MEASURED (2026-06-03 ~16:50, API path, real NPU) — RESULT_matmul_perf_*.log
- matmul M=1024 N=512 K=2048 fp16: **tilelang(API) 0.1123ms vs torch-NPU 0.0142ms = 0.13× (tilelang ~8× SLOWER)**, numerically correct (maxdiff 6.25e-2, fp16 tol).
- Consistent with sinkhorn: **tilelang on NPU, even when compiling correctly, is currently SLOWER than torch/CANN.** For gemm, torch.matmul dispatches to CANN's hand-tuned Cube gemm (~8× faster than this basic-tiling tilelang example kernel).
- **Complete perf answer (both classes now measured):** vector/sinkhorn 0.06–0.45× torch (#100 multi-launch tax) · gemm/matmul 0.13× torch (basic tiling vs CANN-tuned). → tilelang-port is FUNCTIONALLY viable (compiles + numerically correct, both classes) but NOT yet perf-competitive vs torch/CANN on either class. Perf would need: #100 fix (vector single-kernel) + gemm kernel tuning (tiling/pipelining to match CANN).

## CLEANUP + COMMIT-DECISION (2026-06-03 ~16:58)
- **Removed the `[T32.10-DIAG]` llvm::errs print from `_dev.cc`** (source clean now; running .so still has it but it's harmless — only prints on DEV-mode nd2nz, and gemm runs API mode. Next rebuild clears it.)
- **Commit/PR decision is OWNER's** (not autonomous): the nd2nz tensor-dst fix is real + working (API path matmul/sparse_mla pass), BUT it lives in a workspace with **51 dirty files** on branch `blue/fix/sparse-mla-heads-lt-block-h` (which is PR #96 = the unrelated heads<block_H example fix). Issues: (a) committing codegen work to the #96 branch mixes unrelated concerns; (b) the 51-file dirty state is my mid-investigation T32 work — not all coherent/finished, committing blindly captures half-done work. Proper handling = separate branch for the codegen nd2nz fix + understand the full 51-file diff first. This needs owner direction (how to structure) + is outward-facing if PR'd. NOT done autonomously.
- The working fix is durably captured HERE (this doc) + in the dirty working tree; safe from loss.

## ★ CRITICAL BUILD-TARGET DISCOVERY (2026-06-03 ~18:28) — I'd been rebuilding the WRONG .so all along
- The runtime loads **`libtilelang_module.so`** + `libtilelangir.so` (verified via /proc/self/maps), NOT `libtilelang.so`. **`DTypetoMLIRType` / the device codegen lives in `libtilelang_module.so`** (nm: 4 DTypetoMLIRType symbols there, 0 in libtilelangir).
- ALL my earlier rebuilds were `make tilelang` → `libtilelang.so` (WRONG target). `libtilelang_module.so` was **stale ~16 days** the whole time. → **every codegen fix I "tested" earlier never actually loaded.**
- Correct target = **`make -j128 tilelang_module`** (also needed: `ln -sf /usr/lib/x86_64-linux-gnu/libzstd.so.1 .../libzstd.so` — the link needs libzstd.so dev symlink, missing in sgl_probe).
- **IMPLICATION — honesty reconcile (IMPORTANT)**: the earlier "matmul / sparse_mla_fwd All check passed!" results were on `libtilelang.so`-rebuilds = **STALE codegen (WITHOUT my nd2nz fix)**. So those kernels **passed WITHOUT my nd2nz fix being loaded** → **my nd2nz `_api.cc` fix was NOT load-bearing for them; they already compiled on the stock API codegen.** Must correct the report claim "my nd2nz fix made matmul work" → matmul/sparse_mla already worked on the API path; my nd2nz fix was unnecessary for them (and untested until now). The REAL value of finding the right target is the FP8 work below.

## FP8 — now unblocked through the whole OPEN stack; wall is bishengir backend (2026-06-03 ~18:30)
With `libtilelang_module.so` correctly rebuilt (all my fixes finally live), fp8 progressed layer by layer:
1. ✅ TVM string parse (`String2DLDataType`): float8_e4m3fn/e4m3/e5m2 + float4_e2m1fn/e2m1 (fixed substr off-by-one).
2. ✅ codegen dtype→MLIR-type (`DTypetoMLIRType` in codegen_npuir_api.cc): fp8 e4m3fn/e5m2 → `getFloat8E4M3FNType`/`getFloat8E5M2Type`. (FP4: no MLIR Float4 builder in fork's MLIR → still unmapped, deeper.)
3. ✅ tilelang codegen emits well-formed fp8 MLIR (`memref<...xf8E4M3FN...>`, nd2nz, etc.) — verified in dumped npuir.
4. ❌ **bishengir-compile (CANN 8.5.0, the closed-ish backend) rejects the fp8 npuir** — "Failed to run BiShengIR pipeline", no specific diagnostic. Same class as bf16 #1199: open layers fixed, CANN-shipped bishengir doesn't process fp8 cbuf/nd2nz.
- **NET**: fp8 is now unblocked through ALL open/fixable layers (parse + MLIR-type + codegen). The remaining wall = **bishengir backend fp8 support** (CANN, not open-layer fixable; fileable like #1199). FP4 has an earlier wall too (no MLIR Float4 builder).
- T.copy fp32→fp8 also hits "T.copy does not support element type casting" (expected — need explicit cast op, that's a kernel-authoring detail not a codegen bug).

## FP8 bishengir wall — PRECISE diagnostic (2026-06-03 ~19:09, standalone bishengir-compile on dumped fp8.npuir)
The earlier "Failed to run BiShengIR pipeline (no diagnostic)" was the Python wrapper swallowing it. Ran `bishengir-compile /tmp/fp8.npuir --enable-hivm-compile=true` standalone → **specific verifier error**:
```
'hivm.hir.store' op failed to verify that operand at idx 0 should have element type
  8-bit signless/unsigned integer | 16-bit signless/unsigned integer | 16-bit float | bfloat16
  | 32-bit signless/unsigned integer | 32-bit float | 64-bit unsigned/signless integer
```
- **The allow-list explicitly includes bf16 but OMITS f8E4M3FN.** → bishengir 0.1.0's HIVM op verifier (`hivm.hir.store`, likely others) **type allow-list excludes float8** types.
- **EXACT same class as bf16 #1199** (HIVM op verifier type-list omission). Note bf16 IS in the list now → bf16 was added/fixed at some point; fp8 just hasn't been. **So fp8 is a fileable AscendNPU-IR/HIVM verifier type-list omission, NOT a fundamental hw limit** — parallels #1199 (which I filed for bf16 vmul/vexp).
- → **upgraded FP8 conclusion**: not a vague "bishengir doesn't support fp8" — it's a **precise HIVM verifier allow-list omission for float8** (open-source bishengir/AscendNPU-IR, fileable like #1199, potentially upstream-fixable). The whole open tilelang stack (parse/MLIR-type/codegen) is done; only the HIVM verifier type-list needs float8 added.

## FP8 — RESOLVED at source level: upstream AscendNPU-IR already added float8 to StoreOp allow-list (2026-06-03, owner's "upgrade CANN" instinct validated)
Located the exact verifier allow-list: `bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td` `def StoreOp` → `OperElemTypeConstraints<[0], [...]>`.
- **Vendored (container, AscendNPU-IR commit `31f69036`)**: `[I8, UI8, I16, UI16, F16, BF16, I32, UI32, F32, UI64, I64]` — **NO float8**.
- **Upstream master (github Ascend/AscendNPU-IR, fetched)**: `[F8E4M3FN, F8E5M2, I1, I8, UI8, I16, UI16, F16, BF16, I32, UI32, F32, UI64, I64]` — **float8 ADDED**.
- → **fp8 support was added upstream AFTER the vendored commit.** Confirms owner's instinct: a newer CANN/AscendNPU-IR HAS fp8 in the verifier. (Same story as bf16: bf16 already in vendored list = was added earlier; fp8 is the next one.)
- **Two unblock paths**:
  1. **Upgrade CANN** to one whose bishengir is built from a newer AscendNPU-IR (owner's path) — `image-upgrade-drill` to find the right CANN image.
  2. **Patch the vendored `.td`**: add `F8E4M3FN, F8E5M2` to StoreOp (and likely other HIVM ops') `OperElemTypeConstraints`, rebuild bishengir (open-source, rebuildable per [[ascendnpu_ir_compile_chain]]) — same approach as my bf16 #1199 fix. Faster than a full CANN upgrade, validates fp8 end-to-end before committing to an image bump.
- **This is the bf16-#1199 pattern, confirmed fixable** (not a closed wall). My whole open tilelang stack (parse/MLIR-type/codegen) is already done → once the verifier allow-list has float8 (via upgrade OR td-patch+rebuild), fp8 tilelang kernels should compile.

## image-upgrade-drill Step-1 discovery: precise CANN/AscendNPU-IR target for fp8 (2026-06-03)
Pinned the exact version boundary (the owner's "find suitable CANN" deliverable):
- **fp8 added to AscendNPU-IR**: commit e5e7c48 "fix fp8 support" **2025-12-30**, merged via !403 (3ea05c9) "merge fix-fp8 into master" **2026-01-20**.
- **Container's bishengir** (CANN 8.5.0): version 0.1.0, built **2026-01-16**.
- **Fork's vendored AscendNPU-IR**: commit `31f69036`, **2026-01-14**.
- → **BOTH predate the fp8 merge (2026-01-20) by ~4–6 days.** That's the entire reason fp8 isn't in the verifier allow-list. **Target = any CANN whose bishengir is built from AscendNPU-IR master ≥ 2026-01-20 (≥ commit 3ea05c9 / !403).** A very small version delta — current is ~4 days too old.
- **Recommendation for the upgrade**:
  - Path 1 (owner's, CANN image upgrade): need a CANN release whose bishengir snapshot is ≥ 2026-01-20 AscendNPU-IR. CANN 8.5.1 (vllm-ascend already moved to it) is the first candidate — verify its bishengir build date / AscendNPU-IR commit ≥ Jan-20 via `npu-image-inspect` on the 8.5.1 image before committing.
  - Path 2 (local, fast validation): bump the fork's `3rdparty/AscendNPU-IR` submodule to ≥ 3ea05c9 (or cherry-pick the 2-commit fp8 diff into the vendored `.td`s), rebuild bishengir. Validates fp8 end-to-end in hours without an image swap; de-risks Path 1.
- **Both are heavy execution + the owner explicitly chose "upgrade CANN" → holding for owner's pick of path before launching either** (a bishengir/AscendNPU-IR rebuild or a CANN image pull are both substantial; not launching unprompted against the stated direction).

## Refined upgrade target (2026-06-03): CANN 9.0.0
- Latest vllm-ascend uses **CANN 9.0.0** (image `quay.io/ascend/cann:9.0.0-910b-ubuntu22.04-py3.12`). 9.0.0 ≫ fp8 merge (2026-01-20) → **its bishengir definitely has fp8**, and likely also fixes #100 PassManager segfault + more gemm/Cube codegen (a newer bishengir could clear several of this session's walls at once).
- **Caveat (honest)**: that tag is **910b (Atlas A2)**; our target is **A3 (V220 / 910_9382)** → need the A3/V220 variant of CANN 9.0.0 (or aarch64/A3 tag). Verify before swapping.
- Target refined: "≥ CANN 8.5.1" → **CANN 9.0.0 (find A3 variant)**. Current standard, real image, confirmed fp8.
- **Execution gate (owner auth needed)**: actually swapping the container base image is heavy + affects the SHARED base that tlrescue + sgl_probe both run on (others' in-flight work) → outward/hard-to-reverse, must NOT do unprompted. Non-destructive discovery (npu-image-inspect on the 9.0.0 A3 image) is fine to continue; the base swap / bishengir rebuild waits for owner go-ahead.

## ★★ FP8 TRUE ROOT CAUSE (2026-06-03, CANN 9.1.0-beta.1 verified) — A3 HARDWARE doesn't support fp8
Pulled `quay.nju.edu.cn/ascend/cann:9.1.0-beta.1-a3-ubuntu22.04-py3.12-devel` (11.9GB) and ran its **bishengir-compile 1.1.0** (built 2026-05-09, AscendNPU-IR 7058cef3) on the same fp8 npuir that failed on 8.5.0:
- 8.5.0 bishengir 0.1.0 error: `'hivm.hir.store' op ... should have element type [list without float8]` (verifier allow-list omission).
- **9.1.0 bishengir 1.1.0 error: `'hivm.hir.store' op Current hardware doesn't support fp8 type`** ← DIFFERENT, and decisive.
- → 9.1.0's verifier **DOES** accept fp8 in the type allow-list (the upstream fix is in) — it gets PAST the type check, then rejects on a **hardware-capability check**: **A3 (V220 / 910_9382) hardware does not support fp8.**
- **CONCLUSION CHANGE (honest, conclusion-flipping)**: fp8 on A3 is **NOT a fixable software/verifier gap** — it's an **A3 hardware limitation**. act_quant / fp8_gemm / fp4_quant / fp4_gemm fp8-paths **fundamentally cannot run on A3 regardless of CANN version or any of my open-stack fixes.** The whole "fp8 dtype-gap / verifier allow-list" line was a software symptom; the real floor is hardware.
- **Signal that A5 may support fp8**: the 9.1.0 image ships `bishengir-compile-a5` / `hivmc-a5` (A5-specific). A5 (arch35, the newer chip) likely has fp8 hardware. So fp8 tilelang is plausibly an **A5 target**, not A3.
- **Why verification mattered**: had I stopped at "upgrade CANN → fp8 unblocked" (the open-stack + upstream-allow-list reasoning all pointed there), I'd have shipped a WRONG conclusion. Running the actual 9.1.0 bishengir surfaced the hardware wall. (verify_meaning_not_just_mechanics.)

## Build + test loop (my own branch — safe)
- cmake fix: `ln -sf /usr/bin/cmake /usr/local/python3.11.14/lib/python3.11/site-packages/cmake/data/bin/cmake` (the Makefile's hardcoded pip-cmake path is gone in sgl_probe).
- backup of _dev.cc before my nd2nz edit: `/tmp/codegen_npuir_dev.cc.bak_nd2nz`.
- Rebuild: in `/home/z00637938/workspace/tilelang-mlir-ascend/build`, `ninja` (or cmake --build). **Watch for zombie-ninja** ([[tilelang_bf16_three_verifier_layers]]: ninja install is the real install step; pgrep ninja before). Rebuild touches the SHARED .so but it's MY branch's build — fine.
- Test matmul: `python3 /tmp/run_matmul2.py` (runpy of examples/gemm/matmul.py, sys.path has workspace, ASCEND_RT_VISIBLE_DEVICES=0, TILELANG_ASCEND_MODE=Developer). PASS = "All check passed!".
- Test sparse_mla: `python3 /tmp/run_mla_asis.py`.
- After both compile: measure gemm/attention perf (the owner's original goal) vs torch.

## Why this matters
gemm/Cube compile is blocked ONLY by these codegen bugs (my own unfinished fix). Fixing → matmul/sparse_mla compile → unblocks gemm/attention perf measurement (currently only sinkhorn/vector measurable, 0.06–0.45× torch #100-locked).
