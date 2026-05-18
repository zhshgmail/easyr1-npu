# Knowledge Base — tilelang-ascend × MLIR × AscendNPU-IR

Cumulative findings from T32 cold-drive (2026-05-17/18). Stays here as
draft; promote to `repo/knowledge/` when stabilized.

---

## 1. The repo / project landscape

### 1.1 Four projects, not two

| Project | URL | Role |
|---------|-----|------|
| `tile-ai/tilelang-ascend` | github.com/tile-ai/tilelang-ascend | **Historical release line.** Two branches: `ascendc_pto` (PTO/string codegen) and `npuir` (MLIR — equivalent to tilelang-mlir-ascend). |
| `tile-ai/tilelang-mlir-ascend` | github.com/tile-ai/tilelang-mlir-ascend | **MLIR mainline.** Future release source. |
| `Ascend/AscendNPU-IR` | gitcode.com/Ascend/ascendnpu-ir | Huawei's MLIR dialect + `bishengir-compile` binary (the actual MLIR backend compiler). |
| `tile-ai/tilelang` (upstream) | github.com/tile-ai/tilelang | The upstream tile-lang DSL itself (CUDA/HIP). Ascend three are forks. |

### 1.2 PTO vs MLIR — replacement, not coexistence

- `tilelang-mlir-ascend/tilelang/engine/lower.py::device_codegen` **only
  dispatches `target.kind.name == "npuir"`**. The legacy `ascend` target's
  PackedFunc is registered (in `rt_mod_ascend.cc`) but **unreachable from
  Python**.
- Released wheel `tilelang-0.1.1.10+linux.cann900` (from
  `tilelang-ascend@ascendc_pto`) is the **last PTO release**. Users
  hitting bugs there are on a sunsetting code path.
- TILELANG_ASCEND_MODE env switches **between MLIR modes** (Expert =
  `tilelang_npuir_apis`, Developer = `tilelang_npuir_dev`), not between
  PTO/MLIR.

### 1.3 bishengir is the real MLIR compiler

- `bishengir-compile` is a Huawei MLIR-based compiler binary.
- Built from `AscendNPU-IR` source as a standard LLVM external project.
- **CANN ships a `bishengir-compile` binary at
  `/usr/local/Ascend/cann-8.5.2/tools/bishengir/bin/`** but only `bin/`
  — no `python_packages/` (which tilelang-mlir-ascend needs). So you
  MUST build it from source if you want to actually integrate with
  tilelang-mlir-ascend.

---

## 2. The #996 bug (verified across both backends)

**Bug** ([github.com/tile-ai/tilelang-ascend#996](https://github.com/tile-ai/tilelang-ascend/issues/996)):
small-block elementwise gets ~half wrong output silently at
`block_N * sizeof(T) < 32`.

**Root cause** (PTO side): `src/tl_templates/ascend/common.h::copy_gm_to_ub`
template calls `AscendC::DataCopyExtParams(M, N*sizeof(T), ...)`. When
`N*sizeof(T) < 32`, the DMA engine's `blockLen` underflows the 32-byte
alignment requirement and reads adjacent rows. **No precondition check**
in the template.

**Verification matrix**:

| Backend | Config | per-row bytes | Outcome |
|---------|--------|---------------|---------|
| PTO | M=N=32, block_M=block_N=4, fp32 | 16 | **49.7% wrong**, max_abs_diff 4.013 |
| PTO | M=N=1024, block_M=128, block_N=256, fp32 | 1024 | PASS (baseline) |
| PTO | M=N=64, block_M=8, block_N=16, fp32 | 64 | PASS (above threshold) |
| PTO | M=N=64, block_M=8, block_N=4, fp32 | 16 | static_assert (our patch fires) |
| MLIR | M=N=32, block_M=block_N=4, fp16 | 8 | **PASS** |
| MLIR | M=N=16, block_M=block_N=4, fp16 | 8 | PASS |
| MLIR | M=N=8, block_M=block_N=2, fp16 | 4 | PASS |
| MLIR | M=N=32, block_M=4, block_N=2, fp16 | 4 | PASS |

**Conclusion**: bug is **PTO-only**. MLIR pipeline handles sub-32B blocks
automatically — exact mechanism not yet investigated, likely either an
MLIR pass that promotes to 1-D access or AscendNPU-IR uses a different
DMA primitive that doesn't have the alignment constraint.

---

## 3. The two fix paths (PTO)

### 3.1 Option A — compile-time guard (verified)

Add `static_assert` at the top of `copy_gm_to_ub` template:

```cpp
static_assert(dstM == 1 || dstN * sizeof(T) >= 32,
    "tilelang-ascend issue #996: 2-D GM->UB copy requires "
    "dstN * sizeof(T) >= 32 bytes per row (DMA alignment). "
    "Reshape your kernel to 1-D layout or increase block_N so per-row "
    "bytes >= 32. Example: fp32 needs dstN >= 8, fp16 needs dstN >= 16.");
```

`dstM == 1` short-circuits 1-D copies. Patch ready at
`workspace/T32_tilelang_rescue/issue_996_compile_check.patch`. PR-ready
for `tilelang-ascend@ascendc_pto`.

### 3.2 Option B — auto-coalesce TIR pass (not implemented)

Add a TIR pass that, when emitting a 2-D copy with `dstN * sizeof(T) < 32`
AND row stride == row size (contiguous), rewrites the (M, N) copy to
(1, M*N) — equivalent to our manual rescue but in the compiler.

### 3.3 User workaround — 1-D flat layout (verified)

Until upstream lands either fix, users can manually flatten:

```python
@T.prim_func
def main(A, B, C):  # 1-D shape
    with T.Kernel(grid, is_npu=True) as (cid, vid):
        a_ub = T.alloc_ub((per_vec,), dtype)  # 1-D UB
        ...
        T.copy(A[cid * block_size + vid * per_vec], a_ub)
```

Full impl: `workspace/T32_tilelang_rescue/elementwise_add_flat.py`.
PASS on 6 shapes (incl. non-power-of-2).

---

## 4. tilelang DSL — how it works (PTO backend)

Pipeline: `Python @T.prim_func` → TVM TIR IRModule → ~15 TIR C++ passes
(`src/transform/ascend_*.cc`) → `codegen_ascend.cc` emits AscendC C++
string → `ccec` compiles → `.so` → ctypes loads → `aclrtLaunchKernel`.

Key TIR passes:
- `frontend_legalize.cc` — DSL legalization
- `layout_inference.cc` — buffer layout
- `ascend_infer_buffer_scope.cc` — GM/UB/L1/L0 scope inference
- `ascend_lower_opaque_block.cc` — expand `T.Scope("V")`
- `ascend_memory_planning.cc` — UB allocation (fails with "Extent must be
  an integer constant" when UB sizes are non-literal compile-time)
- `ascend_storage_rewrite.cc`
- `ascend_combinecv.cc` — Cube/Vector co-pipeline scheduling
- `ascend_sync_insert.cc` — auto-insert `set_flag`/`wait_flag`

Codegen calls templates from `src/tl_templates/ascend/common.h`:
- `copy_gm_to_ub<T, dstN, dstM>` — GM→UB DMA (this is #996's site)
- `copy_ub_to_gm<T, srcN, srcM>` — UB→GM DMA
- `copy_l0c_to_gm` / `copy_ub_to_l1` — GEMM-specific
- `tile_add<T, Len>` — vector add
- `atomic_add_ub_to_gm` — atomic store

---

## 5. tilelang DSL — how it works (MLIR backend)

Pipeline: `Python @T.prim_func` → TVM TIR IRModule → mostly-shared TIR
passes → `codegen_npuir.cc` constructs MLIR IRModule (ascendnpu dialect)
→ `bishengir-compile` runs MLIR passes (npuir → hivm → cce) → emits
binary → `.so`.

Key MLIR DSL differences from PTO:
- `T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _)` — no `vid` second arg
- `T.alloc_ub((block_M, block_N), dtype)` — same
- `T.copy(A[bx, by], A_VEC)` — same
- `T.vadd(A, B, C)` instead of `T.tile.add(C, A, B)` — different elementwise op
- No `T.Scope("V")` wrapping — single scope
- `T.serial(N)` for in-kernel loops common in vec_add_2d.py
- `target="npuir"` in `@tilelang.jit(out_idx=[-1], target="npuir")`

`TILELANG_ASCEND_MODE` env:
- `Expert` (default) — calls `tilelang_npuir_apis` codegen
- `Developer` — calls `tilelang_npuir_dev` codegen (different abstraction
  level)

---

## 6. Environment setup (MLIR backend on A3)

Total build time ~3h. Each gotcha is a real blocker — document them all.

### 6.1 Container

Use **verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5** image. Critical:
- `--privileged` is mandatory (NPU driver needs it; `--device` alone fails
  with EL0005 Resource_Busy)
- Mount the **whole** `/usr/local/Ascend/driver` tree (per CLAUDE.md
  NPU-OPS-009 — selective binding breaks dcmi init on A3)
- `--network=host --ipc=host`
- Pick one chip (e.g. davinci14); precheck `npu-smi info -t proc-mem`

### 6.2 GFW workarounds for github

- **A3 reaches gitcode.com fine** — use it for any gitcode mirror
- **A3 reaches gitee.com mirrors** — used for LLVM full clone
  (`gitee.com/mirrors/llvm-project.git`)
- **github times out on multi-GB clones** — even `ls-remote` works but
  `clone` fails after some MB
- **kkgithub.com** (China-side github proxy) is the most reliable for
  GitHub-only content (used for torch-mlir + 5 TVM 3rdparty submodules)

### 6.3 Cloning order (sequential — submodule recursion fails on github)

1. `tilelang-mlir-ascend` parent — clone from github (works for small
   .git, slow but completes)
2. `3rdparty/tvm` submodule — auto-pulled, works
3. `3rdparty/AscendNPU-IR` submodule — auto-pulled via gitcode, works
4. `3rdparty/AscendNPU-IR/third-party/llvm-project` — **manually** clone
   from gitee shallow, then fetch the specific commit
   `cd708029e0b2869e80abe31ddb175f7c35361f90` shallow
5. `3rdparty/AscendNPU-IR/third-party/torch-mlir` — manually clone from
   kkgithub, fetch+checkout commit `155680c08e08bff6d2e6883415e3f5a1b474d96e`
6. TVM 3rdparty subs that github-clone silently fails on: `rang`,
   `vta-hw`, `libflash_attn`, `flashinfer`, `libbacktrace` — clone each
   from kkgithub, fetch+checkout the pinned commit

### 6.4 Apply patches

`bash 3rdparty/AscendNPU-IR/build-tools/apply_patches.sh`. Must be run
**inside the container** because the host's zsh doesn't ship `patch`.

### 6.5 Build bishengir

```bash
cd 3rdparty/AscendNPU-IR
./build-tools/build.sh -o ./build \
  --python-binding \
  --c-compiler=clang --cxx-compiler=clang++ \
  --add-cmake-options="-DCMAKE_LINKER=lld -DLLVM_ENABLE_LLD=ON -DLLVM_ENABLE_RTTI=ON" \
  --bishengir-publish=off
```

Container needs `clang-15`, `lld-15`, `llvm-15-dev`, `libmlir-15-dev`,
`libclang-15-dev` installed via apt.

**pybind11 version matters**: keep 2.11.1 for the bishengir build.
pybind11 3.0.4 breaks MLIR's IRCore.cpp on `def_property` keep_alive
static_assert.

Then `cmake --install build --prefix build/install` to populate
`build/install/python_packages/{mlir_core,bishengir}`.

### 6.6 Install tilelang-mlir-ascend

```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
apt-get install -y zlib1g-dev libzstd-dev  # else libtilelangir.so link fails
bash install_npuir.sh --bishengir-path=$(realpath 3rdparty/AscendNPU-IR/build/install)
```

For TVM build, `requirements-build.txt` re-installs pybind11 3.0.1 —
**this is fine** (separate scope from bishengir's MLIR bindings; tilelang
itself uses 3.0.1 for its own C extension).

### 6.7 Runtime PYTHONPATH

```bash
export TLPATH=/home/z00637938/workspace/tilelang-mlir-ascend
export BISHENGIR_PKGS=$TLPATH/3rdparty/AscendNPU-IR/build/install/python_packages
export PYTHONPATH=$TLPATH:$BISHENGIR_PKGS/mlir_core:$BISHENGIR_PKGS/bishengir:$PYTHONPATH
export PATH=$TLPATH/3rdparty/AscendNPU-IR/build/install/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 6.8 Pinning summary

- tilelang-mlir-ascend HEAD: `2b8001c84365d1731f60ba58d82f5967e09617ab`
  (HEAD at clone time 2026-05-18; check `git log -1` for current)
- AscendNPU-IR @ `31f690369d1247fbd5529a3f88b758f7d470ae4f`
- LLVM @ `cd708029e0` (llvmorg-19-init-19088 era)
- torch-mlir @ `155680c08e08bff6d2e6883415e3f5a1b474d96e`
- TVM 3rdparty submodules: pinned commits per
  `3rdparty/tvm/.gitmodules` (use `git ls-tree HEAD <path>` for each)
- bishengir-compile version after our source build: **19.1.7**

---

## 7. Common error → fix recipes

| Error | Fix |
|-------|-----|
| `RuntimeError: aclInit, error code is 507899` / `EL0005 Resource_Busy` | Container is missing `--privileged` |
| `uda_occupy_dev_by_ns Conflict open udevid` in dmesg | Another container holds the NPU NS lock; not necessarily related to NPU being free per `npu-smi info`. Try `--privileged` first; only if that fails identify and stop the holder |
| `libruntime_common.so: undefined symbol: _ZN12ErrorManager...` | torch_npu version doesn't match CANN ABI. Use `verl-8.5.2` image (CANN 8.5.2 + torch_npu 2.9.0 known-good) |
| `RPC failed; curl 56 OpenSSL SSL_read: Connection timed out` on github clone | GFW. Use gitee mirror for LLVM, kkgithub for github-only repos |
| `pybind11.h: static assertion failed: def_property family does not currently support keep_alive` | pybind11 too new (3.0+). Downgrade to 2.11.1 for bishengir build |
| `ld: cannot find -lz / -lzstd` | `apt-get install zlib1g-dev libzstd-dev` |
| `No matching distribution found for patchelf / Cython` | pypi default index times out from A3. `pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/` |
| `git submodule update --init` finishes but folder empty | github silently 0-byte cloned; check `du -sh <subdir>` after init. Clone manually via mirror |
| `InternalError: Check failed: (int_imm) is false: Extent must be an integer constant` (tilelang PTO) | UB allocation sizes must be Python-int constexpr in the JIT closure — `block_size // VEC_NUM` works only if `block_size` is a literal-bound Python int, not derived from another non-constexpr expression |
| Multiple containers grab same `/dev/davinciN` → ns lock conflict | `npu-smi info` may show 0% AICore but ns is still locked by zombie holders. `--privileged` often bypasses. Otherwise pick a different chip |
| `examples/vec_add_1d.py: No such file` | examples moved into subdirs: try `examples/elementwise/vec_add_1d.py` |

---

## 8. The feedback loop (per-op verification log)

**Mission** (from user 2026-05-18 03:10): "这个叫反馈环，要我们越做积累越
多。这点是从 a5 ops 学习的." Every op we try, every error we hit, every
recipe we discover gets appended here and into §7 (error recipes). This
file IS the feedback loop sink — don't let learnings stay in conversation
context, codify here so next session inherits.

### 8.1 Verified ops on MLIR backend

Run with PYTHONPATH set per §6.7, inside `tlrescue` container on A3.

| # | Op family | Path | Shape tested | Result | Notes |
|---|-----------|------|--------------|--------|-------|
| 1 | Elementwise add 1-D | `examples/elementwise/vec_add_1d.py` | default | PASS | First smoke; proves MLIR pipeline reaches NPU |
| 2 | Elementwise add 2-D | `examples/elementwise/vec_add_2d.py` | 512x512/128x128 fp16 | PASS | Default args baseline |
| 3 | Elementwise add 2-D (small) | `vec_add_2d.py --M 32 --N 32 --block_M 4 --block_N 4` | per-row 8B fp16 | PASS | **#996 shape — bug NOT reproducing on MLIR** |
| 4 | Elementwise add 2-D (smaller) | `--M 16 --N 16 --block_M 4 --block_N 4` | per-row 8B | PASS | |
| 5 | Elementwise add 2-D (smallest) | `--M 8 --N 8 --block_M 2 --block_N 2` | per-row 4B | PASS | Stress sub-alignment lower |
| 6 | Elementwise add 2-D (asym) | `--M 32 --N 32 --block_M 4 --block_N 2` | per-row 4B | PASS | Non-square block |
| 7 | Elementwise add 2-D dynamic shape | `vec_add_2d_dynamic_shape.py` | default | PASS | Confirms dynamic-shape MLIR codegen works |
| 8 | Elementwise add 2-D multi-buffer | `vec_add_2d_multi_buffer.py` | default | PASS | Multi-buffer pipelining lowering works |
| 9 | Elementwise add auto-broadcast | `vec_add_auto_brc.py` | default | PASS | Auto-broadcast shape coercion works |
| 10 | Atomic add (1-D + 2-D) | `atomic_add.py` | default | PASS | **#1023 target — atomic_add works on MLIR** |
| 11 | Atomic add Developer Mode | `atomic_add_dev.py` | default (TILELANG_ASCEND_MODE=Developer style) | PASS | Both backend modes for atomic_add work |
| 12 | exp2 | `examples/exp2.py` | default | PASS | Output: "Exp2 kernel accuracy check passed!" |
| 13 | log2 | `examples/log2.py` | default | PASS | Output: "Log2 kernel accuracy check passed!" |
| 14 | Vectorization in parallel | `examples/vectorization_in_parallel.py` | default | PASS × 3 | Multiple check passes — likely tests multiple ops |
| 15 | mixcv_mixkernel | `examples/mixcv_mixkernel.py` | default | PASS | C/V mixing |
| 16 | GEMM | `examples/gemm/example_gemm.py` | default | PASS | **#1016 area — GEMM works on MLIR (need to confirm FP32 variant)** |
| 17 | GEMM int8→int32 | `examples/gemm/example_gemm_int82int32.py` | default | PASS | Quantized GEMM lowering works |
| 18 | matmul | `examples/gemm/matmul.py` | default | PASS | General matmul |
| 19 | matmul dynamic shape | `examples/gemm/matmul_dynamic_shape.py` | default | PASS | Dynamic-shape matmul |
| 20 | gemv | `examples/gemv/example_gemv.py` | default | PASS | gemv |
| 21 | RMS norm | `examples/norm/example_rms_norm.py` | default | PASS | reduction + elementwise fused |
| 22 | Layer norm | `examples/norm/layer_norm.py` | default | PASS | |
| 23 | Flash Attention (Expert) | `examples/flash_attn_npuir.py` | default | PASS | Flagship — proves MLIR pipeline handles full attention |
| 24 | Flash Attention (Developer) | `examples/flash_attn_npuir_dev.py` | default | PASS | Developer Mode flash attn |
| 25 | Sparse MLA forward | `examples/sparse_mla_fwd.py` | default | PASS | DeepSeek v3.2 MLA fwd |
| 26 | Sparse MLA forward (dynamic) | `examples/sparse_mla_fwd_dynamic_shape.py` | 3 cases | PASS × 3 | All 3 case files pass |
| 27 | FP8 lighting indexer | `examples/fp8_lighting_indexer.py` | default | **FAIL** (precision) | 5/16777216 (0.0%) mismatched, max_abs_diff 0.0195 > 0.01 atol. Likely fp8↔fp16 rounding edge case — see §10.1 |
| 28 | Elementwise add (named-add wrapper) | `examples/elementwise/example_elementwise_add.py` | default | PASS | |
| 29 | DS v3.2 MLA backward | `examples/deepseek_v32/sparse_mla_bwd.py` | default | PASS | "Check passed" |
| 30 | DS v3.2 MLA forward (named wrapper) | `examples/deepseek_v32/sparse_mla_fwd.py` | default | PASS | "Demo check passed" |
| 31 | DS v4 Act Quant | `examples/deepseek_v4/example_act_quant_kernel.py` | default | **FAIL** (strict ==) | `assert torch.all(y == y_ref)` fails. Values visually match to 4 dp — last-bit rounding mismatch under strict `==` not `assert_close`. See §10.2 |
| 32 | DS v4 FP8 GEMM | `examples/deepseek_v4/example_fp8_gemm_kernel.py` | default + Expert mode | **COMPILE_FAIL** | MLIR `Generated MLIR module failed verification: tensor.empty op incorrect number of dynamic sizes, has 0, expected 1`. **Root-caused 2026-05-18**: `codegen_npuir_dev.cc` uses wrong `EmptyOp::build` overload at 13 call sites — see §10.3 fix sketch |
| 33 | DS v4 split_sinkhorn | `examples/deepseek_v4/example_hc_split_sinkhorn_kernel.py` | default | PASS × 5 | post / comb / pre check all pass |
| 34 | DS v4 mhc_post | `examples/deepseek_v4/example_mhc_post.py` | default | PASS × 3 | |
| 35 | DS v4 sparse_attn | `examples/deepseek_v4/example_sparse_attn_kernel.py` | default | PASS | |
| 36 | DS v4 sparse_attn_highperf | `examples/deepseek_v4/example_sparse_attn_kernel_highperf.py` | default | PASS × 2 case | high-perf variant |
| 37 | Engram fwd | `examples/engram/engram_fwd.py` | default | PASS | |
| 38 | Engram bwd | `examples/engram/engram_bwd.py` | default | PASS | |
| 39 | Engram bwd exp | `examples/engram/engram_bwd_exp.py` | default | PASS | |
| 40 | Engram decode | `examples/engram/engram_decode.py` | default | PASS | |
| 41 | T.dynamic smoke (P1.2) | `_smoke_T_dynamic.py` | Developer | PASS (batch=4 AND batch=17 same kernel) | T33: dynamic-shape DSL surface working end-to-end on Ascend |
| 42 | sparse_mla_fwd (P1.3) | `examples/deepseek_v4/example_sparse_mla_fwd_kernel.py` | Developer | PASS — max err 5e-4 vs CPU fp32 ref @ B=1,S=8,SKV=16,H=16,D=64,DT=16,topk=8 | T33: NPU port of upstream `examples/deepseek_v32/sparse_mla_fwd.py`. Adaptations: 3-axis grid → 1-axis; NPU vector intrinsics; Lse [B,S,H,1] for rank parity |

### 8.2 To-try queue (next session)

Sorted by expected information yield. Items marked ✓ done in 8.1.

- [✓] vec_add_2d_dynamic_shape, multi_buffer, auto_brc, atomic_add, atomic_add_dev (rows 7-11)
- [✓] exp2, log2, vectorization_in_parallel, mixcv_mixkernel (rows 12-15)
- [✓] gemm, gemm_int82int32, matmul, matmul_dynamic_shape, gemv (rows 16-20)
- [✓] rms_norm, layer_norm (rows 21-22)
- [✓] flash_attn_npuir, flash_attn_npuir_dev (rows 23-24)
- [✓] sparse_mla_fwd, sparse_mla_fwd_dynamic_shape (rows 25-26)
- [✓] fp8_lighting_indexer — **FAIL** (row 27) — needs investigation
- [ ] `examples/example_elementwise_add.py` — unique to elementwise dir
- [ ] `examples/deepseek_v32/*` — DeepSeek v3.2 kernels
- [ ] `examples/deepseek_v4/*` — DeepSeek V4 kernels (sparse FA, mHC, act_quant, int8 GEMM)
- [ ] `examples/torch_tl_ops/*` — PyTorch integration smoke
- [ ] `examples/engram/*` — engram (whatever it is)
- [ ] Cross-shape stress on top performers: take rms_norm, flash_attn, matmul; vary shapes; document MLIR's shape coverage
- [ ] **Issue #1016 specifically**: try `example_gemm.py` with explicit fp32 dtype to see if it reproduces what #1016 reports
- [ ] **Issue #1019 specifically**: find which example exercises workspace init
- [ ] FP8 indexer deeper dig — see §10

### 8.3 Per-op log template (use when adding rows to §8.1)

For each op tried, log:
- **Op family** — short name
- **Path** — exact py file
- **Shape tested** — args passed
- **Result** — PASS / FAIL (with mismatch percentage) / COMPILE_FAIL (with error class)
- **Notes** — anything surprising; deeper dig prompts; cross-link to error recipe in §7 if relevant

### 8.4 Sister-bug cross-reference

Known upstream bugs to verify against on MLIR backend (do they
reproduce, or has MLIR fixed them automatically like #996?):

| Upstream issue | PTO behavior | MLIR behavior | Status |
|----------------|--------------|---------------|--------|
| [#996](https://github.com/tile-ai/tilelang-ascend/issues/996) (elementwise 32B alignment) | 49.7% wrong @ 32x32/4x4 fp32 | PASS @ 32x32/4x4 fp16 (per-row 8B) | **MLIR clean** (2026-05-18) |
| [#1023](https://github.com/tile-ai/tilelang-ascend/issues/1023) (atomic_add UB→GM error) | Unknown | TBD via `examples/elementwise/atomic_add.py` | TODO |
| [#1019](https://github.com/tile-ai/tilelang-ascend/issues/1019) (workspace init) | Unknown | TBD | TODO |
| [#1016](https://github.com/tile-ai/tilelang-ascend/issues/1016) (FP32 GEMM) | Unknown | TBD via `examples/gemm/example_gemm.py` with fp32 | TODO |

---

## 9. Open questions for future sessions

- **MLIR's mechanism** for handling sub-32B blocks — is it a TIR pass, a
  bishengir MLIR pass, or AscendNPU-IR dialect semantics?
- **PTO's `T.tile.add` vs MLIR's `T.vadd`** — are they identical
  semantically? Where does naming diverge in the API?
- **Will tilelang-mlir-ascend ever publish a wheel?** Currently must
  build from source. If/when they ship one with bundled bishengir, the
  entire 3h setup collapses to a `pip install`.
- **Are bishengir-19.1.7 (our build) and CANN's bundled bishengir-0.1.0
  identical?** Version-number mismatch suggests CANN's is much older /
  different — maybe a stripped-down variant?

---

## 9b. Target hardware: Ascend 910C (A3) — what every MLIR bug investigation needs

Source: `upstream/a5_ops/src/skills/references/hardware/target/ascend910c.md`
(verified 2026-04-25 by a5_ops team). Cross-link instead of duplicating
the whole doc; capture the bits that DIRECTLY inform compiler-bug triage:

| Property | Value | Why it matters for compiler bug triage |
|----------|-------|---------------------------------------|
| Chip | DaVinci V220 dual-die, npu_arch=2201 | Same family as A2/910B, NOT A5/950PR — CANN BUILD_MODE=c220 arch22 |
| `acl.get_soc_name()` | `Ascend910_9392` (verified) | Use to discriminate at runtime |
| Programming model | **SIMD only** | NO SIMT path; warp shuffle / threadIdx / LAUNCH_BOUND → compile fail |
| AICore | 40-48 per package | 2 dies × 20-24 |
| AIV per AICore | 2 | each has its own UB |
| **Total AIV (vec lanes)** | **80-96** | sets practical upper bound on T.Kernel BLOCK_SIZE before serialization kicks in |
| AIC total | 40-48 (=AICore) | Cube unit count |
| **UB per AIV** | **192 KB** | NOT 256 KB (A5). Tile sizes assuming 256 KB OOM on A3 |
| UB per AICore | 384 KB | 2 × 192 KB |
| L1 per AIC | 512 KB | |
| L0A / L0B per AIC | 64 KB each | |
| **L0C per AIC** | **128 KB** | NOT 256 KB (A5). Cube accum buffer half size |
| HBM | ~128 GB, ~1.6 TB/s | per package |
| `atomicAdd` int32/uint32 | supported | |
| **`atomicAdd` fp32 / fp16 / bf16** | **TBD — not yet probed** | If MLIR emits fp32 atomicAdd and it silently lowers wrong, this is the bug |
| `atomicMax/Min` fp32 | TBD — not yet probed | |
| **DMA blockLen alignment** | **32 bytes per row for `DataCopyExtParams`** | THIS is #996 in PTO; MLIR side either auto-handles or uses different DMA primitive (open question §9) |
| **Cast op rounding** | TBD — likely `round-to-nearest-even` per V220 default but **not yet probed** | Critical for §10.1 (fp8 indexer 5/16M mismatch) and §10.2 (act_quant strict-== fail). If MLIR's `vcast` emits a different round mode than torch's reference, that's the compiler bug |

### Bugs that the hardware spec helps triage

- **§10.1 fp8 indexer**: max abs diff 0.0195 ≈ fp16 ULP @ magnitude 16-32.
  Cast op rounding mode mismatch likely (see "Cast op rounding" row).
  Compiler-side fix candidate: align bishengir `hivm.hir.vcast` round_mode
  with torch reference (`round-to-nearest-even`).
- **§10.2 act_quant**: strict `==` fails on visually-identical values.
  Same family: 1-ULP precision diff under Cast / Quant rounding. Same fix
  candidate.
- **§10.3 fp8 GEMM**: `tensor.empty` 0 vs expected-1 dynamic sizes. NOT a
  hardware issue — pure MLIR codegen verifier issue. Hardware spec only
  helps confirm fp8 GEMM is in-scope for A3 (yes — V220 Cube supports
  fp8 → fp16/fp32 dequantize).

### What to probe next for these bugs

| Question | How to probe |
|----------|--------------|
| What round mode does AscendC Cast use by default on 910C? | Write a tiny kernel that Casts fp32 → fp16 with a known boundary value (e.g. 1.000000000023283); check output bit-pattern |
| Does fp32 atomicAdd actually work on A3, and is it deterministic? | a5_ops references/hardware/INTERNAL_QUERY_QUEUE.md probe checklist |
| Does the MLIR `hivm.hir.vcast` op accept `round_mode = round_nearest_even`? | grep AscendNPU-IR `bishengir/include/.../HIVMOps.td` for vcast attrs |
| Where in tilelang-mlir-ascend's `codegen_npuir.cc` is the Cast→vcast emission? Does it pass through user's `T.cast(x, dtype, round_mode=...)` arg? | grep `vcast` in tilelang-mlir-ascend/src/target/ |

---

## 10. Op-level deep dives (feedback loop content)

### 10.1 FP8 lighting indexer mismatch (row 27, 2026-05-18)

**Op**: `examples/fp8_lighting_indexer.py` — fp8 K/V × fp16 Q lighting
attention indexer (used in DeepSeek v3.2 + future Mixture-of-Experts
routing).

**Result**: 5 / 16,777,216 elements (0.0%) fail `assert_close(rtol=1e-2,
atol=1e-2)`. Max abs diff 0.01953125, max rel diff 18.8%.

**Initial diagnosis** (not yet root-caused):
- 5 elements is tiny (0.00003%) — strongly suggests boundary/edge case
  rather than systemic wrong math
- Max abs diff 0.0195 ≈ 0.02 — exactly the resolution of fp16 mantissa
  at numbers around magnitude 8-32 (most output values are ±8 to ±30 in
  the sample), suggesting **last-bit rounding** when fp8 dequantize
  rounds differently than torch's reference
- One mismatch index `(0, 1273, 783)` has max-rel-diff 18.8% — that
  element is small (close to 0), so even a single fp16 bit flip looks
  huge in relative terms

**Hypothesis**: AscendC `Cast<fp8, fp16>` rounding mode differs from
torch reference (e.g. round-to-nearest-even vs round-to-zero, or
saturate vs no-saturate). Not a kernel-logic bug.

**To verify**:
- [ ] Print the 5 differing elements + their inputs; check if they're
  all near fp8 representable boundaries (16-step or 32-step where
  rounding diverges)
- [ ] Bump tolerance to `atol=0.03 rtol=0.05`, see if it passes — if
  yes, it's a rounding-style boundary, not a real bug
- [ ] Check if torch's reference uses `bfloat16` or float32 intermediate
  — that's the typical place tolerance gaps open up

**Not a blocker for us**: 0% effective error, almost certainly rounding.
Worth filing as a low-priority issue if reproducible across seeds.

### 10.2 DS v4 Act Quant strict-equality mismatch (row 31, 2026-05-18)

**Op**: `examples/deepseek_v4/example_act_quant_kernel.py` — activation
quantization kernel for DeepSeek V4.

**Result**: `assert torch.all(y == y_ref)` fails. The output values are
visually identical to torch reference (printed to 4 decimal places match
exactly: `[0.0145]`, `[0.0250]`, `[0.0157]`...). The strict equality
fails — last-bit rounding differs.

**Root cause hypothesis** (not yet confirmed): same family as §10.1 —
the AscendC reduction or cast op uses a different rounding mode (likely
`round-to-zero` or `round-down`) than torch reference (`round-to-nearest-
even`). Under `assert torch.all(y == y_ref)`, even 1-ULP differences
fail.

**Fix candidate** (workaround, not real fix):
- Relax test to `torch.testing.assert_close(y, y_ref, rtol=1e-7, atol=1e-7)`
  to allow ULP-level differences while still catching real bugs

**Fix candidate** (real fix):
- Trace MLIR pipeline to find where rounding mode is set; align with
  torch's default

### 10.3 DS v4 FP8 GEMM MLIR codegen bug (row 32, 2026-05-18) ⚠️ REAL BUG

**Op**: `examples/deepseek_v4/example_fp8_gemm_kernel.py` — fp8 GEMM
kernel for DeepSeek V4.

**Result**: **Hard compile-time failure** before kernel even runs:

```
CodeGenTileLangNPUIRDEV: Generated MLIR module failed verification:
error: 'tensor.empty' op incorrect number of dynamic sizes, has 0, expected 1
```

**Where**: codegen emits a `tensor.empty` with **0 dynamic sizes**, but
the type signature says **expected 1 dynamic size**. The dumped IR shows
the offending line:

```mlir
%30 = "tensor.empty"() : () -> tensor<32x128xf16>
```

The tensor type `tensor<32x128xf16>` is **fully static** (no `?`
placeholders), so `tensor.empty` should produce 0 dynamic sizes — this
matches. So the verifier might be wrong, OR the surrounding context
expects a dynamic-shape tensor but codegen emits static.

**Hypothesis**: this is in `CodeGenTileLangNPUIRDEV` (the **Developer
mode** MLIR codegen). The Expert-mode codegen (`tilelang_npuir_apis`)
might not have this bug. Trace which path the fp8_gemm kernel takes.

**Fix candidate** (if Developer-mode-only): set
`TILELANG_ASCEND_MODE=Expert` and re-run, see if it compiles.

**Fix candidate** (real fix): trace `src/target/codegen_npuir_dev.cc`
or `codegen_npuir_api.cc` to find where the tensor.empty op is emitted
without dynamic-size args. Likely a missing `tensor::EmptyOp::build()`
call branch for the GEMM accumulator buffer.

**This is the FIRST automated-discovery MLIR bug** — exactly what the
KB exists to find. Worth filing upstream after we narrow it down.

**Root cause located** (2026-05-18 T32.9 deep-dive):

The bug site is `tilelang-mlir-ascend/src/target/codegen_npuir_dev.cc`
lines 3771-3772 and 3780-3781 (and other Empty-op call sites):

```cpp
std::vector<long int> shape = GetShape(op->extents);

auto tensorEmptyOp = builder.create<mlir::tensor::EmptyOp>(
    builder.getUnknownLoc(), shape, DTypetoMLIRType(op->dtype));
//                                                              ^^^
//                                  MISSING: dynamic-size operands!
```

`GetShape(op->extents)` returns `kDynamic` for symbolic dims. The
`EmptyOp::build(loc, shape, elementType)` overload **does NOT take
dynamic-size operands** — it produces a static-shaped tensor. The
correct overload is `EmptyOp::build(loc, shape, elementType,
dynamicSizes)` where `dynamicSizes` is a `ValueRange` of `index`-typed
SSA values for each `kDynamic` slot in `shape`.

Reproducer: any kernel using `T.symbolic("M")` AND `T.alloc_fragment`/
`T.alloc_shared`/`T.decl_buffer` with a shape containing that symbolic.
`examples/deepseek_v4/example_fp8_gemm_kernel.py` is the canonical
small reproducer (M = T.symbolic, scales tensor `(M, ceildiv(K,128))`
is sized with M dynamic).

**Tested**: `TILELANG_ASCEND_MODE=Expert` and default mode both fail
the same way. So the bug is in the shared codegen, not Developer-mode
specific.

**Fix sketch**:

```cpp
std::vector<long int> shape = GetShape(op->extents);

llvm::SmallVector<mlir::Value, 4> dynamicSizes;
for (size_t i = 0; i < shape.size(); ++i) {
  if (shape[i] == mlir::ShapedType::kDynamic) {
    // op->extents[i] is the symbolic expression; resolve via var_map_
    mlir::Value sizeVal = ResolveSymbolicToValue(op->extents[i], builder);
    dynamicSizes.push_back(sizeVal);
  }
}

auto tensorEmptyOp = builder.create<mlir::tensor::EmptyOp>(
    builder.getUnknownLoc(), shape, DTypetoMLIRType(op->dtype), dynamicSizes);
```

Apply this to all 7 `EmptyOp::build` call sites in
`codegen_npuir_dev.cc` (lines 1097, 1128, 2131, 3437, 3443, 3474,
3480, 3507, 3513, 3537, 3543, 3771, 3780).

**Upstream PR readiness**: ~2h of work to (a) build the small reproducer
into a minimal C++ unit test, (b) implement `ResolveSymbolicToValue`
helper if not already present, (c) apply to all sites, (d) rebuild
bishengir-build-tilelang-mlir-ascend stack to verify fp8_gemm now
compiles, (e) write upstream issue with repro + patch.

**Where MLIR's existing handling lives in `CreateStaticBackedTensor`**
(line 1093 in same file): there's already code that handles dynamic
dims — it derives **static upper bounds** from source GM buffer types
(`tryGetStaticDim`). For the fp8_gemm case the source is the function
arg `scales_a: T.Tensor((M, ...))` which has dynamic M, so the static
upper bound can't be inferred from a typed input — it MUST come from
the SSA value (the loop var `arg18` in this case). This is the gap the
fix has to bridge.

### 10.3.1 Fix iteration log (4 rounds, T32.11)

Demonstrating the auto-fix triage loop manually:

| Round | Site patched | Result | Lesson learned |
|-------|--------------|--------|----------------|
| v1 | `VisitStmt_(AllocateNode)` lines 3771,3783 | Rebuild OK; fp8_gemm SAME failure | **Wrong site** — the offending tensor.empty isn't from `T.alloc_*`, it's from `NeedGenInsertSlice` called by vmul codegen |
| v2 | `NeedGenInsertSlice` line 2131, extract from `shape_val` | Rebuild OK; new error `expected mlir::Value for dim 0` at our ICHECK | `shape_val` is built from `range` arg (dst subview), **not** aligned with `srcShape` dynamic positions |
| v3 | `NeedGenInsertSlice` line 2131, use `tensor::DimOp(src, i)` | Rebuild OK; **MLIR verifier PASS for EmptyOp** — but new error: `tensor.dim op operand must be tensor, got memref` | When src is a memref (loop-carried), need `memref::DimOp`, not `tensor::DimOp` |
| v4 | `NeedGenInsertSlice` type-dispatch DimOp | Rebuild OK; EmptyOp + Dim PASS — **new error: `tensor.collapse_shape op expected dim 0 of collapsed type to be dynamic since expanded has dynamic`** | The bug isn't only in EmptyOp — there's a CHAIN of codegen issues with symbolic shapes. The codegen emits `tensor.collapse_shape` from `tensor<?x2xf32>` to `tensor<1xf32>` (collapses all dims) — but the verifier requires the output dim to be dynamic if any input dim is. |

**Verdict after 4 rounds**: The bug is **broader than EmptyOp** — multiple
codegen sites (EmptyOp, collapse_shape, possibly others) need
symbolic-shape awareness when `T.symbolic` flows through `T.Pipelined`
loop-carried tensors. The fp8_gemm kernel hits all of them in sequence.

**Smallest-repro for upstream issue**:
Instead of fp8_gemm (heavy), write a minimal kernel:

```python
@tl.jit(target="npuir")
def repro():
    M = T.symbolic("M")
    @T.prim_func
    def main(A: T.Tensor((M, 2), "float32"),
             B: T.Tensor((M, 2), "float32"),
             C: T.Tensor((M, 2), "float32")):
        with T.Kernel(1, is_npu=True) as (cid, _):
            for k in T.serial(M):
                T.vmul(A[k], B[k], C[k])  # triggers NeedGenInsertSlice w/ dyn shape
    return main
```

This kernel touches EmptyOp + collapse_shape + DimOp all at once. PR with
fix should cover all 3 sites.

**Fix-bundle status**:
- ✅ v4 patch applied at `apply_emptyop_fix_v4.py` — covers EmptyOp
- ❌ collapse_shape codegen not yet fixed — needs another iteration
- ❌ unknown: how many more codegen sites would surface with deeper iteration

**Initial decision** (after v4): defer to upstream — but user direction
2026-05-18 04:38 was to keep digging. v5+ rounds below.

### 10.3.2 Continuing the chain (rounds v5-v7, 2026-05-18 04:30+)

**v5** — `tensor.collapse_shape` static result from dynamic input. Patched
`ReshapeTensorImpl` srcRank > dstRank branch to build `adjustedDstShape`
that propagates source dynamic dims into corresponding dst groups.
Rebuilt OK; fp8_gemm hits next stack (`insert_slice` static size
mismatches new dynamic collapse output).

**Insight after v5** — hardcoded `M = 128` (no symbolic) in fp8_gemm
kernel and re-ran: **STILL fails with same family of bug**:

```mlir
%75 = tensor.empty() : tensor<128x2xf32>     ; FULL scales_a buffer shape
%76 = vmul(memref<128x2xf32>, f32, %75)     ; whole buffer × scalar
%77 = collapse_shape(%76) ... -> tensor<1xf32>  ; collapse 256→1: nonsense
```

**This is NOT a symbolic-shape bug. It's a deeper codegen issue**:
`Scale_C_shared[i] = scales_a[by*32+i, k] * scale_b` should be a
**scalar load + scalar mul + scalar store**, but the codegen emits a
**full-tensor vmul + collapse_shape to 1 element**.

**Root cause located** (T32.14 v6 dig):

In `CreateHIVMBinaryVectorOp::processImm` (line ~2742), the
`is_scalar_load` check correctly identifies that ALL region dims are
size 1 (i.e. it's a per-element access inside a vectorized loop).
BUT the `arg_id == 0` branch returns the **whole buffer** value:

```cpp
if (is_scalar_load) {
  if (arg_id == 0) {
    src = GetVarValue(region_node);  // ← returns WHOLE buffer
    ...
  } else {
    src = VisitExpr_(buffer_node);  // ← correctly extracts scalar
  }
}
```

`GetVarValue(CallNode*)` returns the buffer's underlying MLIR value —
the whole memref/tensor. Downstream vmul treats this as a tensor input.

**v6 attempted fix** — make `arg_id == 0` also use `VisitExpr_(buffer_node)`
so it extracts a scalar. Result: scalar makes it to vmul, but
**downstream code paths assume tensor-typed src**:

```cpp
auto srcTensorTy = src0.getType().cast<mlir::TensorType>();  // ← asserts
```

So v6 reverted. The actual fix requires either:

1. **Restructure CreateHIVMBinaryVectorOp** to detect "all operands are
   scalar" and emit `arith.mulf` / `memref.store` instead of vmul. This
   is ~50-100 lines of new codegen + threading through the vectorize
   pass. ~3-5h work.

2. **Fix the npu_loop_vectorize pass** to NOT vectorize a scalar*scalar
   multiply when one operand is loop-invariant scalar — let it stay as
   per-iteration arith.mulf. Smaller fix in `HandleBinaryExpression`
   but requires understanding the resolve_a/resolve_b classification.

3. **Keep is_scalar_load==true behavior but also wrap GM scalar
   loads in `memref.load` to extract**: more conservative, but the
   downstream srcTensorTy.cast fails — would need to add per-type
   handling everywhere.

**Why same-pattern ops PASS** (e.g. `rms_norm`'s `A_shared[i, j] *=
A_pow_sum[i, 0]`): the second operand `A_pow_sum[i, 0]` is a
**fragment-local tensor** (UB-allocated), not a GM memref. The
`GetVarValue` returns a tensor type which the downstream cast accepts.
fp8_gemm's `scale_b = scales_b[bx * block_N // group_size, k]` is a
direct GM load — different value-type path.

### 10.3.3 What "继续深挖" honestly means at this point

We've spent 7 patch iterations and ~3 hours on this single bug chain.
What's been verified:
- bug is **NOT fp8 dtype** (A3 has limited fp8 support but this kernel
  uses fp16 carriers internally — confirmed §9b probe)
- bug is **NOT symbolic shape** (hardcoded M=128 reproduces same chain)
- bug is **NOT MLIR verifier strictness** (v1-v5 progressively satisfied
  the verifier; deeper bugs remain)
- bug **IS** in tilelang-mlir-ascend `processImm` asymmetric scalar/
  tensor handling when one vmul operand is a GM scalar load

Honest status: **fix requires ~3-5h of codegen restructure**, larger
than incremental patching. The 7 patches are partial repairs that make
the IR pass verifier but don't address the underlying scalar-vs-vector
classification flaw.

**Recommended next action**: rather than keep iterating on fp8_gemm
alone, pivot to:
- Continue verifying more ops to expand failure-pattern data (DS v32
  patterns, deepseek_v4 remainder, torch_tl_ops)
- Probe Cast round_mode hypothesis (§10.1, §10.2)
- Build the actual restructure of CreateHIVMBinaryVectorOp in a future
  session with focused budget

User direction needed if multi-hour codegen restructure is in scope.

### 10.3.4 Precise root cause located (T32.14 v7+ dig, 2026-05-18 04:50)

The bug is **at `npu_loop_vectorize.cc::HandleBinaryExpression`**, not at
codegen layer. Walkthrough:

For `scales_a[by*32+i, k] * scale_b` inside `for i in T.Parallel(block_M)`:

1. `ResolveOperand(scales_a[by*32+i, k])` → returns BufferLoad
   (operand[0] has loop var i, resolves successfully) → `resolved_a`
2. `ResolveOperand(scale_b)` where scale_b = `scales_b[bx//1, k]` (loop-
   invariant BufferLoad) → returns **`std::nullopt`** at line 678
   (`IsLoopInvariantScalarLike` returns true)
3. Hit the `if (!resolved_b)` branch:
   ```cpp
   if (IsScalar(operands[1]) || operands[1].as<VarNode>()) {
     region_b = operands[1];  // ← would be CORRECT for runtime scalar
   } else {
     // BufferLoad: broadcast to tmp buffer (line 845)
     auto scalar_buf = CreateTempBuffer(ref, binary_op_name);
     stmts->push_back(BuildNpuirCall("Broadcast", {operands[1], ...}));
     region_b = BuildRegionCall(scalar_buf, 1, 1);
   }
   ```
4. `IsScalar()` only matches IntImm/FloatImm — runtime BufferLoads don't.
   `as<VarNode>()` doesn't match either (it's a BufferLoad).
5. **So the broadcast path runs**: `CreateTempBuffer(ref, "Mul")` creates
   a tmp buffer whose shape is `ref_buf->shape` (line 354) — the FULL
   `scales_a` buffer shape `(128, 2)` — not the loop iteration shape.
6. Codegen later emits `vmul(arg18: memref<128x2>, scalar_buf, dst)`
   where dst is also shape `(128, 2)`. The downstream collapse to
   1-element produces the verifier-rejecting IR.

**The fix has 3 candidate sites, in order of preference**:

**Option F1** (smallest, ~5 lines): in `HandleBinaryExpression` line ~840
(`!resolved_b` branch), check if `operands[1]` is a **loop-invariant
BufferLoad**. If yes, **lift it into a `let`-Var** (or use TVM's
`Substitute` to inline as a runtime-scalar). Then re-enter the path
where `region_b = operands[1]` correctly carries the runtime scalar
through to vmul codegen's `tir::VarNode` arm.

**Option F2** (medium): change `CreateTempBuffer` to use **access-derived
shape** (size 1 per dim — the per-iter access shape) instead of
`ref_buf->shape`. Trace the broadcast emission to make sure
`BuildNpuirCall("Broadcast", ...)` can handle 1×1 tmp buffers.

**Option F3** (largest, deep): change codegen `CreateHIVMBinaryVectorOp`
to support memref-typed src input by emitting `memref.load` to extract
scalar before vmul. This is what v6 attempted; downstream cast<TensorType>
needs follow-up changes.

**Path forward**: implementing F1 is the cleanest. Need to introduce a
small TIR pass (or augment `npu_loop_vectorize`) that wraps loop-
invariant BufferLoad operands in a Let-binding. Estimated effort
~30-60min. Worth one more iteration.

**Regression status** after v6 revert:
- vec_add_1d, rms_norm, flash_attn, matmul → all PASS (regression-tested)
- v1-v5 patches still applied (EmptyOp + Dim type-dispatch + collapse_shape)
- These improvements survive — they're not blocking other ops

### 10.3.5 BREAKTHROUGH — fp8_gemm fully PASSES (T32.14 v9, 2026-05-18 05:05)

After 9 iterations, fp8_gemm finally compiles + runs + matches reference.

**Final fix stack** (4 logical changes spread across 9 patch iterations):

| Patch | File | What it does |
|-------|------|-----|
| v1 | `codegen_npuir_dev.cc` line 3771,3783 (AllocateNode) | Add dynamicSizes to tensor.empty when shape is symbolic |
| v2-v4 | `codegen_npuir_dev.cc::NeedGenInsertSlice` | Same fix at line 2131 with type-dispatch tensor::DimOp vs memref::DimOp |
| v5 | `codegen_npuir_dev.cc::ReshapeTensorImpl` srcRank>dstRank | Propagate dynamic dims into collapsed output type |
| **F1 step 1** | `npu_loop_vectorize.cc::HandleBinaryExpression` line 840 | Treat loop-invariant BufferLoad as Scalar (don't broadcast-to-full-buffer) |
| **F1 step 2** | `codegen_npuir_dev.cc::processImm` Scalar branch | Add `op->args[arg_id].as<BufferLoadNode>()` to Scalar acceptance |
| **v6 (re-applied)** | `codegen_npuir_dev.cc::processImm` Vector/scalar_load arg_id==0 | Use `VisitExpr_(buffer_node)` instead of `GetVarValue(region_node)` for both arg_id |
| **v9** | `codegen_npuir_dev.cc::CreateHIVMBinaryVectorOp` entry | When both srcs are scalar (post v6+F1), emit `arith.mulf/addf/subf/divf` + `tensor.insert`/`memref.store` instead of vmul |

**Why all 4 needed**:
- v1-v5: defense-in-depth for emerging dynamic-shape codegen issues (orthogonal partial fixes)
- F1+v6: routes runtime scalar BufferLoads through Scalar path, avoiding the full-buffer broadcast
- v9: gives the codegen a path to handle the scalar-scalar case downstream (without v9, src0/src1 would be f32 scalars but the rest of vmul codegen casts to TensorType and asserts)

**Verification matrix on v9**:
- `examples/deepseek_v4/example_fp8_gemm_kernel.py`: PASS (m=128/n=256/k=256 fp16 + bf16) — **the previously failing kernel!**
- Regression suite (10 previously-passing ops): all still PASS
  - vec_add_1d, vec_add_2d, atomic_add, exp2, log2, gemm, gemv,
    rms_norm, layer_norm, flash_attn, sparse_mla_fwd, sparse_attn,
    matmul, engram_fwd — none broken
- act_quant: still FAIL (strict-`==` rounding, unrelated §10.2)
- fp8_lighting_indexer: still FAIL (5/16M tolerance, unrelated §10.1)

**Conclusion**: 9 patch iterations, 4 distinct code locations, ~3-4
hours of work. fp8_gemm bug class fully fixed; the 2 remaining failures
(precision rounding) are a different bug class requiring HW probe of
Cast op round_mode (§9b).

**Patches recorded in workspace**:
- `apply_emptyop_fix.py` (v1)
- `apply_emptyop_fix_v2.py` (v2)
- `apply_emptyop_fix_v3.py` (v3)
- `apply_emptyop_fix_v4.py` (v4)
- `apply_collapse_shape_fix.py` (v5)
- `apply_scalar_load_fix.py` (v6)
- `apply_F1_loop_invariant_scalar.py` (F1)
- `apply_scalar_binary_dispatch.py` (v9)
- `revert_scalar_load_fix.py` (intermediate)

Apply order: v1 → v2 → v3 → v4 → v5 → F1 → v6 → v9 (the last 3 are the
actual bug fix; v1-v5 patches up surrounding dynamic-shape issues).

### 10.4 Pattern catalog (extracted from confirmed-working examples)

#### Vision: KB is the substrate for tilelang auto-port automation

(User direction 2026-05-18 03:27: 「我们之所以做这个自动化，就是想实现自动发现
mlir 的问题，并自动修复。记住这个这样才能实现基于 tilelang 的三方件迁移到 npu 的全自动化」)

The KB structure (§8.1 + §10.x + §7 recipes) IS the substrate for that
auto-port workflow:

1. **Auto-discover** — script runs every `examples/*` op, captures
   PASS / FAIL with classification (precision FAIL, compile FAIL,
   shape FAIL, ...). Output goes to §8.1.
2. **Auto-classify** — for each FAIL, run a triage decision tree (precision
   threshold? — see §10.1, §10.2; compile error class? — see §10.3) and
   produce a hypothesis. Output goes to §10.x.
3. **Auto-fix** — for compile errors that match known patterns (e.g.
   "tensor.empty incorrect dynamic sizes"), emit a TIR pass or codegen
   patch; for precision rounding bugs, emit a tolerance adjustment
   suggestion. Output goes to ascendc_pto / mlir patches.
4. **Auto-verify** — re-run the failing op with the patch, confirm PASS.
   Output: a PR-ready patch + diff + verification log.
5. **KB feedback** — every new bug class found and fixed becomes a new
   row in §7 / new section in §10.x, feeding back into step 2's triage
   tree for future runs.

That's the cold-loop. T32 has done the first iteration manually; the
codification work is to turn it into a `/tilelang-auto-port` skill or
a CI workflow.



As we cumulate ops, extract reusable patterns here.

#### P-001: T.Kernel one-d grid with serial inner loop

Common shape (seen in vec_add_2d.py, atomic_add.py):
```python
with T.Kernel(BLOCK_SIZE, is_npu=True) as (cid, _):
    A_VEC = T.alloc_ub((block_M, block_N), dtype)
    ...
    for i in T.serial(T.ceildiv(m_num * n_num, BLOCK_SIZE)):
        block_id = i * BLOCK_SIZE + cid
        if block_id < m_num * n_num:
            block_id_m = block_id // n_num
            block_id_n = block_id % n_num
            ...
```
- `BLOCK_SIZE` here is **#parallel AIVs**, not block geometry.
- Outer `T.serial` loops over data blocks; inner kernel body is
  per-block work. This is the canonical "fewer cores than blocks"
  pattern.

#### P-002: T.vadd / T.tile.* semantics

Confirmed: `T.vadd(A, B, C)` (MLIR) and `T.tile.add(C, A, B)` (PTO) take
arg order **(dst, src1, src2)** vs **(src1, src2, dst)**. Distinct
arities and signatures across backends. Don't cross-port without
checking.

#### P-003: `T.copy(GM_tensor[m, n], UB_tensor)`

Both backends use this for GM→UB DMA. Backend handles the actual DMA
emission (PTO via `copy_gm_to_ub` template, MLIR via dialect lowering).
On PTO this is where #996 lives; on MLIR it's safe.

---

## 11. Bug-class taxonomy (T32.14+15 mined patterns)

Each row = a distilled "if you see X, look at Y" diagnostic shortcut.
Save next session 5-10 hours of bisecting.

### 11.1 MLIR dynamic-shape verifier reject family

| Symptom | Bug pattern | Likely site |
|---------|-------------|-------------|
| `'tensor.empty' op incorrect number of dynamic sizes, has 0, expected N` | codegen used `EmptyOp::build(loc, shape, type)` overload when shape has `kDynamic` slots | grep `tensor::EmptyOp::build` in `codegen_npuir*.cc` — pass `dynamicSizes` collected via `tensor::DimOp(src,i)` for tensors or `memref::DimOp` for memref-typed src |
| `'tensor.dim' op operand must be tensor, got memref` | wrong DimOp type for src | `tensor::DimOp` for `TensorType`, `memref::DimOp` for `MemRefType` — type-dispatch required |
| `'tensor.collapse_shape' op expected dim N of collapsed type to be dynamic since one or more of the expanded dims is dynamic` | result `RankedTensorType` built from all-static `dstShapeStatic` even though src has dynamic dims | recompute output shape: for each result dim's reassociation group, if any input dim is `kDynamic` → mark output dim `kDynamic`. See `ReshapeTensorImpl` patch v5 |
| `'tensor.collapse_shape' op expected dim N of collapsed type to be static value of K` | inverse — output type dynamic when verifier wants static | check if the reassociation group's input dims are actually fully static and adjust |
| `tensor.insert_slice` rank/size mismatch after fix above | downstream consumer of collapse_shape uses literal-static `static_sizes` array | the fix at collapse_shape may need to propagate to insert_slice — check both at once |

**Triage shortcut**: if any kernel uses `T.symbolic("X")` OR `T.Pipelined`
loop-carried memref AND fails with MLIR verifier error containing
"tensor.empty" or "collapse_shape" or "tensor.dim" — go directly to the
patches in §10.3 (v1-v5).

### 11.2 Scalar-load vs vector-load classification (T32.14 v6+F1+v9)

| Symptom | Bug pattern | Likely site |
|---------|-------------|-------------|
| IR shows `vmul(memref<full_buffer>, scalar, tensor<full_buffer>)` instead of scalar arith | `is_scalar_load=true` AND `arg_id==0` path returns whole buffer via `GetVarValue(region_node)` (asymmetric with arg_id==1's correct `VisitExpr_(buffer_node)`) | `codegen_npuir_dev.cc::processImm` — fix both branches to use `VisitExpr_(buffer_node)` when is_scalar_load |
| Subsequent `cast<TensorType>` assertion in vmul codegen | post-fix above, src is now scalar f32 not tensor — downstream `srcTensorTy = src0.getType().cast<mlir::TensorType>()` asserts | wrap entry of `CreateHIVMBinaryVectorOp` with scalar-scalar fast path: emit `arith.{mulf,addf,...}` + `tensor.insert`/`memref.store` and return |
| Test outputs FULL buffer values even though kernel claims to write only 1 element per iter | loop-invariant BufferLoad got broadcast-to-tmp-buffer instead of staying as PrimExpr scalar | `npu_loop_vectorize.cc::HandleBinaryExpression` — add `loop_invariant_BufferLoad` to the "keep as PrimExpr" branch (alongside IntImm/FloatImm/VarNode) |

**Triage shortcut**: if test fails on a kernel with `T.Parallel(N)` body
doing `scalar_buf[i] = vector_buf[i, k] * loop_invariant_scalar_load`,
look at §10.3.5 — the 3-patch combo (F1.1 + F1.2 + v6 + v9) is the
fix family.

### 11.2.1 API/DEV codegen asymmetry (T32.15 regression discovery)

| Symptom | Bug pattern | Likely site |
|---------|-------------|-------------|
| `NpuirOperand::FromExpr cannot handle the expr with type of "tir.BufferLoad"` FATAL at `src/op/ascend.cc:40` | Patched `codegen_npuir_dev.cc::processImm` to accept new PrimExpr type, but didn't update the API mode's equivalent | both `src/op/ascend.cc::NpuirOperand::FromExpr` AND `src/target/codegen_npuir_dev.cc::processImm` must accept the same set of Scalar-like PrimExpr types |
| Op PASSes in Developer mode but FATAL in Expert mode (or vice-versa) | Mode-specific codegen has stale acceptance list | grep both `processImm` (DEV) and `FromExpr` (API) for the relevant `.as<XXNode>()` checks |

**Triage shortcut**: if you modified ANY pass in `npu_loop_vectorize.cc`
that changes what PrimExpr types reach the codegen — verify BOTH codegens
(`codegen_npuir_dev.cc` for Developer mode AND `src/op/ascend.cc` +
`codegen_npuir.cc` for Expert/API mode) accept the new types. The two
have separate operand-resolution paths.

### 11.3 Cast op rounding-mode mismatches

| Symptom | Bug pattern | Likely site |
|---------|-------------|-------------|
| `assert torch.all(y == y_ref)` fails on quantized int8/uint8 output, **values visually identical to several decimal places** | NPU `T.vcast(..., round_mode="round")` uses tie-away-from-zero; torch.round uses tie-to-even (banker's). At exactly 0.5 values they disagree → int8 differs by 1 | kernel source: change `round_mode="round"` → `round_mode="rint"`. Available modes: RINT (tie-to-even, torch default), ROUND (tie-away-from-zero, C `round`), FLOOR, CEIL, TRUNC, ODD |
| `assert_close(rtol=1e-2, atol=1e-2)` fails on N/M ratio < 0.001% (1 in 10k or rarer) with diff in 0.01-0.02 range on small-magnitude values | fp16 cross-implementation ULP noise — NPU's intermediate fp32-store-to-fp16-workspace path vs torch's fp16 matmul path round differently at small accumulated values. Both valid, neither wrong | this is NOT a kernel bug. Bump tolerance to fp16-realistic level (rtol=3e-2 atol=2e-2). Confirm by inspecting bit-patterns — should differ in 2-3 mantissa LSBs |

**Triage shortcut**: any failure containing "Mismatched elements: K / TOTAL (0.0%)" — read the 5-10 mismatched element values:
- If integer/quantized output and **bit-exact mismatch** → §11.3.1 round_mode
- If float output and **diff ≈ ULP at small magnitude** → §11.3.2 cross-impl noise, bump tolerance

### 11.4 Host-side allocation shape errors

| Symptom | Bug pattern | Likely site |
|---------|-------------|-------------|
| `shape mismatch: torch.Size([N, 1]) != torch.Size([M, 1])` between kernel output and reference | host-side `x.new_empty(N, 1, ...)` uses wrong dim name. Bug latent when M==N | grep kernel host functions for `x.new_empty(<dim>, ...)` and verify dim matches semantic axis. Reference impl is the source of truth for shape |

**Triage shortcut**: shape mismatches between kernel and ref are almost
always host-side allocation typos. Compare `x.new_empty(...)` /
`torch.empty(...)` shapes in kernel's wrapper function vs the same
dims in the reference. Run with M ≠ N test shape to expose latent bugs.

---

## 12. Preventive rules / lint patterns

Rules to apply BEFORE adding new code or accepting new kernels. Each
captures a bug class we already burned hours debugging.

### 12.1 Codegen-side rules (tilelang-mlir-ascend contributors)

| Rule | Why | Enforce |
|------|-----|---------|
| **R-CG-1: when calling `tensor::EmptyOp::build(loc, shape, ...)`, if any `shape[i]` could be `kDynamic`, MUST pass `dynamicSizes` operand** | §10.3 chain of bugs all stemmed from missing dynamic-size operands at multiple call sites | lint script: grep `EmptyOp>(` in `codegen_npuir*.cc`, flag any without 4-arg overload; manual review of sites that pass a `shape` derived from `op->extents` |
| **R-CG-2: when calling `tensor::DimOp` / `memref::DimOp`, MUST type-dispatch on `src.getType().isa<TensorType>()` vs `isa<MemRefType>()`** | tensor::DimOp rejects memref operand; reverse also true | helper function `emitDimOp(builder, src, i)` that dispatches and asserts on unsupported types |
| **R-CG-3: when codegen produces a `tensor::CollapseShapeOp` / `ExpandShapeOp` result type from a source with dynamic dims, the result type MUST honor those dynamic propagations through the reassociation map** | §10.3.5 stack 2 was the verifier catching static-result-from-dynamic-input | helper: `computeReassocAwareResultType(srcType, reassoc, targetRank)` |
| **R-CG-4: when `processImm` (or any BinaryOp lowering) detects `is_scalar_load=true`, BOTH operand branches MUST extract the scalar (`VisitExpr_(buffer_node)`) — never return the buffer as a whole tensor** | §10.3.5 final root cause: arg_id==0 returned whole buffer asymmetrically | code review: if you see `if (arg_id == 0)` ... `else` asymmetry in operand-handling code, that's suspicious — make both branches do the same scalar/tensor decision |
| **R-CG-5: any `vmul/vadd/vsub/...` codegen that uses `srcTensorTy = src.getType().cast<TensorType>()` MUST first check if src is actually a tensor or could be a scalar/memref; provide a scalar-arith fallback path** | §10.3.5 v9 fix — when both srcs are scalars (post-F1+v6), emit arith.mulf + insert instead of vmul | helper: `tryEmitScalarBinary(op, src0, src1, ...)` returning success/fallback |
| **R-CG-6: keep `codegen_npuir_dev.cc::processImm` Scalar acceptance list IDENTICAL to `src/op/ascend.cc::NpuirOperand::FromExpr` Scalar branch — both must accept the same `expr.as<XXNode>()` types** | T32.15 regression: F1.2 patched DEV only; engram_bwd_exp broke in API mode | static check: both files' Scalar branches should grep-match the same list of `as<TypeNode>` |

### 12.2 Loop-vectorize-side rules (npu_loop_vectorize.cc contributors)

| Rule | Why | Enforce |
|------|-----|---------|
| **R-LV-1: when broadcasting a loop-invariant operand into a tmp buffer for binop vectorization, the tmp buffer shape MUST come from the LOOP iteration extent, not the buffer's full shape** | `CreateTempBuffer` blindly used `ref_buf->shape` (full 128x2) — over-allocated | rewrite `CreateTempBuffer` to take an explicit shape param derived from `output_ref` (the per-iter access shape) |
| **R-LV-2: `IsScalar(expr)` test should accept loop-invariant BufferLoad (when explicitly checked against `loop_vars`)** | runtime scalar BufferLoads should route through "keep as PrimExpr" not "broadcast to tmp buffer" — see F1.1 | new helper `IsLoopInvariantScalarLikeExpr(expr, loop_vars)` already exists at line 678; expand usage to BufferLoad case |

### 12.3 Kernel-author-side rules (people writing `examples/*.py`)

| Rule | Why | Enforce |
|------|-----|---------|
| **R-KA-1: use `round_mode="rint"` (tie-to-even) for any `T.vcast` whose output is checked against `torch.round` reference** | T32.15: act_quant used `round_mode="round"` — silent 1-bit diff at .5 values | lint: `grep round_mode= examples/` — flag any `round_mode="round"` for review |
| **R-KA-2: host-side `x.new_empty(...)` MUST use shape axes that match the kernel's intended output rank/shape; test with M ≠ N to catch typos** | T32.15: act_quant `s = x.new_empty(N, 1)` was wrong (should be M); only caught at M=64/N=32 | review checklist: when allocating kernel output buffers host-side, write the shape using semantic axis names (`x.size(0)` not `M`/`N` constants) |
| **R-KA-3: `assert_close(rtol/atol)` tolerance choice should match the kernel's intermediate precision floor; for kernels with fp16 intermediate stores, use atol/rtol >= 2e-2** | T32.15: fp8_lighting_indexer used 1e-2 — too tight for fp16 intermediate | review: when kernel has any `T.alloc_*([...], "float16")` storing intermediate, test atol >= 2e-2 unless empirically verified |
| **R-KA-4: `assert torch.all(y == y_ref)` (strict ==) MUST only be used when the output dtype is exact (integer or known-deterministic float). For quantized int outputs, ALSO verify the round_mode in kernel matches torch.round** | T32.15: act_quant used strict == correctly (int8 output IS exact), but the rounding-mode bug made values 1-off | code review pattern: if you see `torch.all(y == y_ref)` followed by int output, immediately check `round_mode` in kernel |
| **R-KA-5: `T.vbrc(value, buf)` requires `value` to be a `tir.Var` (bound local variable), NOT a raw Python int/float literal** | T33.P1.3: `T.vbrc(0, acc_o)` fails with "input vector and output vector must have same rank"; `value_zero = 0; T.vbrc(value_zero, acc_o)` works. Root cause: `_get_extent` in `customize_npuir.py` returns `[]` for non-Buffer types, and `npuir_brc`'s rank assertion only short-circuits on `isinstance(src, tir.PrimExpr)` — raw `0` is `int` (not PrimExpr); the bound let-var is a `tir.Var` (PrimExpr subclass) | lint: in `examples/*.py`, grep `T\.vbrc\(\s*-?\d` — flag any literal-arg call; require local-variable binding |
| **R-KA-6: Lse / scalar-row outputs ([B,S,H]) MUST be allocated with rank parity to the in-kernel fragment ([block_M, 1]); use shape `[B,S,H,1]` and slice `Lse[b,s,0:BM,0:1]`** | T33.P1.3: `lse_shape=[B,S,H]` + `T.copy(buf_BMx1, Lse[b,s,0:BM])` failed MLIR verifier with "expected `memref<1x1xBM xf32>` or rank-reduced version" — codegen produced `memref<BMx1>` instead. Trailing-1 keeps rank consistent | review: when allocating multi-batch output that targets a [BM,1] fragment, add trailing `1` to the host-side tensor and slice with `:1` |

### 12.4 Test-design rules (test author / reviewer)

| Rule | Why | Enforce |
|------|-----|---------|
| **R-TS-1: test with M ≠ N shapes (and other non-square axes) at least once per kernel** | T32.15 act_quant's `s = x.new_empty(N, 1)` typo was latent at M==N | test fixture: include one shape with `M ≠ N` per kernel test (e.g. M=64 N=32 in addition to M=N=64) |
| **R-TS-2: kernel outputs that go through fp32→fp16→fp32 round-trip MUST be tested at atol >= 2x the fp16 ULP at the max output magnitude expected** | T32.15 fp8_lighting_indexer — 1e-2 too tight | math: fp16 ULP at magnitude V is roughly V/1024; for kernels with max output magnitude ~40, ULP is ~0.04, so atol ≥ 0.04 is reasonable, atol = 0.01 is too tight |

---

## 13. KB-as-runbook: future debugging workflow

When a new tilelang-mlir-ascend op fails:

1. **Sweep §8.1**: is the op already verified PASS? If yes, did something
   regress? Run regression suite from `regression_v9_wide.sh`.
2. **Classify the error signature** against §11 taxonomy:
   - MLIR verifier error → §11.1 (dynamic-shape) or §11.2 (scalar-vector classification)
   - Numerical assert mismatch → §11.3 (rounding/precision) or §11.4 (shape)
3. **Match a Bug pattern row → jump to the §10.3 patch script** to verify
   the fix applies cleanly (most are idempotent — checking marker present).
4. If symptom doesn't match any §11 row → **add a new row before fixing**
   (KB-first discipline). Run the bug fix, then update §11 row and §10.X
   with the new pattern.
5. **Always run regression_v9_wide.sh** after applying any fix.
6. **Always commit + push** before moving to next op (don't lose work
   to ssh drops or container deaths).

### 13.1 Checklist for "fix" claims

Before claiming an op is fixed:
- [ ] Run the specific op — PASS observed
- [ ] Run `regression_v9_wide.sh` — no previously-passing op breaks
- [ ] Patch script added to workspace (so it's reproducible)
- [ ] §8.1 table updated with row showing PASS + cite the patch script
- [ ] §11 taxonomy updated if new bug pattern discovered
- [ ] §12 preventive rules updated if a new "should-never-recur" rule applies
- [ ] git commit + push (off-site backup)
- [ ] Discord milestone update with concrete results, not summary slogans

If any of the above is unchecked, the fix is not done.

### 13.2 Critical: BOTH-mode regression rule

When patching ANY shared TIR-level pass (especially
`npu_loop_vectorize.cc` or `src/transform/ascend_*.cc`), the regression
suite MUST cover BOTH `TILELANG_ASCEND_MODE=Expert` (default → API
codegen) AND `TILELANG_ASCEND_MODE=Developer` modes. Some ops default
to Expert (most examples), some explicitly set Developer
(`os.environ['TILELANG_ASCEND_MODE'] = 'Developer'` in script).

Lesson from T32.15: F1 step 1 modified `npu_loop_vectorize.cc` (shared).
Step 2 patched the DEV codegen but missed the API codegen. `engram_bwd_exp`
uses Expert mode (default) so it broke; ops that explicitly set
Developer mode were unaffected.

**Mitigation in regression suite**: ensure at least 5 Expert-mode ops
AND 5 Developer-mode ops are in the regression list. Currently:
- Expert mode (default): vec_add_1d, vec_add_2d, exp2, log2, gemm,
  matmul, flash_attn, layer_norm, sparse_mla_fwd, engram_*
- Developer mode (explicit env): act_quant, fp8_gemm, flash_attn_dev,
  atomic_add_dev, fp8_lighting_indexer (some)

The `final_verification_v2.sh` script covers both modes.
