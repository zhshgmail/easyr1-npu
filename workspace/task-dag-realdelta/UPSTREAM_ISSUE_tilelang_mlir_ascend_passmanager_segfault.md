# issue — tile-ai/tilelang-mlir-ascend: MLIR PassManager segfault on a valid NPU vector kernel (softmax+sinkhorn shape)

> STATUS: **POSTED 2026-06-03 → https://github.com/tile-ai/tilelang-mlir-ascend/issues/100** (owner confirmed). No agent trailer (per issue_pr_no_agent_signature). Labels bug/A5 attempted but my account lacks label permission on this repo → maintainer to label. Title + body verified intact post-publish.

## Title
`[NPU] MLIR pass pipeline (Pipeline.run) segfaults on a valid NPU vector kernel — near-identical kernel compiles fine`

## Environment
- tilelang-mlir-ascend fork v0.1.1.030 (dev build from source, `build/`)
- CANN 8.5.0, bishengir-compile 0.1.0 (e4e2ba9841d1, 2026-01-16)
- torch 2.8, torch_npu, Python 3.11, Ascend NPU (single davinci device)
- `target="npuir"`, `TILELANG_ASCEND_MODE=Developer`

## Summary
A valid NPU vector kernel (softmax + sinkhorn-style row/col normalization, `alloc_shared` + `is_npu=True` + block intrinsics `T.vsub/vexp/vdiv/reduce_max/reduce_sum`) **segfaults (SIGSEGV, exit 139) during the in-process MLIR pass pipeline**, before any `.npuir` is emitted. A structurally near-identical kernel (same op set, same buffers, only the unrolled-loop boundary differs) compiles fine. This is deterministic per-kernel-structure.

## Crash location (faulthandler, main thread)
```
File ".../tilelang/tladapter/utils.py", line 103 in run        # self._pp.run(mlir_str) — MLIR PassManager C++ FFI
File ".../tilelang/engine/lower.py", line 286 in lower
File ".../tilelang/jit/jit_npu.py", line 1178 in compile
File ".../tilelang/cache/kernel_cache.py", line 181 in cached_npu
```
SIGSEGV is inside the C++ MLIR pass pipeline (`Pipeline.run`), i.e. the linalg→npuir lowering passes in the bundled compiler — not in subprocess bishengir-compile (the crash precedes npuir emission).

## Minimal repro — CRASHES (deterministic, 2/2)
```python
import tilelang as tl, tilelang.language as T
N, HC, EPS = 4, 8, 1e-6
@tl.jit(target="npuir")
def s():
    @T.prim_func
    def main(ci: T.Tensor((N,HC,HC),"float32"), co: T.Tensor((N,HC,HC),"float32")):
        with T.Kernel(N, is_npu=True) as (i,_):
            c=T.alloc_shared((HC,HC),"float32"); rs=T.alloc_shared((HC,1),"float32")
            cs=T.alloc_shared((1,HC),"float32"); rm=T.alloc_shared((HC,1),"float32")
            T.copy(ci[i,:,:],c)
            T.reduce_max(c,rm,dim=1); T.vsub(c,rm,c); T.vexp(c,c)
            T.reduce_sum(c,rs,dim=1); T.vdiv(c,rs,c)
            T.reduce_sum(c,cs,dim=0); T.vdiv(c,cs,c)
            for _ in range(3):
                T.reduce_sum(c,rs,dim=1); T.vdiv(c,rs,c)
                T.reduce_sum(c,cs,dim=0); T.vdiv(c,cs,c)
            T.copy(c,co[i,:,:])
    return main
s()   # SIGSEGV here
```

## Near-identical control — COMPILES FINE (deterministic, 3/3)
Same op set, same 4 buffers, 9 reductions — only the unrolled-loop boundary moves the first row/col-norm into the loop:
```python
@tl.jit(target="npuir")
def s_ok():
    @T.prim_func
    def main(ci: T.Tensor((N,HC,HC),"float32"), co: T.Tensor((N,HC,HC),"float32")):
        with T.Kernel(N, is_npu=True) as (i,_):
            c=T.alloc_shared((HC,HC),"float32"); rs=T.alloc_shared((HC,1),"float32")
            cs=T.alloc_shared((1,HC),"float32"); rm=T.alloc_shared((HC,1),"float32")
            T.copy(ci[i,:,:],c)
            T.reduce_max(c,rm,dim=1); T.vsub(c,rm,c); T.vexp(c,c)
            for _ in range(4):
                T.reduce_sum(c,rs,dim=1); T.vdiv(c,rs,c)
                T.reduce_sum(c,cs,dim=0); T.vdiv(c,cs,c)
            T.copy(c,co[i,:,:])
    return main
s_ok()   # BUILT OK
```

## What we ruled out (so the report is accurate)
- NOT a missing op: `reduce_max/reduce_sum/vsub/vexp/vdiv/vadd/npuir_transpose` all lower fine individually.
- NOT "too many column (dim=0) reductions": pure `reduce_sum dim=0 × 5` in a loop → BUILT OK; mixed row+col × 5 with softmax → BUILT OK.
- NOT op-count: `reduce_sum dim=1 × 20` → BUILT OK.
- The crash is sensitive to the exact unrolled IR structure near a size threshold → looks like a codegen/PassManager memory bug, not a semantic limit.

## Impact / workaround
Blocks compiling DeepSeek-V4 inference `hc_split_sinkhorn` (sinkhorn_iters=20) as a single kernel. Workaround that WORKS (verified BUILT OK + numerically correct on NPU, max_abs_diff 8.9e-08 vs torch ref): decompose into smaller kernels — a `sink_softmax` kernel + a single-iteration `sink_iter` kernel looped on the host. The smaller kernels stay below the structure threshold.

## Ask
- Is `Pipeline.run` / the linalg→npuir pass pipeline expected to handle this kernel? If so, this is a PassManager crash to fix; if there is a known UB/buffer-pressure limit, please document it so we can early-fail with a clear message instead of SIGSEGV.
