# porting_lessons — index

Lessons learned while porting upstream libraries to Ascend NPU. Read before
starting a new port and before claiming a port complete. Append a new lesson
when a new failure mode is discovered.

**Schema**: [`_schema.md`](_schema.md)

**Cross-link**: 与 PoC 报告 §4 的 `P-COMP/P-API/P-REG/P-ENV/P-CONF-*` 一一对应,见
[`MILES_DSV4_NPU_POC_REPORT.md` §4.0 总表](../../MILES_DSV4_NPU_POC_REPORT.md#40-一张总表快速-grep)。

**Tool**: 用 `/npu-adapt-assist <error-trace>` 自动按 trigger / symptom 匹配 — 见
[`src/skills/npu-adapt-assist/`](../../../../src/skills/npu-adapt-assist/README.md)。

---

## Keyword / alias grep 表(retrieval-friendly)

Worker / 用户先 grep 这一节按关键词找到 cookbook ID,然后跳到 [§ By layer](#by-layer) 拿 link。
每个 cookbook 在 frontmatter 里有完整 `trigger[]` / `symptom_in_wild[]`,这里只放一行别名摘要供快速 grep。

| 现象 / 关键词 | Cookbook | P-CLASS |
|---|---|---|
| `AttributeError.*RMSNorm.*bias`, `'RMSNorm' object has no attribute 'bias'`, fused_split_qk_norm, sgl_kernel_npu RMSNorm | `sglang-002` | P-API-2 |
| `ImportError.*vllm`, `ImportError.*sglang`, namespace package shadow, sys.path '/' or '', PathFinder editable install, `+empty` finder | `cross-layer-008` | P-ENV-2 |
| sparse_mla_fwd NaN, NS≥2, NS=2/4/8, online softmax cross-iter accumulator, bishengir ExtendedCanonicalizer, R-KA-16, scf.for iter_args | `bishengir-001` | P-COMP-1 |
| ub overflow, bishengir 30s, UB cap 64KB / 192KB, CheckUBBudget, block_M tuning, per-alloc breakdown | `tilelang-001` | P-COMP-2 |
| `ModuleNotFoundError.*apex.transformer`, fused_apply_rotary_pos_emb_thd, mindspeed apex_adaptation gap, GLM-5 fuse_rope, fuse_rope ImportError | `mindspeed-002` | P-API-1 |
| flash_attn stub, `find_spec.*flash_attn`, mindspeed create_dummy, vllm rotary embedding crash, RL single-process import order | `cross-layer-010` / `mindspeed-001` | P-ENV-5 |
| RuntimeError narrow, FusedMoE `_load_w13`, update_weights_from_disk, MoE weight reload, stacked_params_mapping reload, RL weight sync | `sglang-003` | P-REG-1 |
| triton vs triton-ascend, `triton/backends/compiler.py` conflict, `cannot import name 'Language'`, xgrammar triton, pip install order | `triton-ascend-002` | P-ENV-1 |
| sglang Engine fork bomb, multiprocessing spawn, `__main__` guard missing, Engine init hang, context already set | `sglang-001` | P-ENV-4 |
| `T.vbrc.*rank check`, vbrc raw int, tilelang vbrc literal, vbrc rank-0 mismatch | `tilelang-002` | P-CONF-1 |
| container NPU invisible, host npu-smi OK container empty, `uda_occupy_dev_by_ns`, Ray raylet zombie, NPU ns lock | `cross-layer-011` | P-CONF-2 |
| `ASCEND_RT_VISIBLE_DEVICES=1`, single chip filter, container device_count=0, davinci single mount, RT_VISIBLE id-vs-count confusion | `cross-layer-009` | P-ENV-3 |
| miles DSAMLA, lighting_indexer port, sparse_mla port, NPU tilelang dispatcher, `q.is_npu`, miles 4 算子 NPU port | `miles-001` | P-API-3 |
| LLVM version mismatch, libtriton.so bishengir-compile, MLIR text format diverges, "custom op X unknown" binary boundary | `triton-ascend-001` | (legacy) |
| `_TORCH_VERSION_BUILT_FOR`, vllm-ascend torch ABI guard, C++ side never set, 14 iters silent | `vllm-ascend-001` | (legacy) |
| fixc image tag, image rebuild proof, `.so` torch ABI verification, `ldd` + symbol + native op call | `vllm-ascend-002` | (legacy) |
| vllm shim plugin-init order, `vllm.v1.sample.*`, `vllm.v1.spec_decode.*`, find_spec + lazy `__getattr__` | `vllm-ascend-003` | (legacy) |

`(legacy)` 标的是 PoC §4 之前已存在的 cookbook;无 `P-CLASS-N` 标号但仍可用。


## By layer

### cross-layer

- [`cross-layer-001-pip-install-is-not-port.md`](cross-layer-001-pip-install-is-not-port.md) — `pip install --no-deps` / overlay wheel / consumer-side shim do NOT constitute upstream port
- [`cross-layer-002-v14-as-multilayer-judgement.md`](cross-layer-002-v14-as-multilayer-judgement.md) — Using V1.4 entropy_loss as THE metric confuses which layer owns the bug
- [`cross-layer-003-skip-lower-layer-bench.md`](cross-layer-003-skip-lower-layer-bench.md) — Moving to layer N+1 while layer N has no independent benchmark hides faults
- [`cross-layer-004-prebuilt-wheel-is-not-port-contribution.md`](cross-layer-004-prebuilt-wheel-is-not-port-contribution.md) — Installing Ascend-prebuilt torch_npu / triton_ascend wheel != contributing a port
- [`cross-layer-005-no-conclusion-without-investigation.md`](cross-layer-005-no-conclusion-without-investigation.md) — Don't declare "out of scope / blocked / AIL team territory" before investigating dependency graph. OSS deps often place flags in source submodules, not binaries.
- [`cross-layer-006-compiler-build-time-budgeting.md`](cross-layer-006-compiler-build-time-budgeting.md) — Stop saying "30 minutes" for LLVM/MLIR/bishengir builds. Default unit is hours plural. Don't predict; report ninja [X/Y] milestones.
- [`cross-layer-007-walk-through-is-not-real-run.md`](cross-layer-007-walk-through-is-not-real-run.md) — Agent walk-through reports validate the plan, not the artifact. Always require on-A3 import + operational smoke before claiming end-to-end PASS.
- [`cross-layer-008-sys-path-root-namespace-shadow.md`](cross-layer-008-sys-path-root-namespace-shadow.md) — **P-ENV-2**. `sys.path` 含 `''` 或 `'/'` 让 Python `PathFinder` 优先解析为隐式 namespace package,屏蔽 `vllm +empty` / `sglang` 编辑式 install。脚本头 `sys.path = [p for p in sys.path if p not in ("", "/")]`。
- [`cross-layer-009-ascend-rt-visible-single-chip-trap.md`](cross-layer-009-ascend-rt-visible-single-chip-trap.md) — **P-ENV-3**. `ASCEND_RT_VISIBLE_DEVICES` 是 device id 白名单不是 count;单 chip 容器里 set=`1` 会过滤掉唯一一颗 (id=0)。
- [`cross-layer-010-vllm-mindspeed-flash-attn-stub-collision.md`](cross-layer-010-vllm-mindspeed-flash-attn-stub-collision.md) — **P-ENV-5**. Single-process RL on NPU 需要 import vllm BEFORE mindspeed;MindSpeed `create_dummy=True` stub flash_attn 让 vllm `find_spec` 误判存在 → crash。
- [`cross-layer-011-uda-namespace-lock-diagnosis.md`](cross-layer-011-uda-namespace-lock-diagnosis.md) — **P-CONF-2**. 容器看不到 NPU 而 host 看得到时,先 `dmesg | grep uda_occupy_dev_by_ns`,不是 driver bug — 别的进程(常常是 zombie Ray raylet)占了 ns 锁。

### triton-ascend

- [`triton-ascend-001-llvm-version-must-match-bishengir.md`](triton-ascend-001-llvm-version-must-match-bishengir.md) — triton-ascend libtriton.so and bishengir-compile must be built against the same LLVM source; MLIR text format diverges across major LLVM versions and produces misleading "custom op X unknown" parse errors at the binary boundary.
- [`triton-ascend-002-packaging-conflict-with-mainline-triton.md`](triton-ascend-002-packaging-conflict-with-mainline-triton.md) — **P-ENV-1**. triton-ascend and mainline triton both register-overwrite `triton/backends/compiler.py`; only the later install is functional. Workaround on NPU hosts is `pip uninstall -y triton && pip install --force-reinstall --no-deps triton-ascend`. Don't file at triton-lang/triton-ascend (closed not-planned 2026-05-29) — the conflict belongs upstream of triton-ascend.

### vllm-ascend

- [`vllm-ascend-001-torch-version-built-for-unverified.md`](vllm-ascend-001-torch-version-built-for-unverified.md) — `_TORCH_VERSION_BUILT_FOR` constant only read in Python; C++ side never set it; guard silently returned `False` for 14 iters
- [`vllm-ascend-002-fix-c-image-name-is-not-proof.md`](vllm-ascend-002-fix-c-image-name-is-not-proof.md) — Image tagged `fixc` does not prove the `.so` was rebuilt against the new torch ABI. Only `ldd` + symbol inspection + first native op call do.
- [`vllm-ascend-003-shim-plugin-init-order.md`](vllm-ascend-003-shim-plugin-init-order.md) — Shims targeting `vllm.v1.sample.*` / `vllm.v1.spec_decode.*` must use `find_spec` + lazy `__getattr__` to survive vllm-ascend's plugin-init phase.

### sglang

- [`sglang-001-engine-spawn-main-guard.md`](sglang-001-engine-spawn-main-guard.md) — **P-ENV-4**. sglang `Engine` 用 multiprocessing spawn;顶层脚本必须 `if __name__ == "__main__"` guard,否则 fork bomb / 死锁。
- [`sglang-002-rmsnorm-bias-attribute-getattr.md`](sglang-002-rmsnorm-bias-attribute-getattr.md) — **P-API-2**. `sgl_kernel_npu fused_split_qk_norm.py` 直读 `RMSNorm.bias` → AttributeError;改 `getattr(layernorm, "bias", None)` 即可。PR #531。
- [`sglang-003-fusedmoe-reload-narrow-stacked-mapping.md`](sglang-003-fusedmoe-reload-narrow-stacked-mapping.md) — **P-REG-1**. `/update_weights_from_disk` reload path 不 honor `stacked_params_mapping`,在 FusedMoE `_load_w13` `narrow(0, 4096) > dim 1408` crash;dense 路径 OK。Issue #26794。

### mindspeed

- [`mindspeed-001-create-dummy-flash-attn-stub.md`](mindspeed-001-create-dummy-flash-attn-stub.md) — **P-ENV-5** MindSpeed 侧。`create_dummy=True` 在 sys.modules 装 stub flash_attn,与 vllm `find_spec` 共存时让 vllm 误判 → consumer 必须 vllm 先 import。
- [`mindspeed-002-apex-fused-rope-thd-shim-gap.md`](mindspeed-002-apex-fused-rope-thd-shim-gap.md) — **P-API-1**. MindSpeed 缺 `apex.transformer.functional.fused_apply_rotary_pos_emb_thd` shim;38 行 self-contained pure-torch fallback + 1 行 `pm.register_patch`。PR #3509。

### bishengir

- [`bishengir-001-extended-canonicalizer-drops-cross-iter-accum.md`](bishengir-001-extended-canonicalizer-drops-cross-iter-accum.md) — **P-COMP-1** (R-KA-16). `ExtendedCanonicalizer` pass 在 DPS in-place `vmul(acc_l, correction, acc_l)` 上误判 RHS 未用 → 删跨 iter softmax 累加器 → `sparse_mla_fwd` NS≥2 全 NaN。Issue Ascend/AscendNPU-IR#251。Bisect 配方 `memory/bishengir_iter_args_bisect_recipe.md`。

### tilelang

- [`tilelang-001-check-ub-budget-early-fail.md`](tilelang-001-check-ub-budget-early-fail.md) — **P-COMP-2**. bishengir 30s 后才报 UB overflow + 无 alloc breakdown;Python pass `CheckUBBudget` 在 npuir 阶段 <1s 给 per-alloc breakdown + 建议 `block_M`。PR tile-ai/tilelang-mlir-ascend#80。
- [`tilelang-002-vbrc-needs-bound-local.md`](tilelang-002-vbrc-needs-bound-local.md) — **P-CONF-1**. `T.vbrc(0, buf)` 在 raw int 上 rank check 失败;先 `zero = T.constant(0, dtype=buf.dtype)` 再传。

### miles

- [`miles-001-dsamla-tilelang-npu-port-pattern.md`](miles-001-dsamla-tilelang-npu-port-pattern.md) — **P-API-3**. miles 4 个 DSAMLA tilelang 算子(lighting_indexer_fwd/bwd、sparse_mla_fwd/bwd)在 NPU 上的 port 模式:dispatcher hook + per-op layout(head split, block_size, UB cap, R-KA-16 mitigation)。PR radixark/miles#1246。
