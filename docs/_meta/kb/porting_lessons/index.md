# porting_lessons — index

Lessons learned while porting upstream libraries to Ascend NPU. Read before
starting a new port and before claiming a port complete. Append a new lesson
when a new failure mode is discovered.

**Schema**: [`_schema.md`](_schema.md)

## By layer

### cross-layer

- [`cross-layer-001-pip-install-is-not-port.md`](cross-layer-001-pip-install-is-not-port.md) — `pip install --no-deps` / overlay wheel / consumer-side shim do NOT constitute upstream port
- [`cross-layer-002-v14-as-multilayer-judgement.md`](cross-layer-002-v14-as-multilayer-judgement.md) — Using V1.4 entropy_loss as THE metric confuses which layer owns the bug
- [`cross-layer-003-skip-lower-layer-bench.md`](cross-layer-003-skip-lower-layer-bench.md) — Moving to layer N+1 while layer N has no independent benchmark hides faults
- [`cross-layer-004-prebuilt-wheel-is-not-port-contribution.md`](cross-layer-004-prebuilt-wheel-is-not-port-contribution.md) — Installing Ascend-prebuilt torch_npu / triton_ascend wheel != contributing a port
- [`cross-layer-005-no-conclusion-without-investigation.md`](cross-layer-005-no-conclusion-without-investigation.md) — Don't declare "out of scope / blocked / AIL team territory" before investigating dependency graph. OSS deps often place flags in source submodules, not binaries.

### triton-ascend

- [`triton-ascend-001-llvm-version-must-match-bishengir.md`](triton-ascend-001-llvm-version-must-match-bishengir.md) — triton-ascend libtriton.so and bishengir-compile must be built against the same LLVM source; MLIR text format diverges across major LLVM versions and produces misleading "custom op X unknown" parse errors at the binary boundary.

### vllm-ascend

- [`vllm-ascend-001-torch-version-built-for-unverified.md`](vllm-ascend-001-torch-version-built-for-unverified.md) — `_TORCH_VERSION_BUILT_FOR` constant only read in Python; C++ side never set it; guard silently returned `False` for 14 iters
- [`vllm-ascend-002-fix-c-image-name-is-not-proof.md`](vllm-ascend-002-fix-c-image-name-is-not-proof.md) — Image tagged `fixc` does not prove the `.so` was rebuilt against the new torch ABI. Only `ldd` + symbol inspection + first native op call do.
