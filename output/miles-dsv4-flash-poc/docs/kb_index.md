# miles+DSv4-Flash PoC — KB cookbook 沉淀索引

> 本 PoC 期间发现的所有 NPU 适配 bug + 解决方案,按 a5_ops `porting_lessons` schema 沉淀。每条都可被 `/npu-adapt-assist` 自动检索匹配。
>
> 完整列表 + keyword grep 表见 [`docs/_meta/kb/porting_lessons/index.md`](../../../docs/_meta/kb/porting_lessons/index.md);schema 见 [`_schema.md`](../../../docs/_meta/kb/porting_lessons/_schema.md)。

## 编译器 bug

| Cookbook | 摘要 | 上游 |
|---|---|---|
| [`bishengir-001`](../../../docs/_meta/kb/porting_lessons/bishengir-001-extended-canonicalizer-drops-cross-iter-accum.md) | R-KA-16: `ExtendedCanonicalizer` pass 在 DPS in-place `vmul(acc_l, ..., acc_l)` 上误判 RHS 未用 → 删跨 iter softmax 累加器 → `sparse_mla_fwd` NS≥2 全 NaN | Issue [`Ascend/AscendNPU-IR#251`](https://gitcode.com/Ascend/AscendNPU-IR/issues/251) |
| [`tilelang-001`](../../../docs/_meta/kb/porting_lessons/tilelang-001-check-ub-budget-early-fail.md) | bishengir 30s 后才报 UB overflow + 无 alloc breakdown;Python pass `CheckUBBudget` 在 npuir 阶段 <1s 给 per-alloc breakdown + 建议 block_M | PR [`tile-ai/tilelang-mlir-ascend#80`](https://github.com/tile-ai/tilelang-mlir-ascend/pull/80) |

## 上游 API 缺失

| Cookbook | 摘要 | 上游 |
|---|---|---|
| [`mindspeed-002`](../../../docs/_meta/kb/porting_lessons/mindspeed-002-apex-fused-rope-thd-shim-gap.md) | MindSpeed 缺 `apex.transformer.functional.fused_apply_rotary_pos_emb_thd` shim;38 行 self-contained pure-torch fallback | PR [`Ascend/MindSpeed#3509`](https://gitcode.com/Ascend/MindSpeed/merge_requests/3509) |
| [`sglang-002`](../../../docs/_meta/kb/porting_lessons/sglang-002-rmsnorm-bias-attribute-getattr.md) | `sgl_kernel_npu fused_split_qk_norm.py` 直读 `RMSNorm.bias` → `AttributeError`;改 `getattr(layernorm, "bias", None)` | PR [`sgl-project/sgl-kernel-npu#531`](https://github.com/sgl-project/sgl-kernel-npu/pull/531) |
| [`miles-001`](../../../docs/_meta/kb/porting_lessons/miles-001-dsamla-tilelang-npu-port-pattern.md) | miles 4 个 DSAMLA tilelang 算子(`lighting_indexer_fwd/bwd`、`sparse_mla_fwd/bwd`)在 NPU 上的 port 模式 | PR [`radixark/miles#1246`](https://github.com/radixark/miles/pull/1246) |

## 上游回归

| Cookbook | 摘要 | 上游 |
|---|---|---|
| [`sglang-003`](../../../docs/_meta/kb/porting_lessons/sglang-003-fusedmoe-reload-narrow-stacked-mapping.md) | sglang `/update_weights_from_disk` reload path 不 honor `stacked_params_mapping`,在 FusedMoE `_load_w13` `narrow(0, 4096) > dim 1408` crash | Issue [`sgl-project/sglang#26794`](https://github.com/sgl-project/sglang/issues/26794) |

## 环境 / 容器 / 打包(KB-only)

| Cookbook | 摘要 | 上游 |
|---|---|---|
| [`cross-layer-008`](../../../docs/_meta/kb/porting_lessons/cross-layer-008-sys-path-root-namespace-shadow.md) | `sys.path` 含 `''` 或 `'/'` 让 Python `PathFinder` 优先解析为隐式 namespace package,屏蔽 `vllm +empty` / `sglang` 编辑式 install | KB-only |
| [`cross-layer-009`](../../../docs/_meta/kb/porting_lessons/cross-layer-009-ascend-rt-visible-single-chip-trap.md) | `ASCEND_RT_VISIBLE_DEVICES` 是 device id 白名单不是 count;单 chip 容器里 set=`1` 会过滤掉唯一一颗(id=0) | KB-only |
| [`cross-layer-010`](../../../docs/_meta/kb/porting_lessons/cross-layer-010-vllm-mindspeed-flash-attn-stub-collision.md) | Single-process RL on NPU 需要 import vllm BEFORE mindspeed;MindSpeed `create_dummy=True` stub flash_attn 让 vllm `find_spec` 误判 | KB-only |
| [`cross-layer-011`](../../../docs/_meta/kb/porting_lessons/cross-layer-011-uda-namespace-lock-diagnosis.md) | 容器看不到 NPU 而 host 看得到 → 先 `dmesg | grep uda_occupy_dev_by_ns`(常常是 zombie Ray raylet 占了 ns 锁) | KB-only |
| [`mindspeed-001`](../../../docs/_meta/kb/porting_lessons/mindspeed-001-create-dummy-flash-attn-stub.md) | MindSpeed `create_dummy=True` stub flash_attn 行为(对偶视角,见 cross-layer-010) | KB-only |
| [`sglang-001`](../../../docs/_meta/kb/porting_lessons/sglang-001-engine-spawn-main-guard.md) | sglang `Engine` 用 multiprocessing spawn;顶层脚本必须 `if __name__ == "__main__"` guard | KB-only |
| [`tilelang-002`](../../../docs/_meta/kb/porting_lessons/tilelang-002-vbrc-needs-bound-local.md) | `T.vbrc(0, buf)` 在 raw int 上 rank check 失败;先 `zero = T.constant(0, dtype=buf.dtype)` 再传 | KB-only |

## 反向 grep 表(快速从 error trace 找 cookbook)

完整 keyword 表在 [`docs/_meta/kb/porting_lessons/index.md` 顶部](../../../docs/_meta/kb/porting_lessons/index.md#keyword--alias-grep-表retrieval-friendly),不在本文件复述。或直接用 skill:

```bash
/npu-adapt-assist "<paste error trace>"
```
