# V4 NPU-adapter: torch-fallback → native torch_npu op swap (task #310)

实测把 V4 bf16 路径里的 torch 占位 fallback 换成已验证数值等价的 native `torch_npu`
算子。**只换数值经实测等价的**;不等价的（RoPE）保留 fp32 torch 路径（且 fp32 比
native bf16 kernel 精度更高，保留是更优选择）。所有结论都来自在 sgl_probe 容器
(Ascend910 A3, torch_npu 2.8.0.post2, CANN 8.5.0) 上真跑的等价 harness,不是推断。

## 等价性实测结论

| fallback 函数 | native 算子 | 实测 max_abs_err vs torch ref | 判定 |
|---|---|---|---|
| `fused_q_norm_rope` / `fused_k_norm_rope_flashmla` 的 RMSNorm 部分 | `npu_rms_norm(x, ones, eps)` | **0.000e+00** (bit-exact) | ✅ 换 |
| `moe.silu_and_mul_clamp` | `npu_clipped_swiglu(dim=-1, alpha=1.0, limit=lim, bias=0.0, interleaved=False)` | **≤2.5e-1 abs / ≤0.77% rel**(纯 bf16 ulp;clamp 内 bit-exact 0.0) | ✅ 换 |
| RoPE 复数乘部分 | `npu_rotary_mul` half | 4.34 | ❌ 不换(rotate-half ≠ V4 interleaved-complex 约定) |
| RoPE 复数乘部分 | `npu_rotary_mul` interleave | 561002 runtime error | ❌ 该 shape/path 不支持 |
| RoPE 复数乘部分 | `npu_apply_rotary_pos_emb` BSND | 4.22 | ❌ 同 convention 不匹配 |
| RoPE 复数乘部分 | `npu_apply_rotary_pos_emb` BNSD | runtime fail | ❌ |

### 为什么 RoPE 不换
V4 的 RoPE 是 `freqs_cis`(complex64)对 `reshape(..., rope_dim//2, 2)` 的**复数逐对相乘**
(interleaved-pair 约定)。native `npu_rotary_mul`/`npu_apply_rotary_pos_emb` 用的是
rotate-half(前半/后半切分)约定,数值差 ~4.3,**不是同一个变换**。interleave 模式在
本 CANN 上对该 shape 报 561002 不支持。而且 V4 fallback 在 **fp32 复数域**算 RoPE,
比任何 bf16 native rope kernel 精度更高 —— 保留 fp32 torch 复数乘是正确选择,不是妥协。

### 为什么 swiglu 参数是 alpha=1.0 / bias=0.0 / interleaved=False
V4 公式: `out = silu(gate).clamp(-lim,lim) * up.clamp(-lim,lim)`,其中 `silu(x)=x*sigmoid(x)`
(plain SiLU, β=1),gate/up 是 last-dim 前/后切分。参数扫描(itertools 全组合实测):
- `alpha=1.702`(GELU-swish 斜率)→ 错(V4 是 plain SiLU,不是 GELU 近似)
- `bias=1.0` → 错(多了平移)
- `interleaved=True`(奇偶切分)→ 错(差 ~38;V4 是前后切分)
- **`alpha=1.0, bias=0.0, interleaved=False` → 0.77% rel(bf16 噪声),clamp 内 bit-exact** ✅

## e2e 验证(关键 gate)
换完两处 native op 后,**整个 V4 RL loop 重跑通过**(不是只编译/单测,是端到端):
```
[v4-rl2] distinct_vs_step0=5/5  step_to_step_changes=5/5
[v4-rl2] === V4 RL LOOP PASS — attention weight-sync changes inference
         (rollout->weight-update->rollout closed) ===
EXIT=0
```
call-site 已确认 native patch 真在路径上执行(非被 OPT flag 旁路):
- `deepseek_v4.py:423` 无条件调 `fused_q_norm_rope`(→ 走 `_rmsnorm` native)
- `deepseek_v2.py:364` 当 `swiglu_limit is not None`(V4 满足)走 `silu_and_mul_clamp`(→ native)

(收尾的 `kill_process_tree ... not reaped` RuntimeError 是 Engine atexit teardown 的
已知噪声,发生在 PASS 打印 + EXIT=0 之后,不是本次运行失败。)

## patch 形态
两处都是 **native-first + torch-fallback** 守卫(`if x.is_npu: native else torch`),
非 NPU 环境自动回落,不破坏 CUDA/CPU 路径 —— 适合作为上游 PR 的形态。
快照: `_elementwise_NATIVE.py` / `_moe_NATIVE.py`。harness: `_native_op_equiv*.py` /
`_swiglu_*.py`。

## 状态
- ✅ task #310 native op swap 完成并 e2e 验证(RMSNorm + clipped_swiglu)
- 这提升了 held 批量 PR 的质量(production-grade native,而非 torch 占位)
- RoPE 保留 fp32 复数乘(已论证是更优而非妥协),不在本次 swap 范围
- 仍 HELD:等整个 miles+V4 RL loop 真训练 e2e 通了再统一批量提 PR
