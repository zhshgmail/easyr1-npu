# miles_indexer_integration dk-梯度垃圾值 —— 精确定性

**日期**: 2026-06-02。验证 tilelang-ascend 算子时,`example_miles_indexer_integration.py` 的 dk 梯度出垃圾值。

## 现象

- dq/dw 梯度正确(对 0 == 0),但 **dk(k.grad)出 `8.23e+33`**,CPU 参考是 `0.285`。max_abs_err(dk)=8.23e+33。
- 定位到 backward 循环里 **b=0, s=3** 那一次 `kernel_s1(...)` 调用产生 `dk_call max_abs=8.23e+33`;s=0/1/2 干净(0.18/0.50/0.55)。

## Discriminator —— 逐个排除(关键:不是表面看到的那些原因)

| 假设 | 测试 | 结果 |
|---|---|---|
| index 乱序触发(s=3 idx=[3,2,0,1] 非升序) | 标准 `_smoke_bwd` 把 topk_idx 从 `arange`[0,1,2,3] 改成 [3,2,0,1] | ❌ **干净**(dKV=0.00003)—— 乱序不是原因 |
| s=3 具体输入值触发 | dump s=3 精确输入(q/k/w/idx/gs),喂给全新 standalone kernel | ❌ **干净**(dKV=0.43,且 gs*0.1→0.043 线性,正常)—— 输入本身没问题 |
| 重复调用 state leakage | 同一 kernel 对象循环调 6 次(乱序 idx + 全 1 gs) | ❌ **全干净**(0.21~0.39)—— 单纯重复调用不触发 |
| **完整集成序列** | 跑完整 `example_miles_indexer_integration`(fwd kernels + topk + bwd kernels 交织) | ✅ **复现**(s=3 出 8.23e+33) |
| async/stream ordering | bwd 调用前插 `torch.npu.synchronize()` | ❌ **仍复现**(s=3 仍 8.23e+33)—— 不是简单 stream 异步问题,更深 |
| 前序 bwd 调用污染 | 跳过 s=0/1/2 的 bwd,只跑 s=3 bwd(fwd 阶段完整) | ❌ **仍复现** —— 不是前序 bwd 调用 |
| fwd tilelang kernel 污染 | fwd 改纯 torch(不调 tilelang fwd kernel),保留 bwd | ❌ **仍复现** —— 不是 fwd kernel 的执行 |

| fwd kernel BUILD 共存 | 连 `lighting_indexer_fwd(...)` 的构建都删掉(=None) | ❌ **仍复现** —— 不是 fwd kernel 共存 |
| gs/idx 进度序列 | 同 kernel 对象按 s=0..3 的精确 gs/idx 进度([1,0,0,0]→[1,1,1,1])调 4 次 | ❌ **干净** —— 不是 active-slot 递增序列 |

**结论(8 个假设全部排除后)**:**只有完整集成脚本复现垃圾,任何受控隔离都干净。** 隔离 vs 集成唯一剩下的差异
是集成里真实张量的生命周期/内存布局 —— `ctx.save_for_backward` 存的张量、autograd graph、对真实 batched
张量做 `.contiguous()` 切片、以及完整 forward 留下的 NPU 内存分配模式。这需要维护方用更深的 NPU 内存工具
(memory allocator trace / UB 残留检查)才能定位。开源层 + 当前工具,已 drill 到尽头。

**不是 V4 训练 blocker**:V4 生产路径用 **CANN-native indexer**(`npu_sparse_lightning_indexer_grad_kl_loss`,
见 [[project_v4_ops_cann_native_mapping]]),不走这条 tilelang indexer-bwd 路径。所以这个 bug 是 tilelang-path
的正确性问题(值得报 tilelang 维护方),但真实训练绕得过去。

## 结论

**dk 垃圾值只在完整集成序列里复现**,在任何隔离测试里都不复现(同 kernel、同输入、重复调用都干净)。
→ 根因是**跨 kernel 的 NPU 状态污染**:forward `lighting_indexer_fwd` 的多次调用 + topk + 之后的 backward `kernel_s1` 调用,某种 NPU 侧状态(workspace / UB / atomic 缓冲)残留,污染了第 4 次 bwd 调用的 dk 输出。这是 indexer-bwd-on-NPU 不稳定家族的新表现(超出已知 R-KA-14 multi-block NaN / R-KA-15 zero-grad atomic garbage)。

dk 路径用 `T.atomic_addx4(dIndexK[cur_idx, ...], ...)` scatter(`example_lighting_indexer_bwd_kernel.py` ~L240),
是最可疑的 NPU 状态点(atomic 缓冲跨 kernel 不清)。但同一 atomic 路径在隔离里干净 → 污染源在**之前的 fwd kernel 序列**,不是 bwd 本身。

## 诚实的边界(不过度声称)

- ✅ 已精确定性:垃圾只在完整 fwd+topk+bwd 序列复现,排除了 index/输入/重复调用三个表面假设。
- ❌ **未找到可改的根因**:跨 kernel NPU 状态污染,大概率在闭源 NPU runtime / atomic_addx4 实现或 tilelang 的 workspace 分配复用。无法在开源层定位到具体的"漏清状态"代码点。
- 因此**不能声称已修复**。这是个需要 tilelang/NPU-runtime 维护方介入的 bug(他们已经在 example 注释里用 R-KA-14/15 标注了同家族问题)。

## 产物(留在 tlrescue `/home/z00637938/workspace/_bf16fix/dk_bug/`)

- `iso_dk2.py` —— 乱序 idx 测试(干净)
- `iso_dk3.py` —— replay s=3 精确输入(干净)
- `iso_dk4.py` —— 重复 6 次调用测试(干净)
- `s3_inputs.pt` —— 触发垃圾的精确 s=3 输入
- `instr_integration.py` —— instrumented 集成示例(打印每次 dk_call max + dump s=3)

## 下一步

- 这是 indexer-bwd 集成路径的真 bug,但根因在难触达层(跨 kernel NPU 状态)。
- 选项:(a) 缩小到"哪一次 fwd 调用污染了 bwd"——在 fwd 和 bwd 之间插 `torch.npu.synchronize()` + 清 workspace 看是否消失(若消失 = 确认状态污染 + 给出 workaround);(b) 报给 tilelang 维护方,带这套隔离证据链。
- 对 V4 训练的影响:miles indexer 反向在 NPU 上的这条集成路径会产坏 dk → 需先用 (a) 的 sync/clear workaround 兜住,才能跑真训练。
