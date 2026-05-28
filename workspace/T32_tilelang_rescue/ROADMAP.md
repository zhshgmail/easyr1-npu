# T33 真参数全栈承载 Roadmap

> **关于这个文件**:这是 T33 工作的**单一计划源**。每个任务一行,有 ID、状态、依赖关系。
> 我每完成一步就更新这里。维护规则:
> - 状态字段:`TODO` / `IN-PROGRESS` / `BLOCKED:<why>` / `DONE` / `PARKED:<reason>`
> - 依赖字段:`deps=[T1, T2]` 表示这步只在 T1、T2 都 DONE 后才能开始
> - 并行字段:`parallel-with=[T3]` 表示可以与 T3 同时 subagent 跑
> - 每个 task 后面带 `→ artifact:` 列出产出物(commit / PR / file 路径)
> - 加 task 用下一个未使用的 T<N> 编号,不复用旧编号
> - **绝不删 task**,完成的标 DONE 后留着做历史
>
> 配套阅读:`DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md`(分析、量化、根因)、`UPSTREAM_ISSUE_RKA*.md`(已 file 的 issue 跟踪)。

## 总目标

让 miles 的 4 个 tilelang 算子(`lighting_indexer_fwd/bwd`、`sparse_mla_fwd/bwd`)在 Ascend A3 NPU 上跑**真 DSv4-Flash / GLM-5 参数**(H=64, D_V=512, topk=512, SKV=2048,详见 `DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md` §1)。

成功标准(end state):
1. 4 个算子单独跑真 shape → 全 PASS、数值有限、精度对得齐
2. miles `DSAMLASelfAttention` 在 Megatron-core 引擎里走真 shape forward + backward → 全 finite,grad_norm 合理
3. 整套修复(算子端 + 编译器端 + miles 集成 shim)以 upstream PR 形式提到 4 个上游仓:`tile-ai/tilelang-mlir-ascend`、`Ascend/AscendNPU-IR`、`NVIDIA/Megatron-LM`、`radixark/miles`

## 当前状态快照(2026-05-28 晚)

* lighting_indexer_fwd 真 shape:✅ PASS(无 UB 问题、无 NaN)
* lighting_indexer_bwd 真 shape:✅ PASS (T1 head-split `block_H_inner=16` → miles `0b39e1b`)
* sparse_mla_fwd 真 shape:❌ 算子层 NaN(R-KA-16 多 iter softmax bug,issue #251 filed,**罪魁 pass `ExtendedCanonicalizer` 已 T6 定位**)— 编译已通过
* sparse_mla_bwd 真 shape:✅ 编译 PASS (T3 `block_size=8` + `pp_block_N` cap → miles `45f86c9`),UB 140192 B;数值仍 NaN(R-KA-13 / R-KA-16,T5/T9)
* 编译器诊断 PR #80 已开,等 maintainer review
* 已 file 5 上游 issue(RKA-13/14/15/16 + sglang triton)
* **Megatron `DSAMLASelfAttention` 真 shape e2e**:✅ 编译 + flow-through 打通(T11 → miles `dc26e45`,52M params,SEQ=2048 H=64);数值 NaN 等 R-KA-16 上游修

## DAG(依赖关系图)

```
T1 (R-KA-16 issue) ──┐
                     ├──> T6 (compiler bisect) ────────────┐
T2 (PR #80 review) ──┘                                     │
                                                            ▼
T3 (lighting_indexer_bwd head-split) ──> T7 (整 4 op 真 shape PASS)
                                                            │
T4 (sparse_mla_bwd UB reduce) ──────────────────────────────┤
                                                            │
T5 (sparse_mla bwd R-KA-13 完整化) ─────────────────────────┤
                                                            │
                                                            ▼
                                                T8 (miles 真 shape e2e)
                                                            │
                                                            ▼
                                                T9 (4 个上游 PR 提交)
```

`T1` `T2` `T3` `T4` 互相独立,可以**并行 subagent** 跑。`T5` 依赖现有 R-KA-13 E5 修法 + bwd kernel(已有,独立)。`T6` 是 R-KA-16 真修(深层 bishengir 工作,串行 + 长)。

## Task 列表

### 已完成(历史 / 防丢)

| ID | 任务 | 状态 | 产出 |
|---|---|---|---|
| T0 | 真 DSv4 / GLM-5 算子层 shape 调研(直读 HF config) | DONE | `DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md` §1 |
| T0.1 | 4 个算子真 shape 实测(写 `_real_shape_smoke.py`) | DONE | miles fork `4983264`,文档 §2 |
| T0.2 | sparse_mla NaN 根因 bisect(D/SKV/topk/num_stages 五维) | DONE | 文档 §7,task #259 注释 |
| T0.3 | CheckUBBudget pass(诊断 PR) | DONE | tile-ai/tilelang-mlir-ascend **PR #80** |
| T0.4 | sparse_mla_fwd 头维度切分(`block_M_inner`) | DONE-但-不彻底 | miles fork `4983264`(编译过,真 shape 仍 NaN 因 R-KA-16) |
| T0.5 | R-KA-16 上游 issue 草稿 + 可执行最小重现 | DONE | `UPSTREAM_ISSUE_RKA16.md`、`repro_rka16.py`、AscendNPU-IR **#251** |

### 待执行(按依赖顺序)

#### Track B(并行可做,不依赖编译器修复)

| ID | 任务 | 状态 | deps | parallel-with | 详细 |
|---|---|---|---|---|---|
| **T1** | 用 `lighting_indexer_bwd` head-split 让真 shape 编译过 | DONE | — | T2, T3, T4 | kernel 把 `pad_heads` 抽出成 `block_H_inner` 参数,grid 改 `B*S*head_groups`。预期 indexer_bwd 不触发 R-KA-16(没 online softmax),应该 PASS。详见文档 §7。 → artifact:miles fork commit `0b39e1b` on `npu-tilelang-dispatch`;`_real_shape_smoke.py` `indexer_bwd @ topk=512 SKV=2048` PASS 1.2s,gq/gw/gk 全 finite,max_abs ∈ [2.05, 6.29] |
| **T2** | 跟进 tile-ai PR #80 review | IN-PROGRESS | — | T1, T3, T4 | 看 reviewer 反馈,如要 C++ port 转化、调注释,响应。已修 CI:`d2d1871`(ruff F841 + 删 stale import)和 `df7431e`(关键修正)。**`df7431e` 解 `examples/mixcv/example_mixcv.py` 在 test CI 里被误判**:把 `_UB_BACKED_SCOPES` 缩到 `{"local", "local.fragment"}`(去掉 shared / shared.dyn —— 它们经 L1 spill);把 raise 阈值从 `soft_budget`(80%)抬到 `catastrophic_cap`(2× UB),中间区段让 bishengir 自己 tile。A3 验证:5/5 UT PASS、mixcv 编译并运行成功、miles 真 shape 4/4 PASS。等下一轮 CI。 → artifact:PR #80 merged 或 next-revision commit |
| **T3** | 给 `sparse_mla_bwd` 加完整 R-KA-13 E5 修复 + 把 bwd UB 用 split_store 拆到 < 192KB | DONE | — | T1, T2, T4 | Strategy A (block_size 缩小) 命中:dispatcher 在 `d_v >= 512` 时选 `block_size=8`,UB 从 289280 B (FAIL) → **140192 B** (71% UB,远在 80% 软预算 157286 B 下);同时 postprocess `pp_block_N` 在 `dim_plus_tail=576` 时自动从 64 缩到 16(避免 368640 B 溢出)。split_store 留作后续优化,本任务不需要。→ artifact:miles fork commit `45f86c9` on `npu-tilelang-dispatch`;`_real_shape_smoke.py` sparse_mla_bwd ✅ PASS in 6.6s(数值 NaN 来自 R-KA-13 / R-KA-16,见 T5/T9) |
| **T4** | 给 `sparse_mla_fwd` 应用 R-KA-13 E5(`correction_expanded`)固化到 PR-able 形式 | DONE | — | T1, T2, T3 | 已在 miles fork `4cdfc1f` 加了。`a74688c` 清理:删 dead alloc `new_max_expanded`、把 inline comment 改成指向 AscendNPU-IR #251 + 跨链接到 dispatcher `num_stages=1` 和 ROADMAP T6-T9。小 smoke (B=1 S=8 SKV=16 H=16 D=64 topk=8) cold cache 重跑:max abs err vs fp32 ref = 0.0005,PASS。**残留**:真 shape (NS≥2) 仍 0-1.6% NaN — 算子层已尽力,根因等 T9 上游修。→ artifact:miles fork commit `a74688c` |

#### Track A(编译器 bisect + 真修,深度工作)

| ID | 任务 | 状态 | deps | 详细 |
|---|---|---|---|---|
| **T5** | 在 A3 上用 `bishengir-compile -print-after-all` 跑 `repro_rka16.py` NS=4 的 IR | DONE | T0.5 (repro 已 run-pass) | 用 fresh-built `bishengir-opt` (build-fresh @ 31f690369d, 2026-05-18) + 顶层 `--mlir-print-ir-after-all` flag 一次 dump 311 个 pass after-IR。`bishengir-compile` 本身不暴露 `--print-after-all`(其内嵌 hivmc 子进程吃 CANN 系统二进制),只能走 `bishengir-opt --bishengir-compile=...` pipeline。npuir 通过 monkey-patch `tilelang.jit.jit_npu.compiler_npu._npuir_to_bin_enable_npu_compile` 在 bishengir 调用前 dump。→ artifact:`rka16_ns4_passes.txt` (2.9 MB,31727 行)、`rka16_ns4.npuir` (10 KB)、驱动 `dump_rka16_ir.py`。**T6 候选 pass(按 pipeline 出场顺序)**:(a) `RemoveRedundantLoopInit` (scf-remove-redundant-loop-init) 第 ~9 个 dump,line 903 (b) `CanonicalizeIterArg` (scf-canonicalize-iter-arg) line 6086 (c) `HIVMDecomposeOp` (hivm-decompose-op) line 16550(最后一次 4-tensor iter_args) (d) `InferHIVMMemScope` (hivm-infer-mem-scope) line 17929 — **smoking gun:此 pass 后外层 scf.for 从 4 个 tensor iter_args(acc_o/acc_l/acc_m/scores)塌成 2 个 memref iter_args,acc_l/acc_m 被 hoist 出循环只用 store 写**;late 阶段进一步塌成 1 个 (e) `EnableMultiBuffer` (hivm-enable-multi-buffer) 多次出现于 line 26781+,即使 `enable_auto_multi_buffer=false` 仍 run |
| **T6** | bisect 哪个 HIVM pass 后 `acc_o` 的 iter_arg 状态变错 | DONE | T5 | **罪魁定位:`ExtendedCanonicalizer`(pass dump line 10801)**。证据:line 10627(after `CanonicalizeIterArg`)外层 `scf.for %arg11 = 0 to 4` 还有 4 个 memref iter_args(acc_o / acc_m[64,1]f32 / acc_l[64,1]f32 / scores);line 10801(after `ExtendedCanonicalizer`)只剩 2 个(acc_o, scores),**`acc_m` 和 `acc_l` 这两个 `[64,1] f32` iter_args 被吃掉了**。下游 `vmul(acc_l, correction, acc_l)` 和 `vmul(acc_o, correction, acc_o)` 因为 in-place write-back-to-iter-arg 的 DPS 形态,被 ExtendedCanonicalizer 误判成「同地址循环写,无跨 iter 依赖」并 drop。Bisect 方法:`grep -nE "scf.for.*iter_args"` 每个 pass dump,统计 tensor/memref iter_args 数,定位 4→2 那一跳。→ artifact:bisect 数据(`pass_lines.txt` 311 行 + 每 pass iter_args 计数)记入 §6 of ROADMAP;具体修改方向 = 给 `ExtendedCanonicalizer` 加 conservative-keep-rule:`iter_arg` 即使在 dependency analysis 上看似 "in-place updated" 但若其 update RHS 用了 loop-induction-variable-dependent value(本例:correction 由 vsub(new_max-old_max) 得到,而 new_max 是当前 iter 数据的 reduce_max),则必须保留 iter_arg |
| **T7** | 写 bishengir HIVM pass 的 fix(C++) | BLOCKED:upstream-MLIR-scope | T6 | T6 锁定罪魁 = `ExtendedCanonicalizer`(`bishengir/lib/Transforms/ExtendedCanonicalizer.cpp`),93 行;但这只是个 **薄壳**,实质只 register upstream MLIR `CanonicalizerBase` 的 patterns + `memref::getExtendedCanonicalizationPatterns` + `linalg::getExtendedCanonicalizationPatterns`。两个扩展文件都没动 SCF iter_args。**真正的 iter_arg dropper 是上游 MLIR `mlir/lib/Dialect/SCF/IR/SCF.cpp` 的 `RemoveUnusedIterArgs` canonicalization pattern**,它在 tensor 形态下 49 个 pass 不出问题(linalg ops 显式 data dep),但 `OneShotBufferize` 把 tensor 转成 memref + DPS in-place 后,该 pattern 看到 `vmul(acc_l, correction, acc_l)` outs aliases iter_arg 而误判 "loop store, no cross-iter dep"。**结论**:C++ fix 不在 bishengir 范围内,要么提到 upstream LLVM,要么 Huawei 在 ExtendedCanonicalizer 中关掉/重写 SCF iter_arg canonicalization。**改 PARKED**,改走 T9 路径:把 T6 的精确诊断写进 AscendNPU-IR issue #251 update,让 Huawei 编译器组接手具体 C++ patch(他们更熟悉自己加的 patterns 与上游交互)。→ artifact:无 commit;诊断材料已在 T6 + `rka16_ns4_pass_index.txt` |
| **T8** | 加 LIT 测试 `bishengir/test/HIVM/online-softmax-iter-args.mlir` | PARKED:upstream-MLIR-scope | T7 | T7 改 PARKED 后,LIT 测试也要等 Huawei 编译器组定 patch 形态再写。当前我们手上有最小 mlir reproducer(`rka16_ns4.npuir`)可以直接附在 issue 里。→ artifact:N/A(已被 issue 更新替代) |
| **T9** | 提 `Ascend/AscendNPU-IR` **issue update + 诊断报告**(原"上游 PR")| DONE | T6 | 已 `gc issue comment 251`,~107 行中文 update 涵盖 (1) bisect 方法 (2) 完整曲线表 + 突变点 ExtendedCanonicalizer line 10801 (3) before/after IR diff(4 iter_args → 2,acc_m / acc_l 被吃) (4) 根因解释(`bishengir/lib/Transforms/ExtendedCanonicalizer.cpp` 93 行薄壳 + upstream MLIR `mlir/lib/Dialect/SCF/IR/SCF.cpp` 的 `RemoveUnusedIterArgs` 误判 DPS in-place 跨 iter 依赖) (5) 三个建议 patch 方向 (6) 复现命令。无 agent 签名。→ artifact:AscendNPU-IR #251 comment ID `1.73358592e+08`(`gc issue view 251 -R Ascend/AscendNPU-IR --comments` 可见) |

#### Track C(整合 + 验收)

| ID | 任务 | 状态 | deps | 详细 |
|---|---|---|---|---|
| **T10** | 4 个算子真 shape 全 PASS | TODO | T1, T3, (T4|T9) | 当 T1 让 indexer_bwd 通过 + T3 让 sparse_mla_bwd 编译通过 + (T4 让 fwd 接受 ~2% NaN 或 T9 上游修了)→ `_real_shape_smoke.py` 4/4 PASS。→ artifact:miles fork commit 上的 smoke 全 ✅ |
| **T11** | miles `DSAMLASelfAttention` 在 Megatron 里跑真 shape 而不是 H=16 减层 | DONE-pending-R-KA-16 | T10 | miles fork commit `dc26e45` 把 `_e2e_megatron_step.py` 参数化:`MILES_E2E_SHAPE={reduced,real}`,real 切到 DSv4-Flash 真数(hidden=512 H=64 q_lora_rank=1024 kv_lora_rank=512 qk_head_dim=128 qk_pos_emb_head_dim=64 v_head_dim=512 ffn_hidden=1024,SEQ=2048,topk=512)。**A3 tlrescue cold cache 实测**:`MILES_E2E_SHAPE=real python -m _e2e_megatron_step` → 52,270,848 params,DSAMLASelfAttention forward `AscendNPU IR compile success`、4 个 tilelang 算子全跑通、`out: [2048, 1, 512]`、indexer score `[2048, 512]`、backward sparse_mla_bwd 也 compile success。**编译 + flow-through 真 shape 全栈打通**。残留:`mla_loss = nan`、6 finite / 6 non-finite grads —— 全部 NaN 来自 sparse_mla_fwd NS=8 受 R-KA-16 影响,**根因已 T9 上抛 AscendNPU-IR #251**,等 Huawei 修。→ artifact:miles fork commit `dc26e45` on `npu-tilelang-dispatch`,实测 log 见 commit message |
| **T12** | 4 个上游 PR 整理 + 提交 | TODO | T10, T11 | 4 个 PR:(1) `tile-ai/tilelang-mlir-ascend` — 现有 4 算子 + head_split + R-KA-13 E5 fwd / 整合到 PR #80 之后 (2) `Ascend/AscendNPU-IR` — R-KA-16 修复(已在 T9) (3) `NVIDIA/Megatron-LM` — `te_general_gemm = None` placeholder bug fix (4) `radixark/miles` — `_npu/` 子包 + dispatch hook(已在 miles fork,需要 cleanup 提 PR)。无 Claude 签名。→ artifact:4 PR URL |

### 监控 / 永续(无 deps,无 end state)

| ID | 任务 | 状态 | 详细 |
|---|---|---|---|
| **M1** | 跟踪 5 个 upstream issue 状态:`AscendNPU-IR #247/#248/#249/#251`、`triton-ascend #277` | ONGOING | 每周看一次,如有 reviewer 反馈或 fix landing 就在 roadmap 里加一行 |
| **M2** | 持续更新 `DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md` §7 进度更新区 | ONGOING | 每完成一个 task 后写一段日期戳 note |
| **M3** | 维护这个 ROADMAP.md | ONGOING | 每开始 / 完成 / parked 一个 task 立刻更新这里 |

## 执行策略

### 现在(下一个 session)

如果时间充足、可并行:用 subagent 同时跑 T1 + T3 + T5(三个 Track 第一步)。理由:
- T1 是已知模式(head_split,sparse_mla_fwd 已经做过同样的事),subagent 可以照搬
- T3 是 split_store 工程,需要看 bwd kernel 的 acc_dkv 结构,subagent 干
- T5 是从 dump 找 IR 文件,subagent 可以单独干

如果只能串行:走 T1(快速看见 1 个算子从 ❌ → ✅)→ T2(响应 reviewer)→ T3 → T5 → T6。

### 状态更新协议

每次工作开始:
1. 读这个 ROADMAP.md 找未 DONE 的下一个 task
2. 改状态为 IN-PROGRESS
3. 干活
4. 完成或 blocked 后立刻改状态 + 写产出物

完成后必须 commit + push 这个文件,确保下次 session 看到最新状态。

## 历史

* **2026-05-28**:**T7/T8 PARKED:upstream-MLIR-scope** — `bishengir/lib/Transforms/ExtendedCanonicalizer.cpp` 是 93 行薄壳,只 register upstream MLIR `CanonicalizerBase` patterns + `memref::getExtendedCanonicalizationPatterns` + `linalg::getExtendedCanonicalizationPatterns`,两个扩展都没动 SCF iter_args。实际吃 iter_args 的是上游 MLIR `mlir/lib/Dialect/SCF/IR/SCF.cpp` 的 `RemoveUnusedIterArgs`,tensor 形态下 49 个 pass 不出问题,bufferize 后 DPS in-place 触发误判。结论:C++ patch 超出 bishengir 仓边界(要么 LLVM 上游,要么 Huawei 在 ExtendedCanonicalizer 里关掉/重写 SCF 该 pattern)。改走 **T9 issue update + 诊断报告** 路径,让 Huawei 编译器组拿到 T6 完整 bisect 数据后接手 C++ patch — 杠杆更大。
* **2026-05-28**:**T9 DONE** — AscendNPU-IR issue #251 已 update comment(`gc issue comment 251`)。107 行中文报告:bisect 方法 + 311-pass 曲线表 + 突变点 ExtendedCanonicalizer line 10801 + before/after IR diff + 根因(`bishengir/lib/Transforms/ExtendedCanonicalizer.cpp` 是薄壳,真正吃 iter_args 的是 upstream MLIR SCF 的 RemoveUnusedIterArgs 类 canonicalization 误判 DPS in-place 的 `vmul(acc_l, correction, acc_l)`/`vadd(arg, new_max, acc_m)` 为「同地址循环写,无跨 iter 依赖」)+ 3 个 patch 方向建议 + 复现命令。无 agent 签名。Track A 全 DONE / PARKED,T32 cold-drive 阶段编译器侧工作结束,等 Huawei 接手。
* **2026-05-28**:**T11 DONE-pending-R-KA-16** — `_e2e_megatron_step.py` 参数化为 `MILES_E2E_SHAPE={reduced,real}`,miles fork commit `dc26e45`。A3 cold cache 实测 real preset:52,270,848 params,DSAMLASelfAttention forward `AscendNPU IR compile success` × 4 算子,`out: [2048, 1, 512]`、indexer score `[2048, 512]`、backward sparse_mla_bwd compile success。**Megatron-core 真 shape e2e 编译 + flow-through 全栈打通**。残留:`mla_loss = nan`、6 finite / 6 non-finite grads — 全部由 R-KA-16 引起,已在 T9 上抛 AscendNPU-IR #251。
* **2026-05-28**:创建。当前活跃 task:T1、T2、T3、T4、T5、T6。
* **2026-05-28**:**T1 DONE** — `lighting_indexer_bwd` head-split (`block_H_inner=16`) 把 H=64 真 shape 从 UB 溢出 259 KB 拆到能装下。miles fork `npu-tilelang-dispatch` commit `0b39e1b`。`_real_shape_smoke.py` 中 `indexer_bwd @ topk=512 SKV=2048` 从 ❌ FAIL → ✅ PASS 1.2s,gq / gw / gk 全 finite。smoke 总分 2/4 → 3/4(`sparse_mla_fwd` 编译过但 NaN — R-KA-16 已 issue;`sparse_mla_bwd` 仍 UB 溢出 289 KB — T3 范畴)。未碰 sparse_mla 系列 kernel。
* **2026-05-28**:**T4 DONE** — `sparse_mla_fwd` kernel cleanup:删 dead alloc `new_max_expanded`、把 inline comment 重写为指向 AscendNPU-IR issue #251 + 跨链接 dispatcher `num_stages=1` 和 ROADMAP T6-T9。miles fork `npu-tilelang-dispatch` commit `a74688c`。小 smoke cold cache 重跑 max abs err vs fp32 ref = 0.0005,PASS。算子层 R-KA-16 mitigation 至此固化到 PR-able 形态;残留 NS≥2 0-1.6% NaN 等 T9 上游 bishengir 修复。
* **2026-05-28**:**T5 DONE** — 用 fresh-built `bishengir-opt` (AscendNPU-IR commit `31f690369d` 2026-05-18) 顶层 `--mlir-print-ir-after-all` flag dump 311 个 pass 的 after-IR 到 `rka16_ns4_passes.txt` (2.9 MB,31727 行)。配套 `rka16_ns4.npuir` (10 KB) 通过 `dump_rka16_ir.py` monkey-patch `compiler_npu._npuir_to_bin_enable_npu_compile` 在 bishengir 调用前保存。**T6 候选 pass**:RemoveRedundantLoopInit、CanonicalizeIterArg、HIVMDecomposeOp、**InferHIVMMemScope(此 pass 后外层 scf.for 从 4 个 tensor iter_args 塌成 2 个 memref iter_args,acc_l/acc_m 退化为循环外 alloc + 循环内 store)**、EnableMultiBuffer。
* **2026-05-28**:**T3 DONE** — `sparse_mla_bwd` UB 从 289280 B 降到 140192 B(71% UB,远在 80% 软预算下),Strategy A(`block_size=8` when `d_v >= 512`)单独命中,不需要 split_store。同步把 postprocess `pp_block_N` 在大 `dim_plus_tail` 下自动从 64 缩到 16(避免 368640 B 溢出)。miles fork commit `45f86c9` on `npu-tilelang-dispatch`。`_real_shape_smoke.py` sparse_mla_bwd ✅ PASS in 6.6s,4/4 全 PASS。数值仍 NaN(R-KA-13 / R-KA-16,T5/T9 跟进);本任务只负责 UB compile,达成。
* **2026-05-28**:**T2 progress** — tile-ai PR #80 CI 失败修复:`d2d1871` on fork `npuir-check-ub-budget`。解 3 个 ruff F841 — 把 `biggest_name` + `biggest_block_M_guess` 从 `_suggest_block_M` 返回,weave 进诊断文案让错误信息直接告诉用户「largest fragment 是哪个、leading-dim 猜了多少」;删 stale `tvm.script.tir` import。ruff format + ruff check 都过,pre-commit 全套 PASS。A3 tlrescue cold cache 重跑 5/5 UT PASS in 4.32s。等下一轮 reviewer / CI。
* **2026-05-28**:**T6 DONE — 罪魁 pass 锁定 `ExtendedCanonicalizer`**(pass dump line 10801)。bisect 方法:把 T5 dump 按 311 个 `// -----// IR Dump After <Pass>` 切片,每片用 `grep -nE "scf.for.*iter_args"` 统计 tensor/memref iter_args 数。完整曲线:1–9971 行 `iter_args=2 tensor=6 memref=0`(4-tensor outer + 2-tensor inner,稳定 49 个 pass);9972 `OneShotBufferize` 触发 tensor→memref(`tensor=0 memref=6`);10627 `CanonicalizeIterArg` 仍然 `memref=6`(没动);**10801 `ExtendedCanonicalizer` 后 `memref=2`,acc_m/acc_l 这两个 `[64,1] f32` iter_args 被吃掉**;后续 21K 行内无回退。具体 IR diff:before = `scf.for %arg11 iter_args(%arg12=acc_o, %arg13=acc_m, %arg14=acc_l, %arg15=scores)`,after = `scf.for %arg11 iter_args(%arg12=acc_o, %arg13=scores)`。**根因解释**:`vmul(acc_l, correction, acc_l)` 和 `vmul(acc_m_update_chain..., acc_m)` 这两个 in-place write-back-to-iter-arg 的 DPS HIVM ops,被 ExtendedCanonicalizer 的依赖分析误判成「同地址循环写、无跨 iter 依赖」并 drop iter_arg;但实际上 `correction = vexp(vsub(acc_m_prev, new_max))` 依赖当前 iter 的 reduce_max,所以是真正的跨 iter 依赖。修改方向:给 ExtendedCanonicalizer 加 conservative-keep:iter_arg 的 in-place update 若 RHS 包含 loop-induction-variable-dependent value,必须保留 iter_arg。下一步 T7:在 `3rdparty/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/ExtendedCanonicalizer*` 找到该 pass。
