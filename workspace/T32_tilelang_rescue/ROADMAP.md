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
* sparse_mla_fwd 真 shape:❌ 算子层 NaN(R-KA-16 多 iter softmax bug,issue #251 filed)— 编译已通过
* sparse_mla_bwd 真 shape:❌ 算子层依赖 fwd + bwd 自己 UB 溢出 289KB(T3)
* 编译器诊断 PR #80 已开,等 maintainer review
* 已 file 5 上游 issue(RKA-13/14/15/16 + sglang triton)

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
| **T2** | 跟进 tile-ai PR #80 review | IN-PROGRESS | — | T1, T3, T4 | 看 reviewer 反馈,如要 C++ port 转化、调注释,响应。 → artifact:PR #80 merged 或 next-revision commit |
| **T3** | 给 `sparse_mla_bwd` 加完整 R-KA-13 E5 修复 + 把 bwd UB 用 split_store 拆到 < 192KB | TODO | — | T1, T2, T4 | bwd 已知有 R-KA-13 E5 partial workaround(commit `502c29f`),但 UB 仍 289KB。需要:(1) 把 `acc_dkv [BS=32, D=512] fp32 = 64KB` 用 split_store(每次只算 BS=16 半,scatter atomic_addx4 两次)拆 (2) `block_size=32` → 16 缩小 BS 维度。→ artifact:miles fork commit + 真 shape bwd compile PASS(数值仍 R-KA-13 待修) |
| **T4** | 给 `sparse_mla_fwd` 应用 R-KA-13 E5(`correction_expanded`)固化到 PR-able 形式 | DONE | — | T1, T2, T3 | 已在 miles fork `4cdfc1f` 加了。`a74688c` 清理:删 dead alloc `new_max_expanded`、把 inline comment 改成指向 AscendNPU-IR #251 + 跨链接到 dispatcher `num_stages=1` 和 ROADMAP T6-T9。小 smoke (B=1 S=8 SKV=16 H=16 D=64 topk=8) cold cache 重跑:max abs err vs fp32 ref = 0.0005,PASS。**残留**:真 shape (NS≥2) 仍 0-1.6% NaN — 算子层已尽力,根因等 T9 上游修。→ artifact:miles fork commit `a74688c` |

#### Track A(编译器 bisect + 真修,深度工作)

| ID | 任务 | 状态 | deps | 详细 |
|---|---|---|---|---|
| **T5** | 在 A3 上用 `bishengir-compile -print-after-all` 跑 `repro_rka16.py` NS=4 的 IR | TODO | T0.5 (repro 已 run-pass) | 直接调用 `bishengir-compile` CLI,而不是从 tilelang JIT 走。需要 dump tilelang 生成的 npuir 文件路径,提取 IR,然后 `bishengir-compile /tmp/kernel.npuir --print-after-all 2>&1 > /tmp/passes.txt`。→ artifact:`/tmp/passes.txt` 里每个 pass 后的 IR |
| **T6** | bisect 哪个 HIVM pass 后 `acc_o` 的 iter_arg 状态变错 | TODO | T5 | 从 T5 dump 里搜 `acc_o` 在 scf.for iter_args 列表里是否一直出现、ConstantFolding 后是否被 elimitate、buffer alloc 是否被 demote。看是否在 `CanonicalizeIterArg` / `RemoveRedundantLoopInit` / `EnableMultiBuffer` 中的某个 pass 后出问题。→ artifact:罪魁 pass 名称 + 修改方向 |
| **T7** | 写 bishengir HIVM pass 的 fix(C++) | TODO | T6 | 修改 `3rdparty/AscendNPU-IR/bishengir/lib/Dialect/HIVM/Transforms/<罪魁>.cpp` 或对应 SCF pass。具体要等 T6 才知道。→ artifact:AscendNPU-IR commit |
| **T8** | 加 LIT 测试 `bishengir/test/HIVM/online-softmax-iter-args.mlir` | TODO | T7 | 验证修复后 NaN 不再出现。最小 LIT test:scf.for with iter_args carrying acc_o,跑修复后应该保留 acc_o 在 iter_args 里、bishengir 编出来的 .o 跑 NPU 不出 NaN。→ artifact:test file |
| **T9** | 提 `Ascend/AscendNPU-IR` 上游 PR | TODO | T7, T8 | gc pr create,无 Claude 签名,带 (1) issue #251 reference (2) 单元测试 (3) 真 shape e2e PASS 截图。→ artifact:Ascend/AscendNPU-IR PR URL |

#### Track C(整合 + 验收)

| ID | 任务 | 状态 | deps | 详细 |
|---|---|---|---|---|
| **T10** | 4 个算子真 shape 全 PASS | TODO | T1, T3, (T4|T9) | 当 T1 让 indexer_bwd 通过 + T3 让 sparse_mla_bwd 编译通过 + (T4 让 fwd 接受 ~2% NaN 或 T9 上游修了)→ `_real_shape_smoke.py` 4/4 PASS。→ artifact:miles fork commit 上的 smoke 全 ✅ |
| **T11** | miles `DSAMLASelfAttention` 在 Megatron 里跑真 shape 而不是 H=16 减层 | TODO | T10 | 把 `_e2e_megatron_step.py` 的 `MLATransformerConfig` 换成 DSv4-Flash 真数(num_attention_heads=64, q_lora_rank=1024, kv_lora_rank=512, qk_head_dim=128, qk_pos_emb_head_dim=64, v_head_dim=512, hidden_size=4096 但单卡装不下时降到 hidden=512 + 1 layer)。验证 forward + backward + Adam step 都 finite。→ artifact:miles fork commit + 实测结果 |
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

* **2026-05-28**:创建。当前活跃 task:T1、T2、T3、T4、T5、T6。
* **2026-05-28**:**T1 DONE** — `lighting_indexer_bwd` head-split (`block_H_inner=16`) 把 H=64 真 shape 从 UB 溢出 259 KB 拆到能装下。miles fork `npu-tilelang-dispatch` commit `0b39e1b`。`_real_shape_smoke.py` 中 `indexer_bwd @ topk=512 SKV=2048` 从 ❌ FAIL → ✅ PASS 1.2s,gq / gw / gk 全 finite。smoke 总分 2/4 → 3/4(`sparse_mla_fwd` 编译过但 NaN — R-KA-16 已 issue;`sparse_mla_bwd` 仍 UB 溢出 289 KB — T3 范畴)。未碰 sparse_mla 系列 kernel。
* **2026-05-28**:**T4 DONE** — `sparse_mla_fwd` kernel cleanup:删 dead alloc `new_max_expanded`、把 inline comment 重写为指向 AscendNPU-IR issue #251 + 跨链接 dispatcher `num_stages=1` 和 ROADMAP T6-T9。miles fork `npu-tilelang-dispatch` commit `a74688c`。小 smoke cold cache 重跑 max abs err vs fp32 ref = 0.0005,PASS。算子层 R-KA-16 mitigation 至此固化到 PR-able 形态;残留 NS≥2 0-1.6% NaN 等 T9 上游 bishengir 修复。
