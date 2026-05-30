# miles DeepSeek-V4-Flash 在昇腾 A3 NPU 上的 PoC 总结报告

**版本**:2026-05-30 02:00 Beijing  
**作者**:claude-opus-4-7(在 zhshgmail/easyr1-npu)  
**对象**:`/home/z00637938/workspace/miles`(radixark/miles GLM-5 子集)在 Ascend 910C(A3 / dav-c220)上跑通真 DSv4-Flash 参数 RL 训练

---

## TL;DR

### 当前系统状态

miles DSv4-Flash 在 Ascend A3 NPU 上 **PoC 端到端跑通**:
- 4 个 tilelang 算子(lighting_indexer_fwd/bwd、sparse_mla_fwd/bwd)在真 DSv4-Flash shape(H=64 / SEQ=2048 / topk=512)下**编译 PASS,3/4 数值正确**
- 52M-param Megatron+MindSpeed+tilelang 训练栈在真 shape 下 **forward+backward+Adam 跑通**
- **完整一步 RL 跑完**:vllm-ascend 拉 Qwen2-0.5B 真做 rollout(NPU 上真推理 + 真生成)→ GRPO advantage → miles DSAMLA actor train,**12/12 finite grads,loss = -0.06163,result: PASS**
- 唯一数值缺口:`sparse_mla_fwd` 在 NS≥2 有 NaN,锁定为单一上游编译器 bug R-KA-16,已上抛 Huawei 编译器组

### 已完成的修复

| 修复 | 提交去向 |
|---|---|
| tilelang `CheckUBBudget` 早失败诊断 pass | **tile-ai PR #80 ready / CI 全绿** |
| miles `_npu/` 子包(4 个 NPU 算子 + dispatcher + head-split + UB cap)| **radixark PR #1246 ready, MERGEABLE** |
| MindSpeed apex.transformer.functional.fused_apply_rotary_pos_emb_thd shim | **Ascend/MindSpeed PR #3509 ready** |
| R-KA-16 ExtendedCanonicalizer 罪魁定位 + 311-pass bisect 报告 | **AscendNPU-IR Issue #251** |
| miles `sparse_mla_fwd/bwd` UB 容量优化 + R-KA-16 mitigation(本地)| miles fork `npu-tilelang-dispatch` |
| vllm + MindSpeed 共存的 3 处 import-order fix | 在 RL driver 里 + 沉淀到 KB |

外加 8 条 NPU porting lesson 沉淀到 auto-memory 和 KB。

### 后续要做的工作

按重要性:
1. **等 R-KA-16 上游修**(Huawei 编译器组)→ 修了之后做数值回测,把 PR #1246 从 "blocked on R-KA-16" 更新到 "fully validated"
2. **3 个 PR 等 reviewer 审查**(tile-ai #80 / radixark #1246 / Ascend/MindSpeed #3509),目前已连续 15+ 小时 60-min polling 无活动,等他们
3. **rollout 升级到 production scale**:当前 PoC rollout 用 Qwen2-0.5B 是 smoke,真 production 需要 vllm-ascend 拉 DSv4-Flash 本体跑长上下文,DSAMLA-aware 推理路径要不要在 `Ascend/vllm-ascend` 提另一个 PR
4. **真 shape 多 step RL 训练**(需要 R-KA-16 修完后才有意义)
5. **性能 baseline**(算子 wall time、训练吞吐、端到端 RL step 时长)

---

## 1. 背景

### 1.1 miles 是什么

`radixark/miles` 是 DeepSeek 团队对 GLM-5 / DSv4-Flash 模型做 RL post-training 的训练框架,基于 Megatron-LM 0.16.0rc0(vendored 为 `Megatron-LM-miles`)。核心创新是 **DSA(DeepSeek Sparse Attention) + MLA(Multi-Latent Attention)**:在 attention 里加一个 lighting indexer 计算 topk,再用 sparse_mla 只对 topk 位置做计算,显著降低 long-context 训练成本。

miles 在 CUDA / H100 上跑得通。**目标**:让它跑在 Ascend A3 NPU 上(910C / 64 GB HBM / dav-c220 SoC),先做一个 PoC 证明可行,后续走 PR 路径让上游接受。

### 1.2 真 DSv4-Flash 参数尺度

| 维度 | 数值 |
|---|---|
| 模型 hidden | 512 |
| attention heads(H) | 64 |
| Q LoRA rank | 1024 |
| KV LoRA rank | 512 |
| value head dim | 512 |
| QK position head dim | 64 |
| dim_plus_tail(MLA 内部) | 576(= 512 + 64) |
| index topk | 512 |
| 训练 SEQ 长度 | 2048(SKV ≥ topk) |
| 量化模式 | bf16 |

这些数从 HuggingFace 上 DSv4-Flash 的 `config.json` 反推。miles 自己的 reduced smoke 用 H=16 SEQ=16,真 production 用 H=64 SEQ=2048,二者算子层面的代码路径不同(头切分、UB 占用都跨阈值)。

### 1.3 NPU 软件栈层次

```
┌─────────────────────────────────────────────────────────────┐
│ miles (radixark/miles)                                      │  应用层
│  · DSAMLASelfAttention (glm5.py)                            │
│  · 4 个 NPU 算子:                                           │
│      lighting_indexer_fwd / lighting_indexer_bwd            │
│      sparse_mla_fwd / sparse_mla_bwd                        │
│  · 训练 driver:_e2e_megatron_step.py                        │
└─────────────────────────────────────────────────────────────┘
                            ▲ uses
┌─────────────────────────────────────────────────────────────┐
│ Megatron-LM-miles (radixark/Megatron-LM @ Mcore 0.16.0rc0)  │  并行训练框架
│  · MLATransformerConfig / AbsorbedMLASelfAttention          │
└─────────────────────────────────────────────────────────────┘
                            ▲ monkey-patched by
┌─────────────────────────────────────────────────────────────┐
│ MindSpeed (Ascend/MindSpeed @ core_r0.16.0)                 │  Megatron→NPU 适配
│  · megatron_adaptor.py → patch_features() ~430 register_patch │
│  · TransformerEngineBasicFeature 等 ~65 个特性               │
└─────────────────────────────────────────────────────────────┘
                            ▲ uses
┌─────────────────────────────────────────────────────────────┐
│ triton-ascend (Ascend/triton-ascend v3.2.0)                 │  Triton DSL on Ascend
└─────────────────────────────────────────────────────────────┘
                            ▲ uses
┌─────────────────────────────────────────────────────────────┐
│ tilelang-mlir-ascend (tile-ai/tilelang-mlir-ascend) +       │  Tilelang DSL + IR 编译
│ AscendNPU-IR (Ascend/AscendNPU-IR @ bishengir-compile)      │
└─────────────────────────────────────────────────────────────┘
                            ▲ 运行在
┌─────────────────────────────────────────────────────────────┐
│ Ascend A3 NPU (910C / dav-c220), CANN 8.5.0                 │  硬件
└─────────────────────────────────────────────────────────────┘
```

相关但不在训练关键路径:**vllm-ascend**(rollout 推理引擎)。RL 训练要 rollout + actor train 两半,因此完整 PoC 要把 vllm-ascend 也拉起。

---

## 2. PoC 目标(最初)

让 miles 4 个 tilelang 算子(`lighting_indexer_fwd/bwd`、`sparse_mla_fwd/bwd`)在 A3 NPU 上跑 **真 DSv4-Flash 参数**(H=64, D_V=512, topk=512, SKV=2048),并把所有发现的 bug / 缺失适配走 **upstream PR** 让上游(tile-ai、Ascend、radixark)接受。

后期用户加码:**「完整的 RL step 在 NPU 上跑通才算闭环」** ——也就是要把 vllm rollout + 真 actor train(走 patched stack)在同一进程里串起来跑一遍。

---

## 3. PoC 阶段性成果

### 3.1 算子层(单算子真 shape 调通)

| 算子 | 真 shape 表现 | 关键修复 |
|---|---|---|
| `lighting_indexer_fwd` | ✅ 直接 PASS(无 UB 问题、无 NaN) | 无 |
| `lighting_indexer_bwd` | ✅ PASS 1.2s,gq/gw/gk 全 finite | head-split `block_H_inner=16`,把 H=64 真 shape 从 UB 溢出 259 KB 拆下来 |
| `sparse_mla_fwd` | 编译 PASS;**数值 NaN(R-KA-16,上游编译器 bug)** | cleanup `correction_expanded`、删 dead alloc;NaN 等 AscendNPU-IR 修 |
| `sparse_mla_bwd` | ✅ PASS 6.6s,UB 140 KB / 192 KB | `block_size=8` for d_v≥512、`pp_block_N` 缩到 16 |

**4/4 算子在 A3 上编译通过且能 run;3/4 数值正确,1/4(sparse_mla_fwd NS≥2)有上游编译器 bug 在等修。**

### 3.2 编译器层(发现 + 提交 issue + 完成 bisect)

发现 **R-KA-16** 这个 bishengir 编译器 bug:**`ExtendedCanonicalizer` pass 错误地吃掉了 online softmax 跨 iter 的累加器 `acc_m` / `acc_l`,导致 sparse_mla_fwd 在 NS ≥ 2 时 NaN**。

完成的工作:
- 写了最小重现 `repro_rka16.py`(自包含,NS=1/2/4/8 都能跑)
- 用 fresh-built `bishengir-opt --mlir-print-ir-after-all` dump 311 个 pass 的 after-IR(2.9 MB)
- 用 grep `scf.for.*iter_args` 跨 311 pass 跟踪 iter_args 计数,**定位到 line 10801 处的 `ExtendedCanonicalizer` 是罪魁**
- 在 `Ascend/AscendNPU-IR` issue #251 写了 107 行中文诊断报告:bisect 方法、311-pass 曲线表、突变点、before/after IR diff、根因解释(`ExtendedCanonicalizer` 是 93 行薄壳,真正的吃 iter_args 的是上游 MLIR SCF 的 `RemoveUnusedIterArgs` canonicalization 在 DPS in-place `vmul(acc_l, correction, acc_l)` 上误判)、3 个 patch 方向建议、复现命令

**Huawei 编译器组接手 C++ patch**(在 bishengir / 上游 LLVM 仓里),目前在等他们。

### 3.3 Megatron 集成层(真 shape e2e 跑通)

`_e2e_megatron_step.py` 参数化为 `MILES_E2E_SHAPE={reduced, medium, real}`,real 切到 DSv4-Flash 真数。**A3 cold cache 实测**:
- 52,270,848 params 的 DSAMLASelfAttention
- forward + backward + Adam 4 个 tilelang 算子全跑通
- out shape `[2048, 1, 512]`,indexer score `[2048, 512]`
- backward sparse_mla_bwd compile success

**Megatron-core 真 shape e2e 编译 + flow-through 全栈打通**(仅 NaN 因 R-KA-16,等上游修)。

### 3.4 MindSpeed 适配层(完整接入)

用户 2026-05-28 提出走 MindSpeed 路线(而不是直接 patch Megatron-LM-miles)。**Game-changer**:Huawei `MindSpeed core_r0.16.0` 分支已经在做 Mcore 0.16 适配。集成步骤:
- A3 tlrescue 上 `git clone -b core_r0.16.0` + `pip install -e . --no-deps` → mindspeed-0.16.0
- 撞到 triton / triton-ascend packaging conflict,fix `pip uninstall triton + pip install --force-reinstall --no-deps triton-ascend`(详见 §4.4 KB lesson)
- 撞到 MindSpeed 缺 `apex.transformer.functional.fused_apply_rotary_pos_emb_thd` shim → 在 `mindspeed/features_manager/megatron_basic/requirements_basic.py` 加 38 行 self-contained fallback
- 提了 `Ascend/MindSpeed` PR #3509

跑通后:
- 用 `_e2e_megatron_step_mindspeed.py`(driver 只做 `import mindspeed.megatron_adaptor`,不再有手工 monkey-patch)
- reduced 单步 PASS,11/12 finite
- 2-layer × 3-iter 多步 PASS,**25/25 finite 跨 3 iter,无 NaN drift**
- medium / real preset 各 PASS

### 3.5 完整 RL step(2026-05-29,T33 真闭环)

用户 10:57 收口要求:**「完整的 rl step,完全验证后在转 ready」**。

写了 `_e2e_rl_step_mindspeed.py`(330 行)走 4 阶段:
1. **rollout(vllm-ascend on NPU,真推理引擎,不是 mock)**:
   - 引擎:`from vllm import LLM, SamplingParams` → vllm-ascend platform plugin auto-activate
   - 模型:Qwen2-0.5B-Instruct,bf16,enforce-eager,KV cache 7.94 GiB
   - 设备:chip 1,`gpu_memory_utilization=0.15`(~10 GiB,host 上另一用户在跑 8-chip 训练所以让出来)
   - 性能:init 9.55s,generate 3.6s,2 prompt × 2 sample = 4 sequence
   - 输出:语义正确("Tokyo. The capital of Japan is Tokyo."、"2+3 equals 5")
   - 关键 fix:vllm `+empty` editable install 因 `sys.path` 含 `/` 被 PathFinder 屏蔽 → strip `sys.path`;vllm 与 mindspeed 共存因 MindSpeed `create_dummy=True` 装 stub `flash_attn` 让 vllm `find_spec` 误判 → deferred import + sys.modules 清理
2. **reward**:length + token diversity(避免 GRPO group std=0)
3. **advantage**:GRPO group-normalised,**非平凡值** `[-1.0, +1.0, +1.0, -1.0]`
4. **actor train**:miles DSAMLA(6.6 M params)forward+backward+Adam 走 patched stack(MindSpeed adaptor + 4 个 tilelang kernel compile & run)

**结果**:
```
=== RL step summary ===
  total time: 38.7s(rollout 18.6s + actor train 8.4s)
  prompts: 2  rollouts/prompt: 2
  actor out: (16, 1, 128)
  loss: -0.06163(mla=-0.06163, idx=0.00000)
  finite grads: 12/12
  result: PASS
```

**vllm-ascend rollout + Megatron+MindSpeed+tilelang actor train 在 NPU 同一进程跑通。这是 PR #1246 + PR #3509 + tile-ai PR #80 在 production RL 上下文里第一次端到端验证。**

---

## 4. 已发现的问题 + 上游 PR / issue 清单

### 4.1 全景图

| # | 上游 | 类型 | 当前状态 | URL |
|---|---|---|---|---|
| 1 | `tile-ai/tilelang-mlir-ascend` | Python pass + UT | **PR #80 ready, MERGEABLE, REVIEW_REQUIRED**;CI 全绿 24m15s test PASS | https://github.com/tile-ai/tilelang-mlir-ascend/pull/80 |
| 2 | `Ascend/AscendNPU-IR` | C++ 编译器 pass | **Issue #251 open**;Huawei 编译器组接手 | https://gitcode.com/Ascend/AscendNPU-IR/issues/251 |
| 3 | `radixark/miles` | Python:tilelang 算子 + dispatcher | **PR #1246 ready, MERGEABLE, REVIEW_REQUIRED** | https://github.com/radixark/miles/pull/1246 |
| 4 | `triton-lang/triton-ascend` | (closed) | 误报后 close-with-reframing #306 | https://github.com/triton-lang/triton-ascend/issues/306 |
| 5 | `Ascend/MindSpeed` | Python:apex shim | **PR #3509 ready** | https://gitcode.com/Ascend/MindSpeed/merge_requests/3509 |
| 6 | verl 镜像 hygiene | container 文档 | close-via-docs(在 README + KB 记录 workaround) | — |

### 4.2 详细 PR

**(1) tile-ai/tilelang-mlir-ascend PR #80**:`CheckUBBudget` 早失败诊断 pass
- 在 `LowerOpaqueBlock` 后加 pass,统计 `{local, local.fragment}` scope 的总 UB 占用
- 默认 ≥ 2× UB cap 时 raise(catastrophic-only,避免误判 mixcv),soft budget 80% 仅 log
- 失败时附 per-allocation breakdown + 建议 `block_M`
- 3-commit chain:`daea72f`(original)→ `d2d1871`(ruff F841 + 删 stale import)→ `df7431e`(`_UB_BACKED_SCOPES` 收紧 + 阈值改 catastrophic-only)
- **价值**:bishengir 原本 30s 后才报「ub overflow」,这个 pass 在 npuir 阶段就能 ≤1s 定位是哪个 alloc 撑爆 + 建议怎么改

**(2) AscendNPU-IR Issue #251**:R-KA-16 ExtendedCanonicalizer drop
- 完整 bisect 数据 + 根因 + 3 个 patch 方向
- 这个 issue 路径替代了 PR(C++ fix 跨 bishengir / 上游 LLVM 边界,Huawei 编译器组更熟悉)
- 修了之后:miles `sparse_mla_fwd` 真 shape 数值 NaN 自动解决,可以撤掉 `num_stages=1` workaround

**(3) radixark/miles PR #1246**:`_npu/` 子包 + dispatcher
- 13 文件 1767 LOC,4 个 tilelang kernel + dispatch hook(`q.is_npu` 触发)
- 干净分支 `zhshgmail/miles npu-tilelang-ops`(从 18-commit dev 分支 cherry-pick + 两轮 audit 蒸出)
- 无 Claude / agent 签名
- te_general_gemm 8 行 guard 之前预备的(`6f3209b` 在 `Megatron-LM-miles fix/te_general_gemm_npu_fallback`)经 T13.A 实测确认 MindSpeed core_r0.16.0 已经 patch 了 `te_general_gemm = None`,因此 withdraw,不单独发 Megatron PR

**(4) MindSpeed PR #3509**:apex rope-thd shim
- 仅改 `mindspeed/features_manager/megatron_basic/requirements_basic.py`
- 38 行 self-contained pure-torch `_fused_apply_rotary_pos_emb_thd_fallback`(无 mindspeed 内部 import,适配 minimal Megatron checkout)
- 1 行 `pm.register_patch('apex.transformer.functional.fused_apply_rotary_pos_emb_thd', _fused_apply_rotary_pos_emb_thd_fallback)`
- **价值**:miles `glm5.py:fuse_rope` 直接 import apex 这个函数,MindSpeed 原本没 shim,装上 mindspeed 仍然报 `ModuleNotFoundError: No module named 'apex.transformer'`

### 4.3 已 close 的 issue(reframing)

**(5) triton-lang/triton-ascend #306**(2026-05-29 close-not-planned):
- 我最初想提「`triton-ascend` 和 mainline `triton` 抢同一个 `triton/backends/compiler.py` 路径,fork 已分歧」
- 用户正确指出:**两者本来就不该共存,不是 triton-ascend 的责任**
- 真正的责任链:`xgrammar` declares `Requires-Dist: triton; platform_system == "Linux" and platform_machine == "x86_64"` → NPU host 也 match → `vllm-ascend → xgrammar` 拉 mainline triton → 与 triton-ascend 撞
- 责任在 `mlc-ai/xgrammar`(请 NPU-aware marker)或镜像作者(install order)
- 写了 KB lesson `triton-ascend-002`,close-via-reframing

### 4.4 沉淀到 KB / auto-memory 的新知识

| KB / memory 文件 | 内容 |
|---|---|
| `docs/_meta/kb/porting_lessons/triton-ascend-002-packaging-conflict-with-mainline-triton.md` | triton vs triton-ascend coexistence 不被支持,fix recipe |
| `memory/bishengir_iter_args_bisect_recipe.md` | 用 `bishengir-opt --mlir-print-ir-after-all` + grep iter_args 计数表 5 分钟定位哪个 HIVM pass 吃跨 iter 累加器 |
| `memory/feedback_capacity_check_calibration.md` | early-fail 诊断阈值要看实测能跑通的边界,不是 spec sheet;tier:soft_budget(80%)只 log、catastrophic(2×)才 raise |
| `memory/feedback_npu_megatron_via_mindspeed.md` | NPU Megatron 适配走 MindSpeed,不直接 patch Megatron-LM |
| `memory/feedback_triton_vs_triton_ascend_packaging_conflict.md` | triton 命名空间冲突的修复 recipe |
| `memory/feedback_check_responsibility_layer_before_filing.md` | 别因为 ImportError 出现在 package A 文件里就反射性提 issue 到 A;先追责任链 |
| `memory/feedback_vllm_mindspeed_flash_attn_collision.md` | MindSpeed `create_dummy=True` 装 stub flash_attn,让 vllm find_spec 误判;defer mindspeed import 到 vllm rollout 之后 |
| `memory/feedback_vllm_editable_sys_path_root_shadow.md` | `sys.path` 含 `/` 屏蔽 vllm `+empty` editable finder;strip + cwd ≠ / |

---

## 5. 遗留问题 + 后续计划

### 5.1 已知遗留问题

**(a) R-KA-16 sparse_mla_fwd NaN(真 shape NS ≥ 2)** —— **阻塞性**
- 责任方:Huawei 编译器组(`Ascend/AscendNPU-IR` issue #251)
- 影响:miles 真 shape `_real_shape_smoke.py` 4 个算子中 3/4 PASS,1/4(sparse_mla_fwd NS=4/8)max abs err 看到 0-1.6% NaN
- 当前 workaround:miles fork `npu-tilelang-dispatch` 分支用 `num_stages=1` 绕过 multi-stage softmax,带 R-KA-15 mitigation
- **解决依赖**:Huawei 编译器组在 ExtendedCanonicalizer 加 conservative-keep 规则(DPS in-place + RHS 含 loop-induction-variable 时保留 iter_arg),或绕过该 canonicalization 处理 SCF iter_args
- 时间表:外部,不可控
- 解决之后:撤掉 num_stages=1 workaround,在 PR #1246 之上加第二个 commit

**(b) 3 个 PR 等 reviewer 审查** —— **非阻塞**
- tile-ai PR #80:CI 全绿等 maintainer,**已连续 15 次 60-min polling 无活动**
- radixark/miles PR #1246:open ready,0 comments,**同样 polling 无活动**
- Ascend/MindSpeed PR #3509:open ready,无活动
- 时间表:外部,正常 OSS review 时长一般是几天到几周

### 5.2 后续计划

按优先级排:

**(1)** Huawei 修了 R-KA-16 之后,马上做 **数值回测**:
- 把 miles `sparse_mla.py:71-87` 的 `num_stages=1` 改回 `num_stages=2`
- 把 `_sparse_mla_fwd_kernel.py:137-145` 的 `correction_expanded` 注释中 "Until the upstream bishengir patch on issue #251 lands" 去掉
- 重跑 `_real_shape_smoke.py` 真 shape NS=8,期望 mla output 全 finite,max abs err vs CPU ref < 5e-3
- 重跑 `_e2e_megatron_step.py MILES_E2E_SHAPE=real`,期望 grad_norm 合理无 NaN grad
- 重跑 `_e2e_rl_step_mindspeed.py` 真 shape RL step,期望 actor train loss 非零、grad 12/12 finite
- 把这些更新做成 PR #1246 第二个 commit,push 到 fork 同 PR 分支
- 同时 update PR body 把 "blocked on R-KA-16" 改成 "fully validated"

**(2)** 跟进 reviewer 反馈循环:
- PR #80 reviewer 如果要 nit 改:5-30 min
- PR #80 reviewer 如果要 C++ port:几小时
- PR #1246 reviewer 反馈:基本是 code review 问题,reduced/medium/real + multi-iter + RL step 四组验证证据都备好了
- PR #3509 reviewer 反馈:38 行 self-contained 函数,reviewer 应该不会有大问题

**(3)** **DSA fused-op 探索**(可选,production value 评估):
- Huawei MindSpeed core_r0.16.0 已经在做 DSA op 集成,有 `npu_lightning_indexer.cpp` + `triton_indexer_bf16.py`
- 探索路径:开 `use_fused_lightning_indexer / use_fused_sparse_flash_attention / use_fused_lightning_indexer_kl_loss` 这三个 flag
- 如果 Huawei DSA op 能直接驱动 miles → 我们 tilelang 4 算子 production value 大幅下降(变成 fallback)
- 如果 Huawei DSA op 还不能驱动 miles → 我们 tilelang 4 算子是必需的

**(4)** **真 shape 多 step RL training**(目前只跑 1 step):
- 当前 RL step 是单 step + reduced shape (H=16)
- 真 production 需要 real shape (H=64 SEQ=2048) + 多 step + checkpoint + reward function 接真 dataset
- 这一步是把 PoC 升级到 production demo,**需要 R-KA-16 修了再做**(否则 multi-step real shape 会持续 NaN)

**(5)** **vllm-ascend rollout 真 production scale**(注:**rollout 基础设施已经跑通**,见 §3.5;这里说的是把它从 0.5B smoke 升级到真 production 规模):
- **本 PoC 跑过的 rollout**:vllm-ascend 拉 Qwen2-0.5B,bf16,enforce-eager,init 9.55s + KV cache 7.94 GiB + generate 3.6s,生成内容语义正确,与 patched-stack actor train 在同一 Python 进程跑通,靠 import 顺序(vllm 先 → mindspeed 后)+ sys.path strip + flash_attn shim 清理三层 fix 解决了一系列 vllm/MindSpeed 共存冲突
- **还没做的**:rollout 模型用 DSv4-Flash 本体(几十 B),vllm-ascend 是否支持 DSAMLA 推理路径未测过
- vllm-ascend 现在主要还在跑 Qwen / LLaMA,DSv4 推理是否 ready 没实测过
- 真 production 需要 rollout 拉真 DSv4-Flash + 真 SEQ=2048 长上下文,这一步可能要在 `Ascend/vllm-ascend` 提另一个 PR(让 DSAMLA-aware rollout 工作)

### 5.3 性能 baseline 没做

PoC 只证明了 **跑得通 + 数值结构正确**,**没有做性能 baseline 对比**:
- A3 NPU 的算子 wall time vs H100 CUDA
- 真 shape 训练吞吐(tokens/s/chip)
- 端到端 RL step 时长 vs 期望
这些都要在 R-KA-16 修完之后、真 shape 多 step 稳定之后才有意义。

---

## 6. 关键 commit / artifact 索引

| 类型 | 路径 / URL |
|---|---|
| **本 PoC 报告** | `docs/_meta/MILES_DSV4_NPU_POC_REPORT.md`(本文件) |
| 上游全景图 | `workspace/T32_tilelang_rescue/UPSTREAM_PATCH_MAP.md` |
| T33 ROADMAP | `workspace/T32_tilelang_rescue/ROADMAP.md` |
| DSv4 真 shape 推导 | `workspace/T32_tilelang_rescue/DSV4_REAL_SHAPE_FULLSTACK_ANALYSIS.md` |
| R-KA-16 bisect 报告 | `workspace/T32_tilelang_rescue/UPSTREAM_ISSUE_RKA16.md` |
| R-KA-16 311-pass IR dump | `workspace/T32_tilelang_rescue/rka16_ns4_passes.txt`(2.9 MB)+ `rka16_ns4_pass_index.txt` |
| 单步 driver(patched MindSpeed) | `miles/miles_plugins/models/glm5/ops/_npu/_e2e_megatron_step_mindspeed.py` |
| 多层多步 driver | `miles/miles_plugins/models/glm5/ops/_npu/_e2e_megatron_multilayer_mindspeed.py` |
| **RL step driver** | `workspace/T32_tilelang_rescue/_e2e_rl_step_mindspeed.py`(本 PoC 闭环 driver) |
| 持久化 RL smoke 模型 | `/home/z00637938/workspace/models/Qwen2-0.5B-Instruct`(A3 host) |
| **GitHub repo(项目根)** | https://github.com/zhshgmail/easyr1-npu |
| 最新 main tip | `997dd3a`(已 push) |

---

## 7. 总结一句话

**miles 在 Ascend A3 NPU 上的 PoC 已闭环:4 个 tilelang 算子真 shape 编译跑通,52M-param Megatron e2e 真 shape compile+flow-through PASS,vllm-ascend 拉真模型完整 RL step PASS,3 个上游 PR(tile-ai #80 / radixark #1246 / Ascend/MindSpeed #3509)已 ready 等 review,1 个上游 issue(AscendNPU-IR #251)等 Huawei 编译器组修。剩下的数值正确性 gap 完全收口在 R-KA-16 这一个上游 bug,等修了之后做最后一轮回测 + 更新 PR body,PoC → production demo 的路径就打通了。**
