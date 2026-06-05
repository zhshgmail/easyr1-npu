# M3 handoff — CANN-native DSv4 dispatcher PoC (for an intranet-A3 agent)

> blue 写于 2026-06-05。我（blue）只能看到一台 A3(`115.190.166.102`),它被他人 117-sglang 集群 + 2 容器占满(16 davinci 全 mount,HBM 基线但设备被占)。无法在它上面起干净容器跑 M3。需要内网另一台 A3 + 能看到它的 agent 接手。本文件是接手指南。

## 0. 一句话

把最新 `radixark/miles` main 的 DSv4 训练算子(纯 GPU tilelang)在 NPU 运行层接 **torch_npu CANN-native** 算子,做单算子 fwd+bwd e2e 数值验证,达 PR-bar。**不是 tilelang re-port**——CANN-native 已验证全覆盖。

## 1. 起点(git)

- repo: `github.com/zhshgmail/easyr1-npu` main `8dcc34c`(本文件所在)。
- 计划全文: `workspace/task-dag-realdelta/REBASE_ON_LATEST_MILES_PLAN_2026-06-05.md`(M1-M5)。
- 映射依据: `M1_LATEST_MILES_USECASE_UPSTREAM_MAP_2026-06-05.md` + `MILES_REBASE_ASSESSMENT_2026-06-05.md`。
- miles 最新 main 在 `/home/z00637938/workspace/miles`(origin=radixark/miles;`git fetch origin` 取 main `74198b45`)。

## 2. 已完成(M1/M2/M4,本地,独立 agent 验证零 REFUTED)

- M1 差异基线 + UPSTREAM_FORKS miles 行;M2 cookbook 重整;M4 DSV4 report §九。详见计划 doc「进度」段。

## 3. M3 要做的(接手者执行)

1. 在最新 `radixark/miles` main 建分支 `blue/fix/dsv4-npu-cann-native-dispatcher`(或你的 slug)。
2. DSv4 plugin 的 6 个算子调用点(实查):
   - `deepseek_v4.py:35` `sparse_attn_tilelang` → **`npu_nsa_select_attention`(+`_grad`)**
   - `v4_indexer.py` `batched_indexer_fwd` → **`npu_lightning_indexer`(+`_grad`)**
   - `compressor.py` `DeepSeekV4Compressor` → **`npu_nsa_compress_attention`(+`npu_nsa_compress_grad`)**
   - `precision_aligned_ops.py` `linear_bf16_fp32` → 纯 torch(A3 直接可用,无需换)
   - `qat.py` `fp8_simulate_qat` → fp8,**A3 硬件墙,QAT-off 路不调,先不接**
   - `RMSNorm`(compressor.py:14)→ 保留纯 torch FP32 或 `npu_rms_norm`
3. dispatcher 用 `q.is_npu` 分流(同 glm5 `_npu/` 模式,见 `miles-001` cookbook 的 dispatcher hook 写法)。
4. **★ M3 头号风险(预研发现,务必先验):** 最新 main `sparse_mqa_fwd_interface(q,kv,attn_sink,topk_idxs,...)` 多了 **`attn_sink[H]fp32`**(softmax sink),而 `npu_nsa_select_attention` 的 native 签名里没有显式 attn_sink。**第一件事 = A3 上确认 attn_sink 怎么接**:(a) native 是否有隐藏/新版 attn_sink 支持,(b) 否则 native softmax 后做 sink 调整(post-hoc lse 合并)。torch_npu C-binding 签名 `*args/**kwargs` 不可 introspect,只能实跑确认。
   - native 约束(已验):TND layout,bf16,D_qk=192/D_v=128,select_block_size=64(仅 64),select_block_count=16,KV S≥1024(64 倍数),G=Nq/Nkv≤32,topk int32 [0,S2/64)。errno 561103 = shape/param 违反。
5. UT(高覆盖,达 `feedback_pr_quality_bar`):每算子 fwd+bwd vs 参考(tilelang GPU 版或 torch naive)数值对;attn_sink 路;dtype/layout 还原;dispatcher 分流。
6. A3 e2e:单算子真机跑通 fwd+bwd 数值对,出 e2e 报告。
7. 完成回写 repo(report §九.4 + 计划 doc M3 状态 + 新 cookbook 若需),并**独立 agent 验证**(per-milestone 纪律)。

## 4. 容器(需重建)

- 内网 A3 上需起带 NPU 的容器(参考 CLAUDE.md「外部基础设施」+ `npu-container-runner` SKILL:bind `/home/z00637938→同路径`、device davinci、`--shm-size`)。CANN 8.5(torch_npu 2.9.0 有上述 native op,已验)即可;不需 9.1.0(fp8 在 A3 是硬件墙,与 M3 bf16 路无关)。
- 资源:M3 单算子 e2e **1–2 张 davinci 足够**,不需要 16 张。

## 5. 红线(blue 的纪律,接手者请沿用)

- PR-bar: 只有验证过(测试+高 UT+e2e 报告)的补丁才提 PR;不自动提(M5 待 owner)。
- 上游 PR/issue 无 agent 签名,按 radixark/miles 规范。
- 诚实分类: 符号存在 ≠ 执行正确;e2e 数要真落盘。
