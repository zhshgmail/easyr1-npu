# SESSION_HANDOVER — V4 真路径在 NPU 上 attempt — **PASS 2026-06-01 06:36 UTC**

> 本文件是 auto-compact 防丢失保险。任何时候 agent 被 compact 后接手,**先读这里**。

## 🔖 LATEST IN-FLIGHT STATE (2026-06-01 ~09:25Z — read first)

- **本轮新增 (repo HEAD `afd2c7d`, pushed)**:
  - ✅ **#310 native op swap DONE + e2e verified**: V4 torch fallback → native torch_npu。换了 `npu_rms_norm`(bit-exact 0.0)+ `npu_clipped_swiglu(alpha=1.0,bias=0.0,interleaved=False)`(bf16-ulp 等价)。RoPE **不换**(实测 npu_rotary_mul/apply_rotary 是 rotate-half 约定,差 4.3;V4 是 interleaved-complex;fp32 torch 更准)。换完 V4 RL loop 重跑 PASS(5/5 distinct, EXIT=0)。snapshot+harness+findings 在 `workspace/v4_attempt_2026_06_01/native_op_snapshots/`(commit `9348868`)。
  - ✅ **V4 训练侧 op gap inventory**: `workspace/v4_attempt_2026_06_01/V4_TRAINING_SIDE_OP_GAP.md`。真训练侧在 `miles-v4-extracted/.../models/deepseek_v4/ops/`。训练侧 e2e 卡 **6 个 TileLang kernel**(sinkhorn/act_quant/indexer fwd+bwd/sparse_mla fwd+bwd)+ 3 个纯 torch module(compressor/hc/rope)。**交叉验证**了 op-gen 的 CPU-truth model.py 跟真训练 kernel `sinkhorn.py` 的 pre/post/comb + 迭代结构逐行一致。
  - ⏸ **#311 hc_split_sinkhorn op-gen 重跑 → 停在 await_user_decision**: a5_ops 三个 bug(task#28/29/30)merge 到 origin/main `30f385e7` 我 pull 重跑。routing 走对了(没 IL 误路由 ✓)。但发现**第 4 个 bug**:task#28 name-gate 漏了 `kw_brief.py:172` call-site,它还用旧 tag-based `is_fa_class` → 给 worker 注入 "STOP DO NOT AUTHOR" → 两 gate 矛盾。worker probe-first 顶住没乱写,handoff 诊断(在 `workspace/hc_split_sinkhorn/PROGRESS.md`)。已报 main + 给出一行 fix(gate 换 `is_attention_named(workspace.name)`)。**memory `a5ops_fa_gate_two_callsites.md`**。team 修 push 后 pull → resume `orchestrator.py hc_split_sinkhorn --lane 0`(workspace 停在 await_user_decision,会接着走)。
  - ✅ **#313 DONE — 推翻假设**(repo `c41393f`): 起了 **`miles_v4_train_probe` 容器**(verl-8.5.2 镜像,无 davinci 挂载 = 无 UDA 风险,miles_plugins 解到 `/opt/miles`)+ editable 装 megatron-core 0.16.0rc0 + mindspeed 0.16.0(复刻 tlrescue setup)。实测:**只有 `ops/utils.py` 纯 torch import OK**;compressor + hyper_connection **import 即拉 tilelang 且 forward 真调** act_quant / hc_split_sinkhorn(hyper_connection.py:55);rope 需 `miles_megatron_plugins`。**结论:训练侧 module 不可拆"纯 torch 先跑/tilelang 后补",6 个 TileLang kernel 是 forward 硬依赖** → hc_split_sinkhorn(#311)在训练侧 e2e 真关键路径上。
  - **tlrescue megatron/mindspeed 来源**(供复刻): editable installs,`megatron-core 0.16.0rc0`→`/home/z00637938/workspace/Megatron-LM-miles`,`mindspeed 0.16.0`→`/home/z00637938/workspace/MindSpeed-clone`(都是 host 挂载卷)。verl-8.5.2 **镜像本身不含** megatron/mindspeed,是 tlrescue 里 editable 装的。
  - **两条线收敛到同一 blocker**: #311(sinkhorn op-gen,等 main 修 kw_brief:172 一行)= 训练侧 forward 硬依赖。main 那个 fix 解锁的不只一个 kernel,是训练侧 e2e 第一块。fix push 后:pull a5_ops → resume `orchestrator.py hc_split_sinkhorn --lane 0`。
  - **container 清单**(本 session): `miles_v4_train_probe`(新,训练侧 import 验证,可复用/可删)。a5ops-a3(op-gen build,davinci2)。sgl_probe(V4 推理 PASS,davinci1)。tlrescue(sacred,davinci14)。

- **两个 V4 milestone DONE + pushed**: generate() PASS (`fc2f486`) + RL loop PASS (`d65c219`, synth-delta 占位). 14-gap 分析 + fp8/bf16 分析已 push.
- **上游 PR 决策**: V4 sglang-NPU adapter patch **不零散提**,留 fork,等整个 miles+V4 RL loop 真 e2e(真训练非 synth-delta)再批量 PR. Issue draft HELD 在 `workspace/v4_attempt_2026_06_01/UPSTREAM_ISSUE_sglang_v4_npu.md`. (memory `project_v4_upstream_pr_batch_after_e2e.md`)
- **a5_ops 协作群**(Discord `1501649396922712105`,我 alias=blue `1494824059966324897`): 我是 **consumer 不是 harness dev**;**main(`1489941073735450704`)是我唯一 PoC**,op-gen 问题只找 main. 报了 3 个 a5_ops bug → tilelang task#28(is_fa_class)+ main task#29(ref_preflight npu-id)+ task#30(manifest input_stats). team 修完 push gitcode 我 pull 重跑 hc_split_sinkhorn op-gen. (memory `reference_a5ops_authors_discord_group.md`)
- **新铁律**: ① 绝不用 console-only 问答(AskUserQuestion/picker)——user 只看 Discord(memory `feedback_no_console_only_questions.md`)② a5_ops 更新随时 pull(KB 更新)
- **a5_ops op-gen 环境**: build 容器 `a5ops-a3` on davinci2(易容 A3 host),`.ascendc_env` TARGET=a3 → SSH alias `easyr1-a3`. hc_split_sinkhorn workspace ref_preflight=RUNNABLE,卡在 FA-误判 router(等 task#28).
- **discord 插件已 patch**: `~/.claude/plugins/cache/.../discord/0.0.4/server.ts` 让 allowBots 生效 + 杀孤儿 bun 进程(bot↔bot 通信修复). plugin 升级会 revert,需重打(memory 有 recipe).
- **easyr1-npu repo HEAD**: `789b891`(upstream issue draft). a5_ops local main 落后 origin ~39 commits,op-gen 前先 pull.

## 🎉 PASS

```
[v4-min] sgl 0.5.12.post2.dev434+gb13d3d18c
[v4-min] Engine init OK in 27.7s
[v4-min] generate done in 0.9s
[v4-min] output: [{'text': '醺报废', 'output_ids': [122081, 112435], ...}]
```

`DeepseekV4ForCausalLM` 真 V4 model class 在 Ascend A3 NPU bf16 跑通 `llm.generate()`。Shape-correct,非 numerical correct(1-layer reduced fab + 14 PoC workarounds — REPORT §0.5 + `workspace/v4_attempt_2026_06_01/README.md`)。

Commit: `fc2f486 V4 PoC PASS: DeepseekV4ForCausalLM generate() returns on Ascend A3 NPU bf16`,已 push 到 `origin/main`。

## 🎉 PASS #2 — V4 RL LOOP CLOSED (2026-06-01)

rollout → weight-update → re-rollout 全闭环,5/5 步 weight-sync 都改变 rollout 输出。
- 绕 #26794:`Engine.update_weights_from_tensor`(只推 attention 5 个权重,不碰 MoE experts)
- weight delta 是 seeded synthetic 占位(miles V4 训练侧算子未移植);plumbing 已 prove
- Commit `d65c219`,push 到 origin/main
- Artifact:`workspace/v4_attempt_2026_06_01/v4_RL_LOOP_PASS_log_2026_06_01.txt` + `_v4_rl_loop_tensor_PASS.py`

## 进行中(in-flight,session 末状态)

1. **hc_split_sinkhorn AscendC op-gen** — 用 a5_ops `/ascendc-op-gen` 生成唯一需要的 V4 vector 算子(native NPU 无对应)。已 deploy a5_ops skills(`bash src/deploy.sh` global),配 preflight(A3 target → `easyr1-a3` SSH alias = 115.190.166.102:443,build 容器 `a5ops-a3` on davinci2)。修了 a5_ops orchestrator gap:`_run_ref_preflight_bootstrap` 没传 lane→`--npu-id`(默认 1,容器只有 davinci2=index 0 → aclInit 107001)。fix 在 `a5_ops/src/scripts/orchestrator/orchestrator.py`(working tree,**未 commit 到 a5_ops main** — 那是 user 项目,作者群 + main agent 协调)。op-gen 重跑中(invocation #3),清了 stale ref_runnable.json + pyc。
2. **native NPU op 替换**(task #310,质量提升,非阻塞)— RL loop 已用 torch fallback PASS;native 替换路径 verified(`npu_clipped_swiglu` / `npu_apply_rotary_pos_emb` / `npu_kv_rmsnorm_rope_cache_v2`),记在 README 末。

## a5_ops 作者 Discord 群(新)

group `1501649396922712105`,我的 alias **"blue"**(bot id 1494824059966324897,role=平台端到端移植到NPU+用a5_ops)。main agent role id `1489941073735450704`。等 main agent @ 我再自我介绍。详见 `memory/reference_a5ops_authors_discord_group.md`。

下面历史记录保留,因为它含有完整的 attempt 链 + 调试方法,对后续上游 PR 工作有用。
---



## 工作约束(user-stated,permanent)

1. **不假设时区/不假设 today/tomorrow**。任何 "等用户上线 / 明天再做" 类逻辑禁止。
2. **token 不耗尽就继续**。失败 / 卡住 / 需要确认时,自己向前推一步,不停下等用户。
3. **没有 milestone PASS,就如实写没有,绝不偷换概念**。先前 V3.2 替换事件造成严重信任损失,memory `deception_under_closure_pressure_2026_06_01.md` 永久生效。
4. **V4 真路径不能换模型**(2026-06-01 03:21 user 明确:「不能再出现换模型的情况了」),减层 PoC 必须仍是 `DeepseekV4ForCausalLM` 真 V4 schema。

## 当前位置(session 暂存状态)

### Repo head
- `0e5de06` — V4 attempt narrowed to IPC dispatch contradiction(已 push 到 `origin/main`)

### 子项目目录
- `output/miles-dsv4-flash-poc/` — V3.2 PoC sub-project,顶部有 §0 Disclosure 标明 V3.2 替换事件
- `workspace/v4_attempt_2026_06_01/` — 本 V4 attempt 全部 artifact + 完整 README(挂着多个 update 段,最新在文末)
- `workspace/T32_tilelang_rescue/v4_real_truth/` — sglang 上游 V4 真 source ground truth(deepseek_v4.py 2259 行 + DeepSeekV4Config + HF v4_real_config.json)

### A3 实测环境(sgl_probe 容器 on chip 1)
- Image: `lmsysorg/sglang:main-cann8.5.0-a3`
- sglang: `0.5.12.post2.dev434+gb13d3d18c`
- Fab ckpt: `/host-models/dsv4_REAL_1layer_fab/` (`DeepseekV4ForCausalLM`, MoE active, compress_ratios=[4], 1.3B params, sliding_window=256, quantization_config=None)
- 已 patched sglang 源(注意:容器内的 sglang 源已被 trace 修改,有 `[SCHED]` `[SCHED-LOOP]` `[TM]` `[TM-B]` `[RR]` `[TRACE]` prints。)
- 备份:`/tmp/_sched_bak.py`, `/tmp/_tm_bak.py`, `/tmp/_dsv4_bak.py`
- HBM 用量:~2.5 GB / 64 GB,健康
- watchdog `/tmp/_hbm_watchdog.sh` 在 background(PID 1029257 或后续),55 GB cap

### 上一次 attempt 卡在哪
**`recv_requests()` 矛盾**(见 `workspace/v4_attempt_2026_06_01/README.md` 末尾):
- TM 端 `_send_batch_request` 完成 ✓
- scheduler 端 `request_receiver._pull_raw_reqs` 返回 1 item ✓([RR] trace)
- BUT 同一个 receiver,在 `scheduler.event_loop_overlap` while-loop 里调用 `recv_requests()` 永远返回 `truthy=False` 空 list ✗([SCHED-LOOP] trace)

## 接手该做什么(in priority order)

### Step 1:解释 `[RR]` 和 `[SCHED-LOOP]` 的矛盾
两个候选假设要验证:

**假设 A:`[RR]` 出现是 sticky 一次 health check,后面所有 [SCHED-LOOP] 都是真实 main loop 的空 recv。**
- 验证方法:在 `request_receiver.recv_requests` 入口加 `print('[RR] called from', id(self), threading.current_thread().name)`
- 同时在 `event_loop_overlap` 的 `recv_requests()` 调用前加 `print('[SCHED-LOOP] about to call recv_requests on', id(self.request_receiver))`
- 对比两个 id 是否相同。如果同一 `id`,那必然是同一函数被同一 receiver 调用,矛盾。如果不同,说明有 2 个 receiver。

**假设 B:scheduler 的 recv 是先 broadcast,然后 attn_tp_rank != 0 收到 None,被过滤成空。**
- 看 `_broadcast_reqs_across_ranks`:`broadcast_pyobj` 在 `attn_tp_size > 1` 时把请求复制给所有 attn ranks。tp_size=1 应该跳过。但可能有 NPU-specific path。
- 检查 `_broadcast_reqs_across_ranks` 全文,加 print 哪些 rank 看到了请求。

### Step 2:如果 Step 1 不能立刻解决
- 写一个 vllm-ascend V4 path 的替代 attempt。vllm-ascend `deepseek_v4_fp8` validator 是单一 `raise ValueError`(`vllm/platforms/interface.py:check_quantization_supported`),可以 monkey-patch 它的 `supported_quantization` 加 `"deepseek_v4_fp8"`,然后让 vllm-ascend V4 model class(native NPU ops:`torch_npu.npu_rotary_mul`、`torch.ops._C_ascend.npu_hc_pre/post`)接管。
- A3 上 `tlrescue` 容器已经 pull 了 community vllm main + vllm-ascend main(`/vllm` 和 `/vllm-ascend` editable installs),`vllm_ascend.models.deepseek_v4` import OK。卡点是 fp8 validator + GDN drift(后者已 try/except 绕过)。

### Step 3:如果两条都跑不通 / 时间深入太长
- 写完整 upstream issue 到 sgl-project/sglang:V4 + device=npu + bf16 路径 generate hangs at IPC,Engine init + KV pool 全 OK。
- 也写 vllm-ascend 一个:V4 不支持 bf16,需要 `dequantize-fp8-to-bf16` 真路径 + 暴露 `weight_scale_inv` 公开 API 给 fab 工具用。

## 必读 memory(每次接手 / compact 后)

`/home/z00637938/.claude/projects/-home-z00637938-workspace-easyr1-npu/memory/MEMORY.md` 完整索引。重点条目:
- `deception_under_closure_pressure_2026_06_01.md` — 永久 anti-pattern,closure pressure → invent fact → harm user trust
- `verify_architecture_class_against_huggingface_truth.md` — arch class 选择必须 verify 真 HF config
- `pr_polling_must_check_reviews_field.md` — PR polling 必须看 reviews[]
- `check_ci_on_my_own_push_before_declaring_done.md` — push 完必须等 CI
- `a3_chip_economy.md` — A3 是共享机器,每次操作前 precheck npu-smi
- `end_to_end_vs_described.md` — 不假装 "described == done"
- `no_user_schedule_references.md` — **绝不**说「等你睡觉前 / tomorrow」类话

## 接手时不要做的事

- 不要 declare "V4 NPU PoC PASS" 除非 `llm.generate(["Hi"], max_new_tokens=2)` 真返回非空字符串
- 不要修改 PROJECT.json 的 status 为 complete
- 不要重新写 V3.2 fab 假装是 V4
- 不要假设时区 / today / tomorrow
- 不要 stop polling 或 stop attempt 等用户回复
- 不要把 sglang 容器内的 trace 改动 commit 到 `easyr1-npu/main`(那些是 sglang 源码)— 只 commit 我们自己的 fab + driver

## 接手时该做的事

1. 立刻 `git log --oneline -5` 看上次 push
2. 读 `workspace/v4_attempt_2026_06_01/README.md` 末尾,确定上次卡在哪
3. 按 "Step 1 → Step 2 → Step 3" 顺序推进
4. 每个里程碑 / 每次 commit / 每次 push 必须 Discord 报告(chat_id `1494825170399924366`)
5. 维护本 handover 文件 — 推进了就更新「当前位置」段

## 紧急联系

- A3 host: `ssh -p 443 root@115.190.166.102`
- sgl_probe 容器仍在 chip 1,patched sglang 源仍在内
- Discord chat: `1494825170399924366`
