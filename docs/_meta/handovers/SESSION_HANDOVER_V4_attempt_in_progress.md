# SESSION_HANDOVER — V4 真路径在 NPU 上 attempt(in-progress,never stop)

> 本文件是 auto-compact 防丢失保险。任何时候 agent 被 compact 后接手,**先读这里**。
> 这次 session **不能 stop**,除非真正 token 用尽。user 已明确:不假设时区、不假设「今天/明天」、只要能前进就不停。

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
