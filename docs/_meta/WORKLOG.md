# WORKLOG — 当前 session 汇报入口

**用途**：user 2026-04-23T22:26Z 指令——"这么多工作，你分成一个工作序列从第一个任务开始做，所有任务都完成才算完成。你记录工作的文档先进行维持重构，把这些任务背景和状态放进去，用这个文件来汇报状态"。

本文件**必须**：
- 每完成一个 sub-task 立刻更新状态
- 每发现新问题立刻追加一行到未来任务
- auto-compact 之后下一 session 的第一入口就是读本文件
- user 想看状态就读这一份

## 2026-04-24T01:20Z 重新排序（user correction）

User 指正：我之前跳过了 vllm-ascend C 扩展针对 torch 2.11 ABI 的真正重编，靠 `VLLM_BATCH_INVARIANT=1` 回退路径是作为 vllm-ascend 开发者不该接受的做法。真正的 vllm 0.20 NPU port 必须从 **`.so` 针对 torch 2.11 真正重编** 开始。

## 核心任务（新序列）

### Phase 1 — vllm-ascend C 扩展针对 torch 2.11 ABI 重编

目标：让 `vllm_ascend_C.so` 真正用 torch 2.11 的 headers / libtorch 链接，让 `_TORCH_VERSION_BUILT_FOR="2.11"` 常量正确注入，让 `_torch_abi_safe_for_custom_ops()` guard 返回 True，让 native NPU custom ops 可以在 torch 2.11 下原生调用不 SIGSEGV。

- [ ] T1.1 进 iter 18 container，`nm -D vllm_ascend_C*.so | grep TORCH_VERSION` / `ldd` 看 `.so` 实际链接的 libtorch 版本
- [ ] T1.2 看 vllm-ascend source 里 `_TORCH_VERSION_BUILT_FOR` 常量是哪里设置的（CMakeLists 或 C++ source），确认它为什么没被写进 compiled `.so`
- [ ] T1.3 修改 source 让 `_TORCH_VERSION_BUILT_FOR` 真的被 export 成 module attribute，值来自 `TORCH_VERSION` macro 或 CMake 的 `Torch_VERSION`
- [ ] T1.4 在 Fix C 风格的 overlay container 里重编：`python3 setup.py build_ext --inplace` with CMakeLists widened to accept 2.11.x，检查出来的 `.so` 大小、`ldd` 链接版本、`_TORCH_VERSION_BUILT_FOR` 常量
- [ ] T1.5 用 `python3 -c "from vllm_ascend.utils import _torch_abi_safe_for_custom_ops; print(_torch_abi_safe_for_custom_ops())"` 验证 guard 返回 True
- [ ] T1.6 跑 native custom op reproducer（`torch.ops._C_ascend.npu_add_rms_norm_bias(...)`）确认不 segfault + 返回值数值合理

### Phase 2 — 在真 torch 2.11 native custom ops 上验证 vllm 0.20 推理

- [ ] T2.1 build iter 19 overlay：Phase 1 的 `.so` + 之前的 14 个 vllm 0.20 py-level drift patch
- [ ] T2.2 不设 `VLLM_BATCH_INVARIANT`（guard 应返回 True 走 native path）
- [ ] T2.3 跑推理对照（token diff vs baseline，prompt + greedy）— 确认 native path 下 token 也对
- [ ] T2.4 如果不对 → 说明 14 个 py patch 其中有假设 batch-invariant fallback 的 bug，要分别查
- [ ] T2.5 推理对照过了才算"vllm 0.20 推理真的跑通"

### Phase 3 — 查 EasyR1 侧 vllm 0.20 API 适配

- [ ] T3.1 看 `upstream/EasyR1/verl/workers/rollout/vllm_rollout/` 下所有文件
- [ ] T3.2 对比 vllm 0.18 vs 0.20 的 LLM 类 / SamplingParams / RequestOutput / logprobs 返回格式 breaking change
- [ ] T3.3 找出 EasyR1 这侧需要跟着改的地方（可能有：构造参数、return 解析、logprobs 字段名、log_prob vs cumulative_logprob 等）
- [ ] T3.4 patch EasyR1 到 vllm 0.20 兼容
- [ ] T3.5 commit 到 zhshgmail/EasyR1 的新 branch（ascend-port-vllm0200）

### Phase 4 — 训练场景端到端验证

- [ ] T4.1 训练测试：Qwen2-0.5B + math12k GRPO 2 步，sampling params 按实际训练（temperature=0.6, top_p=0.95, multi-sample）
- [ ] T4.2 比较 entropy_loss：目标落入 [1.21, 1.34]
- [ ] T4.3 如果还偏离，instrument rollout logprobs vs actor logprobs 对比找 root cause

### Phase 5 — 之前未完的结构工作（降级为低优先级）

- [ ] T5.1 确认 src/ 结构重组（commit 45ae36b 已 push，验证 install-skills 能用）
- [ ] T5.2 确认 docs/ 重组（commit a373612 已 push）
- [ ] T5.3 workspace/ 本地重组（commit c9bf1a6 已 push）
- [ ] T5.4 GLOSSARY 更新 day0/upgrade deprecated（commit 7595145 已 push）

## 现状（2026-04-24T01:20Z 本次更新时）

**推理能跑对吗（8 token greedy + 单 prompt）**：是，iter 18 image 输出和 baseline 一字不差。
**推理在训练实际使用的参数下能跑对吗（temperature=0.6, batch, long prompt, multi-sample）**：没验证过。
**训练能跑完吗**：能。训练 2 步 + checkpoint 成功。
**训练数字对吗**：不对。entropy_loss = 3.213 vs baseline 1.275，差 2.5 倍。不在可接受 band 内。

**根因的本质**（2026-04-24 user 指正后的重新理解）：
- vllm-ascend 的 `vllm_ascend_C.so` 是按 torch 2.9 ABI 编的
- 我用 Fix C overlay 让它"能装上"，但 `_TORCH_VERSION_BUILT_FOR` 常量没正确注入，`.so` 也可能根本没用 torch 2.11 的 headers 重链接
- `_torch_abi_safe_for_custom_ops()` guard 永远返回 False → `_CUSTOM_OP_ENABLED=False` → 所有 custom op（包括训练 backward 涉及的 linear）全部走 batch-invariant Python fallback
- 这个 fallback 路径对推理够用，对训练不够：actor forward 过程中的 RMSNorm / linear / attention 数值精度和 native NPU path 不一致，优化器更新后 actor 的分布和 rollout 脱钩 → entropy_loss 爆

**正确顺序**：先修 `.so`（Phase 1）→ 验证推理在 native path 仍对（Phase 2）→ 修 EasyR1 API 适配（Phase 3）→ 训练验证（Phase 4）。

## 历史 commit（本 session 已落）

| Commit | Summary |
|---|---|
| `7595145` | Phase 4 (旧)：GLOSSARY removes day0/upgrade binary distinction |
| `c9bf1a6` | Phase 3 (旧)：rewrite workspace/ path refs to per-upstream layout |
| `45ae36b` | Phase 2 (旧)：restructure src/ into {skills,scripts}/ per-upstream |
| `a373612` | Phase 1 (旧)：restructure docs/ into per-upstream subdirs + _meta + _archive |
| `01c65b8` | docs(worklog): create docs/_meta/WORKLOG.md |
| `42b8266` | docs: add MODULE-PORT-STATUS.md |
| `cc5ec81` | docs: add GLOSSARY.md |
| `9d2bb81` | docs(vllm-day0): mark vllm 0.20 drift resolved at iter 15 |

## vllm-ascend trace-branch commit（zhshgmail/vllm-ascend）

| Commit | Summary |
|---|---|
| `0ab16321` | [BugFix] vllm 0.20: patched bind_kv_cache must assign single tensor |
| `ca384b09` | [BugFix] vllm 0.20: register kv_cache as stacked tensor |
| `3665da07` | [BugFix] vllm 0.20: override forward_includes_kv_cache_update=False + add do_kv_cache_update |
| `910b2f3d`..`149393a0` | iter 1-15 vllm 0.20 drift patches |
| `87b507ed`, `caa55fed`, `7c2078e7`, `ab26a534` | torch 2.11 Fix B+/C 原 commits |
