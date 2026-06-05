# Postmortem: 为什么我突然失去对项目的正确认知（2026-06-02 "推理是假的" 误判）

## 缺失的信息(根因)

这个 session 从 auto-compact 摘要起步。摘要**详细保留了最近活跃的那条线**(tilelang/bf16/算子验证的逐步细节),但**丢掉了"V4 各部分到底是怎么被跑通的"这层知识**,具体丢的是:

1. **工作配方(working recipe)**:sglang V4 推理跑通靠的是 **torch-fallback —— 关掉所有 tilelang、monkey-patch 走纯 torch**(`_sglang_v4_minimal_PASS.py`),**不是编译 tilelang**。这个"哪条路才是 work 的"完全没进摘要。
2. **架构决策 + 理由**:tilelang 在推理/训练两侧都是**可选优化**,不是必需;训练侧用 CANN-native 替换,推理侧用 torch fallback。这个"为什么不用 tilelang"的判别没进摘要。
3. **verified-vs-unverified 的证据指针**:哪些是有 log 的真结果、哪些是转述旧 log。

**结果链**:摘要给了我"tilelang 是核心、正在死磕版本"的画面,但抹掉了"tilelang 其实可选、torch-fallback 才是 work 的路"。于是我:
- 接着去**编译 tilelang**(撞 0.1.8-DSL-vs-fork 版本错配),
- 撞墙后**应激否定**,把真实跑通过的推理判成"假/blocked",
- 又因为之前**没把工作配方记成可复现脚本/笔记**,复现时 re-derive 走错路,坐实了"看起来在说谎"。

**一句话**:compaction 保留了"症状级的最近战术细节",丢了"决策级 + 配方级 + 证据级"的知识。后者才是重现/接手的关键。

## 两个修复(owner 指令)

### 1) CLAUDE.md 指导 auto-compact 保留缺失信息类型 —— 已改
在 `## Compact Instructions` 加了 **"Guidance to the auto-compact SUMMARIZER"**:必须 verbatim 保留 ① 工作配方(脚本+env+容器+model)② 架构决策+理由(哪条 work、哪条被否及为何)③ verified-vs-unverified + 证据指针 ④ 用户的诚信纠正。空间不够时,**先砍战术流水账,绝不砍这四类**。
并在 writer-side 表新增两行:"A result was made to RUN/PASS → `RESULT_<slug>_<date>.md`(配方+原始输出+为何选这条+limits)+ `_runlogs/` 时间戳日志";"key 架构决策+理由 → `project_*.md` 带判别器"。

### 2) 该做什么笔记,让后人/后 session 不会重蹈、不会以为我在说谎 —— 格式已定 + 已示范
**每个"跑通/PASS"的结果,落一份 `RESULT_<slug>_<date>.md`,含四件套:**
1. **可复现配方**:能直接 run 的脚本路径 + 让它 work 的确切 env/flag/monkey-patch + 容器 + model 路径。(不是散文描述,是能照着跑的。)
2. **捕获的原始输出**:那行证据(`output_ids=[...]` / `grad_norm=...`),不是"PASS"二字。
3. **为何这条路、不是另一条**:判别器(如"torch-fallback,不是 tilelang-compile,因为 sglang V4 对每个 tilelang kernel 都有 torch fallback")。
4. **verified-vs-unverified + limits**:本次亲验?限定(减层/占位 loss/随机权重)?
**外加**:每次 run 写时间戳日志到 `_runlogs/`。报结果必附 log 路径。
**已示范**:`SGLANG_V4_INFERENCE_PASS_2026-06-02.md`(推理配方+原始 output_ids+为何 torch-fallback+limits)。

## 为什么这能防"以为我说谎"
人类/后 session 复现时,照着 RESULT 文件的**可复现配方**跑,会拿到**记录里的同一原始输出**(如同样的 output_ids、同量级 grad_norm)——证据自洽,不需要相信我的话。配方缺失(本次)才是"复现不出 → 怀疑造假"的根。记忆教训:[[feedback_every_run_must_have_timestamped_log]]、[[deception_under_closure_pressure_2026_06_01]]。
