# 通用 NPU-port 无条件规则（cross-expert）

> **任何 NPU 上游移植 expert 的 worker 必须在 Phase A 写任何代码前读完本文。**
>
> 本文是 **cross-expert 通用** 的 process rule 和 universal trap。每个具体的 expert
> 还有自己的 `ALWAYS_LOADED_RULES.md`，加自己的 **OL-03 denylist** 和 **OL-08
> 可编辑路径白名单**，其余规则引用本文即可。
>
> **为什么要无条件**：这些都是"下笔之前不知道，事后查就晚了"的规则；并且是
> Stop/PreToolUse hook 会机械校验的最小集。

---

## 1. 通用 meta 规则

### OL-01: 代码改完必过 py_compile，禁止"看起来没问题就提交"

写任何代码修改后，Phase C 完成前：

```bash
bash <expert>/scripts/static_check.py --files <edited-files> --import-package <caller-package>
```

必须 exit 0。exit 1/2 = Phase C 未通过，必须修。

**反例**（实锤 2026-04-22 round 2 easyr1）：agent 把 `from ..utils.device import ...`
插到 `from ..utils.torch_functional import (` 的裸括号里，肉眼扫不出来，
py_compile 立即 SyntaxError。

**为什么无条件**：agent 幻觉式错误的特征就是"它不知道自己错了"。py_compile
是机械的，不信任人眼扫。

### OL-02: 声称 PASS 必须有 log 文件路径 + 关键数值

任何 PROGRESS.md / RESULTS.md 里写 PASS / green / closed / end-to-end 之前：

1. 必须在同一文档引用具体 log 文件路径
2. 该 log 文件必须存在
3. 必须引用到里面的关键数值（entropy_loss、entropy_loss 在 band 内、或其它
   expert 自定义的断言指标）

**反例**（NPU-OPS-010，实锤 2026-04-21）：prior session 把手工跑 V1.4 成功
说成 "P1 end-to-end closed"，没有数值兜底就广告给客户。

**为什么无条件**：G3 invariant。Stop hook 会拦截不符合的 claim；worker 自己
一开始就按规则走更干净。

### OL-04: 使用 session-specific image tag，不要复用别人/过去的 image

每次 build 产出的 image 必须用 session-specific tag，例如
`<expert-image-prefix>:<SESSION_TAG>`。禁止复用任何已存在的 tag（过去手工
build 的、别的 round agent 留下的、旧 drill 留下的）。

使用已存在 tag = 走捷径 = 绕过了你应该自己 build 出来的代码路径。

**例外**：user 通过 `--reuse-image <TAG>` 明确提供预构建 image 时，skip build；
此时 Phase C 仅校验 image 存在（`docker image inspect <TAG>`），provenance 记
`user-provided`（不计入 cold-drive build 证据）。

### OL-04b: Session 结束前必须清理临时 image/container

Session 退出前（无论 PASS/FAIL/stuck），必须清理本 session 创建的临时产物。
A3 磁盘是共享的。

**清理对象**：
- `<expert-image-prefix>:<SESSION_TAG>` 及其 iter 变种
- session tag 命名的容器
- `/tmp/round-deploy.bundle` 等 scp 中转产物

**不清理**：
- user 提供的 image（OL-04 例外路径）
- 基础镜像（`quay.io/ascend/verl:*` 等）
- 其他 session 留下的 image/container（不是你的别动）

**不要**：`docker system prune` / `docker image prune` 这类泛清理——会误伤
别的 session / user 持有的资源。

Worker 退出前必须在 PROGRESS.md 记 `Cleanup: {clean|partial|skipped <reason>}`。

### OL-05: A3 是共享 host，任何占 NPU 动作前都先 precheck

**任何**要占 NPU 的 A3 动作之前（smoke、docker run 挂 davinci、ray start、
`run-npu-container.sh` 等）**必须**：

```bash
ssh -p 443 root@$A3_HOST "npu-smi info | head -30"
ssh -p 443 root@$A3_HOST 'for i in 0 1 2 3; do npu-smi info -t proc-mem -i $i 2>&1 | grep -E "Process id|Chip ID"; done'
```

确认打算用的 chips AICore=0% + HBM 接近静态 + 没有 `Process id:` 行。
发现非 idle → 停手，不抢，报 user。

**反例**：2026-04-20..21 连续几天踩过"别人跑着的 Ray raylet 锁了 UDA namespace，
容器启动失败"的坑（NPU-OPS-009）。

### OL-05b: 用最少数量的 NPU

选 chip 数量时用"跑得通的最小"，不是"刚好能用的最大"。
共享 host 上占多 chip 跑非必要测试 = 侵占他人配额。
Expert 各自在 Phase D 入口 PROGRESS.md 记 `NPU-usage-plan: <rung/smoke → chip-count>`。

### OL-06: A3 github 不通（GFW），用 scp / bundle 同步代码

**关键事实**：A3 host 的 `git clone https://github.com/...` **会 timeout 2+ 分钟**。
ping 通不代表 github 可达。

**做法**：
- Agent 产出代码在 x86 host，本地 commit + push 到 github fork
- A3 端不 clone，而是 `scp` tar 或 `git bundle` → A3 → `git fetch bundle`
  接入 local checkout

**反例**：2026-04-22 spawn agent 试图 A3 上直接 `git clone` 消耗 2 分钟才发现超时。

### OL-07: Huaweicloud pypi mirror 不稳，优先 aliyun + 加 timeout

**Dockerfile 里**：`pip install` 不要依赖 huaweicloud mirror（至少要有 timeout +
fallback）。

**规则**：
- Dockerfile 第一 pip install 必须 `--default-timeout=60` + `|| 使用 aliyun fallback`
- 详见每个 expert 的 `patterns/domains/dockerfile.md`

**反例**：2026-04-22 round 2 Dockerfile build 卡在 `pip install triton-ascend==3.2.0`（走 huaweicloud）超过 50 min 无响应。

### OL-09: 所有 claim 带 provenance

Artifact 里每个 "PASS / produced / 验证" 类断言必须有 `produced_by:` 字段。
合法值：
- `<expert-name>-worker`（worker agent 产）
- `orchestrator`（orchestrator skill 产）
- `user-provided`（OL-04 例外路径：user 给的 image/artifact）
- `human-intervention`（人手工做的，**Stage 0 CLOSED 不允许**）

不标 provenance = Stop hook reject。

### OL-10: 失败 ≠ 乱改，要按 ERROR_CORRECTIONS 分类

Phase D 断言失败时，**先 grep expert 自己的 `references/ERROR_CORRECTIONS.md`**
把 traceback/症状分类到 EC-NNN。对应 EC 有 root cause + fix。**不要凭直觉改**。

若无 EC 匹配，先读 `patterns/domains/` 里相关的 .md，再不行记 new finding 到
PROGRESS.md 的 "unclassified failures" 段，`stuck` 或 `@review-fail` 退出。

---

## 2. Expert-specific 规则（每个 expert 在自己的 ALWAYS_LOADED_RULES.md 里补）

以下两条因为**随 expert 变化**，所以每个 expert 在自己的
`references/ALWAYS_LOADED_RULES.md` 里写本地版本，不在这里定义：

- **OL-03**: denylist（禁读文件列表；每个 expert 有不同的"答案"历史文档需要
  屏蔽，例如 easyr1-expert 屏蔽 `ascend-port*` 分支 git log，
  transformers-upgrade-expert 屏蔽 `ascend-port-transformers-upgrade` 分支）。
- **OL-08**: 可编辑路径白名单（每个 expert 的 agent 能 edit 什么路径；
  PreToolUse hook 会拦截违规 Edit）。

---

## 3. Phase 加载顺序（通用）

1. **Phase A 开始**：读本文 + expert 的 `ALWAYS_LOADED_RULES.md` + `KB_INDEX.md`
2. **Phase B 前**：按需加载 expert 的 `patterns/domains/<relevant>.md`
3. **Phase C 前**：加载 `patterns/domains/dockerfile.md`（若 expert 产 image）
4. **Phase D fail 时**：defer-load `ERROR_CORRECTIONS.md`
5. **任何时候**：可 grep `PLATFORM_BUGS.md`

---

## 4. 退出协议（通用）

- 全部断言 PASS → exit `done`
- 同 signature 卡死 ≥3 次 → exit `@upstream-probe` (Stage 1+) 或 `stuck` (Stage 0)
- 任何 OL 违反（尤其 OL-03 denylist 读取）→ exit `@review-fail`

**退出前必须**：在 PROGRESS.md 写 `Worker signed: <expert-name>-worker <timestamp>`。
Stop hook 会查这行。
