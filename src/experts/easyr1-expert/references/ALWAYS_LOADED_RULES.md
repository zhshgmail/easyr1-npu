# EasyR1-port 必读规则（无条件加载）

> **Worker 必须在 Phase A 开始写任何代码前读完此文件。**
>
> 这里是**跨 port 通用的 process rule 和 universal trap** —— 不是 op-specific 技术细节。
> 具体 pattern（device dispatch、Ray shim、attention backend 等）在 `patterns/domains/*.md`，按需加载。
>
> **为什么这些要无条件**：它们都是"如果不在下笔前知道，后面再查也晚了"的规则。

---

## 1. Meta 规则（防止幻觉式错误）

### OL-01: NPU 代码改完必过 py_compile，禁止"看起来没问题就提交"

**前置步骤**：写任何代码修改后，在提交/标记 Phase C 完成前：

```bash
bash src/experts/easyr1-expert/scripts/static_check.py --files <edited-files> --import-package verl
```

必须 exit 0。exit 1/2 = Phase C 不通过，必须修。

反例（实锤 2026-04-22 round 2）：agent 把 `from ..utils.device import ...` 插到 `from ..utils.torch_functional import (` 的裸括号里，人眼扫代码不会发现（看起来都是 `from ... import`），但 py_compile 立刻 SyntaxError。

**为什么无条件**：agent 幻觉式错误特征是"它不知道自己错了"。py_compile 是机械的，不信任人眼扫。

### OL-02: 声称 smoke PASS 必须有 log 文件路径 + entropy_loss 数值

**前置步骤**：任何 PROGRESS.md / RESULTS.md 里写 "V1.4 PASS" 之前：

1. 必须在同一文档引用具体 log 文件路径 `/tmp/${NPU_USER}/easyr1-logs/<tag>.log`
2. 该 log 文件必须存在
3. 必须 grep 到 `entropy_loss:` 行，数值在 `references/SMOKE_BASELINE.md` 给定的 baseline band 内

反例（实锤 2026-04-21..22，NPU-OPS-010）：prior session 把手工跑 V1.4 成功说成 "P1 end-to-end closed"，没用这条规则兜底就广告给客户。

**为什么无条件**：G3 invariant。Stop hook 会拦截不符合规则的 claim，但 worker 自己一开始就按这条走更干净。

### OL-03: 不得读 denylist 里的历史文档（诚实 cold-drive 前提）

在 agent context 里**禁读**：

- `docs/_meta/HANDOVER.md`（§6 有过去具体命令）
- `docs/easyr1/porting-journal.md`（手工工作日志）
- `docs/transformers/transformers-upgrade-drill.md`
- `docs/transformers/UPGRADE-DRILL-STATUS.md`
- `docs/_archive/P2-WORKFLOW.md`
- `docs/_archive/skill-dry-run-2026-04-20.md`
- `docs/easyr1/DELIVERABLE.md`
- `docs/_archive/codex-*.md`
- `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md`（会偏向"参考过去经验"思维）
- `docs/_meta/design.md` / `docs/easyr1/dep-matrix.md` / `docs/easyr1/PORT-SUMMARY.md`（历史版本）
- `docs/easyr1/easyr1-dep-chain-audit.md`（答案 spoiler）
- `docs/_archive/handoff-2026-04-19.md`
- **zhshgmail/EasyR1 的 `ascend-port*` 分支的任何内容（git log、git show、grep、甚至 `git diff main..ascend-port`）**

读了这些 = 作弊，round 被作废。

**为什么无条件**：agent 读到就不是 cold-drive 了。看似"参考"等同于抄答案。

### OL-04: 使用专家镜像 tag，不要复用别人/过去的 image

**前置步骤**：每次 build 产出的 image 必须用 session-specific tag，例如 `easyr1-npu:round3-{timestamp}`。禁止：

- `easyr1-npu:ascend-port`（过去手工 build）
- `easyr1-npu:ascend-port-e2e`（round 1 agent 复用）
- `easyr1-npu:round2`（round 2 agent build，可能有 SyntaxError）
- 任何已存在的 tag

使用已存在 tag = 走捷径 = 绕开了你应该自己 build 出来的代码路径。

**例外**：user 通过 `--reuse-image <TAG>` 明确提供预构建 image 时，skip build，直接用 user 给的 tag。此时 Phase C 仅验证 image 存在（`docker image inspect <TAG>`）+ 记录 image_id 到 PROGRESS.md（标注 `produced_by: user-provided`，不计入 cold-drive build 证据）。

### OL-04b: 每次 session 跑完要清理临时 image/container

**规则**：session 结束前（无论 PASS/FAIL/stuck），清理本 session 创建的临时产物。A3 磁盘是共享的。

**清理对象**：
- `easyr1-npu:{SESSION_TAG}` 和它的 iter 变种（`easyr1-npu:{SESSION_TAG}-iter1` 等）
- 本 session 启动的容器（通过 `run-npu-container.sh` 创建的，名称模式 `easyr1-*-{SESSION_TAG}-*` 或时间戳匹配）
- `/tmp/round-deploy.bundle` 等 scp 中转产物

**不清理**：
- user 提供的 image（OL-04 例外路径）
- 基础镜像 `quay.io/ascend/verl:*`
- 其他 session 留下的 image/container（不是你的就不要删）

**清理方式**（Stop hook 或 `@review-fail` 前）：
```bash
# 删除本 session tag 的所有 image
ssh -p 443 root@$A3_HOST "docker images --format '{{.Repository}}:{{.Tag}}' | grep -E '^easyr1-npu:${SESSION_TAG}' | xargs -r docker rmi -f"

# 停+删本 session 启动的容器
ssh -p 443 root@$A3_HOST "docker ps -a --format '{{.Names}}' | grep ${SESSION_TAG} | xargs -r docker rm -f"

# 临时 bundle
ssh -p 443 root@$A3_HOST "rm -f /tmp/round-deploy.bundle"
```

**不要**：`docker system prune` / `docker image prune` 这类泛清理——会删掉别的 session 或 user 持有的资源。只删本 session 打了 tag 的。

Worker 退出前必须在 PROGRESS.md 记录 `Cleanup: {clean|partial|skipped <reason>}`。

### OL-05: A3 是共享 host，任何 A3 动作前都先看 chip 再抢

### OL-05: A3 是共享 host，任何 A3 动作前都先看 chip 再抢

**前置步骤**：**任何**要占 NPU 的 A3 动作之前（smoke、docker run 挂 davinci、ray start 等）**必须**：

```bash
ssh -p 443 root@$A3_HOST "npu-smi info | head -30"
# 以及针对打算使用的 npu：
ssh -p 443 root@$A3_HOST 'for i in 0 1 2 3; do npu-smi info -t proc-mem -i $i 2>&1 | grep -E "Process id|Chip ID"; done'
```

确认打算用的 chips AICore=0% + HBM 接近静态（2-3GB）+ 没有 `Process id:` 行。发现非 idle → 停手，不抢，报给 user。

**不是仅限 smoke**：也包括 `deploy_to_a3.sh` 触发 docker build（build 不占 chip 但验证阶段会）、`run-npu-container.sh`（启动即挂 chip）、任何 `--device=/dev/davinciN` 的 docker run。

反例：2026-04-20..21 连续几天踩过"别人跑着的 Ray raylet 锁了 UDA namespace，我们容器启动失败"的坑（NPU-OPS-009）。

### OL-05b: 用最少数量的 NPU

**规则**：选 chip 数量时用"跑得通的最小"，不是"刚好能用的最大"。

- **V1.1 device smoke**：1 chip 够了（`--chips 0`）。不要默认 2 chip。
- **V1.3 rollout smoke**：1 chip 够了（单 prompt rollout）。不要默认 2 chip。
- **V1.4 2-chip GRPO**：2 chip 是这一步的 baseline 定义（DP=2），不能再减。
- **V1.5/V2.2 4-chip**：Stage 0 **默认跳过**。只在 user 明确要求时跑。
- **V2.1 padding_free**：2 chip（和 V1.4 同），Stage 0 可选。

共享 host 上占 4 chip 跑非必要测试 = 侵占他人配额。默认只跑必须 rung（V1.1+V1.3+V1.4），共占 2 chip 峰值。

Worker 在 Phase D 入口必须在 PROGRESS.md 记录 `NPU-usage-plan: <rung → chip-count>` 给 orchestrator 审核。

### OL-06: A3 github 不通（GFW），用 scp / bundle 同步代码

**关键事实**：A3 host 的 `git clone https://github.com/...` **会 timeout 2+ 分钟**。已经 ping 通不代表 github 可达。

**做法**：
- Agent 产出代码在 x86 host，本地 commit + push 到 github fork
- A3 端不 clone，而是 `scp` 代码 tar 或 `git bundle` → A3 → `git fetch bundle` 接入 local checkout

反例：2026-04-22 spawn agent 试图 A3 上直接 `git clone` 消耗 2 分钟才发现超时。

### OL-07: Huaweicloud pypi mirror 不稳，优先 aliyun + 加 timeout

**Dockerfile 里**：`pip install triton-ascend` 不要依赖 huaweicloud mirror（至少要有 timeout + fallback）。

反例：2026-04-22 round 2 Dockerfile build 卡在 `pip install triton-ascend==3.2.0`（走 huaweicloud）超过 50 min 无响应，最终手工 kill。

**规则**：
- Dockerfile 第一 pip install 必须 `--default-timeout=60` + `|| 使用 aliyun fallback`
- 见 `patterns/dockerfile.md` 模板

### OL-08: 不擅自 Edit 非自己 agent scope 的代码

Stage 0：`easyr1-port-worker` 可 Edit 的范围：
- ✅ `upstream/EasyR1/verl/**/*.py`（port 工作）
- ✅ `upstream/EasyR1/Dockerfile.npu*`（Dockerfile）
- ✅ `upstream/EasyR1/examples/qwen2_0_5b_math_grpo_npu_smoke*.sh`（smoke）
- ✅ `upstream/EasyR1/requirements*.txt`
- ✅ 自己的 `workspace/easyr1-port-{ts}/` 目录
- ❌ `src/experts/easyr1-expert/**`（expert 本身不改）
- ❌ `docs/` / `knowledge/`（历史记录）
- ❌ 其他 host / 其他用户的目录

PreToolUse hook 会拦截违规 Edit。

### OL-09: 所有 claim 带 provenance

Artifact 里每个 "PASS / produced / 验证" 类断言必须有 `produced_by:` 字段。合法值：
- `easyr1-port-worker` (agent 产)
- `orchestrator` (orchestrator skill 产)
- `human-intervention` (人手工做的，**Stage 0 CLOSED 不允许**)

不标 provenance = Stop hook reject。

### OL-10: 失败 ≠ 失败，要按 ERROR_CORRECTIONS 分类

Phase D smoke fail 时，**先 grep `references/ERROR_CORRECTIONS.md`** 把 traceback 分类到 EC-NNN。对应 EC 有 root cause + fix。**不要凭直觉改**。

如果没找到 EC 匹配，先读 patterns/domains/ 里相关的 .md，再不行记一个 new finding 到 PROGRESS.md 的 "unclassified failures" 段，报告给用户。

---

## 2. Phase 加载顺序

1. **Phase A 开始**：读本文（ALWAYS_LOADED）+ `KB_INDEX.md`
2. **Phase B 前**：根据要做的事，按需加载 `patterns/domains/<relevant>.md`
3. **Phase C 前**：加载 `references/dockerfile.md`（在 `patterns/domains/` 下）
4. **Phase D fail 时**：defer-load `ERROR_CORRECTIONS.md`，grep 错误匹配 EC-NNN
5. **任何时候**：可 grep `PLATFORM_BUGS.md` 查已知平台 bug（NPU-BUG-001..004）

---

## 3. 退出协议

- 所有 rung (V1.1/V1.3/V1.4) PASS → exit "done"
- 同 signature 卡死 ≥3 次 → exit "@smoke-probe" (Stage 1+ 才有 probe；Stage 0 视为 FAILED)
- 任何 OL 被违反（尤其 OL-03 denylist 读取）→ exit "@review-fail"

**退出前必须**：签名 PROGRESS.md（写 `Worker signed: easyr1-port-worker <timestamp>`），Stop hook 会查这行。

---

## 4. 关键文件路径参考

- KB：`src/experts/easyr1-expert/references/`
- Scripts：`src/experts/easyr1-expert/scripts/`
- Workspace：`workspace/easyr1-port-{timestamp}/`（session 临时）
- Port branch：`upstream/EasyR1/ ascend-port-{session-tag}`
- A3 smoke logs：`/tmp/${NPU_USER}/easyr1-logs/<tag>.log`
