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

- `docs/HANDOVER.md`（§6 有过去具体命令）
- `docs/porting-journal.md`（手工工作日志）
- `docs/transformers-upgrade-drill.md`
- `docs/UPGRADE-DRILL-STATUS.md`
- `docs/P2-WORKFLOW.md`
- `docs/skill-dry-run-2026-04-20.md`
- `docs/DELIVERABLE.md`
- `docs/codex-*.md`
- `docs/design/SKILLS_ARCH_TARGET.md`（会偏向"参考过去经验"思维）
- `docs/design.md` / `docs/dep-matrix.md` / `docs/PORT-SUMMARY.md`（历史版本）
- `docs/easyr1-dep-chain-audit.md`（答案 spoiler）
- `docs/handoff-2026-04-19.md`
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

### OL-05: A3 是共享 host，先看 chip 再抢

**前置步骤**：跑 smoke 前 **必须**：

```bash
ssh -p 443 root@$A3_HOST "npu-smi info | head -20"
```

确认打算用的 chips AICore=0% + HBM 很低。不抢别人正在用的 chip（`Process id: xxx` 不为空的行）。

反例：2026-04-20..21 连续几天踩过"别人跑着的 Ray raylet 锁了 UDA namespace，我们容器启动失败"的坑（NPU-OPS-009）。

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
