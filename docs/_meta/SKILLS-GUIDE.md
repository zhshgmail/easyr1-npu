# Skills 使用手册 — 从 0 开始做一次 EasyR1→NPU 移植

**目标读者**：拿到本仓希望**重做一次移植**的人——可能是新版本 EasyR1、可能是新 base image、甚至可能是一个完全不同的 RL 框架（veRL、OpenRLHF、TRL）。

**本手册回答**：
1. 每个 skill 做什么？输入输出是什么？
2. 从 0 开始移植按什么顺序调这些 skill？
3. 哪一步输出作为下一步的输入？
4. 哪些决策点需要人拍板、哪些可以交给 skill 自动推进？

配套文档：
- `PORT-GUIDE.md` — 如果你只是要**跑起来**（不是重做），读那本
- 每个 skill 自己的 `SKILL.md`（在 `skills/*/SKILL.md`）是 authoritative reference

---

## 1. 8 个 skill 一览

| Skill | 一句话说明 | 适用频率 |
|---|---|---|
| `npu-image-inspect` | 给一个 Ascend NPU docker image，抽出它装了啥（CANN、python、torch_npu、transformers 等），emit `knowledge/images/<slug>.md` | 每个新目标 image 一次 |
| `dep-gap-detect` | 给定 requirements.txt + 目标 image inventory，**自动判断** 所有依赖的 A/B/C/D/E 分级。D=0 → 场景 P1 走标准升级；D≥1 → 场景 P2，必须先完成 NPU 适配 | 新 EasyR1 commit / 新 target image 之前先跑这个 |
| `npu-code-path-sweep` | 扫一个 Python 源码树找所有 GPU-only 调用点（`torch.cuda.*` / `"cuda"` / `flash_attn` / `nccl`），emit `docs/code-path-sweep-<repo>.md` | 每个新移植目标一次，或大版本升级后 |
| `npu-container-runner` | 启动 NPU 容器，设备 passthrough + bind mount + chip 占用检查 + HCCL 环境变量 | 每次跑 smoke 都用 |
| `upstream-branch-hygiene` | 纪律规范：所有上游修改走 `ascend-port` 分支 + 本地 push + NPU host pull，**绝不**在容器内直接改 site-packages | 贯穿始终 |
| `ray-npu-shim` | 一个 ~100 LOC 的 drop-in Python 模块，解决 Ray 不识别 NPU 的三个问题（resource 注册 + `ASCEND_RT_VISIBLE_DEVICES` 被 Ray 2.55 清掉 + `VLLM_ASCEND_ENABLE_NZ=0`） | 每个 Ray-based RL 框架移植都要用 |
| `image-upgrade-drill` | 把一次 image / 框架大版本升级走完 7 步演练（baseline → 预检 → 建 drill 分支 → iterate code break → 跑 bug probe → 20-step trajectory → 报告 + cherry-pick），产出 `docs/<target>-upgrade-drill.md` | CANN 8.5.x → 8.5.y、transformers 4→5、vllm 0.13→0.18 这种升级 |
| `codex-review` | 用本地 codex CLI 做第二眼审查，作为 sign-off 代理 | 每个重要里程碑 |

---

## 2. 部署 skills + 拉 upstream 参考代码

**2.1 部署 skills 到 Claude Code**

本仓的 skills 不是 Python 包，是 markdown + shell 脚本。让 Claude Code 看到它们：

```bash
cd easyr1-npu
bash scripts/install-skills.sh                 # symlink 到 ~/.claude/skills/
# 或指定路径
bash scripts/install-skills.sh --skills-dir /path/to/other/skills/
# 卸载
bash scripts/install-skills.sh --undeploy
```

部署后在 Claude Code 里 `/` 能看到 `npu-image-inspect`、`ray-npu-shim` 等。详见 `install-skills.sh --help`。

**2.2 拉 upstream 参考代码**

移植过程中 skill 和人都要 grep 上游库的源码（找 CUDA-only 调用、比对 API 改名、查 upstream bug 是否已修）。`fetch-upstream.sh` 做这件事：

```bash
# 只拉 essential（EasyR1 fork）
bash scripts/fetch-upstream.sh

# 拉全部 essential + optional（verl/transformers/torch-npu/vllm-ascend/triton-ascend）
bash scripts/fetch-upstream.sh --include-optional
```

默认拉到 `../upstream/`（跟 `easyr1-npu/` 同级）。已存在就 `git fetch` 刷新；否则 `git clone`。idempotent，安全重跑。

**要 clone 哪些**：
- **essential**：`EasyR1`（你的 port 目标；fork 到自己 namespace 才能 push）
- **optional**：其它参考库。磁盘紧就不拉；skill 的 `Step 3 (version-align upstream refs)` 需要 `upstream/torch-npu/` 等 ref 源码时再拉

---

## 3. 从 0 开始移植的 9 步 workflow

假设你拿到一个**新的 EasyR1 commit**（或另一个 RL 框架）+ **一个候选目标 image**，下面是推荐的调用顺序。

### Step 0 — 前置（读文档 + 拉代码，20 分钟）

- 读 `PORT-GUIDE.md` §1-§4 了解我们做过什么（尤其 §1 "为什么不需要改 upstream 库"）
- 读 `knowledge/npu-patterns.md` 的 23 个 stable ID 标题（不用细读，标题就够，遇到 bug 能回来 grep）
- 读 `knowledge/upstream-refs.md` 了解 upstream ref 怎么对齐（**不能默认 review master**）
- 跑 `bash scripts/fetch-upstream.sh --include-optional` 把 upstream 参考代码拉下来（上面 §2.2）

### Step 1 — 摸清目标 image（skill: `npu-image-inspect`）

```bash
bash scripts/inspect-ascend-image.sh quay.io/ascend/verl:verl-X.Y.Z-a3-...
```

**输出**：`knowledge/images/<slug>.md`，包含 CANN 版本、torch_npu、transformers、vllm_ascend、triton_ascend、完整 pip freeze、NPU-BUG-001 triton 完整性检测。

**决策点**：检查输出的 `## Triton-ascend integrity check` —— 如果有 warning，你的 Dockerfile 必须加 `pip install --force-reinstall --no-deps triton-ascend==<version>`。

### Step 1.5 — 自动判断场景 P1 vs P2（skill: `dep-gap-detect`）

这一步**必跑**。判断新 EasyR1 / 新 image 组合是否引入了需要 NPU 额外开发的依赖（场景 P2），还是只需要标准升级（场景 P1）。

```bash
# 提取 EasyR1 master 的 requirements
git -C ../upstream/EasyR1 show main:requirements.txt > /tmp/easyr1-reqs.txt

# 跑检测
bash scripts/dep-gap-detect.sh \
  --reqs /tmp/easyr1-reqs.txt \
  --image-inventory knowledge/images/<slug>.md \
  --out /tmp/gap-report.md
```

**决策点**：
- 退出码 0（**D = 0，场景 P1**）→ 继续 Step 2 的 drill 流程
- 退出码 1（**D ≥ 1，场景 P2**）→ **停下来**。按 [`P2-WORKFLOW.md`](P2-WORKFLOW.md) 的端到端流程走（识别 gap → 分档 tier 1/2/3 → 建任务 → 执行 → 验证 → 接回 P1）。task 登记在 [`docs/easyr1/npu-adaptation-tasks.md`](npu-adaptation-tasks.md)

这个 skill 的内置 PACKAGE_RULES 就是 NPU 生态知识库的编码。识别出新 pattern 要更新它（见 `skills/dep-gap-detect/SKILL.md`）。

### Step 2 — 基础设施预检（skill 里 `image-upgrade-drill` Step 2；新项目也要做）

```bash
# NPU-OPS-007: image 里有 pip index 配置吗？
docker run --rm --entrypoint cat <image> /etc/pip.conf 2>/dev/null || echo 'NO_PIP_CONF'

# NPU-OPS-008: 我们 pin 的 wheel 在 huaweicloud 还有吗？
for pkg in triton-ascend torch-npu vllm-ascend; do
  curl -sL --max-time 30 "https://mirrors.huaweicloud.com/ascend/repos/pypi/$pkg/" | \
    grep -coE "$pkg[._-][^\"<>]*\.whl" | xargs -I{} echo "$pkg: {} wheels"
done

# NPU-OPS-006: host docker daemon 的代理配置
systemctl cat docker | grep -E 'HTTP_PROXY|NO_PROXY'
```

**决策点**：
- `/etc/pip.conf` 空 → Dockerfile 里烤 `ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/`
- huaweicloud 返回 0 wheels → 切到 aliyun 或别的 mirror
- daemon 有代理 → 加 `NO_PROXY` 再 pull

**这一步省下来的时间**：1-2 小时（每个坑**反应式**踩要 15-45 分钟，**预检式**查 30 秒）。

### Step 3 — 对齐上游 ref（手动，10 分钟）

编辑 `knowledge/upstream-refs.md`，对新 image 写一行新记录，列出 torch-npu / vllm-ascend / transformers / triton-ascend 对应的 branch/tag。依据：每个上游的 README 兼容性表。

```bash
# 在本地 upstream/ 里切 ref，方便后面 grep 原始代码
cd upstream/torch-npu && git checkout origin/v2.8.0-7.3.0
cd upstream/vllm-ascend && git checkout origin/releases/v0.13.0
# etc
```

### Step 4 — Fork + 建 port 分支（skill: `upstream-branch-hygiene`）

```bash
# 在 GitHub / GitCode fork 目标 repo 到 zhshgmail（或你的 namespace）
gh repo fork hiyouga/EasyR1 --clone=false
gh api --method PATCH repos/zhshgmail/EasyR1 -f visibility=private

# Local clone 加 personal remote
cd upstream/EasyR1
git remote add personal https://github.com/zhshgmail/EasyR1.git
git checkout -b ascend-port origin/main
```

**分支命名约定**：
- `main` = upstream mirror（**不改**）
- `ascend-port` = 我们的 port 发布分支
- `ascend-port-<target>-upgrade` = 升级演练分支（drill）
- 所有 drill-only commit 带 `[drill]` 前缀，用于 Step 7 的 cherry-pick 边界区分

### Step 5 — 扫 CUDA-only 调用点（skill: `npu-code-path-sweep`）

```bash
bash scripts/code-path-sweep.sh "$HOME/workspace/easyr1-npu/upstream/EasyR1"
```

**输出**：`docs/easyr1/code-path-sweep-EasyR1.md`，按 `NPU-CP-001` / `NPU-CP-002` / ... 分段的 hit 表格。

**决策点**：每个 hit 三选一
- **现在修** —— 写 device accessor + 替换（参考 §3.2、3.3 的 `verl/utils/device.py` pattern）
- **deferred** —— 标注到 `npu-gap-plan.md`，先让 smoke 跑再说
- **N/A** —— 假阳性（注释里、dead code）

### Step 6 — 应用 5 个 archetype 改动

按 `PORT-GUIDE.md §3.7` 的 5 档 archetype 顺序推：

1. **设备 dispatch**：copy `verl/utils/device.py` 模式
2. **版本 compat shim**：`try/except` import、`hasattr` gate、vllm 0.13 的 `vllm.lora.models` rename 等
3. **Ray 集成**：`ray_npu_shim.py` drop-in（skill `ray-npu-shim`）—— 把 `ray.init` 换成 `ray_init_npu_aware`、`num_gpus` 换成 `apply_actor_options`、等等。详见 `skills/ray-npu-shim/SKILL.md`
4. **Vendoring**：`from flash_attn import X` → `from transformers.integrations.npu_flash_attention import X`
5. **Dockerfile + smoke 脚本**：基于我们的 `Dockerfile.npu` + `examples/qwen2_0_5b_math_grpo_npu_smoke*.sh` 改

### Step 7 — 走 smoke 梯子（skill: `npu-container-runner`）

按 V1.1 → V1.3 → V1.4 → V1.5 → V2.1 → V2.2 顺序跑，fail-fast。

```bash
bash scripts/run-npu-container.sh --chips 0,1 \
  -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

**每级 fail 怎么办**：
1. **第一件事 grep `npu-patterns.md`**（24 条 stable ID 命中率非常高）
2. 没命中 → 写新的 stable ID，按 `Symptom / Root cause / Fix / Commit ref / Generalizable rule` schema 加一条
3. 修复 → commit → push personal → NPU host pull → 重跑

**baseline 数值**：见 `scripts/smoke/README.md`。step 1 的 entropy_loss 漂移 > ±5% 说明依赖 drift，先停下来比版本。

### Step 8 — 升级演练（可选；skill: `image-upgrade-drill`）

**什么时候用**：目标 image 同时升级了多个主版本（CANN 8.x → 8.y、transformers 4→5、vllm 0.1x → 0.1y）。走完这个 skill 会产出一份带数字的 drill report（预测 vs 实际成本、LOC 变动、bug probe 结果），作为"能不能切到新 image"的决策依据。

**输出**：`docs/<target>-upgrade-drill.md` + 一个带 `[drill]` 前缀的 commit 序列 + backward-compat cherry-pick 清单。

### Step 9 — Sign-off（skill: `codex-review`）

```bash
# 写 review prompt 到文件
cat > /tmp/review-prompt.txt <<'EOF'
You are reviewing the EasyR1 NPU port ...
Files to read: ...
Output: STRENGTHS / ISSUES / SUGGESTIONS / VERDICT
EOF

cat /tmp/review-prompt.txt | \
  codex exec --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox
```

**为什么要 bypass sandbox**：共享 host 上 bubblewrap 会 block，导致 codex 读不到本地文件又不报错，改成基于 inference 瞎编。一定加这个 flag。

---

## 4. 决策点速查表

| 时刻 | 谁决定 | 看哪里 |
|---|---|---|
| 用哪个 target image | 你 / user | `knowledge/images/*.md` 的已验证版本；`PORT-GUIDE.md §2` |
| 一个 CUDA-only 调用要现修还是 defer | 你 | `npu-patterns.md` 是否已覆盖；`docs/easyr1/npu-gap-plan.md` 是否已计划 |
| 新发现的 bug 是否要加 stable ID | 你 | 判断标准：下次还会撞吗？撞了能通过这个 ID 的 "Generalizable rule" 行在 5 分钟内定位吗？能就加 |
| 是否切新 image（upgrade） | user | `image-upgrade-drill` 输出的 PASS / BLOCKED verdict + cost report |
| 跑 smoke 用哪些 chip | 你 + A3 状态 | `npu-smi info` + `npu-container-runner` 的自动检查 |

---

## 5. 移植成本参考

基于两次数据点（v1 初次移植 + v2 transformers 升级演练）：

| 场景 | 预计工时 | LOC 变更 |
|---|---|---|
| 同 EasyR1 版本、换一台机器 | ~半天 | 0 LOC（全是 setup） |
| EasyR1 小版本升级（无主版本 dep 变化） | 0.5-1 天 | 2-4 处 catalog pattern 命中 |
| 移植到另一个 Ray-based RL 框架（OpenRLHF / TRL） | 1-1.5 天 | 框架 CUDA-only 调用点重跑 sweep + shim |
| Transformers / vllm / torch_npu 大版本同时升级 | ~1 小时（代码） + ~1 小时（踩基础设施坑） | 2-4 LOC（backward-compat try/except） |

代码成本一旦 helper layer（`verl/utils/device.py` + `hasattr`-gated vllm imports）稳定就很低。**基础设施成本反而是大头** —— 每个新 image 平均 1-2 个 infra 坑，每个未预检踩一次 10-15 分钟。Step 2 的预检节省时间。

---

## 6. 当 skill 自己不写代码时 —— 三档责任划分

**第一性原则**：本项目目标是让 EasyR1 master 跑在 A3 上。遇到 NPU 生态 gap 时**必须识别 + 驱动解决**，不能用 "不在 scope" 绕过。但"驱动解决"不意味着本仓直接写所有代码；下面是三档责任划分：

### 档 1：本仓直接做

- EasyR1 自己的源码改动（device 路由、版本 compat shim、Ray 集成、Dockerfile）
- Python 层 shim / fork 来桥接 CUDA-only 包 → NPU 替代（e.g. flash_attn → `transformers.integrations.npu_flash_attention`）
- 向 vllm-ascend / triton-ascend / torch_npu 的 **Python 层**提 issue 或 PR
- 识别 NPU 适配 gap，记入 `docs/easyr1/npu-adaptation-tasks.md`（待建）

### 档 2：委托给姐妹项目 / 独立仓（本仓 track 适配，实现在别的仓）

- **新 CANN 算子（kernel math）的实现**：
  - A3 kernel 精度验证 → 委托给 [`ascend-fused-accuracy-probe`](https://gitcode.com/zhengshencn_hwca/ascend-fused-accuracy-probe)
  - A5 kernel 生成 → 委托给 [`a5_ops`](https://gitcode.com/zhengshencn_hwca/a5_ops)
  - A3 kernel 生成 → 有类似的独立仓（按需调用）
  - **本仓做的事**：识别 "EasyR1 需要某个 fused op 但没现成 NPU 实现" → 建 `docs/easyr1/npu-adaptation-tasks.md` 任务 → 协调姐妹项目完成 → 接回 EasyR1 使用

- **torch_npu C++ 扩展层的 op 实现**：同上，委托给 Ascend PyTorch 团队 / 相关 kernel 项目。本仓 track

- **多节点 / 跨 host HCCL 调优**：超出单节点 scope 的部分，建任务；可能委托给专门的 HCCL 调优项目或内部团队

### 档 3：只能向 Ascend 团队提需求（我们做不了 workaround 的情况）

- **CANN runtime 框架级的 bug**（ACL C 层本身 crash、HCCL C 协议层冲突、driver 层问题）—— 这类我们没权限改 runtime 代码，能做的是**提 issue + 写 workaround + 记录进 `npu-patterns.md` stable ID + 等 Ascend 团队修**

---

**关键**：档 2 和档 3 的"委托"和"提需求"**不是推卸**。每一条 gap 都要：
1. 识别并记录（哪个功能需要、当前状态、blocker 在哪）
2. 建对应任务到 `docs/easyr1/npu-adaptation-tasks.md`（谁来做、预计什么时间、怎么 track）
3. Track 进度到完成
4. 完成后接回 EasyR1 使用

"不在 scope" 不是可接受的结束状态。

---

## 6.5 Port-expert skills（适配上游新版本，2026-04-24 新增）

除了上面 8 个 EasyR1 移植 skill，还有 3 个 **port-expert skill**，用来应对**上游库发新版本**时 vllm-ascend / torch_npu 出现 API drift 的情况。这套 skill 独立运行，不依赖前面的 8 个。

### 目标读者

你是某个**上游库的维护者**或者**近用户**，上游发了新版本但 NPU 适配还没跟上。典型场景：
- torch 发布 2.12，torch_npu 还锁在 2.11 → 用 `torch-npu/port-expert`
- vllm 主仓合并了 API 重构，vllm-ascend 没跟上 → 用 `vllm-ascend/port-expert`
- 想验证你刚写的 compat shim 两条分支都对 → 用 `drift-port-validate`

### 三个 skill

| Skill 名 | 做什么 | 关键工具 | 用户指南 |
|---|---|---|---|
| `vllm-ascend/port-expert` | 扫 vllm 新版本 diff，找出 vllm-ascend 受影响的符号，按 F1-F8 族匹配并建议修法 | `scripts/kb_drive_test.py`, `scripts/sweep.sh` | [`docs/vllm-ascend/PORTING-GUIDE.md`](../vllm-ascend/PORTING-GUIDE.md) |
| `torch-npu/port-expert` | 扫 torch 新版本里所有 `torch._<private>` 符号，找出 torch_npu 受影响的导入路径搬家 | `scripts/extract_imports.py`, `scripts/check_drift.py`, `scripts/check_sig_drift.py` | [`docs/torch-npu/PORTING-GUIDE.md`](../torch-npu/PORTING-GUIDE.md) |
| `_shared/drift-port-validate` | 验证写好的 compat shim 两条分支（OLD upstream 保留 + NEW upstream fallback）都对 | `references/templates/*_verify_{old,new}.py` | 无专属用户指南，在每个 port-expert 的 `/drift-port-validate` 步骤里被调用 |

### 典型调用链

```
用户 → /vllm-ascend-day0 或 /torch-npu-day0
         ↓ (orchestrator skill 调起)
         ├─ Phase 0/0.5: 运行扫描器（sweep.sh / check_drift.py）列出所有漂移
         ├─ Phase 1-3: 按 F1-F8 族写 compat shim
         ├─ Phase 4: 调 /drift-port-validate 验证 shim
         └─ Phase 5-6: push 到 fork branch + 生成 PR 材料
```

### 与主 8 skill 的关系

- 主 8 skill 解决"EasyR1 在 A3 上跑起来 + 新版本升级整个 RL 栈"
- port-expert 3 skill 解决"某一个上游库的新版本漂移到 NPU 适配"

两套**不重叠**：port-expert 假设 NPU base image 已经有某一旧版本，你要把上游最新版本桥接进去。主 skill 假设你要做的是端到端 RL 栈的一次完整迁移。

### F 族漂移分类（摘要）

F1-F8 是**跨上游通用**的漂移形状，最早在 vllm-ascend 沉淀，后被 torch_npu 复用并加了 F2-path-move 变种：

- **F1** 符号被删：`try/except import` fallback 到本地 class / 新 upstream 路径
- **F2-rename** 类改名：别名导入
- **F2-path-move**（torch_npu 扩展）同名符号搬家：`try/except` 两条 import
- **F3** 函数签名变：运行时 `inspect.signature` sniff
- **F4** 返回值类型变：duck-type / shim
- **F5** buffer API 迁移：compat helper 函数
- **F6** kv_cache 契约变
- **F7** 基类新增必须字段
- **F8** 基类新增必须方法

完整描述见 [`src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md`](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md)。

**2026-04-24 late 扫描器覆盖矩阵**：

| Family | 工具 | 模式 |
|---|---|---|
| F1 / F2-rename / F3 / F5-suspect | `kb_drive_test.py` + `sweep.sh` | 逐 commit / commit-range |
| F4（返回值类型漂移） | `check_f4.py` | tag-range AST diff |
| F7（基类新属性） | `check_f7_f8.py` | tag-range AST diff |
| F8（基类新方法） | `check_f7_f8.py` | tag-range AST diff |
| F6（kv_cache 契约） | **仍需手动** | 运行时断言，非结构化 |

典型 Mode Sweep 一次会跑 3 个工具：`sweep.sh` + `check_f4.py` + `check_f7_f8.py`。

## 7. 相关文档

- 每个 skill 的权威说明：`src/skills/<name>/SKILL.md`
- 坑目录：`knowledge/npu-patterns.md`
- 上游 ref 对齐：`knowledge/upstream-refs.md`
- 跑起来手册（非复现）：`PORT-GUIDE.md`
- 当前状态 + 未结工作：`HANDOVER.md`
- v2 drill 完整报告（`image-upgrade-drill` skill 的首个实例）：`transformers-upgrade-drill.md`
- Port-expert 专用：[`docs/vllm-ascend/PORTING-GUIDE.md`](../vllm-ascend/PORTING-GUIDE.md), [`docs/torch-npu/PORTING-GUIDE.md`](../torch-npu/PORTING-GUIDE.md)
