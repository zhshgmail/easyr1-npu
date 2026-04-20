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

## 1. 7 个 skill 一览

| Skill | 一句话说明 | 适用频率 |
|---|---|---|
| `npu-image-inspect` | 给一个 Ascend NPU docker image，抽出它装了啥（CANN、python、torch_npu、transformers 等），emit `knowledge/images/<slug>.md` | 每个新目标 image 一次 |
| `npu-code-path-sweep` | 扫一个 Python 源码树找所有 GPU-only 调用点（`torch.cuda.*` / `"cuda"` / `flash_attn` / `nccl`），emit `docs/code-path-sweep-<repo>.md` | 每个新移植目标一次，或大版本升级后 |
| `npu-container-runner` | 启动 NPU 容器，设备 passthrough + bind mount + chip 占用检查 + HCCL 环境变量 | 每次跑 smoke 都用 |
| `upstream-branch-hygiene` | 纪律规范：所有上游修改走 `ascend-port` 分支 + 本地 push + NPU host pull，**绝不**在容器内直接改 site-packages | 贯穿始终 |
| `ray-npu-shim` | 一个 ~100 LOC 的 drop-in Python 模块，解决 Ray 不识别 NPU 的三个问题（resource 注册 + `ASCEND_RT_VISIBLE_DEVICES` 被 Ray 2.55 清掉 + `VLLM_ASCEND_ENABLE_NZ=0`） | 每个 Ray-based RL 框架移植都要用 |
| `image-upgrade-drill` | 把一次 image / 框架大版本升级走完 7 步演练（baseline → 预检 → 建 drill 分支 → iterate code break → 跑 bug probe → 20-step trajectory → 报告 + cherry-pick），产出 `docs/<target>-upgrade-drill.md` | CANN 8.5.x → 8.5.y、transformers 4→5、vllm 0.13→0.18 这种升级 |
| `codex-review` | 用本地 codex CLI 做第二眼审查，作为 sign-off 代理 | 每个重要里程碑 |

---

## 2. 部署 skills 到 Claude Code

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

---

## 3. 从 0 开始移植的 7 步 workflow

假设你拿到一个**新的 EasyR1 commit**（或另一个 RL 框架）+ **一个候选目标 image**，下面是推荐的调用顺序。

### Step 0 — 前置（读文档，15 分钟）

- 读 `PORT-GUIDE.md` 了解我们做过什么
- 读 `knowledge/npu-patterns.md` 的 23 个 stable ID 标题（不用细读，标题就够，遇到 bug 能回来 grep）
- 读 `knowledge/upstream-refs.md` 了解 upstream ref 怎么对齐（**不能默认 review master**）

### Step 1 — 摸清目标 image（skill: `npu-image-inspect`）

```bash
bash scripts/inspect-ascend-image.sh quay.io/ascend/verl:verl-X.Y.Z-a3-...
```

**输出**：`knowledge/images/<slug>.md`，包含 CANN 版本、torch_npu、transformers、vllm_ascend、triton_ascend、完整 pip freeze、NPU-BUG-001 triton 完整性检测。

**决策点**：检查输出的 `## Triton-ascend integrity check` —— 如果有 warning，你的 Dockerfile 必须加 `pip install --force-reinstall --no-deps triton-ascend==<version>`。

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

**输出**：`docs/code-path-sweep-EasyR1.md`，按 `NPU-CP-001` / `NPU-CP-002` / ... 分段的 hit 表格。

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
1. **第一件事 grep `npu-patterns.md`**（23 条 stable ID 命中率非常高）
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
| 一个 CUDA-only 调用要现修还是 defer | 你 | `npu-patterns.md` 是否已覆盖；`docs/npu-gap-plan.md` 是否已计划 |
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

## 6. 什么时候这些 skill 不够用

- 需要**写新 CUDA-equivalent 算子**（例如某个 fused attention 在 NPU 上不存在）—— 出 `skills/` 范围，参考姐妹项目 `ascend-fused-accuracy-probe` 的 `fused-op-accuracy-probe` skill
- 需要**跨多 host RDMA / HCCL 调优** —— 本项目 skills 只覆盖单节点；多节点是 `DELIVERABLE.md` 标的 known debt
- 需要**改 CANN 本身或 torch_npu 的 C++ 层** —— 那是 gitcode Ascend 仓的工作，超出本项目 scope

---

## 7. 相关文档

- 每个 skill 的权威说明：`skills/<name>/SKILL.md`
- 坑目录：`knowledge/npu-patterns.md`
- 上游 ref 对齐：`knowledge/upstream-refs.md`
- 跑起来手册（非复现）：`PORT-GUIDE.md`
- 当前状态 + 未结工作：`HANDOVER.md`
- v2 drill 完整报告（`image-upgrade-drill` skill 的首个实例）：`transformers-upgrade-drill.md`
