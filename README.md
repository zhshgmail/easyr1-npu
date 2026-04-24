# easyr1-npu

**把 EasyR1（`hiyouga/EasyR1`）移植到 Ascend 910C (A3) NPU，并沉淀一套可复用的"GPU RL 框架→NPU"移植 skills。**

本仓交付两件事：
1. **EasyR1 在 A3 上能跑** —— 代码改动在 [`zhshgmail/EasyR1`](https://github.com/zhshgmail/EasyR1) 的 **`ascend-port`** 分支（20 个 commit）
2. **一套可复用的移植 skills** —— 下一次 EasyR1 版本升级 / CANN 升级 / 别的 RL 框架（OpenRLHF / TRL）移植都能套用

> **📣 2026-04-20 Scope 说明（勘误）**：之前版本的 README / SKILLS-GUIDE 把"改 NPU 上游库"和"改 CANN / torch_npu C++ 层"都打成 ❌ 不在 scope 是**错误的表述**。
>
> 正确的区分（按"谁来做"分三档）：
> - ✅ **本仓直接做**：EasyR1 源码改动、Python 层 shim / fork、向 vllm-ascend / triton-ascend / torch_npu 上游提 issue 或 PR、识别 NPU 适配 gap 并 track
> - 🤝 **委托给姐妹项目 / 独立仓**：CANN 算子实现、kernel 数值精度验证 —— 用 `ascend-fused-accuracy-probe`（A3 kernel 验证）和 `a5_ops`（A5 kernel 生成，A3 有类似的独立仓）这些专门的项目做。本仓**识别 gap + 建接口 + track 适配**，不把 kernel commit 放本仓
> - 📣 **提需求给 Ascend 团队**：CANN runtime 框架级的 bug（ACL C 层本身的 crash、HCCL C 层协议问题）—— 这类我们做 workaround + 提 issue，修是 Ascend 团队的事
>
> **关键原则**：如果新 EasyR1 版本依赖一个 NPU 还没覆盖的包 / 算子，"不在 scope" 不是可接受答案 —— 项目目标就死了。**必须建任务推动适配**，哪怕具体 commit 落到别的仓。见路径 4 + `docs/easyr1/npu-adaptation-tasks.md`（待建）。

---

## 选你要走的路径

本仓服务**四类用户**，按你的目标选：

| # | 你想要 | 路径 | 读哪本 |
|---|---|---|---|
| 1 | **只想在 A3 上把 EasyR1 跑起来** | 用 v1 已验证发布路径（`ascend-port` 分支 + CANN 8.5.0 image） | [`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md) |
| 2 | **复现 / 自动化 EasyR1 移植** | 从 0 跑移植流程（用 8 个 skill + `fetch-upstream.sh`） | [`docs/_meta/SKILLS-GUIDE.md`](docs/_meta/SKILLS-GUIDE.md) |
| 3 | **用带新依赖（transformers 5 / CANN 8.5.1）的 EasyR1** | 看 drill 演练当前状态 + 自行验证 | [`docs/transformers/UPGRADE-DRILL-STATUS.md`](docs/transformers/UPGRADE-DRILL-STATUS.md) |
| 4 | **复现"把 EasyR1 + 新依赖一起移植到 NPU"的自动流程** | 走 `image-upgrade-drill` skill 7 步流程 | [`docs/_meta/SKILLS-GUIDE.md`](docs/_meta/SKILLS-GUIDE.md) §8 + [`docs/transformers/UPGRADE-DRILL-STATUS.md`](docs/transformers/UPGRADE-DRILL-STATUS.md) |
| 5 | **把 vllm-ascend 适配到一个新版本的 vllm**（扫描 drift + 按族写 shim + 验证） | 用 `/vllm-ascend-day0` skill + kb_drive_test 扫描器 + `/drift-port-validate` 验证 | [`docs/vllm-ascend/PORTING-GUIDE.md`](docs/vllm-ascend/PORTING-GUIDE.md) |
| 6 | **把 torch_npu 适配到一个新版本的 torch**（扫描私有模块搬家 + 按 F2-path-move 族写 shim） | `sweep.sh` 一键跑 4 个 scanner (extract_imports + check_drift + check_sig_drift + check_f7_f8) → 按 F-family 模板写 `torch_npu/compat/` shim → `/drift-port-validate` 验证 | [`docs/torch-npu/PORTING-GUIDE.md`](docs/torch-npu/PORTING-GUIDE.md) |
| 7 | **把 transformers 适配到一个新版本**（NPU 面很小，90% outcome A） | Stage 0 快速决策树：byte-compare NPU 集成文件 + ALL_ATTENTION_FUNCTIONS key 集合，通常无需 A3 | [`src/skills/transformers/port-expert/SKILL.md`](src/skills/transformers/port-expert/SKILL.md) |
| 8 | **把 triton-ascend 适配到新版本 triton**（Fork 模式，不是 plugin） | `git rebase --onto <new-triton-tag> <current-base>` → 手动解决 conflict（KB 列了 5 个历史 surface）→ rebuild + NPU smoke | [`src/skills/triton-ascend/port-expert/SKILL.md`](src/skills/triton-ascend/port-expert/SKILL.md) |

---

## 路径 1：只想把 EasyR1 跑起来（最快 ~30 分钟）

**前置**：A3 host（≥ 2 空闲 chip）+ Docker + HuggingFace 访问（国内走镜像）。

**你不需要**：clone 任何 upstream 库（`torch-npu` / `transformers` 等）。image 里全有。

**仓库**：本仓（`zhshgmail/easyr1-npu`）和 EasyR1 fork（`zhshgmail/EasyR1`）都是 **public**，直接 clone 即可。

```bash
# 1. Pull 镜像（国内走 NJU mirror；国外直接 quay.io/ascend）
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
docker tag quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest \
           quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest

# 2. Clone 本仓（拿 runner 脚本）+ EasyR1 fork 的 ascend-port 分支
mkdir -p "$HOME/workspace" && cd "$HOME/workspace"
git clone https://github.com/zhshgmail/easyr1-npu.git
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git

# 3. Build 层叠 image（~3 分钟）
cd EasyR1 && docker build -t easyr1-npu:ascend-port -f Dockerfile.npu .

# 4. 下载模型
mkdir -p "/data/$USER/models"
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen2-0.5B-Instruct \
  --local-dir "/data/$USER/models/Qwen2-0.5B-Instruct"

# 5. 跑 V1.4 smoke（~8 分钟；期望 entropy_loss step1 ≈ 0.991）
cd "$HOME/workspace/easyr1-npu"
bash scripts/run-npu-container.sh --chips 0,1 \
    --live-source "$HOME/workspace/EasyR1" \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

完整细节 + 期望数值 + 遇到问题怎么办 → [`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)

---

## 路径 2：复现 / 自动化 EasyR1 移植

适合：新版本 EasyR1 发布后你想自动化重跑这套移植流程；或要移植另一个 Ray-based RL 框架（OpenRLHF / TRL）到 NPU。

**额外需要**：upstream 库的本地 clone（供 skill grep 参考代码）。

```bash
# 1. Clone 本仓
git clone https://github.com/zhshgmail/easyr1-npu.git
cd easyr1-npu

# 2. 部署 skills 到 Claude Code（在本地开发机上，不是 A3）
bash scripts/install-skills.sh     # symlink skills/* 到 ~/.claude/skills/

# 3. 拉 upstream 参考代码（默认只拉 essential，加 --include-optional 拉全部）
bash scripts/fetch-upstream.sh --include-optional

# 4. 按 SKILLS-GUIDE.md 的 9 步 workflow 走
```

完整 workflow（什么 skill 在什么时候调用、输入输出是什么、决策点在哪）→ [`docs/_meta/SKILLS-GUIDE.md`](docs/_meta/SKILLS-GUIDE.md)

---

## 路径 3：用新依赖的 EasyR1（transformers 5 / 新 image）

**当前状态**：drill 分支验证过**在新 image（8.5.2）上跑通 2-step + 20-step smoke**，2 个 backward-compat fix 已 cherry-pick 到 `ascend-port` 分支。

**但**：没跑完整 V1.1 → V2.2 smoke 梯子，没做长训练收敛验证。**不推荐生产切过去**。

详细状态 + caveat + 要不要切的决策建议 → [`docs/transformers/UPGRADE-DRILL-STATUS.md`](docs/transformers/UPGRADE-DRILL-STATUS.md)

### 路径 3.5：社区刚发了新 transformers / vllm / torch，NPU 没跟上 —— Day-0 场景

这是**真正最难**的场景：community 刚 release 新版本，NPU 生态还没 ship 配套 image / wheel，你又想用。`/transformers-day0` / `/vllm-day0` / `/torch-day0` skills 就是为这个设计的。

**0 交互示例**（实测 2026-04-23，community transformers 5.6.0 前一天发布，NPU 原本无适配）→ [`docs/torch-npu/examples/transformers-5.6.0-day0.md`](docs/torch-npu/examples/transformers-5.6.0-day0.md)

示例包含：1 行 skill 调用、skill 在 5 个 Phase 里做什么、预期 3 种 outcome（works-as-is / forward-port / blocked）、反作弊独立 verify 步骤、真实测得的数字（step-1 entropy_loss=1.31 在 v2 band 内）。

---

## 路径 4：复现"EasyR1 + 新依赖自动移植"的流程

适合：NPU 软件栈升级（CANN 9.x、torch_npu 2.10、transformers 6 之类）发布后，你要评估 "我们的 EasyR1 port 能不能跟上新 image"，**以及**识别出哪些新依赖是 NPU 生态还没适配的，驱动相应的 NPU 适配任务。

**第一性原则重申**：本项目的目标是"让 EasyR1 master 在 A3 上跑"。如果新 EasyR1 版本或新 image 引入一个 NPU 还没适配的依赖，**我们不能说"不在 scope" 就结束** —— 那等于放弃项目目标。正确的做法是**识别这个 gap，建 NPU 适配任务，推动它完成**（可能是我们自己做，可能是协调 Ascend / upstream 团队做）。见 [`docs/easyr1/npu-adaptation-tasks.md`](docs/easyr1/npu-adaptation-tasks.md)（live task registry）+ [`docs/_archive/P2-WORKFLOW.md`](docs/_archive/P2-WORKFLOW.md)（端到端 P2 workflow 设计）。

**本路径明确在 scope**：
- 验证新 image 的一整套新依赖（torch_npu、vllm_ascend、triton_ascend、transformers 等）跟我们的 EasyR1 `ascend-port` 分支是否兼容，做数值 smoke 验证
- 修复 EasyR1 自己源码里的版本 compat 问题（e.g. transformers 5 改了 import 路径 → EasyR1 里加 try/except）
- 换 base image + rebuild 层叠 image
- **识别 NPU 适配 gap**：新 EasyR1 依赖某个 CUDA-only 包、或某个包的 NPU 移植还没跟上 → 用 `dep-gap-detect` skill 自动识别，建任务 track。见 [`docs/easyr1/npu-adaptation-tasks.md`](docs/easyr1/npu-adaptation-tasks.md)（live registry）+ [`docs/_archive/P2-WORKFLOW.md`](docs/_archive/P2-WORKFLOW.md)（端到端 workflow）
- **协调 NPU 适配工作落地**：比如需要给 vllm-ascend / triton-ascend 提 issue 或 PR、或跟 Ascend 团队提适配需求、或自己写 shim/fork

**本路径的工作委托给姐妹项目**（本仓识别 + 建接口 + track，具体实现在别的仓）：
- 🤝 **新 CANN 算子实现 / kernel 精度验证** → 委托给 `ascend-fused-accuracy-probe`（A3 kernel 验证）、`a5_ops`（A5 kernel 生成，A3 有类似的独立项目）这类专门 kernel 项目。本仓识别 "EasyR1 在 A3 上需要某个 fused attention / fused softmax 但没有现成 NPU 实现"，建 NPU 适配任务，协调姐妹项目完成

**真正上报给 Ascend 团队**（只有这一档是我们做不了 workaround 就只能等的）：
- 📣 **CANN runtime 的 C 层框架 bug**（ACL runtime 本身 crash、HCCL C 协议层问题）—— 我们能做的是提 issue + 做 workaround + 等修复

**Python 层的 shim、上游库的 issue、vllm-ascend 的移植协调、triton-ascend 的 wheel 整理 —— 都直接在本仓驱动**（哪怕具体 commit 是别的仓 merge 的）。

用 [`image-upgrade-drill`](skills/image-upgrade-drill/SKILL.md) skill 的 7 步演练流程。产物：
- 一份带数字的 drill report（预测 vs 实际成本、LOC 变动、bug probe 结果）
- backward-compat commit 序列，可 cherry-pick 进 `ascend-port`
- **NPU 适配 gap 清单**：哪些新依赖还没 NPU 适配，各自需要的工作量 / 谁去做 / 当前状态
- 新 stable ID 条目加到 `knowledge/npu-patterns.md`
- PASS / BLOCKED 决策依据（区分"EasyR1 代码 compat 完成" vs "NPU 适配 gap 完成"两件事）

**当前状态**：
- 🟡 **`ascend-port` V1.4 smoke 单 rung PASS on 两套 image**（2026-04-22，手动跑）：step1=0.991 exact on 8.5.0；step1=1.275 on 8.5.2（新 baseline）。**没**跑 V1.1/V1.3/V1.5/V2.1/V2.2 完整 ladder；**没**让 skill chain 端到端冷启动驱动。见 [`docs/_meta/HANDOVER.md §6.2`](docs/_meta/HANDOVER.md) + `knowledge/npu-patterns.md#npu-ops-010`（针对"skill 闭环"的诚实定义）
- ✅ **NPU 适配 gap 清单已建立**：[`docs/easyr1/npu-adaptation-tasks.md`](docs/easyr1/npu-adaptation-tasks.md)。EasyR1 master 当前 D 类 blocker 为 0（见 [`docs/easyr1/easyr1-dep-chain-audit.md`](docs/easyr1/easyr1-dep-chain-audit.md)），所以 tier-2 active 任务为空；tier-3 有 2 条（BUG-003/004）等上游修
- ✅ **P2 端到端 workflow 已设计**：[`docs/_archive/P2-WORKFLOW.md`](docs/_archive/P2-WORKFLOW.md)（"EasyR1 需要 NPU 没覆盖的东西时怎么闭环"）。但**尚未在真实 D ≥ 1 场景下实测**（当前 D = 0 还没真的触发过）—— 下次遇到时按 workflow 跑 + 修文档
- 🟡 **"skill 自动化端到端复现" 没充分证明**：2026-04-20 dry-run 发现当时 SKILL.md 直接写了答案（已修 commit `66c5ce9`）。agent 能否在**真未知 break** 下独立发现 gap 未验证。等下次真升级做 clean test

完整 skill 说明 → [`skills/image-upgrade-drill/SKILL.md`](skills/image-upgrade-drill/SKILL.md)
v2 drill 的首次实证报告 → [`docs/transformers/transformers-upgrade-drill.md`](docs/transformers/transformers-upgrade-drill.md)
Skill dry-run 验证 → [`docs/_archive/skill-dry-run-2026-04-20.md`](docs/_archive/skill-dry-run-2026-04-20.md)

---

## `zhshgmail/EasyR1` fork 的分支结构

Fork 上有 4 个分支，每个用途不同。**用户只切 `ascend-port`**，其它都是开发用。

| 分支 | 对应什么 | 用户该用吗？ |
|---|---|---|
| `main` | upstream `hiyouga/EasyR1:main` mirror；无 NPU 改动 | ❌ clone 默认分支会拿不到 NPU 代码 |
| **`ascend-port`** | `main` + 20 个 NPU port commit；已 backward-compat 兼容 v1/v2 两套 image | ✅ **发布分支，就用这个** |
| `ascend-port-transformers-upgrade` | `ascend-port` + 9 个 `[drill]` commit；实验性 Dockerfile + BUG probe 脚本 | ❌ drill 演练分支 |
| `ascend-port-transformers-upgrade-reproduce` | 从 `ascend-port` 派生的 skill 复现验证分支 | ❌ 复现实验分支 |

**关键**：
- `main` ≠ 发布分支。**一定用 `git clone -b ascend-port`**
- drill 分支跟 `ascend-port` **完全隔离**（drill 只加了新文件 `Dockerfile.npu-852`，没改 `Dockerfile.npu`）
- drill 的 backward-compat fix 通过 cherry-pick 进 `ascend-port`，**用 `try/except` + `hasattr` 写法同时兼容 transformers 4/5 和 vllm 0.13/0.18**

---

## 仓库布局

```
easyr1-npu/                           ← 本仓（github.com/zhshgmail/easyr1-npu）
├── README.md                         ← 你在这里
├── CLAUDE.md                         ← 项目指令（给 Claude Code 用）
├── docs/
│   ├── PORT-GUIDE.md                 ← 路径 1：怎么跑起来
│   ├── SKILLS-GUIDE.md               ← 路径 2 & 4：怎么用 skill 自动化移植（P1 workflow）
│   ├── UPGRADE-DRILL-STATUS.md       ← 路径 3：升级演练当前状态
│   ├── P2-WORKFLOW.md                ← 路径 4 补充：EasyR1 需要 NPU 没覆盖的依赖时的端到端 workflow
│   ├── NEXT-SESSION-STARTER.md       ← **冷启动 10 分钟 checklist（最新来的 agent 先读）**
│   ├── HANDOVER.md                   ← 当前状态 + 未结工作（**session 起手必读**）
│   ├── DOCS-CONVENTION.md            ← **文档组织 convention + 归属 map**（贡献者 / agent 必读）
│   ├── DELIVERABLE.md                ← 正式 sign-off 文档
│   ├── easyr1-dep-chain-audit.md     ← EasyR1 master 依赖 A/B/C/D/E 分级审计
│   ├── npu-adaptation-tasks.md       ← NPU 适配 task 清单（档 1/2/3）
│   ├── transformers-upgrade-drill.md ← v2 升级演练完整报告（drill 实证）
│   ├── skill-dry-run-2026-04-20.md   ← Skill 有效性 dry-run 验证记录
│   ├── porting-journal.md            ← 时间线日志
│   ├── design.md                     ← 原始需求 + 任务拆解
│   ├── skills-design.md              ← skills 系统架构设计
│   ├── dep-matrix.md                 ← 早期 GPU↔NPU 依赖对照（被 easyr1-dep-chain-audit 取代）
│   ├── npu-gap-plan.md               ← 早期 gap 计划（被 npu-adaptation-tasks 取代）
│   ├── PORT-SUMMARY.md               ← 早期 playbook（被 PORT-GUIDE + SKILLS-GUIDE 取代）
│   ├── handoff-2026-04-19.md         ← (旧) 初版 handoff，被 HANDOVER.md 替代
│   └── codex-*.md                    ← codex 独立 review 存档
├── skills/                           ← 8 个可复用 skill
│   ├── npu-image-inspect/
│   ├── npu-code-path-sweep/
│   ├── npu-container-runner/
│   ├── upstream-branch-hygiene/
│   ├── ray-npu-shim/
│   ├── image-upgrade-drill/
│   ├── dep-gap-detect/               ← 自动判断新依赖是否需要 NPU 适配（P1 vs P2）
│   └── codex-review/
├── scripts/
│   ├── install-skills.sh             ← 部署 skills 到 ~/.claude/skills/
│   ├── fetch-upstream.sh             ← 拉 upstream 库（路径 2 用）
│   ├── run-npu-container.sh          ← NPU 容器 launcher（路径 1 + 2 都用）
│   ├── inspect-ascend-image.sh       ← image 体检（skill 内部）
│   ├── dep-gap-detect.sh             ← 依赖 A/B/C/D/E 自动分级（skill 内部）
│   ├── code-path-sweep.sh            ← CUDA-only 扫描（skill 内部）
│   ├── smoke_v11_device.py           ← V1.1/V1.2 smoke
│   ├── smoke_v13_rollout.py          ← V1.3 smoke
│   └── smoke/README.md               ← 完整 smoke 梯子索引
└── knowledge/
    ├── npu-patterns.md               ← **NPU 坑 stable ID 目录**
    ├── smoke-ladder-convention.md    ← smoke 命名约定
    ├── upstream-refs.md              ← image→upstream ref 对应表
    ├── easyr1-master-deps.md         ← EasyR1 requirements.txt 原始提取
    ├── verl-master-deps.md           ← veRL 对照（EasyR1 是 veRL 的精简 fork）
    ├── cann-9-0-x-install.md         ← CANN 9 安装笔记
    └── images/                       ← 每个目标 image 的 pip freeze + inventory

开发机额外有 ~/workspace/upstream/（跟 easyr1-npu 同级，不 track）：
    ├── EasyR1/                       ← zhshgmail/EasyR1 fork
    ├── verl/                         ← GPU 参考
    ├── torch-npu/
    ├── vllm-ascend/
    ├── triton-ascend/
    └── transformers/
```

每类文件的**归属规则 + 更新触发**见 [`docs/_meta/DOCS-CONVENTION.md`](docs/_meta/DOCS-CONVENTION.md)。新建文件前先查那里的 place-of-record map。

`upstream/` 不是本仓的一部分。走路径 2 的人用 `scripts/fetch-upstream.sh` 现拉现用，拉到 `../upstream/`（跟 `easyr1-npu/` 同级，不 track）。

---

## 前置条件

**路径 1（只跑 EasyR1）**：
- Ascend 910C (A3) NPU host（≥ 2 空闲 chip）
- NPU driver ≥ 25.5.0
- Docker ≥ 24
- ≥ 20 GB 空闲磁盘
- 能访问 HuggingFace（国内用 hf-mirror）

**路径 2 / 4（做或复现自动化移植）**：
- 路径 1 全部 +
- Claude Code CLI ≥ 2.0 安装好
- 开发机能访问 GitHub / GitCode（fetch-upstream.sh 用）
- `gh` CLI（`gh auth login` 登过）
- `codex` CLI（用于 `codex-review` skill）—— 可选

---

## 维护 + 文档 convention

**第一性原则**：README 是入口 + 索引，**不**往 README 塞具体内容。重要文档从 README 出发**都能**通过链接 2 跳以内可达。

完整规则见 [`docs/_meta/DOCS-CONVENTION.md`](docs/_meta/DOCS-CONVENTION.md)，贡献者（人 + agent）**必读**。简要摘要：

- **每类信息有唯一归属（single source of truth）**。完整 place-of-record map 见 DOCS-CONVENTION.md §1
  - 常见入口：坑目录 `knowledge/npu-patterns.md` · 日记 `docs/easyr1/porting-journal.md` · 状态 `docs/_meta/HANDOVER.md` · 适配任务 `docs/easyr1/npu-adaptation-tasks.md` · 依赖审计 `docs/easyr1/easyr1-dep-chain-audit.md`
- **什么时候更新什么**：见 DOCS-CONVENTION §2 的触发表
- **Language**：项目文档中文，代码 / commit / SKILL.md frontmatter 英文（详见 DOCS-CONVENTION §4）
- **Session 起手**：读 README → **NEXT-SESSION-STARTER** → HANDOVER → DOCS-CONVENTION（4 篇）就能接手。详见 DOCS-CONVENTION §7
- **术语对不上时**：所有 "Fix A/B/B+/C"、"Level 1-4"、"outcome A/B/C-patch/C-report"、"V1.x smoke rung"、"session-tag" 的**单一权威定义**在 [`docs/_meta/GLOSSARY.md`](docs/_meta/GLOSSARY.md)。如果本 repo 里某处术语和 GLOSSARY 冲突，以 GLOSSARY 为准。
- **当前 session 汇报入口**（任何查看状态从这里开始）：[`docs/_meta/WORKLOG.md`](docs/_meta/WORKLOG.md)——任务序列 + 当前状态 + V1.3/V1.4 debug backlog + 重构进度。**每完成一步或发现新问题都要更新**。
- **模块化 NPU 移植总览**：所有上游模块（transformers / torch_npu / vllm-ascend / vllm 等）的 port 状态、skill、trace branch、V1.3/V1.4 结果统一在 [`docs/_meta/MODULE-PORT-STATUS.md`](docs/_meta/MODULE-PORT-STATUS.md)。**每做完一次 port session 加一行**。
- **进行中 session 的工作计划 + 工作记录**：活跃 session 的 ground-truth log 在 `workspace/<session-tag>/PROGRESS.md`（**每完成一小步或发现新问题都要追加**），auto-compact 之后下一 session 要依赖它续跑。活跃 session 列表见 MODULE-PORT-STATUS §"活跃 session"。

commit message **不要**加 Claude 相关文字（`CLAUDE.md` 规定）。

---

## 相关项目

**姐妹项目**：[`ascend-fused-accuracy-probe`](https://gitcode.com/zhengshencn_hwca/ascend-fused-accuracy-probe)（private）—— 检测融合算子精度对齐。两个项目都跑在同一个 A3 host 上，互不干扰但共享一些硬件 / 软件事实知识。
