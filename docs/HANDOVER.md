# 项目交接（HANDOVER） — easyr1-npu + 依赖移植 skills

**给下一个 session / 另一个 dedicated agent 接手本项目时用**。只读这一篇文档 + `README`/`CLAUDE.md` 就能接着干。本文件补那些**不在其他 md 里、但下一个人必须知道的一次性 / stateful / 未解决事项**。

最新更新：2026-04-19（初版，替换老 `handoff-2026-04-19.md`）

---

## 0. 一句话项目定位

把 **EasyR1（master tip，2026-04 月）** 移植到 **Ascend 910C (A3) NPU**，并沉淀一套**可复用的"GPU-RL 框架移植到 NPU" skills 库**——目的不只是把 EasyR1 跑起来，而是建立**未来 EasyR1 小版本升级 / 类似 RL 框架（OpenRLHF/TRL）移植**都能套用的系统化流程。

**姐妹项目**：`ascend-fused-accuracy-probe`（gitcode private，handover 在它的 `docs/HANDOVER.md`）——用来检测融合算子数值精度，和本项目平行互补。两者都被 user 同一个 Discord channel 跟踪。

---

## 1. 仓库布局（关键！读任何 md 前先理解这个）

```
~/workspace/easyr1-npu/
├── CLAUDE.md                              ← 本仓根的 instructions（必读）
├── repo/                                   ← 我们的 git-tracked deliverable
│   ├── docs/
│   │   ├── HANDOVER.md                    ← 本文件
│   │   ├── DELIVERABLE.md                 ← v1/v2 的正式 sign-off 文档
│   │   ├── PORT-SUMMARY.md                ← 移植 playbook（"下次怎么干"的 step-by-step）
│   │   ├── design.md                      ← 原始需求 + 任务拆解（大部分已落实）
│   │   ├── transformers-upgrade-drill.md  ← v2 drill 完整报告
│   │   ├── skills-design.md               ← skills 系统架构 V0.3
│   │   ├── porting-journal.md             ← 移植过程日志
│   │   ├── dep-matrix.md                  ← GPU↔NPU 依赖对照
│   │   ├── npu-gap-plan.md                ← 识别的 gap + 计划
│   │   ├── codex-signoff.md / v2.md       ← codex 独立 review 记录
│   │   ├── codex-review-skills-audit.md   ← skills 架构审计
│   │   └── handoff-2026-04-19.md          ← (旧) 初版 handoff，被本文件替代
│   ├── skills/                            ← 7 个可复用 skill
│   │   ├── npu-image-inspect/
│   │   ├── npu-code-path-sweep/
│   │   ├── npu-container-runner/
│   │   ├── upstream-branch-hygiene/
│   │   ├── codex-review/
│   │   ├── ray-npu-shim/
│   │   └── image-upgrade-drill/           ← v2 drill 沉淀出的新 skill
│   ├── scripts/                           ← skills 用到的 shell 脚本
│   │   ├── inspect-ascend-image.sh
│   │   ├── code-path-sweep.sh
│   │   └── run-npu-container.sh
│   └── knowledge/
│       ├── npu-patterns.md                ← **23 条 stable ID 目录**（CP×7 BUG×4 ENV×4 OPS×8）
│       ├── smoke-ladder-convention.md
│       ├── upstream-refs.md               ← 各 upstream 对应哪个 image 的 ref
│       └── images/                        ← `inspect-ascend-image.sh` 产出
├── upstream/                              ← 每个子目录都是独立 git clone + branch
│   ├── EasyR1/                            ← github.com/hiyouga/EasyR1（fork 到 zhshgmail）
│   ├── verl/                              ← GPU 参考对照
│   ├── torch-npu/                         ← gitcode Ascend/pytorch
│   ├── vllm-ascend/
│   ├── triton-ascend/
│   └── transformers/
└── (没有 src/ 顶层 Python 包——skills 是 skill md 格式，scripts 是 bash)
```

**ground rule**（见 `CLAUDE.md`）：**upstream 修改都走 git branch（`ascend-port`）**，**不** 维护独立 patch 文件。

---

## 2. 远端环境（A3 host）当前状态 — **不要重做**

**A3 host**：`ssh -p 443 root@115.190.166.102`（**SSH 端口 443**，非默认 22，key auth）
- x86_64 / openEuler 22.03 LTS / glibc 2.34 / kernel 5.10.0-60.18
- NPU driver 25.5.0.b050（ascendhal 7.35.23，Innerversion C23）
- **共享机器**：host 上有别人的 `chj_roll` conda env、`pedantic_gagarin` container；**别动**，不争 chip 0 以外的 NPU
- `/var/lib/docker` 占 93%——装新 image 前先看 `docker images | sort -k 7` 清冗余
- A3 在 **GFW 内**（见 memory `a3_is_firewalled.md`），本地到 A3 走 SSH 跳板；web 爬在本地做，git sync 到 A3

### 2.1 已就绪的 Docker image（`docker images` 结果）

| Image tag | 用途 |
|---|---|
| `quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest` | **v1 生产 image**（14 GB），CANN 8.5.0 + torch_npu 2.8 + transformers 4.57 + vllm_ascend 0.13 |
| `quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5` | **v2 drill image base**（24 GB），CANN 8.5.1 + torch_npu 2.9 + transformers 5.3.0.dev0 + vllm 0.18 |
| `quay.nju.edu.cn/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5` | **同上，NJU CN 镜像** retag 而来 |
| `easyr1-npu:ascend-port` | v1 可直接跑的 image |
| `easyr1-npu-852:drill` | **v2 drill 成功 image**（含 triton-ascend force-reinstall），用于验证 transformers-5 / vllm-0.18 兼容 |
| `easyr1-npu-852:drill-reproduce` | v2 drill 复现用 image（子 agent 卡在 `no_init_weights` 没跑完；详见 §6） |

### 2.2 docker daemon 特殊配置（**很重要**）

A3 的 docker daemon **默认走一个代理** `100.66.1.4:7897`，这个代理时灵时不灵。已经在 `/etc/systemd/system/docker.service.d/http-proxy.conf` 的 `NO_PROXY` 加了 `quay.nju.edu.cn` 能直连。如果后续要 pull 新 registry 的 image 卡住，先看 `NPU-OPS-006`。

### 2.3 A3 文件系统关键路径

- `/home/z00637938/workspace/easyr1-npu/upstream/EasyR1/` — git clone 存在，v1 branch 和 v2 drill branch 都有
- `/tmp/z00637938/easyr1-logs/` — 所有 smoke run 的 log（V1.1 → V2.2 + drill）
- `/tmp/z00637938/easyr1_smoke_ckpt*/` — V1.x/V2.x smoke 产出的 FSDP checkpoint（几 GB 到几十 GB）
- `/tmp/z00637938/reproduce/` — v2 drill **复现用**的 workspace（子 agent 留下的残局，详见 §6）
- `/tmp/drill_launch.sh` — v2 drill 启动脚本
- `/data/z00637938/` — 用户约定的大文件存放地（models / weight / hf-cache）
- `/data/z00637938/hf-cache/` — HuggingFace cache，通过 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载

### 2.4 A3 chip 使用约定

16 个 chip（0..15），我们的 smoke 默认用 **`0,1,2,3`**。跑之前务必 `npu-smi info` 看 HBM-Usage、AICore%——别人正在用的 chip 别抢。

---

## 3. Git 分支状态（在 `upstream/EasyR1/` 内）

```
* ascend-port-transformers-upgrade-reproduce     ← 子 agent 冷启动 v2 drill 复现用的分支，head 8452b37
  ascend-port-transformers-upgrade               ← **v2 drill 主分支**，head 2fd9337 已 PASS
  ascend-port                                    ← **v1 生产分支**，两个 backward-compat 修复已合入
  main                                           ← upstream hiyouga/EasyR1 的 mirror
  remotes/origin/*                               ← 对应 hiyouga
  remotes/personal/*                             ← 对应 zhshgmail（我们的 private fork）
```

**关键 commit 指针**（记一下免得下一个 agent 每次 git log）：

| 分支 | Head | 含义 |
|---|---|---|
| `ascend-port` | `ecce71d` | v1 完整 + 两个 drill-origin backward-compat fix |
| `ascend-port-transformers-upgrade` | `2fd9337` | v2 drill 20-step smoke 全绿的状态 |
| v2 drill 关键 commit | `55bb730` | `no_init_weights` 从 `modeling_utils` 搬到 `initialization` 的兼容 patch |
| v2 drill 关键 commit | `d213f01` | vllm 0.18 `SamplingParams.eos_token_id` 只读 property 跳过 |
| v2 drill 关键 commit | `63ae428` | NPU-BUG-003 probe 脚本（`use_torch_compile=true`） |
| v2 drill 关键 commit | `a18d1f8` | NPU-BUG-004 fix（删 triton 的 amd/nvidia backends） |

v1 的两个修复（`1f716ea` + `ecce71d`）已经 cherry-pick 到 `ascend-port`，也已经 push 到 `personal/ascend-port`。

---

## 4. Smoke 测试梯子（**成功的里程碑**）

### V1.x（CANN 8.5.0 image，v1 生产）

| Smoke | 状态 | 含义 |
|---|---|---|
| V1.1 | ✅ | device 基本可用 |
| V1.2 | ✅ | ray + fsdp 初始化 |
| V1.3 | ✅ | rollout (vllm-ascend) |
| V1.4 | ✅ | 完整 GRPO 2-step 训练，entropy_loss=1.434 成为**基准** |
| V1.5 | ✅ | 4-chip multi-card HCCL |
| V2.1 | ✅ | padding_free=True on NPU（通过 transformers 的 `npu_flash_attention` integration） |
| V2.2 | ✅ | 4-chip + ulysses_size=2 + padding_free 全绿 |

### V2.x drill（CANN 8.5.1 image，升级演习）

| Smoke | 状态 | 含义 |
|---|---|---|
| drill 2-step | ✅ | **entropy_loss=1.434 完全匹配 V1.4 基准** |
| drill 20-step | ✅ | 全 20 步稳定（entropy_loss ∈ [1.31, 1.83]，grad_norm max ~3.2，no HCCL/vector core 错误） |
| BUG-003 probe | 挂 | 确认 BUG-003 在 CANN 8.5.1 **没修**（inductor 路径返回数值错误 + delayed crash） |

---

## 5. 知识库（`repo/knowledge/npu-patterns.md`）— 23 条 stable ID

catalog 每一条都有**统一 schema**：`Symptom / Root cause / Fix / Commit ref / Generalizable rule`。

| 类别 | 数量 | 示例 |
|---|---|---|
| `NPU-CP-NNN`（code patterns） | 7 | CP-001 torch.cuda.* 全面扫除；CP-007 npu_flash_attention integration |
| `NPU-BUG-NNN`（platform bugs） | 4 | BUG-001 triton-ascend 装残；BUG-003 inductor log_probs crash；BUG-004 triton 3.6 + triton-ascend 3.2 冲突 |
| `NPU-ENV-NNN`（env/config） | 4 | ENV-001 HF_ENDPOINT mirror；ENV-002 VLLM_ASCEND_ENABLE_NZ=0 |
| `NPU-OPS-NNN`（operational） | 8 | OPS-003 shared-host chip 争用；**OPS-006 docker daemon HTTP_PROXY 死掉**；OPS-007 base image 无 pip.conf；OPS-008 huaweicloud pypi mirror 不稳 |

下一个 agent 遇到任何"这个 NPU 问题之前见过吗？"的时刻，**第一件事 grep `npu-patterns.md`**。

---

## 6. 未结工作（**重要——别漏**）

### 6.1 v2 drill reproduce — **子 agent 卡住 8+ 小时**

我之前 spawn 了一个 isolated agent `ac31ee809fdc8dc57` 做"从零用 `image-upgrade-drill` skill 复现 v2 drill"的验证实验。它：

- ✅ 走完 Step 1-3（infra 预检、drill 分支 `ascend-port-transformers-upgrade-reproduce`、build image `easyr1-npu-852:drill-reproduce`）
- ✅ 撞上预期第一个 API break `no_init_weights`，log 在 `/tmp/z00637938/reproduce/logs/v22_reproduce_20260419_163542.log`
- 🟥 **然后 silent 卡住**。Harness 级问题（context 满 或 超时），不是 skill 问题

**怎么办**：
1. drill 分支 + image 都留着，接手的 agent 可以直接在主 session 里**从 `no_init_weights` 断点继续**——已知后续 2 步（修 fsdp_workers.py 的 import + SamplingParams patch）都有 commit 参考
2. 或者判决复现实验失败，在 `skills/image-upgrade-drill/SKILL.md` 补一条"**不要把全部 7 步丢给单个 isolated agent**，harness 会超时，要拆成人+agent 接力或分段子 agent"的 note
3. **不要清 `/tmp/z00637938/reproduce/` 和 `easyr1-npu-852:drill-reproduce`**——留着给下一次复现用

### 6.2 `ascend-port` 的两个 cherry-pick 已 push 但**未经过生产 image 实测**

`1f716ea` + `ecce71d` 两个 fix 是在 drill image 上写的，backward-compat 理论上 8.5.0 也 ok——但 **没在 8.5.0 container 里重跑过 V2.2 smoke 确认没 regression**。P1，建议补。

### 6.3 A3 上易踩坑：`build_ascendc.py` 的 SoC 字符串 ≠ `acl.get_soc_name()`

本机 `acl.get_soc_name()` 返回 `Ascend910_9382`，但 `build_ascendc.py -v` 和 opp 目录用的是 `Ascend910B3` / `Ascend910_93`。CANN 9 的 `auto_tune/update_repository.py` 把 `Ascend910_9382` 映射到 short name `Ascend910_93`。**实战**：v2 drill 我用 `-v Ascend910_9382` 都编译过——但如果用户给的 kernel 不能，换 `Ascend910B3` 试试。

### 6.4 BUG-003 跨版本持续跟进

- CANN 8.5.0：直接 crash
- CANN 8.5.1（drill image）：silent 数值损坏 + delayed crash，**更糟**

**监控条件**：torch_npu 升到 2.10 或 triton-ascend 升到 3.3+ 后 reprobe。BUG-003 的 "Generalizable rule" 已经写了"启用 torch.compile 前要比 entropy_loss vs eager baseline"。

### 6.5 PORT-SUMMARY.md 升级里的 N 个 open follow-up（非阻塞）

详见 `PORT-SUMMARY.md` §"Known debt" 和 `transformers-upgrade-drill.md` §"Follow-ups status"。都标过状态。

---

## 7. Skills 系统（本项目的第二个交付物）

7 个 shipped skill + 3 个 deferred。完整表见 `repo/docs/skills-design.md`（V0.3）的 status table。

**最重要 / 最通用的三个**（下一次 EasyR1 小升级 / RL 框架移植都会用到）：

1. **`skills/image-upgrade-drill/`** — v2 drill 沉淀出的 7 步 playbook。下次 CANN 8.5.3 / 9.0 正式出了，照着跑就能验证兼容。**第一手实证**在 `docs/transformers-upgrade-drill.md`
2. **`skills/npu-code-path-sweep/`** — scan 一个 GPU-only codebase 找 `torch.cuda.*` / `"cuda"` 字符串 / `"nccl"` / `flash_attn` 这类 callsite。产出 `docs/code-path-sweep-<framework>.md`，每个 hit 带修复建议
3. **`skills/ray-npu-shim/`** — drop-in 的 Python 模块解决 Ray 不识别 NPU、`ASCEND_RT_VISIBLE_DEVICES` 被 Ray 2.55 清掉的问题（NPU-CP-003 + NPU-BUG-002 + NPU-ENV-002 三合一）

**skill 目录约定**：每个 skill 一个 `SKILL.md`（必有），可选 `scripts/`、`references/`。skills 不 deploy 到 `~/.claude/skills/`（本项目没写 install.sh），因为是"本项目用"不是"跨项目复用"——和 `ascend-fused-accuracy-probe` 的 skills 部署模式不同。

---

## 8. 工具链 + 环境依赖（需要什么 / 在哪）

### 本地（开发 / scan / web 爬）

- `gh` 已登录，用于 github / 拿 release
- `gc` 已登录（`zhengshencn_hwca`），用于 gitcode（NPU 仓主要在 gitcode）
- `docker` 可用
- `codex` CLI 已装（`codex exec --dangerously-bypass-approvals-and-sandbox`），用于独立 review。见 memory `codex_review_usage.md`——**必须** 加 bypass sandbox 否则 codex 跑不动
- `dev-browser` skill 装了（`/home/z00637938/.claude/plugins/cache/dev-browser-marketplace/...`），用于抓 JS-rendered 页（hiascend.com 所有页面都 JS-rendered，中文版才有完整内容）

### A3（smoke 跑 / kernel 编译）

见 §2。**一切关键操作在 `run-npu-container.sh` 启动的 container 里做**，直接在 host 上改东西会污染别的 user。

---

## 9. 姐妹项目 `ascend-fused-accuracy-probe` 关系

2026-04-19 user 让我创建了新项目 `ascend-fused-accuracy-probe`（gitcode private `zhengshencn_hwca/ascend-fused-accuracy-probe`），用来**检测融合算子精度对齐**。本项目和它的关系：

- **平行**：probe 和 easyr1-npu 各管各的，但都跑在**同一个 A3 host**上
- **共享知识**：probe 的 `knowledge/a3-a5-soc-mapping.md`、`cann-9-0-x-install.md`、`ascend-doc-source-gotchas.md` 和本项目的 `npu-patterns.md` 是同一套硬件/软件事实的不同切片。两个项目的 knowledge **互相引用**
- **A3 资源共享**：probe 在 A3 上留了个 `afap-cann9` container；easyr1-npu 在 A3 上有 `easyr1-npu-852:drill*` image 和 `/tmp/z00637938/easyr1-logs`。**互不干扰，但下一个 agent 要清 A3 时小心别误删别的项目的**

user 在**同一个 Discord channel**（`1494825170399924366`）跟踪**两个项目**。有 probe 相关的 issue，**应该引导到 probe 项目的 HANDOVER.md**。

---

## 10. 不在任何 md 里但下一个 agent 要知道的 user 偏好

来自 user 的 memory + 历次对话（都已加到 `~/.claude/projects/-home-z00637938-workspace-easyr1-npu/memory/*.md`，但归拢在这里一并列）：

- **commit message**：**不要** 加 `Co-Authored-By: Claude` / `🤖 Generated with Claude`。全局 `~/.claude/CLAUDE.md` 规则
- **项目级文档默认中文**：README、knowledge、findings、设计文档都中文；**代码注释、commit message、SKILL.md frontmatter `description` 保持英文**（机器可读面稳定）。见 memory `project_docs_language.md`
- **Discord 主通信**：chat_id `1494825170399924366`。终端用户看不到，milestone / 要等答复的都必须 Discord。不要 > 15 分钟不更新（memory `discord_cadence.md`）
- **工具能用 ≠ 任务能做成**：不要声称"我用 X 能 Y"没 end-to-end 实测。见 `ascend-fused-accuracy-probe/knowledge/tool-capability-realism.md`（姐妹项目的），本项目也适用
- **不要给 A/B/C 选项等 user 拍板**：当依赖链明显时自己推进。见 memory `decision_patterns_to_avoid.md`
- **version-aware reviews**：review NPU 上游时**不能只看 master**，要看目标 image 对应的 branch/tag。见 memory `version_aware_reviews.md` + `repo/knowledge/upstream-refs.md`
- **codex review** 在共享 host 必须加 `--dangerously-bypass-approvals-and-sandbox`（bubblewrap 会 block）。memory `codex_review_usage.md`
- **A3 是共享机器**、**在 GFW 内**——web 工作在本地、git sync 到 A3。memory `a3_server.md` + `a3_is_firewalled.md` + `a3_docker_proxy.md`

---

## 11. 下一步候选工作 — 给 agent 的 top picks

按 "价值/可做性" 排序：

### P0（马上可以开）

1. **为 v2 drill reproduce 补结论**：要么修复 isolated agent harness 限制（拆成分段 agent），要么正式判"single-agent 复现失败，手动分步走"。更新 `image-upgrade-drill/SKILL.md` 的 "When not to use" 一段，记录这个已知局限
2. **在 8.5.0 生产 image 回归测 `ascend-port` 的两个 cherry-pick**：跑一次 V1.4 + V2.2 smoke，确认 `1f716ea`（`no_init_weights` try/except）和 `ecce71d`（SamplingParams 只读 property 跳过）**在 transformers 4.57 / vllm 0.13 下不 regress**。**30 分钟**的工作，但项目完整性关键

### P1

3. **EasyR1 upstream pull**：hiyouga/EasyR1 master 可能又有新 commit。`git fetch origin main && git log ascend-port..origin/main --oneline` 看要不要 rebase 或 merge
4. **补齐 "RL 框架移植" skill 文档**：现在 skills 都聚焦单次 EasyR1 port；下一次要移植 OpenRLHF / TRL 时要用。考虑在 `skills-design.md` 里加一节 "for arbitrary Ray-based RL frameworks"

### P2

5. **跟进 CANN 9.0 / torch_npu 2.10 release**：user 明确说过 v2 是验证演习、正式切 8.5.2 要 user 决策。等新 CANN 出再跑一轮 drill
6. **BUG-003 long-running reprobe**：等 triton-ascend 3.3 release

### P3（nice-to-have）

7. **把 A3 上的 `easyr1-npu-852:drill*` image 推到某个可共享的 private registry**（gitcode container registry？huawei cloud？）——目前只存在 A3 本地 docker，机器坏了没备份
8. **skills-design.md 里 deferred 的几个 skill**（`dep-diff`、`version-align`、`gap-plan`）：如果下次移植发现手动做这些很痛就动工；否则留着

---

## 12. 交接 checklist — 下一个 agent 启动时做这几件事

1. [ ] `cd ~/workspace/easyr1-npu && cat repo/docs/HANDOVER.md`（本文件）
2. [ ] `cat CLAUDE.md`（项目根 instructions）
3. [ ] 快速过 `repo/docs/PORT-SUMMARY.md`（playbook）+ `repo/docs/transformers-upgrade-drill.md`（v2 实证）
4. [ ] 过 `repo/knowledge/npu-patterns.md` 的 23 条 stable ID 标题（遇到 bug 能对号入座）
5. [ ] `cd upstream/EasyR1 && git log --oneline -5 ascend-port` + `git log --oneline -5 ascend-port-transformers-upgrade`
6. [ ] `ssh -p 443 root@115.190.166.102 "npu-smi info \| head -15"` 确认 host 可达 + 看 chip 占用
7. [ ] `docker ps -a` on A3 确认 container / image 状态（§2.1）
8. [ ] 看 Discord channel `1494825170399924366` 最近 20 条，了解用户关注点
9. [ ] 看姐妹项目 `~/workspace/ascend-fused-accuracy-probe/docs/HANDOVER.md`（同一 user，可能交叉问 probe 的事）
10. [ ] 把本文件 §6 的 open items + §11 候选工作转成自己的 TaskCreate list
11. [ ] 跟 user 同步当前状态 + 打算做什么

---

## 13. 关键信息变更时怎么更新

本文件**必须随状态更新**，不要让下一个人读到过期信息：

- 完成一个 §6 的 open item → 在 §6 画删除线 + 写入 git log 到"完成记录"
- 加一条 `npu-patterns.md` stable ID → §5 的数量表 +1
- A3 上清掉一个 image / container → §2.1 删一行
- 新 EasyR1 upstream 拉 → §3 的 commit 指针更新

**交接永远不结束**——每次 handoff 更新，下次接手的人就省一次"这文件还能信吗"的怀疑。
