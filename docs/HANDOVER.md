# 项目交接（HANDOVER） — easyr1-npu + 依赖移植 skills

**给下一个 session / 另一个 dedicated agent 接手本项目时用**。只读这一篇文档 + `README`/`CLAUDE.md` 就能接着干。本文件补那些**不在其他 md 里、但下一个人必须知道的一次性 / stateful / 未解决事项**。

最新更新：**2026-04-23**（今天大幅扩展：Stage 2 三个 shim-adapt experts + Stage 3 两个 Day-0 experts + user-facing example guide + Day-0 reframing 进 design doc）。本次更新覆盖 §3 (git 分支)、§6 (未结工作)、§7 (skills 系统)、§11 (下一步)。上一版是 2026-04-19（Stage 0 only）。

**本轮 session 的关键 pivot**（user 2026-04-22T21:51）：原先 Stage 2 三个 upgrade experts 假设"NPU 已经 ship 好新版 image"——这只是**最简单的 shim-adapt 场景**。真问题是 **Day-0**：community 刚发版、NPU 生态没跟上，用户还想跑。新建 Stage 3 `transformers-day0-expert` + `vllm-day0-expert` 才是**真正的产品**。详见 [`design/SKILLS_ARCH_TARGET.md`](design/SKILLS_ARCH_TARGET.md) §Day-0 Reframing。

---

## 0. 一句话项目定位

把 **EasyR1（master tip，2026-04 月）** 移植到 **Ascend 910C (A3) NPU**，并沉淀一套**可复用的"GPU-RL 框架移植到 NPU" skills 库**——目的不只是把 EasyR1 跑起来，而是建立**未来 EasyR1 小版本升级 / 类似 RL 框架（OpenRLHF/TRL）移植**都能套用的系统化流程。

**姐妹项目**：`ascend-fused-accuracy-probe`（gitcode private，handover 在它的 `docs/HANDOVER.md`）——用来检测融合算子数值精度，和本项目平行互补。两者都被 user 同一个 Discord channel 跟踪。

---

## 1. 先读这两份（本文件是 transit state，不是布局 / convention）

- **仓库布局** → 见 [`../README.md`](../README.md)（"仓库布局" 段）。**HANDOVER 不再重复它**，布局改了改 README，不要回来改这里
- **每类文档归属 + 更新 convention** → 见 [`DOCS-CONVENTION.md`](DOCS-CONVENTION.md)
- **项目 instructions + working preferences** → 见 [`../CLAUDE.md`](../CLAUDE.md)

**本文件的职责**：**transit state** —— 当前分支 head、当前 A3 状态、未结工作、下一个 agent 要知道但不稳定的一次性事项。任何**稳定的** convention / 布局 / 设计规则不写在这里。

**ground rule**（见 `CLAUDE.md`）：**upstream 修改都走 git branch（`ascend-port`）**，**不**维护独立 patch 文件。

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

**更新 2026-04-23**：从 Stage 0 baseline（`ascend-port-round*`）扩到 Stage 2 和 Stage 3 experts 的 fixture / ascend-day0-* 分支。以下只列"还活着 / 下一 session 可能要参考"的分支。

```
* ascend-day0-vllm-vllm-day0-wetrun-20260423-0226  ← **vllm-day0 wet-run 当前分支**，
                                                     head a17d089。overlay image 已
                                                     build 在 A3，等 chips 释放可复
                                                     跑 V1.3/V1.4（见 §6.1）

  fixture/transformers-loosened-for-day0           ← transformers-day0 用的 fixture
                                                     (d448278)；upper bound 放到 <6.0
  fixture/vllm-loosened-for-day0                   ← vllm-day0 用的 fixture
                                                     (09456bf)；upper bound 放到 <0.20

  ascend-port-round4-20260422-0702                 ← **baseline-working v1 port ref**，
                                                     含 CP-001..006 + canonical smoke
                                                     config；Stage 2 experts 用作
                                                     UPSTREAM_REF，Stage 3 day0 experts
                                                     默认 OL-03 denylist 但允许 checkout
  ascend-port-round3-20260422-0514                 ← round 3 cold-drive 分支，image
                                                     easyr1-npu:round3-20260422-0514
                                                     保留在 A3（v1 baseline）

  ascend-port-transformers-upgrade                 ← **v2 drill 主分支**，head 2fd9337
  ascend-port-transformers-upgrade-reproduce       ← v2 drill 复现，head 8452b37

  ascend-port                                      ← v1 生产分支，ecce71d
  main                                             ← hiyouga/EasyR1 mirror @ dd71bbd
```

**关键 commit 指针**（下一 agent 的快速索引）：

| 分支 | Head | 含义 |
|---|---|---|
| `main` | `dd71bbd` | upstream master tip，NPU-未 port 原始状态 |
| `fixture/transformers-loosened-for-day0` | `d448278` | transformers-day0 fixture（1 行：`<5.0.0` → `<6.0.0`） |
| `fixture/vllm-loosened-for-day0` | `09456bf` | vllm-day0 fixture（1 行：加 `,<0.20.0`） |
| `ascend-port-round4-20260422-0702` | `367e313` | Stage 0 end-state，含完整 CP-001..006 + canonical smoke config，**易过 V1.1/V1.3/V1.4** |
| `ascend-port-day0-trans-trans-day0-wetrun-20260423-0109` | `?` | transformers-day0 wet-run 产物（5.6.0 on v2）— 分支在 A3，本地也有 |
| `ascend-day0-vllm-vllm-day0-wetrun-20260423-0226` | `a17d089` | vllm-day0 wet-run 产物（0.19.1 on v2）— smoke 待跑 |
| `ascend-port` | `ecce71d` | v1 生产分支 |
| `ascend-port-transformers-upgrade` | `2fd9337` | v2 drill（transformers 5.3.0.dev0） |

所有这些分支都 push 到 `personal` (zhshgmail/EasyR1)。

---

## 4. Smoke 测试梯子（**成功的里程碑**）

### V1.x（CANN 8.5.0 image，v1 生产）

| Smoke | 状态 | 含义 |
|---|---|---|
| V1.1 | ✅ | device 基本可用 |
| V1.2 | ✅ | ray + fsdp 初始化 |
| V1.3 | ✅ | rollout (vllm-ascend) |
| V1.4 | ✅ | 完整 GRPO 2-step 训练，step1 entropy_loss=0.991 → step2 1.263（8.5.0 image 基准） |
| V1.5 | ✅ | 4-chip multi-card HCCL |
| V2.1 | ✅ | padding_free=True on NPU（通过 transformers 的 `npu_flash_attention` integration） |
| V2.2 | ✅ | 4-chip + ulysses_size=2 + padding_free 全绿 |

### V2.x drill（CANN 8.5.1 image，升级演习）

| Smoke | 状态 | 含义 |
|---|---|---|
| drill 2-step | ✅ | 8.5.2 image 上 step1 entropy_loss=1.434（注意：8.5.2 基准 ≠ 8.5.0 基准——drill 报告里做 v2 自己的内部一致性比较，不是 v2↔v1 数值对比） |
| drill 20-step | ✅ | 全 20 步稳定（entropy_loss ∈ [1.31, 1.83]，grad_norm max ~3.2，no HCCL/vector core 错误） |
| BUG-003 probe | 挂 | 确认 BUG-003 在 CANN 8.5.1 **没修**（inductor 路径返回数值错误 + delayed crash） |

---

## 5. 知识库（`repo/knowledge/npu-patterns.md`）— 24 条 stable ID

catalog 每一条都有**统一 schema**：`Symptom / Root cause / Fix / Commit ref / Generalizable rule`。

| 类别 | 数量 | 示例 |
|---|---|---|
| `NPU-CP-NNN`（code patterns） | 7 | CP-001 torch.cuda.* 全面扫除；CP-007 npu_flash_attention integration |
| `NPU-BUG-NNN`（platform bugs） | 4 | BUG-001 triton-ascend 装残；BUG-003 inductor log_probs crash；BUG-004 triton 3.6 + triton-ascend 3.2 冲突 |
| `NPU-ENV-NNN`（env/config） | 4 | ENV-001 HF_ENDPOINT mirror；ENV-002 VLLM_ASCEND_ENABLE_NZ=0 |
| `NPU-OPS-NNN`（operational） | 9 | OPS-003 shared-host chip 争用；**OPS-006 docker daemon HTTP_PROXY 死掉**；OPS-007 base image 无 pip.conf；OPS-008 huaweicloud pypi mirror 不稳；**OPS-009 container 无法访问 NPU（host-level driver state 坏）** |

下一个 agent 遇到任何"这个 NPU 问题之前见过吗？"的时刻，**第一件事 grep `npu-patterns.md`**。

---

## 6. 未结工作（**重要——别漏**）

### 6.0 ⏸️ vllm-day0 wet-run smoke follow-up — **等 chips 释放**

**状态**：**OUTCOME B, Phase A/B/C done; Phase D BLOCKED on shared-host contention.** 这是今天（2026-04-23）最迫切的"半成品"。

- session: `vllm-day0-wetrun-20260423-0226`
- target: community vllm 0.19.1（发布于 2026-04-18，NPU 生态无 matching vllm-ascend）
- overlay image built: `easyr1-npu-vllm0191:vllm-day0-wetrun-20260423-0226` (sha e54c5d03f378, 28.1GB) **保留在 A3**
- branch: `ascend-day0-vllm-vllm-day0-wetrun-20260423-0226` (a17d089)，4 commits off fixture
- **Phase D (V1.3 + V1.4 smoke) 未跑**：另有 root session 在 A3 全部 16 chip 占 Ray workers。agent 按 OL-05 诚实 PARTIAL 退出

**怎么做 follow-up**（等 chips 释放时下一 session 接手）：
1. 确认 `npu-smi info -t proc-mem -i 0` 和 `-i 1` 返回 **没有** `Process id:` 行（chips idle）。目前有一个 Monitor-based watchdog 在 session 启动时 `bash /tmp/chip_watchdog.sh` 运行中，注意它可能已经 timeout 了——重开一个即可。
2. 直接用 `--reuse-image easyr1-npu-vllm0191:vllm-day0-wetrun-20260423-0226` 对照 session 用的 branch `ascend-day0-vllm-vllm-day0-wetrun-20260423-0226`，复跑 V1.3 + V1.4。**不要 rebuild**。
3. 预期 **outcome A-with-informational-note**（vllm 0.19.1 的 gpt_oss MoE triton kernel import error 非致命，Qwen2-0.5B 不碰）。V1.4 step-1 entropy_loss 应该和 vllm 0.18 baseline 差不多（在 v2 band [1.21, 1.34] 内）。
4. 跑完把结果回写 `workspace/vllm-day0-vllm-day0-wetrun-20260423-0226/PROGRESS.md` 的 Phase D/E 段（不在 git），然后 `git commit -m "vllm-day0 follow-up: V1.3/V1.4 smoke after shared-host release"` 到 `ascend-day0-vllm-...` 分支。

**Day-0 架构发现已记**（commit cdf4ff8）：fixture=master 的 day0 session 需要 OL-08 给 shim 文件 + smoke harness 权限。transformers-day0 / vllm-day0 双方已扩展。这条**不用再动代码**，只是要记得下次冷启 day0 session 时可能触发该路径。

### 6.1 v2 drill reproduce — **子 agent 卡住 8+ 小时**

我之前 spawn 了一个 isolated agent `ac31ee809fdc8dc57` 做"从零用 `image-upgrade-drill` skill 复现 v2 drill"的验证实验。它：

- ✅ 走完 Step 1-3（infra 预检、drill 分支 `ascend-port-transformers-upgrade-reproduce`、build image `easyr1-npu-852:drill-reproduce`）
- ✅ 撞上预期第一个 API break `no_init_weights`，log 在 `/tmp/z00637938/reproduce/logs/v22_reproduce_20260419_163542.log`
- 🟥 **然后 silent 卡住**。Harness 级问题（context 满 或 超时），不是 skill 问题

**怎么办**：
1. drill 分支 + image 都留着，接手的 agent 可以直接在主 session 里**从 `no_init_weights` 断点继续**——已知后续 2 步（修 fsdp_workers.py 的 import + SamplingParams patch）都有 commit 参考
2. 或者判决复现实验失败，在 `skills/image-upgrade-drill/SKILL.md` 补一条"**不要把全部 7 步丢给单个 isolated agent**，harness 会超时，要拆成人+agent 接力或分段子 agent"的 note
3. **不要清 `/tmp/z00637938/reproduce/` 和 `easyr1-npu-852:drill-reproduce`**——留着给下一次复现用

### 6.2 `ascend-port` 的两个 cherry-pick — 🟡 **2026-04-22 V1.4 smoke manually PASS on 8.5.0 + 8.5.2（这是一根 rung，不是完整 ladder）**

> 这个 section 的历史版本一度标成 "P1 端到端闭环 ✅"。**那是误导性表述**：我们只手工跑了 V1.4 一根 rung，没跑 V1.1 / V1.3 / V1.5 / V2.1 / V2.2，也没让 skill chain 冷启动驱动。真正的"端到端闭环"要 second actor 从 skill docs 复现出同样结果，那个 bar 还没到。见 `knowledge/npu-patterns.md#npu-ops-010` 和 memory `end_to_end_vs_described.md`。

`1f716ea` + `ecce71d` 两个 fix 在 **8.5.0 image 上 V1.4 smoke 实测 PASS**：
- entropy_loss step1 = **0.991**（exact match baseline）
- entropy_loss step2 = **1.263**（exact match baseline）
- Validation + checkpoint 都干净完成
- **"backward-compat 理论" → "backward-compat 实测"** ✅

实测过程中发现（也修了）两个 bug：

1. **`run-npu-container.sh` bind 配置漏了** —— 缺 `/usr/local/dcmi`、bind 整个 `/usr/local/Ascend/driver` 而不是 `lib64` 子目录、缺 `/etc/ascend_install.info`。Container DCMI 初始化失败，`dcmi model initialized failed -8020` → `npu get board type failed -9005` → Ray 报 `Total available GPUs 0`。修复见 commit **`b3f7a0f`**。
2. **2026-04-20 对 NPU-OPS-009 的 root cause 诊断是错的** —— 当时读 dmesg 里 `uda_occupy_dev_by_ns Conflict open udevid` 得出"僵尸 Ray raylet 锁 UDA namespace"结论，跑了 `device_hot_reset.sh` 试图修"驱动泄漏"，结果把 PCI 卡掉到 BIOS 都 enumerate 不回来，被迫 reboot。**真正的 root cause 是我们自己容器 bind 缺三个文件**（见上一条）。NPU-OPS-009 已重写，保留错误诊断作为 anti-pattern 教训。
3. **`--user` flag 必须显式传 `z00637938`**（ssh as root 时 `$USER=root`，script 不会自动 bind `/data/z00637938`）。文档补丁作为 UX TODO。

**2026-04-22 实测**：
- **8.5.0 image** (03:43 UTC): step1=0.991, step2=1.263 — exact match V1.4 baseline。Log `/tmp/z00637938/easyr1-logs/v14_regression_fixed2_20260422-034304.log`
- **8.5.2 drill image** (04:35 UTC): step1=1.275, step2=0.895 (grad_norm 2.07，合理 learning dynamics)。Drill 报告里只有 V2.2 config 数值（1.434），V1.4 config 在 8.5.2 上以前没 baseline —— 今天首次建立。Log `/tmp/z00637938/easyr1-logs/v14_drill_852_fresh_20260422-042253.log`。

踩过的坑（修完了）：
- **round 2 false positive**：smoke 从 #26 留下的 checkpoint 续跑（`Found latest checkpoint: .../global_step_2, will resume from it`），没做新训练。`rm -rf /tmp/z00637938/easyr1_smoke_ckpt` 再跑 round 3 OK
- **chips 2,3 + drill image 组合失败**（原因未查，chips 0,1 OK）—— 写进 HANDOVER 防下次

**复跑命令**（v1 image）：
```bash
cd /home/z00637938/workspace/easyr1-npu
rm -rf /tmp/z00637938/easyr1_smoke_ckpt
bash repo/scripts/run-npu-container.sh --chips 0,1 --user z00637938 \
  --image easyr1-npu:ascend-port \
  --live-source /home/z00637938/workspace/easyr1-npu/upstream/EasyR1 \
  -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

drill image 同样，`--image easyr1-npu-852:drill`。

### 6.3 A3 上易踩坑：`build_ascendc.py` 的 SoC 字符串 ≠ `acl.get_soc_name()`

本机 `acl.get_soc_name()` 返回 `Ascend910_9382`，但 `build_ascendc.py -v` 和 opp 目录用的是 `Ascend910B3` / `Ascend910_93`。CANN 9 的 `auto_tune/update_repository.py` 把 `Ascend910_9382` 映射到 short name `Ascend910_93`。**实战**：v2 drill 我用 `-v Ascend910_9382` 都编译过——但如果用户给的 kernel 不能，换 `Ascend910B3` 试试。

### 6.4 BUG-003 跨版本持续跟进

- CANN 8.5.0：直接 crash
- CANN 8.5.1（drill image）：silent 数值损坏 + delayed crash，**更糟**

**监控条件**：torch_npu 升到 2.10 或 triton-ascend 升到 3.3+ 后 reprobe。BUG-003 的 "Generalizable rule" 已经写了"启用 torch.compile 前要比 entropy_loss vs eager baseline"。

### 6.5 PORT-SUMMARY.md 升级里的 N 个 open follow-up（非阻塞）

详见 `PORT-SUMMARY.md` §"Known debt" 和 `transformers-upgrade-drill.md` §"Follow-ups status"。都标过状态。

---

## 7. Skills 系统 —— **2026-04-23 大改**

**重要 pivot**：skills 不再只是"手动 playbook"，演化成 **多 expert 架构 with orchestrator**。分两层：

### 7.1 Expert 架构（新）

目录布局：

```
src/experts/
├── _shared/                          ← cross-expert 模板层
│   ├── references/
│   │   └── ALWAYS_LOADED_UNIVERSAL.md  ← OL-01/02/04/04b/05/05b/06/07/09/10
│   ├── scripts/
│   │   ├── static_check.py           ← py_compile + dry-import + --container-import-image
│   │   └── cleanup_session.sh
│   ├── hooks/                        ← reference-copy starting points
│   ├── templates/
│   │   └── state_machine_skeleton.yaml
│   └── README.md                     ← "what's shared, how to fork new expert"
│
├── easyr1-expert/                    ← **Stage 0 — D=0 consumer port**
│                                       covers EasyR1 master/dd71bbd port in full
│                                       cold-drive-T0.3 validated 2026-04-22 round 4
│                                       (step-1 entropy_loss 0.958 in v1 band)
│
├── dep-analysis-expert/              ← Stage 1 — read-only dep classifier
│                                       emits P1/P2/P2a/P2b scenario + task_plan
│                                       routing. Version-aware since bb82ff0.
│
├── transformers-upgrade-expert/      ← Stage 2 — SHIM-ADAPT for stack bumps already
├── vllm-upgrade-expert/                shipped in a known NPU image (v1→v2 etc.)
├── torch-npu-upgrade-expert/
│
├── transformers-day0-expert/         ← **Stage 3 — THE REAL PROBLEM**
│                                       community release NOT YET shipped in NPU image
│                                       wet-run PASS 2026-04-23 on transformers 5.6.0
│                                       (outcome A; overlay 27GB preserved on A3)
├── vllm-day0-expert/                 ← **Stage 3 — 第 2 个 Day-0** 新建 2026-04-23
│                                       wet-run PARTIAL: Phase A/B/C PASS, Phase D
│                                       smoke blocked on shared chips (see §6.0)
│                                       Overlay 28GB preserved on A3.
│                                       torch-day0-expert 尚未建（user 说 torch 最易，低优先）

src/orchestrators/
└── npu-port/                         ← **全链 orchestrator**
                                        /npu-port --consumer-ref X --candidate-image Y
                                        → dep-analysis → (P2a shim-adapt | P2b day0)
                                        → consumer port → re-verify → RESULTS.md
                                        E2E wet-run PASS 2026-04-22 (fixture P2 chain)
```

每个 expert 都有：
- `README.md`（产品定义 + 何时用 / 何时不用）
- `SKILL.md`（/skill-name 入口 + workflow overview）
- `agent.md`（worker 详细 Phase A/B/C/D/E brief，含 watchdog-safe discipline）
- `state_machine.yaml`（G1/G2/G3 invariants + phase-specific rules）
- `references/ALWAYS_LOADED_RULES.md`（expert-specific OL-03 denylist + OL-08 edit scope）
- `references/KB_INDEX.md`（symptom → pattern/EC 路由）
- `references/patterns/domains/*.md`（per-domain deep knowledge）
- `references/ERROR_CORRECTIONS.md` / `PLATFORM_BUGS.md` / `SMOKE_BASELINE.md`（深度 expert 还有这些）
- `scripts/`（deploy_to_a3.sh / smoke_validate.sh / static_check.py / cleanup_session.sh 的 fork）
- `hooks/check_stop_worker.sh` + `check_edit_scope.sh`（G2/G3 / G1 enforcement）
- `SHARED_VERSION.txt`（pin 到 `_shared/` 的哪个 git sha）

### 7.2 关键 commit 指针（下一 session 接手时快速定位）

| 场景 | commit |
|---|---|
| `_shared/` 骨架 | 9b71ce4 |
| transformers-upgrade-expert（shim-adapt） | 6c063e3 |
| dep-analysis-expert | a27b7c9 |
| `/npu-port` orchestrator | 7d49a9d |
| vllm-upgrade-expert | 3b8a77d |
| torch-npu-upgrade-expert | 9b3e89d |
| **Day-0 reframing 进 design doc** | **88f3a3e** |
| transformers-day0-expert | 351269c |
| vllm-day0-expert | 27f2134 |
| Day-0 wet-run 后的 OL-08 refinement | cdf4ff8 |
| User-facing 5.6.0 example guide | 2f12f2e |

### 7.3 老的"skills 7+3"目录还在吗？

`repo/skills/` 目录里的 `image-upgrade-drill/` / `npu-code-path-sweep/` / `ray-npu-shim/` 等**仍存在但已被 experts 架构覆盖**：
- `image-upgrade-drill/` 的 playbook 内容现在是 transformers-upgrade-expert 的 pattern/domain 文件
- `npu-code-path-sweep/` 的工具脚本被 easyr1-expert 的 `code_path_sweep.sh` 吸收
- `ray-npu-shim/` 的 shim 代码被 easyr1-expert `patterns/domains/ray_integration.md` 吸收

**保留老 `repo/skills/`**：a) 文档里有引用；b) 下一 session 可能要做"从老 skills 提取剩余有用内容到 experts"的收尾工作。不要删，标为 `legacy` 即可。

**skill 目录约定**：每个 expert 一个 `SKILL.md`（必有），其它可选。skills 不 deploy 到 `~/.claude/skills/`（本项目没写 install.sh），因为是"本项目用"不是"跨项目复用"——和 `ascend-fused-accuracy-probe` 的 skills 部署模式不同。

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

## 11. 下一步候选工作 — 给 agent 的 top picks（2026-04-23 刷新）

按 "价值/可做性" 排序：

### P0（马上可以开；不需要 user 确认）

1. **vllm-day0 wet-run follow-up** — 等 A3 chips 0,1 释放，用 `--reuse-image easyr1-npu-vllm0191:vllm-day0-wetrun-20260423-0226` 复跑 V1.3 + V1.4 smoke 完成 session §6.0。预期 outcome A-with-note，~15min。这是**今天唯一显式的 BLOCKED**。
2. **Push 当前所有本地 commit 到远端**（如果 HANDOVER 更新后 session 结束忘了）。当前 `main` 最新 = `cdf4ff8`；`upstream/EasyR1` 上 `fixture/*` 和 `ascend-day0-vllm-*` 分支也都 push 到 `personal`。

### P1

3. **OL-05c 正式化**（从 §6.0 经验）：给 `_shared/references/ALWAYS_LOADED_UNIVERSAL.md` 加"共享 host chip 占用 > X 分钟 → PARTIAL exit + re-attempt recipe"子句。目前是散落在 vllm-day0 session 的手写 smoke-blocked.md 里。
4. **Minimal smoke harness 模板** — `transformers-day0-expert/references/patterns/domains/smoke-harness-minimal.md` + 同名 vllm-day0 版本。今天 vllm-day0 agent 当场手写了 minimal smoke scripts（because master 没有），模板化能省下一 session 重新发明。
5. **`/npu-port` Day-0 端到端 E2E** — 串 `dep-analysis → transformers-day0（或 vllm-day0）→ easyr1-port`。今天的 /npu-port E2E 只覆盖了 shim-adapt 路径（P2a）；Day-0 路径（P2b）的 task_plan 路由规则已加（cdf4ff8），但实跑没验证过。需要 fixture 分支 + chips free。~1.5h 预算。
6. **KB 补漏**：今天 vllm-day0 wet-run 的 4 个 commit（0187b7b / a0f5204 / a17d089 / Dockerfile.overlay-vllm0191）**还没进 vllm-day0 expert 的 SMOKE_BASELINE.md / ERROR_CORRECTIONS.md**。如果 §P0.1 follow-up 跑成功，顺便把 vllm 0.19.1 的 fresh baseline 写进 SMOKE_BASELINE.md。

### P2（等有 trigger / 有动机再做）

7. **torch-day0-expert** — user 说 torch 最简单（torch_npu 跟社区很紧），但目前没 trigger（torch 2.9 已在 v2 image）。等社区发 2.11 + NPU 没跟上时再建。今天**故意不 build**（见 SKILLS_ARCH_TARGET.md V3.0 "don't pre-build hypothetical"）。
8. **legacy `repo/skills/` 清理** — 老目录里的 image-upgrade-drill / npu-code-path-sweep / ray-npu-shim 内容已经迁到 experts；要么删要么明确标 `legacy/`。见 §7.3。

### P3（nice-to-have）

9. **image registry backup** — A3 上 5 个 custom images（v1/round3/v2/trans56/vllm0191）只存在本地 docker；机器坏了没备份。考虑推到 gitcode container registry 或 huawei cloud。
10. **EasyR1 upstream pull** — `git fetch origin main && git log ascend-port-round4-...|head`，看有没有新 commit 值得 rebase。最近已经 dd71bbd，不是紧急。
11. **BUG-003 long-running reprobe** — 等 triton-ascend 3.3 release。
12. **CANN 9.0 / torch_npu 2.10 release 跟进** — user 明确说过 v2 是验证演习、正式切 8.5.2+ 要 user 决策。等新 CANN 出再跑一轮 drill。

### 用户 2026-04-23 的长期方向

来自今天的 2 条 reframe：

- **"要解决 day 0 支持新版本社区软件的问题"**（user 2026-04-23T00:45）：skills 的核心价值在 Day-0 expert；shim-adapt 只是副产品。
- **"这个实例要记录下来，指导用户如何用 skills 0 交互完成"**（user 2026-04-23T02:06）：每次 Day-0 成功验证都要写 `docs/examples/<skill>-<trigger>.md`（transformers-5.6.0-day0.md 是首例）；user-facing 文档是 skill 的一部分，不是附件。

**对下一 session**：保持这两条优先级。不要回到"写更多 shim-adapt expert"的老路。

---

## 12. 交接 checklist — 下一个 agent 启动时做这几件事

1. [ ] `cat README.md`（仓根入口）
2. [ ] `cat docs/HANDOVER.md`（本文件）
3. [ ] `cat docs/DOCS-CONVENTION.md`（**必读**：每类信息归属到哪个文档，什么时候更新什么，避免你重新计划文档组织）
4. [ ] `cat CLAUDE.md`（项目根 instructions + working preferences）
5. [ ] 快速过 `docs/PORT-GUIDE.md`（v1 跑法）+ `docs/SKILLS-GUIDE.md`（重做移植流程）+ `docs/UPGRADE-DRILL-STATUS.md`（drill 状态）
6. [ ] 过 `knowledge/npu-patterns.md` 的 stable ID 标题（遇到 bug 能对号入座）
7. [ ] `cd upstream/EasyR1 && git log --oneline -5 ascend-port` + 看分支结构
8. [ ] `ssh -p 443 root@115.190.166.102 "npu-smi info | head -15"` 确认 host 可达 + 看 chip 占用
9. [ ] `docker ps -a` on A3 确认 container / image 状态（§2.1）
10. [ ] 看 Discord channel `1494825170399924366` 最近 20 条，了解用户关注点
11. [ ] 看姐妹项目 `~/workspace/ascend-fused-accuracy-probe/docs/HANDOVER.md`（同一 user，可能交叉问 probe 的事）
12. [ ] 把本文件 §6 的 open items + §11 候选工作转成自己的 TaskCreate list
13. [ ] 跟 user 同步当前状态 + 打算做什么

---

## 13. 关键信息变更时怎么更新

本文件**必须随状态更新**，不要让下一个人读到过期信息：

- 完成一个 §6 的 open item → 在 §6 画删除线 + 写入 git log 到"完成记录"
- 加一条 `npu-patterns.md` stable ID → §5 的数量表 +1
- A3 上清掉一个 image / container → §2.1 删一行
- 新 EasyR1 upstream 拉 → §3 的 commit 指针更新

**交接永远不结束**——每次 handoff 更新，下次接手的人就省一次"这文件还能信吗"的怀疑。
