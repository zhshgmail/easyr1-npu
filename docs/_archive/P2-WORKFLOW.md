# P2 场景工作流 — EasyR1 引入"NPU 生态未覆盖"依赖时的端到端流程设计

> **⚠️ 本文档是"设计"，不是"已验证可用"**。
>
> 以下所有 workflow 都**尚未在真实 D ≥ 1 的场景下端到端执行过**（目前 D = 0）。tier-1 的子步骤大部分有历史先例（flash-attn shim、liger-kernel gate、try/except imports 等都真做过），但把它们当作"跟 skill 驱动的链路"端到端跑通**从未发生**。对一个需要 tier-2 kernel 委托的 gap 更是连单次实测都没有。
>
> 不要把本文档的存在当作 "P2 场景已闭环" 的证据。它只是**下次真遇到 D ≥ 1 时的出发点**。

**本文档回答**：当 `dep-gap-detect` skill 返回 **D ≥ 1**（有 CUDA-only dep 没有 NPU 替代）时，怎么从"发现 gap"推到"接回 EasyR1 能跑"。

**定位**：
- P1 工作流（无新适配需求）—— 见 [`SKILLS-GUIDE.md`](SKILLS-GUIDE.md) 9 步
- P2 工作流（本文档）—— 在 P1 基础上，插入 tier 1/2/3 适配工作 + 验收关卡
- 实际在 [`npu-adaptation-tasks.md`](npu-adaptation-tasks.md) 登记 track

> **Scope 的第一性原则**（见 [`README.md`](../README.md) 📣 callout 和 [`SKILLS-GUIDE.md §6`](SKILLS-GUIDE.md#6)）：EasyR1 需要的东西 NPU 没覆盖时，**不能用"不在 scope"绕过**。本文档就是把这个原则变成可执行流程。

---

## 1. P2 的触发

P2 场景被两种情况触发：

**A. 主动触发 —— 新 EasyR1 commit / 新 base image 时的 pre-flight**

走 P1 workflow 的 Step 1.5（`dep-gap-detect` skill）时，如果退出码 `1` (D ≥ 1)，**停在这里，进入 P2**。不要继续 Step 2 的 drill。

**B. 被动触发 —— 跑 smoke 时才发现**

少数情况 `dep-gap-detect` 漏报（例如包名经过 rename、或被 transitive 引入不在 `requirements.txt` 里），smoke 跑起来才报 `ImportError` / `ModuleNotFoundError` / `Op not supported on NPU`。这时候倒查依赖链，识别出 D 类 package，同样进入 P2。

---

## 2. 识别 + 分档（10-30 分钟）

对每个 D 类 package，按决策树决定 tier：

```
这个 package 是什么？
├── Python-only / wrapper（e.g. 一个 import 层的包装库）
│   └── 能在 EasyR1 源码里 shim / fork / try-except 绕过？
│       ├── 能 → Tier 1（本仓 shim 或 EasyR1 fork 修）
│       └── 不能（核心功能必须走这个 API） → 下一个分支
├── 依赖一个 NPU 上不存在的 kernel（fused op / custom CUDA kernel）
│   └── Tier 2（委托 kernel 项目）
├── 依赖 torch_npu C++ ATen 层缺失的 op
│   └── Tier 2（委托 Ascend PyTorch 团队 / 相关 kernel 项目）
├── 依赖 vllm-ascend / triton-ascend / transformers 的某个 NPU bug
│   └── Tier 1（Python 层 issue + workaround）或 Tier 3（如果是 C 层）
└── CANN runtime C 层 / driver 层 bug
    └── Tier 3（提 issue 给 Ascend 团队 + workaround）
```

每一条都**登记到 [`npu-adaptation-tasks.md`](npu-adaptation-tasks.md)**，格式见该文档顶部说明。

---

## 3. Tier 1 — 本仓直接做

**典型场景**：
- EasyR1 用了一个 CUDA-only 的辅助包（像 `flash-attn` 的 `bert_padding`、`liger-kernel` 的 triton loss），NPU 不需要这个包、或可以用 `torch.xxx` / `transformers.integrations.*` 替代
- 上游某个 Python 库有 NPU 相关 bug，workaround 可以在 EasyR1 代码里加
- 版本 compat shim（e.g. transformers 5 把 `no_init_weights` 搬家 → `try/except` import）

**Step-by-step**：

1. **在 EasyR1 fork `ascend-port` 分支上实现 shim / patch**。参考已有 pattern（`verl/utils/device.py`、`verl/utils/npu_flash_attn_utils.py`、`ray-npu-shim` skill）
2. **NPU-CP-NNN 命名**：如果是可复用 pattern（将来别的移植会遇到），在 [`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md) 加一条 stable ID
3. **更新 `dep-gap-detect.sh` 的 `PACKAGE_RULES`**：这个包从 D 挪到 C（CUDA-only 但本仓 shim 绕过）。下次自动判断就不会再报 blocker
4. **单元测试 / x86 side probe**（如果可行）：在本地 x86 上起 python，确认 shim 不破坏 GPU 路径（很多 shim 用 `if is_npu_available()` gate，要确保 CUDA 侧 return upstream 原逻辑）
5. **push `ascend-port`** → 上 A3 smoke 验证（V1.1 → V1.4）
6. **接回 P1 workflow**：D=0 后走 `image-upgrade-drill` Step 3 以后

**验收标准**：
- dep-gap-detect 重跑显示该包变 Tier C（不再是 D）
- V1.4 smoke 在目标 image 上 PASS（entropy_loss 在合理 band）
- （可选）20-step smoke 稳定

**已有 Tier-1 实例**（可作为模板）：
- `flash-attn.bert_padding` → 纯 torch 重写到 `verl/utils/npu_flash_attn_utils.py`（commit `da2487f`）
- `flash-attn` attention → `transformers.integrations.npu_flash_attention`（commit `fbaa983`）
- `liger-kernel` → 挪进 `[gpu]` extras，NPU 跳过（commit `7ee0f0b`）
- `no_init_weights` import 搬家 → try/except（commit `1f716ea`）
- `SamplingParams.eos_token_id` read-only property → `isinstance + fset is None` filter（commit `ecce71d`）

---

## 4. Tier 2 — 委托姐妹项目（kernel / ATen op 实现）

**典型场景**：
- EasyR1 的新 feature 要求一个 fused op（e.g. fused LayerNorm + activation），NPU 上不存在现成实现
- 上游推出了新模型架构需要自定义 CUDA kernel，NPU 要相应实现

**本仓的职责边界**：**识别 + 建任务 + track + 接回 EasyR1**，**不在本仓写 kernel**。

### 4.1 委托接口 — 姐妹项目做什么

| 项目 | 地址 | 做什么 | 不做什么 |
|---|---|---|---|
| `ascend-fused-accuracy-probe` | gitcode.com/zhengshencn_hwca/ascend-fused-accuracy-probe | A3 上**验证** fused op 的数值精度（跟参考实现比对） | kernel 从零实现（需要 kernel 源码已存在） |
| `a5_ops` | gitcode.com/zhengshencn_hwca/a5_ops | A5（910B、910_93）kernel **生成** —— 基于 AscendNPU-IR / ATC / 手工 AscendC | — |
| A3 kernel 专属项目 | （按需确认 / 建立） | A3（A3 生态 + 910B 最新 SoC）kernel 生成 | — |

### 4.2 Step-by-step

1. **在 [`npu-adaptation-tasks.md`](npu-adaptation-tasks.md) 建 T2-NNN 任务**，字段：
   - `Task`：具体 op 名称 + 数学描述（shape、dtype、broadcasting rules）
   - `Delegated to`：点名哪个姐妹项目
   - `Interface`：EasyR1 要调的 Python API（e.g. `from npu_ops import fused_xxx`）
   - `Acceptance`：EasyR1 集成后能跑什么 smoke
   - `Status`：OPEN / IN-PROGRESS (delegated) / VERIFIED / INTEGRATED

2. **在姐妹项目建对应 issue / PR**（格式按各自项目 convention）。本仓的 task 链接到那个 issue 的 URL，便于 cross-ref

3. **本仓并行准备 integration shim**：
   - 在 EasyR1 `ascend-port` 分支写一个 `verl/utils/<op>_shim.py` 先用 `torch.xxx` 兜底（性能差但能跑）
   - 标注 TODO 指向 T2-NNN
   - V1.4 smoke 能继续跑（虽然慢），不把整个项目阻塞在 kernel 完成前

4. **姐妹项目完成 kernel 后**：
   - 他们发 wheel / whl / 入 CANN 主仓
   - 本仓 `dep-gap-detect` 的 `PACKAGE_RULES` 加一条：`<new-op-pkg>=A:reason` 或 `=B`（取决于是原生 NPU 还是需要 import shim）
   - 替换兜底 shim 为真 kernel 调用
   - 数值 validation（跟 sister project 的 probe 数据对比，确保 EasyR1 这边调用姿势对）
   - V1.4 smoke 重跑，entropy_loss 对比：用真 kernel vs 兜底 shim，可以有轻微 drift 但不应该是数量级差别

5. **接回 P1 workflow**：跑 `image-upgrade-drill` 的 smoke ladder

### 4.3 验收标准

- 姐妹项目的 probe / CI 显示 op 数值精度达标（fp16 / bf16 绝对误差小于 tolerance band）
- `dep-gap-detect` 显示该 op 从 D 变 A/B
- EasyR1 集成 smoke（V1.4 + V2.2）PASS
- 如果有性能要求：一个 benchmark run 数据进 `docs/<op>-perf.md`

### 4.4 已有 Tier-2 实例

**当前空**。EasyR1 master 的依赖审计（[`easyr1-dep-chain-audit.md`](easyr1-dep-chain-audit.md)）显示 D=0，今天（2026-04-22）没有 active tier-2 任务。未来 EasyR1 升级 / 新模型需求出现时按上述流程建。

---

## 5. Tier 3 — 提需求给 Ascend 团队

**典型场景**：
- CANN runtime C 层 crash（像 `aclnnNonzero` 的 vector-core 错误 —— NPU-BUG-003 就属于这类）
- HCCL 多节点通信协议 bug
- Driver 层 / kernel module 本身 hang

**本仓的职责边界**：**识别 + 提 issue + 写 workaround + track + 等修**。不直接改 CANN / kernel module。

### 5.1 Step-by-step

1. **确认是 tier 3 而不是 tier 1/2**：问自己三个问题
   - 用户态有没有绕过方法？（e.g. 关一个 feature flag、换 API 调用方式）—— 有 → tier 1
   - Python 层能 wrapper？—— 能 → tier 1
   - 只能改 C 层？—— yes → tier 3

2. **做最小可复现**：写一个 `knowledge/bug-probes/<id>_probe.py`，在最小输入上复现 crash。这是 Ascend 团队最想看到的东西

3. **登记为 NPU-BUG-NNN stable ID**（[`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md)）
   - Symptom / Root cause（推测）/ Fix（= workaround 当前，上游修复后替换）/ Commit ref / Generalizable rule
   - 保留 probe 脚本路径的链接

4. **给 Ascend 团队的 channel**：
   - Ascend gitcode issue 仓（CANN / torch-npu / vllm-ascend / triton-ascend 各自仓的 issues）
   - 或内部 bug tracker（如果有 access）
   - 引用 NPU-BUG-NNN + 最小复现

5. **本仓写 workaround**：通常在 EasyR1 config 级别 gate 一个 risky feature（e.g. `use_torch_compile=false` 绕 BUG-003）

6. **Track 上游修复**：在 [`npu-adaptation-tasks.md`](npu-adaptation-tasks.md) 的 T3-NNN 记下：
   - 监控哪个上游版本可能修（e.g. "triton-ascend 3.3+ / torch_npu 2.10+"）
   - 一旦新版本发，重跑 probe 验证，修好了移除 workaround

### 5.2 验收（这种情况"完成"意味着）

- Workaround 上了 `ascend-port`，smoke PASS
- NPU-BUG-NNN stable ID 在 catalog 里
- 上游 issue 提了，URL 记在 adaptation-tasks.md
- 监控条件清晰（"上游发什么版本 / 改哪个 commit 后重测"）

### 5.3 已有 Tier-3 实例

- **T3-001 NPU-BUG-003**: triton-ascend inductor log_probs 数值损坏 → workaround `use_torch_compile=false`，等 triton-ascend 3.3+
- **T3-002 NPU-BUG-004**: upstream triton 3.6 + triton-ascend 3.2 共存 → Dockerfile 删 `backends/amd/nvidia/`，等 triton-ascend 对齐 upstream triton

---

## 6. 多档混合的情况

一个 D 类依赖可能牵涉多档：
- 先 Python shim 抢跑（tier 1）
- 同时委托 kernel 真实现（tier 2）
- 过程中踩到 runtime bug（tier 3）

**流程**：三档的 task **并行登记**，用 `blockedBy` 字段串起来。shim（tier 1）先跑让 EasyR1 能用；kernel（tier 2）做完后替换 shim；runtime bug（tier 3）单独 track。

示例 DAG：
```
T1-XXX (shim 兜底)  ─┐
                     ├─→ EasyR1 V1.4 smoke PASS（第一版）
T3-XXX (runtime)  ──┘
                          ↓  
T2-YYY (kernel 完成)  ──→ EasyR1 切换真 kernel → V1.4 smoke 二次验证
```

---

## 7. P2 workflow 的 skill 支撑

现有 skill 能覆盖 P2 的大部分步骤。缺的那些标记为 future skill：

| P2 步骤 | Skill | 状态 |
|---|---|---|
| 识别 gap | `dep-gap-detect` | ✅ 已有 |
| 分档决策 | — | 人工（本文档决策树）；future skill: `classify-gap-tier` |
| Tier 1 shim 实现 | 参考 `ray-npu-shim` / `npu-code-path-sweep` | ✅ 已有 |
| Tier 2 委托建 issue | — | 人工；future skill: `delegate-to-kernel-project` |
| Tier 3 bug probe | 参考 `image-upgrade-drill §5` 的 `bug003_probe` | ✅ 概念已有 |
| 登记 task | — | 人工（adaptation-tasks.md 格式）；future skill: `adaptation-task-register` |
| 接回 EasyR1 + smoke | `npu-container-runner` + smoke ladder | ✅ 已有 |

---

## 8. 触发点（下次遇到 P2 时怎么知道看这篇）

- `dep-gap-detect.sh` exit code 1 时，其输出 markdown 会引导到这里
- [`SKILLS-GUIDE.md §3 Step 1.5`](SKILLS-GUIDE.md) 的决策点 "D ≥ 1 → STOP, file task, adapt"
- [`PORT-GUIDE.md §7.5`](PORT-GUIDE.md) 的 FAQ "新依赖版本对不上怎么办"
- [`README.md 路径 4`](../README.md) "复现'EasyR1 + 新依赖自动移植'的流程"

这几处都已经写了，但没有专门文档承载 P2 完整 workflow —— **本文档填这个空**。

---

## 9. 尚未验证的部分

**本文档是设计**，不是实测过的流程：
- Tier 2 委托给姐妹项目的跨仓协调流程**在真实 D ≥ 1 场景下没跑过**（当前 D = 0）
- Tier 3 issue → Ascend 修复 → 本仓 workaround 移除这个完整环，我们只走过前半部分（BUG-003/004 都还是 OPEN）
- 多档混合的 DAG 调度**纯理论**

下次真遇到 D ≥ 1 时按本流程走，**同步修本文档**（哪一步跟预期不符、哪一步缺关键信息 / 工具）。

---

## 10. 相关文档

- [`SKILLS-GUIDE.md`](SKILLS-GUIDE.md) — P1 workflow（P2 的前置 + 后续都在 P1 里）
- [`npu-adaptation-tasks.md`](npu-adaptation-tasks.md) — 实际 task 登记和 track 的地方
- [`easyr1-dep-chain-audit.md`](easyr1-dep-chain-audit.md) — EasyR1 当前依赖 A/B/C/D/E 分档，确认 D=0 的数据依据
- [`knowledge/npu-patterns.md`](../knowledge/npu-patterns.md) — NPU-CP/BUG/ENV/OPS stable ID 目录
- [`skills/image-upgrade-drill/SKILL.md`](../skills/image-upgrade-drill/SKILL.md) — 升级演练 skill，包含 bug probe pattern
- [`skills/dep-gap-detect/SKILL.md`](../skills/dep-gap-detect/SKILL.md) — 自动判断 D 类 blocker 的 skill
