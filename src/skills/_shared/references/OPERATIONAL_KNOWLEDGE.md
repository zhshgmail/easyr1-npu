# OPERATIONAL_KNOWLEDGE — OL-XX catalog with grep keywords

> 跨 skill 的 OL-XX（Operational Lessons）总索引。每行 = 一个 OL + 关键词 + 触发场景 + 详情位置。
>
> **使用方法**：worker 遇到问题时，先 `grep -i '<keyword>' OPERATIONAL_KNOWLEDGE.md`，找到匹配的 OL 再 load 对应文件读细节。
>
> **维护规则**：
> 1. OL-XX 分两类：
>    - **universal**（所有 day-0 expert 适用）：详情写在 `ALWAYS_LOADED_UNIVERSAL.md`
>    - **expert-specific**（特定 expert 适用）：详情写在 `<expert>/references/ALWAYS_LOADED_RULES.md`
> 2. 新增 OL：先在本文件加一行，再写详情文件
> 3. OL-XX 编号一旦发出**不可复用**（即使废弃）；废弃的 OL 加 `⚠ deprecated` 标记保留
>
> 最后更新：2026-05-15（T29.2 首次成文）。

---

## Universal OL（所有 day-0 / port expert 适用）

详情见 [`ALWAYS_LOADED_UNIVERSAL.md`](ALWAYS_LOADED_UNIVERSAL.md)。

| OL | Title | Keywords / Aliases | When |
|---|---|---|---|
| **OL-01** | 代码改完必过 py_compile | py_compile, static_check, SyntaxError, agent 幻觉, exit 0 | 写任何代码修改后，Phase C 完成前 |
| **OL-02** | 声称 PASS 必须有 log 路径 + 关键数值 | provenance, log path, entropy_loss, reward_score, evidence | 任何 outcome A/B/C 声明 |
| **OL-03** | denylist：不读历史文档（每 expert 自定义） | cold-drive, honest replay, do-not-read, fork branch denylist | Phase A 之前 |
| **OL-04** | 使用 session-specific image tag | image tag, session tag, do not reuse, ascend-day0-<session> | 任何 docker build / pull |
| **OL-04b** | Session 结束前清理临时 image / container | cleanup, docker rmi, orphan, disk space | Session end |
| **OL-05** | A3 共享 host，任何 NPU 动作前 precheck | npu-smi info, chip occupancy, shared host, 抢 chip | 任何用 chip 动作前 |
| **OL-05b** | 用最少数量的 NPU | minimal chip, 2-chip default, V1.4 vs V1.5 | 任何 V1.x smoke 启动 |
| **OL-06** | A3 github 不通（GFW），用 scp / bundle 同步 | GFW, github, scp, git bundle, A3 firewall | A3-side git operations |
| **OL-07** | Huaweicloud pypi 不稳，优先 aliyun | pypi mirror, aliyun, timeout, hf-mirror | pip install on A3 |
| **OL-08** | edit scope 白名单（每 expert 自定义） | edit scope, allowed paths, agent territory | 任何 Edit / Write 操作 |
| **OL-09** | 所有 claim 带 provenance | provenance, commit ref, file path, line range | 写 PR_MATERIAL.md / handover / outcome |
| **OL-10** | 失败 ≠ 乱改，先 ERROR_CORRECTIONS 分类 | EC, error classification, root cause, do not patch over | 编译/运行失败时 |
| **OL-11** | PyTorch 2.11+ docker build 内不能 import torch | _import_device_backends, CANN libs, py_compile only at build, NPU-OPS-007 | Day-0 涉及 torch 2.11+ overlay 构建 |
| **OL-12** | `VLLM_BATCH_INVARIANT` module-import-time cached | env var, vllm 0.18, set before import vllm, plugin init order | vllm-ascend 触发 |
| **OL-13** | `ASCEND_RT_VISIBLE_DEVICES` 是容器内 index，非 host phy-id | container index, host phy-id, --device /dev/davinciN, NPU-OPS-012, T25.5 helper bug | 任何 NPU 容器启动 |
| **OL-14** | SSH-as-root 必传 `NPU_USER=<workspace-owner>` | $USER=root, /home/root empty, HFValidationError, NPU-OPS-013 | A3 ssh -p 443 root@... |
| **OL-15** | A3 host 上 `repo/` 必须是 git clone | NPU-OPS-014, stale v0 layout, repo/scripts/ vs repo/src/scripts/, hand-copy | 任何 SKILL 命令 cite `repo/...` 路径 |
| **OL-16** | Outcome 分类不可越级声明 | A / A-with-note / B / C-patch / C-report, outcome ladder, PR_MATERIAL | 任何 day-0 完成声明 |
| **OL-17** | fork-branch 命名约定 `ascend-port/<target-version-slug>` | ascend-port, target version slug, lowercase hyphen, branch convention | 推送任何 fork branch |
| **OL-18** | walk-through PASS ≠ 真证据 | cross-layer-007, agent walk-through, plan vs artifact, on-A3 import smoke | 任何 outcome 声明前 |
| **OL-32** | NPU 容器 bind set 单一权威源 | bind set canonical source, npu-container-runner SKILL.md, /dev/devmm_svm, /dev/hisi_hdc, /dev/davinci_manager, NPU-OPS-009/011/012/013/014, DEBT-2 | 写任何 docker run / bind path 时；其他 doc 必须 cite npu-container-runner |

---

## Expert-specific OL

每 expert 在自己的 `<expert>/references/ALWAYS_LOADED_RULES.md` 里定义。

### vllm-ascend-day0

详情见 [`vllm-ascend/port-expert/references/ALWAYS_LOADED_RULES.md`](../../vllm-ascend/port-expert/references/ALWAYS_LOADED_RULES.md)。

| OL | Title | Keywords |
|---|---|---|
| OL-03 | vllm-ascend denylist | vllm-ascend cold-drive, prior PR_MATERIAL |
| OL-08 | vllm-ascend edit scope | `vllm_ascend/compat/`, `vllm_ascend/{_310p,ops,spec_decode}/`, 不动 community vllm |
| OL-19 | plugin-init lazy import（F2-path-move）| find_spec, lazy __getattr__, vllm.v1.sample.*, vllm.v1.spec_decode.*, plugin init order, vllm-ascend-003 |

### torch-npu-day0

详情见 [`torch-npu/port-expert/references/ALWAYS_LOADED_RULES.md`](../../torch-npu/port-expert/references/ALWAYS_LOADED_RULES.md)。

| OL | Title | Keywords |
|---|---|---|
| OL-03 | torch-npu denylist | torch-npu cold-drive, prior PR_MATERIAL |
| OL-08 | torch-npu edit scope | `torch_npu/compat/`, 不动 community torch core |
| OL-20 | F7/F8 验证不能只看 AST，要 grep call sites | check_f7_f8.py, safe-inherit 验证, NPU subclass usage | T25.2 cold-drive 时通过 grep 验证 5 个 F7/F8 全部 safe inherit |

### transformers-day0

详情见 [`transformers/port-expert/references/ALWAYS_LOADED_RULES.md`](../../transformers/port-expert/references/ALWAYS_LOADED_RULES.md)。

| OL | Title | Keywords |
|---|---|---|
| OL-03 | transformers denylist | transformers cold-drive, prior PR_MATERIAL |
| OL-08 | transformers edit scope | `src/transformers/integrations/npu_*`，**不动** transformers core |
| OL-21 | upstream tag 名带 `v` 前缀 | v5.4.0 not v5.4, fatal invalid object name, git show <tag> | T25.3 cold-drive 抓到的 KB tag-naming gap |

### triton-ascend-port

详情见 [`triton-ascend/port-expert/references/KB_INDEX.md`](../../triton-ascend/port-expert/references/KB_INDEX.md)。

| OL | Title | Keywords |
|---|---|---|
| OL-22 | triton-ascend 是 vendored fork，不是 plugin | git merge --no-commit --no-ff，不用 rebase，不用 F1-F8 分类 |
| OL-23 | bishengir LLVM 版本必须匹配 libtriton.so | cross-binary IR pipeline, LLVM 19 vs 22, custom op 'to' unknown, triton-ascend-001 |

### sglang-npu-day0

详情见 [`sglang/port-expert/references/KB_INDEX.md`](../../sglang/port-expert/references/KB_INDEX.md)。

| OL | Title | Keywords |
|---|---|---|
| OL-24 | sglang NPU 3-axis 版本协调 | sglang main / sgl-kernel-npu / cann+torch_npu, image tag encodes 2-of-3, pip show 看第三轴 |
| OL-25 | sglang day-0 不开 fork | upstream actively maintained, GitHub issue handoff, no ascend-port branch |

### integrated-overlay-build

详情见 [`_shared/integrated-overlay-build/SKILL.md`](../integrated-overlay-build/SKILL.md)。

| OL | Title | Keywords |
|---|---|---|
| OL-26 | base image 必须 ship 真 vllm（非 +empty stub） | base image version axis, vllm 0.20.0 real, easyr1-npu-vllm0200:iter20-abi-both |
| OL-27 | 集成 overlay 不在这里修单上游 bug | overlay scope, fix upstream first then come back, T22 lessons |

---

## OL-XX 编号空间

- 已用：OL-01..OL-27（universal: 01-18，含 04b/05b；expert-specific: 19-27）
- 下一可用：OL-28
- ⚠ deprecated：（无，目前都活跃）

## 见也

- [`ALWAYS_LOADED_UNIVERSAL.md`](ALWAYS_LOADED_UNIVERSAL.md) — universal OL 详情（OL-01..OL-18）
- [`../patterns/F-family-taxonomy.md`](../patterns/F-family-taxonomy.md) — F1-F8 + F2-path-move 漂移分类
- [`../../../knowledge/npu-patterns.md`](../../../knowledge/npu-patterns.md) — NPU-CP/BUG/ENV/OPS 29 个 stable ID（与 OL 正交：NPU-XXX 描述硬件/环境陷阱，OL 描述 process rule）
- [`../../../docs/_meta/kb/porting_lessons/`](../../../docs/_meta/kb/porting_lessons/) — cross-layer 教训（已成文的 reflection）
- [`../../../docs/_meta/kb/challenge_patterns/`](../../../docs/_meta/kb/challenge_patterns/) — 11 条 self-critic 模板
