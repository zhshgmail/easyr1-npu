# `src/skills/` — 所有 skills，按上游 repo 分类

## 结构

```
src/skills/
├── _shared/                          跨 upstream 共用的 skills + 模板
│   ├── references/                   OL rules, universal patterns
│   ├── scripts/                      shared scripts
│   ├── upstream-branch-hygiene/      通用 git 工作流
│   ├── codex-review/                 评审流程
│   ├── dep-gap-detect/               依赖 gap 扫描
│   ├── npu-image-inspect/
│   ├── npu-code-path-sweep/
│   ├── npu-container-runner/
│   ├── ray-npu-shim/
│   └── image-upgrade-drill/
│
├── orchestrators/                    跨 skill 的调度层
│   └── npu-port/                     /npu-port 斜杠命令，编排 port-expert
│
├── vllm-ascend/
│   └── port-expert/                  端到端 port（合并了 day0 + upgrade）
│       ├── SKILL.md                  /vllm-ascend-port 斜杠命令
│       ├── agent.md
│       ├── state_machine.yaml
│       ├── references/
│       └── scripts/
│
├── vllm/port-expert/                 (+ _legacy-upgrade/ 待合并)
├── torch-npu/port-expert/            (+ _legacy-upgrade/)
├── transformers/port-expert/         (+ _legacy-upgrade/)
├── easyr1/port-expert/               consumer port（不是 upstream）
└── dep-analysis/expert/              meta 依赖分析工具
```

## 为什么这么组织

每个上游 repo 在这里有独立目录，该上游的所有 skill / agent brief / scripts / references 都放里面。`_shared/` 放跨上游通用的东西。`orchestrators/` 放调度层。

**day0 / upgrade 不再分两个 expert**：端到端 port 工作是**同一件事**（改 Huawei-owned upstream 的代码 / shim consumer / 验证 V1.3/V1.4 PASS / 沉淀 KB），硬拆两个 expert 是制造假边界。`port-expert/_legacy-upgrade/` 保留旧 upgrade-expert 的内容作为历史归档，后续应并入主 port-expert/。

## 如何安装为 Claude Code 斜杠命令

```bash
bash src/scripts/install-skills.sh
```

递归扫 `src/skills/**/SKILL.md`，读每个文件的 YAML frontmatter `name:`，symlink 到 `~/.claude/skills/<name>/`。

**装哪些**：带 `name:` frontmatter 的 `SKILL.md`。
**不装**：纯 `agent.md` brief（被 orchestrator 通过 filesystem path 读，不走 Claude Code skill registry）

## 卸载

```bash
bash src/scripts/install-skills.sh --undeploy
```
