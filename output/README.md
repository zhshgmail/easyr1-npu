# `output/` — 独立子项目的标准 home

> 借鉴 a5_ops 的 `output/<slug>/` 模式。每个独立 sub-project(PoC、port、integration、harness、audit)在本目录下有自己的子目录,**所有该项目对外可见的成果(README、报告、复现步骤、PR 索引、artifacts)都汇总在那里**。
>
> 学习 audit:[`workspace/a5_ops_audit_2026_05_31/FINDINGS.md`](../workspace/a5_ops_audit_2026_05_31/FINDINGS.md)

## 当前 sub-projects

| Slug | Kind | Status | 一句话 |
|---|---|---|---|
| [`miles-dsv4-flash-poc`](miles-dsv4-flash-poc/) | poc | active | 让 `radixark/miles` + DeepSeek-V4-Flash 在 Ascend A3 NPU 上跑通 RL 后训练;5 PR + 2 Issue + 13 KB cookbook + `/npu-adapt-assist` skill |

## 添加新 sub-project

```bash
SLUG=<your-slug>     # 必须 kebab-case
cp -r output/_project_template output/${SLUG}
# 编辑 output/${SLUG}/PROJECT.json - 填 slug / kind / source / chip / created
# 编辑 output/${SLUG}/README.md - 填一句话 / status 表 / dir tree / 快速复现
# 写 output/${SLUG}/docs/REPORT.md, REPRODUCE.md
# 把这一行加到本文件的「当前 sub-projects」表
```

PROJECT.json 必须符合 [`_project_schema/PROJECT.schema.json`](_project_schema/PROJECT.schema.json)。校验:

```bash
python3 -c "
import json, jsonschema, sys
sys.exit(jsonschema.validate(
    json.load(open('output/<slug>/PROJECT.json')),
    json.load(open('output/_project_schema/PROJECT.schema.json'))
) is None and 0 or 1)
"
```

## 与其他顶层目录的关系

| 顶层目录 | 用途 | 与 output/ 的关系 |
|---|---|---|
| `docs/` | **项目级** 跨 sub-project 的协调文档(ARCHITECTURE / ROADMAP / UPSTREAM_FORKS / DOCS-CONVENTION) | 不放 sub-project specific 内容;sub-project specific 报告都在 `output/<slug>/docs/REPORT.md` |
| `docs/_meta/kb/` | KB cookbook(跨 sub-project 共享的 porting lesson + challenge pattern) | sub-project 沉淀的 cookbook 写到这里;sub-project 自己的 `docs/kb_index.md` 链接过去 |
| `src/` | 源码(skills、scripts) | sub-project 产生的 skill 改动写到这里;`output/<slug>/PROJECT.json` 的 `skill_changes` 字段索引过去 |
| `knowledge/` | 跨项目硬件知识(npu-patterns、upstream-refs、image snapshots) | sub-project 不放硬件知识;直接 use |
| `workspace/` | **session 期临时 + 大型 forensic 不可移**(R-KA-16 IR dump 等) | sub-project 的 reproducer 默认在 `output/<slug>/artifacts/scripts/`,但跨 sub-project 共享的 forensic asset 留在 workspace,用 `artifacts/README.md` 写索引(miles-dsv4-flash-poc 走这条路) |
| `tests/` | 项目级 tests(skill cold-drive 等) | sub-project 内的 tests(如 reviewer feedback UT)走对应上游 fork 分支,不在本仓 |

## 历史 redirect

| 老路径 | 新路径 |
|---|---|
| `docs/_meta/MILES_DSV4_NPU_POC_REPORT.md` | [`output/miles-dsv4-flash-poc/docs/REPORT.md`](miles-dsv4-flash-poc/docs/REPORT.md) — 2026-06-01 重组(借鉴 a5_ops `output/<slug>/` 模式) |
