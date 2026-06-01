# `<project-slug>` — `<one-line title>`

> **kind**: poc / port / integration / harness / audit
> **status**: active / complete / blocked / archived
> **target**: Ascend 910C A3 / 950PR A5 / 910B A2 / any
> **owner**: claude-opus-4-7

## 一句话

`<what this delivers — 2-3 sentences max>`

## 当前状态

| 字段 | 值 |
|---|---|
| 进度 | `<3/5 阶段完成>` |
| 阻塞项 | `<list, or 无>` |
| 下一步 | `<single sentence>` |

## 上游 PR / Issue

| # | 上游 | 类型 | 状态 | URL |
|---|---|---|---|---|
| 1 | `<owner/repo>` | PR / Issue | open / merged | `<url>` |

## 目录结构

```
output/<project-slug>/
├── PROJECT.json          # 元数据(schema:output/_project_schema/PROJECT.schema.json)
├── README.md             # 本文件 — 入口 + 状态 + 索引
├── docs/
│   ├── REPORT.md         # 详细成果报告(问题/方法/证据/结果)
│   ├── REPRODUCE.md      # 复现步骤(0 → 全栈跑通)
│   └── kb_index.md       # 此项目沉淀的 KB cookbook 列表 + 链接
└── artifacts/
    ├── upstream-prs.md   # PR / Issue 的逐项 status + 反馈历史
    ├── scripts/          # reproducer / driver / probe 脚本
    └── <other>/          # 其他大文件(IR dump、screenshot 等)
```

## 快速复现

```bash
# <2-3 步快速指令>
```

详细见 [`docs/REPRODUCE.md`](docs/REPRODUCE.md)。

## 沉淀

- KB cookbooks: `<list with relative links into docs/_meta/kb/porting_lessons/>`
- Skill 改动: `<list>`
- Memory entries: `<list>`
