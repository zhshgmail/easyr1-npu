# Postmortem: T30 design v1 重蹈 a5_ops `SAFETY_NET_NAME_COUPLING` 覆辙 — 2026-05-16

> **Self-postmortem**：本仓自己的设计文档 v1 在借鉴 a5_ops 后**重蹈了** a5_ops 自己 2 周前在 [SAFETY_NET_NAME_COUPLING_2026_05_14.md](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md) 写过的 mode-coupling 失效模式。Codex review 抓出来。本文 codify 教训，避免下次。

**Date**: 2026-05-16
**Severity**: MEDIUM — design 阶段被外部 review 抓出，无 production 损失；如果 v1 ship 实现，sglang day-0（无 fork branch）会立即撞墙
**Cost**: codex review 2 轮（共 ~120K token）+ v2/v3/v4 三次设计迭代 ~3 小时
**Reporter**: Codex review #2 (2026-05-16)
**Status**: fix landed commits `385f9a9` (v4 design) + `685de73` (P0i/P0j/P0p/P0q ship)

## Summary

T30 设计 v1 → v2 → v3 三轮中，每轮都暴露 mode-coupling 类问题：

- **v1**: scanner / gate path 参数没参数化（"workspace 必有 PR_MATERIAL.md" 当默认）→ codex #1 抓出
- **v2**: 加了 `Mode` enum + per-mode `required_artifacts` 表 → 看起来 mode-aware 了
- **v3**: 借 a5_ops adversarial audit M2 时，**直接复制 a5_ops `verification.json` 字面 grep**到我们的 scanner — 这是 cargo-cult；我们仓里根本没 `verification.json` 文件；同时 §2.3 mode required-field 表和 §4.4 gate required artifact 列表写成两份相互矛盾 → codex #2 抓出
- **v4**: 引入 `mode_dispatch.py` 单一权威（Mode/Outcome/ArtifactRole/EvidenceType + REQUIRED_* dicts），所有 downstream consumer 派生自这一个文件；scanner anti-cycle 从字面 grep 改为 dynamic forbidden-path set（按当前 manifest derive）；outcome enum 只保留 Python 形式（A_WITH_NOTE，不是 A-with-note）

**讽刺的是** a5_ops 自己的 SAFETY_NET_NAME_COUPLING postmortem 写的就是 "safety net coupling to file names is fragile by construction" —— 然后我借鉴它时**复制了它的文件名**，立刻撞同一个洞。

## What happened (timeline)

- 2026-05-15 20:11Z: user 问 A3 NPU 可访问；启动 T29 a5_ops 借鉴落地（之前的工作）
- 2026-05-15 20:30Z: user 让起 T30 — 学 a5_ops "测试网 + 安全网 + 反馈环"
- 2026-05-15 20:35Z: T30 design v1 写好
- 2026-05-15 20:40Z: codex review #1 (~55K tokens) 抓出：v1 markdown regex 太脆 / over-engineering 6 个 gate / 没区分 mode / `pr_material` 硬编码 → fixes → v2
- 2026-05-15 23:15Z: 拉 a5_ops 看他们的 adversarial audit，挪了 M2 anti-cycle 进设计
- 2026-05-15 23:30Z: v3 写好；commit `73b28c2`
- 2026-05-16 03:55Z: codex review #2 (~70K tokens) 抓出 v3 6 类回归：
  - **R1**: M2 cargo-cult `verification.json`
  - **R2**: 路径解析 bug（manifest 在 workspace/<skill-tag>/，但 verifier path 用 repo-root 风格）
  - **R3**: M5 self_challenge.notes 字符串扫 = security theater
  - **R4**: bad fixtures 全丢 `validate_manifest()` 但 gate fraud 走不同层
  - **R5**: M3 deferral 事实错（codex 自己去 grep 仓库验证 /npu-port SKILL 已 spawn）
  - **R6**: §2.3 ↔ §4.4 schema/gate dispatch drift（pr_material optional vs required；A-with-note vs A-w-note vs A_WITH_NOTE）
- 2026-05-16 04:15Z: v4 落地 commit `385f9a9`
- 2026-05-16 04:30Z: P0h.0 实现 + 16 tests PASS
- 2026-05-16 05:00Z: P0i + P0j + P0p + P0q 全部实现 + 24 tests PASS
- 2026-05-16 05:15Z: hook wire commit `42df04c`

## Root cause

**Architectural** (multi-layer):

1. **借鉴时未做 local translation**: a5_ops 借鉴是大方向对，但**每条 rule 必须重新表述为本地 artifact + 本地 attack surface**。v3 把 a5_ops 字面文件名 `verification.json` 抄进我们 scanner，对应物在我们这里是 `claim_manifest.yaml`；同理 a5_ops 的 `pass_a_runner.py` 在我们这里是任意 `verifier_scripts[i].path`。
2. **没把 review feedback 沉淀进规则**: codex #1 R2 ("model workflow modes explicitly") 已经说了要 mode enum；v2 加了，但只用于 schema 校验，没把 gate 重构成 mode-derive；§4.4 gate 代码仍硬编码 `PR_MATERIAL.md`。两层（schema vs gate）独立写，drift。
3. **多个 dispatch 表分散**: v2/v3 每节自己写自己的 required-field（§2.3 一份、§4.4 一份、§3.6 fixtures 暗含一份），没有 single source of truth。
4. **outcome naming convention 自己没定**: v2 `A-with-note` (prose) vs v3 `A_WITH_NOTE` (Python) vs hybrid。我自己写的时候 alternate use；codex 抓到三种写法同时存在。

**Immediate cause**: 没把 codex #1 的 R2 (mode-aware dispatch) 落地成代码模块，只是文档说法。

## How it slipped past every layer

| Layer | What it was supposed to do | What actually happened |
|---|---|---|
| 设计原则 §0.2 #4 | "Safety net 不能 couple 到具体文件名" | v1 自己违反；v3 借鉴 a5_ops 时再次违反 |
| Codex review #1 | 抓 v1 mode-coupling | 抓到了 → 推 v2；但 v2 修了 schema 没修 gate；codex 没追第二轮直到 v3 |
| 自审 | 写 v3 时应自检 "M2 实现是 generic 还是字面复制" | 没自检；M2 抄了字面 `verification.json` |
| Codex review #2 | 抓 v3 | 抓到了 6 类问题；本次 review 是关键防御层 |
| Sanity suite | 跑 schema_bad fixtures 验证 | T31 之前不存在 |

**核心**: codex review 是唯一接住的层。如果 user 没要求 codex review，v3 会直接 ship 实现，撞 sglang day-0 (issue_only mode) 立刻爆雷。

## Fix (landed `385f9a9` + `685de73`)

- **Code (mode_dispatch.py)**: 新建 `src/scripts/safety/mode_dispatch.py` 作 single source of truth — `Mode` / `Outcome` / `ArtifactRole` / `EvidenceType` enums + `REQUIRED_ARTIFACTS` / `REQUIRED_EVIDENCE` / `REQUIRED_VALIDATION_LEVEL` dicts。**所有 downstream consumer（schema validator / scanner / gate / sanity suite）禁止 redeclare**；sanity-suite `test_no_competing_enum_declarations` grep 强制。

- **Scanner (scan_outcome_claims.py)**: M2 从字面 `grep "verification.json"` 改为 **dynamic forbidden-path set**：`_collect_forbidden_paths()` 按当前 manifest evidence[].path + artifacts[].paths derive；scanner 扫 verifier body 是否含任一 forbidden path。Self-adjusts per manifest，无 a5_ops 字面字符串。

- **Gate (finalize_day0_check.py)**: 4 gate（codex #1 R3 砍）；mode-derive 自 `mode_dispatch.required_artifacts(mode)`；no hardcoded `PR_MATERIAL.md` literal in gate code。

- **Schema**: `_manifest_relative_to_repo_root` + `repo_root_anchor` 字段，路径解析显式 `(repo_root / v.path).resolve()` + `is_relative_to()` 检查。

- **Outcome canonical**: 只有 Python enum 形式 (`A_WITH_NOTE`)；`test_outcome_enum_canonical_only` 扫所有 good fixtures 拒 prose 变种。

- **DEBT**: ROADMAP §6 加 DEBT-8 (cross-backend joint regression — 当前只 codex；a5_ops 用 Opus + V4 双 backend 抓到 Opus 漏的攻击)。

## What we couldn't fix yet

- **Cross-backend pre-ship review**: codex 是单 backend；未来引入 gpt-4o / gemini 作 second-opinion review。当前 DEBT-8 跟踪。
- **Design 阶段无 mechanical test**: T31.x sanity suite 现在能跑 gate fitness，但**只对实现，不对 design 本身**；design doc 自己的 drift（如 §2.3 vs §4.4 字段说法不一）是 prose-level，sanity suite 抓不到。未来可加 design-doc 自身的 lint（如 grep "outcome.*A-with-note" 报 fail）。

## Lessons (永久记录)

1. **借鉴外部设计时必须做 local translation**：每条 rule 重新表述为**本仓 artifact + 本仓 attack surface**；禁止字面复制对方文件名 / 对方 fixture 名 / 对方 codex symbol。Codified as: `ANTI_PRESSURE_PROTOCOLS.md` P-future-N (not yet written; candidate for future addition).

2. **Mode-aware dispatch 必须用单一权威表**：不允许"§2.3 一份 schema 视角的 required，§4.4 一份 gate 视角的 required"。设计文档 + 实现共用同一个 `mode_dispatch.py` 数据结构。Codified as: §2.5 canonical mode_dispatch_table in design v4 + `src/scripts/safety/mode_dispatch.py` + sanity-suite `test_no_competing_enum_declarations`.

3. **Outcome naming convention 第一时间定死**：项目 lifecycle 早期就要选 prose 还是 enum 形式，全文统一。Codified as: `mode_dispatch.Outcome` Python enum (canonical) + `test_outcome_enum_canonical_only` sanity test bans prose variants in fixtures.

4. **Codex review 是当前唯一的"独立第二意见"层**：本仓没有 cross-backend joint regression（DEBT-8）；codex 每个 design iteration 必跑；user 别忘了要 review。Codified as: ROADMAP §6 DEBT-8 + workflow rule in CLAUDE.md (future addition candidate).

5. **设计文档 prose 容易 drift**：sanity suite 抓不到 design doc 本身的不一致；codex review 是唯一防御。设计稿改任何 dispatch 表 / outcome / mode 字段时，**必须同步搜全文** + 必须 codex review。Codified as: workflow rule (informal; could become a sanity-suite test that lints design doc for inconsistent dispatch tables).

6. **"借鉴 a5_ops 自己的 postmortem" 也救不了你**：a5_ops 自己的 SAFETY_NET_NAME_COUPLING postmortem 写的就是文件名 coupling 的反模式，我读了它仍然在 v3 里复制了 a5_ops 的文件名。**抽象警告会被忽略，具体事故锚点能触发识别** — 这也是 ANTI_PRESSURE_PROTOCOLS 强调 incident anchor 而不是抽象 rule 的原因。本 postmortem 就是这条 incident anchor。

---

## 见也

- 源教训: a5_ops [`docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md`](https://gitcode.com/zhengshencn_hwca/a5_ops/blob/main/docs/postmortem/SAFETY_NET_NAME_COUPLING_2026_05_14.md)
- Codex review #1: [`docs/_meta/design/review/2026-05-15_codex_review_T30_design.md`](../design/review/2026-05-15_codex_review_T30_design.md)
- Codex review #2: [`docs/_meta/design/review/2026-05-16_codex_review_T30_design_v3.md`](../design/review/2026-05-16_codex_review_T30_design_v3.md)
- Design v4: [`docs/_meta/design/TEST_SAFETY_FEEDBACK_DESIGN.md`](../design/TEST_SAFETY_FEEDBACK_DESIGN.md)
- ROADMAP: P0f / P0g / P0m / P0n （design 4 轮 iteration 全部跟踪），DEBT-8 (cross-backend regression)
- 实现 commits: `385f9a9` (design v4) → `7f68880` (P0h.0) → `685de73` (P0i/P0j/P0p/P0q)
