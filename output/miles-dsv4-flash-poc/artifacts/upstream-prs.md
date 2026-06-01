# miles+DSv4-Flash PoC — 上游 PR / Issue 反馈历史

> 实时机读 metadata 在 [`../PROJECT.json`](../PROJECT.json) 的 `upstream_prs` 字段。
> 本文件是人类可读的反馈时间线 + 状态详情。

---

## 1. `tile-ai/tilelang-mlir-ascend` PR #80 — CheckUBBudget 早失败诊断 pass

**类型**: PR · **当前状态**: OPEN, REVIEW_REQUIRED, CI 全绿 · **HEAD**: `6b5c8df`

URL: https://github.com/tile-ai/tilelang-mlir-ascend/pull/80

**时间线**:
- 2026-05-15 ~ 05-23:写 + 三轮 self-audit(原始 `daea72f` → ruff F841 `d2d1871` → `_UB_BACKED_SCOPES` 收紧 + catastrophic-only 阈值 `df7431e`)
- 2026-05-28: gemini-code-assist 自动 review,提 4 条 line-level(1 HIGH `mod.attrs=None` crash + 3 MEDIUM)
- 2026-05-31 10:54: push `27e5c54` — 应用 reviewer 4 条建议 + 加 5 个 UT
- 2026-05-31 11:04: **CI `test / test` FAILURE on 27e5c54** — 我把 `node.buffer_var.name` 改成 `.name_hint`(reviewer 建议),但 tilelang-vendored TVM 的 FFI `__getattr__` 对 `.name_hint` 抛 AttributeError。local 只做了 source-grep 没真跑 import
- 2026-05-31 11:21: push `6b5c8df` — 引入 `_var_name` defensive helper(try `.name_hint` → fall back `.name` → 从 `str()` 提 leading identifier);两 call site 统一走 helper;UT 加 fake-Var 验证两 fallback path
- 2026-05-31 11:25: **CI 9 SUCCESS + 3 SKIPPED,0 FAILURE** ✓

**经验**: memory `check_ci_on_my_own_push_before_declaring_done.md`(push 完必须等 CI 出结果再宣告 done)

---

## 2. `Ascend/AscendNPU-IR` Issue #251 — R-KA-16(`ExtendedCanonicalizer` 删跨 iter 累加器)

**类型**: Issue · **当前状态**: OPEN, labelled `triage-review`

URL: https://gitcode.com/Ascend/AscendNPU-IR/issues/251

**时间线**:
- 2026-05-23: 提交 issue body(107 行中文诊断报告 + bisect 方法 + 311-pass 曲线表 + before/after IR diff + 根因解释 + 3 patch 方向建议 + reproducer)
- 2026-05-29 02:24:`zhengshencn_hwca` update — 把 311-pass dump 切片成 `rka16_ns4_pass_index.txt`,line 10801 的 `ExtendedCanonicalizer` 是罪魁
- 2026-05-29 09:08:Huawei reviewer `SL25` 加 `triage-review` label — issue 已分配到 triage 队列,但还没有具体 owner 或 patch ETA

**未解决依赖**: Huawei 编译器组在 `ExtendedCanonicalizer` 加 conservative-keep 规则,或绕过 SCF iter_args canonicalization 处理 DPS in-place

---

## 3. `radixark/miles` PR #1246 — `_npu/` 子包 + 4 DSAMLA 算子

**类型**: PR · **当前状态**: OPEN, REVIEW_REQUIRED, MERGEABLE · **HEAD**: `ff0161cc0`

URL: https://github.com/radixark/miles/pull/1246

**时间线**:
- 2026-05-25 ~ 05-28:从 18-commit 开发分支两轮 audit 蒸出 `npu-tilelang-ops` 干净分支(`d03db2c`):13 文件 1767 LOC,4 个 tilelang kernel + dispatcher + head-split + UB cap + R-KA-16 mitigation
- 2026-05-28: gemini-code-assist 提 **6 条 HIGH** line-level review:
  - 5 个 negative-sentinel 漏 guard(`cur_idx == -1` 在 NPU 上直接 OOB / corrupt)
  - 1 个 intrinsic 拼写 `T.atomic_addx4` → 正确名 `T.npuir_atomic_addx4`
  - **1 个 production bug — `_MAGIC_THRESHOLD = 1e3` 在 AMP 下吃合法 loss-scaled gradient**(AMP loss-scale=65536 时,合法 grad 0.1 → 6553.6 > 1e3 被 silently zeroed)
- 2026-05-31 10:51: push `ff0161cc0` — 全部 6 条 fix + 9 个 source-level UT(本地 9/9 PASS + negative-test 过)

**Production-bug catch 价值**: reviewer 抓到的 AMP issue 单独就值得 PR 流程的存在 — memory `pr_polling_must_check_reviews_field.md`(`gh pr view comments[]` 漏看 reviewer review body)是从这条 finding 起的教训

---

## 4. `Ascend/MindSpeed` MR #3509 — apex fused-rope-thd shim

**类型**: gitcode MR · **当前状态**: OPEN, no human reviewer activity

URL: https://gitcode.com/Ascend/MindSpeed/merge_requests/3509

**时间线**:
- 2026-05-28:提交 — 38 行 self-contained pure-torch `_fused_apply_rotary_pos_emb_thd_fallback` + 1 行 `pm.register_patch('apex.transformer.functional.fused_apply_rotary_pos_emb_thd', ...)`
- 2026-05-30:author email amend(huawei → gmail)force-push,SHA `9aa2f75f`
- 2026-05-31:10 条 comment 全是 `ascend-robot` docs CI(都 skip 了),无 human review

---

## 5. `sgl-project/sgl-kernel-npu` PR #531 — `fused_split_qk_norm` RMSNorm.bias getattr fix

**类型**: PR · **当前状态**: OPEN, REVIEW_REQUIRED · **HEAD**: `3c08165`

URL: https://github.com/sgl-project/sgl-kernel-npu/pull/531

**时间线**:
- 2026-05-30:4 行 patch — `q_a_layernorm.bias` → `getattr(q_a_layernorm, "bias", None)`(4 sites)
- 2026-05-30:gemini-code-assist auto-review:"no review comments, no feedback"(4 行 patch 太小 reviewer 没话说)

---

## 6. `sgl-project/sglang` Issue #26794 — `/update_weights_from_disk` FusedMoE reload narrow regression

**类型**: Issue · **当前状态**: OPEN, 0 comments

URL: https://github.com/sgl-project/sglang/issues/26794

**时间线**:
- 2026-05-30:提交 issue — initial-load 通过、`/update_weights_from_disk` reload 触发 `RuntimeError: start (0) + length (4096) exceeds dimension size (1408)` at `fused_moe_triton/layer.py:482 _load_w13`;reload path 不 honor `stacked_params_mapping`,把 stacked 4096-dim 整体往 1408-dim slot 塞。dense-only fab 路径已 prove plumbing 正常,等 maintainer 回复 reload path 是否应该跟 initial-load 共用 stacked mapping

---

## 7. `triton-lang/triton-ascend` Issue #306 — triton vs triton-ascend coexistence(closed)

**类型**: Issue · **当前状态**: closed not-planned

URL: https://github.com/triton-lang/triton-ascend/issues/306

**时间线**:
- 2026-05-29 早:误以为 `triton-ascend` 和 mainline `triton` 抢同一个 `triton/backends/compiler.py` 是 triton-ascend 的责任,提 issue
- 2026-05-29 同日:user 正确指出**两者本来就不该共存**,真正的责任链 `xgrammar` declares `Requires-Dist: triton; platform_system == "Linux" and platform_machine == "x86_64"` → NPU host 也 match → `vllm-ascend → xgrammar` 拉 mainline triton → 与 triton-ascend 撞;责任在 `mlc-ai/xgrammar`(请 NPU-aware marker)或镜像作者(install order)
- 2026-05-29:closed not-planned + 写 KB cookbook `triton-ascend-002` + close-with-reframing comment

**经验**: memory `feedback_check_responsibility_layer_before_filing.md`(别因为 `ImportError` 出现在 package A 文件里就反射性提 issue 到 A;先追责任链:谁的 install 把两个东西放到同一 site-packages?那个层才是责任方)

---

## 8. `zhengshencn_hwca/a5_ops` MR `blue/pr/kw-brief-fa-gate-name-align` — kw_brief FA-gate 漏改 call-site(task#28 follow-up)

**类型**: gitcode MR(a5_ops 内部 harness)· **当前状态**: pushed, MR 待 main open/merge · **branch HEAD**: kw_brief fix + 5-case regression test

URL: https://gitcode.com/zhengshencn_hwca/a5_ops/merge_requests/new?source_branch=blue/pr/kw-brief-fa-gate-name-align

**时间线**:
- 2026-06-01:`/ascendc-op-gen hc_split_sinkhorn` 重跑(team 修完 task#28/29/30 后)→ routing 走对(kw 路径),但 kernel-worker 收到的 brief 仍含 "STOP — DO NOT AUTHOR / emit structural_rewrite_needed" FA-escalation block。根因:task#28 把**路由** gate(`should_use_tilelang_il` → name-based `is_attention_named`)修了,但漏了**第二个 call-site** `briefs/kw_brief.py:172` `_fa_class_design_absent_emit_block`,它还用 tag-based `is_fa_class(op_class)`(对 `fused`+`softmax` tag 误判 True)。两 gate disagree → worker 拒绝乱写、handoff await_user_decision、不产出 kernel。
- 2026-06-01:本地 working-tree 先打 fix 解锁 op-gen(用过的同款 stopgap 模式),验证 `is_attention_named("hc_split_sinkhorn")==False`(正常 author)/ `3_FusionAttention`==True(FA-STOP 保留)→ resume op-gen run#2,worker-2 收到正常 Vector brief、直接 author。
- 2026-06-01:按 owner 指示提 MR(branch `blue/pr/kw-brief-fa-gate-name-align`)—— fix + `test_task28_followup_kw_brief_name_gate.py`(5 cases green)。

**经验**: memory `a5ops_fa_gate_two_callsites.md`(FA-class gating 有两个 call-site:router + kw_brief;name-gate 修复要查两处)。worker probe-first 顶住 "STOP" 指令(P8)、自己钉死根因到 file:line —— 是 op-gen harness worker 该有的行为。task#31(classifier 发 distinguishing FA tag)落地后两 gate 可回退 tag-based。
