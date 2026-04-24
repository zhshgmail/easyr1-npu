# vllm-ascend 适配新版 vllm 的移植指南

## 这份指南给谁看

你是 **vllm-ascend 维护者** 或 **社区贡献者**，想把 vllm-ascend 适配到一个新版本的 vllm（社区主仓）。典型场景：

- vllm 发了新版本，里面有 API 变更，vllm-ascend 还没跟上
- 你刚 rebase 到 vllm 最新 main，发现 plugin 装不进去
- 你想在 "合并窗口之前"  判断下一版 vllm 会引入哪些会打破 vllm-ascend 的改动

## 这份指南交付什么

一套 **KB + 工具 + skill** 组合：

1. **4 个 scanner + 1 个 sweep wrapper**：
   - `kb_drive_test.py` —— 给一个 vllm commit SHA 扫 F1/F2-rename/F3/F5-suspect
   - `sweep.sh` —— 跨 commit-range 一键跑所有 scanner + 交叉对比 KB
   - `check_f4.py` —— F4 返回值类型漂移（AST）
   - `check_f7_f8.py` —— F7/F8 基类新属性/新方法（AST）
2. **Pattern KB** —— 把 vllm API drift 分成 8 族（F1 删符号 / F2 改名 / F3 改签名 / F4 返回值 / F5 buffer API / F6 kv_cache / F7 新必须字段 / F8 新必须方法），每族带对应的 fix template。
3. **验证 skill** `/drift-port-validate` —— 在 A3 上快速确认你写的 compat shim 两条分支（OLD vllm pass-through / NEW vllm fallback）都正确。
4. **skill 入口** `/vllm-ascend-day0` —— 把上面串起来的主 skill。

全套可以在 **不懂 vllm 内部实现** 的情况下完成一个 drift port：scanner 告诉你哪些 symbol 坏了，KB 告诉你怎么修，验证 skill 确认你修对了。

**推荐流程**（2026-04-24 起）：一条 `sweep.sh` 命令跑完所有 scanner，看输出决定是否需要按 F-family 模板写 compat shim。

## 工作流（共 5 步）

### 步骤 1 —— 扫描：哪些变更会打破 vllm-ascend？

**一键 sweep**（推荐）：

```bash
cd <easyr1-npu-repo>
./src/skills/vllm-ascend/port-expert/scripts/sweep.sh \
  --commit-range v0.20.0..origin/main \
  --vllm-path /path/to/vllm \
  --vllm-ascend-path /path/to/vllm-ascend
```

一条命令跑完：
- `kb_drive_test.py` 对 range 内每个 commit 扫 F1/F2-rename/F3/F5-suspect
- `check_f4.py` 对整个 range 扫 F4 返回值类型漂移
- `check_f7_f8.py` 对整个 range 扫 F7/F8 基类新属性/方法
- 去重 + 和 KB 案例注册比对，报告 novel 发现

**或单独调用 kb_drive_test 针对一个特定 SHA**：

```bash
python3 src/skills/vllm-ascend/port-expert/scripts/kb_drive_test.py \
  --vllm-ref <TARGET_VLLM_SHA> \
  --vllm-path <path-to-vllm-checkout> \
  --vllm-ascend-path <path-to-vllm-ascend-checkout> \
  --kb-dir src/skills/vllm-ascend/port-expert/references
```

输出：`proposal.md` + `summary.json`。`summary.json` 里每个 drift 包含：

- `symbol` —— 被 vllm 删掉 / 改名 / 改签名的符号
- `kind` —— drift 的形状（`removed_symbol` / `renamed` / `sig_change`）
- `family` —— 匹配到 KB 的哪一族（`F1` ~ `F8`）
- `ascend_sites` —— vllm-ascend 里受影响的调用点数量

如果 `summary.json` 的 `unmatched` 为 0 且 `impact_ascend` > 0，说明扫描器找到了可以按 KB 模板修的 drift。

### 步骤 2 —— 读 KB 对应的 fix template

扫描器给你 `family` 字段后，打开对应章节：

| Family | 场景 | 读这里 |
|---|---|---|
| F1 | `ImportError: cannot import name 'X'` | [patterns/domains/vllm-api-drift.md §F1](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |
| F2 | `AttributeError: ... has no attribute 'OldName'` | [§F2](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |
| F3 | `TypeError: func() missing/unexpected argument` | [§F3](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |
| F4 | 先前是 scalar 的返回值现在变 NamedTuple | [§F4](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |
| F5 | 多个 `.np` / `.copy_to_gpu` / `CpuGpuBuffer` 调用点同时坏 | [§F5](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |
| F6 | `AssertionError: attn_layer.kv_cache must be single tensor` | [§F6](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |
| F7 | `AttributeError: 'NPUX' has no attribute 'Y'` where Y 是基类新加的字段 | [§F7](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |
| F8 | `NotImplementedError` 基类新加的方法 | [§F8](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) |

每一族都给你一个 try/except 或 sig-sniff 的 template，**目标是保持"同一棵 vllm-ascend 源码树能同时对旧版和新版 vllm build 成功"**，不是硬切版本。

### 步骤 3 —— 按 template 写 compat shim

约定放位置：`vllm_ascend/compat/<symbol>.py`。

**F1 例子**（删符号）：

```python
# vllm_ascend/compat/shared_fused_moe.py
try:
    from vllm.model_executor.layers.fused_moe.shared_fused_moe import (
        SharedFusedMoE as _UpstreamSharedFusedMoE,
    )
    SharedFusedMoE = _UpstreamSharedFusedMoE
    _UPSTREAM_HAS_SHARED_FUSED_MOE = True
except ImportError:
    _UPSTREAM_HAS_SHARED_FUSED_MOE = False
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    class SharedFusedMoE(FusedMoE):  # type: ignore[no-redef]
        ...  # 等价的本地实现
```

然后把原来 `from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE` 的地方换成 `from vllm_ascend.compat.shared_fused_moe import SharedFusedMoE`。

**关键点**：

- `_UPSTREAM_HAS_<X>` 是标志位，验证 skill 靠它确认走了哪条分支。**必须导出**。
- 本地 fallback 要保留和上游原类同等的结构承诺（比如继承链），否则下游 `isinstance` / `issubclass` 会漏掉你的类。

### 步骤 4 —— 验证（drift-port-validate skill）

写完 compat shim 后，调用：

```
/drift-port-validate patch-summary="F1 compat shim for SharedFusedMoE at vllm_ascend/compat/shared_fused_moe.py, 2 callsites swapped"
```

这个 skill 会在 A3 的一个 container 里：

- 模拟 "OLD vllm 还在"：确认 shim 走 try 分支 → 指向上游
- 模拟 "NEW vllm 已经删了"：确认 shim 走 except 分支 → 指向本地 fallback

两边都必须 PASS。**通过不代表跑得动 inference**，只代表 shim 结构正确。Inference 级验证见步骤 5。

### 步骤 5 —— 推 fork branch + 提 PR

约定分支名：`<TARGET_VLLM_VERSION>_auto_porting`（e.g. `vllm-main_auto_porting`）。

每个 drift 一个 commit，commit message 里写明：

- 上游 vllm commit SHA
- 上游 vllm PR 编号
- 对应 KB family
- 动过的 vllm-ascend 源文件

PR 描述里引用本指南路径 + 附上 `kb_drive_test.py` 的 `summary.json` 作为 "为什么这个 drift 需要修" 的证据。

## 参考执行（2026-04-24，作者首次全流程跑通）

目标：vllm main (post-v0.20.0) 上两个 F1 drift
- `SharedFusedMoE` 被 PR #35782（commit `5e584ce9e`）删掉
- `DefaultMoERunner` 被 PR #40560（commit `809d83c2d`）合并改名到 `MoERunner`

流程：
1. 扫描 156 个 post-0.20.0 commit → 命中 2 个 F1（另外有 1 个假阳性被 filter 过滤）
2. 按 F1 template 写 `vllm_ascend/compat/shared_fused_moe.py` + `default_moe_runner.py`
3. 改 3 个源文件的 import（2 个 fused_moe.py）
4. drift-port-validate 跑出 **OLD: 4/4 + NEW: 5/5 = 9/9 PASS**
5. 推到 `zhshgmail/vllm-ascend` branch `vllm-main_auto_porting`

完整 commit 在 [fork branch](https://github.com/zhshgmail/vllm-ascend/tree/vllm-main_auto_porting)：
- `2fb41e8f` F1 SharedFusedMoE
- `08fa6f85` F1/F2 DefaultMoERunner

## 下一步延伸

- 扫描器现在只覆盖 F1 / F2 / F3 三个族。F4 / F5 / F6 / F7 / F8 还要补检测规则。
- A3 cold-drive 目前只做结构检查（shim 两分支 PASS）。完整 inference 检查（V1.3 rollout token-by-token diff）等下游使用侧再做。
- vllm-ascend 层做完后，**下一层是 torch_npu**。torch_npu 的移植遵循同一套"manual → KB → skill"顺序，不先写 skill 骨架。

## 相关文档

- [`docs/vllm-ascend/KB-SUMMARY.md`](KB-SUMMARY.md)（待建） —— KB 结构总览（platform / patterns / 索引 / 案例注册 / self-critic）
- [`src/skills/vllm-ascend/port-expert/SKILL.md`](../../src/skills/vllm-ascend/port-expert/SKILL.md) —— 主 skill 入口（orchestrator 读这个）
- [`src/skills/vllm-ascend/port-expert/references/KB_INDEX.md`](../../src/skills/vllm-ascend/port-expert/references/KB_INDEX.md) —— KB 索引（症状→ family 路由表 + 案例注册）
- [`src/skills/_shared/drift-port-validate/SKILL.md`](../../src/skills/_shared/drift-port-validate/SKILL.md) —— 验证 skill
