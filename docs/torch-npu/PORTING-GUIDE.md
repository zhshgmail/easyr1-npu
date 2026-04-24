# torch_npu 适配新版 torch 的移植指南

## 这份指南给谁看

你是 **torch_npu 维护者**（或 Ascend/pytorch 贡献者），想把 torch_npu 适配到一个新版本的 torch（社区主仓）。典型场景：

- torch 发了新 minor 版本（比如 2.11 → 2.12），torch_npu 还没跟上
- 你刚跑 torch nightly 发现 `import torch_npu` 挂掉
- 你想提前判断 torch 下一个 release 会引入哪些打破 torch_npu 的改动

## 为什么 torch_npu 和 vllm-ascend 的适配不一样

| 上游 | 公开 API 稳定性 | 主要 drift 形状 | 影响面 |
|---|---|---|---|
| vllm | 公共 API 稳定 | F1（删符号）/ F2（改名）/ F7（基类加字段） | vllm-ascend 几十个调用点 |
| torch | `torch._inductor` / `torch._dynamo` 等**私有** API 频繁重构 | **F2-path-move**（符号保留，路径搬家）占主导 | torch_npu 60+ 个文件 import `torch._*` |

简单说：vllm 破坏你是偶尔；torch 破坏你是经常，而且多半是"这个符号还在，但换了个位置"。

## 工作流（共 5 步，顺序：manual → KB → skills，不先写脚本）

### 步骤 1 —— 扫描：哪些 torch 私有符号会被移走？

**一键扫描**（`sweep.sh` 会按顺序调用四个 scanner）：

```bash
cd <easyr1-npu-repo>
./src/skills/torch-npu/port-expert/scripts/sweep.sh \
  --baseline v2.11.0 \
  --target v2.12.0-rc3 \
  --pt-repo /path/to/community-pytorch-checkout \
  --torch-npu-path /path/to/torch-npu-checkout
```

或**分步调用**（都在 `src/skills/torch-npu/port-expert/scripts/`）：

```bash
# 第一步：提取 torch_npu 里所有 from torch._* import 对
cd <easyr1-npu-repo>
python3 src/skills/torch-npu/port-expert/scripts/extract_imports.py \
  --root /path/to/torch-npu-checkout/torch_npu \
  > /tmp/imports_by_module.txt

# 第二步：F1/F2-path-move 扫描（符号是否还在旧路径）
python3 src/skills/torch-npu/port-expert/scripts/check_drift.py \
  --pt-repo /path/to/community-pytorch-checkout \
  --pairs-file /tmp/imports_by_module.txt \
  --baseline-tag v2.11.0 \
  --target-tag v2.12.0-rc3 \
  --out /tmp/drift.json

# 第三步：F3 签名变化扫描
python3 src/skills/torch-npu/port-expert/scripts/check_sig_drift.py \
  --pt-repo /path/to/community-pytorch-checkout \
  --pairs-file /tmp/imports_by_module.txt \
  --baseline-tag v2.11.0 \
  --target-tag v2.12.0-rc3 \
  --out /tmp/sig_drift.json

# 第四步：F7/F8 基类新属性/新方法扫描（AST class-scope）
python3 src/skills/torch-npu/port-expert/scripts/check_f7_f8.py \
  --pt-repo /path/to/community-pytorch-checkout \
  --torch-npu-path /path/to/torch-npu-checkout \
  --baseline-tag v2.11.0 \
  --target-tag v2.12.0-rc3 \
  --out /tmp/f78_scan.json
```

所有 4 个脚本的 CLI 共用 `--pt-repo` / `--baseline-tag` / `--target-tag` flag，便于串联调用。

输出结果：
- `check_drift.py`：列出所有"baseline 有但 target 没有"的符号，每条带 drift 类型（`at-original` / `submodule` / `not-here` / `mod-gone`）
- `check_sig_drift.py`：列出签名变化，按 cosmetic / additive-with-defaults / potentially-breaking 分类
- `check_f7_f8.py`：列出 torch_npu 子类化的 parent 类上新增的公开属性/方法
- `check_sig_drift.py`：列出签名变化，区分 cosmetic-only（PEP 604）/ additive-with-defaults（非破坏性）/ potentially-breaking（需要修）

2026-04-24 v2.11.0→v2.12.0-rc3 实际扫描结果：437 个 (mod, symbol) 对 → 1 个真实 F1 drift（`torch._inductor.codecache::Union`）+ 37 个签名变化（21 cosmetic + 7 additive + 9 可能破坏但绝大多数也是 cosmetic typing 没真破坏）。

**高风险扫描顺序**（torch_npu 历史 import 最多的私有模块，先扫这些）：

1. `torch._inductor.utils` (~29 sites)
2. `torch._inductor.virtualized` (~20)
3. `torch._inductor.codegen.triton` (~15)
4. `torch._dynamo.utils` (~12)
5. `torch._inductor.ir` (~11)
6. `torch._inductor.codegen.common` (~10)
7. `torch._inductor.codecache` (~10)
8. `torch._inductor.scheduler` (~9)
9. `torch._inductor.codegen.simd` (~9)
10. `torch._dynamo.device_interface` (~9)
11. `torch._C` (root) (~9)

### 步骤 2 —— 读 KB 对应的 fix template

对应 pattern：**F2-path-move**（在 vllm-ascend port-expert 的 KB 里定义，torch_npu 复用同一条）。

位置：[`src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md`](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) §F2-path-move

模板简化版：

```python
# torch_npu/compat/<symbol_group>.py
try:
    # 旧 torch 版本路径（比如 torch 2.11 仍在老位置）
    from torch._inductor.utils import FloorDiv, ModularIndexing  # type: ignore
    _SOURCE = "torch._inductor.utils"
except ImportError:
    # 新 torch 版本路径（torch 2.12 搬家到这里）
    from torch.utils._sympy.functions import FloorDiv, ModularIndexing  # type: ignore
    _SOURCE = "torch.utils._sympy.functions"

__all__ = ["FloorDiv", "ModularIndexing"]
```

**关键约定**：
- 放 `torch_npu/compat/<module>.py`，不是散在 `torch_npu/_inductor/` 各处
- `_SOURCE` 字段可选但建议保留，方便 debug 时看走了哪条分支
- 同一批移动的符号（比如 sympy 相关的一组）建议放同一个 compat 文件

### 步骤 3 —— 改 import 点

```bash
sed -i 's|from torch._inductor.utils import ModularIndexing, FloorDiv|from torch_npu.compat.sympy_functions import ModularIndexing, FloorDiv|' <affected_files>
```

注意：如果原 import 行既有搬走的符号又有留在原位的符号，要拆成两行。例如：

原：`from torch._inductor.utils import sympy_index_symbol, ModularIndexing, FloorDiv, sympy_product`

改后：
```python
from torch._inductor.utils import sympy_index_symbol, sympy_product
from torch_npu.compat.sympy_functions import ModularIndexing, FloorDiv
```

### 步骤 4 —— 验证（/drift-port-validate skill）

调用验证 skill，在 A3 container 里 stub torch 环境两次（OLD + NEW）分别确认 shim 走对分支：

```
/drift-port-validate patch-summary="F2-path-move shim for FloorDiv+ModularIndexing at torch_npu/compat/sympy_functions.py, 3 callsites swapped"
```

看 skill 文档：[`src/skills/_shared/drift-port-validate/SKILL.md`](../../src/skills/_shared/drift-port-validate/SKILL.md)

9/9 checks PASS 才算过。

### 步骤 5 —— 推 fork 分支 + 提 PR

约定分支名：`<target-torch-version>_auto_porting`，例 `torch-2.12_auto_porting`。

**特殊说明**：torch_npu 的 upstream 在 gitcode.com/Ascend/pytorch。如果你还没有 gitcode fork，先创（需要交互式 web 操作）。PR 走 gitcode merge request 流程。

Commit message 里写明：
- 上游 torch 版本区间（`v2.11.0..v2.12.0-rc3`）
- 对应 KB family（F2-path-move）
- 动过的 torch_npu 源文件
- 引用 KB 案例注册行

## 参考执行（2026-04-24，首次 torch_npu drift port）

目标：torch v2.11.0 → v2.12.0-rc3

流程：
1. 扫描 torch_npu v2.11.0 分支，共 66 个文件 import `torch._*`
2. 抽 `torch._inductor.utils` 的 18 个符号，对比 v2.12.0-rc3
3. 发现 2 个漂移：`FloorDiv` + `ModularIndexing` 从 `torch._inductor.utils` 搬到 `torch.utils._sympy.functions`
4. 定位 3 个 torch_npu 源文件受影响
5. 按 F2-path-move 模板写 `torch_npu/compat/sympy_functions.py`
6. `sed` 改 3 个文件的 import
7. py_compile 5 个文件 PASS
8. 本地 commit `2d81f06c8` on `torch-2.12_auto_porting`（gitcode fork 待创建再 push）

**这是"manual → KB → skills"闭环的首个 torch_npu 案例**。KB 里已登记：[`src/skills/torch-npu/port-expert/references/KB_INDEX.md` §"2026-04-24 torch 2.11 → 2.12 inductor path drift"](../../src/skills/torch-npu/port-expert/references/KB_INDEX.md)

## 下一步延伸

- ~~**scanner 脚本**：给 torch_npu 写对应的 `kb_drive_test.py`~~ ✅ 已完成（2026-04-24）：三步扫描链路已就位，见上文步骤 1
- **F3/F5 检测**：torch._inductor 函数签名变化（F3）和 IR class 重构（F5）是下一步
- **C++ ABI 漂移**：torch_npu 有 `torch._C` 层的 9 个 import，这些可能遇到 F9 族（ABI 变化）。F9 族还没在 KB 里建立

## 相关文档

- [`src/skills/torch-npu/port-expert/references/KB_INDEX.md`](../../src/skills/torch-npu/port-expert/references/KB_INDEX.md) —— torch_npu 专用 KB 索引 + 案例注册
- [`src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md`](../../src/skills/vllm-ascend/port-expert/references/patterns/domains/vllm-api-drift.md) —— 通用 drift pattern 家族（F1-F8 + F2-path-move）
- [`src/skills/_shared/drift-port-validate/SKILL.md`](../../src/skills/_shared/drift-port-validate/SKILL.md) —— 验证 skill
- [`docs/vllm-ascend/PORTING-GUIDE.md`](../vllm-ascend/PORTING-GUIDE.md) —— 姐妹指南（vllm-ascend 端）
