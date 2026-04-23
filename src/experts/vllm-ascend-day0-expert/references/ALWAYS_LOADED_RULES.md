# vllm-ascend-day0-worker 专属规则

> 读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` 后再读本文。

## OL-03 denylist

**禁读**：
- 通用：`docs/HANDOVER.md` / `docs/porting-journal.md` / `docs/*-drill.md`
- 其他 expert 的 workspace：`workspace/torch-day0-*` 只读 deploy-/manual-/
  analysis- 里的 PROGRESS.md 和 ONBOARDING.md，不读 analysis 里的
  intermediate artifacts（太长，误导）；`workspace/vllm-day0-*`、
  `workspace/transformers-day0-*`、`workspace/*-upgrade-*`、
  `workspace/easyr1-port-*`
- `upstream/EasyR1` 上 `ascend-port*` / `round*` 分支（easyr1-expert 的域）

**允许读**：
- 本 expert 自己的 `references/**`
- **vllm-ascend 源码**（`upstream/vllm-ascend/`，所有 ref） —— 这是本
  expert 的工作对象
- **vllm 源码**（`upstream/vllm/` / image 里的 `/vllm/`） —— 用于定位
  consumer 的 import 链、cached env var 点
- `torch_npu` 源码（`upstream/torch-npu/`） —— 当 delta 涉及 torch_npu
  API 时读其 API surface
- torch 源码（WebFetch / 镜像） —— 用于理解 dispatcher / ABI 变化
- **上游 Day-0 session 的 ONBOARDING.md 和 PR material** —— 这是 base
  image 的契约，必读

## OL-08 edit scope

**可写**：
- `$WORKSPACE/**`
- `upstream/vllm-ascend/**/*.py` 在 `ascend-day0-<TARGET_DELTA>-<SESSION_TAG>`
  分支上 —— 本 expert 的主要交付物
- `upstream/vllm-ascend/vllm_ascend/**/*.py` 允许新增文件（helper
  函数等），但 **不要**重写架构级模块；最小 invasive。
- 一次性临时 COPY 补丁文件（Dockerfile.overlay-vllm-ascend-fix*）在
  `$WORKSPACE/<deploy-dir>/`

**禁写**：
- 社区 `upstream/vllm/**` —— 社区决定，我们不改
- 社区 `upstream/pytorch/**` —— 同上
- 其它华为开源仓（`upstream/torch-npu/`、`upstream/triton-ascend/`、
  `upstream/transformers/src/transformers/integrations/npu_*`） —— 那些
  是 **其它** day-0 expert 的 C-patch 域；如果 delta 真的要到那边修，
  说明 target 选错了或者 session 要先回到上游 expert
- `src/experts/**` 自身
- `vllm_ascend_C.cpython-*.so` —— 这是编译产物，靠 rebuild 不是 patch
  手写；C++ 改动属于 Fix C tech debt 不归本 session

## Outcome 矩阵

| Outcome | 含义 | 行动 |
|---|---|---|
| A | 存在的 vllm-ascend 已处理（可能 main tip 修过了，image 太旧） | 只 ship overlay + notes；切换建议用户升级 vllm-ascend |
| B | 不改代码，靠 env-var / CLI flag 配置解决 | ONBOARDING 写用法，KB 记 |
| C-patch | 改 vllm-ascend 源码，smoke PASS | ascend-day0-<delta>-<TAG> 分支，V1.3 smoke PASS，PR material |
| C-report | 根因在社区上游，我们无权改 | blocker-report.md |

## Fix level 选择顺序

同一个 bug 可能有多层 fix。按**最小 invasive**顺序试：

1. **Env-var / CLI flag** — 不改 vllm-ascend 任何代码，只靠消费者配置
   环境变量（如 `VLLM_BATCH_INVARIANT=1` 关掉 broken 的 custom op 路径）
2. **Python-layer guard at plugin entry point** — 在 `vllm_ascend/__init__.py`
   自动设 env var，让用户无感；同时在 `enable_custom_op()` 或相邻
   checkpoint 加 defense-in-depth
3. **Python-layer source edit in affected call site** — 如 forward_oot
   添加 version-gated branch
4. **C++ source edit** — 需要重编 `.so`，属于长期 Fix C，不归本 session

Level 1-3 是本 session 的工作；Level 4 记进 PR material 的
"follow-up work" 和独立的 tech debt task。

## Target delta 来源

本 expert 通常**不自己选 target**，而是从上游 Day-0 session 的 handoff
接到：
- torch-day0 session 说"torch 2.11 layer OK，vllm-ascend C++ ext ABI
  对不上" → 进来做 torch-2.11 delta
- vllm-day0 session 说"vllm 0.20.0 删了 symbol X 但 vllm-ascend main
  还 import 它" → 进来做 vllm-0.20.0 delta

从 ONBOARDING.md 的 "known-broken" 段读 base image 的已知问题；那通常
就是本 session 的 target。

## 特殊的 vllm-ascend gotcha（已踩）

- **`vllm_is_batch_invariant()` 是 cached-at-import**：vllm 的
  `vllm/model_executor/layers/batch_invariant.py` 在 line 997 把
  `VLLM_BATCH_INVARIANT` env var 读到 module-level 常量。所以 env
  var 必须在**任何 vllm 模块 import 之前**设好才有效。在
  `enable_custom_op()` 里面设（那时 vllm 已经 imported）**太晚**。
  解决：在 `vllm_ascend/__init__.py` 做，plugin entry point 在 vllm
  导入之前就触发。
- **editable install 给了我们机会**：base image 里 vllm-ascend 是
  `pip install -e`（`Editable project location: /vllm-ascend`），
  COPY 覆盖 `/vllm-ascend/vllm_ascend/*.py` 就生效，不需要 pip
  reinstall。
- **C++ extension 段错误看起来像 Python bug**：`.so` 能 load +
  `TORCH_LIBRARY_IMPL` 注册能成功，但第一次 op call 就 SIGSEGV。
  Python stack 顶部会显示 `torch._ops.py:1269 __call__`，容易误以为
  Python 层出 bug。检查方法：`docker run --rm <image> python3 -c 'import
  vllm_ascend.vllm_ascend_C; print(dir(torch.ops._C_ascend))'` —
  `name` 以外 attr 存在但是 call 时炸 = C++ ABI drift。

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: vllm-ascend-day0-worker <ISO-8601-UTC>`
