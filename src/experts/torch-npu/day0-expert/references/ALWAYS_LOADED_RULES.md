# torch-day0-worker 专属规则

> 读 `../../_shared/references/ALWAYS_LOADED_UNIVERSAL.md` 后再读本文。

## OL-03 denylist

**禁读**：
- 通用：`docs/_meta/HANDOVER.md` / `docs/easyr1/porting-journal.md` / `docs/_archive/P2-WORKFLOW.md`
  / `docs/easyr1/DELIVERABLE.md` / `docs/_archive/codex-*.md` / `docs/_meta/design.md` /
  `docs/easyr1/dep-matrix.md` / `docs/easyr1/PORT-SUMMARY.md` / `docs/easyr1/easyr1-dep-chain-audit.md`
  / `docs/_archive/handoff-2026-04-19.md` / `docs/_archive/skill-dry-run-2026-04-20.md` /
  `docs/transformers/UPGRADE-DRILL-STATUS.md` / `docs/transformers/transformers-upgrade-drill.md`
- `upstream/EasyR1` 上 `ascend-port*` / `round3-*` / `round4-*` /
  `ascend-port-e2e-*` 分支
- 其他 expert 的 workspace：`workspace/vllm-day0-*/`、
  `workspace/transformers-day0-*/`、`workspace/vllm-upgrade-*/`、
  `workspace/transformers-upgrade-*/`、`workspace/torch-npu-upgrade-*/`（**
  这一项特别注意**：torch-npu-upgrade-expert 的结论假定已有匹配版本 ship
  到 image，Day-0 场景正好相反，读它的结论会误导）、
  `workspace/easyr1-port-*`、`workspace/dep-analysis-*`、`workspace/npu-port-*`

**允许读**：
- 本 expert 自己的 `references/**`
- 消费者 repo 源码（UPSTREAM_REF）
- **上游 PyTorch + Ascend/pytorch 源码、release notes、CHANGELOG**
  （WebFetch / git fetch / pip download） —— Day-0 证据的核心源头
- **PyPI torch-npu release history**（rc wheel 是 Day-0 的 target）
- **CANN release notes / 版本配对表**（确定 rc wheel 对应的 CANN 版本）
- `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` 的 **Day-0 reframing 段**

## OL-08 edit scope

**可写**：
- `$WORKSPACE/**`
- `upstream/<consumer>/Dockerfile.overlay-torch<M><m>*` —— overlay Dockerfile
- `upstream/<consumer>/requirements*.txt` —— 只松 torch / torch_npu 版本上
  界（fixture 路径），不碰其它 dep
- **如 outcome C-patch**：
  - `upstream/torch-npu/**/*.py` 以及 `upstream/torch-npu/torch_npu/csrc/**`
    在 `ascend-day0-torch<M><m>-<SESSION_TAG>` 分支上 —— 产出 torch_npu
    upstream patch 是本 skill 的正当交付物（skill 的受众就是 Ascend/pytorch
    team）
  - 同样适用于其它华为开源适配层：`upstream/triton-ascend/**`（如果 Day-0
    触发 triton-ascend 的适配 gap）、`upstream/transformers/src/transformers/integrations/npu_*`
  - **不改**社区 PyTorch 本身（`upstream/pytorch/**`，社区的决定我们无权
    替 rep）
  - patch 要走 git branch，不直接 mutate image 内 `.py`；overlay image
    通过 `pip install git+...@<branch>` 或 `COPY <patched-files>` 把
    patch 装进去

**禁写**：
- 任何 `verl/**/*.py`（是 easyr1-expert / 其它 day0 expert 的 domain）
- `Dockerfile.npu` / `Dockerfile.npu-852` / `Dockerfile.overlay-vllm*` /
  `Dockerfile.overlay-trans*` —— 其它 expert 的 domain
- 社区 PyTorch (`upstream/pytorch/**`) 本身
- `src/experts/**` 自身

## Outcome 矩阵

| Outcome | 含义 | 该做什么 |
|---|---|---|
| A | 直接 pip overlay + runtime smoke 6/6 PASS | 写 PROGRESS + 部署 artifacts + ONBOARDING |
| A-with-note | smoke PASS 但 dep 树中有 known gap 不在 smoke 覆盖内 | ship overlay + 在 ONBOARDING 的 known-broken 段里记 |
| B | smoke fails on consumer-side pin loosen | 放松 requirements + smoke PASS |
| C-patch | smoke FAIL，修 torch_npu（或其它华为开源）能解 | 在 `upstream/torch-npu/` 开 `ascend-day0-torch<M><m>-<TAG>` 分支改，rebuild overlay 装 patch，smoke PASS，打包 PR-ready diff |
| C-report | fix 要社区 PyTorch 改；或 fix 超出 skill 领域 | blocker-report 最小复现 + suggested fix 交给 Ascend/pytorch team |

**目标是 A / A-with-note / C-patch 且 PASS**。C-report 只在真没法自己
解的时候用。

## Target 选择的 pre-probe（必做）

在 `pip overlay torch==<TARGET>` 之前，先 probe：

1. `cd upstream/torch-npu && git fetch origin --tags`
   - `git tag --list '*<TARGET>*'` 看有没有对应 release tag（只有 source
     tag 不算 stable，有 `v<X>.<Y>.<Z>` GitHub release 才算）
   - `git log origin/main -S '<关键 symbol>' -- setup.py requirements.txt`
     查 Ascend/pytorch main 有没有适配痕迹
2. PyPI: `pip index versions torch-npu` 或 `curl pypi.org/pypi/torch-npu/json`
   —— 看有没有 rc wheel（Day-0 session 的 install 对象）
3. CANN 版本配对：`upstream/torch-npu/README.md` 的兼容表 + base image
   的 CANN 版本 (`knowledge/images/<image>.md`)。**必须匹配或 base 版本
   ≥ rc 要求的一个 patch 以内**（e.g. rc1 标 CANN 8.5.0，base 有 8.5.1
   可接受；base 有 8.4.0 或 8.6.x 要 re-probe）。
4. **Stop 条件**：
   - Ascend/pytorch main 已经适配 + 有 stable release → session 没
     skill 价值，换更新 target
   - rc wheel 不存在 → session 被 Ascend release 节奏阻塞，emit
     advisory，标 deferred
   - base image CANN 和 rc requirements 差距超过一个 patch → fail-fast
     with "need base image refresh first"

记这步的发现到 PROGRESS.md 的 Phase A。

## 特殊的 torch-day0 gotcha（已踩）

- **PyTorch 2.11 `_import_device_backends()`**：`import torch` 会 eager
  加载所有注册的 device backend（含 torch_npu），触发 CANN lib 的
  runtime link。**docker build 容器里没 mount CANN devices → build-time
  `import torch` 会炸 `ImportError: libascend_hal.so`**。
  - 解决：build-time 不做 `import torch`；用 `python3 -m py_compile`
    或 `ast.parse` 做语法检查；runtime 再做 import smoke。
- **C++ extension ABI drift**：torch 2.11 的 dispatcher 内部布局变了，
  pre-2.11 编译的 `.so`（vllm_ascend_C, 其它第三方 PrivateUse1 扩展）
  会在 op-call 时 SIGSEGV，而不是 import-time 报错。Phase 5（下游 expert）
  是专门的 C-patch 领域，torch-day0 只负责把 torch 层跑通。
- **rc wheel 版本约束**：PyPI `torch-npu==2.11.0rc1` 的 `requires_dist`
  硬 pin `torch==2.11.0`。安装 combo 要用 `torch==2.11.0+cpu`（aarch64
  或 x86 看 image）从 `download.pytorch.org/whl/cpu/`，不要用社区 dev
  build 对抗 rc pin。

PreToolUse hook 拦违规 Edit。

## 退出签名

`Worker signed: torch-day0-worker <ISO-8601-UTC>`
