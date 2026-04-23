# Example: Day-0 NPU support for transformers 5.6.0 (zero-interaction)

> **Status**: 实测 2026-04-23，transformers 5.6.0（社区 2026-04-22 发布，NPU
> 生态原本没跟上）通过 `/transformers-day0` skill 在 NPU 上跑通 V1.1+V1.3+V1.4，
> step-1 entropy_loss=1.310156412422657，在 v2 baseline band [1.21, 1.34] 内。
> 全程零人为介入。

## 场景

你是 EasyR1 用户：
- 消费者（EasyR1 / 你自己的 RL 框架）想用 community 最新 `transformers==5.6.0`
- 当前 NPU base image（`quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5`）
  ship 的是 transformers 5.3.0.dev0
- NPU 生态没 ship 5.6.0 适配镜像
- 想知道：**能不能用？能用就直接用；不能用的话缺哪块？**

这正是 `/transformers-day0` skill 解决的 Day-0 场景。

## 零交互调用（3 步准备 + 1 步 skill）

### 步骤 1：消费者 repo 放松 transformers 上界（fixture）

如果消费者的 `requirements.txt` 把 transformers 钉在 `<5.0.0`，先在一个
fixture 分支把上界放松（这一步用户自己 commit，不是 skill 做）：

```bash
cd /home/$USER/workspace/easyr1-npu/upstream/EasyR1
git checkout -b fixture/transformers-loosened-for-day0
sed -i 's|transformers>=4.54.0,<5.0.0|transformers>=4.54.0,<6.0.0|' requirements.txt
git add requirements.txt
git commit -m "fixture: loosen transformers upper bound for day0 probe"
```

Commit hash（你的会不同）：`d448278`

### 步骤 2：Pre-flight（orchestrator / 用户手动跑一次）

```bash
# chip precheck (OL-05)
ssh -p 443 root@115.190.166.102 'for i in 0 1; do npu-smi info -t proc-mem -i $i 2>&1 | grep "Process id" | head -1; done'
# empty = idle, OK to proceed

# disk headroom (overlay 约 +1GB)
ssh -p 443 root@115.190.166.102 'df -h / | tail -1'
```

### 步骤 3：调用 skill（零交互核心步骤）

```
/transformers-day0 \
    --target-transformers-version 5.6.0 \
    --base-image easyr1-npu-852:trans-upg-e2e-20260422-2200 \
    --upstream-ref fixture/transformers-loosened-for-day0
```

（不带 `--session-tag` 时 skill 自动生成；不带 `--base-image` 时默认 v2。）

**就这一条命令。skill 剩下全部自己做**。

## Skill 实际执行什么（用户不必 memorize，仅供理解）

| Phase | Skill 动作 | ~耗时 |
|---|---|---|
| A (Analyze) | docker run pip install 5.6.0 进 base image，probe `transformers.integrations.npu_flash_attention` sig + ALL_ATTENTION_FUNCTIONS 扩张 / 删减 + modeling_utils 私有 hook | ~3min |
| B (Decide) | 根据 drift 决 A/B/C：A=works-as-is / B=forward-port 适配器 / C=blocked | 即时 |
| C (Build overlay) | 写 `Dockerfile.overlay-trans56`（`FROM base-image` + `pip install transformers==5.6.0`），git commit，docker build 到新 image tag | ~3min（pip overlay 很快） |
| D (Smoke) | V1.1 chip 0（device + pad/unpad）→ V1.3 chip 0（rollout）→ V1.4 chips 0,1（2-step GRPO，v2 band [1.21,1.34]） | ~15min |
| E (Exit) | PROGRESS.md 签字、cleanup 保留 overlay image、emit Handoff JSON | ~1min |

Total 约 25 min。

## 实测 2026-04-23 结果（Day-0 outcome A 的具体数字）

- overlay image: `easyr1-npu-trans56:trans-day0-wetrun-20260423-0109` (sha 509d9dce7292, 27GB)
- V1.1 marker: `ALL SMOKE CHECKS PASSED` ✓
- V1.3 marker: `V1.3 ROLLOUT SMOKE PASSED` ✓
- V1.4 step-1 entropy_loss: **1.310156412422657** in v2 band [1.21, 1.34] ✓
- V1.4 step-2 entropy_loss: 1.286271445453167
- V1.4 step-2 val reward: 0.01799999736249447

结论：**transformers 5.6.0 在 NPU 生态上 Day-0 可用**，数值与 v2 image
原生的 5.3.0.dev0 基本一致。vllm 0.18 的 `transformers<5.0.0` pip pin 是
metadata-only，runtime 接受 5.6。

## 如果 skill exit 不是 outcome A 怎么办

### Outcome B (forward-port)

skill 会自动把改过的 `npu_flash_attention.py` 作为补丁 `COPY` 进
overlay image，再重 smoke。你看到 `provenance: transformers-day0-worker`
+ `shims_applied: [...]` 就是它在 forward-port 了。一般 1 个 kwarg
兼容 / 1 个 attention-key handler。

### Outcome C (blocked)

skill 不会硬着头皮上。它会在 `$WORKSPACE/blocker-report.md` 精准指出
到底缺哪块（torch_npu 的某个 op / vllm-ascend 的某个 API / transformers
NPU FA 的某个 hook）。这是你拿去和 NPU 团队 / upstream 上报的**证据**。
blocker 报告示例：

```
blocker: transformers.integrations.npu_flash_attention.some_new_hook
required_by: Gemma4 model default attn_implementation on 5.6
npu_ecosystem_status: not yet published
workaround: attn_implementation='sdpa' in consumer config OR wait
```

Skill **不会**虚构可用结果；如果 blocker 真的存在，它会如实 C。

## 反作弊校验（用户可以自己做）

skill 跑完后，**不要只信 Handoff JSON**。下面 3 条命令独立拉磁盘，
对比 skill 报的数字：

```bash
# 1. overlay image 真的存在
ssh -p 443 root@115.190.166.102 "docker images | grep <SESSION_TAG>"

# 2. V1.4 jsonl 里的 entropy_loss 和 skill 报的一致（16 位数字 match）
ssh -p 443 root@115.190.166.102 "cat /home/z00637938/workspace/easyr1-npu/upstream/EasyR1/checkpoints/easy_r1/v14_smoke/experiment_log.jsonl | head -3" | python3 -c "
import json,sys
for line in sys.stdin:
    r = json.loads(line.strip())
    if r.get('actor', {}).get('entropy_loss'):
        print(f\"step {r['step']}: entropy_loss={r['actor']['entropy_loss']}\")
"

# 3. V1.1 / V1.3 markers 真的在 log 里
ssh -p 443 root@115.190.166.102 "grep -c 'ALL SMOKE CHECKS PASSED' /tmp/z00637938/easyr1-logs/V1.1-...-<SESSION_TAG>-*.log"
ssh -p 443 root@115.190.166.102 "grep -c 'V1.3 ROLLOUT SMOKE PASSED' /tmp/z00637938/easyr1-logs/V1.3-...-<SESSION_TAG>-*.log"
```

如果三条都核验通过 + Handoff JSON 对得上，该 skill 的这次 run 可信。
这叫 **orchestrator G2 re-verify**，是 skill 架构的抗作弊层。

## 为什么它能零交互

- **skill 自带 KB**：`references/patterns/domains/api-drift-scan.md` 里是
  机械的 probe 协议，agent 按步骤做，不需要人"拍脑袋"。
- **skill 自带决策树**：`KB_INDEX.md §"Quick symptoms → classification"`
  把常见 drift 模式映射到 A/B/C，agent 不需要创造判断。
- **skill 有 Stop hook 守卫**：agent 想跳步 / 伪造 PASS，Stop hook 会拦
  下（G2 container dry-import，G3 必须 cite log + numeric evidence，
  OL-09 必须有 MODE/TARGET/BASE/HANDOFF 字段）。
- **skill 有 PreToolUse hook 守卫**：agent 想改 `verl/**/*.py` 或其它非
  scope 文件会被拦下（OL-08：day0 agent 只能碰 Dockerfile.overlay-trans* +
  `$WORKSPACE/patches/`）。
- **fresh baseline 协议**：如果目标 transformers 是首次测，没有历史 band
  比对，skill 会把测出的数字 record 成"for version X the baseline is Y"，
  之后别人用同版本就有 baseline 用。

## skill 本身的 Day-0 findings（来自本例）

skill 跑这次发现了 2 条模板层缺陷，**已经进 KB**（commit a534033）：

1. Dockerfile 里**不要**在 build-time 做 `from transformers.integrations
   import npu_flash_attention` — 它 trigger torch_npu 去 dlopen
   `libascend_hal.so`，docker build 沙箱没 NPU device 挂载就挂。
   Build-time 只做版本号 check；runtime import 留给 smoke container。
2. NPU-BUG-003（torch.compile inductor crash）在 transformers minor
   bump 后仍然复现；canonical V1.4 smoke 的 `use_torch_compile=false`
   覆盖。新 day0 session 继承这个 workaround 一般不会重踩。

## 下次再来一次更快

一旦本例的 overlay image 在 A3 上（本例 `easyr1-npu-trans56:...`），
后续用户可以直接 `--reuse-image` 而不是重 build：

```
/easyr1-port \
    reproduce \
    --reuse-image easyr1-npu-trans56:trans-day0-wetrun-20260423-0109
```

从 ~25min 缩到 ~15min（省掉 Phase A/B/C）。

## 相关文档

- `docs/_meta/design-subdocs/SKILLS_ARCH_TARGET.md` §"Day-0 Reframing" —— 这个 skill 存在的
  原因：社区有了但 NPU 没跟上的场景才是真问题，shim-adapt 只能解决 NPU 已
  ship 的情况
- `src/experts/transformers/day0-expert/README.md` —— skill 定义 + 当日
  realtime baseline
- `src/experts/transformers/day0-expert/references/patterns/domains/api-drift-scan.md`
  —— probe 协议 + 本例观察到的 ALL_ATTENTION_FUNCTIONS 扩张细节
- `src/experts/transformers/day0-expert/references/patterns/domains/overlay-image.md`
  —— Dockerfile.overlay 模板 + build-time import trap 警告
