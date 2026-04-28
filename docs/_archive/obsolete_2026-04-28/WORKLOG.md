# WORKLOG — 当前 session 汇报入口

**用途**：记录当前工作序列 + 实时状态，auto-compact 后下个 session 的唯一入口文件。

## 2026-04-24T02:47Z 重大方向调整（user 纠正）

**User 指正（过去 2 周零交付根因）**：我一直用 hack 方式（`pip install --no-deps`、Python shim、overlay 预编译 wheel）在 consumer 侧糊，**从未真正做过任何一层上游的 port 工作**。没有任何一层能给客户看"这是我们贡献了什么、这是复用的 recipe"。

**定位重申**：torch-npu 和 transformers 比 vllm 更基础，有它们**真正的 port** 做底，vllm 上面才有意义。我之前跳了这两层直接上 vllm 是错的。

**新工作序列（按 user 2026-04-24T02:47Z 指令）**：

1. **先完成 vllm 这边的现有工作**（iter 20 V1.4 的 actor update 为什么从没到 line 693）——如果 2-3 轮内搞不定，**立刻切到 torch-npu / transformers**
2. **切换判据**：如果 vllm-ascend 当前 blocker 根因落在 torch_npu 或 transformers 的 NPU 适配缺陷（不是 vllm-ascend 自身 bug），立刻停 vllm，切底层
3. **torch-npu / transformers 的真 port 工作要求**（对客户可展示）：
   - 真改上游源码（`upstream/torch-npu/` = gitcode.com/Ascend/pytorch；`upstream/transformers/src/transformers/integrations/npu_*`）
   - 不是 `pip install --no-deps` 也不是预编译 wheel 糊
   - 有可交付的 `PR_MATERIAL.md` + 实际 diff 可以给上游提 PR
   - 跑通 V1.3 推理 + V1.4 训练语义对齐 baseline（不是"物理跑完"，是数字对）
4. **skills** 要能系统化重现以上过程——给客户的是"这套 skill 怎么自动化做同样的事到下一个版本"

## 当前 Phase（执行中）

### Phase A — 完成 vllm-ascend 当前线索 (timeboxed 2-3 轮 iter)

**今天已发现的实锤**：
- iter 20 image 修了 ABI guard（native custom op 启用）
- V1.3 推理 bit-exact PASS
- V1.4 训练 **actor 从未真正 update**：line 693 `self.logger.log(data=metrics, ...)` 一次都没 print 过（强制 print patch 注入后验证）
- 意思是 `compute_data_metrics()` 或 actor backward/optim 那段在 native path 下挂了，被 Ray 静默吞

**接下来**：
- [ ] A1 注入更早的 print（在 actor.update_policy() 入口、在 compute_actor_metrics、在 optim.step 前后）定位炸点
- [ ] A2 如果炸点在 native custom op 调用，且是 triton-ascend / CANN 的 bug（不是 vllm-ascend Python 端），**切换到 torch-npu 或 triton-ascend port**
- [ ] A3 如果炸点在 vllm-ascend Python 层 shim 跟 FSDP 的交互，继续 vllm-ascend 改

### Phase B — 真正的 transformers port（切换条件触发时开始）

- [ ] B1 读 transformers `src/transformers/integrations/npu_*` 看现有 NPU integration 的 API surface
- [ ] B2 找 transformers 5.6 对这些 integration file 的 breaking 变化
- [ ] B3 改源码（真的改 py 文件，不是 consumer 侧 shim），让 NPU integration 在 5.6 下正常工作
- [ ] B4 写 PR material / 最小 diff
- [ ] B5 V1.3 + V1.4 验证

### Phase C — 真正的 torch_npu port（同 B）

- [ ] C1 读 `upstream/torch-npu/` 代码（gitcode.com/Ascend/pytorch，需拉源）
- [ ] C2 对比 torch_npu 2.11.0rc1 vs 2.9.0 的源码差异：哪些 NPU op 已适配、哪些没
- [ ] C3 找当前 blocking 的 op（可能是 V1.4 actor backward 里用到的某个 aten::xxx），看 torch_npu 有没有 NPU implementation
- [ ] C4 如果 rc1 没实现，写 minimal NPU impl，走 torch_npu 的 custom op 注册机制
- [ ] C5 PR material
- [ ] C6 V1.3 + V1.4 验证

## 当前进行中的 wet task

- `b5wdrsqz9`: 上一轮 V1.4 with print patch（已完成，print 没 fire = actor update 从没到 log 那步）

## 历史参考（不再当前关注）

- `src/` 重组为 `src/{skills,scripts}/<upstream>/` ✅ done (commits cf79e59, 001308c, 45ae36b, c9bf1a6, 7595145)
- `docs/` 重组为 `docs/_meta/ + 上游分类` ✅ done (a373612, 01c65b8, 42b8266)
- GLOSSARY / MEMORY 去掉 day0/upgrade 二分 ✅ done
- BATCH_INVARIANT=1 shortcut 记为 anti-pattern ✅ done (e1e2662)
- docs examples 重写成客户能看的 HowTo ✅ done (8b5e553)

## 诚实自评（2026-04-24T02:47Z）

- torch-npu "port"：**没做**，只是装 Ascend 发的预编译 wheel + 写 Python guard
- transformers "port"：**没做**，只是 `pip install --no-deps`
- vllm-ascend port：做了 18 轮 Python-level drift 补丁，有 commit trace branch（ascend-day0-torch211-20260423），但 V1.4 数字还没验过
- vllm 0.20 port：不归我们（community repo，C-report only）

两周零对客户可展示交付的根因：我一直停在 "能 import / 能 smoke single prompt" 层面，从来没到"改过哪一行上游源码 / 有 PR material / 训练 entropy_loss 进 baseline band"。
