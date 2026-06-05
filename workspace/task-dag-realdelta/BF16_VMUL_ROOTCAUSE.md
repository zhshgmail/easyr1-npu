# bf16 在 tilelang-ascend 上算错/编不过 —— 根因定位与修改验证

**日期**: 2026-06-02
**触发**: flash_attn bf16 输出错误(73.5% vs fp32 参考),softmax 隔离后定位到向量算术。owner 指令"你要做修改验证"——不是只提 issue,要真改+验。

## 结论(一句话)

bf16 向量乘(`hivm.hir.vmul`)在整条编译链上被 **HIVM dialect 的 `VMulOp` type-constraint 拒绝**,因为该约束漏列 BF16。根因**不在 tilelang-ascend,在 AscendNPU-IR 的 HIVM op 定义**。开源那份(`.td`)我已修复并验证 verifier 放行;但部署用的 **CANN 8.5.1 自带闭源 `hivmc` 0.1.0** 把同样的旧 verifier 编死在二进制里,改不到 —— 这是真正的墙。

## 三道 verifier(逐道验证)

| 层 | 二进制/库 | bf16 vmul 校验(改之前) | 我能改? | 改之后 |
|---|---|---|---|---|
| ① tilelang 进程内 codegen | `libtilelang.so` / `libtilelang_module.so`(静态链 HIVM dialect) | 拒绝 | ✅ 开源 | 重编后**放行** |
| ② `bishengir-compile` 独立 verifier | `build/bin/bishengir-compile`(开源 AscendNPU-IR) | 拒绝 | ✅ 开源 | 重编后**放行** |
| ③ `hivmc` 后端 verifier | `/usr/local/Ascend/cann-8.5.1/tools/bishengir/bin/hivmc`(闭源,0.1.0/2026-02-13) | **拒绝** | ❌ 闭源 | **撞墙** |

## 修复内容(开源可改的部分)

`3rdparty/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td`:

```diff
 def VExpOp : HIVM_ElementwiseUnaryOp<"vexp",
-     OperElemTypeConstraints<[0, 1], [F16, F32]>,
+     OperElemTypeConstraints<[0, 1], [F16, BF16, F32]>,

 def VMulOp : HIVM_ElementwiseBinaryOp<"vmul",
-     OperElemTypeConstraints<[0, 1], [I16, I32, F16, F32, I64]>,
+     OperElemTypeConstraints<[0, 1], [I16, I32, F16, BF16, F32, I64]>,
```

VExpOp 改动严格说不是 bf16 路径必需(exp2 lowering 里 vexp 已被 cast 到 f32 跑),但 vexp 本身在其他 kernel 可能直接 bf16,一并补上更一致。真正卡住 exp2-bf16 的是 **VMulOp**(ln2 标量乘那步)。

## 验证过程(关键证据)

1. **IR dump 定位**:exp2-bf16 lower 出的 IR 是 `vmul(bf16) → vcast bf16→f32 → vexp(f32) → vcast f32→bf16`。vexp 已 f32,以 bf16 执行并报错的是 **vmul**。(避免了"改错 op"——第一次确实改错改了 vexp。)
2. **逐层重编验证**:
   - 重编 `bishengir-compile` 二进制 → 不够,因为 tilelang 进程内校验走 `libtilelang*.so`,没 shell out。
   - `ninja install`(把改后的 HIVM dialect `.a` + 头文件推到 `build/install/`)+ 重编 `libtilelangir.so` / `libtilelang.so` / `libtilelang_module.so` → 进程内 verifier 放行(`CodeGenTileLangNPUIRAPI failed verification` 报错消失)。
   - 随后流程推进到 `bishengir-compile` 子进程 → 我重编的二进制 verifier 也放行。
   - 最终撞 `hivmc`:`error: 'hivm.hir.vmul' op failed to verify that operand at idx 0 and 1 should have element type [i16, i32, f16, f32, i64]`(无 BF16)。
3. **hivmc 闭源铁证**:`strings hivmc | grep` 显示其 vmul verifier 合法类型列表无 BF16;但**同一 hivmc 的单目 op 错误信息含 `bfloat16`**(bf16 字样出现 467 次)——说明不是"硬件不支持 bf16",而是**这个老 hivmc 的 vmul verifier 那一处遗漏 BF16**。
4. **上游核对**:gitcode `Ascend/AscendNPU-IR` master 最新版 `VMulOp`/`VExpOp` **仍然漏 BF16** —— 是当前真实的上游缺口,不是已修复待发布。

## 诚实的未决项(不能过度声称)

- ✅ 已验证:三道 verifier 中开源的两道,加 BF16 后**放行**(编译期不再拒绝)。
- ❌ **未验证**:bf16 vmul **数值是否正确**。被闭源 hivmc 0.1.0 挡住,拿不到 device 上的 bf16 vmul 结果。无法排除"老 hivmc 的 vmul verifier 故意不列 BF16,是因为后端 AICore 对 bf16 双目乘有真实限制"这一可能(虽然单目 op 含 bfloat16 让"纯遗漏"更可能)。
- 因此**不能声称"bf16 已修复可用"**,只能声称"verifier 层根因已定位+开源部分已修复放行;功能正确性需在带匹配新版 hivmc 的 CANN 上由维护方验证"。

## CANN 版本核查(owner 建议的 9.1.0 beta)

用 Playwright 抓 hiascend.com 社区下载页确认:
- 部署机当前:CANN 8.5.1,其 `hivmc` = **0.1.0 / 2026-02-13**。
- 社区最新:**CANN 9.1.0-beta.1 / 2026-05-15**,有完整包(`Ascend-cann-toolkit_9.1.0-beta.1_linux-aarch64.run` 含 bishengir/hivmc + `Ascend-cann-nnal_*.run` + 各产品 ops 包)。比部署版新很多。
- **必要但不充分**:升级 CANN 能解掉 flag 兼容(新 hivmc 认新 flag);但 bf16 vmul verifier 缺口在**所有当前 AscendNPU-IR ref 上都存在**:
  - master(`de20e14a`):VMulOp/VExpOp 仍 `[…,F16,F32,…]`,无 BF16
  - **最新稳定 tag `v1.1.0-post2`(`428ab8fd`):同样无 BF16**
  - 部署的 hivmc 0.1.0:无 BF16
  → 因此 **CANN 9.1.0-beta 若从 master 或 v1.1.0-post2 构建,其 hivmc 仍会拒绝 bf16 vmul,升级 CANN 单独不解决**。除非 9.1.0-beta 从某个含修复的未公开分支构建(无法确认)。**唯一可靠解 = 上游接受修复(PR #1199)并随之进入 CANN 构建。** 故无需下载 gated 的 CANN toolkit 验证 —— 跨所有 ref 的核查已给出高置信结论。

## 解法路径

1. **升级 CANN 9.1.0-beta**:解 flag 兼容 + 大概率带更新的后端;但需 `strings` 新 hivmc 核实 vmul verifier 是否已含 BF16(master 仍漏,故不保证)。
2. **上游 PR**(根本解):向 `Ascend/AscendNPU-IR` 提 `.td` 补 BF16 的 PR。PR body 必须诚实标注"verifier 放行已验证,bf16 vmul 数值正确性请维护方在带新 hivmc 的环境验证;参照同文件单目 op verifier 已含 bfloat16"。无 agent 签名,套用 repo issue/PR 模板。

## 产物

- `bf16_verifier_fix.patch` —— 2 行 `.td` 修复(PR 内容)
- `hivmc_vmul_verifier_proof.txt` —— hivmc 闭源二进制 verifier 字符串证据
- `test{2..6}_out.txt` —— 逐层重编后的 verifier 推进证据链
- 重编产物(留在 tlrescue):`build/bin/bishengir-compile`、`build/libtilelang*.so`(均含 BF16 verifier)
