# tilelang × tilelang-ascend × tilelang-mlir-ascend × AscendNPU-IR
# 完整架构 + 时序解析

> 解释 tile-ai 组织下的三个 tilelang 相关 repo 和 Huawei 的 AscendNPU-IR
> 之间的关系，以及一段 tilelang Python 代码到 NPU 二进制经过的所有步骤。
> 所有结论基于今天（2026-05-17）拉到的源码。

---

## 1. 四个 repo 的真实关系（修正版）

```
                          tile-ai / TileLang (上游 DSL + TVM 工具链)
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │   ascendc_pto branch     │  ◄── tilelang-ascend 这条线
                          │   后端：String codegen    │     字符串拼 AscendC C++
                          │   编译器：ccec           │     已 release，wheel
                          │   3rdparty: tvm/cutlass/  │
                          │     catlass/pto-isa/shmem│
                          └──────────────────────────┘
                                       
                          ┌──────────────────────────┐
                          │   npuir branch           │  ◄── tilelang-ascend 这条线
                          │   等价于下面的            │     （这两个 repo 是镜像！）
                          │   tilelang-mlir-ascend   │     
                          └──────────────────────────┘

           ┌─────────────────────────────────────────────────┐
           │  tile-ai / tilelang-mlir-ascend (独立 repo)      │
           │  后端：MLIR + bishengir                          │
           │  编译器：bishengir-compile                       │
           │  3rdparty: tvm + AscendNPU-IR (submodule)        │
           └─────────────────────────────────────────────────┘
                                       │
                                       │ 通过 submodule 引用
                                       ▼
                ┌────────────────────────────────────┐
                │  Huawei / AscendNPU-IR              │  ◄── Huawei 独立项目
                │  (gitcode.com/Ascend/ascendnpu-ir)  │     MLIR-based NPU 编译器
                │  生成: bishengir-compile binary     │     Apache-2.0
                │  third-party: llvm-project (~2GB)   │     bishengir = Huawei "毕昇"
                │                  torch-mlir         │     编译器品牌
                └────────────────────────────────────┘
```

**关键 finding（用户指出的）**：

| Repo | URL | 角色 | 是否 upstream |
|------|------|------|--------------|
| `tilelang-ascend` | github.com/tile-ai/tilelang-ascend | **历史 release repo**：ascendc_pto 是当前 release path；npuir branch 是 MLIR 试点 | **YES** (我们沿用) |
| `tilelang-mlir-ascend` | github.com/tile-ai/tilelang-mlir-ascend | **MLIR 路线主仓**：从 tilelang-ascend npuir branch 抽出来独立维护的 MLIR 后端版本，submodule 干净 | **YES** (用户要求新加) |
| `AscendNPU-IR` | gitcode.com/Ascend/ascendnpu-ir | **Huawei 的 MLIR 编译器**：生成 `bishengir-compile` 二进制，是 MLIR 后端的真正"compiler"。tilelang-mlir-ascend 通过 `--bishengir-path` 消费它 | **YES** (Huawei 单独维护) |
| `tile-ai/tile-lang` (上游 GPU 版) | github.com/tile-ai/tilelang | **上游 GPU/CUDA tile-lang**，Ascend 三个都是它的 fork | YES (理论上) |

**所以最简化的心智模型**：

- tilelang-ascend **ascendc_pto** branch ≈ **历史 1.0 release，字符串拼 AscendC**
- tilelang-mlir-ascend ≈ **MLIR 2.0 主路径，调 bishengir-compile**
- AscendNPU-IR ≈ **Huawei 提供的 bishengir 编译器源码 + MLIR dialect**

**Bug 可能藏在哪**（用户问题：「bug 可能出现在多个地方」）：
- ascendc_pto 上：bug 在 `tilelang-ascend/src/transform/ascend_*.cc` (TIR pass) 或
  `src/tl_templates/ascend/common.h` (codegen 模板)。**#996 就是这条线的 common.h bug**
- tilelang-mlir-ascend 上：bug 可能在
  - `tilelang-mlir-ascend/src/target/codegen_npuir.cc` (TIR → MLIR)
  - `tilelang-mlir-ascend/src/transform/*.cc` (TIR pass，与 ascendc_pto 可能共用大部分)
  - AscendNPU-IR 的 MLIR pass（pass infra 在 bishengir 内）
  - AscendNPU-IR 的 LLVM lowering（最底层）

**适配新版本 tilelang（用户问题：「是否两边都要调」）**：
- 答案：**绝大多数 NPU-specific 改动都要落两次**，因为：
  - 两个 repo 的 `src/transform/ascend_*.cc` 大部分是同一组 pass（TIR-level，与后端无关）
  - 但 codegen 后端不同：ascendc_pto 是 `codegen_ascend.cc` (string)，
    mlir 是 `codegen_npuir.cc` (MLIR API)，**必须分别改**
  - `tl_templates/ascend/common.h` 是 ascendc_pto 独有；MLIR 路径用 AscendNPU-IR 的 dialect op，不走 .h
- 实际工作量：~70% 重叠（TIR pass），~30% 各做一遍（codegen + template/dialect）
- **理想的工作流**：
  1. 在 ascendc_pto 上写出修复（更熟、release path）
  2. cherry-pick 到 tilelang-mlir-ascend (TIR pass)
  3. 在 tilelang-mlir-ascend 上对 MLIR 后端做对应调整
  4. 如果改动碰到 AscendNPU-IR dialect 行为，向 Huawei 提 issue/PR

---

## 2. tilelang 脚本编译 + 执行时序（ascendc_pto 后端，超详细）

这是 `examples/elementwise/elementwise_add.py --m 1024 --n 1024` 实际跑
的全部步骤。

### 阶段 A: Python 进程启动 → JIT 触发

```
T+0      python3 elementwise_add.py --m 1024 --n 1024
T+0.01   import tilelang  (5-10s, 触发 cython adapter JIT 编译，首次)
T+0.5    tilelang.cache.clear_cache()  # 用户调，清掉旧缓存
T+0.5    func = vec_add(M, N, 128, 256)
         ▼
         @tilelang.jit(out_idx=[-1]) 装饰器拦截调用
         ▼
         tilelang/jit/__init__.py::jit() 创建 JITKernel 对象
         此时还没编译，只是占位
T+1.0    a = torch.randn(M, N).npu()    # 触发 torch_npu 第一次初始化
T+2.0    b = torch.randn(M, N).npu()    # NPU mem alloc
T+2.5    c = func(a, b)                  # 这一行触发真正编译
```

### 阶段 B: TileLang 编译流水（func 第一次调用时）

```
T+2.5    JITKernel.__call__(a, b)
         ▼
         检查 .cache/tilelang/jit_kernels/ 是否有 hit
         ▼ (cache miss → 编译)
         tilelang/engine/lower.py::lower()
         ▼
         Step B1: 构造 TVM IRModule
                  - 把 @T.prim_func 装饰的 main 函数提取成 PrimFunc
                  - 包装成 IRModule
                  
         Step B2: 跑 transform passes (~15 个 C++ pass)
                  ┌────────────────────────────────────────────────┐
                  │ tilelang/transform/__init__.py::PassConfig    │
                  │ 把 pass 链按顺序传给 TVM 的 Pass infra：       │
                  │                                                │
                  │   frontend_legalize.cc                         │
                  │   layout_inference.cc                          │
                  │   if_stmt_binding.cc                           │
                  │   ascend_infer_buffer_scope.cc                 │
                  │   ascend_collect_buffer_shape.cc               │
                  │   ascend_lower_opaque_block.cc                 │
                  │   ascend_lower_parallel_to_vector.cc           │
                  │   ascend_memory_planning.cc  ← (Extent fail)  │
                  │   ascend_storage_rewrite.cc                    │
                  │   ascend_combinecv.cc       (CV 协作流水)      │
                  │   ascend_sync_insert.cc     (auto set_flag)    │
                  │   ascend_vid_reduction.cc                      │
                  │   ascend_workspace_reduction.cc                │
                  │   ...                                          │
                  │                                                │
                  │ 每个 pass：TIR IRModule → TIR IRModule         │
                  │ pass 顺序很重要，前一个的输出是下一个的输入    │
                  └────────────────────────────────────────────────┘
                  
         Step B3: codegen → AscendC C++ 字符串
                  ┌────────────────────────────────────────────────┐
                  │ src/target/codegen_ascend.cc                  │
                  │ 把 lowered TIR 翻译成 .cce 文件内容：          │
                  │                                                │
                  │   #include "tl_templates/ascend/common.h"     │
                  │   extern "C" __global__ __aicore__ void       │
                  │   vec_add_kernel(GM_ADDR A, GM_ADDR B,        │
                  │                  GM_ADDR C) {                  │
                  │     using namespace AscendC;                  │
                  │     LocalTensor<float> a_ub = ...;            │
                  │     copy_gm_to_ub<float, 256, 64>(            │
                  │         a_ub, A_gm, ...);                     │
                  │     // ← 这里调 common.h 里的模板             │
                  │     tile_add<float, 16384>(c_ub, a_ub, b_ub); │
                  │     copy_ub_to_gm<float, 256, 64>(            │
                  │         C_gm, c_ub, ...);                     │
                  │   }                                            │
                  └────────────────────────────────────────────────┘
                  
         Step B4: 写入临时 .cce 文件
                  /tmp/tilelang_jit_xxxx/vec_add_kernel.cce
                  
         Step B5: 调 ccec 编译
                  subprocess.run([
                    "ccec",
                    "-O2", "--cce-aicore-arch=dav-c220-cube",
                    "-I", "/usr/local/Ascend/...",  # AscendC headers
                    "-I", "$tilelang_install/src/tl_templates/ascend/",  # 我们的模板
                    "-c", "vec_add_kernel.cce", "-o", ".../vec_add_kernel.o"
                  ])
                  ccec = Huawei 的 NPU C++ 编译器（clang-based）
                  这一步如果 common.h 有 static_assert，会在这里 fail（option A）
                  
         Step B6: 链接 .o → .so
                  ld -shared vec_add_kernel.o ... → vec_add_kernel.so
                  
         Step B7: 通过 cython adapter 加载 .so
                  tilelang.jit.adapter.cython.adapter::CythonAdapter
                  用 ctypes.CDLL 加载 .so
                  存到 cache: ~/.cache/tilelang/jit_kernels/<hash>/
```

### 阶段 C: 真正跑 NPU

```
T+8      JITKernel 已编译好，回到 func(a, b) 的调用
         ▼
         CythonAdapter.invoke(a, b, c_out)
         ▼
         调 .so 里的 kernel_launcher 符号
         ▼
         kernel_launcher 内部调 aclrtLaunchKernel(
           kernel_func, grid, block, stream, args
         )
         ▼
         NPU 上每个 AIC tile 启动 2 个 AIV，开始执行：
           T+8.001  AIV 0 + AIV 1 拿到 cid=0
           T+8.002  T.copy(A[..], a_ub) → DataCopyPad
                    DMA 引擎：256*sizeof(float)=1024B blockLen ≥ 32B ✓
                    GM 数据搬到 UB
           T+8.005  T.barrier_all() → set_flag MTE2_V, wait_flag MTE2_V
           T+8.006  T.tile.add → AscendC::Add 向量 SIMD 加法
           T+8.008  T.barrier_all() → set_flag V_MTE3, wait_flag V_MTE3
           T+8.009  T.copy(c_ub, C[..]) → DataCopyPad UB→GM
           ...
         T+8.5    所有 cid 完成 → kernel 返回
         T+8.5    Python 这边的 c 张量已经有结果
T+9      torch.testing.assert_close(c, ref_c, ...)
T+9.1    print("Kernel Output Match!")
```

### 阶段 D: 第二次/后续调用

```
重复同一函数（缓存命中）：
  T+0    func = vec_add(...)
  T+0.01 hash 命中 .cache/tilelang/jit_kernels/<hash>/
  T+0.05 直接加载 .so，跳过 Step B
  T+0.05 func(a, b) → 直接 launch
```

---

## 3. MLIR 路线（tilelang-mlir-ascend + bishengir）的差异

### Step B3 (codegen) → 完全不同

```
              ascendc_pto                        tilelang-mlir-ascend
              ───────────                        ───────────────────
TIR ────► codegen_ascend.cc                  TIR ────► codegen_npuir.cc
            │                                            │
            │ string拼 .cce                              │ 调 MLIR Python API
            ▼                                            │ 构造 mlir::Module
       AscendC C++ 字符串                                ▼
            │                                       MLIR IRModule
            │ ccec 编译                               (用 ascendnpu dialect ops)
            ▼                                            │
        vec_add.so                                       │ bishengir-compile
                                                         │ (从 AscendNPU-IR 来的二进制)
                                                         ▼
                                                    一系列 MLIR pass
                                                    （ascendnpu → hivm → cce）
                                                         │
                                                         ▼
                                                    bishengir 内置后端 lowering
                                                         │
                                                         ▼
                                                    vec_add.so
```

最大差别在于：MLIR 路线把 codegen 后的所有优化都搬到 **MLIR pass infra**
里，可以叠加 LLVM 上游生态的 pass；ascendc_pto 路线一旦 codegen 出字符串
就交给 ccec，几乎不再做高层优化。

### Step B2 (TIR passes) → 基本一样

两条路线的 `src/transform/ascend_*.cc` 大部分一样（TIR-level passes），
但有差异：
- ascendc_pto 有 `ascend_combinecv.cc`（CV 协作流水）这种针对字符串 codegen 的特殊化
- mlir 路径在 codegen 之前可能少几个 pass，因为后续 MLIR pass 能补上

---

## 4. 谁修哪个 bug：决策表

| Bug 类型 | 在哪里修 |
|---------|---------|
| AscendC 模板逻辑错（如 #996 DMA align） | `tilelang-ascend ascendc_pto` 的 `tl_templates/ascend/common.h` |
| TIR pass 漏检查（如「Extent must be integer」） | 两条线的 `src/transform/ascend_*.cc` 都改 |
| MLIR dialect op 语义问题 | 上 AscendNPU-IR（Huawei 那边）提 issue |
| MLIR pass bug | 上 AscendNPU-IR |
| bishengir 后端代码生成错 | 上 AscendNPU-IR |
| Python frontend bug (`T.alloc_ub` 参数没传对) | 两条线的 `tilelang/language/*.py` |

---

## 5. 我们今天进度（截至 2026-05-17 T+8h）

### ✅ ascendc_pto 线
- 环境：tlrescue 容器（verl-8.5.2 image，CANN 8.5.2，chip 14）
- 源码：`/home/z00637938/workspace/tilelang-ascend` HEAD `b925cbe`
- 编译：从源码编译成功 → `tilelang-0.1.4+ubuntu.22.4.cann852-*.whl`
- 修复 option A：static_assert 已 apply 到 `common.h`，rebuild + 验证：
  - 1024×1024 baseline 仍 PASS
  - 32×32 / 4×4 case 现在编译期清晰报错 + 给出 remedy
  - 16×16 / 4×16 边界 case PASS（dstN*sizeof≥32）
  - 16×16 / 4×4 边界 case 报错（dstN*sizeof<32）
- artifacts: `OPTION_A_RESULT.md` + `issue_996_compile_check.patch`

### 🚧 tilelang-mlir-ascend 线
- 源码：`/home/z00637938/workspace/tilelang-mlir-ascend` (8.9 GB，含 LLVM)
- 编译：未试（需要 bishengir-compile 二进制，下一步任务）
- 验证 #996 在这条线是否复现：未做

### 🚧 AscendNPU-IR
- 源码：`tilelang-mlir-ascend/3rdparty/AscendNPU-IR/` 已 clone
- 编译：未试（需 LLVM + ninja + 大约 ~30min）
- 产物目标：`bishengir-compile` binary

---

## 6. 待办（按用户当前指示）

1. **同时 track 两个 upstream repo**：把 `tilelang-mlir-ascend` 加到
   `knowledge/sister-projects.md` 类似 a5_ops 的方式（不做 submodule，
   pin-by-commit 手动 sync）
2. **跑通 tilelang-mlir-ascend 环境**：
   - 编译 AscendNPU-IR → 拿到 `bishengir-compile`
   - 用 `install_npuir.sh --bishengir-path=...` 安装 tilelang-mlir-ascend
   - 验证可以跑 `examples/vec_add_1d.py`
3. **在 tilelang-mlir-ascend 上重现 #996**：用 ascendc_pto 的同一 reproducer
4. **如果 #996 也在 MLIR 路径暴露**：定位是 TIR pass 漏检 还是 MLIR
   lowering 漏检；分别提 PR
5. **如果 #996 在 MLIR 路径已自动 OK**：用 option C（auto-coalesce）的思路
   反推 MLIR 路径是怎么做的，移植回 ascendc_pto
