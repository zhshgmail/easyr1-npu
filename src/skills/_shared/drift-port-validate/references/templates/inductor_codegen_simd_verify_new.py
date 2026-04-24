"""NEW-upstream path verify for torch_npu.compat.inductor_codegen_simd."""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/inductor_codegen_simd.py"

# OLD path empty -> fallback
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.codegen"] = types.ModuleType("torch._inductor.codegen")
sys.modules["torch._inductor.codegen.simd"] = types.ModuleType("torch._inductor.codegen.simd")

# NEW target: simd_kernel_features has all three
new_skf = types.ModuleType("torch._inductor.codegen.simd_kernel_features")
class NewDisableReduction: pass
class NewEnableReduction: pass
class NewSIMDKernelFeatures: pass
new_skf.DisableReduction = NewDisableReduction
new_skf.EnableReduction = NewEnableReduction
new_skf.SIMDKernelFeatures = NewSIMDKernelFeatures
sys.modules["torch._inductor.codegen.simd_kernel_features"] = new_skf

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("DisableReduction is NewDisableReduction", mod.DisableReduction is NewDisableReduction)
check("EnableReduction is NewEnableReduction", mod.EnableReduction is NewEnableReduction)
check("SIMDKernelFeatures is NewSIMDKernelFeatures", mod.SIMDKernelFeatures is NewSIMDKernelFeatures)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
