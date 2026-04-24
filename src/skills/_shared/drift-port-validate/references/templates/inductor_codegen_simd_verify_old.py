"""OLD-upstream path verify for torch_npu.compat.inductor_codegen_simd."""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/inductor_codegen_simd.py"

old_simd = types.ModuleType("torch._inductor.codegen.simd")
class OldDisableReduction: pass
class OldEnableReduction: pass
class OldSIMDKernelFeatures: pass
old_simd.DisableReduction = OldDisableReduction
old_simd.EnableReduction = OldEnableReduction
old_simd.SIMDKernelFeatures = OldSIMDKernelFeatures

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.codegen"] = types.ModuleType("torch._inductor.codegen")
sys.modules["torch._inductor.codegen.simd"] = old_simd

# NEW-path target stubbed empty
sys.modules["torch._inductor.codegen.simd_kernel_features"] = types.ModuleType(
    "torch._inductor.codegen.simd_kernel_features"
)

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("DisableReduction is OldDisableReduction", mod.DisableReduction is OldDisableReduction)
check("EnableReduction is OldEnableReduction", mod.EnableReduction is OldEnableReduction)
check("SIMDKernelFeatures is OldSIMDKernelFeatures", mod.SIMDKernelFeatures is OldSIMDKernelFeatures)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
