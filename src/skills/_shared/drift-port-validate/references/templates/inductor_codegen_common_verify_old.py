"""OLD-upstream path verify for torch_npu.compat.inductor_codegen_common."""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/inductor_codegen_common.py"

old_common = types.ModuleType("torch._inductor.codegen.common")
class OldIndentedBuffer: pass
def old_free_symbol_is_type(*a, **kw): return "old"
old_common.IndentedBuffer = OldIndentedBuffer
old_common.free_symbol_is_type = old_free_symbol_is_type

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.codegen"] = types.ModuleType("torch._inductor.codegen")
sys.modules["torch._inductor.codegen.common"] = old_common

# NEW-path targets stubbed empty
sys.modules["torch._inductor.utils"] = types.ModuleType("torch._inductor.utils")
sys.modules["torch.utils"] = types.ModuleType("torch.utils")
sys.modules["torch.utils._sympy"] = types.ModuleType("torch.utils._sympy")
sys.modules["torch.utils._sympy.symbol"] = types.ModuleType("torch.utils._sympy.symbol")

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("IndentedBuffer is OldIndentedBuffer", mod.IndentedBuffer is OldIndentedBuffer)
check("free_symbol_is_type is old impl", mod.free_symbol_is_type is old_free_symbol_is_type)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
