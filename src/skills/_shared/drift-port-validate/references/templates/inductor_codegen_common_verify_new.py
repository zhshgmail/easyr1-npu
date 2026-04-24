"""NEW-upstream path verify for torch_npu.compat.inductor_codegen_common."""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/inductor_codegen_common.py"

# OLD parent, empty -> fallback
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.codegen"] = types.ModuleType("torch._inductor.codegen")
sys.modules["torch._inductor.codegen.common"] = types.ModuleType("torch._inductor.codegen.common")

# NEW path 1: torch._inductor.utils has IndentedBuffer
new_utils = types.ModuleType("torch._inductor.utils")
class NewIndentedBuffer: pass
new_utils.IndentedBuffer = NewIndentedBuffer
sys.modules["torch._inductor.utils"] = new_utils

# NEW path 2: torch.utils._sympy.symbol has free_symbol_is_type
new_symbol = types.ModuleType("torch.utils._sympy.symbol")
def new_free_symbol_is_type(*a, **kw): return "new"
new_symbol.free_symbol_is_type = new_free_symbol_is_type
sys.modules["torch.utils"] = types.ModuleType("torch.utils")
sys.modules["torch.utils._sympy"] = types.ModuleType("torch.utils._sympy")
sys.modules["torch.utils._sympy.symbol"] = new_symbol

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("IndentedBuffer is NewIndentedBuffer", mod.IndentedBuffer is NewIndentedBuffer)
check("free_symbol_is_type is new impl", mod.free_symbol_is_type is new_free_symbol_is_type)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
