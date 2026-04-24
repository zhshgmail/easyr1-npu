"""NEW-upstream path verify for torch_npu.compat.sympy_functions.

NEW path: torch._inductor.utils no longer has FloorDiv/ModularIndexing;
they live at torch.utils._sympy.functions.
Shim should take the except branch; exported symbols are the NEW stubs.
"""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/sympy_functions.py"

# Stub OLD path WITHOUT the symbols so `from ... import FloorDiv` raises ImportError
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.utils"] = types.ModuleType("torch._inductor.utils")  # empty

# Stub NEW path WITH the symbols
new_fns = types.ModuleType("torch.utils._sympy.functions")
class NewFloorDiv: pass
class NewModularIndexing: pass
new_fns.FloorDiv = NewFloorDiv
new_fns.ModularIndexing = NewModularIndexing
sys.modules["torch.utils"] = types.ModuleType("torch.utils")
sys.modules["torch.utils._sympy"] = types.ModuleType("torch.utils._sympy")
sys.modules["torch.utils._sympy.functions"] = new_fns

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("_SOURCE == 'torch.utils._sympy.functions'", mod._SOURCE == "torch.utils._sympy.functions")
check("FloorDiv is NewFloorDiv (NEW identity)", mod.FloorDiv is NewFloorDiv)
check("ModularIndexing is NewModularIndexing (NEW identity)", mod.ModularIndexing is NewModularIndexing)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
