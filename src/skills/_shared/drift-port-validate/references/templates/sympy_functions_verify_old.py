"""OLD-upstream path verify for torch_npu.compat.sympy_functions.

OLD path: symbols FloorDiv, ModularIndexing live at torch._inductor.utils.
Shim should take the try branch; exported symbols are the OLD stubs.
"""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/sympy_functions.py"

# Stub the OLD path WITH the symbols
old_utils = types.ModuleType("torch._inductor.utils")
class OldFloorDiv: pass
class OldModularIndexing: pass
old_utils.FloorDiv = OldFloorDiv
old_utils.ModularIndexing = OldModularIndexing

# Parent stubs
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.utils"] = old_utils

# Also stub NEW path parents WITHOUT the symbols (fallback won't be taken)
sys.modules["torch.utils"] = types.ModuleType("torch.utils")
sys.modules["torch.utils._sympy"] = types.ModuleType("torch.utils._sympy")
sys.modules["torch.utils._sympy.functions"] = types.ModuleType("torch.utils._sympy.functions")

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("_SOURCE == 'torch._inductor.utils'", mod._SOURCE == "torch._inductor.utils")
check("FloorDiv is OldFloorDiv (OLD identity)", mod.FloorDiv is OldFloorDiv)
check("ModularIndexing is OldModularIndexing (OLD identity)", mod.ModularIndexing is OldModularIndexing)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
