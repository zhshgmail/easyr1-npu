"""OLD-upstream path verify for torch_npu.compat.inductor_ir.

OLD path: FloorDiv, ModularIndexing, LoopBody, Reduction, ReductionHint
all live at torch._inductor.ir. Shim should take try branch on every
symbol.
"""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/inductor_ir.py"

# Stub OLD path WITH all symbols
old_ir = types.ModuleType("torch._inductor.ir")
class OldFloorDiv: pass
class OldModularIndexing: pass
class OldLoopBody: pass
class OldReduction: pass
class OldReductionHint: pass
old_ir.FloorDiv = OldFloorDiv
old_ir.ModularIndexing = OldModularIndexing
old_ir.LoopBody = OldLoopBody
old_ir.Reduction = OldReduction
old_ir.ReductionHint = OldReductionHint

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.ir"] = old_ir

# NEW-path parents stubbed empty (fallback branches must not be taken)
sys.modules["torch.utils"] = types.ModuleType("torch.utils")
sys.modules["torch.utils._sympy"] = types.ModuleType("torch.utils._sympy")
sys.modules["torch.utils._sympy.functions"] = types.ModuleType("torch.utils._sympy.functions")
sys.modules["torch._inductor.loop_body"] = types.ModuleType("torch._inductor.loop_body")
sys.modules["torch._decomp"] = types.ModuleType("torch._decomp")
sys.modules["torch._decomp.decompositions"] = types.ModuleType("torch._decomp.decompositions")
sys.modules["torch._inductor.runtime"] = types.ModuleType("torch._inductor.runtime")
sys.modules["torch._inductor.runtime.hints"] = types.ModuleType("torch._inductor.runtime.hints")

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("FloorDiv is OldFloorDiv", mod.FloorDiv is OldFloorDiv)
check("ModularIndexing is OldModularIndexing", mod.ModularIndexing is OldModularIndexing)
check("LoopBody is OldLoopBody", mod.LoopBody is OldLoopBody)
check("Reduction is OldReduction", mod.Reduction is OldReduction)
check("ReductionHint is OldReductionHint", mod.ReductionHint is OldReductionHint)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
