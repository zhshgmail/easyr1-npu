"""NEW-upstream path verify for torch_npu.compat.inductor_ir.

NEW path: FloorDiv → torch.utils._sympy.functions
          ModularIndexing → torch.utils._sympy.functions
          LoopBody → torch._inductor.loop_body
          Reduction → torch._decomp.decompositions
          ReductionHint → torch._inductor.runtime.hints
All of these must resolve through the except branch.
"""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/inductor_ir.py"

# Stub OLD path as an EMPTY module so every `from torch._inductor.ir import X` fails
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._inductor"] = types.ModuleType("torch._inductor")
sys.modules["torch._inductor.ir"] = types.ModuleType("torch._inductor.ir")  # empty

# Stub NEW paths with the symbols
new_sympy = types.ModuleType("torch.utils._sympy.functions")
class NewFloorDiv: pass
class NewModularIndexing: pass
new_sympy.FloorDiv = NewFloorDiv
new_sympy.ModularIndexing = NewModularIndexing
sys.modules["torch.utils"] = types.ModuleType("torch.utils")
sys.modules["torch.utils._sympy"] = types.ModuleType("torch.utils._sympy")
sys.modules["torch.utils._sympy.functions"] = new_sympy

new_loop_body = types.ModuleType("torch._inductor.loop_body")
class NewLoopBody: pass
new_loop_body.LoopBody = NewLoopBody
sys.modules["torch._inductor.loop_body"] = new_loop_body

new_decomp = types.ModuleType("torch._decomp.decompositions")
class NewReduction: pass
new_decomp.Reduction = NewReduction
sys.modules["torch._decomp"] = types.ModuleType("torch._decomp")
sys.modules["torch._decomp.decompositions"] = new_decomp

new_hints = types.ModuleType("torch._inductor.runtime.hints")
class NewReductionHint: pass
new_hints.ReductionHint = NewReductionHint
sys.modules["torch._inductor.runtime"] = types.ModuleType("torch._inductor.runtime")
sys.modules["torch._inductor.runtime.hints"] = new_hints

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check("FloorDiv is NewFloorDiv", mod.FloorDiv is NewFloorDiv)
check("ModularIndexing is NewModularIndexing", mod.ModularIndexing is NewModularIndexing)
check("LoopBody is NewLoopBody", mod.LoopBody is NewLoopBody)
check("Reduction is NewReduction", mod.Reduction is NewReduction)
check("ReductionHint is NewReductionHint", mod.ReductionHint is NewReductionHint)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
