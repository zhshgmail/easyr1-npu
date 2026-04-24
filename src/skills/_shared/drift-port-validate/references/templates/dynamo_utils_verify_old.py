"""OLD-upstream path verify for torch_npu.compat.dynamo_utils."""
import sys, types, importlib.util

SHIM = "/home/z00637938/workspace/easyr1-npu/upstream/torch-npu/torch_npu/compat/dynamo_utils.py"

old_dyn = types.ModuleType("torch._dynamo.utils")
class OldUnsupportedFakeTensorException(Exception): pass
old_dyn.UnsupportedFakeTensorException = OldUnsupportedFakeTensorException

sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch._dynamo"] = types.ModuleType("torch._dynamo")
sys.modules["torch._dynamo.utils"] = old_dyn

# NEW path stubbed empty
sys.modules["torch._subclasses"] = types.ModuleType("torch._subclasses")
sys.modules["torch._subclasses.fake_tensor"] = types.ModuleType("torch._subclasses.fake_tensor")

spec = importlib.util.spec_from_file_location("shim_under_test", SHIM)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

results = []
def check(name, ok):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")

check(
    "UnsupportedFakeTensorException is OldUnsupportedFakeTensorException",
    mod.UnsupportedFakeTensorException is OldUnsupportedFakeTensorException,
)

npass = sum(1 for _, ok in results if ok)
print(f"RESULT {npass}/{len(results)}")
sys.exit(0 if npass == len(results) else 1)
