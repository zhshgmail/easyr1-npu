"""F1 compat-shim verification harness.

Runs inside a container where vllm is installed but the target drift
(removed SharedFusedMoE / DefaultMoERunner) has not actually happened.
We simulate NEW vllm by stubbing `vllm.model_executor.*` parent packages
and NOT providing the leaf modules the compat shim tries to import.
This forces the shim to take its ImportError fallback path.

Success criteria (printed as structured lines):
- NEW_VLLM_SHARED_UPSTREAM_FALSE — shim detected upstream absent
- NEW_VLLM_SHARED_RESOLVES_LOCAL — SharedFusedMoE resolves to local shim
- NEW_VLLM_SHARED_ISSUBCLASS_FUSEDMOE — local shim is still FusedMoE subclass
- NEW_VLLM_DEFAULT_UPSTREAM_FALSE — shim detected upstream absent
- NEW_VLLM_DEFAULT_RESOLVES_LOCAL — DefaultMoERunner resolves to MoERunner alias
"""
import sys
import types
import importlib.util


def stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# Build stub vllm hierarchy — enough for parent package imports to succeed
stub_module("vllm")
stub_module("vllm.model_executor")
stub_module("vllm.model_executor.layers")
stub_module("vllm.model_executor.layers.fused_moe")

class _StubFusedMoE:
    pass

stub_module("vllm.model_executor.layers.fused_moe.layer", FusedMoE=_StubFusedMoE)

# For the F1/F2 default_moe_runner fallback, stub the new moe_runner module
class _StubMoERunner:
    pass

stub_module("vllm.model_executor.layers.fused_moe.runner")
stub_module(
    "vllm.model_executor.layers.fused_moe.runner.moe_runner",
    MoERunner=_StubMoERunner,
)

# DO NOT stub:
#   vllm.model_executor.layers.fused_moe.shared_fused_moe
#   vllm.model_executor.layers.fused_moe.runner.default_moe_runner
# Their ImportError is what we want the shim to catch.


def load_compat(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


shared = load_compat(
    "/vllm-ascend/vllm_ascend/compat/shared_fused_moe.py", "test_shared"
)
default = load_compat(
    "/vllm-ascend/vllm_ascend/compat/default_moe_runner.py", "test_default"
)

# Assertions
checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(f"{name}: {'PASS' if cond else 'FAIL'}")


check("NEW_VLLM_SHARED_UPSTREAM_FALSE", shared._UPSTREAM_HAS_SHARED_FUSED_MOE is False)
check(
    "NEW_VLLM_SHARED_RESOLVES_LOCAL",
    shared.SharedFusedMoE.__module__ == "test_shared",
)
check(
    "NEW_VLLM_SHARED_ISSUBCLASS_FUSEDMOE",
    issubclass(shared.SharedFusedMoE, _StubFusedMoE),
)

check(
    "NEW_VLLM_DEFAULT_UPSTREAM_FALSE",
    default._UPSTREAM_HAS_DEFAULT_MOE_RUNNER is False,
)
check(
    "NEW_VLLM_DEFAULT_RESOLVES_LOCAL",
    default.DefaultMoERunner is _StubMoERunner,
)

passed = sum(1 for _, ok in checks if ok)
total = len(checks)
print(f"\nRESULT {passed}/{total}")
sys.exit(0 if passed == total else 1)
