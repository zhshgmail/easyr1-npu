"""F1 compat-shim verification — OLD vllm pass-through path.

Simulate OLD vllm by stubbing the upstream modules with the symbols
present. The shim should take its try-branch (NOT the except).
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


stub_module("vllm")
stub_module("vllm.model_executor")
stub_module("vllm.model_executor.layers")
stub_module("vllm.model_executor.layers.fused_moe")

class _StubFusedMoE:
    pass

stub_module("vllm.model_executor.layers.fused_moe.layer", FusedMoE=_StubFusedMoE)

# OLD vllm: provide the upstream modules with their classes
class _OldSharedFusedMoE(_StubFusedMoE):
    pass

stub_module(
    "vllm.model_executor.layers.fused_moe.shared_fused_moe",
    SharedFusedMoE=_OldSharedFusedMoE,
)

stub_module("vllm.model_executor.layers.fused_moe.runner")

class _OldDefaultMoERunner:
    pass

stub_module(
    "vllm.model_executor.layers.fused_moe.runner.default_moe_runner",
    DefaultMoERunner=_OldDefaultMoERunner,
)


def load_compat(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


shared = load_compat(
    "/vllm-ascend/vllm_ascend/compat/shared_fused_moe.py", "test_shared_old"
)
default = load_compat(
    "/vllm-ascend/vllm_ascend/compat/default_moe_runner.py", "test_default_old"
)

checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(f"{name}: {'PASS' if cond else 'FAIL'}")


check(
    "OLD_VLLM_SHARED_UPSTREAM_TRUE", shared._UPSTREAM_HAS_SHARED_FUSED_MOE is True
)
check(
    "OLD_VLLM_SHARED_RESOLVES_UPSTREAM",
    shared.SharedFusedMoE is _OldSharedFusedMoE,
)

check(
    "OLD_VLLM_DEFAULT_UPSTREAM_TRUE",
    default._UPSTREAM_HAS_DEFAULT_MOE_RUNNER is True,
)
check(
    "OLD_VLLM_DEFAULT_RESOLVES_UPSTREAM",
    default.DefaultMoERunner is _OldDefaultMoERunner,
)

passed = sum(1 for _, ok in checks if ok)
total = len(checks)
print(f"\nRESULT {passed}/{total}")
sys.exit(0 if passed == total else 1)
