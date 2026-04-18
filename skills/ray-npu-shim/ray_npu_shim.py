# Copyright 2026 easyr1-npu port authors.
# Licensed under Apache 2.0.
#
# Drop-in helpers that make Ray work on Ascend NPU with the fewest changes
# to the host framework. See repo/skills/ray-npu-shim/SKILL.md for the
# integration recipe.

from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def is_npu_available() -> bool:
    """True iff torch_npu loads and reports an NPU."""
    try:
        import torch
        import torch_npu  # noqa: F401 — registers 'npu' as a torch device

        return bool(torch.npu.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_ray_resource_name() -> str:
    """Ray treats CUDA GPUs as the builtin ``"GPU"`` resource with ``num_gpus``
    sugar. NPUs don't get that sugar — Ray has to be told about them via the
    ``resources={"NPU": n}`` option, and ``available_resources()`` returns them
    under the ``"NPU"`` key. Use this helper everywhere we talk to Ray about
    accelerator counts.
    """
    return "NPU" if is_npu_available() else "GPU"


def _npu_runtime_env_defaults() -> dict[str, str]:
    """Env vars every NPU Ray actor needs. Keep minimal; don't leak framework-
    specific vars here (those belong in the caller's runtime_env)."""
    return {
        # NPU-BUG-002: Ray 2.55+ wipes *_VISIBLE_DEVICES on actor spawn when
        # the actor has num_gpus=0/None. We claim NPU via `resources` not
        # num_gpus, so the wipe blinds torch_npu inside the actor. Turn it
        # off so the visibility list propagates.
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
        # NPU-ENV-002: vllm-ascend's FRACTAL_NZ weight layout corrupts param
        # sync in RL scenarios. Disable globally inside the cluster; if an
        # inference-only workload wants NZ perf, set the env var back to 1
        # externally before calling ray_init_npu_aware.
        "VLLM_ASCEND_ENABLE_NZ": "0",
    }


def ray_init_npu_aware(**kwargs: Any):
    """Thin wrapper over ``ray.init`` that adds NPU-specific resources and
    runtime_env defaults. Forwards every other kwarg untouched.

    Example::

        from ray_npu_shim import ray_init_npu_aware
        if not ray.is_initialized():
            ray_init_npu_aware(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true"}})
    """
    import ray

    runtime_env = dict(kwargs.pop("runtime_env", None) or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    for k, v in _npu_runtime_env_defaults().items():
        env_vars.setdefault(k, v)
    runtime_env["env_vars"] = env_vars
    kwargs["runtime_env"] = runtime_env

    if is_npu_available():
        import torch

        resources = dict(kwargs.pop("resources", None) or {})
        resources.setdefault(get_ray_resource_name(), int(torch.npu.device_count()))
        kwargs["resources"] = resources

    return ray.init(**kwargs)


def apply_actor_options(options: dict, num_accel: int) -> dict:
    """Set the correct accelerator-claim option on a Ray actor/task options dict.

    On CUDA hosts, sets ``options["num_gpus"] = num_accel`` (the Ray sugar).
    On NPU hosts, sets ``options["resources"] = {"NPU": num_accel}`` (the
    explicit custom-resource request — Ray's num_gpus sugar is CUDA-only).

    Mutates and returns the dict for chaining.
    """
    if get_ray_resource_name() == "GPU":
        options["num_gpus"] = num_accel
    else:
        options.setdefault("resources", {})[get_ray_resource_name()] = num_accel
    return options


def placement_bundle(num_cpus: int = 1, num_accel: int = 1) -> dict:
    """Build a placement-group bundle that requests ``num_cpus`` CPUs plus
    ``num_accel`` of whichever accelerator is active. Use instead of
    hardcoding ``{"CPU": n, "GPU": m}``.
    """
    bundle: dict[str, int | float] = {"CPU": num_cpus}
    if num_accel > 0:
        bundle[get_ray_resource_name()] = num_accel
    return bundle


__all__ = [
    "is_npu_available",
    "get_ray_resource_name",
    "ray_init_npu_aware",
    "apply_actor_options",
    "placement_bundle",
]
