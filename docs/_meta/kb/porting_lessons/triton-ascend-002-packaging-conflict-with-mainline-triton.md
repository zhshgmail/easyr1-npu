---
id: triton-ascend-002
date: 2026-05-29
layer: triton-ascend
title: triton-ascend and mainline triton can both be on disk, but only the later install is functional
trigger:
  - "tlrescue / quay.io/ascend/verl container fresh pull"
  - "install vllm-ascend and triton-ascend into the same Python env"
  - "see ImportError: cannot import name 'Language' or 'AttrsDescriptor' from triton.backends.compiler"
symptom_in_wild:
  - "ImportError: cannot import name 'Language' from 'triton.backends.compiler' (after pip install triton-ascend)"
  - "ImportError: cannot import name 'AttrsDescriptor' from 'triton.backends.compiler' (after pip install triton)"
  - "import mindspeed.megatron_adaptor → cascade ImportError because triton stack is broken"
  - "import triton works but `from triton.backends import backends` returns wrong backend set for your hardware"
root_cause: >
  triton-ascend (NPU DSL) and mainline triton (GPU DSL) both register their
  own `triton/backends/compiler.py` and `triton/backends/__init__.py` files
  via pip-package-managed file ownership. The two forks of `compiler.py`
  diverge on exported symbols: mainline triton 3.6 exports `class Language`
  (needed by `triton/backends/amd/compiler.py` and `nvidia/compiler.py`);
  triton-ascend 3.2 exports `AttrsDescriptor` (needed by
  `triton/backends/ascend/compiler.py`). Whichever was installed later wins
  the file; the other's backend immediately breaks on first import.

  vllm-ascend pulls xgrammar; xgrammar declares
  `triton; platform_system == "Linux" and platform_machine == "x86_64"`
  which fires on Ascend NPU hosts (also Linux x86_64) and brings mainline
  triton in. A separate `pip install triton-ascend` collides. The two
  packages were never designed to coexist, and neither team plans to defend
  against it (verified at triton-lang/triton-ascend#306 close 2026-05-29).
mistake_pattern: "two pip packages register-overwriting the same import path, neither knowing the other exists"
correction:
  - "Pick ONE per environment. For NPU training paths the answer is triton-ascend; mainline triton is dead weight (xgrammar's triton-using inference-grammar path is not exercised in training)."
  - "Workaround recipe (verified): `pip uninstall -y triton && pip install --force-reinstall --no-deps triton-ascend`. After this, `import triton` returns 3.2.0 with `backends == {'ascend': ...}`. miles real-shape e2e (H=64, SEQ=2048, 52M-param DSAMLASelfAttention) runs end-to-end through MindSpeed + tilelang."
  - "Don't file an issue at triton-lang/triton-ascend asking for Provides-Dist/Conflicts — the project doesn't consider mainline-triton coexistence a supported configuration, and the conflict's actual cause is one layer up (xgrammar's broad-but-incorrect-on-NPU dep declaration, OR the container author who installed both)."
  - "For a project-level NPU container, document the recipe in setup_env.sh and gate the container's smoke on `import triton; assert 'ascend' in triton.backends.backends`."
  - "If escalating: the real upstream targets are (a) mlc-ai/xgrammar — request platform marker that doesn't fire on NPU hosts; or (b) Huawei's container/image authors — install-order fix in the Dockerfile."
evidence:
  - "tlrescue empirical test 2026-05-29: with both `triton 3.6.0` and `triton-ascend 3.2.0` on disk, install order decides which backend works. ascend-first then mainline-overlay → `triton.backends.ascend` ImportError on AttrsDescriptor while mainline amd/nvidia work. mainline-first then ascend-overlay → `triton.backends.amd.compiler` ImportError on Language while ascend works."
  - "PyPI xgrammar metadata: `triton; platform_system == \"Linux\" and platform_machine == \"x86_64\"` (no NPU exclusion)."
  - "PyPI vllm-ascend metadata: `Requires: ... xgrammar>=0.1.30 ...` (transitive triton)."
  - "Misfiled at triton-lang/triton-ascend#306, then closed as not-planned with reframing comment 2026-05-29."
  - "User Discord 2026-05-29T07:35Z: '不用纠结这个。把这个问题记录下来，作为知识，以后知道怎么做。'"
---
