"""Probe: 2-step rollout + weight-update on sglang-on-NPU.

Goal: empirically check whether sglang's rollout + /update_weights_from_disk
path works on Ascend A3 NPU, since the user heard about a bf16 issue on
this combination and wants 2 steps verified before wiring miles' real RL
trainer.

Setup:
  * Sglang HTTP server in sgl_probe (Qwen2-0.5B-Instruct, bf16)
  * This script runs on host (or inside sgl_probe — uses requests)
  * Step 1: rollout 2 prompts -> save weights back to disk (no change)
           -> POST /update_weights_from_disk -> rollout again
  * Step 2: same
  * Each rollout uses the SAME tiny prompt set; we just want correctness,
    not learning signal.

Verifies:
  - sglang accepts bf16 weight loads from disk
  - /update_weights_from_disk returns success after a no-op save
  - rollout works before AND after the update (no internal corruption)
  - no NaN in outputs

Why "simplest dataset": no training data needed. The two prompts ARE the
dataset. No reward function, no advantage, no actor train. Pure rollout +
weight-update plumbing.
"""
import os
import sys
import shutil
import time
import json

# Same fix as smoke scripts
sys.path = [p for p in sys.path if p not in ("", "/")]

import requests

SGLANG_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000")
ORIGINAL_MODEL = "/host-models/Qwen2-0.5B-Instruct"
# We need an alternate model path so /update_weights_from_disk gets a
# different directory. The two directories can have the exact same
# weights -- sglang just needs to think it's loading "fresh" weights.
COPY_MODEL = "/host-models/Qwen2-0.5B-Instruct-copy"

PROMPTS = [
    "What is 2+3?",
    "The capital of Japan is",
]


def wait_healthy(base_url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/health_generate", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def make_copy_dir():
    """Make a working copy of the model dir so /update_weights_from_disk
    has a distinct path to load from. Identical bytes -- we're just
    exercising the load path."""
    if os.path.exists(COPY_MODEL):
        return
    print(f"  copying {ORIGINAL_MODEL} -> {COPY_MODEL} ...")
    shutil.copytree(ORIGINAL_MODEL, COPY_MODEL)


def rollout(prompts, label):
    """sglang /generate. Returns list[str], one per prompt."""
    print(f"[{label}] rollout {len(prompts)} prompts ...")
    out = []
    t0 = time.time()
    for p in prompts:
        r = requests.post(
            f"{SGLANG_URL}/generate",
            json={"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 8}},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            data = data[0]
        text = data.get("text", "")
        # NaN/empty detection
        has_nan = "nan" in text.lower() or "inf" in text.lower()
        out.append({"prompt": p, "text": text, "has_nan_marker": has_nan})
    print(f"[{label}] rollout done in {time.time()-t0:.1f}s")
    for o in out:
        print(f"  {o['prompt']!r}\n    -> {o['text']!r}{'  [NAN!]' if o['has_nan_marker'] else ''}")
    return out


def update_weights_from_disk(path, label):
    print(f"[{label}] /update_weights_from_disk path={path} ...")
    t0 = time.time()
    r = requests.post(
        f"{SGLANG_URL}/update_weights_from_disk",
        json={"model_path": path},
        timeout=180,
    )
    print(f"[{label}] response in {time.time()-t0:.1f}s: status={r.status_code}")
    try:
        data = r.json()
        print(f"  body: {json.dumps(data, indent=2)[:300]}")
        return data.get("success", False)
    except Exception as e:
        print(f"  decode err: {e}; body={r.text[:300]}")
        return False


def main():
    if not wait_healthy(SGLANG_URL):
        raise RuntimeError(f"sglang server at {SGLANG_URL} not healthy after 120s")

    # Make second model path
    make_copy_dir()

    # Round 0: baseline rollout (with original weights)
    r0 = rollout(PROMPTS, "round-0 baseline")

    # Step 1: update weights to copy dir; rollout again
    ok1 = update_weights_from_disk(COPY_MODEL, "step-1 update")
    if not ok1:
        print("[step-1] update_weights returned non-success; checking rollout anyway")
    r1 = rollout(PROMPTS, "step-1 rollout")

    # Step 2: update back to original; rollout again
    ok2 = update_weights_from_disk(ORIGINAL_MODEL, "step-2 update")
    if not ok2:
        print("[step-2] update_weights returned non-success; checking rollout anyway")
    r2 = rollout(PROMPTS, "step-2 rollout")

    # Summary
    print("\n=== 2-step sglang NPU rollout + weight-update summary ===")
    print(f"  baseline rollout : {[o['text'][:30] for o in r0]}")
    print(f"  step-1 update    : ok={ok1}")
    print(f"  step-1 rollout   : {[o['text'][:30] for o in r1]}")
    print(f"  step-2 update    : ok={ok2}")
    print(f"  step-2 rollout   : {[o['text'][:30] for o in r2]}")

    all_finite = not any(o['has_nan_marker'] for r in (r0, r1, r2) for o in r)
    all_non_empty = all(o['text'].strip() for r in (r0, r1, r2) for o in r)

    if ok1 and ok2 and all_finite and all_non_empty:
        print(f"\n[smoke] PASS")
        return 0
    else:
        print(f"\n[smoke] FAIL: ok1={ok1} ok2={ok2} finite={all_finite} non_empty={all_non_empty}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
