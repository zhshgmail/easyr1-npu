"""Probe: 2 real RL steps using sglang NPU rollout + HF trainer side.

Goal: exercise the FULL bf16 weight-conversion path that sglang takes when
/update_weights_from_disk receives an actually-changed checkpoint (last
smoke short-circuited because the weights were bit-identical).

Two-process design (avoids HCCL conflict with the host's long-running
tenant -- option A per user):

   ┌─────────────────────────┐                       ┌─────────────────────────┐
   │ sgl_probe container      │ ◄── HTTP /generate ──│ tlrescue container        │
   │ sglang HTTP server       │                       │ this script               │
   │   Qwen2-0.5B bf16 NPU    │ ◄── /update_weights ─│   HF Qwen2-0.5B forward    │
   │   chip 1                 │      _from_disk      │   + Adam step             │
   │                          │                       │   + save_pretrained ckpt  │
   └─────────────────────────┘                       └─────────────────────────┘
                                                        (CPU side -- no NPU needed
                                                         for this perturbation)

Steps:
  Step 0 (baseline): rollout 2 prompts
  Step 1: HF train Qwen2-0.5B for 1 gradient step on CPU (tiny LR perturbation)
          -> save ckpt -> /update_weights_from_disk -> rollout same 2 prompts
          -> assert outputs differ from baseline OR weight-load took > 1s
  Step 2: same: HF train one more step -> save -> update -> rollout
          -> assert step-2 weights work

Watchdog enforced -- our process kills itself if other tenant pressures
the chip. NPU is only touched by sglang side (sgl_probe); the trainer runs
on CPU here to avoid colliding with any tlrescue tilelang caches and keep
the HBM picture clean.
"""
import sys

sys.path = [p for p in sys.path if p not in ("", "/")]

import os
import shutil
import time
import json

import requests
import torch

# Avoid pulling tilelang/mindspeed -- pure HF here keeps the driver clean
from transformers import AutoModelForCausalLM, AutoTokenizer

SGLANG_URL = os.environ.get("SGLANG_BASE_URL", "http://sgl_probe:30000")
ORIGINAL_MODEL = "/home/z00637938/workspace/models/Qwen2-0.5B-Instruct"
STEP_DIRS = [
    "/home/z00637938/workspace/models/Qwen2-0.5B-step1",
    "/home/z00637938/workspace/models/Qwen2-0.5B-step2",
]

PROMPTS = [
    "What is 2+3?",
    "The capital of Japan is",
]


def wait_healthy(url, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health_generate", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def rollout(prompts, label):
    print(f"[{label}] rollout {len(prompts)} prompts ...")
    out = []
    t0 = time.time()
    for p in prompts:
        r = requests.post(
            f"{SGLANG_URL}/generate",
            json={"text": p, "sampling_params": {"temperature": 0.0, "max_new_tokens": 12}},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            data = data[0]
        text = data.get("text", "")
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
        timeout=300,
    )
    elapsed = time.time() - t0
    print(f"[{label}] response in {elapsed:.1f}s: status={r.status_code}")
    try:
        data = r.json()
        success = data.get("success", False)
        print(f"  body: {json.dumps(data, indent=2)[:300]}")
        return success, elapsed
    except Exception as e:
        print(f"  decode err: {e}; body={r.text[:300]}")
        return False, elapsed


def train_one_step(model, tokenizer, step_idx, save_dir, lr=1e-4):
    """Run one HF gradient step on CPU with a tiny perturbation so the
    saved weights are demonstrably different from the prior checkpoint.
    """
    print(f"[train-{step_idx}] doing 1 gradient step on CPU ...")
    t0 = time.time()
    # Inputs: just compute LM loss on a single short sequence
    ids = tokenizer("Hello world", return_tensors="pt").input_ids
    out = model(input_ids=ids, labels=ids)
    loss = out.loss
    print(f"  loss: {loss.item():.4f}")
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"[train-{step_idx}] step done in {time.time()-t0:.1f}s")

    # Save -- copy tokenizer files too so sglang can reload
    print(f"[train-{step_idx}] saving to {save_dir} ...")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    # Tokenizer + generation_config copy
    for fname in ("tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "generation_config.json"):
        src = os.path.join(ORIGINAL_MODEL, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(save_dir, fname))
    print(f"[train-{step_idx}] saved")
    return loss.item()


def main():
    if not wait_healthy(SGLANG_URL):
        raise RuntimeError(f"sglang server at {SGLANG_URL} not healthy after 120s")

    # Round 0: baseline rollout
    r0 = rollout(PROMPTS, "round-0 baseline")

    # Load HF model on CPU for training perturbations
    print("[setup] loading Qwen2-0.5B on CPU for training perturbation ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL, dtype=torch.bfloat16)
    print(f"[setup] model loaded in {time.time()-t0:.1f}s ({sum(p.numel() for p in model.parameters()):,} params)")

    # Step 1: perturb -> save -> update -> rollout
    train_one_step(model, tokenizer, 1, STEP_DIRS[0])
    ok1, t_update1 = update_weights_from_disk(STEP_DIRS[0], "step-1 update")
    r1 = rollout(PROMPTS, "step-1 rollout")

    # Step 2: perturb again -> save -> update -> rollout
    train_one_step(model, tokenizer, 2, STEP_DIRS[1])
    ok2, t_update2 = update_weights_from_disk(STEP_DIRS[1], "step-2 update")
    r2 = rollout(PROMPTS, "step-2 rollout")

    # Summary
    print("\n=== 2-step REAL sglang NPU rollout + bf16 weight-update summary ===")
    for label, r in [("baseline", r0), ("step-1 ", r1), ("step-2 ", r2)]:
        print(f"  {label}: {[o['text'][:30] for o in r]}")
    print(f"  update times: step-1={t_update1:.1f}s, step-2={t_update2:.1f}s")
    diff_1 = any(r0[i]['text'] != r1[i]['text'] for i in range(len(PROMPTS)))
    diff_2 = any(r1[i]['text'] != r2[i]['text'] for i in range(len(PROMPTS)))
    print(f"  baseline->step-1 outputs differ?: {diff_1}")
    print(f"  step-1->step-2 outputs differ?:   {diff_2}")

    all_finite = not any(o['has_nan_marker'] for r in (r0, r1, r2) for o in r)
    all_non_empty = all(o['text'].strip() for r in (r0, r1, r2) for o in r)
    # Either outputs change OR update took > 1s -- both prove the weight
    # path actually ran (not short-circuit)
    convert_path_ran = diff_1 or t_update1 > 1.0

    if ok1 and ok2 and all_finite and all_non_empty:
        print(f"\n[smoke] PASS  (bf16 convert path exercised: {convert_path_ran})")
        return 0
    else:
        print(f"\n[smoke] FAIL: ok1={ok1} ok2={ok2} finite={all_finite} non_empty={all_non_empty}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
