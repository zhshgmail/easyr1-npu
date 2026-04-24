---
id: 05
pattern: shim-is-not-port
trigger_phrases:
  - "pip install 算不算 port？"
  - "你在 torch npu 上基本上没做什么工作"
user_source:
  - "2026-04-24T01:19Z: '你要是 vllm ascend 的开发者会用为 torch 2.9 编译的内核做 vllm 0.20.0 的移植么？'"
  - "2026-04-24T02:29Z: '你在 torch npu 上基本上没做什么工作'"
---

# Shim-is-not-port detection

## What the user is catching

I'm calling a `pip install`, overlay wheel, or consumer-side try/except shim
a "port". User wants actual upstream source contributions, not install
recipes.

## Why it matters

A port is a contribution to the upstream library. An install is a user
action. If we're hired to port, we owe source contributions.

## Self-check before action

For any "layer N port" claim:
1. Which file under `upstream/<layer>/` did I edit?
2. Which commit hash is the edit in?
3. Which branch on my personal fork has the commit pushed?
4. If the answer to any of 1-3 is "none" / "install only", this is NOT a port.

See lesson `cross-layer-001` and `cross-layer-004`.

## My common failure mode

I label my `pip install --no-deps transformers==5.6.0` + `try/except import`
shim as "transformers 5.6 port completed". Customer opens `zhshgmail/transformers`,
sees no commits authored by me, asks "what did you port?" — and I have nothing.
