# M3 — attn_sink BACKWARD closed-form derivation (2026-06-09, scan-agent)

Companion to the fwd adapter (`nsa_attn_sink_adapter.py`). Goal: get the gradients of the
sink-adjusted output `o_sink` w.r.t. `q, k, v, attn_sink`, **reusing the native
`npu_nsa_select_attention_grad`** for the bulk + a small analytic correction (the Jacobian
cross-term blue flagged).

## Forward recap (per row i, head h, over the selected keys n)

```
s_n      = scale · <q_i, k_n>
m        = max_n s_n                       # softmax_max  = native out1  (take [...,0])
e_n      = exp(s_n − m)
Z        = Σ_n e_n                         # softmax_sum  = native out2  (take [...,0])
S        = exp(a_h − m)                     # sink term, a_h = attn_sink[h]
D        = Z + S                            # sink-adjusted denominator
num      = Σ_n e_n v_n
o_native = num / Z                          # native out0  (no sink)
o_sink   = num / D = o_native · (Z/D) = o_native · r,   r ≡ Z/D ∈ (0,1]
```

So the fwd is just a per-(row,head) scalar rescale `r = Z/D = sum_exp/denom` (already validated).

## Backward

Upstream cotangent `g = ∂L/∂o_sink` (shape T,N,D_v). Define two per-(row,head) scalars:

```
r = Z / D
β = <g, o_native>          # dot over D_v  (== <g, o_sink>/r)
```

### Key identity (lets us reuse native_grad)

`o_sink = r · o_native`, and `r` depends on q,k (through Z, m). Product rule:

```
∂L/∂x = Σ g · ∂o_sink/∂x
      = Σ g · ( r·∂o_native/∂x + o_native·∂r/∂x )
      = NativeGrad_x(  g·r  )  +  β · ∂r/∂x            for x ∈ {q,k}
```

- **First term** = the native backward fed a *scaled cotangent* `dout' = g·r`. This is the
  "first-order `sum_exp/denom` scaling" intuition.
- **Second term** `β·∂r/∂x` = the **Jacobian cross-term** (blue's warning: first-order alone is
  NOT enough when the sink is non-negligible).

### The cross-term ∂r/∂x (max-shift cancels — clean)

```
∂r = [S·∂Z − Z·∂S] / D²
∂Z = Σ_n e_n(∂s_n − ∂m) = Σ_n e_n ∂s_n − Z·∂m
∂S = −S·∂m
⇒ S·∂Z − Z·∂S = S·Σ_n e_n ∂s_n − S·Z·∂m + Z·S·∂m = S·Σ_n e_n ∂s_n     (∂m cancels)
⇒ ∂r/∂x = (S / D²) · Σ_n e_n · ∂s_n/∂x
```

With `s_n = scale·<q_i,k_n>` ⇒ `∂s_n/∂q_i = scale·k_n`, `∂s_n/∂k_n = scale·q_i`.

### Final closed-form

```
dout'  = g · r                                   # r broadcast over D_v
dv     = NativeGrad_v(dout')                     # r doesn't touch v
dq_i   = NativeGrad_q(dout')_i + β·(S/D²)·scale·Σ_n e_n k_n
dk_n   = NativeGrad_k(dout')_n + β·(S/D²)·scale·e_n·q_i
da_h   = − Σ_{rows i→h} β · (Z·S / D²)            # ∂o_sink/∂a = −o_native·Z·S/D², dotted with g = −β·Z·S/D²
```

`Σ_n e_n k_n` is a second (unnormalized) attention-style contraction of the native exp-scores with
**K** (instead of V) — the only genuinely new compute the sink backward needs.

### Numerical boundary (must test)

- **Small / very-negative sink** (`S ≪ Z`): `r→1`, `S/D²→0` ⇒ cross-term→0, `dq/dk→native`,
  `dv→native` — backward degenerates to the native grad (sanity check #1).
- **Sink comparable to Z** (`S ~ Z`): cross-term is first-order-significant. Validation must show
  **first-order-only (drop β·∂r/∂x) FAILS** vs autograd on dq/dk, while the **full form PASSES** —
  this is the empirical proof the cross-term is required (blue's key point).

## Validation plan

- **Stage A (pure torch, CPU — math proof)**: build dense-with-sink reference, autograd → ref
  dq/dk/dv/da; compute the closed-form using a torch stand-in for native fwd/grad; show full-form
  matches + first-order-only fails at large sink.
- **Stage B (real kernel, NPU `a5ops-a3-scan`)**: native fwd + `npu_nsa_select_attention_grad`
  (dout'=g·r) + analytic corrections vs torch autograd-with-sink. Closes the milestone with blue's
  independent 8.5-A3 cross-verify + code review (per-milestone discipline, same as fwd).
