# M3 bwd native-grad convention — diagnostic chain (2026-06-09, blue×scan-agent)

Cross-verify investigation of why `npu_nsa_select_attention_grad` dq/dk/dv mismatch
a torch autograd reference. Converged via blue's A3 experiments + scan-agent's CANN-source reading.

## Confirmed
1. **arg-order correct** (scan-agent grep + torch schema): `grad(grad, query, key, value, attention_out, softmax_max, softmax_sum, topk_indices, scale, head_num, sbs, sbc, *, atten_mask, actual_seq_qlen, actual_seq_kvlen)`.
2. **softmax_max/sum MUST be 8-wide as-is + dout bf16** (blue: 1-wide / fp32 dout → ERROR). LSE form == as-is (no change).
3. **★ topk_indices shape = `[T1, Nkv, sbc]`, NOT `[T1, Nq, sbc]`** (scan-agent, official doc). Selection is per-KV-head; G=Nq/Nkv query-heads in a group SHARE the KV-head's selected blocks. The wrong `[Tq,Nq,sbc]` shape was masked by all-select (any head slice identical) — root cause of "dense fwd matches 7e-4 but sparse breaks".
4. **fwd semantics = block-level softmax over selected blocks** (blue: topk=[block0,block5] → o matches softmax over elements [0:64]∪[320:384] to 1.3e-3). Block = 64 contiguous KV; select sbc blocks; q-heads in group share; softmax over the gathered KV.
5. **With CORRECT shape `[Tq,Nkv,sbc]` + group-shared ref: fwd matches 1.2e-3 ✓** (m3_bwd_correct.py).

## OPEN (the remaining point)
- Even with fwd-matched ref, **native grad dq/dk cos≈0.05 (near-zero), dv cos≈0.49 (partial)**. So it's a genuine native-bwd convention issue in the **score-gradient (softmax-Jacobian) / G-head aggregation** path, NOT a reference bug (fwd matches).
- dv partially right (P^T·g, no Jacobian) vs dq/dk broken → points at the P-backprop or dk G-head accumulation.
- **scan-agent source lane**: reading fwd/bwd `SelectAndGather` + dq/dk compute for: (a) intra-block dense? (b) dk G-head aggregation (4 q-heads → 1 kv-head)? (c) layout/transpose.
- blue harness `m3_bwd_correct.py` (fwd-matched ref + grad-compare) ready to validate any fix.

## NOTE on attn_sink bwd
The sink math (da = −β·Z·S/D²) was real-machine VALIDATED 9.9e-3 (scan-agent Stage B) — sink correction is correct and INDEPENDENT of this native-grad issue. Once native dq/dk/dv convention is resolved, the sink corrections (blue-reviewed closed-form, Stage A 5.6e-16) layer on top cleanly.

## UPDATE 2 (2026-06-09, after 10 probes) — bwd grad mismatch is NOT a simple convention; ruled out everything arithmetic + structural

Despite fwd matching 1.2e-3 in EVERY config, native `npu_nsa_select_attention_grad` dq/dk/dv
never matches a torch-autograd reference. Systematically ruled out:
- **D_i term** (P∘(dP−D_i)): all 3 dS variants → native dq cos≈0.053 (m3_ds_diag.py).
- **scale**: ratio test native/ref is NON-constant (dv mean0.196 std0.297, dk median 0.0) → not a missing scalar (m3_ratio.py).
- **gather-order / .sort()**: built gather-based ref in STORED topk order + scatter dk/dv back to full [Tkv,D] real positions via index_add → dk cos STILL 0.0045, dv 0.57 (m3_gather_ref.py).
- **index-space**: compared at real KV positions (full Tkv), not gathered rep → still wrong.

Signature: dv (P^T@dO, no Jacobian) stuck at cos≈0.5 across all variants despite P matching fwd
to 1e-3 — abnormal; the most-constrained gradient shouldn't be half-wrong if P+dO are right.

**Conclusion**: this is NOT a reference-construction bug we can fix by matching a convention.
Either (a) a required fwd "training-mode" / extra cached state the bwd needs that we're not
providing, (b) `dO`/`attention_out` interpreted differently than out0/g, or (c) this grad op in
CANN 8.5 / torch_npu 2.9 does not produce an autograd-matching gradient for this config.
No official autograd-comparison test bundled locally to ground-truth against.

**Next (scan-agent source lane)**: determine if fwd needs a training flag / returns extra
bwd-state; or whether there's a known limitation. **blue's 10 probes (durable here) rule out the
arithmetic/structural hypotheses** — further blind probing is diminishing returns.

**Fallback for M3**: attn_sink FWD is closed (validated). If native bwd can't be made to match
autograd, the training-side bwd path may need: tilelang sparse_mla_bwd, OR use the fwd-matched
torch reference's autograd as the NPU bwd impl reference (build dq/dk/dv from native fwd + the
verified sparse-softmax autograd). Decision pending scan-agent's source verdict + owner.
