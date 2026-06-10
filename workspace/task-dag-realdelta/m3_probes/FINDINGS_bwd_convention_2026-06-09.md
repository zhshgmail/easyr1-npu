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
