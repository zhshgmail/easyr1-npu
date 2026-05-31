# sglang-miles branch analysis (2026-05-30)

Task #275: investigate whether we need to switch the PoC baseline from
`sgl-project/sglang main` to the `sglang-miles` branch.

## Branch state

- branch HEAD: `e94a84e0`
- ahead of main by: 38 commits
- behind main by: 650 commits
- files changed: 134

So this branch is **significantly behind main** in routine improvements,
but carries 38 miles-specific commits that the upstream maintainers
sequence as `[N/14] [sglang-miles]`.

## The 14 miles-specific commits

| # | sha | feature | upstream PR |
|---|---|---|---|
| 1 | f5b3fd2b | True on-policy training support for FSDP2 | #18639 |
| 2 | 1a831978 | **R3 (Rollout Routing Replay) DeepEP and MTP support** | #18642 |
| 3 | 5bd6c6bc | INT4 QAT for RL | #18565 |
| 4 | ded3a783 | PD disaggregation for RL | #18646 |
| 5 | b5697ff8 | MTP related fix | #18647 |
| 6 | 6efbb87d | VLM multimodal fallback fixes | #18781 |
| 7 | 0b891eb2 | Better TITO token return | #19731 |
| 8 | 1ca4d304 | Cross-turn token after last user message | #20066 |
| 9 | 0b891eb2 | **P2P weight update support and fixes** | #21278, #22663 |
| 10 | 02449619 | Pause-aware weight update deadlock fixes | #22754, #22623 |
| 11 | 329828ba | R3 on PD disaggregation mini_lb | #22916 |
| 12 | 6ce6b5f2 | PD pause handling | #23672, #23887 |
| 13 | 44f350d1 | KimiK2 raw tool call id parser | #25196 |
| 14 | e687c743 | True on-policy qwen_dense support | (no separate PR) |
| extra | 53e04836 | Cherry-pick GemmaRMSNorm gemma_weight buffer storage fix | #26429 |

## Cross-check: are these features available on main?

`#21278 P2P Weight Update features for miles` -- merged 2026-04-01 to
**main** (then cherry-picked into sglang-miles). HTTP routes added by
this PR (`/init_weights_send_group_for_remote_instance`,
`/send_weights_to_remote_instance`,
`/get_remote_instance_transfer_engine_info`, etc.) ARE present in our
v0.5.12.post2 image. Verified by grep of
`/sgl-workspace/sglang/python/sglang/srt/entrypoints/http_server.py`.

`#18642 R3 DeepEP and MTP support` -- this is the DeepEP/MTP extension
of R3 (which itself shipped in v0.5.7 via #12162). The base R3 plumbing
(plain `return_routed_experts`) is in our v0.5.12.post2 image and
verified working on NPU (task #276 + #277).

`#22916 R3 on PD disaggregation mini_lb` -- merged to main 2026-03 too
(this commit lineage shows it on main first, then to miles).

## Conclusion: stay on main

For our current PoC scope:

- **R3 base path**: on main, verified working on NPU.
- **R3 DeepEP/MTP extensions**: needed only if we enable DeepEP a2a or
  MTP. Our miles_local config doesn't (TP=1, no DeepEP, no MTP), so
  the main-branch R3 is sufficient.
- **P2P weight update**: this is the workaround for Bug B #26794
  (FusedMoE reload narrow regression). The endpoints are on main and
  in our image. Switching the PoC to sglang-miles branch is NOT
  required to access P2P -- we can call those endpoints directly from
  our current image.
- **FSDP2 / INT4 QAT / PD disaggregation**: not in our PoC scope yet.

**Recommendation**: do NOT switch the PoC baseline. The features we
need (R3 base + P2P weight update) are on main and in our image.
If we later need DeepEP-specific R3 or PD-specific R3, only then
revisit sglang-miles.

The 650 commits gap between sglang-miles HEAD and main also makes
sglang-miles a worse base for our work: we'd inherit known regressions
that main has already fixed.

## Implication for task #275

Task closed as **WORKAROUND: stay on main** -- no branch switch needed.
The miles team's sglang-miles branch is their own RL-integration staging
area; consuming it from our PoC would entail tracking + rebasing their
patchset, which doesn't pay off when the relevant features are already
on main.
