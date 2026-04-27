# Upstream forks ledger

> The single authoritative table of (a) which personal-account fork
> we use for each NPU upstream, (b) what branch carries our work,
> (c) where the PR_MATERIAL.md handoff document is.

Branch naming convention across all forks:

```
ascend-port/<target-version-slug>
```

- `<target-version-slug>` = the upstream version we're adapting to,
  written in lowercase with hyphens (e.g. `vllm-main`, `torch-2.12-rc3`,
  `transformers-v5.4`, `triton-v3.6.0`).
- One slug = one branch. If we run a follow-up against a newer target
  (e.g. torch 2.13-rc1), open a new `ascend-port/torch-2.13-rc1`
  branch alongside the old one. Don't rebase / overwrite history.
- Prefix `ascend-port/` avoids colliding with upstream-owned branches
  like `release/v1.1.x` or community feature branches.

## Active branches

| Upstream | Personal fork | Active branch | Latest case status | PR_MATERIAL |
|---|---|---|---|---|
| vllm-ascend | [`github.com/zhshgmail/vllm-ascend`](https://github.com/zhshgmail/vllm-ascend) | [`ascend-port/vllm-main`](https://github.com/zhshgmail/vllm-ascend/tree/ascend-port/vllm-main) | 13/13 cold-drive PASS (3 F1/F2 shims) | [`PR_MATERIAL.md`](https://github.com/zhshgmail/vllm-ascend/blob/ascend-port/vllm-main/PR_MATERIAL.md) |
| torch-npu | [`gitcode.com/zhengshencn_hwca/pytorch`](https://gitcode.com/zhengshencn_hwca/pytorch) | [`ascend-port/torch-2.12-rc3`](https://gitcode.com/zhengshencn_hwca/pytorch/tree/ascend-port/torch-2.12-rc3) | 6/6 cold-drive PASS (1 F2-path-move shim) | [`PR_MATERIAL.md`](https://gitcode.com/zhengshencn_hwca/pytorch/blob/ascend-port/torch-2.12-rc3/PR_MATERIAL.md) |
| transformers | [`github.com/zhshgmail/transformers`](https://github.com/zhshgmail/transformers) | [`ascend-port/transformers-v5.4`](https://github.com/zhshgmail/transformers/tree/ascend-port/transformers-v5.4) | outcome A (no NPU drift, no patch) | [in harness repo: `docs/transformers/PR_MATERIAL_v5.4_outcome_A.md`](PR_MATERIAL_v5.4_outcome_A.md) |
| triton-ascend | [`gitcode.com/zhengshencn_hwca/triton-ascend`](https://gitcode.com/zhengshencn_hwca/triton-ascend) | [`ascend-port/triton-v3.6.0`](https://gitcode.com/zhengshencn_hwca/triton-ascend/tree/ascend-port/triton-v3.6.0) | code-side fixes complete (9 drifts); NPU smoke blocked on bishengir LLVM-22 release; vendor 6/6 baseline PASS separately | [`PR_MATERIAL.md`](https://gitcode.com/zhengshencn_hwca/triton-ascend/blob/ascend-port/triton-v3.6.0/PR_MATERIAL.md) |

## Reading order for a maintainer of one of these upstreams

1. Open this ledger row for your upstream.
2. Click the `ascend-port/<target>` branch link → see the diff vs your `main`.
3. Click the `PR_MATERIAL.md` link → read the per-fix rationale, reproducer, validation status, known limitations.
4. Apply the diff to your own fork (`git cherry-pick` or your project's submission process), open a PR.

## Updating this ledger

When a skill cold-drive lands a new `ascend-port/...` branch:
1. Push the branch to the matching personal fork.
2. Push `PR_MATERIAL.md` at branch root.
3. Add or update the row in this table.
4. The README and `STATUS` (when present) link to this ledger; do
   not duplicate the table elsewhere.
