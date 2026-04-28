# EasyR1 on Ascend 910C (A3) — onboarding

One-page quickstart. Two validated paths today:

| Path | Stack | Use case |
|---|---|---|
| **v1** (production) | verl-8.5.0-a3 image + EasyR1 ascend-port | "I just want EasyR1 working today, the most-tested combo." |
| **v2** (integrated) | vllm 0.20 + torch 2.11 + the 4 NPU-upstream ports overlaid + EasyR1 master | "I want the newest stack with NPU upstream gaps closed; first-validated 2026-04-28." |

## Path v1 — fastest, most-tested

```bash
docker pull quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest
git clone -b ascend-port https://github.com/zhshgmail/EasyR1.git
git clone https://github.com/zhshgmail/easyr1-npu.git
NPU_USER=<your-workspace-owner> \
  bash easyr1-npu/src/scripts/run-npu-container.sh \
    --chips 0,1 \
    --image quay.nju.edu.cn/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-latest \
    --live-source ./EasyR1 \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

Full guide: [`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md).

## Path v2 — integrated overlay (newest)

```bash
NPU_USER=<your-workspace-owner> \
  bash easyr1-npu/src/scripts/run-npu-container.sh \
    --chips 0,1 \
    --image easyr1-npu:integrated-20260427 \
    --live-source ./EasyR1 \
    -- bash /opt/easyr1/examples/qwen2_0_5b_math_grpo_npu_smoke.sh
```

The image `easyr1-npu:integrated-20260427` is built on the A3 host
from `easyr1-npu-vllm0200:iter20-abi-both` + 4 ascend-port overlay
layers. Reproducible build context: `/tmp/integrated-overlay/Dockerfile`
on the developer host (file is in this repo's git history under
`docs/_meta/RL_INTEGRATION_PLAN.md` §T22.5 references).

Full guide: [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](docs/easyr1/PORT-GUIDE-v2-integrated.md).

## Hard prerequisites (both paths)

- A3 host (Ascend 910C, x86 + NPU cards, openEuler / Ubuntu)
- CANN ≥ 8.5.0 driver installed at `/usr/local/Ascend/`
- Docker with privileged + `--ipc=host` capability
- ≥ 50 GB free in `/data/<your-workspace-owner>` for HuggingFace cache + checkpoints

## Common gotchas (both paths)

1. **`ssh -p 443 root@…`**: SSH-as-root makes `$USER=root`; the
   container runner's default mounts `/home/root` (empty). **Always
   pass `NPU_USER=<workspace-owner>`** so the right home/data/tmp
   dirs bind in.
2. **chip indices**: `--chips 0,1` is in-container indices, NOT
   host phy-id. Host pre-check `npu-smi info -t proc-mem -i N`
   before launch; runner script handles the mapping.
3. **NPU container env**: don't hand-roll `docker run`; use
   `easyr1-npu/src/scripts/run-npu-container.sh`. Hand-rolled containers
   miss critical bind-mounts (`/etc/ascend_install.info`,
   `/etc/ascend_driver.conf`, `/etc/ascend_filelist.info`); see
   `knowledge/npu-patterns.md` NPU-OPS-009 / NPU-OPS-011.

## Where things live

- Project state: [`README.md`](README.md) — current progress + 8 user paths
- All upstream fork branches: [`docs/_meta/UPSTREAM_FORKS.md`](docs/_meta/UPSTREAM_FORKS.md)
- v1 path detail: [`docs/easyr1/PORT-GUIDE.md`](docs/easyr1/PORT-GUIDE.md)
- v2 integrated path detail: [`docs/easyr1/PORT-GUIDE-v2-integrated.md`](docs/easyr1/PORT-GUIDE-v2-integrated.md)
- Skills usage (for upstream maintainers): [`docs/_meta/SKILLS-USAGE.md`](docs/_meta/SKILLS-USAGE.md)
- Knowledge base: [`knowledge/npu-patterns.md`](knowledge/npu-patterns.md), [`docs/_meta/kb/porting_lessons/`](docs/_meta/kb/porting_lessons/)

## Issues + iteration

- Open issue / question: github.com/zhshgmail/easyr1-npu/issues
- Discord: project channel (internal)
