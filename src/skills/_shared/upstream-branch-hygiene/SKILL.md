---
name: upstream-branch-hygiene
description: Enforces the "modify locally, push to personal fork, pull on remote" workflow for every upstream dependency being ported. Prevents work loss from shared servers, container teardown, or host cleanup. Use whenever you need to change code in upstream/<repo>/ — whether that's EasyR1, transformers, vllm-ascend, torch-npu, triton-ascend, or any other dependency we're porting from.
---

# upstream-branch-hygiene

## When to use

Invoke this skill whenever you're about to:
- Change a file under `upstream/<repo>/`.
- Commit a change to an upstream repo's `ascend-port` (or equivalent) branch.
- Push that change to a remote where the NPU server will pull from.
- Pull the change on the NPU host to exercise it.

Also invoke it when **reviewing** someone else's proposed change to an upstream, to check it follows the discipline.

## When NOT to use

- Changes that live entirely in this repo (our own project source-of-truth — `easyr1-npu` itself, not the things under `upstream/`). Those push straight to `github.com/zhshgmail/easyr1-npu`, no `personal` dance.
- One-shot debug edits on the remote that are expected to be thrown away within the same SSH session (the user called these out as "非常小的 debug，不需要记录的" — "very small debugs don't need recording"). Still keep those short; never let them accumulate.

## The rule

**All non-trivial modifications to upstream dependencies must flow:**

```
local edit → local commit → git push personal <branch> → remote git pull personal <branch>
```

**Never:**
- Edit source on the NPU host directly in a checkout that might be pulled from elsewhere later (e.g. inside a container's bind-mounted source dir) without the change also landing in a personal fork branch.
- Hot-patch installed packages inside a container (`pip install --force-reinstall` excepted when it's captured in the Dockerfile).
- Rely on `origin/<upstream>` for anything we've changed — once we modify, `personal/<branch>` is the source of truth.

## Why (the incident this exists to prevent)

The A3 host `115.190.166.102` is **shared**. Other users operate on `/home/baymax/`, `/home/wjq/`, etc. There is no guarantee that our container, our filesystem, or our branch checkouts will still be there tomorrow:
- Containers get pruned to reclaim disk (root fs was 93% used at onboarding).
- Admins may wipe `/root/` or `/tmp/`.
- A failed `docker system prune` can take out stopped containers carrying uncommitted edits.

If an important change only exists in a running container's filesystem or in a detached checkout on the NPU host, it is one `docker rm` away from being lost. All porting work on this project has had >1 hour of effort per change; losing one is expensive.

## Setup (once per repo)

For each upstream repo we may modify, on the **local workstation**:

1. Fork on GitHub/GitCode if not already done. GitHub forks default public — immediately flip to private via API:
   ```bash
   gh repo fork <owner>/<repo> --clone=false --default-branch-only=false
   gh api --method PATCH repos/zhshgmail/<repo> -f visibility=private
   ```
   For GitCode: `gc repo fork <owner>/<repo>`. Private toggle via web UI or `gc repo create --private` pattern.

2. In the local `upstream/<repo>/` clone, set up remotes:
   ```bash
   cd upstream/<repo>
   # 'origin' should still point at the upstream project (hiyouga/EasyR1, huggingface/transformers, etc.)
   git remote -v
   # Add 'personal' pointing at our fork:
   git remote add personal https://github.com/zhshgmail/<repo>.git   # or gitcode equivalent
   git remote -v
   ```

3. Record the upstream+personal pair in `knowledge/upstream-refs.md` under the "Private forks" section so future sessions see it.

## Working flow (every change)

From the **local workstation**, inside `upstream/<repo>/`:

1. **Branch**: `git checkout -b <task>` or `git checkout ascend-port` (the default branch name we use for NPU port work).
2. **Edit** locally — use editor + tests on the x86 box where feasible.
3. **Commit** with a descriptive message. Follow project commit style when possible. Do NOT include Claude attribution per the user's global instruction.
4. **Push**: `git push personal <branch>`.
5. **Pull on NPU host** (from a cold shell, or via `ssh`):
   ```bash
   cd "$HOME/workspace/easyr1-npu/upstream/<repo>"
   GITHUB_TOKEN=$(gh auth token) GIT_ASKPASS=/root/.git-askpass-github.sh \
     git pull --ff-only personal <branch>
   ```
   (The askpass helper is at `/root/.git-askpass-github.sh` on the A3 host, set up once at onboarding.)
6. **Rebuild**: if the change affects the docker image (Dockerfile or anything it COPYs), rebuild on the NPU host with `docker build -t <tag> -f Dockerfile.npu .`. Otherwise, a bind-mount of the source into the container picks up the change without rebuild.

## Handling the NPU host's "don't edit directly" rule

The NPU host has a full git checkout at `$HOME/workspace/easyr1-npu/upstream/<repo>/`. It is **read-only in spirit** — we use it for `git pull` and nothing else.

If a change is needed urgently and you're already on the NPU host:

1. Make the edit.
2. **Before closing the session**, push it from the NPU host to `personal`:
   ```bash
   cd "$HOME/workspace/easyr1-npu/upstream/<repo>"
   git diff                  # review
   git add -A && git commit -m "..."
   GITHUB_TOKEN=... GIT_ASKPASS=/root/.git-askpass-github.sh git push personal <branch>
   ```
3. **Pull that change back on the local workstation** before doing anything else there, so the two checkouts don't diverge.

This is a **last resort**, not the default. Prefer local edit → push → pull.

## Container discipline

All NPU containers run with the three user-scoped bind mounts so source is preserved:
- `/home/${NPU_USER}:/home/${NPU_USER}`
- `/data/${NPU_USER}:/data/${NPU_USER}`
- `/tmp/${NPU_USER}:/tmp/${NPU_USER}`

(`NPU_USER` defaults to `$USER` — see `scripts/run-npu-container.sh`.)

Containers are created with `--rm` so they go away on exit. Source must live outside — either via bind mount (for iterative work) or baked into the image via `COPY` (for reproducible layers). **Never** leave an edited file in an unmounted container path.

## Gotchas we've seen

- **`cd && ...` in one-line ssh**: `ssh host "cd /path && cmd"` works, but `ssh host "VAR=\$(locally-substituted) cd /path && cmd"` can have the `cd` get eaten by shell substitution. Use `;` instead of `&&` or wrap in an explicit `bash -c` when both env vars and cd are in play.
- **GitHub forks default public**: always run the private-flip API call right after `gh repo fork`.
- **`--no-deps` / `--force-reinstall` inside a running container**: those don't persist after container destruction. If the fix is needed every time, encode it in the Dockerfile (see `Dockerfile.npu`'s triton-ascend reinstall).
- **`pip install -e .` inside a container** leaves an editable link to the build-time COPY path. If you bind-mount a different host path over it, imports resolve to the bind-mount (sometimes surprising if you wanted the baked version). Check `pip show <pkg>` `Editable project location` when debugging.

## Review checklist

When reviewing a proposed upstream change:

- [ ] Was it made on a local workstation, not the NPU host?
- [ ] Is the commit message accurate and free of "Claude" attribution?
- [ ] Does `git log personal/<branch>` show the commit?
- [ ] Does the NPU host's checkout show the commit after `git pull personal <branch>`?
- [ ] Has any downstream image rebuild happened if the change touches the Dockerfile layer?
- [ ] Is the change documented in `docs/easyr1/porting-journal.md` if it's non-trivial?

If any box is unchecked, the change is not properly landed yet.
