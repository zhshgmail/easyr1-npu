// dag.workflow.js — deterministic DAG executor for the task-dag-planner skill.
//
// Runs a caller-supplied task DAG in topological order: nodes with no
// inter-dependency are grouped into the same wave and fanned out with
// parallel(); dependent nodes wait for their wave's predecessors. After
// execution a Review phase spawns one adversarial verifier per node to
// challenge the node agent's claimed output. Returns { completed, failed,
// verifications }.
//
// Contract notes:
//   - meta below is a PURE LITERAL (no computed values, no interpolation).
//   - The DAG is read from the global `args` (the Workflow tool injects it).
//   - We intentionally avoid the three forbidden runtime builtins: the
//     millisecond-timestamp call, the random-float call, and the argless
//     date constructor. Ordering is derived structurally from deps, not wall
//     clock; ids are taken from the caller, never generated from a clock.
//
// Expected args shape (inline example):
//   args = {
//     task_dag: {
//       nodes: [
//         { id: "sweep",   prompt: "Run npu-code-path-sweep on the repo",      deps: [],                 parallelizable: true  },
//         { id: "deps",    prompt: "Classify runtime deps A/B/C/D/E",          deps: [],                 parallelizable: true  },
//         { id: "fixes",   prompt: "Apply the 5 archetype NPU port fixes",     deps: ["sweep", "deps"],  parallelizable: false },
//         { id: "image",   prompt: "Build the integrated overlay image",       deps: ["fixes"],          parallelizable: false },
//         { id: "smoke",   prompt: "Run V1.4 GRPO smoke and assert numerics",  deps: ["image"],          parallelizable: false }
//       ]
//     }
//   }
//   // sweep + deps run in wave 0 (parallel); fixes in wave 1; image, smoke serial after.

export const meta = {
  name: "task-dag",
  description:
    "Deterministic DAG executor: topologically sorts task_dag.nodes by deps, fans out independent siblings in parallel waves, chains dependents, then adversarially verifies each node's claimed output.",
  phases: ["plan", "execute", "review"],
};

// ---------------------------------------------------------------------------
// Phase: plan — validate the DAG and group nodes into topological waves.
// ---------------------------------------------------------------------------

// Read the DAG out of the global args, tolerating either { task_dag: {...} }
// or a bare { nodes: [...] } at the top level.
function readNodes() {
  const dag = (args && args.task_dag) ? args.task_dag : args;
  const nodes = (dag && Array.isArray(dag.nodes)) ? dag.nodes : [];
  return nodes;
}

// Normalize a single node, defaulting missing fields. parallelizable defaults
// to true (a node with no special serialization need can share a wave).
function normalizeNode(node) {
  return {
    id: String(node.id),
    prompt: String(node.prompt == null ? "" : node.prompt),
    deps: Array.isArray(node.deps) ? node.deps.map(String) : [],
    parallelizable: node.parallelizable !== false,
  };
}

// Kahn topological sort that buckets ready nodes into ordered waves. A wave is
// the set of nodes whose deps are all satisfied by earlier waves. Within a
// wave, nodes that opt out of parallelizable are peeled into their own
// singleton waves so they act as serialization barriers. Detects cycles and
// dangling deps and returns them as structured errors rather than throwing.
function planWaves(nodes) {
  const byId = {};
  for (const n of nodes) byId[n.id] = n;

  const errors = [];

  // Dangling-dep check.
  for (const n of nodes) {
    for (const d of n.deps) {
      if (!byId[d]) {
        errors.push({ id: n.id, reason: "missing dependency: " + d });
      }
    }
  }

  // Remaining-dependency count per node.
  const remaining = {};
  for (const n of nodes) {
    remaining[n.id] = n.deps.filter((d) => byId[d]).length;
  }

  const settled = {};
  const waves = [];
  let safety = nodes.length + 1;

  while (Object.keys(settled).length < nodes.length && safety > 0) {
    safety = safety - 1;

    // Nodes whose deps are all settled and that are not yet settled.
    const ready = nodes
      .filter((n) => !settled[n.id] && remaining[n.id] === 0)
      .map((n) => n.id);

    if (ready.length === 0) break; // no progress => cycle (handled below)

    // Split the ready set: parallelizable nodes share one wave; each
    // non-parallelizable node becomes its own singleton barrier wave.
    const parallelWave = ready.filter((id) => byId[id].parallelizable);
    const serialIds = ready.filter((id) => !byId[id].parallelizable);

    if (parallelWave.length > 0) waves.push(parallelWave);
    for (const sid of serialIds) waves.push([sid]);

    for (const id of ready) settled[id] = true;
    // Decrement remaining counts for dependents of just-settled nodes.
    for (const n of nodes) {
      if (settled[n.id]) continue;
      remaining[n.id] = n.deps.filter((d) => byId[d] && !settled[d]).length;
    }
  }

  // Anything unsettled is part of a dependency cycle.
  for (const n of nodes) {
    if (!settled[n.id]) {
      errors.push({ id: n.id, reason: "unsettled (dependency cycle)" });
    }
  }

  return { waves, errors, byId };
}

// ---------------------------------------------------------------------------
// Schemas — structured returns so downstream stages can branch deterministically.
// ---------------------------------------------------------------------------

const nodeResultSchema = {
  type: "object",
  properties: {
    id: { type: "string" },
    status: { type: "string", enum: ["done", "failed"] },
    output: { type: "string" },
    artifacts: { type: "array", items: { type: "string" } },
  },
  required: ["id", "status", "output"],
};

const verificationSchema = {
  type: "object",
  properties: {
    id: { type: "string" },
    verdict: { type: "string", enum: ["pass", "fail", "uncertain"] },
    rationale: { type: "string" },
    evidence: { type: "array", items: { type: "string" } },
  },
  required: ["id", "verdict", "rationale"],
};

// ---------------------------------------------------------------------------
// Prompt builders.
// ---------------------------------------------------------------------------

function executionPrompt(node, byId) {
  const depLines = node.deps.length
    ? node.deps.map((d) => "  - " + d).join("\n")
    : "  - (none)";
  return [
    "You are the node agent for DAG node id=" + node.id + ".",
    "Upstream dependencies (already completed):",
    depLines,
    "",
    "TASK:",
    node.prompt,
    "",
    "Do the work. When done, return the result schema with status=done and a",
    "concise factual `output` describing what you produced. List concrete",
    "artifact paths (absolute) in `artifacts`. If you could not complete the",
    "task, return status=failed and put the blocking reason in `output` —",
    "do NOT fabricate success.",
  ].join("\n");
}

function reviewPrompt(node, result) {
  return [
    "You are an ADVERSARIAL verifier for DAG node id=" + node.id + ".",
    "The node agent claimed the following output:",
    "",
    "--- CLAIMED OUTPUT ---",
    result && result.output ? result.output : "(no output captured)",
    "--- END CLAIMED OUTPUT ---",
    "",
    "Claimed status: " + (result && result.status ? result.status : "unknown"),
    "Claimed artifacts: " +
      (result && Array.isArray(result.artifacts) && result.artifacts.length
        ? result.artifacts.join(", ")
        : "(none listed)"),
    "",
    "The original task was:",
    node.prompt,
    "",
    "Your job is to DISPROVE the claim, not to rubber-stamp it. Independently",
    "check the artifacts exist and actually contain what was claimed. Treat",
    "reduced-scope / smoke-only / stub / synthetic-placeholder results as NOT",
    "equivalent to the full claim — call those out explicitly. Return verdict:",
    "  pass      = claim independently confirmed by evidence",
    "  fail      = claim contradicted, or evidence missing/fabricated",
    "  uncertain = cannot verify either way; say what evidence is missing",
    "Put concrete file paths / command outputs you checked in `evidence`.",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Workflow body.
// ---------------------------------------------------------------------------

export default async function run() {
  // ----- Phase: plan -----
  phase("plan");

  const rawNodes = readNodes();
  if (rawNodes.length === 0) {
    log("plan: no nodes supplied in args.task_dag.nodes — nothing to run");
    return { completed: [], failed: [], verifications: [] };
  }

  const nodes = rawNodes.map(normalizeNode);
  const { waves, errors, byId } = planWaves(nodes);

  log(
    "plan: " +
      nodes.length +
      " node(s) sorted into " +
      waves.length +
      " wave(s); " +
      errors.length +
      " structural error(s)",
  );
  waves.forEach((wave, i) => {
    log("plan: wave " + i + " = [" + wave.join(", ") + "]");
  });

  const completed = [];
  const failed = [];

  // Nodes implicated in cycles / missing deps are failed up front and never
  // dispatched (their inputs can never be satisfied deterministically).
  const blockedIds = {};
  for (const e of errors) {
    if (!blockedIds[e.id]) {
      blockedIds[e.id] = true;
      failed.push({ id: e.id, status: "failed", output: "plan error: " + e.reason });
    }
  }

  // ----- Phase: execute -----
  phase("execute");

  // resultsById accumulates every node's structured result so the Review
  // phase can challenge each one. A node is dispatched only if it was not
  // blocked at plan time and all its deps actually completed.
  const resultsById = {};

  function depsSatisfied(node) {
    return node.deps.every((d) => resultsById[d] && resultsById[d].status === "done");
  }

  for (let w = 0; w < waves.length; w = w + 1) {
    const waveIds = waves[w].filter((id) => !blockedIds[id]);

    // Partition this wave into runnable vs skipped (a predecessor failed).
    const runnable = waveIds.filter((id) => depsSatisfied(byId[id]));
    const skipped = waveIds.filter((id) => !depsSatisfied(byId[id]));

    for (const id of skipped) {
      const r = {
        id,
        status: "failed",
        output: "skipped: an upstream dependency did not complete",
      };
      resultsById[id] = r;
      failed.push(r);
      log("execute: wave " + w + " SKIP " + id + " (upstream failed)");
    }

    if (runnable.length === 0) continue;

    log(
      "execute: wave " +
        w +
        " dispatching [" +
        runnable.join(", ") +
        "]" +
        (runnable.length > 1 ? " in parallel" : ""),
    );

    // A single runnable node executes inline; multiple independent nodes fan
    // out through parallel() as a barrier — the wave does not advance until
    // every thunk resolves.
    let waveResults;
    if (runnable.length === 1) {
      const node = byId[runnable[0]];
      const r = await agent(executionPrompt(node, byId), {
        label: "exec:" + node.id,
        phase: "execute",
        schema: nodeResultSchema,
      });
      waveResults = [r];
    } else {
      const thunks = runnable.map((id) => {
        const node = byId[id];
        return () =>
          agent(executionPrompt(node, byId), {
            label: "exec:" + node.id,
            phase: "execute",
            schema: nodeResultSchema,
          });
      });
      waveResults = await parallel(thunks);
    }

    // Record results, normalizing the id back onto each (agents may omit it).
    runnable.forEach((id, idx) => {
      const r = waveResults[idx] || {};
      const norm = {
        id,
        status: r.status === "done" ? "done" : "failed",
        output: r.output == null ? "" : String(r.output),
        artifacts: Array.isArray(r.artifacts) ? r.artifacts : [],
      };
      resultsById[id] = norm;
      if (norm.status === "done") {
        completed.push(norm);
      } else {
        failed.push(norm);
      }
    });
  }

  // ----- Phase: review -----
  phase("review");

  // Adversarially verify every node that actually ran (completed OR failed —
  // a "failed" self-report is itself a claim worth checking, and a "done" may
  // be over-claimed). Skipped/plan-blocked nodes have nothing to verify.
  const reviewable = nodes
    .map((n) => n.id)
    .filter((id) => resultsById[id] && !blockedIds[id])
    .filter((id) => resultsById[id].output !== "skipped: an upstream dependency did not complete");

  let verifications = [];
  if (reviewable.length === 0) {
    log("review: nothing executed to verify");
  } else {
    log("review: adversarially verifying " + reviewable.length + " node(s)");
    const verifyThunks = reviewable.map((id) => {
      const node = byId[id];
      const result = resultsById[id];
      return () =>
        agent(reviewPrompt(node, result), {
          label: "verify:" + id,
          phase: "review",
          schema: verificationSchema,
        });
    });
    const rawVerifs = await parallel(verifyThunks);
    verifications = reviewable.map((id, idx) => {
      const v = rawVerifs[idx] || {};
      return {
        id,
        verdict:
          v.verdict === "pass" || v.verdict === "fail" ? v.verdict : "uncertain",
        rationale: v.rationale == null ? "" : String(v.rationale),
        evidence: Array.isArray(v.evidence) ? v.evidence : [],
      };
    });
  }

  // A node that self-reported done but failed adversarial review is demoted
  // from completed to failed so the caller never trusts an unverified claim.
  for (const v of verifications) {
    if (v.verdict === "fail") {
      const idx = completed.findIndex((c) => c.id === v.id);
      if (idx >= 0) {
        const demoted = completed.splice(idx, 1)[0];
        demoted.status = "failed";
        demoted.output =
          demoted.output + " | review FAIL: " + v.rationale;
        failed.push(demoted);
        log("review: demoting " + v.id + " (claimed done, verification failed)");
      }
    }
  }

  log(
    "done: completed=" +
      completed.length +
      " failed=" +
      failed.length +
      " verifications=" +
      verifications.length,
  );

  return { completed, failed, verifications };
}
