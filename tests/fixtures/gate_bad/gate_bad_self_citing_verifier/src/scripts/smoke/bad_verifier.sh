#!/bin/bash
# FRAUD: this verifier reads the claim's smoke log path AND echoes the
# grep_assertion back as if it had independently measured it. Anti-cycle
# violation: data-flow depends on the claim file itself.
#
# M2 scanner must reject this — it references a path that appears in
# the manifest's evidence[].path set.

LOG_PATH="tests/fixtures/gate_bad/gate_bad_self_citing_verifier/workspace/smoke.log"
if [ -f "$LOG_PATH" ]; then
  echo "verifier claims: $(cat $LOG_PATH | grep ok)"
fi
echo "fake-independent measurement: ok"
