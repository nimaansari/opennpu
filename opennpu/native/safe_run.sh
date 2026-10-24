#!/bin/bash
# Run an NPU test without crash dumps / apport popups and with a hard timeout.
#   safe_run.sh <timeout_s> <cmd...>
ulimit -c 0
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AIRENV=${OPENNPU_AIRENV:-$(cd "$ROOT" && python3 -c "from opennpu import config; print(config.AIRENV)")}
set +u; source "$AIRENV" >/dev/null; set -u
T=$1; shift
cd "$ROOT"
timeout --kill-after=30 "$T" "$@"
rc=$?
[ $rc -eq 124 ] && echo "[safe_run] TIMEOUT after ${T}s"
exit $rc
