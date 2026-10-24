#!/bin/bash
# Run one heavy model/NPU test without risking the desktop.
#   npu_test.sh <mem_GB> <max_seconds> <command...>      (env for the test: NPU_TEST_ENV="A=1 B=2")
#
# - one test at a time: the test runs as the user unit "opennpu-test"; a second start is refused
# - refuses to start unless MemAvailable >= mem_GB + headroom (NPU_TEST_HEADROOM, default 6 GB, at
#   least 4), or while the OpenNPU server has a model loaded
# - hard memory cap with no swap (the test dies inside its own cgroup, not the system), killed first
#   if the whole system runs out of memory (OOMScoreAdjust=1000), and stopped after max_seconds
# - Ctrl-C / kill of this script stops the test too (a timeout around systemd-run alone leaves the
#   unit running: two 17 GB tests then ran at once and the OOM killer took the desktop, 2026-10-17)
set -u
[ $# -ge 3 ] || { echo "usage: npu_test.sh <mem_GB> <max_seconds> <command...>" >&2; exit 64; }
MEM=$1; SECS=$2; shift 2
UNIT=opennpu-test
if systemctl --user is-active --quiet "$UNIT"; then
  echo "[npu_test] another test is running ($UNIT); not starting" >&2; exit 75
fi
systemctl --user reset-failed "$UNIT" 2>/dev/null
avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
HEAD=${NPU_TEST_HEADROOM:-6}; [ "$HEAD" -lt 4 ] && HEAD=4
if [ "$avail" -lt $((MEM + HEAD)) ]; then
  echo "[npu_test] ${avail} GB available, need $((MEM + HEAD)) GB; not starting" >&2; exit 75
fi
ps_json=$(curl -s -m 3 127.0.0.1:11540/api/ps 2>/dev/null)
case "$ps_json" in *'"name"'*)
  echo "[npu_test] the OpenNPU server has a model loaded: $ps_json; unload it first" >&2; exit 75;;
esac
envs=(-E OMP_WAIT_POLICY=PASSIVE)
for kv in ${NPU_TEST_ENV:-}; do envs+=(-E "$kv"); done
trap 'systemctl --user stop "$UNIT" 2>/dev/null' EXIT
trap 'exit 130' INT TERM
# in the background + wait: bash runs a trap only after a foreground child exits, `wait` returns at once
systemd-run --user --quiet --collect --pipe --unit="$UNIT" "${envs[@]}" \
  -p MemoryMax="${MEM}G" -p MemorySwapMax=0 -p OOMScoreAdjust=1000 -p RuntimeMaxSec="$SECS" \
  "$@" &
wait $!
