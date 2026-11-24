#!/usr/bin/env bash
# RunPod entrypoint for atari-rl.   MODE=train|smoke (default train)
#
# Lifecycle: exit if the run is already DONE -> pull the run dir (resume) -> export pod cost ->
# periodic GCS sync -> atari-rl $MODE. No ROM step: ale-py bundles the Atari ROMs.
# The EXIT trap is installed first, so on ANY exit (finish, CRITICAL health, crash, stall exit 3,
# missing config) the run dir is synced and this pod terminates itself — billing always stops.
#
# GCS works only while the 1 h GCS_ACCESS_TOKEN lasts (startup + first syncs); after that the durable
# channel is MLflow (health, videos, checkpoints, best model). RunPod injects RUNPOD_POD_ID.
set -uo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
APP_DIR="${APP_DIR:-/app}"
POD_SELF="${APP_DIR}/runpod/scripts/pod_self.sh"
GCS_TIMEOUT="${GCS_TIMEOUT:-300}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"
SYNC_PID=""
SYNC_ENABLED=0

log() { echo "=== $* ==="; }

gcs() {
    local token_file="${WORKSPACE_DIR}/control/gcs_token"
    if [ -s "$token_file" ]; then
        CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$token_file")" timeout "$GCS_TIMEOUT" gsutil "$@"
    else
        timeout "$GCS_TIMEOUT" gsutil "$@"
    fi
}

sync_up() {
    [ "$SYNC_ENABLED" = 1 ] || return 0
    gcs -m -q rsync -r "$RUN_DIR" "${GCS_ROOT}/runs/${RUN_ID}" || log "sync to GCS failed (token expired? MLflow has the artifacts)"
}

finish() {
    local rc=$?
    trap - EXIT
    [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
    log "final sync (exit code $rc)"
    sync_up
    bash "$POD_SELF" remove || log "POD SELF-TERMINATION FAILED — remove it in the RunPod console"
    exit "$rc"
}
trap finish EXIT

for var in ATARIRL_GAME RUN_ID GCS_BUCKET; do
    [ -n "${!var:-}" ] || { log "$var not set"; exit 2; }
done
MODE="${MODE:-train}"
GAME="$ATARIRL_GAME"
GCS_ROOT="gs://${GCS_BUCKET}/atari-rl"
RUN_DIR="${WORKSPACE_DIR}/runs/${RUN_ID}"
export ATARIRL_RUNS_DIR="${WORKSPACE_DIR}/runs" ATARIRL_RUN_ID="$RUN_ID"
# A slow tracker must not stall training long enough to trip the 5-min watchdog.
export MLFLOW_HTTP_REQUEST_MAX_RETRIES="${MLFLOW_HTTP_REQUEST_MAX_RETRIES:-2}"
export MLFLOW_HTTP_REQUEST_TIMEOUT="${MLFLOW_HTTP_REQUEST_TIMEOUT:-20}"
mkdir -p "$RUN_DIR" "${WORKSPACE_DIR}/control"

if [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
    (umask 077 && printf '%s' "$GCS_ACCESS_TOKEN" > "${WORKSPACE_DIR}/control/gcs_token")
fi

DONE_STATUS="$(gcs cat "${GCS_ROOT}/runs/${RUN_ID}/DONE" 2>/dev/null || true)"
if [ "$DONE_STATUS" = budget ]; then
    # Probe-then-extend: a run stopped by its budget cap resumes from its last checkpoint.
    log "extending run ${RUN_ID} (stopped on budget)"
    gcs -q rm "${GCS_ROOT}/runs/${RUN_ID}/DONE" || true
elif [ -n "$DONE_STATUS" ]; then
    log "run ${RUN_ID} already finished (${DONE_STATUS}), nothing to do"
    exit 0
fi
SYNC_ENABLED=1

log "pulling run dir (resume)"
gcs -m -q rsync -r "${GCS_ROOT}/runs/${RUN_ID}" "$RUN_DIR" 2>/dev/null || log "no prior state, fresh run"
rm -f "${RUN_DIR}/DONE"

ATARIRL_COST_PER_HR_USD="$(bash "$POD_SELF" cost)"
export ATARIRL_COST_PER_HR_USD
log "pod cost ${ATARIRL_COST_PER_HR_USD} USD/h"

sync_loop() {
    # Kill our own sleep on TERM, otherwise it outlives the loop and keeps stdout open.
    local sleep_pid=""
    trap '[ -n "$sleep_pid" ] && kill "$sleep_pid" 2>/dev/null; exit 0' TERM
    while true; do
        sleep "$SYNC_INTERVAL" &
        sleep_pid=$!
        wait "$sleep_pid"
        sync_up
    done
}
sync_loop &
SYNC_PID=$!

log "atari-rl ${MODE} (game=${GAME}, run=${RUN_ID})"
atari-rl "$MODE"
exit $?
