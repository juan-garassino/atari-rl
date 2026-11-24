#!/usr/bin/env bash
# RunPod entrypoint for atari-rl.
#
#   MODE=train|smoke   (default train)
#
# Lifecycle: GCS auth -> exit if the run is already DONE -> pull the run dir (resume) ->
# export pod cost -> periodic sync -> atari-rl $MODE ->
# final sync -> terminate this pod. The EXIT trap guarantees the last two steps, so
# billing stops whatever happens (finish, CRITICAL health, crash, stall exit code 3).
#
# Required env: RUN_ID, GCS_BUCKET, ATARIRL_GAME (pong|breakout); GCS_ACCESS_TOKEN (short-lived, refreshed by
# refresh-token.yml into $WORKSPACE_DIR/control/gcs_token). RunPod injects RUNPOD_POD_ID.
set -uo pipefail

MODE="${MODE:-train}"
GAME="${ATARIRL_GAME:?ATARIRL_GAME not set}"
RUN_ID="${RUN_ID:?RUN_ID not set}"
GCS_BUCKET="${GCS_BUCKET:?GCS_BUCKET not set}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
APP_DIR="${APP_DIR:-/app}"
SYNC_INTERVAL="${SYNC_INTERVAL:-600}"

GCS_ROOT="gs://${GCS_BUCKET}/atari-rl"
RUN_DIR="${WORKSPACE_DIR}/runs/${RUN_ID}"
TOKEN_FILE="${WORKSPACE_DIR}/control/gcs_token"
POD_SELF="${APP_DIR}/runpod/scripts/pod_self.sh"

export ATARIRL_RUNS_DIR="${WORKSPACE_DIR}/runs" ATARIRL_RUN_ID="$RUN_ID"
mkdir -p "$RUN_DIR" "$(dirname "$TOKEN_FILE")"

log() { echo "=== $* ==="; }

if [ -n "${GCS_ACCESS_TOKEN:-}" ]; then
    umask 077
    printf '%s' "$GCS_ACCESS_TOKEN" > "${TOKEN_FILE}.new" && mv "${TOKEN_FILE}.new" "$TOKEN_FILE"
    umask 022
fi

gcs() {
    # Re-read the token before every call: refresh-token.yml rewrites it every 50 min.
    if [ -s "$TOKEN_FILE" ]; then
        CLOUDSDK_AUTH_ACCESS_TOKEN="$(cat "$TOKEN_FILE")" gsutil "$@"
    else
        gsutil "$@"
    fi
}

sync_up() {
    gcs -m -q rsync -r "$RUN_DIR" "${GCS_ROOT}/runs/${RUN_ID}" || log "sync to GCS failed (retrying next interval)"
}

SYNC_PID=""
finish() {
    local rc=$?
    trap - EXIT
    [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
    log "final sync (exit code $rc)"
    sync_up
    bash "$POD_SELF" remove
    exit "$rc"
}

if gcs -q stat "${GCS_ROOT}/runs/${RUN_ID}/DONE" 2>/dev/null; then
    log "run ${RUN_ID} already DONE, nothing to do"
    bash "$POD_SELF" remove
    exit 0
fi

trap finish EXIT

log "pulling run dir (resume)"
gcs -m -q rsync -r "${GCS_ROOT}/runs/${RUN_ID}" "$RUN_DIR" 2>/dev/null || log "no prior state, fresh run"

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
