"""Entrypoint behaviour with stubbed gsutil / curl / atari-rl on PATH (no cloud, no pod)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

GSUTIL = r'''#!/usr/bin/env bash
echo "gsutil $*" >> "$CALLS"
[ -n "${STUB_GSUTIL_SLEEP:-}" ] && case "$*" in *rsync*) exec sleep "$STUB_GSUTIL_SLEEP" ;; esac
case "$*" in
  *cat*DONE*) [ -n "${STUB_DONE:-}" ] && { echo "$STUB_DONE"; exit 0; } || exit 1 ;;
  *rm*DONE*) exit 0 ;;
  *rsync*) [ "${STUB_RSYNC_FAIL:-0}" = 1 ] && exit 1 || exit 0 ;;
esac
exit 0
'''
CURL = r'''#!/usr/bin/env bash
echo "curl $*" >> "$CALLS"
case "$*" in
  *costPerHr*) echo '{"data":{"pod":{"costPerHr":0.5}}}' ;;
  *podTerminate*)
    n=$(grep -c podTerminate "$CALLS")
    if [ "$n" -le "${STUB_TERMINATE_FAILS:-0}" ]; then echo '{"errors":[{"message":"try again"}]}'; else echo '{"data":{"podTerminate":null}}'; fi ;;
esac
'''
ATARI_RL = r'''#!/usr/bin/env bash
echo "atari-rl $* game=${ATARIRL_GAME:-} cost=${ATARIRL_COST_PER_HR_USD:-} retries=${MLFLOW_HTTP_REQUEST_MAX_RETRIES:-} timeout=${MLFLOW_HTTP_REQUEST_TIMEOUT:-}" >> "$CALLS"
exit "${STUB_TRAIN_RC:-0}"
'''


def _trainer_ran(calls: str) -> bool:
    return any(line.startswith("atari-rl ") for line in calls.splitlines())


def _run(tmp_path: Path, **env_overrides) -> tuple[int, str, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in {"gsutil": GSUTIL, "curl": CURL, "atari-rl": ATARI_RL}.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    calls = tmp_path / "calls.log"
    calls.touch()
    ws = tmp_path / "ws"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}", "CALLS": str(calls), "WORKSPACE_DIR": str(ws),
        "APP_DIR": str(REPO), "RUN_ID": "t", "GCS_BUCKET": "bucket", "MODE": "train",
        "ATARIRL_GAME": "breakout", "RUNPOD_POD_ID": "pod1", "RUNPOD_API_KEY": "k", "SYNC_INTERVAL": "3600",
        "HOME": str(tmp_path), "POD_SELF_BACKOFF": "0",
    }
    env.update(env_overrides)
    env = {k: v for k, v in env.items() if v is not None}
    proc = subprocess.run(["bash", str(REPO / "runpod" / "entrypoint.sh")], env=env, capture_output=True,
                          text=True, timeout=60)
    return proc.returncode, calls.read_text(), ws


def test_entrypoint_done_marker_exits(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_DONE="done")
    assert rc == 0
    assert "podTerminate" in calls
    assert not _trainer_ran(calls)


def test_no_rom_step_and_atari_gcs_root(tmp_path):
    rc, calls, _ = _run(tmp_path)
    assert rc == 0
    assert "roms" not in calls and "stable_retro" not in calls
    assert "gs://bucket/atari-rl/runs/t" in calls
    assert "atari-rl train game=breakout" in calls


def test_sync_failure_does_not_stop_training(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_RSYNC_FAIL="1")
    assert rc == 0
    assert "atari-rl train" in calls
    assert "podTerminate" in calls


def test_cost_exported_to_trainer(tmp_path):
    _, calls, _ = _run(tmp_path)
    assert "cost=0.5" in calls


def test_trainer_failure_still_syncs_and_terminates(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_TRAIN_RC="3")
    assert rc == 3
    lines = calls.splitlines()
    train_idx = next(i for i, line in enumerate(lines) if line.startswith("atari-rl "))
    after = "\n".join(lines[train_idx:])
    assert "rsync" in after and "podTerminate" in after


def test_local_run_without_pod_id_skips_terminate(tmp_path):
    rc, calls, _ = _run(tmp_path, RUNPOD_POD_ID="")
    assert rc == 0
    assert "podTerminate" not in calls


def test_missing_game_fails_fast(tmp_path):
    rc, calls, _ = _run(tmp_path, ATARIRL_GAME="")
    assert rc != 0
    assert not _trainer_ran(calls)


def test_missing_var_still_terminates(tmp_path):
    rc, calls, _ = _run(tmp_path, ATARIRL_GAME=None)
    assert rc != 0
    assert "podTerminate" in calls
    assert not _trainer_ran(calls)


def test_mlflow_retry_limits_exported(tmp_path):
    _, calls, _ = _run(tmp_path)
    assert "retries=2 timeout=20" in calls


def test_terminate_retries_until_success(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_TERMINATE_FAILS="2")
    assert rc == 0
    assert calls.count("podTerminate") == 3


def test_gsutil_wrapped_in_timeout(tmp_path):
    """A hung gsutil is killed instead of blocking the pod (and its billing) forever."""
    rc, calls, _ = _run(tmp_path, GCS_TIMEOUT="1", STUB_GSUTIL_SLEEP="100")
    assert rc == 0
    assert _trainer_ran(calls)


def test_done_marker_does_not_sync(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_DONE="done")
    assert rc == 0
    assert "rsync" not in calls  # never overwrite a finished run with an empty local dir


def test_token_writer_removed():
    assert not (REPO / "runpod" / "scripts" / "token_writer.sh").exists()


def test_budget_stopped_run_is_extended(tmp_path):
    rc, calls, _ = _run(tmp_path, STUB_DONE="budget")
    assert rc == 0
    assert any(" rm " in line and "DONE" in line for line in calls.splitlines())
    assert any(line.startswith("atari-rl ") for line in calls.splitlines())
