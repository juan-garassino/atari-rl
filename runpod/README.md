# RunPod runbook

The image is built by CI (`build-push.yml`) — never `docker build` on the laptop. It contains no ROM file:
ale-py ships the Pong and Breakout ROMs inside its wheel, so the pod has no ROM step at all.

## Launch

GitHub → Actions → **deploy-runpod** → inputs `game` (pong|breakout), `mode` (train|smoke), `total_steps`,
`n_envs` (≈ vCPUs), `max_hours`, `max_eur` (or `gh workflow run deploy-runpod.yml -f game=pong`).
The workflow mints a 1 h GCS token via WIF and creates the pod. GCS is only needed in the first minutes
(resume state); after that the token may expire — live metrics, health, eval videos, the best model and the final
checkpoint go through MLflow (`--serve-artifacts`, basic auth).

## What the pod does (`entrypoint.sh`)

1. Exits (and removes itself) if `gs://garassino-ml-artifacts/atari-rl/runs/<RUN_ID>/DONE` exists.
2. Pulls the run dir (resume from the newest checkpoint).
3. Reads its own USD/h (`pod_self.sh cost`) → `ATARIRL_COST_PER_HR_USD`; training stops at
   min(`max_hours`, `max_eur` worth of hours). Defaults: Pong 1.5 EUR, Breakout 3.5 EUR, 6 h.
4. Syncs the run dir to GCS every `SYNC_INTERVAL` (600 s) while the token lasts; failures never stop training.
5. Runs `atari-rl $MODE`; on any exit (incl. missing config): final sync, then `pod_self.sh remove`, retried 5×
   (billing stops). If even that fails, the log says so — remove the pod in the RunPod console.

## Watch a run

```bash
make status RUN=<id>       # from MLflow: health status + reasons, eval score, fps
```

Everything else (videos, best_model.zip, health.json, final checkpoint): MLflow experiment `atari-rl`,
run name = RUN_ID. Needs `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` in your shell (and as GitHub
secrets for the pod).
