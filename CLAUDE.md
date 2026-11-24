# CLAUDE.md — atari-rl

PPO on Atari Pong / Breakout (ale-py), trained on RunPod. Sibling of `../RL-sega-rl` (the template): keep the
two repos' trainer, health monitor, MLflow tracking and runpod/ files in sync by copy-paste, not a shared package.

## Module map (`src/atari_rl/`)

| Module | Purpose |
|---|---|
| `config.py` | `Settings` (`ATARIRL_` prefix); per-game defaults: steps (pong 3M / breakout 10M), `max_eur` (1.5 / 3.5), `TARGET_SCORE` (18 / 200); `hours_budget` = min(max_hours, EUR cap ÷ pod USD/h) |
| `envs/atari.py` | `make(cfg, eval_mode)`: ALE v5 → `Monitor` → SB3 `AtariWrapper` → `ActionInfo`. Eval: unclipped, full games, `FireOnLifeLoss` |
| `envs/factory.py` | `make_single` (one env thunk) and `make_vec_env` (SubprocVecEnv + VecFrameStack(4)) |
| `monitoring/health.py` | `HealthMonitor` rules + `HealthCallback` (writes `health.json`, stall watchdog, stops on CRITICAL) — verbatim from sega-rl |
| `monitoring/mlflow_tracking.py` | No-op unless `MLFLOW_TRACKING_URI`; inactive until `start()` succeeds; `MLflowOutputFormat` streams SB3 metrics |
| `monitoring/video.py` | mp4 writer for eval replays |
| `evaluate.py` | `run_eval` (raw score), `EvalCallback` (best model by mean score, uploads video + best.json) |
| `train.py` | `run()`: build/resume PPO, `CheckpointKeeper`, `BudgetCallback`, `HealthReporter`, env-crash → CRITICAL, `DONE` marker |
| `status.py` | `atari-rl status <run>`: live run state from MLflow |
| `cli.py` | argparse: `train`, `eval`, `smoke`, `status` |

`runpod/`: `Dockerfile` (CUDA, uv, non-root, CI-built only), `entrypoint.sh` (DONE check → resume pull → pod cost →
periodic GCS sync → `atari-rl $MODE` → final sync → pod self-terminate), `scripts/pod_self.sh`, `scripts/token_writer.sh`.

## Env vars

`ATARIRL_GAME` (pong|breakout), `ATARIRL_N_ENVS`, `ATARIRL_TOTAL_STEPS`, `ATARIRL_MAX_HOURS`, `ATARIRL_MAX_EUR`,
`ATARIRL_COST_PER_HR_USD` (set by the entrypoint), `ATARIRL_EVAL_FREQ`, `ATARIRL_EVAL_MAX_STEPS`,
`ATARIRL_STICKY_PROB`, `ATARIRL_PARAMS_FILE` (JSON of PPO kwargs); every field is documented inline in
`config.py`. Pod: `RUN_ID`, `GCS_BUCKET` (root `gs://$GCS_BUCKET/atari-rl`), `GCS_ACCESS_TOKEN`, `MLFLOW_TRACKING_URI`
(+ `MLFLOW_TRACKING_USERNAME/PASSWORD`), `RUNPOD_API_KEY`; RunPod injects `RUNPOD_POD_ID`.

## Gotchas

- **Training reward ≠ score.** Training rewards are clipped to ±1 and an episode is one life; eval and
  `rollout/ep_rew_mean` are raw full-game scores. Compare runs on `eval/mean_score`.
- **Breakout stalls without FIRE.** Training relaunches the ball via `FireResetEnv`, so the agent never learns FIRE;
  eval wraps `FireOnLifeLoss`. Eval games are also capped at `eval_max_steps` (10k agent steps).
- No VecMonitor: per-env `Monitor` sits under `AtariWrapper` (SB3 `make_atari_env` convention); adding VecMonitor
  would overwrite its stats with clipped per-life ones.
- `info["buttons"]` is the one-hot action (from `ActionInfo`) so the shared button-spam health rule works.
- The training process hosts no emulator (all envs are SubprocVecEnv workers); eval runs one in-process env.
- Local machine: one heavy job at a time, niced, ≤ 2 envs. Never `docker build` locally.
- `.github/workflows` (ci, build-push, deploy-runpod) are copied from sega-rl separately.
