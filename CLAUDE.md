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
| `cli.py` | argparse: `train` / `smoke` (dispatch on `ATARIRL_ALGO`), `eval` (loads SB3 or scratch zips), `cartpole`, `status`, `videos` |

### From-scratch PPO (`src/atari_rl/scratch/`, `ATARIRL_ALGO=scratch`)

Plain PyTorch + numpy, no SB3 import anywhere in `scratch/` (a test enforces it). One **console-agnostic core**
plus one **connector** per console; the core never imports a console module (also test-enforced), so Genesis
(stable-retro) / Game Boy (PyBoy) only add a file under `connectors/`. Design: `docs/superpowers/specs/2026-11-24-scratch-ppo-design.md`.

| Module | Purpose |
|---|---|
| `connectors/base.py` | `Connector` protocol (`name`, `observation_spec`, `n_actions`, `frame_stack`, `share_encoder`, `ppo_overrides`, `make_env(rank, *, eval)`, `eval_metrics(episodes)`, `best_key(metrics)`); `ImageSpec` / `VectorSpec` (obs are always dicts); `DictObservation` |
| `connectors/atari.py` | `AtariConnector` (pong/breakout): ALE v5 + wrappers, eval = unclipped full games + `FireOnLifeLoss`; metrics incl. `win_rate` (Pong: outscored the CPU; Breakout: ≥ target) |
| `connectors/atari_wrappers.py` | Hand-written NoopReset, MaxAndSkip, EpisodicLife, FireReset, WarpFrame(84 gray), ClipReward, FireOnLifeLoss — frame-identical to SB3's `AtariWrapper` (tested) |
| `connectors/cartpole.py` | `CartPoleConnector`: sanity check only, separate actor/critic MLPs |
| `connectors/__init__.py` | `connector_for(cfg)`: the only place that picks a console (called by the CLI) |
| `nets.py` | `ActorCritic` built from the obs spec: NatureCNN per image key, MLP per vector key, concatenated; orthogonal init (√2 / 0.01 policy / 1 value) |
| `vec_env.py` | `SubprocVecEnv` (forkserver, auto-reset in worker, `info["final_obs"]`), `InProcessVecEnv`, `FrameStack` (batched, zero-filled on reset), `make_vec_env(connector, n)` |
| `buffer.py` | `compute_gae` (terminated → 0, truncated → bootstrap V(final obs)) + `RolloutBuffer` (dict obs, uint8 images) |
| `rollout.py` | `RolloutCollector`: n_steps batched act/step, truncation bootstrap values, episode stats, action counts |
| `ppo.py` | `PPOConfig` (design defaults; overrides: connector ← `ATARIRL_PARAMS_FILE`, unknown keys rejected), clipped surrogate, value loss (clip off), `ppo_update` → SB3-named metrics |
| `checkpoint.py` | zip = `meta.json` + `model.pt` + `optimizer.pt`; same file names as SB3 (`ckpt_*.zip`, `best_model.zip`); `is_scratch_checkpoint`; `weights_only` loads |
| `evaluate.py` | `run_eval(agent, connector, ...)`: greedy full games, first one filmed → `connector.eval_metrics` |
| `health_guard.py` | Feeds the unchanged `HealthMonitor`, writes `health.json` atomically, stall watchdog thread |
| `metrics.py` | `MetricsWriter`: log line + TensorBoard + MLflow, SB3 metric names |
| `trainer.py` | `run(cfg, connector, env_factory)`: same contract as `train.run` (resume, ckpt/eval cadence, budget, DONE, MLflow); reuses `HealthReporter`, `_stall_handler`, `torch_threads`, `ENV_CRASH_ERRORS` from `train.py` |
| `cartpole.py` | `train_cartpole()`: the bare PPO loop on one screen (`atari-rl cartpole`) |
| `seeding.py` | `seed_everything(seed)` → generator for minibatch shuffling |

`runpod/`: `Dockerfile` (CUDA, uv, non-root, CI-built only), `entrypoint.sh` (DONE check → resume pull → pod cost →
periodic GCS sync → `atari-rl $MODE` → final sync → pod self-terminate; the EXIT trap is set before config checks), `scripts/pod_self.sh` (cost / self-terminate, retried 5× and checked).

## Env vars

`ATARIRL_GAME` (pong|breakout), `ATARIRL_N_ENVS`, `ATARIRL_TOTAL_STEPS`, `ATARIRL_MAX_HOURS`, `ATARIRL_MAX_EUR`,
`ATARIRL_COST_PER_HR_USD` (set by the entrypoint), `ATARIRL_EVAL_FREQ`, `ATARIRL_EVAL_MAX_STEPS`,
`ATARIRL_STICKY_PROB`, `ATARIRL_PARAMS_FILE` (JSON of PPO kwargs; scratch keys = `PPOConfig` fields, e.g. `n_minibatches`
not `batch_size`), `ATARIRL_ALGO` (sb3|scratch, default sb3); every field is documented inline in
`config.py`. Pod: `RUN_ID`, `GCS_BUCKET` (root `gs://$GCS_BUCKET/atari-rl`), `GCS_ACCESS_TOKEN`, `MLFLOW_TRACKING_URI`
(+ `MLFLOW_TRACKING_USERNAME/PASSWORD`), `RUNPOD_API_KEY`; RunPod injects `RUNPOD_POD_ID`.

## Gotchas

- **Training reward ≠ score.** Training rewards are clipped to ±1 and an episode is one life; eval and
  `rollout/ep_rew_mean` are raw full-game scores. Compare runs on `eval/mean_score`.
- **Breakout stalls without FIRE.** Training relaunches the ball via `FireResetEnv`, so the agent never learns FIRE;
  eval wraps `FireOnLifeLoss`. Eval games are also capped at `eval_max_steps` (10k agent steps).
- No VecMonitor: per-env `Monitor` sits under `AtariWrapper` (SB3 `make_atari_env` convention); adding VecMonitor
  would overwrite its stats with clipped per-life ones.
- GCS token on the pod lasts 1 h and is not refreshed; anything that must survive goes to MLflow (experiment
  `mlflow-artifacts:/atari-rl`; `train.run` refuses to start, DONE `mlflow_unavailable`, if MLflow is configured but
  the upload probe fails). Checkpoints and best_model.zip upload as they are written.
- Every exit path writes `DONE` (`done|critical|budget|stalled|mlflow_unavailable`) and closes the MLflow run.
- Eval calls `HealthCallback.heartbeat()` per step so long evals don't trip the 5-min stall watchdog.
- Torch is pinned to 2.14.1 so the CPU lock and the image's cu126 wheel are the same version.
- `info["buttons"]` is the one-hot action (from `ActionInfo`) so the shared button-spam health rule works.
- The training process hosts no emulator (all envs are SubprocVecEnv workers); eval runs one in-process env.
- Local machine: one heavy job at a time, niced, ≤ 2 envs. Never `docker build` locally.
- `.github/workflows` (ci, build-push, deploy-runpod) are copied from sega-rl separately.
- **Scratch vs SB3 checkpoints share file names** but not formats: a scratch run refuses to resume from an SB3
  `ckpt_*.zip` (CRITICAL `crash`) — switch algo only with a new run id. `atari-rl eval` auto-detects the format
  (scratch zips have `meta.json`) and, for scratch, takes the game from the checkpoint when `--game` is omitted.
- Scratch clip range anneals linearly (design default); the SB3 path keeps a constant 0.1. Scratch minibatch =
  n_envs·n_steps/4 (512 at 16 envs) vs SB3 256.
- Scripts that start scratch subprocess envs need an `if __name__ == "__main__":` guard (forkserver re-imports
  `__main__` in every worker; without it each worker re-runs the script).
- `deploy-runpod.yml` has no `algo` input yet: a scratch pod run needs `ATARIRL_ALGO=scratch` added to the pod env.
