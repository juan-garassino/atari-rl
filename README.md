# atari-rl

PPO agents for Atari **Pong** and **Breakout** (`GAME=pong|breakout`), trained on RunPod with live
training-health monitoring that stops a broken run and removes the pod. Part of the RL series — one repo per
console, copied from `RL-sega-rl` (same trainer, health monitor, MLflow tracking and RunPod lifecycle).

Why Atari: it is the canonical PPO benchmark, so the numbers are comparable with published results and the
pipeline gets a cheap, well-understood sanity check before the harder Sega / Game Boy games.

## Stack

Python 3.12 · uv · `ale-py` (Pong/Breakout ROMs ship inside the wheel — no ROM import, nothing in git) ·
gymnasium ALE v5 (`frameskip=1`, sticky actions 0.25) · SB3 `AtariWrapper` (noop reset, frame skip 4 + max-pool,
episodic life, fire reset, 84×84 gray, reward clipping) · 4-frame stack · SB3 PPO `CnnPolicy` with rl-zoo
Atari defaults · MLflow (`garassino-mlflow`, experiment `atari-rl`).

## Setup

```bash
make install   # uv sync --extra dev (CPU torch)
make test      # full suite, niced (~3 min on the laptop)
make smoke     # 2-env Pong, 2k steps, niced; ~1 min
make help
```

## From-scratch PPO (`ATARIRL_ALGO=scratch`)

`src/atari_rl/scratch/` is our own PPO — plain PyTorch + numpy, no Stable-Baselines3 — written to be read
on camera: one responsibility per file, every formula commented. It honours the same contract as the SB3 path
(run dir, checkpoints, DONE, budget, `health.json`, MLflow metric names), so `status`, `videos` and `play` work
unchanged. SB3 stays the default until scratch matches it on a paid Pong probe.

```
connectors/   the only console-specific code (Atari now; Genesis / Game Boy later)
  base.py       Connector protocol + observation spec (dict of image / vector inputs)
  atari.py      ALE env, eval scoring, win definition      atari_wrappers.py  hand-written Atari preprocessing
  cartpole.py   sanity-check connector
nets.py       actor-critic built from the obs spec (NatureCNN / MLP per input, concatenated)
vec_env.py    subprocess envs + batched frame stacking
rollout.py    play n_steps in every env → buffer
buffer.py     rollout storage + GAE(λ) (terminated vs truncated)
ppo.py        clipped surrogate, value loss, entropy bonus, minibatch epochs
trainer.py    the run loop (checkpoints, eval, health, budget, MLflow)
cartpole.py   the whole PPO loop on one screen
```

```bash
make cartpole                                    # sanity check: solves CartPole-v1 (500/500) in ~30k steps
make smoke-scratch                               # 2-env Pong smoke through the full run contract
ATARIRL_ALGO=scratch uv run atari-rl train       # or --algo scratch
uv run atari-rl eval --model runs/<id>/best_model.zip --episodes 1 --video play.mp4   # SB3 or scratch zip
```

Scratch checkpoints are zips with `meta.json` + `model.pt` + `optimizer.pt` (same file names as SB3's, told apart
by `meta.json`). Hyper-parameters: `ATARIRL_PARAMS_FILE` with `PPOConfig` field names (e.g. `{"n_minibatches": 8}`).

## Train on RunPod

The image is built by CI only (`runpod/Dockerfile`, never locally). Launch via the **deploy-runpod** GitHub
workflow (`make launch GAME=pong` or `GAME=breakout`; extra inputs `ARGS="-f n_envs=32 -f max_eur=2"`).
The pod resumes from the newest checkpoint, trains until the step target, CRITICAL health, or the time/EUR
budget, then syncs and terminates itself. Details: [`runpod/README.md`](runpod/README.md).

| Game | Steps | EUR cap | Target (raw score, mean of eval games) |
|---|---|---|---|
| Pong | 3M | 1.5 | ≥ +18 |
| Breakout | 10M | 3.5 | ≥ 200 |

Targets are logged (`eval/target_reached`, `best.json`), never enforced.

## Health and results

- Every PPO update: KL, clip fraction, entropy collapse, explained variance, reward plateau, FPS drop / hang,
  NaNs, action spam → `health.json` HEALTHY / WARN / CRITICAL (CRITICAL stops the run).
- Every 250k steps: 5 deterministic **full games with unclipped rewards** → `eval/mean_score` is the real game
  score; best model by mean score; one replay video.
- `rollout/ep_rew_mean` is also the raw full-game score (Monitor sits under the Atari wrappers).
- `make status RUN=<run id>` prints live state from MLflow; videos, `best_model.zip`, `health.json` and the final
  checkpoint are MLflow artifacts of the run named `<run id>`.

## Run ledger

| Date | Run id | Game | Steps | Pod | Cost (RunPod, card) | Best eval score | Health | Notes |
|---|---|---|---|---|---|---|---|---|
