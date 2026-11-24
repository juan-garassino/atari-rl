# From-scratch PPO — design

Date: 2026-11-24 · Status: approved (Juan: "start the from scratch ppo"; scope confirmed: own PPO + CNN +
vectorized runner; keep emulators and gymnasium's env API; Pong first)

## Why
Full control and readability for the RL series. Stable-Baselines3 hides the algorithm; Juan wants every line
of the learning loop to be ours, explainable on video, and tunable (CNN, losses, schedules, throughput).

## Goals
- `src/atari_rl/scratch/`: PPO (rollout buffer, GAE(λ), clipped surrogate, value loss, entropy bonus,
  minibatch epochs, grad-norm clip, LR/clip annealing), Nature-CNN actor-critic, subprocess vector env with
  frame stacking — plain PyTorch + numpy, one responsibility per file, no SB3 import anywhere in `scratch/`.
- Same outside contract as the SB3 path: `ATARIRL_ALGO=scratch|sb3` (default stays `sb3` until scratch is proven),
  same Settings, run dir layout, checkpoints, DONE marker, budget cap, health.json, MLflow metrics names,
  eval videos + best model, `atari-rl status/videos/play` all keep working.
- Health monitor reused unchanged (`HealthMonitor.update(HealthSnapshot)`), fed directly by the scratch
  trainer (approx_kl, clip_fraction, entropy, explained_variance, ep_rew_mean, fps, action histogram, NaN check, watchdog heartbeat).
- Throughput at least SB3's on the same pod (SB3 Pong probe: ~300 fps on 32 vCPU).

## Non-goals
- Rewriting emulators; dropping gymnasium; other algorithms (DQN etc.); GPU-specific tricks.
- Porting to sega-rl / gameboy-rl (next step, after Pong proves it).

## Correctness bar (before any paid run)
- Unit tests: GAE matches a hand-computed reference incl. episode boundaries and truncation bootstrap;
  clipped loss sign/shape; advantage normalization; value-clipping off by default; deterministic seeding.
- CartPole-v1 sanity (MLP head, local, < 2 min): reaches ≥ 450 mean return — proves the learning loop.
- Pong local smoke (2 envs, few k steps) runs through the full train.run contract (checkpoint, DONE, health.json).
- Paid probe: Pong, same cap as `pong-probe-2` (€0.50) → train score trend must be ≥ SB3's (−20.1 → −18.6 by ~490k).

## Defaults (CleanRL / SB3-zoo Atari PPO)
n_envs 16, n_steps 128, minibatches 4, epochs 4, lr 2.5e-4 linear, clip 0.1 linear, γ 0.99, λ 0.95,
ent 0.01, vf 0.5, max_grad_norm 0.5, orthogonal init (√2 trunk, 0.01 policy head, 1 value head), obs/255.
