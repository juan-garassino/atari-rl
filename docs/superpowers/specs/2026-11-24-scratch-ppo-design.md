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
- sega-rl (later, optional).

## Phase 2 — Pokémon (Juan: "for at least the pokemon and the atari")
After Pong proves the core: copy `scratch/` into gameboy-rl (workspace convention: copy, no shared package) and
add a multi-input actor-critic for PWhiddy v2's dict observation (screens 3×72×80 → CNN; health/level/badges/
events/map/recent_actions → MLP/embedding; concatenated trunk). Resume is not possible from the SB3 checkpoint
(different parameter layout) — the scratch Pokémon run starts fresh from `has_pokedex_nballs.state`.

## Correctness bar (before any paid run)
- Unit tests: GAE matches a hand-computed reference incl. episode boundaries and truncation bootstrap;
  clipped loss sign/shape; advantage normalization; value-clipping off by default; deterministic seeding.
- CartPole-v1 sanity (MLP head, local, < 2 min): reaches ≥ 450 mean return — proves the learning loop.
- Pong local smoke (2 envs, few k steps) runs through the full train.run contract (checkpoint, DONE, health.json).
- Paid probe: Pong, same cap as `pong-probe-2` (€0.50) → train score trend must be ≥ SB3's (−20.1 → −18.6 by ~490k).

## Defaults (CleanRL / SB3-zoo Atari PPO)
n_envs 16, n_steps 128, minibatches 4, epochs 4, lr 2.5e-4 linear, clip 0.1 linear, γ 0.99, λ 0.95,
ent 0.01, vf 0.5, max_grad_norm 0.5, orthogonal init (√2 trunk, 0.01 policy head, 1 value head), obs/255.

## Phase 3 — all connectors proven (Juan: "try with a bit of gameboy and sega as well ... to know the connectors work")
Home: `~/Code/005-products/033-rl-arcade` (core + `connectors/{atari,genesis,gameboy}`).
Each connector gets a short paid probe (~€0.50–1, CPU pod) with a video:
- atari: Pong — train score trend ≥ SB3 probe.
- genesis: Sonic Green Hill 1 (stall termination, 7 contest actions) — eval level progress rises (SB3 reached 54% by 369k);
  SF via the 21-move set once sega-rl's SF v2 shows learning.
- gameboy: Pokémon from has_pokedex_nballs.state, dict obs (screens CNN + vector MLP) — explore reward / visited map
  tiles rise; replay video via play.
