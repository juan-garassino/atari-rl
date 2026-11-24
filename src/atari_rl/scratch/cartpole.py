"""CartPole-v1 sanity run: the whole PPO loop on one screen, no run-contract plumbing.

    for each iteration:
        anneal learning rate (and clip range)
        collect n_steps × n_envs transitions with the current policy   (rollout.py)
        compute GAE advantages and returns                              (buffer.py)
        n_epochs × n_minibatches clipped-surrogate gradient steps       (ppo.py)

If this does not reach ~500 (the episode cap), the bug is in the core, not in Atari. `atari-rl cartpole`.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from atari_rl.scratch.buffer import RolloutBuffer
from atari_rl.scratch.connectors.cartpole import SOLVED, CartPoleConnector
from atari_rl.scratch.evaluate import run_eval
from atari_rl.scratch.nets import ActorCritic
from atari_rl.scratch.ppo import PPOConfig, linear_schedule, ppo_update, set_learning_rate
from atari_rl.scratch.rollout import RolloutCollector
from atari_rl.scratch.seeding import seed_everything
from atari_rl.scratch.vec_env import make_vec_env

logger = logging.getLogger(__name__)


@dataclass
class CartPoleResult:
    steps: int
    iterations: int
    seconds: float
    train_mean_return: float  # mean of the last 20 training episodes
    eval_mean_return: float  # greedy policy, eval_episodes full episodes
    solved_at_step: int | None  # first step where the last-20 training mean reached 475
    final_params_checksum: float


def train_cartpole(total_steps: int = 100_000, n_envs: int = 4, seed: int = 1, eval_episodes: int = 10,
                   stop_when_solved: bool = False) -> CartPoleResult:
    shuffle_rng = seed_everything(seed)
    device = torch.device("cpu")
    connector = CartPoleConnector()
    ppo = PPOConfig.from_sources(connector.ppo_overrides, None)
    venv = make_vec_env(connector, n_envs, in_process=True)  # CartPole steps in µs: pipes would cost more
    agent = ActorCritic(connector.observation_spec, connector.n_actions, share_encoder=connector.share_encoder)
    optimizer = torch.optim.Adam(agent.parameters(), lr=ppo.learning_rate, eps=1e-5)
    buffer = RolloutBuffer(ppo.n_steps, n_envs, connector.observation_spec, device)
    collector = RolloutCollector(venv, agent, buffer, device, seed=seed)

    recent: deque[float] = deque(maxlen=20)
    steps, iterations, solved_at = 0, 0, None
    start = time.time()
    n_iterations = total_steps // (n_envs * ppo.n_steps)
    for iterations in range(1, n_iterations + 1):
        progress = steps / total_steps
        set_learning_rate(optimizer, linear_schedule(ppo.learning_rate, progress) if ppo.anneal_lr else ppo.learning_rate)
        clip = linear_schedule(ppo.clip_range, progress) if ppo.anneal_clip else ppo.clip_range

        stats = collector.collect(ppo.gamma, ppo.gae_lambda)
        metrics = ppo_update(agent, optimizer, buffer, ppo, clip, shuffle_rng)
        steps += stats.steps
        recent.extend(float(e["r"]) for e in stats.episodes)

        mean_ret = float(np.mean(recent)) if recent else 0.0
        if solved_at is None and len(recent) == recent.maxlen and mean_ret >= SOLVED:
            solved_at = steps
        if iterations % 40 == 0:
            logger.info("step %d  return(last 20) %.1f  kl %.4f  clipfrac %.3f  ev %.3f  entropy %.3f", steps,
                        mean_ret, metrics["approx_kl"], metrics["clip_fraction"], metrics["explained_variance"],
                        metrics["entropy"])
        if stop_when_solved and solved_at is not None:
            break
    venv.close()
    seconds = time.time() - start
    ev = run_eval(agent, connector, eval_episodes, max_steps=500, seed=seed + 10_000)
    checksum = float(sum(p.detach().double().sum().item() for p in agent.parameters()))
    return CartPoleResult(steps, iterations, seconds, float(np.mean(recent)) if recent else 0.0, ev["mean_score"],
                          solved_at, checksum)
