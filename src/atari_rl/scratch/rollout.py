"""Play the current policy in all envs for n_steps and fill the rollout buffer.

One loop iteration = one batched forward pass for all N envs + one parallel env step. No per-env Python work
except reading episode stats, so the main process stays cheap next to the emulators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from atari_rl.scratch.buffer import RolloutBuffer
from atari_rl.scratch.nets import ActorCritic, obs_to_tensor
from atari_rl.scratch.vec_env import stack_obs


@dataclass
class RolloutStats:
    steps: int  # env steps taken (n_steps · n_envs)
    episodes: list[dict[str, Any]] = field(default_factory=list)  # info["episode"] of every finished game
    action_counts: np.ndarray | None = None  # how often each action was picked (health: button spam)


class RolloutCollector:
    """Owns the "current observation" of every env between rollouts, so episodes continue across updates."""

    def __init__(self, venv, agent: ActorCritic, buffer: RolloutBuffer, device: torch.device, seed: int | None,
                 heartbeat: Callable[[], None] | None = None):
        self.venv, self.agent, self.buffer, self.device = venv, agent, buffer, device
        self.heartbeat = heartbeat or (lambda: None)
        self.obs = venv.reset(seed)
        self.n_actions = agent.policy_head.out_features

    @torch.no_grad()
    def collect(self, gamma: float, lam: float) -> RolloutStats:
        n_steps, n_envs = self.buffer.n_steps, self.buffer.n_envs
        stats = RolloutStats(steps=n_steps * n_envs, action_counts=np.zeros(self.n_actions, np.int64))
        for t in range(n_steps):
            actions, logprobs, values = self.agent.act(obs_to_tensor(self.obs, self.device))
            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = self.venv.step(actions_np)

            # Truncated (time limit, not a real ending): bootstrap from V(true final obs), see buffer.py.
            final_values = np.zeros(n_envs, np.float32)
            cut = np.flatnonzero(truncated & ~terminated)
            if cut.size:
                final_obs = stack_obs([infos[i]["final_obs"] for i in cut])
                final_values[cut] = self.agent.value(obs_to_tensor(final_obs, self.device)).cpu().numpy()

            self.buffer.add(t, self.obs, actions, logprobs, values, rewards, terminated, truncated, final_values)
            stats.episodes += [info["episode"] for info in infos if "episode" in info]
            stats.action_counts += np.bincount(actions_np, minlength=self.n_actions)
            self.obs = next_obs
            self.heartbeat()

        last_values = self.agent.value(obs_to_tensor(self.obs, self.device))
        self.buffer.finish(last_values, gamma, lam)
        return stats
