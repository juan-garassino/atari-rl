"""Rollout storage and Generalized Advantage Estimation (GAE-λ).

One PPO iteration collects `n_steps` transitions from each of `n_envs` envs into fixed (T, N, ...) tensors,
computes advantages once, then hands out shuffled minibatches for several epochs of updates.

Timing convention (read this before the math): row t holds
    obs_t, action_t, logprob_t, value_t = V(obs_t), reward_t, terminated_t, truncated_t
where reward/terminated/truncated are the *result* of taking action_t — they describe the same transition.

Terminated vs truncated matters:
- terminated: the game really ended (lost a life / game over / pole fell). The future is worth 0.
- truncated:  a time limit cut the episode short. The future is NOT worth 0, so we bootstrap from the value
              of the real final observation, V(final_obs_t), which the env hands back before auto-resetting.
Either way the next row belongs to a new episode, so the advantage must not carry across it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from atari_rl.scratch.connectors.base import ImageSpec, ObsSpec


def compute_gae(
    rewards: torch.Tensor,  # (T, N)
    values: torch.Tensor,  # (T, N)  V(obs_t)
    terminated: torch.Tensor,  # (T, N) bool
    truncated: torch.Tensor,  # (T, N) bool
    final_values: torch.Tensor,  # (T, N)  V(final_obs_t), only read where truncated
    last_values: torch.Tensor,  # (N,)  V(obs_T): the obs after the rollout, used to bootstrap the last row
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (advantages, returns), both (T, N). returns = advantages + values (the value-function target).

    δ_t = r_t + γ · V(next_t) − V(obs_t)           (TD error of one step)
    A_t = δ_t + γλ · A_{t+1}  (only if the episode continues after t)
    with V(next_t) = 0 if terminated, V(final_obs_t) if truncated, else V(obs_{t+1}).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(last_values)
    episode_ended = terminated | truncated
    for t in reversed(range(T)):
        next_value = last_values if t == T - 1 else values[t + 1]
        next_value = torch.where(truncated[t], final_values[t], next_value)
        next_value = torch.where(terminated[t], torch.zeros_like(next_value), next_value)
        delta = rewards[t] + gamma * next_value - values[t]
        continues = (~episode_ended[t]).float()
        next_advantage = delta + gamma * lam * continues * next_advantage
        advantages[t] = next_advantage
    return advantages, advantages + values


@dataclass
class Batch:
    """A flat slice of the rollout (B = minibatch size) — everything one PPO gradient step needs."""

    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    logprobs: torch.Tensor  # log π_old(a|s) at collection time
    values: torch.Tensor  # V_old(s) at collection time
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """Fixed-size (n_steps × n_envs) storage, reused every iteration. Image obs stay uint8 to save memory."""

    def __init__(self, n_steps: int, n_envs: int, obs_spec: ObsSpec, device: torch.device):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        T, N = n_steps, n_envs
        self.obs = {
            key: torch.zeros((T, N, *spec.shape), dtype=torch.uint8, device=device) if isinstance(spec, ImageSpec)
            else torch.zeros((T, N, spec.size), dtype=torch.float32, device=device)
            for key, spec in obs_spec.items()
        }
        self.actions = torch.zeros((T, N), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((T, N), device=device)
        self.values = torch.zeros((T, N), device=device)
        self.rewards = torch.zeros((T, N), device=device)
        self.terminated = torch.zeros((T, N), dtype=torch.bool, device=device)
        self.truncated = torch.zeros((T, N), dtype=torch.bool, device=device)
        self.final_values = torch.zeros((T, N), device=device)
        self.advantages = torch.zeros((T, N), device=device)
        self.returns = torch.zeros((T, N), device=device)

    def add(self, t: int, obs: dict[str, np.ndarray], actions: torch.Tensor, logprobs: torch.Tensor,
            values: torch.Tensor, rewards: np.ndarray, terminated: np.ndarray, truncated: np.ndarray,
            final_values: np.ndarray | torch.Tensor) -> None:
        for key, buf in self.obs.items():
            buf[t] = torch.as_tensor(obs[key], device=self.device)
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.values[t] = values
        self.rewards[t] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.terminated[t] = torch.as_tensor(terminated, device=self.device)
        self.truncated[t] = torch.as_tensor(truncated, device=self.device) & ~self.terminated[t]
        self.final_values[t] = torch.as_tensor(final_values, dtype=torch.float32, device=self.device)

    def finish(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        """Call once the rollout is full: fills advantages and returns."""
        self.advantages, self.returns = compute_gae(self.rewards, self.values, self.terminated, self.truncated,
                                                    self.final_values, last_values, gamma, lam)

    def minibatches(self, n_minibatches: int, generator: torch.Generator) -> Iterator[Batch]:
        """One epoch: a random permutation of all T·N samples, cut into n_minibatches equal slices."""
        size = self.n_steps * self.n_envs
        if size % n_minibatches:
            raise ValueError(f"n_steps*n_envs={size} is not divisible by n_minibatches={n_minibatches}")
        flat = lambda x: x.reshape(size, *x.shape[2:])  # noqa: E731  (T, N, ...) → (T·N, ...)
        obs = {k: flat(v) for k, v in self.obs.items()}
        fields = [flat(x) for x in (self.actions, self.logprobs, self.values, self.advantages, self.returns)]
        order = torch.randperm(size, generator=generator).to(self.device)
        mb_size = size // n_minibatches
        for start in range(0, size, mb_size):
            idx = order[start:start + mb_size]
            yield Batch({k: v[idx] for k, v in obs.items()}, *(f[idx] for f in fields))
