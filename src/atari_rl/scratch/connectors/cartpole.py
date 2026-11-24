"""CartPole-v1: the sanity-check connector. If PPO can't hold the pole for ~500 steps, nothing else will work.

Not a console — it exists to prove the core learns (and that the core really is console-agnostic).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from atari_rl.scratch.connectors.base import DictObservation, ObsSpec, VectorSpec

SOLVED = 475.0  # gymnasium's reward_threshold for CartPole-v1 (max 500)


@dataclass(frozen=True)
class CartPoleConnector:
    name: str = "cartpole"
    n_actions: int = 2
    frame_stack: int = 1
    share_encoder: bool = False  # separate actor/critic MLPs, as in CleanRL's ppo.py
    # CleanRL ppo.py defaults where they differ from the Atari ones in PPOConfig.
    ppo_overrides: dict[str, Any] = field(default_factory=lambda: {"clip_range": 0.2, "anneal_clip": False})

    @property
    def observation_spec(self) -> ObsSpec:
        return {"state": VectorSpec(4)}

    def make_env(self, rank: int, *, eval: bool) -> gym.Env:
        env = gym.make("CartPole-v1", render_mode="rgb_array" if eval else None)
        return DictObservation(RecordEpisodeStatistics(env), "state")

    def eval_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        scores = np.array([e["score"] for e in episodes], dtype=np.float64)
        return {"mean_score": float(scores.mean()), "std_score": float(scores.std()), "max_score": float(scores.max()),
                "mean_length": float(np.mean([e["length"] for e in episodes])),
                "target_reached": float(scores.mean() >= SOLVED)}

    def best_key(self, metrics: dict[str, float]) -> tuple[float, ...]:
        return (metrics["mean_score"],)
