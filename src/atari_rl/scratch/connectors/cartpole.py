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
    # rl-zoo's tuned PPO CartPole-v1 config (8 envs × 32 steps there; 4 × 64 here = the same 256-sample batch).
    # CleanRL's untuned defaults (lr 2.5e-4, ent 0.01) also learn — the curve matched SB3 with the same settings —
    # but plateau noisily around 300-400 training return; this config solves CartPole in ~100k steps.
    ppo_overrides: dict[str, Any] = field(default_factory=lambda: {
        "n_steps": 64, "n_minibatches": 1, "n_epochs": 20, "learning_rate": 1e-3, "clip_range": 0.2,
        "ent_coef": 0.0, "gamma": 0.98, "gae_lambda": 0.8})

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
