"""Atari 2600 (ALE) connector: Pong and Breakout.

Everything the PPO core must not know about Atari lives here: ALE env ids, the preprocessing stack
(atari_wrappers.py), reward clipping for training, full unclipped games for eval, and what a "win" is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics

from atari_rl.config import TARGET_SCORE, Settings
from atari_rl.scratch.connectors.atari_wrappers import (
    ClipReward,
    EpisodicLife,
    FireOnLifeLoss,
    FireReset,
    MaxAndSkip,
    NoopReset,
    WarpFrame,
)
from atari_rl.scratch.connectors.base import DictObservation, ImageSpec, ObsSpec

GAME_IDS = {"pong": "ALE/Pong-v5", "breakout": "ALE/Breakout-v5"}
N_ACTIONS = {"pong": 6, "breakout": 4}  # ALE minimal action sets (checked against ALE in the tests)
SCREEN = 84


def _won(game: str, score: float) -> bool:
    """Pong: you scored more points than the CPU. Breakout (no opponent): reached the target score."""
    return score > 0 if game == "pong" else score >= TARGET_SCORE[game]


@dataclass(frozen=True)
class AtariConnector:
    game: str
    frame_skip: int = 4
    frame_stack: int = 4
    sticky_prob: float = 0.25  # ALE v5 repeat_action_probability (Machado et al. 2018)
    noop_max: int = 30
    name: str = "atari"
    share_encoder: bool = True
    ppo_overrides: dict[str, Any] = field(default_factory=dict)  # PPOConfig defaults are the Atari ones

    @classmethod
    def from_settings(cls, cfg: Settings) -> AtariConnector:
        return cls(game=cfg.game, frame_skip=cfg.frame_skip, frame_stack=cfg.frame_stack,
                   sticky_prob=cfg.sticky_prob)

    @property
    def n_actions(self) -> int:
        return N_ACTIONS[self.game]

    @property
    def observation_spec(self) -> ObsSpec:
        return {"pixels": ImageSpec((self.frame_stack, SCREEN, SCREEN))}

    def make_env(self, rank: int, *, eval: bool) -> gym.Env:
        import ale_py

        gym.register_envs(ale_py)
        env = gym.make(GAME_IDS[self.game], frameskip=1, repeat_action_probability=self.sticky_prob,
                       render_mode="rgb_array")
        env = RecordEpisodeStatistics(env)  # raw full-game score, below every wrapper that changes rewards/episodes
        env = NoopReset(env, self.noop_max)
        env = MaxAndSkip(env, self.frame_skip)
        if not eval:
            env = EpisodicLife(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireReset(env)
        env = WarpFrame(env, SCREEN)
        env = FireOnLifeLoss(env) if eval else ClipReward(env)
        return DictObservation(env, "pixels")

    def eval_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        scores = np.array([e["score"] for e in episodes], dtype=np.float64)
        mean = float(scores.mean())
        return {
            "mean_score": mean,  # raw game score: unclipped rewards over a full game (all lives)
            "std_score": float(scores.std()),
            "max_score": float(scores.max()),
            "mean_length": float(np.mean([e["length"] for e in episodes])),  # agent steps per game
            "target_reached": float(mean >= TARGET_SCORE[self.game]),
            "win_rate": float(np.mean([_won(self.game, s) for s in scores])),
            "target": TARGET_SCORE[self.game],  # constant; kept so best.json says what "solved" means
        }

    def best_key(self, metrics: dict[str, float]) -> tuple[float, ...]:
        return (metrics["mean_score"],)

    @property
    def target_score(self) -> float:
        return TARGET_SCORE[self.game]
