from __future__ import annotations

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecFrameStack

from atari_rl.config import Settings


def make_single(cfg: Settings, rank: int, eval_mode: bool = False) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        from atari_rl.envs import atari

        env = atari.make(cfg, eval_mode=eval_mode)
        env.reset(seed=cfg.seed + rank)
        return env

    return _init


def make_vec_env(cfg: Settings, n_envs: int | None = None) -> VecEnv:
    """SubprocVecEnv + 4-frame stack. Episode stats come from each env's Monitor (raw score), not VecMonitor."""
    n = n_envs or cfg.n_envs
    venv = SubprocVecEnv([make_single(cfg, i) for i in range(n)], start_method="forkserver")
    return VecFrameStack(venv, cfg.frame_stack, channels_order="last")
