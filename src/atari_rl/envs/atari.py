"""Pong / Breakout on ale-py (ROMs bundled) with SB3's standard Atari preprocessing."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

from atari_rl.config import Settings

GAME_IDS = {"pong": "ALE/Pong-v5", "breakout": "ALE/Breakout-v5"}
FIRE = 1  # ALE action index of FIRE in both games' minimal action sets


class ActionInfo(gym.Wrapper):
    """Adds info["buttons"] = one-hot of the Discrete action, so HealthCallback's button-spam rule works unchanged."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        buttons = np.zeros(self.action_space.n, np.int8)
        buttons[int(action)] = 1
        info["buttons"] = buttons
        return obs, reward, terminated, truncated, info


class FireOnLifeLoss(gym.Wrapper):
    """Eval only: presses FIRE once after a lost life.

    Training ends the episode on a lost life and FireResetEnv relaunches the ball, so the agent never learns
    to press FIRE itself; in a full eval game Breakout would otherwise stall after the first life.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get("lives", self.lives)
        if lives < self.lives and not (terminated or truncated):
            obs, extra, terminated, truncated, info = self.env.step(FIRE)
            reward += extra
        self.lives = info.get("lives", lives)
        return obs, reward, terminated, truncated, info


def make_ale(game: str, sticky_prob: float) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)
    return gym.make(GAME_IDS[game], frameskip=1, repeat_action_probability=sticky_prob, render_mode="rgb_array")


def make(cfg: Settings, eval_mode: bool = False) -> gym.Env:
    """Training: clipped rewards, episode per life. Eval: raw game score, full games (all lives)."""
    # Monitor under AtariWrapper records raw full-game scores (rollout/ep_rew_mean = game score).
    env: gym.Env = Monitor(make_ale(cfg.game, cfg.sticky_prob))
    if eval_mode:
        env = FireOnLifeLoss(AtariWrapper(env, frame_skip=cfg.frame_skip, terminal_on_life_loss=False,
                                          clip_reward=False))
    else:
        env = AtariWrapper(env, frame_skip=cfg.frame_skip)
    return ActionInfo(env)
