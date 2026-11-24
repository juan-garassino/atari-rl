"""Env contract per game, on ale-py's bundled ROMs (no ROM import anywhere)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv
from stable_baselines3.common.monitor import Monitor

from atari_rl.config import Settings
from atari_rl.envs.atari import FIRE, ActionInfo, FireOnLifeLoss, make

N_ACTIONS = {"pong": 6, "breakout": 4}


def _chain(env: gym.Env) -> list[type]:
    types = []
    while True:
        types.append(type(env))
        if not hasattr(env, "env"):
            return types
        env = env.env


@pytest.mark.parametrize("game", ["pong", "breakout"])
def test_train_step_contract(game):
    env = make(Settings(game=game))
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (84, 84, 1) and obs.dtype == np.uint8
        assert env.action_space == Discrete(N_ACTIONS[game])
        for _ in range(20):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            assert reward in (-1.0, 0.0, 1.0)  # clipped for training
            assert info["buttons"].shape == (N_ACTIONS[game],) and info["buttons"].sum() == 1
    finally:
        env.close()


def test_train_wrapper_chain_monitor_below_clipping():
    """Monitor sits under AtariWrapper so rollout/ep_rew_mean is the raw full-game score."""
    chain = _chain(make(Settings(game="breakout")))
    assert chain.index(ClipRewardEnv) < chain.index(EpisodicLifeEnv) < chain.index(Monitor)


def test_eval_wrapper_chain_unclipped_full_games():
    chain = _chain(make(Settings(game="breakout"), eval_mode=True))
    assert ClipRewardEnv not in chain and EpisodicLifeEnv not in chain
    assert FireOnLifeLoss in chain and chain[0] is ActionInfo


def test_breakout_train_episode_is_one_life_monitor_is_full_game():
    env = make(Settings(game="breakout"))
    try:
        env.reset(seed=0)
        lives_lost, info = 0, {}
        for _ in range(20_000):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                lives_lost += 1
                if "episode" in info:
                    break
                env.reset()
        assert "episode" in info and lives_lost == 5
    finally:
        env.close()


def test_eval_render_is_rgb_frame():
    env = make(Settings(game="pong"), eval_mode=True)
    try:
        env.reset(seed=0)
        frame = env.render()
        assert frame.shape == (210, 160, 3) and frame.dtype == np.uint8
    finally:
        env.close()


class _StubLives(gym.Env):
    """Loses a life on step 2; records actions."""

    def __init__(self):
        self.observation_space = Box(0, 255, (84, 84, 1), np.uint8)
        self.action_space = Discrete(4)
        self.actions: list[int] = []

    def reset(self, *, seed=None, options=None):
        self.actions = []
        return np.zeros((84, 84, 1), np.uint8), {"lives": 5}

    def step(self, action):
        self.actions.append(int(action))
        lives = 4 if len(self.actions) >= 2 else 5
        return np.zeros((84, 84, 1), np.uint8), 1.0, False, False, {"lives": lives}


def test_fire_on_life_loss_presses_fire_once():
    stub = _StubLives()
    env = FireOnLifeLoss(stub)
    env.reset()
    _, r1, *_ = env.step(0)
    _, r2, *_ = env.step(0)
    _, r3, *_ = env.step(0)
    assert stub.actions == [0, 0, FIRE, 0]
    assert (r1, r2, r3) == (1.0, 2.0, 1.0)  # the FIRE step's reward is kept


def test_action_info_one_hot():
    env = ActionInfo(_StubLives())
    env.reset()
    *_, info = env.step(3)
    assert info["buttons"].tolist() == [0, 0, 0, 1]
