"""Atari connector + our own preprocessing wrappers (src/atari_rl/scratch/connectors/atari*.py).

The strongest check: with the same seed and actions, our training env produces exactly the frames and
rewards of SB3's AtariWrapper stack used by the SB3 path (tests may import SB3; scratch/ may not).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from atari_rl.config import Settings
from atari_rl.scratch.connectors import connector_for
from atari_rl.scratch.connectors.atari import AtariConnector
from atari_rl.scratch.connectors.atari_wrappers import MaxAndSkip
from atari_rl.scratch.connectors.base import ImageSpec


@pytest.mark.parametrize("game", ["pong", "breakout"])
def test_matches_sb3_preprocessing_exactly(game):
    from atari_rl.envs.atari import make as make_sb3

    cfg = Settings(game=game)
    ours = AtariConnector.from_settings(cfg).make_env(0, eval=False)
    theirs = make_sb3(cfg)
    o1, _ = ours.reset(seed=11)
    o2, _ = theirs.reset(seed=11)
    assert np.array_equal(o1["pixels"], o2[:, :, 0])
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = int(rng.integers(ours.action_space.n))
        o1, r1, t1, tr1, _ = ours.step(a)
        o2, r2, t2, tr2, _ = theirs.step(a)
        assert np.array_equal(o1["pixels"], o2[:, :, 0]) and r1 == r2 and t1 == t2 and tr1 == tr2
        if t1 or tr1:
            o1, _ = ours.reset()
            o2, _ = theirs.reset()
    ours.close()
    theirs.close()


def test_spec_and_actions_from_settings():
    c = AtariConnector.from_settings(Settings(game="breakout", frame_stack=4))
    assert c.observation_spec == {"pixels": ImageSpec((4, 84, 84))}
    assert c.n_actions == 4 and c.frame_stack == 4 and c.share_encoder
    for game in ("pong", "breakout"):
        env = AtariConnector(game=game).make_env(0, eval=True)
        assert env.action_space.n == AtariConnector(game=game).n_actions
        env.close()


def test_connector_for_settings():
    assert isinstance(connector_for(Settings(game="pong")), AtariConnector)


def test_training_env_life_is_an_episode_but_score_is_the_full_game():
    env = AtariConnector(game="breakout").make_env(0, eval=False)
    env.reset(seed=0)
    terminations, game_over_info = 0, None
    for _ in range(5000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert reward in (-1.0, 0.0, 1.0)  # clipped
        if terminated or truncated:
            terminations += 1
            if "episode" in info:
                game_over_info = info["episode"]
                break
            env.reset()
    env.close()
    assert terminations == 5  # one per life
    assert game_over_info is not None and game_over_info["l"] > 0


def test_eval_env_plays_full_unclipped_game():
    env = AtariConnector(game="pong").make_env(0, eval=True)
    obs, _ = env.reset(seed=0)
    total, frame = 0.0, env.render()
    for _ in range(10_000):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            break
    env.close()
    assert terminated and -21 <= total <= -15  # a random agent loses the whole game
    assert info["episode"]["r"] == total
    assert frame.shape == (210, 160, 3)


class _Frames(gym.Env):
    """Emits frames 1, 2, 3, ... and reward 1 per step."""

    observation_space = gym.spaces.Box(0, 255, (1,), np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.array([0], np.uint8), {}

    def step(self, action):
        self.t += 1
        return np.array([self.t], np.uint8), 1.0, False, False, {}


def test_max_and_skip_repeats_action_and_max_pools_last_two_frames():
    env = MaxAndSkip(_Frames(), skip=4)
    env.reset()
    obs, reward, *_ = env.step(0)
    assert reward == 4.0 and obs.tolist() == [4]
    obs, reward, *_ = env.step(0)
    assert obs.tolist() == [8]


def test_eval_metrics_and_win_definition():
    pong = AtariConnector(game="pong")
    eps = [{"score": 3.0, "length": 900}, {"score": -2.0, "length": 800}, {"score": 21.0, "length": 700}]
    m = pong.eval_metrics(eps)
    assert m["mean_score"] == pytest.approx(22 / 3) and m["max_score"] == 21.0 and m["mean_length"] == 800.0
    assert m["win_rate"] == pytest.approx(2 / 3)  # Pong: a game is won when you outscore the CPU
    assert m["target_reached"] == 0.0  # mean 7.3 < 18
    assert pong.best_key(m) == (m["mean_score"],)
    breakout = AtariConnector(game="breakout")
    assert breakout.eval_metrics([{"score": 250.0, "length": 1}])["win_rate"] == 1.0  # ≥ target 200
