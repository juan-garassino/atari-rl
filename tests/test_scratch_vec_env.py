"""Vector envs + frame stacking (src/atari_rl/scratch/vec_env.py).

Toy envs run in-process (InProcessVecEnv); the subprocess runner is exercised with the CartPole connector,
which lives in the package so forkserver workers can import it.
"""
from __future__ import annotations

from functools import partial

import gymnasium as gym
import numpy as np
import pytest

from atari_rl.scratch.connectors.cartpole import CartPoleConnector
from atari_rl.scratch.vec_env import FrameStack, InProcessVecEnv, SubprocVecEnv, make_vec_env


class _CounterEnv(gym.Env):
    """Frame value = step index within the episode (+1); episode lasts `length` steps, then terminates
    (or truncates when truncate=True). Lets every stacked frame be checked by value."""

    def __init__(self, length=3, truncate=False):
        self.length, self.truncate = length, truncate
        self.observation_space = gym.spaces.Dict({"pixels": gym.spaces.Box(0, 255, (2, 2), np.uint8)})
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0

    def _obs(self):
        return {"pixels": np.full((2, 2), self.t + 1, np.uint8)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        end = self.t >= self.length
        info = {"episode": {"r": float(self.t), "l": self.t}} if end else {}
        return self._obs(), 1.0, end and not self.truncate, end and self.truncate, info


def _frames(obs, env=0):
    return obs["pixels"][env, :, 0, 0].tolist()


def test_frame_stack_zero_fills_then_rolls():
    venv = FrameStack(InProcessVecEnv([lambda: _CounterEnv(length=10)]), n_stack=3, image_keys=["pixels"])
    obs = venv.reset(seed=0)
    assert obs["pixels"].shape == (1, 3, 2, 2)
    assert _frames(obs) == [0, 0, 1]
    obs, *_ = venv.step(np.array([0]))
    assert _frames(obs) == [0, 1, 2]
    obs, *_ = venv.step(np.array([0]))
    assert _frames(obs) == [1, 2, 3]


def test_auto_reset_returns_final_obs_and_clears_stack():
    venv = FrameStack(InProcessVecEnv([lambda: _CounterEnv(length=2)]), n_stack=3, image_keys=["pixels"])
    venv.reset(seed=0)
    venv.step(np.array([0]))
    obs, rewards, terminated, truncated, infos = venv.step(np.array([0]))
    assert terminated.tolist() == [True] and truncated.tolist() == [False]
    assert infos[0]["final_obs"]["pixels"][:, 0, 0].tolist() == [1, 2, 3]  # the true last stacked state
    assert _frames(obs) == [0, 0, 1]  # first obs of the next episode, old frames wiped
    assert infos[0]["episode"]["r"] == 2.0


def test_truncation_flag_passes_through():
    venv = InProcessVecEnv([lambda: _CounterEnv(length=1, truncate=True)])
    venv.reset(seed=0)
    _, _, terminated, truncated, infos = venv.step(np.array([1]))
    assert truncated.tolist() == [True] and terminated.tolist() == [False]
    assert "final_obs" in infos[0]


def test_step_does_not_mutate_previous_obs():
    venv = FrameStack(InProcessVecEnv([lambda: _CounterEnv(length=10)]), n_stack=2, image_keys=["pixels"])
    first = venv.reset(seed=0)
    venv.step(np.array([0]))
    assert _frames(first) == [0, 1]


def test_subprocess_runner_matches_in_process_with_same_seeds():
    c = CartPoleConnector()
    sub = make_vec_env(c, n_envs=2, seed=3)
    local = make_vec_env(c, n_envs=2, seed=3, in_process=True)
    try:
        a, b = sub.reset(seed=3), local.reset(seed=3)
        np.testing.assert_array_equal(a["state"], b["state"])
        for _ in range(30):
            actions = np.array([0, 1])
            ra, rb = sub.step(actions), local.step(actions)
            np.testing.assert_array_equal(ra[0]["state"], rb[0]["state"])
            assert ra[2].tolist() == rb[2].tolist()
    finally:
        sub.close()
        local.close()


def test_dead_worker_surfaces_as_env_crash_error():
    c = CartPoleConnector()
    venv = SubprocVecEnv([partial(c.make_env, 0, eval=False), partial(c.make_env, 1, eval=False)])
    try:
        venv.reset(seed=0)
        venv.processes[1].kill()
        venv.processes[1].join()
        with pytest.raises((EOFError, BrokenPipeError, ConnectionResetError)):
            venv.step(np.array([0, 0]))
    finally:
        venv.close()
