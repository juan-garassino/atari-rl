"""Rollout collection, checkpoints and the CartPole loop (scratch/rollout.py, checkpoint.py, cartpole.py)."""
from __future__ import annotations

import json
import zipfile

import gymnasium as gym
import numpy as np
import pytest
import torch

from atari_rl.scratch import checkpoint
from atari_rl.scratch.buffer import RolloutBuffer
from atari_rl.scratch.cartpole import train_cartpole
from atari_rl.scratch.connectors.atari import AtariConnector
from atari_rl.scratch.connectors.base import VectorSpec
from atari_rl.scratch.connectors.cartpole import CartPoleConnector
from atari_rl.scratch.nets import ActorCritic, obs_to_tensor
from atari_rl.scratch.ppo import PPOConfig
from atari_rl.scratch.rollout import RolloutCollector
from atari_rl.scratch.vec_env import InProcessVecEnv


class _TimeLimited(gym.Env):
    """State = step count; every episode is cut by a time limit after 3 steps (truncated, never terminated)."""

    observation_space = gym.spaces.Dict({"s": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)})
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return {"s": np.array([0.0], np.float32)}, {}

    def step(self, action):
        self.t += 1
        end = self.t == 3
        info = {"episode": {"r": 3.0, "l": 3}} if end else {}
        return {"s": np.array([float(self.t)], np.float32)}, 1.0, False, end, info


def test_collector_bootstraps_truncation_from_final_obs_value():
    torch.manual_seed(0)
    agent = ActorCritic({"s": VectorSpec(1)}, n_actions=2)
    buf = RolloutBuffer(5, 2, {"s": VectorSpec(1)}, torch.device("cpu"))
    venv = InProcessVecEnv([_TimeLimited, _TimeLimited])
    collector = RolloutCollector(venv, agent, buf, torch.device("cpu"), seed=0)
    stats = collector.collect(gamma=0.99, lam=0.95)
    assert buf.truncated[2].all() and not buf.truncated[[0, 1, 3, 4]].any()
    with torch.no_grad():
        expected = agent.value({"s": torch.tensor([[3.0], [3.0]])})  # V(final obs), not V(next episode's first obs)
    assert torch.allclose(buf.final_values[2], expected)
    assert [e["r"] for e in stats.episodes] == [3.0, 3.0]
    assert stats.action_counts.sum() == 10 and stats.steps == 10
    assert buf.obs["s"][3, :, 0].tolist() == [0.0, 0.0]  # row 3 starts the new episode


def test_collector_calls_heartbeat_every_step():
    beats = []
    agent = ActorCritic({"s": VectorSpec(1)}, n_actions=2)
    buf = RolloutBuffer(4, 1, {"s": VectorSpec(1)}, torch.device("cpu"))
    RolloutCollector(InProcessVecEnv([_TimeLimited]), agent, buf, torch.device("cpu"), seed=0,
                     heartbeat=lambda: beats.append(1)).collect(0.99, 0.95)
    assert len(beats) >= 4


def test_checkpoint_roundtrip(tmp_path):
    c = AtariConnector(game="pong")
    torch.manual_seed(0)
    agent = ActorCritic(c.observation_spec, c.n_actions)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3, eps=1e-5)
    agent(obs_to_tensor({"pixels": np.ones((1, 4, 84, 84), np.uint8)}, torch.device("cpu")))[1].sum().backward()
    opt.step()  # non-empty optimizer state
    path = checkpoint.save(tmp_path / "ckpt.zip", agent, opt, connector=c, ppo=PPOConfig(),
                           num_timesteps=1234, n_updates=5)
    assert checkpoint.is_scratch_checkpoint(path)
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("meta.json"))
    assert meta["connector"] == "atari" and meta["connector_config"]["game"] == "pong"

    loaded = checkpoint.load(path)
    assert loaded.num_timesteps == 1234 and loaded.n_updates == 5 and loaded.ppo == PPOConfig()
    for a, b in zip(agent.state_dict().values(), loaded.agent.state_dict().values()):
        assert torch.equal(a, b)
    opt2 = torch.optim.Adam(loaded.agent.parameters(), lr=1e-3, eps=1e-5)
    opt2.load_state_dict(loaded.optimizer_state)
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
    obs = {"pixels": np.random.default_rng(0).integers(0, 255, (3, 4, 84, 84), dtype=np.uint8)}
    assert np.array_equal(agent.predict(obs), checkpoint.load_agent(path).predict(obs))


def test_sb3_zip_is_not_a_scratch_checkpoint(tmp_path):
    sb3 = tmp_path / "sb3.zip"
    with zipfile.ZipFile(sb3, "w") as z:
        z.writestr("data", "{}")
        z.writestr("policy.pth", b"")
    assert not checkpoint.is_scratch_checkpoint(sb3)
    assert not checkpoint.is_scratch_checkpoint(tmp_path / "missing.zip")


def test_cartpole_loop_runs_and_is_seed_deterministic():
    results = [train_cartpole(total_steps=1024, n_envs=2, seed=5, eval_episodes=1) for _ in range(2)]
    a, b = results
    assert a.steps == 1024 and a.iterations == 8  # 1024 / (2 envs × 64 steps)
    assert a.final_params_checksum == b.final_params_checksum
    assert a.eval_mean_return == b.eval_mean_return


@pytest.mark.parametrize("seed", [0])
def test_cartpole_different_seed_differs(seed):
    a = train_cartpole(total_steps=512, n_envs=2, seed=seed, eval_episodes=1)
    b = train_cartpole(total_steps=512, n_envs=2, seed=seed + 1, eval_episodes=1)
    assert a.final_params_checksum != b.final_params_checksum
