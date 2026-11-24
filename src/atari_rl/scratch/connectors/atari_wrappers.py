"""The standard Atari 2600 preprocessing (Mnih et al. 2015; Machado et al. 2018), written out by hand.

Training stack, innermost first (same order as SB3's AtariWrapper, so frames match it exactly):

    ALE v5 (frameskip=1, sticky actions 0.25)
    RecordEpisodeStatistics   raw full-game score → info["episode"]   (sits under everything that alters rewards/episodes)
    NoopReset(30)             random number of NOOPs after reset → varied starting states
    MaxAndSkip(4)             repeat the action 4 frames, sum rewards, max-pool the last 2 frames (Atari sprites flicker)
    EpisodicLife              losing a life ends the *training* episode (value bootstrapping gets a clear signal)
    FireReset                 press FIRE after reset so games like Breakout actually start
    WarpFrame                 RGB 210×160 → gray 84×84
    ClipReward                sign(reward) ∈ {−1, 0, +1}: the same learning rate works across games

Eval drops EpisodicLife and ClipReward (full games, real score) and adds FireOnLifeLoss.
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np

NOOP, FIRE = 0, 1  # ALE minimal action set indices shared by Pong and Breakout


class NoopReset(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # seeded by reset(seed=...)
        obs, info = None, {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(NOOP)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        self.last_two = np.zeros((2, *env.observation_space.shape), dtype=env.observation_space.dtype)

    def step(self, action):
        total_reward, terminated, truncated, info = 0.0, False, False, {}
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self.skip - 2:
                self.last_two[0] = obs
            if i == self.skip - 1:
                self.last_two[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break  # the frame returned on a terminal step is never used for learning
        return self.last_two.max(axis=0), total_reward, terminated, truncated, info


class EpisodicLife(gym.Wrapper):
    """A lost life looks like the end of an episode to the learner; the emulator only resets on game over."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.game_over = terminated or truncated
        lives = self.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.game_over:
            obs, info = self.env.reset(**kwargs)
        else:  # just a lost life: keep playing the same game with one NOOP
            obs, _, terminated, truncated, info = self.env.step(NOOP)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.unwrapped.ale.lives()
        return obs, info


class FireReset(gym.Wrapper):
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(FIRE)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: int = 84):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(0, 255, (size, size), np.uint8)

    def observation(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ClipReward(gym.RewardWrapper):
    def reward(self, reward) -> float:
        return float(np.sign(float(reward)))


class FireOnLifeLoss(gym.Wrapper):
    """Eval only: press FIRE once after a lost life. Training relaunches the ball via FireReset, so the agent
    never learns FIRE itself; without this a full Breakout eval game would stall after the first life."""

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
