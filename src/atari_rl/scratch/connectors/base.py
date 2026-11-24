"""The contract between the console-agnostic PPO core and one console (Atari, Genesis, Game Boy, ...).

The core (nets, buffer, ppo, vec_env, rollout, trainer, evaluate, checkpoint) only ever talks to a
`Connector`. Everything that knows about a specific emulator — how to build the env, how to preprocess
frames, whether rewards are clipped, what counts as a "win" — lives behind this interface.

Observations are always a dict of numpy arrays, one entry per named input:

    Atari:    {"pixels": uint8 (84, 84)}                    → network sees (4, 84, 84) after frame stacking
    CartPole: {"state": float32 (4,)}
    Pokémon:  {"screens": uint8 (3, 72, 80), "stats": float32 (k,)}   (phase 2, no core change needed)

`observation_spec` describes what the *network* sees (after the core's frame stacking), so the core can build
the right encoder per key: a CNN for every ImageSpec, an MLP for every VectorSpec, concatenated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Union

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class ImageSpec:
    """A uint8 image input, channels first, e.g. (4, 84, 84) = 4 stacked gray frames. Scaled /255 by the net."""

    shape: tuple[int, int, int]


@dataclass(frozen=True)
class VectorSpec:
    """A flat float32 feature vector, e.g. CartPole's 4 state values or Pokémon's party levels."""

    size: int


ObsSpec = dict[str, Union[ImageSpec, VectorSpec]]


def spec_to_json(spec: ObsSpec) -> dict[str, Any]:
    """Serialisable form, stored in checkpoints so a model can be rebuilt without its connector."""
    out: dict[str, Any] = {}
    for key, s in spec.items():
        if isinstance(s, ImageSpec):
            out[key] = {"kind": "image", "shape": list(s.shape)}
        else:
            out[key] = {"kind": "vector", "size": s.size}
    return out


def spec_from_json(data: dict[str, Any]) -> ObsSpec:
    return {key: ImageSpec(tuple(s["shape"])) if s["kind"] == "image" else VectorSpec(int(s["size"]))
            for key, s in data.items()}


class Connector(Protocol):
    """One console/game family. Must be picklable: env workers receive it to build their env.

    Attributes
    - name:            short id stored in checkpoints ("atari", "cartpole", ...).
    - observation_spec: what the network sees per obs key (after frame stacking).
    - n_actions:       size of the Discrete action space.
    - frame_stack:     how many consecutive frames the core stacks for every *image* key (1 = none).
                       Stacking expects single-channel (H, W) frames from the env.
    - share_encoder:   True = one encoder feeds both heads (Atari, CleanRL/SB3 CnnPolicy);
                       False = separate actor and critic encoders (CleanRL CartPole).
    - ppo_overrides:   PPO hyper-parameters that differ from the core defaults for this console.
    """

    name: str
    observation_spec: ObsSpec
    n_actions: int
    frame_stack: int
    share_encoder: bool
    ppo_overrides: dict[str, Any]

    def make_env(self, rank: int, *, eval: bool) -> gym.Env:
        """One env with dict observations. Training env: shaped/clipped rewards are fine.
        Eval env (eval=True): the real game score per step, a full game per episode, render() → RGB frame.
        Both must put info["episode"] = {"r": raw score, "l": length} when a real game ends.
        """
        ...

    def eval_metrics(self, episodes: list[dict[str, Any]]) -> dict[str, float]:
        """Eval episodes ({"score", "length", "info"} each) → metrics logged as eval/<name>, incl. the win/target."""
        ...

    def best_key(self, metrics: dict[str, float]) -> tuple[float, ...]:
        """Ordering key for best-model selection (higher is better)."""
        ...


class DictObservation(gym.ObservationWrapper):
    """Wraps a plain-array observation into {key: obs} so every env speaks the core's dict format."""

    def __init__(self, env: gym.Env, key: str):
        super().__init__(env)
        self.key = key
        self.observation_space = gym.spaces.Dict({key: env.observation_space})

    def observation(self, observation: np.ndarray) -> dict[str, np.ndarray]:
        return {self.key: observation}
