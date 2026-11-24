"""Actor-critic networks, built from the connector's observation spec.

    obs dict ──► one encoder per key ──► concat ──► features ──┬─► policy head → logits over actions
                 ImageSpec  → NatureCNN (512)                   └─► value head  → V(s)
                 VectorSpec → MLP (64)

Atari: a single "pixels" key and a shared encoder — exactly SB3's CnnPolicy / CleanRL's ppo_atari.
CartPole: a single "state" key and separate actor/critic encoders (CleanRL's ppo.py).
A dict-obs console (Pokémon: screens + stats) needs no change here: it just has two keys.

Initialisation (the "37 implementation details of PPO" recipe): orthogonal weights, zero biases, gain √2 for
hidden layers, 0.01 for the policy head (start near-uniform → high initial entropy), 1 for the value head.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from atari_rl.scratch.connectors.base import ImageSpec, ObsSpec, VectorSpec


def layer_init(layer: nn.Module, std: float = math.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class NatureCNN(nn.Module):
    """Mnih et al. 2015 (Nature DQN) encoder: 3 convs + 512-unit linear. Input uint8 (C, 84, 84) → 512 features."""

    def __init__(self, spec: ImageSpec, out_dim: int = 512):
        super().__init__()
        c, h, w = spec.shape
        convs = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():  # infer the flattened size (3136 for 84×84) instead of hard-coding it
            n_flat = convs(torch.zeros(1, c, h, w)).shape[1]
        self.net = nn.Sequential(*convs, layer_init(nn.Linear(n_flat, out_dim)), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float() / 255.0)  # pixels are stored as uint8 0..255; the net sees 0..1


class MLPEncoder(nn.Module):
    """Two tanh layers for flat feature vectors."""

    def __init__(self, spec: VectorSpec, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(spec.size, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class ObsEncoder(nn.Module):
    """One encoder per observation key, outputs concatenated in sorted-key order (stable across runs)."""

    def __init__(self, spec: ObsSpec):
        super().__init__()
        self.keys = sorted(spec)
        self.encoders = nn.ModuleDict(
            {k: NatureCNN(spec[k]) if isinstance(spec[k], ImageSpec) else MLPEncoder(spec[k]) for k in self.keys})
        self.out_dim = sum(e.out_dim for e in self.encoders.values())

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.encoders[k](obs[k]) for k in self.keys], dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, spec: ObsSpec, n_actions: int, share_encoder: bool = True):
        super().__init__()
        self.actor_encoder = ObsEncoder(spec)
        self.critic_encoder = self.actor_encoder if share_encoder else ObsEncoder(spec)
        self.policy_head = layer_init(nn.Linear(self.actor_encoder.out_dim, n_actions), std=0.01)
        self.value_head = layer_init(nn.Linear(self.critic_encoder.out_dim, 1), std=1.0)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """(logits (B, n_actions), value (B,))."""
        features = self.actor_encoder(obs)
        critic_features = features if self.critic_encoder is self.actor_encoder else self.critic_encoder(obs)
        return self.policy_head(features), self.value_head(critic_features).squeeze(-1)

    def value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value_head(self.critic_encoder(obs)).squeeze(-1)

    def act(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout time: sample a ~ π(·|s). Returns (action, log π(a|s), V(s))."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: dict[str, torch.Tensor], actions: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update time: re-score the stored actions under the current weights. (log π(a|s), entropy, V(s))."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    @torch.no_grad()
    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        """Numpy in, numpy actions out (eval / play). Deterministic = the most likely action."""
        logits, _ = self(obs_to_tensor(obs, next(self.parameters()).device))
        actions = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
        return actions.cpu().numpy()


def obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in obs.items()}
