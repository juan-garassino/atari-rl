"""The PPO update: clipped surrogate + value loss − entropy bonus, minibatch epochs, gradient-norm clipping.

Reference: Schulman et al. 2017; implementation details follow CleanRL's ppo_atari.py and SB3's PPO, so
the numbers we log (approx_kl, clip_fraction, ...) mean exactly what they mean in SB3.

    total loss = policy_gradient_loss + vf_coef · value_loss − ent_coef · entropy
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from atari_rl.scratch.buffer import RolloutBuffer
from atari_rl.scratch.nets import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    """CleanRL / SB3-zoo Atari defaults (design doc). Override per run with a JSON params file."""

    n_steps: int = 128  # env steps per env per rollout
    n_minibatches: int = 4  # rollout (n_steps · n_envs samples) is cut into this many minibatches
    n_epochs: int = 4  # passes over the rollout per update
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True  # linear decay to 0 over the run
    clip_range: float = 0.1  # ε: how far the probability ratio may move from 1
    anneal_clip: bool = True  # linear decay to 0 over the run
    clip_range_vf: float | None = None  # value clipping, off by default (it tends to hurt; see 37-details blog)
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantage: bool = True  # per minibatch, like SB3 and CleanRL

    def with_overrides(self, overrides: dict[str, Any]) -> PPOConfig:
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown PPO params {sorted(unknown)}; known: {sorted(known)}")
        return replace(self, **overrides)

    @classmethod
    def from_sources(cls, connector_overrides: dict[str, Any], params_file: Path | None) -> PPOConfig:
        """Core defaults ← the connector's overrides ← the run's params file (highest priority)."""
        cfg = cls().with_overrides(connector_overrides)
        if params_file:
            cfg = cfg.with_overrides(json.loads(Path(params_file).read_text()))
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def linear_schedule(start: float, progress: float) -> float:
    """progress = fraction of the run done (0 → 1); the value decays linearly from start to 0."""
    return start * (1.0 - min(max(progress, 0.0), 1.0))


def normalize_advantages(adv: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (adv - adv.mean()) / (adv.std() + eps)


def clipped_surrogate_loss(new_logprob: torch.Tensor, old_logprob: torch.Tensor, advantages: torch.Tensor,
                           clip_range: float) -> tuple[torch.Tensor, dict[str, float]]:
    """L_clip = −E[min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)], ratio = π_new(a|s) / π_old(a|s).

    The min makes the objective pessimistic: once the ratio has moved ε in the direction the advantage wants,
    the clipped term is constant and its gradient is 0, so one rollout can't drag the policy too far.
    """
    log_ratio = new_logprob - old_logprob
    ratio = log_ratio.exp()
    unclipped = advantages * ratio
    clipped = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        # KL(old‖new) estimator k3 from http://joschu.net/blog/kl-approx.html — always ≥ 0, low variance.
        approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
        clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean().item()
    return loss, {"approx_kl": approx_kl, "clip_fraction": clip_fraction}


def value_loss(new_values: torch.Tensor, returns: torch.Tensor, old_values: torch.Tensor,
               clip_range_vf: float | None) -> torch.Tensor:
    """MSE between V(s) and the GAE return target (SB3 convention: no ½ factor; vf_coef scales it)."""
    if clip_range_vf is None:
        return ((new_values - returns) ** 2).mean()
    clipped = old_values + torch.clamp(new_values - old_values, -clip_range_vf, clip_range_vf)
    return torch.max((new_values - returns) ** 2, (clipped - returns) ** 2).mean()


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """1 − Var(y − ŷ)/Var(y): 1 = the critic predicts returns perfectly, 0 = no better than a constant, < 0 = worse."""
    var_y = torch.var(y_true).item()
    return float("nan") if var_y == 0 else 1.0 - torch.var(y_true - y_pred).item() / var_y


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def ppo_update(agent: ActorCritic, optimizer: torch.optim.Optimizer, buffer: RolloutBuffer, cfg: PPOConfig,
               clip_range: float, generator: torch.Generator) -> dict[str, float]:
    """n_epochs × n_minibatches gradient steps on one finished rollout. Returns SB3-named train/ metrics."""
    pg_losses, v_losses, entropies, kls, clip_fracs, grad_norms = [], [], [], [], [], []
    loss = torch.zeros(())
    for _ in range(cfg.n_epochs):
        for mb in buffer.minibatches(cfg.n_minibatches, generator):
            new_logprob, entropy, new_values = agent.evaluate(mb.obs, mb.actions)
            adv = normalize_advantages(mb.advantages) if cfg.normalize_advantage else mb.advantages

            pg_loss, stats = clipped_surrogate_loss(new_logprob, mb.logprobs, adv, clip_range)
            v_loss = value_loss(new_values, mb.returns, mb.values, cfg.clip_range_vf)
            entropy_mean = entropy.mean()
            loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)  # norm *before* clipping
            optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy_mean.item())
            kls.append(stats["approx_kl"])
            clip_fracs.append(stats["clip_fraction"])
            grad_norms.append(grad_norm.item())

    # How much of the return variance the critic explained *before* this update (values from collection time).
    ev = explained_variance(buffer.values.flatten(), buffer.returns.flatten())
    entropy = float(np.mean(entropies))
    return {
        "policy_gradient_loss": float(np.mean(pg_losses)),
        "value_loss": float(np.mean(v_losses)),
        "entropy": entropy,
        "entropy_loss": -entropy,  # SB3's name and sign
        "approx_kl": float(np.mean(kls)),
        "clip_fraction": float(np.mean(clip_fracs)),
        "loss": loss.item(),  # last minibatch, like SB3
        "explained_variance": ev,
        "grad_norm": float(np.mean(grad_norms)),
        "clip_range": clip_range,
    }
