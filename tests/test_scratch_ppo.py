"""PPO loss pieces and one update (src/atari_rl/scratch/ppo.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from atari_rl.scratch.buffer import RolloutBuffer
from atari_rl.scratch.connectors.base import VectorSpec
from atari_rl.scratch.nets import ActorCritic
from atari_rl.scratch.ppo import (
    PPOConfig,
    clipped_surrogate_loss,
    explained_variance,
    linear_schedule,
    normalize_advantages,
    ppo_update,
    value_loss,
)


def test_surrogate_is_scalar_and_minus_advantage_at_ratio_one():
    old = torch.log(torch.tensor([0.2, 0.5, 0.3]))
    adv = torch.tensor([1.0, -2.0, 0.5])
    loss, stats = clipped_surrogate_loss(old.clone(), old, adv, clip_range=0.2)
    assert loss.shape == ()
    assert loss.item() == pytest.approx(-adv.mean().item())  # we *minimise* −A·ratio
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-7) and stats["clip_fraction"] == 0.0


def test_surrogate_gradient_pushes_good_actions_up():
    new = torch.zeros(1, requires_grad=True)
    loss, _ = clipped_surrogate_loss(new, torch.zeros(1), torch.tensor([1.0]), clip_range=0.2)
    loss.backward()
    assert new.grad.item() < 0  # gradient descent raises log π(a) for a positive advantage


@pytest.mark.parametrize("adv,log_ratio", [(1.0, 0.5), (-1.0, -0.5)])
def test_surrogate_clipped_side_has_no_gradient(adv, log_ratio):
    """Ratio already past 1±ε in the advantage's direction: no further push (that is PPO's trust region)."""
    new = torch.tensor([log_ratio], requires_grad=True)
    loss, stats = clipped_surrogate_loss(new, torch.zeros(1), torch.tensor([adv]), clip_range=0.2)
    loss.backward()
    assert new.grad.item() == 0.0
    assert stats["clip_fraction"] == 1.0
    assert loss.item() == pytest.approx(-adv * (1.2 if adv > 0 else 0.8))


def test_surrogate_pessimistic_side_keeps_gradient():
    """Ratio moved the *wrong* way (bad action made likelier): the unclipped term wins and still corrects it."""
    new = torch.tensor([0.5], requires_grad=True)
    loss, _ = clipped_surrogate_loss(new, torch.zeros(1), torch.tensor([-1.0]), clip_range=0.2)
    loss.backward()
    assert new.grad.item() > 0


def test_approx_kl_positive_when_policy_moves():
    old = torch.log(torch.full((4,), 0.25))
    new = old + torch.tensor([0.1, -0.1, 0.2, -0.3])
    _, stats = clipped_surrogate_loss(new, old, torch.ones(4), clip_range=0.1)
    assert stats["approx_kl"] > 0 and stats["clip_fraction"] == 0.75


def test_normalize_advantages():
    adv = torch.tensor([1.0, 2.0, 3.0, 10.0])
    out = normalize_advantages(adv)
    assert out.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert out.std().item() == pytest.approx(1.0, abs=1e-3)
    assert torch.argsort(out).tolist() == torch.argsort(adv).tolist()


def test_value_clipping_off_by_default():
    assert PPOConfig().clip_range_vf is None
    new, ret, old = torch.tensor([3.0]), torch.tensor([0.0]), torch.tensor([0.0])
    assert value_loss(new, ret, old, None).item() == pytest.approx(9.0)  # plain MSE
    # clipped: prediction may move at most 0.5 from the old value → max(9, 0.25) is still 9 (pessimistic)
    assert value_loss(new, ret, old, 0.5).item() == pytest.approx(9.0)
    assert value_loss(torch.tensor([3.0]), torch.tensor([4.0]), torch.tensor([0.0]), 0.5).item() == pytest.approx(12.25)


def test_explained_variance():
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert explained_variance(y, y) == pytest.approx(1.0)
    assert explained_variance(torch.full((4,), 2.5), y) == pytest.approx(0.0)
    assert np.isnan(explained_variance(y, torch.ones(4)))  # undefined for a constant target


def test_linear_schedule():
    assert linear_schedule(2.5e-4, 0.0) == 2.5e-4
    assert linear_schedule(2.5e-4, 0.5) == pytest.approx(1.25e-4)
    assert linear_schedule(0.1, 1.0) == 0.0


def test_config_defaults_match_design():
    c = PPOConfig()
    assert (c.n_steps, c.n_minibatches, c.n_epochs, c.learning_rate, c.clip_range) == (128, 4, 4, 2.5e-4, 0.1)
    assert (c.gamma, c.gae_lambda, c.ent_coef, c.vf_coef, c.max_grad_norm) == (0.99, 0.95, 0.01, 0.5, 0.5)
    assert c.anneal_lr and c.anneal_clip


def test_config_overrides_and_params_file(tmp_path):
    assert PPOConfig().with_overrides({"gamma": 0.9}).gamma == 0.9
    with pytest.raises(ValueError, match="batch_size"):
        PPOConfig().with_overrides({"batch_size": 256})
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"n_steps": 64}))
    assert PPOConfig.from_sources({"ent_coef": 0.0}, pf).n_steps == 64


def _filled_buffer(seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = ActorCritic({"s": VectorSpec(3)}, n_actions=2)
    buf = RolloutBuffer(8, 2, {"s": VectorSpec(3)}, torch.device("cpu"))
    for t in range(8):
        obs = {"s": rng.normal(size=(2, 3)).astype(np.float32)}
        with torch.no_grad():
            a, lp, v = agent.act({"s": torch.as_tensor(obs["s"])})
        buf.add(t, obs, a, lp, v, rng.normal(size=2), rng.random(2) < 0.2, np.zeros(2, bool), np.zeros(2))
    buf.finish(torch.zeros(2), 0.99, 0.95)
    return agent, buf


def test_update_returns_health_metrics_and_changes_weights():
    agent, buf = _filled_buffer()
    before = [p.detach().clone() for p in agent.parameters()]
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3, eps=1e-5)
    cfg = PPOConfig(n_steps=8, n_minibatches=2, n_epochs=2)
    m = ppo_update(agent, opt, buf, cfg, clip_range=0.2, generator=torch.Generator().manual_seed(0))
    for k in ("approx_kl", "clip_fraction", "entropy", "entropy_loss", "explained_variance", "policy_gradient_loss",
              "value_loss", "loss", "grad_norm"):
        assert k in m and np.isfinite(m[k]), k
    assert m["entropy_loss"] == pytest.approx(-m["entropy"])
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.parameters()))


def test_update_is_deterministic_given_seeds():
    params = []
    for _ in range(2):
        agent, buf = _filled_buffer(seed=3)
        opt = torch.optim.Adam(agent.parameters(), lr=1e-3, eps=1e-5)
        ppo_update(agent, opt, buf, PPOConfig(n_steps=8, n_minibatches=2, n_epochs=2), 0.2,
                   torch.Generator().manual_seed(1))
        params.append(torch.cat([p.detach().flatten() for p in agent.parameters()]))
    assert torch.equal(params[0], params[1])
