"""GAE(λ) and rollout storage (src/atari_rl/scratch/buffer.py).

Hand-computed cases use γ=0.5, λ=1 so every number can be checked on paper.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from atari_rl.scratch.buffer import RolloutBuffer, compute_gae
from atari_rl.scratch.connectors.base import ImageSpec, VectorSpec


def _t(rows):
    return torch.tensor(rows, dtype=torch.float32).reshape(len(rows), -1)


def _gae(r, v, term, trunc, final, last, gamma=0.5, lam=1.0):
    return compute_gae(_t(r), _t(v), _t(term).bool(), _t(trunc).bool(), _t(final), torch.tensor(last, dtype=torch.float32),
                       gamma, lam)


def test_no_episode_end_bootstraps_from_last_value():
    # δ1 = 1 + .5*2 - 0 = 2 ; δ0 = 1 + .5*0 - 0 = 1 ; A0 = δ0 + .5*A1 = 2
    adv, ret = _gae([1, 1], [0, 0], [0, 0], [0, 0], [0, 0], [2.0])
    assert adv[:, 0].tolist() == [2.0, 2.0]
    assert ret[:, 0].tolist() == [2.0, 2.0]  # returns = advantages + values


def test_termination_cuts_bootstrap_and_carry():
    # t=1 terminated: δ1 = 1 - 0 = 1 (no next value) ; A0 = 1 + .5*1 = 1.5
    adv, _ = _gae([1, 1], [0, 0], [0, 1], [0, 0], [0, 0], [5.0])
    assert adv[:, 0].tolist() == [1.5, 1.0]


def test_truncation_bootstraps_from_final_observation_value():
    # t=1 truncated by a time limit: the game did not end, so bootstrap from V(final obs) = 4:
    # δ1 = 1 + .5*4 - 0 = 3 ; A0 = 1 + .5*3 = 2.5 (last_value 5 belongs to the NEXT episode and is ignored)
    adv, _ = _gae([1, 1], [0, 0], [0, 0], [0, 1], [0, 4], [5.0])
    assert adv[:, 0].tolist() == [2.5, 3.0]


def test_episode_boundary_inside_rollout():
    # Episode ends at t=0, a new one starts at t=1: A0 must not see rewards from t=1..2.
    # A2 = 1 + .5*2 = 2 ; A1 = 1 + .5*A2 = 2 ; A0 = 1
    adv, _ = _gae([1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0], [2.0])
    assert adv[:, 0].tolist() == [1.0, 2.0, 2.0]


def test_values_subtract_and_lambda_discounts():
    # γ=0.9 λ=0.8, one step then bootstrap: δ = r + γV' - V
    r, v, last = [1.0, 0.0], [0.5, 0.25], [1.0]
    adv, ret = _gae(r, v, [0, 0], [0, 0], [0, 0], last, gamma=0.9, lam=0.8)
    d1 = 0.0 + 0.9 * 1.0 - 0.25
    d0 = 1.0 + 0.9 * 0.25 - 0.5
    assert adv[1, 0].item() == pytest.approx(d1)
    assert adv[0, 0].item() == pytest.approx(d0 + 0.9 * 0.8 * d1)
    assert ret[0, 0].item() == pytest.approx(adv[0, 0].item() + 0.5)


def _reference_gae(r, v, term, trunc, final, last, gamma, lam):
    """Slow, obviously-correct reference: A_t = Σ_k (γλ)^k δ_{t+k} up to the end of t's episode."""
    T, N = r.shape
    adv = np.zeros((T, N))
    for n in range(N):
        deltas, ends = [], []
        for t in range(T):
            if term[t, n]:
                nv = 0.0
            elif trunc[t, n]:
                nv = final[t, n]
            else:
                nv = v[t + 1, n] if t + 1 < T else last[n]
            deltas.append(r[t, n] + gamma * nv - v[t, n])
            ends.append(bool(term[t, n] or trunc[t, n]))
        for t in range(T):
            total, k = 0.0, 0
            while t + k < T:
                total += (gamma * lam) ** k * deltas[t + k]
                if ends[t + k]:
                    break
                k += 1
            adv[t, n] = total
    return adv


def test_matches_reference_on_random_rollouts():
    rng = np.random.default_rng(0)
    T, N = 17, 5
    r = rng.normal(size=(T, N))
    v = rng.normal(size=(T, N))
    term = rng.random((T, N)) < 0.1
    trunc = (rng.random((T, N)) < 0.1) & ~term
    final = np.where(trunc, rng.normal(size=(T, N)), 0.0)
    last = rng.normal(size=N)
    f = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    adv, ret = compute_gae(f(r), f(v), torch.tensor(term), torch.tensor(trunc), f(final), f(last), 0.99, 0.95)
    ref = _reference_gae(r, v, term, trunc, final, last, 0.99, 0.95)
    np.testing.assert_allclose(adv.numpy(), ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret.numpy(), ref + v, rtol=1e-5, atol=1e-5)


def test_buffer_stores_dict_obs_and_yields_every_sample_once_per_epoch():
    spec = {"pixels": ImageSpec((2, 3, 3)), "stats": VectorSpec(2)}
    buf = RolloutBuffer(n_steps=4, n_envs=2, obs_spec=spec, device=torch.device("cpu"))
    for t in range(4):
        obs = {"pixels": np.full((2, 2, 3, 3), t, np.uint8), "stats": np.full((2, 2), t, np.float32)}
        buf.add(t, obs, actions=torch.tensor([t, t]), logprobs=torch.zeros(2), values=torch.zeros(2),
                rewards=np.ones(2), terminated=np.zeros(2, bool), truncated=np.zeros(2, bool),
                final_values=np.zeros(2))
    buf.finish(last_values=torch.zeros(2), gamma=0.99, lam=0.95)
    assert buf.obs["pixels"].dtype == torch.uint8
    seen = []
    for mb in buf.minibatches(n_minibatches=4, generator=torch.Generator().manual_seed(0)):
        assert mb.obs["pixels"].shape == (2, 2, 3, 3) and mb.obs["stats"].shape == (2, 2)
        assert torch.equal(mb.obs["stats"][:, 0].long(), mb.actions)  # rows stay aligned across fields
        seen += mb.actions.tolist()
    assert sorted(seen) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_minibatches_must_divide_batch():
    buf = RolloutBuffer(n_steps=3, n_envs=1, obs_spec={"s": VectorSpec(1)}, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        next(buf.minibatches(n_minibatches=2, generator=torch.Generator()))
