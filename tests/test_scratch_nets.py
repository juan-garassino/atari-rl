"""Actor-critic networks built from an observation spec (src/atari_rl/scratch/nets.py)."""
from __future__ import annotations

import math

import numpy as np
import torch

from atari_rl.scratch.connectors.base import ImageSpec, VectorSpec
from atari_rl.scratch.nets import ActorCritic, obs_to_tensor

ATARI = {"pixels": ImageSpec((4, 84, 84))}


def _atari_obs(n=3, fill=0):
    return {"pixels": torch.full((n, 4, 84, 84), fill, dtype=torch.uint8)}


def test_nature_cnn_shapes():
    net = ActorCritic(ATARI, n_actions=6)
    logits, value = net(_atari_obs())
    assert logits.shape == (3, 6) and value.shape == (3,)


def test_orthogonal_init_head_scales():
    net = ActorCritic(ATARI, n_actions=6)
    w_pi, w_v = net.policy_head.weight, net.value_head.weight
    # orthogonal rows scaled by the gain: W Wᵀ = gain² I
    assert torch.allclose(w_pi @ w_pi.T, 0.01 ** 2 * torch.eye(6), atol=1e-6)
    assert torch.allclose(w_v @ w_v.T, torch.ones(1, 1), atol=1e-5)
    assert torch.count_nonzero(net.policy_head.bias) == 0 and torch.count_nonzero(net.value_head.bias) == 0
    first_conv = net.actor_encoder.encoders["pixels"].net[0]
    w = first_conv.weight.reshape(first_conv.weight.shape[0], -1)
    assert torch.allclose(w @ w.T, 2.0 * torch.eye(w.shape[0]), atol=1e-4)  # gain √2


def test_initial_policy_is_near_uniform():
    """A 0.01-gain policy head starts with ~uniform action probabilities: entropy ≈ ln(n_actions)."""
    net = ActorCritic(ATARI, n_actions=6)
    obs = {"pixels": torch.randint(0, 256, (8, 4, 84, 84), dtype=torch.uint8)}
    _, entropy, _ = net.evaluate(obs, torch.zeros(8, dtype=torch.long))
    assert entropy.mean().item() > 0.99 * math.log(6)


def test_pixels_are_scaled_to_unit_range():
    net = ActorCritic(ATARI, n_actions=6)
    enc = net.actor_encoder.encoders["pixels"]
    x = torch.full((1, 4, 84, 84), 255, dtype=torch.uint8)
    assert torch.allclose(enc(x), enc.net(torch.ones(1, 4, 84, 84)))


def test_vector_spec_uses_mlp_and_separate_encoders():
    net = ActorCritic({"state": VectorSpec(4)}, n_actions=2, share_encoder=False)
    assert net.actor_encoder is not net.critic_encoder
    logits, value = net({"state": torch.zeros(5, 4)})
    assert logits.shape == (5, 2) and value.shape == (5,)


def test_dict_obs_concatenates_image_and_vector_parts():
    spec = {"screens": ImageSpec((3, 72, 80)), "stats": VectorSpec(7)}
    net = ActorCritic(spec, n_actions=8)
    obs = {"screens": torch.zeros(2, 3, 72, 80, dtype=torch.uint8), "stats": torch.zeros(2, 7)}
    logits, value = net(obs)
    assert logits.shape == (2, 8) and value.shape == (2,)
    assert net.actor_encoder.out_dim == 512 + 64


def test_act_and_evaluate_agree():
    torch.manual_seed(0)
    net = ActorCritic({"state": VectorSpec(4)}, n_actions=3)
    obs = {"state": torch.randn(16, 4)}
    actions, logprobs, values = net.act(obs)
    lp, _, v = net.evaluate(obs, actions)
    assert torch.allclose(lp, logprobs) and torch.allclose(v, values)


def test_predict_deterministic_is_argmax():
    net = ActorCritic({"state": VectorSpec(4)}, n_actions=3)
    obs = {"state": np.random.default_rng(0).normal(size=(4, 4)).astype(np.float32)}
    logits, _ = net(obs_to_tensor(obs, torch.device("cpu")))
    assert np.array_equal(net.predict(obs, deterministic=True), logits.argmax(-1).numpy())


def test_same_seed_same_weights():
    torch.manual_seed(7)
    a = ActorCritic(ATARI, n_actions=4)
    torch.manual_seed(7)
    b = ActorCritic(ATARI, n_actions=4)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)
