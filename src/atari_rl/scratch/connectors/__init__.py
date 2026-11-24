"""Console connectors. The PPO core never imports this package's console modules directly; the CLI picks
one with `connector_for(cfg)` and hands it to the trainer.

| Connector | Console | Observation |
|---|---|---|
| `atari.AtariConnector` | Atari 2600 via ALE (pong, breakout) | {"pixels": 4×84×84 uint8} |
| `cartpole.CartPoleConnector` | CartPole-v1 (sanity check only) | {"state": 4 floats} |
"""
from __future__ import annotations

from atari_rl.config import Settings
from atari_rl.scratch.connectors.base import Connector


def connector_for(cfg: Settings) -> Connector:
    from atari_rl.scratch.connectors.atari import AtariConnector

    return AtariConnector.from_settings(cfg)
