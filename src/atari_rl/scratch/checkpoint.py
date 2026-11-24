"""Scratch checkpoint format: a plain zip you can open with any unzip tool.

    ckpt_0000250000.zip / best_model.zip
    ├── meta.json      format tag, connector name + config, obs spec, n_actions, PPO config, timesteps
    ├── model.pt       ActorCritic state_dict (torch.save)
    └── optimizer.pt   Adam state_dict, so a resumed run continues exactly where it stopped

The same file names as the SB3 path (`ckpt_*.zip`, `best_model.zip`) keep `atari-rl videos/play` and the
pod's resume sync unchanged; `is_scratch_checkpoint` tells the two formats apart (SB3 zips have no meta.json).
Tensors are loaded with `weights_only=True`: a downloaded checkpoint cannot run code.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from atari_rl.scratch.connectors.base import Connector, spec_from_json, spec_to_json
from atari_rl.scratch.nets import ActorCritic
from atari_rl.scratch.ppo import PPOConfig

FORMAT = "scratch-ppo/1"


def _tensor_bytes(obj: Any) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def save(path: Path, agent: ActorCritic, optimizer: torch.optim.Optimizer | None, *, connector: Connector,
         ppo: PPOConfig, num_timesteps: int, n_updates: int) -> Path:
    path = Path(path)
    meta = {
        "format": FORMAT,
        "connector": connector.name,
        "connector_config": dataclasses.asdict(connector) if dataclasses.is_dataclass(connector) else {},
        "obs_spec": spec_to_json(connector.observation_spec),
        "n_actions": connector.n_actions,
        "share_encoder": connector.share_encoder,
        "ppo": ppo.to_dict(),
        "num_timesteps": num_timesteps,
        "n_updates": n_updates,
    }
    tmp = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as z:  # write-then-rename: a crash never leaves a half checkpoint
        z.writestr("meta.json", json.dumps(meta, indent=2))
        z.writestr("model.pt", _tensor_bytes(agent.state_dict()))
        if optimizer is not None:
            z.writestr("optimizer.pt", _tensor_bytes(optimizer.state_dict()))
    os.replace(tmp, path)
    return path


def read_meta(path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("meta.json")) if "meta.json" in z.namelist() else None
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError):
        return None


def is_scratch_checkpoint(path: Path) -> bool:
    meta = read_meta(path)
    return meta is not None and meta.get("format") == FORMAT


@dataclass
class Checkpoint:
    agent: ActorCritic
    optimizer_state: dict[str, Any] | None
    ppo: PPOConfig
    num_timesteps: int
    n_updates: int
    meta: dict[str, Any]


def load(path: Path, device: torch.device | str = "cpu") -> Checkpoint:
    meta = read_meta(path)
    if meta is None or meta.get("format") != FORMAT:
        raise ValueError(f"{path} is not a scratch PPO checkpoint ({FORMAT})")
    agent = ActorCritic(spec_from_json(meta["obs_spec"]), meta["n_actions"], share_encoder=meta["share_encoder"])
    with zipfile.ZipFile(path) as z:
        agent.load_state_dict(torch.load(io.BytesIO(z.read("model.pt")), map_location=device, weights_only=True))
        opt_state = (torch.load(io.BytesIO(z.read("optimizer.pt")), map_location=device, weights_only=True)
                     if "optimizer.pt" in z.namelist() else None)
    return Checkpoint(agent.to(device), opt_state, PPOConfig(**meta["ppo"]), int(meta["num_timesteps"]),
                      int(meta["n_updates"]), meta)


def load_agent(path: Path, device: torch.device | str = "cpu") -> ActorCritic:
    return load(path, device).agent
