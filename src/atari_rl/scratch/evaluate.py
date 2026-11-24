"""Evaluation: play full games with the greedy policy in one in-process eval env, optionally film the first.

The connector's eval env reports the real score (Atari: unclipped, all lives); what counts as a win and
which metric picks the best model are the connector's business (`eval_metrics`, `best_key`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from atari_rl.monitoring.video import write_mp4
from atari_rl.scratch.connectors.base import Connector
from atari_rl.scratch.nets import ActorCritic
from atari_rl.scratch.vec_env import FrameStack, make_vec_env


def play_episodes(agent: ActorCritic, connector: Connector, episodes: int, *, max_steps: int,
                  video_path: Path | None = None, heartbeat: Callable[[], None] | None = None,
                  seed: int | None = None, deterministic: bool = True) -> list[dict[str, Any]]:
    """Returns one {"score", "length", "info"} per game. Games are cut at max_steps agent steps."""
    venv = make_vec_env(connector, 1, eval=True, in_process=True)
    raw_env = (venv.venv if isinstance(venv, FrameStack) else venv).envs[0]
    results = []
    try:
        for ep in range(episodes):
            obs = venv.reset(seed if ep == 0 else None)
            frames = [raw_env.render()] if ep == 0 and video_path else None
            score, length, info = 0.0, 0, {}
            for _ in range(max_steps):
                if heartbeat is not None:
                    heartbeat()  # eval takes no training steps; keep the stall watchdog quiet
                action = agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, infos = venv.step(action)
                score += float(reward[0])
                length += 1
                info = infos[0]
                if terminated[0] or truncated[0]:
                    break  # (the env already auto-reset, so don't film its next first frame)
                if frames is not None:
                    frames.append(raw_env.render())
            results.append({"score": score, "length": length, "info": info})
            if frames is not None:
                write_mp4(frames, video_path)
    finally:
        venv.close()
    return results


def run_eval(agent: ActorCritic, connector: Connector, episodes: int, *, max_steps: int,
             video_path: Path | None = None, heartbeat: Callable[[], None] | None = None,
             seed: int | None = None) -> dict[str, float]:
    was_training = agent.training
    agent.eval()
    try:
        played = play_episodes(agent, connector, episodes, max_steps=max_steps, video_path=video_path,
                               heartbeat=heartbeat, seed=seed)
    finally:
        agent.train(was_training)
    return connector.eval_metrics(played)
