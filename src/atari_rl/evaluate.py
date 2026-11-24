from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from atari_rl.config import Settings
from atari_rl.envs.factory import make_single
from atari_rl.monitoring import mlflow_tracking
from atari_rl.monitoring.video import write_mp4

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    mean_score: float  # raw game score: unclipped rewards over a full game (all lives)
    std_score: float
    max_score: float
    mean_length: float  # agent steps per game
    target_reached: float  # 1.0 when mean_score >= the game's TARGET_SCORE


def run_eval(model, cfg: Settings, episodes: int, video_path: Path | None = None,
             max_steps: int | None = None) -> EvalResult:
    """Deterministic full games in one in-process env; the score is the raw game score."""
    max_steps = max_steps or cfg.eval_max_steps
    venv = VecFrameStack(DummyVecEnv([make_single(cfg, rank=10_000, eval_mode=True)]), cfg.frame_stack,
                         channels_order="last")
    raw_env = venv.venv.envs[0]
    scores, lengths = [], []
    try:
        for ep in range(episodes):
            obs = venv.reset()
            frames = [raw_env.render()] if ep == 0 and video_path else None
            ep_score, ep_len = 0.0, 0
            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = venv.step(action)
                ep_score += float(reward[0])
                ep_len += 1
                if frames is not None:
                    frames.append(raw_env.render())
                if done[0]:
                    break
            scores.append(ep_score)
            lengths.append(ep_len)
            if frames is not None:
                write_mp4(frames, video_path)
    finally:
        venv.close()

    mean = float(np.mean(scores))
    return EvalResult(mean_score=mean, std_score=float(np.std(scores)), max_score=float(np.max(scores)),
                      mean_length=float(np.mean(lengths)), target_reached=float(mean >= cfg.target_score))


def score(r: EvalResult) -> tuple[float, ...]:
    """Ordering key for best-model selection (higher is better)."""
    return (r.mean_score,)


class EvalCallback(BaseCallback):
    """Every cfg.eval_freq steps: deterministic eval + one video; keeps best_model.zip."""

    def __init__(self, cfg: Settings, run_dir: Path, eval_fn: Callable[..., EvalResult] = run_eval):
        super().__init__()
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.eval_fn = eval_fn
        self.next_eval = cfg.eval_freq
        self.best: tuple[float, ...] | None = None

    def _on_step(self) -> bool:
        if self.model.num_timesteps < self.next_eval:
            return True
        steps = self.model.num_timesteps
        self.next_eval = (steps // self.cfg.eval_freq + 1) * self.cfg.eval_freq
        video = self.run_dir / "videos" / f"eval_{steps:010d}.mp4"
        result = self.eval_fn(self.model, self.cfg, self.cfg.eval_episodes, video_path=video)
        if video.exists():
            mlflow_tracking.log_artifact(video, "videos")
        for k, v in asdict(result).items():
            self.logger.record(f"eval/{k}", v)
        key = score(result)
        logger.info("eval @%d: %s (target %.0f)", steps, result, self.cfg.target_score)
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(self.run_dir / "best_model.zip")
            (self.run_dir / "best.json").write_text(
                json.dumps({"step": steps, "target": self.cfg.target_score, **asdict(result)}, default=float))
            mlflow_tracking.log_artifact(self.run_dir / "best.json")
        return True
