"""The scratch PPO run loop, honouring the same contract as the SB3 path (`atari_rl.train.run`):

- run dir layout: config.json, checkpoints/ckpt_<steps>.zip (newest keep_ckpts), best_model.zip + best.json,
  videos/eval_<steps>.mp4, health.json, tb/, DONE
- resume from the newest checkpoint in the run dir (timesteps, weights and Adam state continue)
- DONE = done | critical | budget | mlflow_unavailable (| stalled, written by the watchdog) on every exit path
- time budget = cfg.hours_budget, checked before every rollout
- HealthMonitor fed every iteration; CRITICAL stops the run (cfg.stop_on_critical); env crash → CRITICAL
- MLflow: refuses to train if configured but unusable; checkpoints / best model / videos / health uploaded as
  written; metric names identical to SB3's so `atari-rl status` works

Console-agnostic: everything game-specific comes from the `connector` argument.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from atari_rl.config import Settings
from atari_rl.monitoring import mlflow_tracking
from atari_rl.monitoring.health import HealthMonitor, HealthSnapshot
from atari_rl.scratch import checkpoint
from atari_rl.scratch.buffer import RolloutBuffer
from atari_rl.scratch.connectors.base import Connector
from atari_rl.scratch.evaluate import run_eval
from atari_rl.scratch.health_guard import HealthGuard
from atari_rl.scratch.metrics import MetricsWriter
from atari_rl.scratch.nets import ActorCritic
from atari_rl.scratch.ppo import PPOConfig, linear_schedule, ppo_update, set_learning_rate
from atari_rl.scratch.rollout import RolloutCollector
from atari_rl.scratch.seeding import seed_everything
from atari_rl.scratch.vec_env import make_vec_env

# Run-contract helpers shared with the SB3 path (they contain no SB3 code).
from atari_rl.train import ENV_CRASH_ERRORS, HealthReporter, _stall_handler, torch_threads

logger = logging.getLogger(__name__)

LOG_INTERVAL = 10  # iterations between metric dumps (SB3's log_interval in train.py)
EVAL_SEED_OFFSET = 10_000  # eval env seed = cfg.seed + this, never one of the training envs' seeds
_STATUS_RANK = {"HEALTHY": 0, "WARN": 1, "CRITICAL": 2}


@dataclass
class TrainState:
    agent: ActorCritic
    optimizer: torch.optim.Optimizer
    ppo: PPOConfig
    num_timesteps: int = 0
    n_updates: int = 0


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _latest_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(ckpt_dir.glob("ckpt_*.zip"))
    return ckpts[-1] if ckpts else None


def build_or_resume(cfg: Settings, connector: Connector, ckpt_dir: Path, device: torch.device) -> TrainState:
    ppo = PPOConfig.from_sources(connector.ppo_overrides, cfg.params_file)
    resume_from = _latest_checkpoint(ckpt_dir)
    if resume_from is None:
        agent = ActorCritic(connector.observation_spec, connector.n_actions, connector.share_encoder).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=ppo.learning_rate, eps=1e-5)
        return TrainState(agent, optimizer, ppo)
    if not checkpoint.is_scratch_checkpoint(resume_from):
        raise ValueError(f"{resume_from} is not a scratch checkpoint (SB3 run?): use ATARIRL_ALGO=sb3 or a new run id")
    logger.info("resuming from %s", resume_from)
    ckpt = checkpoint.load(resume_from, device)
    optimizer = torch.optim.Adam(ckpt.agent.parameters(), lr=ppo.learning_rate, eps=1e-5)
    if ckpt.optimizer_state is not None:
        optimizer.load_state_dict(ckpt.optimizer_state)
    return TrainState(ckpt.agent, optimizer, ppo, ckpt.num_timesteps, ckpt.n_updates)


class CheckpointKeeper:
    """ckpt_<steps>.zip every cfg.ckpt_freq steps, newest cfg.keep_ckpts kept, each uploaded to MLflow."""

    def __init__(self, cfg: Settings, ckpt_dir: Path, connector: Connector):
        self.cfg, self.ckpt_dir, self.connector = cfg, ckpt_dir, connector
        self.next_save = cfg.ckpt_freq

    def start(self, num_timesteps: int) -> None:
        self.next_save = (num_timesteps // self.cfg.ckpt_freq + 1) * self.cfg.ckpt_freq

    def save(self, state: TrainState) -> Path:
        path = checkpoint.save(self.ckpt_dir / f"ckpt_{state.num_timesteps:010d}.zip", state.agent, state.optimizer,
                               connector=self.connector, ppo=state.ppo, num_timesteps=state.num_timesteps,
                               n_updates=state.n_updates)
        for old in sorted(self.ckpt_dir.glob("ckpt_*.zip"))[: -self.cfg.keep_ckpts]:
            old.unlink()
        mlflow_tracking.log_artifact(path, "checkpoints")  # durable copy: GCS access ends after 1 h
        return path

    def maybe_save(self, state: TrainState) -> None:
        if state.num_timesteps >= self.next_save:
            self.save(state)
            self.next_save = (state.num_timesteps // self.cfg.ckpt_freq + 1) * self.cfg.ckpt_freq


class Evaluator:
    """Every cfg.eval_freq steps: greedy full games + one video; keeps best_model.zip by connector.best_key."""

    def __init__(self, cfg: Settings, run_dir: Path, connector: Connector, metrics: MetricsWriter,
                 heartbeat: Callable[[], None]):
        self.cfg, self.run_dir, self.connector, self.metrics, self.heartbeat = cfg, run_dir, connector, metrics, heartbeat
        self.next_eval = cfg.eval_freq
        best_json = run_dir / "best.json"  # on resume, keep the bar so a worse model can't overwrite the best
        self.best = connector.best_key(json.loads(best_json.read_text())) if best_json.exists() else None

    def start(self, num_timesteps: int) -> None:
        self.next_eval = (num_timesteps // self.cfg.eval_freq + 1) * self.cfg.eval_freq

    def maybe_eval(self, state: TrainState) -> None:
        if state.num_timesteps < self.next_eval:
            return
        steps = state.num_timesteps
        self.next_eval = (steps // self.cfg.eval_freq + 1) * self.cfg.eval_freq
        video = self.run_dir / "videos" / f"eval_{steps:010d}.mp4"
        result = run_eval(state.agent, self.connector, self.cfg.eval_episodes, max_steps=self.cfg.eval_max_steps,
                          video_path=video, heartbeat=self.heartbeat, seed=self.cfg.seed + EVAL_SEED_OFFSET)
        if video.exists():
            mlflow_tracking.log_artifact(video, "videos")
        logger.info("eval @%d: %s", steps, result)
        self.metrics.write({f"eval/{k}": v for k, v in result.items()}, step=steps)
        key = self.connector.best_key(result)
        if self.best is None or key > self.best:
            self.best = key
            checkpoint.save(self.run_dir / "best_model.zip", state.agent, None, connector=self.connector,
                            ppo=state.ppo, num_timesteps=steps, n_updates=state.n_updates)
            (self.run_dir / "best.json").write_text(json.dumps({"step": steps, **result}, default=float))
            mlflow_tracking.log_artifact(self.run_dir / "best.json")
            mlflow_tracking.log_artifact(self.run_dir / "best_model.zip")


def _all_finite(agent: ActorCritic, losses: dict[str, float]) -> bool:
    """NaN check for health: the last loss and every weight."""
    return bool(np.isfinite(losses["loss"])) and all(torch.isfinite(p).all().item() for p in agent.parameters())


def train_loop(cfg: Settings, connector: Connector, venv, state: TrainState, guard: HealthGuard,
               keeper: CheckpointKeeper, evaluator: Evaluator, metrics: MetricsWriter, deadline: float,
               shuffle_rng: torch.Generator, device: torch.device) -> str:
    """Returns done | critical | budget. The caller owns cleanup (final checkpoint, DONE, MLflow)."""
    ppo = state.ppo
    buffer = RolloutBuffer(ppo.n_steps, venv.num_envs, connector.observation_spec, device)
    collector = RolloutCollector(venv, state.agent, buffer, device, seed=cfg.seed, heartbeat=guard.heartbeat)
    episodes: deque[dict] = deque(maxlen=100)  # SB3's ep_info_buffer: rollout/ep_rew_mean over the last 100 games
    keeper.start(state.num_timesteps)
    evaluator.start(state.num_timesteps)
    start_time, start_steps = time.time(), state.num_timesteps
    last_health = (time.monotonic(), state.num_timesteps)
    iteration = 0
    guard.start()

    while state.num_timesteps < cfg.steps:
        if time.time() >= deadline:
            logger.warning("time budget reached, stopping at %d steps", state.num_timesteps)
            return "budget"
        iteration += 1

        # 1. schedules: linear decay over the whole run (progress survives resume via num_timesteps)
        progress = state.num_timesteps / cfg.steps
        lr = linear_schedule(ppo.learning_rate, progress) if ppo.anneal_lr else ppo.learning_rate
        clip = linear_schedule(ppo.clip_range, progress) if ppo.anneal_clip else ppo.clip_range
        set_learning_rate(state.optimizer, lr)

        # 2. play, 3. learn
        stats = collector.collect(ppo.gamma, ppo.gae_lambda)
        state.num_timesteps += stats.steps
        train_metrics = ppo_update(state.agent, state.optimizer, buffer, ppo, clip, shuffle_rng)
        state.n_updates += ppo.n_epochs
        episodes.extend(stats.episodes)

        # 4. health: one snapshot per iteration, same fields the SB3 HealthCallback fills
        now = time.monotonic()
        fps = (state.num_timesteps - last_health[1]) / (now - last_health[0]) if now > last_health[0] else None
        last_health = (now, state.num_timesteps)
        ep_rew_mean = float(np.mean([e["r"] for e in episodes])) if episodes else None
        report = guard.update(HealthSnapshot(
            step=state.num_timesteps,
            frac_done=state.num_timesteps / cfg.steps,
            approx_kl=train_metrics["approx_kl"],
            clip_fraction=train_metrics["clip_fraction"],
            entropy=train_metrics["entropy"],
            explained_variance=train_metrics["explained_variance"],
            ep_rew_mean=ep_rew_mean,
            fps=fps,
            button_freq=(stats.action_counts / stats.action_counts.sum()).tolist(),
            losses_finite=_all_finite(state.agent, train_metrics),
            wall_time=now,
        ))

        # 5. metrics (SB3 names), checkpoints, eval
        done = state.num_timesteps >= cfg.steps
        if iteration % LOG_INTERVAL == 0 or done or report.status == "CRITICAL":
            elapsed = max(time.time() - start_time, 1e-9)
            record = {f"train/{k}": v for k, v in train_metrics.items() if k != "entropy"}
            record.update({
                "train/learning_rate": lr, "train/n_updates": state.n_updates,
                "rollout/ep_rew_mean": ep_rew_mean,
                "rollout/ep_len_mean": float(np.mean([e["l"] for e in episodes])) if episodes else None,
                "time/fps": int((state.num_timesteps - start_steps) / elapsed), "time/iterations": iteration,
                "time/time_elapsed": int(elapsed), "time/total_timesteps": state.num_timesteps,
                "health/status": _STATUS_RANK[report.status],
            })
            metrics.write(record, step=state.num_timesteps)
        keeper.maybe_save(state)
        evaluator.maybe_eval(state)

        if report.status == "CRITICAL":
            logger.error("CRITICAL training health: %s", report.reasons)
            if cfg.stop_on_critical:
                return "critical"

    return "critical" if guard.last_report and guard.last_report.status == "CRITICAL" else "done"


def run(cfg: Settings, connector: Connector, env_factory: Callable[[Settings], object] | None = None) -> str:
    """Train (or resume) until cfg.steps, CRITICAL health, or the time budget.

    Returns done | critical | budget | mlflow_unavailable. DONE, the final checkpoint and the MLflow uploads
    happen on every exit path.
    """
    run_dir = cfg.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.model_dump_json(indent=2))
    deadline = time.time() + cfg.hours_budget * 3600

    if mlflow_tracking.enabled() and not mlflow_tracking.start(cfg, probe=run_dir / "config.json"):
        logger.error("MLflow is configured but unusable — refusing to train without a durable channel")
        (run_dir / "DONE").write_text("mlflow_unavailable")
        return "mlflow_unavailable"
    mlflow_tracking.set_tag("algo", "scratch")

    guard = HealthGuard(HealthMonitor(cfg), run_dir / "health.json", stall_secs=cfg.fps_zero_secs,
                        on_stall=_stall_handler(run_dir), on_report=HealthReporter(run_dir / "health.json"))
    keeper = CheckpointKeeper(cfg, ckpt_dir, connector)
    metrics = MetricsWriter(run_dir)
    env_factory = env_factory or (lambda c: make_vec_env(connector, c.n_envs))
    device = _device()
    if device.type == "cpu":
        torch.set_num_threads(torch_threads(os.cpu_count() or 2, cfg.n_envs))

    status, state, venv = "done", None, None
    try:
        shuffle_rng = seed_everything(cfg.seed)
        venv = env_factory(cfg)
        state = build_or_resume(cfg, connector, ckpt_dir, device)
        evaluator = Evaluator(cfg, run_dir, connector, metrics, heartbeat=guard.heartbeat)
        status = train_loop(cfg, connector, venv, state, guard, keeper, evaluator, metrics, deadline, shuffle_rng,
                            device)
    except ENV_CRASH_ERRORS as e:
        logger.exception("environment worker crashed: %s", e)
        guard.mark_critical("env_crash")
        status = "critical"
    except Exception as e:  # e.g. NaN logits make Categorical raise before any health check can run
        logger.exception("training crashed: %s", e)
        guard.mark_critical("crash")
        status = "critical"
    finally:
        guard.stop()
        if state is not None:
            keeper.save(state)
        if venv is not None:
            try:
                venv.close()
            except Exception:
                logger.exception("env.close failed")
        metrics.close()
        (run_dir / "DONE").write_text(status)
        for artifact in ("health.json", "best.json", "DONE"):
            if (run_dir / artifact).exists():
                mlflow_tracking.log_artifact(run_dir / artifact)
        mlflow_tracking.set_tag("health_status", guard.last_report.status if guard.last_report else "UNKNOWN")
        mlflow_tracking.end(status)

    logger.info("run %s finished: %s at %d steps", cfg.run_id, status, state.num_timesteps if state else 0)
    return status
