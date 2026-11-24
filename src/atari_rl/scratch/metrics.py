"""Where training numbers go: a one-line log, TensorBoard (run_dir/tb) and MLflow.

Metric names are SB3's (train/approx_kl, rollout/ep_rew_mean, time/fps, eval/mean_score, ...), so the
MLflow dashboards and `atari-rl status` read a scratch run exactly like an SB3 run.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

from atari_rl.monitoring import mlflow_tracking

logger = logging.getLogger(__name__)

# The keys shown in the one-line console summary (everything is still logged to TB/MLflow).
_SUMMARY = ("time/total_timesteps", "time/fps", "rollout/ep_rew_mean", "train/approx_kl", "train/clip_fraction",
            "train/entropy_loss", "train/explained_variance", "eval/mean_score")


class MetricsWriter:
    def __init__(self, run_dir: Path, tensorboard: bool = True):
        self.tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(run_dir / "tb"))

    def write(self, metrics: dict[str, float], step: int) -> None:
        finite = {k: float(v) for k, v in metrics.items() if v is not None and math.isfinite(float(v))}
        if self.tb is not None:
            for k, v in finite.items():
                self.tb.add_scalar(k, v, step)
        mlflow_tracking.log_metrics(finite, step=step)  # no-op unless MLflow is active; "/" → "_"
        shown = "  ".join(f"{k.split('/')[-1]}={finite[k]:.4g}" for k in _SUMMARY if k in finite)
        logger.info("%s", shown)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()
