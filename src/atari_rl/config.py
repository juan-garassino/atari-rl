from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Game = Literal["pong", "breakout"]
Algo = Literal["sb3", "scratch"]

_TOTAL_STEPS = {"pong": 3_000_000, "breakout": 10_000_000}
_MAX_EUR = {"pong": 1.5, "breakout": 3.5}
# Raw game score (unclipped, full game) we call "solved"; logged, never enforced.
TARGET_SCORE = {"pong": 18.0, "breakout": 200.0}


def default_total_steps(game: str) -> int:
    return _TOTAL_STEPS[game]


def default_max_eur(game: str) -> float:
    return _MAX_EUR[game]


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATARIRL_", extra="ignore")

    game: Game = "pong"  # ATARIRL_GAME=breakout
    algo: Algo = "sb3"  # ATARIRL_ALGO=scratch (our own PPO in atari_rl/scratch; sb3 stays default until proven)
    run_id: str = Field(default_factory=_utc_run_id)  # ATARIRL_RUN_ID=20261124-150000
    total_steps: int | None = None  # ATARIRL_TOTAL_STEPS=3000000 (None = per-game default)
    n_envs: int = 16  # ATARIRL_N_ENVS=16
    frame_skip: int = 4  # ATARIRL_FRAME_SKIP=4
    frame_stack: int = 4  # ATARIRL_FRAME_STACK=4
    sticky_prob: float = 0.25  # ATARIRL_STICKY_PROB=0.25 (ALE v5 repeat_action_probability)
    seed: int = 0  # ATARIRL_SEED=0
    params_file: Path | None = None  # ATARIRL_PARAMS_FILE=runs/x/ppo_params.json (overrides PPO_DEFAULTS)
    eval_freq: int = 250_000  # ATARIRL_EVAL_FREQ=250000
    eval_episodes: int = 5  # ATARIRL_EVAL_EPISODES=5
    eval_max_steps: int = 10_000  # ATARIRL_EVAL_MAX_STEPS=10000 (agent steps per eval game)
    ckpt_freq: int = 250_000  # ATARIRL_CKPT_FREQ=250000
    keep_ckpts: int = 3  # ATARIRL_KEEP_CKPTS=3
    runs_dir: Path = Path("runs")  # ATARIRL_RUNS_DIR=/workspace/runs
    stop_on_critical: bool = True  # ATARIRL_STOP_ON_CRITICAL=false
    max_hours: float = 6.0  # ATARIRL_MAX_HOURS=6
    max_eur: float | None = None  # ATARIRL_MAX_EUR=3.5 (None = per-game default)
    usd_per_eur: float = 1.05  # ATARIRL_USD_PER_EUR=1.05
    cost_per_hr_usd: float | None = None  # ATARIRL_COST_PER_HR_USD=0.44 (set by the RunPod entrypoint)

    kl_warn: float = 0.05  # ATARIRL_KL_WARN=0.05
    kl_crit: float = 0.2  # ATARIRL_KL_CRIT=0.2
    clipfrac_warn: float = 0.3  # ATARIRL_CLIPFRAC_WARN=0.3
    entropy_collapse_frac: float = 0.2  # ATARIRL_ENTROPY_COLLAPSE_FRAC=0.2
    ev_warn_after: float = 0.1  # ATARIRL_EV_WARN_AFTER=0.1 (fraction of training)
    ev_crit: float = -0.5  # ATARIRL_EV_CRIT=-0.5
    plateau_frac: float = 0.2  # ATARIRL_PLATEAU_FRAC=0.2
    fps_warn_frac: float = 0.5  # ATARIRL_FPS_WARN_FRAC=0.5
    fps_zero_secs: float = 300.0  # ATARIRL_FPS_ZERO_SECS=300
    button_spam_frac: float = 0.9  # ATARIRL_BUTTON_SPAM_FRAC=0.9

    @property
    def steps(self) -> int:
        return self.total_steps if self.total_steps is not None else default_total_steps(self.game)

    @property
    def budget_eur(self) -> float:
        return self.max_eur if self.max_eur is not None else default_max_eur(self.game)

    @property
    def target_score(self) -> float:
        return TARGET_SCORE[self.game]

    @property
    def hours_budget(self) -> float:
        """Wall-clock cap: max_hours, tightened by the EUR cap when the pod's hourly cost is known."""
        if not self.cost_per_hr_usd:
            return self.max_hours
        return min(self.max_hours, self.budget_eur * self.usd_per_eur / self.cost_per_hr_usd)

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id


settings = Settings()
