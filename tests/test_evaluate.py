from __future__ import annotations

import json
import sys

import numpy as np
from gymnasium.spaces import Discrete

from atari_rl.config import Settings
from atari_rl.evaluate import EvalCallback, EvalResult, run_eval, score
from atari_rl.monitoring import mlflow_tracking


class _Logger:
    def __init__(self):
        self.records = {}

    def record(self, key, value, exclude=None):
        self.records[key] = value


class _RandomModel:
    def __init__(self, action_space):
        self.logger = _Logger()
        self.action_space = action_space
        self.saved = []
        self.num_timesteps = 0

    def predict(self, obs, deterministic=True):
        return np.array([self.action_space.sample() for _ in range(len(obs))]), None

    def save(self, path):
        self.saved.append(str(path))
        open(path, "w").close()

    def get_env(self):
        return None


def test_eval_pong_returns_result(tmp_path):
    video = tmp_path / "v.mp4"
    result = run_eval(_RandomModel(Discrete(6)), Settings(game="pong"), episodes=1, video_path=video, max_steps=200)
    assert isinstance(result, EvalResult)
    assert isinstance(result.mean_score, float) and result.mean_length == 200
    assert result.target_reached == 0.0
    assert video.exists() and video.stat().st_size > 0


def test_eval_plays_full_raw_game():
    """A random Pong agent loses the whole game: the CPU reaches 21 points (raw score, not one clipped point)."""
    result = run_eval(_RandomModel(Discrete(6)), Settings(game="pong"), episodes=1, max_steps=10_000)
    assert result.mean_length < 10_000
    assert -21.0 <= result.mean_score <= -15.0


def test_score_key_is_mean_raw_score():
    r = EvalResult(mean_score=12.0, std_score=1.0, max_score=14.0, mean_length=900.0, target_reached=0.0)
    assert score(r) == (12.0,)


def test_mlflow_disabled_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)  # any import attempt would raise
    assert mlflow_tracking.enabled() is False
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.end("done")


def _result(mean):
    return EvalResult(mean_score=mean, std_score=0.0, max_score=mean, mean_length=1000.0,
                      target_reached=float(mean >= 18))


def test_best_model_selection(tmp_path):
    results = iter([_result(5.0), _result(3.0), _result(19.0)])
    cfg = Settings(game="pong", eval_freq=100)
    cb = EvalCallback(cfg, tmp_path, eval_fn=lambda *a, **k: next(results))
    model = _RandomModel(Discrete(6))
    cb.init_callback(model)
    for steps in (100, 200, 300):
        model.num_timesteps = steps
        cb.on_step()
    assert len(model.saved) == 2
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["step"] == 300 and best["mean_score"] == 19.0 and best["target"] == 18.0
    assert model.logger.records["eval/mean_score"] == 19.0
