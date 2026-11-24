"""Scratch trainer honours the SB3 run contract (mirrors tests/test_train.py)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from atari_rl.config import Settings
from atari_rl.scratch import checkpoint
from atari_rl.scratch.connectors.atari import AtariConnector
from atari_rl.scratch.trainer import run
from atari_rl.scratch.vec_env import make_vec_env


def _cfg(tmp_path, **kw):
    base = dict(game="pong", n_envs=2, total_steps=1024, eval_freq=1024, ckpt_freq=512,
                eval_episodes=1, eval_max_steps=100, runs_dir=tmp_path, run_id="t", algo="scratch")
    base.update(kw)
    return Settings(**base)


def _run(cfg, env_factory=None):
    return run(cfg, AtariConnector.from_settings(cfg), env_factory=env_factory)


def test_smoke_writes_artifacts(tmp_path):
    assert _run(_cfg(tmp_path)) == "done"
    rd = tmp_path / "t"
    ckpts = sorted((rd / "checkpoints").glob("*.zip"))
    assert ckpts and all(checkpoint.is_scratch_checkpoint(p) for p in ckpts)
    assert json.loads((rd / "health.json").read_text())["status"] in {"HEALTHY", "WARN"}
    assert (rd / "config.json").exists()
    assert (rd / "DONE").read_text() == "done"
    assert checkpoint.is_scratch_checkpoint(rd / "best_model.zip")
    best = json.loads((rd / "best.json").read_text())
    assert best["target"] == 18.0 and best["step"] == 1024 and "win_rate" in best
    assert list((rd / "videos").glob("eval_*.mp4"))
    assert list((rd / "tb").iterdir())


def test_resume_continues_timesteps(tmp_path):
    _run(_cfg(tmp_path, total_steps=512, eval_freq=10_000))
    first = sorted((tmp_path / "t" / "checkpoints").glob("*.zip"))
    assert first and checkpoint.load(first[-1]).num_timesteps == 512
    _run(_cfg(tmp_path, total_steps=1024, eval_freq=10_000))
    last = sorted((tmp_path / "t" / "checkpoints").glob("*.zip"))[-1]
    assert int(last.stem.split("_")[-1]) >= 1024
    loaded = checkpoint.load(last)
    assert loaded.num_timesteps == 1024 and loaded.n_updates == 2 * 4 * 2  # 2 runs × 2 iterations × 4 epochs


def test_budget_stop_writes_done(tmp_path):
    assert _run(_cfg(tmp_path, max_hours=0.0, eval_freq=10_000)) == "budget"
    assert (tmp_path / "t" / "DONE").read_text() == "budget"


class _RaiseAfter:
    """Vec env stand-in that raises `exc` after n steps (a dying emulator worker looks like EOFError)."""

    def __init__(self, venv, n, exc):
        self.venv, self.n, self.exc = venv, n, exc
        self.num_envs = venv.num_envs

    def reset(self, seed=None):
        return self.venv.reset(seed)

    def step(self, actions):
        self.n -= 1
        if self.n < 0:
            raise self.exc
        return self.venv.step(actions)

    def close(self):
        self.venv.close()


def _factory(n, exc):
    return lambda c: _RaiseAfter(make_vec_env(AtariConnector.from_settings(c), c.n_envs), n, exc)


def test_env_crash_marks_critical(tmp_path):
    status = _run(_cfg(tmp_path, eval_freq=10_000), env_factory=_factory(300, EOFError("worker died")))
    assert status == "critical"
    health = json.loads((tmp_path / "t" / "health.json").read_text())
    assert health["status"] == "CRITICAL" and "env_crash" in health["reasons"]
    assert list((tmp_path / "t" / "checkpoints").glob("*.zip"))


def test_unexpected_exception_marks_critical_and_writes_done(tmp_path):
    status = _run(_cfg(tmp_path, eval_freq=10_000), env_factory=_factory(300, ValueError("nan logits")))
    assert status == "critical"
    rd = tmp_path / "t"
    assert (rd / "DONE").read_text() == "critical"
    assert "crash" in json.loads((rd / "health.json").read_text())["reasons"]


def test_env_factory_failure_still_writes_done(tmp_path):
    def boom(c):
        raise RuntimeError("cannot start emulator")

    assert _run(_cfg(tmp_path), env_factory=boom) == "critical"
    assert (tmp_path / "t" / "DONE").read_text() == "critical"


def test_mlflow_unavailable_aborts_before_training(tmp_path, monkeypatch):
    from atari_rl.monitoring import mlflow_tracking

    monkeypatch.setattr(mlflow_tracking, "enabled", lambda: True)
    monkeypatch.setattr(mlflow_tracking, "start", lambda cfg, probe=None: False)
    assert _run(_cfg(tmp_path)) == "mlflow_unavailable"
    assert (tmp_path / "t" / "DONE").read_text() == "mlflow_unavailable"
    assert not list((tmp_path / "t" / "checkpoints").glob("*.zip"))


def test_mlflow_uploads_and_metric_names_match_sb3(tmp_path, monkeypatch):
    from atari_rl.monitoring import mlflow_tracking

    uploads, logged = [], {}
    monkeypatch.setattr(mlflow_tracking, "log_artifact", lambda p, a=None: uploads.append((str(p), a)))
    monkeypatch.setattr(mlflow_tracking, "log_metrics", lambda m, step: logged.update(m))
    _run(_cfg(tmp_path))
    assert sum(1 for _, a in uploads if a == "checkpoints") >= 2  # periodic + final
    assert any(a == "videos" for _, a in uploads)
    assert any(p.endswith("best_model.zip") for p, _ in uploads)
    for key in ("time/total_timesteps", "time/fps", "train/approx_kl", "train/clip_fraction", "train/entropy_loss",
                "train/explained_variance", "train/value_loss", "train/policy_gradient_loss", "train/loss",
                "train/learning_rate", "train/clip_range", "health/status", "eval/mean_score",
                "eval/target_reached"):
        assert key in logged, key


def test_params_file_override(tmp_path):
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"n_steps": 64, "gamma": 0.9}))
    _run(_cfg(tmp_path, params_file=pf, total_steps=256, eval_freq=10_000))
    last = sorted((tmp_path / "t" / "checkpoints").glob("*.zip"))[-1]
    ppo = checkpoint.load(last).ppo
    assert ppo.n_steps == 64 and ppo.gamma == 0.9


def test_sb3_checkpoint_in_run_dir_is_refused(tmp_path):
    ckpt_dir = tmp_path / "t" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    with zipfile.ZipFile(ckpt_dir / "ckpt_0000000512.zip", "w") as z:
        z.writestr("data", "{}")
    assert _run(_cfg(tmp_path)) == "critical"
    assert "crash" in json.loads((tmp_path / "t" / "health.json").read_text())["reasons"]


def test_health_snapshot_has_ppo_metrics(tmp_path):
    _run(_cfg(tmp_path, eval_freq=10_000))
    last = json.loads((tmp_path / "t" / "health.json").read_text())["last"]
    for k in ("approx_kl", "clip_fraction", "entropy", "explained_variance", "fps"):
        assert last[k] is not None and np.isfinite(last[k]), k
    assert len(last["button_freq"]) == 6 and abs(sum(last["button_freq"]) - 1.0) < 1e-9
    assert last["entropy"] > 1.5  # ~ln(6) = 1.79 for a fresh policy


def test_scratch_core_never_imports_sb3_or_atari():
    """Console-agnostic core: no SB3, no ALE / Atari connector imports outside scratch/connectors/."""
    root = Path(__file__).parents[1] / "src" / "atari_rl" / "scratch"
    for path in root.rglob("*.py"):
        src = path.read_text()
        assert "stable_baselines3" not in src, path
        if "connectors" not in path.parts:
            for banned in ("ale_py", "connectors.atari", "envs.atari", "atari_wrappers"):
                assert banned not in src, (path, banned)
