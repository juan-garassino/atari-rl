from __future__ import annotations

from atari_rl.config import Settings
from atari_rl.monitoring import mlflow_tracking
from atari_rl.monitoring.health import HealthReport
from atari_rl.status import format_status
from atari_rl.train import HealthReporter


class _Rec:
    def __init__(self):
        self.tags, self.artifacts = {}, []

    def set_tag(self, k, v):
        self.tags[k] = v

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((str(path), artifact_path))


def test_health_reporter_tags_on_change_and_throttles_uploads(tmp_path, monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(mlflow_tracking, "set_tag", rec.set_tag)
    monkeypatch.setattr(mlflow_tracking, "log_artifact", rec.log_artifact)
    health_json = tmp_path / "health.json"
    health_json.write_text("{}")
    clock = iter([0.0, 10.0, 700.0])
    reporter = HealthReporter(health_json, upload_every_s=600, clock=lambda: next(clock))

    reporter(HealthReport("HEALTHY", []))
    reporter(HealthReport("WARN", ["approx_kl"]))
    reporter(HealthReport("WARN", ["approx_kl"]))

    assert rec.tags["health_status"] == "WARN"
    assert rec.tags["health_reasons"] == "approx_kl"
    assert len(rec.artifacts) == 2  # t=0 and t=700, not t=10


def test_format_status():
    text = format_status("pong-x", {"health_status": 1.0, "eval_mean_score": 17.5, "rollout_ep_rew_mean": 12.0,
                                     "time_fps": 900.0},
                         {"health_status": "WARN", "health_reasons": "approx_kl", "final_status": None}, step=500_000)
    assert "WARN" in text and "approx_kl" in text and "500,000" in text and "17.5" in text


def test_eval_callback_uploads_video_and_best(tmp_path, monkeypatch):
    from atari_rl.evaluate import EvalCallback, EvalResult

    rec = _Rec()
    monkeypatch.setattr(mlflow_tracking, "log_artifact", rec.log_artifact)

    def fake_eval(model, cfg, episodes, video_path=None):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"mp4")
        return EvalResult(mean_score=3.0, std_score=0.0, max_score=3.0, mean_length=10.0, target_reached=0.0)

    class _Model:
        num_timesteps = 100

        class logger:  # noqa: N801 - SB3 logger stand-in
            @staticmethod
            def record(key, value, exclude=None):
                pass

        def get_env(self):
            return None

        def save(self, path):
            open(path, "w").close()

    cb = EvalCallback(Settings(eval_freq=100), tmp_path, eval_fn=fake_eval)
    cb.init_callback(_Model())
    cb.on_step()
    assert any(a == "videos" for _, a in rec.artifacts)
    assert any(p.endswith("best.json") for p, _ in rec.artifacts)


def test_mlflow_new_helpers_noop_when_disabled():
    assert mlflow_tracking.enabled() is False
    mlflow_tracking.set_tag("a", "b")
    mlflow_tracking.log_artifact(__file__, "x")


class _FakeMlflow:
    def __init__(self, fail_start):
        self.fail_start = fail_start
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append("uri")

    def set_experiment(self, name):
        if self.fail_start:
            raise RuntimeError("401 not authenticated")

    def start_run(self, run_name):
        self.calls.append("start")

    def log_metrics(self, metrics, step=0):
        self.calls.append("metrics")

    def set_tag(self, k, v):
        self.calls.append("tag")

    def log_params(self, p):
        pass

    def end_run(self):
        self.calls.append("end")


def test_failed_start_disables_all_logging(monkeypatch):
    fake = _FakeMlflow(fail_start=True)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.set_tag("k", "v")
    assert "metrics" not in fake.calls and "tag" not in fake.calls  # no stray auto-created runs


def test_successful_start_enables_logging(monkeypatch):
    fake = _FakeMlflow(fail_start=False)
    monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    monkeypatch.setattr(mlflow_tracking, "_active", False)
    mlflow_tracking.start(Settings())
    mlflow_tracking.log_metrics({"a": 1.0}, step=1)
    mlflow_tracking.end("done")
    assert fake.calls.count("metrics") == 1 and "end" in fake.calls
    assert mlflow_tracking._active is False
