from __future__ import annotations

import pytest

from atari_rl.config import TARGET_SCORE, Settings, default_max_eur, default_total_steps


def test_defaults():
    s = Settings()
    assert s.game == "pong"
    assert s.n_envs == 16
    assert s.frame_stack == 4


def test_env_prefix(monkeypatch):
    monkeypatch.setenv("ATARIRL_GAME", "breakout")
    monkeypatch.setenv("ATARIRL_N_ENVS", "4")
    s = Settings()
    assert s.game == "breakout"
    assert s.n_envs == 4


def test_default_total_steps():
    assert default_total_steps("pong") == 3_000_000
    assert default_total_steps("breakout") == 10_000_000


def test_default_max_eur():
    assert default_max_eur("pong") == 1.5
    assert default_max_eur("breakout") == 3.5


def test_targets():
    assert TARGET_SCORE == {"pong": 18.0, "breakout": 200.0}


def test_resolved_budget(monkeypatch):
    monkeypatch.setenv("ATARIRL_GAME", "breakout")
    s = Settings()
    assert s.steps == 10_000_000
    assert s.budget_eur == 3.5
    monkeypatch.setenv("ATARIRL_MAX_EUR", "3")
    assert Settings().budget_eur == 3.0


def test_import_ale_bundled_roms():
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)
    assert {"ALE/Pong-v5", "ALE/Breakout-v5"} <= set(gym.registry)


def test_budget_from_cost(monkeypatch):
    monkeypatch.setenv("ATARIRL_GAME", "breakout")  # 3.5 EUR cap
    monkeypatch.setenv("ATARIRL_COST_PER_HR_USD", "0.5")
    assert Settings().hours_budget == 6.0  # 3.5 EUR buys 7.35 h > 6 h cap
    monkeypatch.setenv("ATARIRL_COST_PER_HR_USD", "1.0")
    assert Settings().hours_budget == pytest.approx(3.5 * 1.05 / 1.0)


def test_budget_without_cost_uses_max_hours():
    assert Settings(max_hours=2.0).hours_budget == 2.0
