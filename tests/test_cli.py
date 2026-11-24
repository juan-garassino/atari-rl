from __future__ import annotations

import pytest

from atari_rl.cli import build_parser, settings_from_args


def test_train_overrides():
    args = build_parser().parse_args(["train", "--game", "breakout", "--steps", "500", "--n-envs", "4",
                                      "--run-id", "x"])
    cfg = settings_from_args(args)
    assert (cfg.game, cfg.steps, cfg.n_envs, cfg.run_id) == ("breakout", 500, 4, "x")


def test_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_subcommands():
    p = build_parser()
    assert p.parse_args(["smoke"]).cmd == "smoke"
    args = p.parse_args(["eval", "--game", "pong", "--model", "m.zip", "--episodes", "2"])
    assert (args.cmd, args.episodes, str(args.model)) == ("eval", 2, "m.zip")
    with pytest.raises(SystemExit):
        p.parse_args(["tune"])
    with pytest.raises(SystemExit):
        p.parse_args(["train", "--game", "sf"])


def test_status_subcommand():
    args = build_parser().parse_args(["status", "pong-20261124-180000"])
    assert (args.cmd, args.run_name) == ("status", "pong-20261124-180000")


def test_videos_command_is_dispatched(monkeypatch, tmp_path):
    import atari_rl.status as status

    from atari_rl.cli import main

    called = []
    monkeypatch.setattr(status, "fetch_videos", lambda run, dest: called.append((run, dest)) or [])
    assert main(["videos", "pong-1", "--dest", str(tmp_path)]) == 0
    assert called and called[0][0] == "pong-1"
