from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from atari_rl.config import Settings

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atari-rl", description="PPO on Atari Pong / Breakout (ale-py).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_overrides(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sp.add_argument("--game", choices=["pong", "breakout"])
        sp.add_argument("--steps", type=int, help="total timesteps (default: per game)")
        sp.add_argument("--n-envs", type=int)
        sp.add_argument("--run-id")
        return sp

    with_overrides(sub.add_parser("train", help="train or resume a run"))
    ev = with_overrides(sub.add_parser("eval", help="evaluate a saved model (raw score) and record a video"))
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--episodes", type=int, default=5)
    ev.add_argument("--video", type=Path)
    sub.add_parser("smoke", help="2-env Pong run, ~2k steps, in a temp dir")
    st = sub.add_parser("status", help="live run state from MLflow (health, eval, fps)")
    st.add_argument("run_name", help="run id = MLflow run name, e.g. pong-20261124-180000")
    vid = sub.add_parser("videos", help="download a run's eval replay videos from MLflow")
    vid.add_argument("run_name")
    vid.add_argument("--dest", type=Path, default=Path.home() / "Videos" / "rl-series")
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {"game": args.game, "total_steps": args.steps, "n_envs": args.n_envs, "run_id": args.run_id}
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def _cmd_train(args: argparse.Namespace) -> int:
    from atari_rl import train

    status = train.run(settings_from_args(args))
    print(status)
    return 0 if status != "critical" else 2


def _cmd_eval(args: argparse.Namespace) -> int:
    from stable_baselines3 import PPO

    from atari_rl.evaluate import run_eval

    model = PPO.load(args.model)
    print(run_eval(model, settings_from_args(args), args.episodes, video_path=args.video))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from atari_rl import train

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Settings(game="pong", n_envs=2, total_steps=2048, eval_freq=1024, ckpt_freq=1024,
                       eval_episodes=1, eval_max_steps=300, runs_dir=Path(tmp), run_id="smoke")
        status = train.run(cfg)
    print(status)
    return 0 if status == "done" else 2


def _cmd_status(args: argparse.Namespace) -> int:
    from atari_rl.status import fetch

    print(fetch(args.run_name))
    return 0


def _cmd_videos(args: argparse.Namespace) -> int:
    from atari_rl.status import fetch_videos

    new = fetch_videos(args.run_name, args.dest)
    print(f"{len(new)} new video(s) in {args.dest / args.run_name}")
    for p in new:
        print(f"  {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    handlers = {"train": _cmd_train, "eval": _cmd_eval, "smoke": _cmd_smoke, "status": _cmd_status}
    return handlers[args.cmd](args)
