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
        sp.add_argument("--algo", choices=["sb3", "scratch"], help="PPO implementation (default: ATARIRL_ALGO or sb3)")
        return sp

    with_overrides(sub.add_parser("train", help="train or resume a run"))
    ev = with_overrides(sub.add_parser("eval", help="evaluate a saved model (raw score) and record a video"))
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--episodes", type=int, default=5)
    ev.add_argument("--video", type=Path)
    sub.add_parser("smoke", help="2-env Pong run, ~2k steps, in a temp dir (honours ATARIRL_ALGO)")
    cp = sub.add_parser("cartpole", help="scratch PPO sanity check on CartPole-v1 (should reach ~500)")
    cp.add_argument("--steps", type=int, default=100_000)
    cp.add_argument("--n-envs", type=int, default=4)
    cp.add_argument("--seed", type=int, default=1)
    st = sub.add_parser("status", help="live run state from MLflow (health, eval, fps)")
    st.add_argument("run_name", help="run id = MLflow run name, e.g. pong-20261124-180000")
    vid = sub.add_parser("videos", help="download a run's eval replay videos from MLflow")
    vid.add_argument("run_name")
    vid.add_argument("--dest", type=Path, default=Path.home() / "Videos" / "rl-series")
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    overrides = {"game": args.game, "total_steps": args.steps, "n_envs": args.n_envs, "run_id": args.run_id,
                 "algo": args.algo}
    return Settings(**{k: v for k, v in overrides.items() if v is not None})


def _run_training(cfg: Settings) -> str:
    """ATARIRL_ALGO picks the PPO implementation; both honour the same run contract."""
    if cfg.algo == "scratch":
        from atari_rl.scratch import trainer
        from atari_rl.scratch.connectors import connector_for

        return trainer.run(cfg, connector_for(cfg))
    from atari_rl import train

    return train.run(cfg)


def _cmd_train(args: argparse.Namespace) -> int:
    status = _run_training(settings_from_args(args))
    print(status)
    return 0 if status != "critical" else 2


def _cmd_eval(args: argparse.Namespace) -> int:
    from atari_rl.scratch import checkpoint

    if checkpoint.is_scratch_checkpoint(args.model):
        return _eval_scratch(args, checkpoint.load(args.model))
    from stable_baselines3 import PPO

    from atari_rl.evaluate import run_eval

    model = PPO.load(args.model)
    print(run_eval(model, settings_from_args(args), args.episodes, video_path=args.video))
    return 0


def _eval_scratch(args: argparse.Namespace, ckpt) -> int:
    from atari_rl.scratch import evaluate
    from atari_rl.scratch.connectors import connector_for

    if args.game is None:  # the checkpoint knows which game it was trained on
        args.game = ckpt.meta.get("connector_config", {}).get("game")
    cfg = settings_from_args(args)
    print(evaluate.run_eval(ckpt.agent, connector_for(cfg), args.episodes, max_steps=cfg.eval_max_steps,
                            video_path=args.video, seed=cfg.seed + 10_000))
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Settings(game="pong", n_envs=2, total_steps=2048, eval_freq=1024, ckpt_freq=1024,
                       eval_episodes=1, eval_max_steps=300, runs_dir=Path(tmp), run_id="smoke")
        status = _run_training(cfg)
    print(status)
    return 0 if status == "done" else 2


def _cmd_cartpole(args: argparse.Namespace) -> int:
    from atari_rl.scratch.cartpole import train_cartpole

    result = train_cartpole(total_steps=args.steps, n_envs=args.n_envs, seed=args.seed)
    print(result)
    return 0 if result.eval_mean_return >= 450 else 2


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
    handlers = {"train": _cmd_train, "eval": _cmd_eval, "smoke": _cmd_smoke, "cartpole": _cmd_cartpole,
                "status": _cmd_status, "videos": _cmd_videos}
    return handlers[args.cmd](args)
