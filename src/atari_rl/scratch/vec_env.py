"""Run N envs in parallel and batch their observations; stack the last k frames.

    main process (policy, one batched forward pass)          worker processes (one emulator each)
        actions (N,) ──── pipe.send("step", a_i) ──────────►  obs, r, term, trunc, info = env.step(a_i)
                                                              if the episode ended: info["final_obs"] = obs
                                                                                    obs = env.reset()
        batched obs ◄──── pipe.recv() ────────────────────────

Auto-reset happens inside the worker, so the main loop never waits for a reset round-trip. The observation
returned for a finished env is already the first one of its *next* episode; the true last observation is in
info["final_obs"] (needed to bootstrap a truncated episode, see buffer.py).

Frame stacking runs in the main process on the whole batch: workers send one 84×84 frame, not four.
Env factories must be picklable (use functools.partial over module-level callables): workers start with
"forkserver", which is safe next to torch's threads.
"""
from __future__ import annotations

import multiprocessing as mp
from functools import partial
from multiprocessing.connection import Connection
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np

from atari_rl.scratch.connectors.base import Connector, ImageSpec

EnvFn = Callable[[], gym.Env]
Obs = dict[str, np.ndarray]
StepResult = tuple[Obs, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]


def _stack(obs_list: Sequence[dict[str, np.ndarray]]) -> Obs:
    """List of per-env dict obs → one dict of (N, ...) arrays."""
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _step_and_autoreset(env: gym.Env, action) -> tuple:
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        info["final_obs"] = obs
        obs, _ = env.reset()
    return obs, reward, terminated, truncated, info


def _worker(remote: Connection, parent_remote: Connection, env_fn: EnvFn) -> None:
    parent_remote.close()  # the parent's end; closing our copy lets the parent see EOF if we die
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(_step_and_autoreset(env, data))
            elif cmd == "reset":
                remote.send(env.reset(seed=data))
            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()  # an exception escapes after this: the process dies and the parent gets EOFError


class SubprocVecEnv:
    """One subprocess per env. A dead worker surfaces as EOFError / BrokenPipeError in the main process."""

    def __init__(self, env_fns: Sequence[EnvFn], start_method: str = "forkserver"):
        ctx = mp.get_context(start_method)
        self.num_envs = len(env_fns)
        pipes = [ctx.Pipe() for _ in env_fns]
        self.remotes = [parent for parent, _ in pipes]
        self.processes = []
        for (parent, child), fn in zip(pipes, env_fns):
            p = ctx.Process(target=_worker, args=(child, parent, fn), daemon=True)
            p.start()
            child.close()  # only the worker holds the child end now
            self.processes.append(p)
        self.closed = False

    def reset(self, seed: int | None = None) -> Obs:
        for i, remote in enumerate(self.remotes):
            remote.send(("reset", None if seed is None else seed + i))
        return _stack([remote.recv()[0] for remote in self.remotes])

    def step(self, actions: np.ndarray) -> StepResult:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]  # all envs step in parallel; we wait for all
        obs, rewards, terminated, truncated, infos = zip(*results)
        return (_stack(obs), np.asarray(rewards, np.float32), np.asarray(terminated, bool),
                np.asarray(truncated, bool), list(infos))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


class InProcessVecEnv:
    """Same interface, envs stepped one after another in this process (eval, tests, debugging)."""

    def __init__(self, env_fns: Sequence[EnvFn]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self, seed: int | None = None) -> Obs:
        return _stack([env.reset(seed=None if seed is None else seed + i)[0] for i, env in enumerate(self.envs)])

    def step(self, actions: np.ndarray) -> StepResult:
        obs, rewards, terminated, truncated, infos = zip(*(_step_and_autoreset(e, a) for e, a in zip(self.envs, actions)))
        return (_stack(obs), np.asarray(rewards, np.float32), np.asarray(terminated, bool),
                np.asarray(truncated, bool), list(infos))

    def close(self) -> None:
        for env in self.envs:
            env.close()


class FrameStack:
    """Keeps the last n_stack frames of every image key: (N, H, W) frames → (N, n_stack, H, W), oldest first.

    A new episode starts from an all-zero stack (SB3 VecFrameStack convention), so the network can tell
    "game just started" from "nothing moved".
    """

    def __init__(self, venv: SubprocVecEnv | InProcessVecEnv, n_stack: int, image_keys: Sequence[str]):
        self.venv, self.n_stack, self.image_keys = venv, n_stack, list(image_keys)
        self.num_envs = venv.num_envs
        self.stacks: dict[str, np.ndarray] = {}

    def _observe(self, obs: Obs) -> Obs:
        out = dict(obs)
        for k in self.image_keys:
            out[k] = self.stacks[k].copy()  # copy: callers keep old observations around
        return out

    def reset(self, seed: int | None = None) -> Obs:
        obs = self.venv.reset(seed)
        for k in self.image_keys:
            frame = obs[k]
            self.stacks[k] = np.zeros((frame.shape[0], self.n_stack, *frame.shape[1:]), frame.dtype)
            self.stacks[k][:, -1] = frame
        return self._observe(obs)

    def step(self, actions: np.ndarray) -> StepResult:
        obs, rewards, terminated, truncated, infos = self.venv.step(actions)
        done = terminated | truncated
        for i in np.flatnonzero(done):  # rebuild the finished episode's last *stacked* observation
            final = dict(infos[i]["final_obs"])
            for k in self.image_keys:
                final[k] = np.concatenate([self.stacks[k][i, 1:], final[k][None]])
            infos[i]["final_obs"] = final
        for k in self.image_keys:
            stack = self.stacks[k]
            stack[:, :-1] = stack[:, 1:]  # drop the oldest frame
            stack[done] = 0  # finished envs start a fresh episode
            stack[:, -1] = obs[k]
        return self._observe(obs), rewards, terminated, truncated, infos

    def close(self) -> None:
        self.venv.close()


def make_vec_env(connector: Connector, n_envs: int, seed: int | None = None, *, eval: bool = False,
                 in_process: bool = False) -> SubprocVecEnv | InProcessVecEnv | FrameStack:
    """n_envs copies of the connector's env (+ frame stacking of image keys when connector.frame_stack > 1).

    `seed` is not applied here; pass it to the first reset() (env i gets seed + i).
    """
    fns = [partial(connector.make_env, rank, eval=eval) for rank in range(n_envs)]
    venv = InProcessVecEnv(fns) if in_process else SubprocVecEnv(fns)
    image_keys = [k for k, s in connector.observation_spec.items() if isinstance(s, ImageSpec)]
    if connector.frame_stack > 1 and image_keys:
        return FrameStack(venv, connector.frame_stack, image_keys)
    return venv
