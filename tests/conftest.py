"""Shared fixtures.

- `clean_env`: strips ATARIRL_* vars so Settings tests see defaults (motivated by src/atari_rl/config.py env_prefix).
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("ATARIRL_") or k == "MLFLOW_TRACKING_URI":
            monkeypatch.delenv(k, raising=False)
