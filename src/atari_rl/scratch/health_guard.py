"""Feeds the shared HealthMonitor (monitoring/health.py, unchanged) without SB3 callbacks.

Same duties as the SB3 path's HealthCallback:
- one HealthSnapshot per PPO iteration → HealthMonitor → health.json (atomic write) → on_report (MLflow tags)
- a watchdog thread: if no heartbeat arrives for stall_secs (a stuck emulator never finishes its rollout), it
  marks the run CRITICAL "stalled" and calls on_stall (which exits the process on the pod).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

from atari_rl.monitoring.health import HealthMonitor, HealthReport, HealthSnapshot

logger = logging.getLogger(__name__)


class HealthGuard:
    def __init__(self, monitor: HealthMonitor, out_path: Path, stall_secs: float | None,
                 on_stall: Callable[[HealthReport], None], on_report: Callable[[HealthReport], None] | None = None):
        self.monitor = monitor
        self.out_path = Path(out_path)
        self.stall_secs = stall_secs
        self.on_stall = on_stall
        self.on_report = on_report
        self.last_report: HealthReport | None = None
        self._last_progress = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def write(self, report: HealthReport) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_name(self.out_path.name + ".tmp")
        tmp.write_text(report.to_json())
        os.replace(tmp, self.out_path)  # readers (GCS sync, status) never see a half-written file
        self.last_report = report

    def update(self, snapshot: HealthSnapshot) -> HealthReport:
        self.heartbeat()
        report = self.monitor.update(snapshot)
        self.write(report)
        if self.on_report is not None:
            self.on_report(report)
        if report.status != "HEALTHY":
            logger.warning("health %s: %s", report.status, ", ".join(report.reasons))
        return report

    def mark_critical(self, reason: str) -> HealthReport:
        report = self.monitor.mark_critical(reason)
        self.write(report)
        return report

    def heartbeat(self) -> None:
        """The run is alive (called every env step and every eval step)."""
        self._last_progress = time.monotonic()

    def start(self) -> None:
        self.heartbeat()
        if self.stall_secs is not None:
            self._thread = threading.Thread(target=self._watch, daemon=True, name="health-watchdog")
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _watch(self) -> None:
        interval = min(5.0, self.stall_secs / 4)
        while not self._stop.wait(interval):
            if time.monotonic() - self._last_progress > self.stall_secs:
                report = self.mark_critical("stalled")
                self.on_stall(report)
                return
