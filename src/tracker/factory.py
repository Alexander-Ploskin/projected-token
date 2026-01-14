from __future__ import annotations
from typing import Any, Dict

from src.tracker.base import MultiTracker, NullTracker, Tracker
from src.tracker.console import ConsoleTracker


def build_tracker(logging_cfg: Dict[str, Any]) -> Tracker:
    backends = logging_cfg.get("backends", ["console"])
    trackers: list[Tracker] = []
    for b in backends:
        if b == "console":
            trackers.append(ConsoleTracker())
        else:
            raise ValueError(f"Unsupported tracker backend: {b} (mock only supports 'console')")
    if not trackers:
        return NullTracker()
    return MultiTracker(trackers)
