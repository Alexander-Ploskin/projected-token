from __future__ import annotations
from typing import Any, Dict

from src.tracker.base import MultiTracker, NullTracker, Tracker
from src.tracker.console import ConsoleTracker


def build_tracker(logging_cfg: Dict[str, Any], log_dir: Optional[str] = None) -> Tracker:
    backends = logging_cfg.get("backends", ["console", "tensorboard"])
    trackers: list[Tracker] = []
    
    for b in backends:
        if b == "console":
            trackers.append(ConsoleTracker())
        elif b == "tensorboard":
            from .tensorboard import TensorboardTracker
            if log_dir is None:
                # Fallback to output_dir if not provided in config
                log_dir = logging_cfg.get("log_dir", "./runs/default")
            trackers.append(TensorboardTracker(log_dir))
        else:
            raise ValueError(f"Unsupported tracker backend: {b} (supports 'console', 'tensorboard')")
    
    if not trackers:
        return NullTracker()
    return MultiTracker(trackers)
