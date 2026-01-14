from __future__ import annotations
import json
from typing import Any, Dict

from .base import Tracker


class ConsoleTracker(Tracker):
    def log_config(self, cfg: Dict[str, Any]) -> None:
        print("[config]")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        kv = " ".join([f"{k}={v:.6g}" for k, v in metrics.items()])
        print(f"[step={step}] {kv}")

    def log_text(self, text: str) -> None:
        print(text)

    def close(self) -> None:
        return None
