from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Tracker(ABC):
    @abstractmethod
    def log_config(self, cfg: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        pass

    @abstractmethod
    def log_text(self, text: str) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class NullTracker(Tracker):
    def log_config(self, cfg: Dict[str, Any]) -> None:
        return None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        return None

    def log_text(self, text: str) -> None:
        return None

    def close(self) -> None:
        return None


class MultiTracker(Tracker):
    def __init__(self, trackers: list[Tracker]):
        self.trackers = trackers

    def log_config(self, cfg: Dict[str, Any]) -> None:
        for t in self.trackers:
            t.log_config(cfg)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        for t in self.trackers:
            t.log_metrics(metrics, step)

    def log_text(self, text: str) -> None:
        for t in self.trackers:
            t.log_text(text)

    def close(self) -> None:
        for t in self.trackers:
            t.close()
