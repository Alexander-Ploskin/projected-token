from __future__ import annotations
from abc import ABC, abstractmethod


class Runner(ABC):
    @abstractmethod
    def run(self) -> None:
        pass
