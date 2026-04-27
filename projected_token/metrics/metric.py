from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    @abstractmethod
    def __call__(self, original: str, rephrased: str) -> dict[str, Any]:
        pass
