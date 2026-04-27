from abc import ABC, abstractmethod
from typing import Iterator, Any


class Dataset(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "Dataset":
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        pass
