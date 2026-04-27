from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    @abstractmethod
    def __call__(self, document: str, prompt_template: str, model_args: dict = {}) -> str:
        pass
