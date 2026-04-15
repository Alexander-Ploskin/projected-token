from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch


class Encoder(ABC):
    """Базовый класс для энкодеров (сжатие документов в векторы)."""
    
    @abstractmethod
    def encode(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Сжать документы в латентные векторы.
        
        Args:
            documents: Список текстов документов
            questions: Опционально список вопросов для query-aware сжатия
            
        Returns:
            Тензор формы [batch_size, hidden_dim] - латентные представления
        """
        pass
    
    @abstractmethod
    def encode_batch(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """Сжать документы в список латентных векторов (по одному на документ).
        
        Args:
            documents: Список текстов документов
            questions: Опционально список вопросов
            
        Returns:
            Список тензоров, каждый формы [hidden_dim]
        """
        pass
    
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Размерность латентного пространства."""
        pass