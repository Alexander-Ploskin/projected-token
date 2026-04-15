import torch
from typing import List, Optional
from transformers import AutoModel

from evaluation.encoders import Encoder


class OscarEncoder(Encoder):
    """OSCAR энкодер - сжимает документы в латентные векторы.
    
    Использует метод compress_documents из OSCAR модели для получения
    латентного представления документа.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        aggregation: str = "mean",
    ) -> None:
        """
        Args:
            model_name_or_path: Путь к модели
            device: Устройство
            torch_dtype: Тип данных
            trust_remote_code: Доверять удаленному коду
            aggregation: Стратегия агрегации mem-токенов:
                - "mean": усреднение (default)
                - "first": первый mem-токен
                - "last": последний mem-токен
                - "max": максимум по каждому измерению
                - "mean_max": конкатенация mean и max
        """
        self._device = torch.device(device)
        self._aggregation = aggregation
        
        device_map = device if device != "cpu" else "cpu"
        
        self._model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        ).eval()
        
        dummy_output = self._model.compress_documents(["test"])
        
        if aggregation == "mean_max":
            self._latent_dim = dummy_output.shape[-1] * 2
        else:
            self._latent_dim = dummy_output.shape[-1]
    
    def _aggregate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Агрегация тензора [batch, num_tokens, hidden] -> [batch, hidden]
        
        Args:
            tensor: Тензор формы [batch_size, num_tokens, hidden_dim]
            
        Returns:
            Тензор формы [batch_size, hidden_dim] или [batch_size, hidden_dim*2] для mean_max
        """
        if self._aggregation == "mean":
            return tensor.mean(dim=1)
        elif self._aggregation == "first":
            return tensor[:, 0, :]
        elif self._aggregation == "last":
            return tensor[:, -1, :]
        elif self._aggregation == "max":
            return tensor.max(dim=1).values
        elif self._aggregation == "mean_max":
            mean_emb = tensor.mean(dim=1)
            max_emb = tensor.max(dim=1).values
            return torch.cat([mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unknown aggregation: {self._aggregation}")
    
    def encode(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Сжать документы в латентные векторы (батч).
        
        Args:
            documents: Список документов
            questions: Опционально вопросы для query-aware сжатия
            
        Returns:
            Тензор [batch_size, latent_dim]
        """
        valid_docs = [doc for doc in documents if doc.strip()]
        
        if not valid_docs:
            return torch.zeros(len(documents), self._latent_dim, device=self._device)
        
        qa_questions = questions if questions else None
        
        with torch.inference_mode():
            compressed = self._model.compress_documents(
                documents=valid_docs,
                questions=qa_questions,
            )
        
        # compressed: [batch_size, num_mem_tokens, hidden_dim]
        aggregated = self._aggregate(compressed)
        
        result = torch.zeros(len(documents), self._latent_dim, device=self._device)
        
        valid_indices = [i for i, doc in enumerate(documents) if doc.strip()]
        for i, idx in enumerate(valid_indices):
            if i < len(aggregated):
                result[idx] = aggregated[i]
        
        return result
    
    def encode_batch(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """Сжать документы в список векторов.
        
        Args:
            documents: Список документов
            questions: Опционально вопросы
            
        Returns:
            Список тензоров [latent_dim]
        """
        valid_docs = [doc for doc in documents if doc.strip()]
        
        if not valid_docs:
            return [torch.zeros(self._latent_dim, device=self._device) for _ in documents]
        
        qa_questions = questions if questions else None
        
        with torch.inference_mode():
            compressed = self._model.compress_documents(
                documents=valid_docs,
                questions=qa_questions,
            )
        
        aggregated = self._aggregate(compressed)
        
        results = []
        valid_idx = 0
        for doc in documents:
            if doc.strip() and valid_idx < len(aggregated):
                results.append(aggregated[valid_idx])
                valid_idx += 1
            else:
                results.append(torch.zeros(self._latent_dim, device=self._device))
        
        return results
    
    @property
    def latent_dim(self) -> int:
        return self._latent_dim
    
    @property
    def aggregation(self) -> str:
        return self._aggregation