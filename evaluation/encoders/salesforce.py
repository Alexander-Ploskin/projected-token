import torch
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from evaluation.encoders import Encoder


class SalesforceEncoder(Encoder):
    """Salesforce/e5-mistral энкодер для получения эмбеддингов.
    
    Использует SentenceTransformer для кодирования текстов.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
    ) -> None:
        self._device = torch.device(device)
        
        self._model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        
        self._latent_dim = self._model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Получить эмбеддинги документов или вопросов (батч).
        
        Args:
            documents: Список документов
            questions: Опционально вопросы - если переданы, кодируются как запросы
            
        Returns:
            Тензор [batch_size, latent_dim]
        """
        is_query = questions is not None
        texts = questions if is_query else documents
        
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        
        if not valid_texts:
            return torch.zeros(len(texts), self._latent_dim, device=self._device)
        
        # E5 требует префикс "query: " для запросов и "passage: " для документов
        if is_query:
            texts_with_prefix = [f"query: {q}" for q in valid_texts]
        else:
            texts_with_prefix = [f"passage: {doc}" for doc in valid_texts]
        
        embeddings = self._model.encode(
            texts_with_prefix,
            convert_to_tensor=True,
            device=self._device,
            show_progress_bar=False,
        )
        
        result = torch.zeros(len(texts), self._latent_dim, device=self._device)
        
        valid_indices = [i for i, t in enumerate(texts) if t and isinstance(t, str) and t.strip()]
        for i, idx in enumerate(valid_indices):
            if i < len(embeddings):
                result[idx] = embeddings[i]
        
        return result
    
    def encode_batch(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """Получить эмбеддинги документов или вопросов (список).
        
        Args:
            documents: Список документов
            questions: Опционально вопросы - если переданы, кодируются как запросы
            
        Returns:
            Список тензоров [latent_dim]
        """
        is_query = questions is not None
        texts = questions if is_query else documents
        
        valid_texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
        
        if not valid_texts:
            return [torch.zeros(self._latent_dim, device=self._device) for _ in texts]
        
        if is_query:
            texts_with_prefix = [f"query: {q}" for q in valid_texts]
        else:
            texts_with_prefix = [f"passage: {doc}" for doc in valid_texts]
        
        embeddings = self._model.encode(
            texts_with_prefix,
            convert_to_tensor=True,
            device=self._device,
            show_progress_bar=False,
        )
        
        results = []
        valid_idx = 0
        for t in texts:
            if t and isinstance(t, str) and t.strip() and valid_idx < len(embeddings):
                results.append(embeddings[valid_idx])
                valid_idx += 1
            else:
                results.append(torch.zeros(self._latent_dim, device=self._device))
        
        return results
    
    @property
    def latent_dim(self) -> int:
        return self._latent_dim