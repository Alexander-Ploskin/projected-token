import torch
from typing import List, Optional
from pathlib import Path
from transformers import AutoModel

from projected_token.encoders import Encoder
from projected_token.encoders.projector import MEMProjector, DistillationProjector


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
                - "flatten": разворачивание всех mem-токенов
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
        elif aggregation == "flatten":
            self._latent_dim = dummy_output.shape[1] * dummy_output.shape[-1]
        else:
            self._latent_dim = dummy_output.shape[-1]
    
    def _aggregate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Агрегация тензора [batch, num_tokens, hidden] -> [batch, hidden]
        
        Args:
            tensor: Тензор формы [batch_size, num_tokens, hidden_dim]
            
        Returns:
            Тензор формы [batch_size, hidden_dim], [batch_size, hidden_dim*2] для mean_max
            или [batch_size, num_mem_tokens*hidden_dim] для flatten
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
        elif self._aggregation == "flatten":
            return tensor.reshape(tensor.shape[0], -1)
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


class OscarProjectorEncoder(Encoder):
    """OSCAR энкодер с проектором для retrieval.
    
    Использует query-independent сжатие (только документы, без вопросов).
    Применяет проектор для преобразования MEM-токенов в эмбеддинги для поиска.
    """
    
    def __init__(
        self,
        oscar_model_name: str,
        projector_path: str = None,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        embed_dim: int = 768,
        pooler: str = "mean",
        num_layers: int = 2,
        dropout: float = 0.1,
        projector_hidden_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            oscar_model_name: Путь к OSCAR модели
            projector_path: Путь к чекпоинту проектора (опционально)
            device: Устройство
            torch_dtype: Тип данных
            trust_remote_code: Доверять удаленному коду
            embed_dim: Размерность выходных эмбеддингов
            pooler: Стратегия агрегации mem-токенов
            num_layers: Количество слоев в проекторе
            dropout: Dropout в проекторе
            projector_hidden_dim: Скрытая размерность в MLP проекторе
        """
        self._device = torch.device(device)
        
        device_map = device if device != "cpu" else "cpu"
        
        self._oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
        ).eval()
        
        if hasattr(self._oscar_model, 'compr') and hasattr(self._oscar_model.compr, 'config'):
            self._oscar_model.compr.config.mean_resizing = False
        
        dummy_output = self._oscar_model.compress_documents(["test"])
        hidden_size = dummy_output.shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")
        
        # Load checkpoint first to get config
        checkpoint = None
        checkpoint_config = None
        if projector_path:
            print(f"Loading projector from {projector_path}")
            checkpoint = torch.load(projector_path, weights_only=False, map_location=device)
            checkpoint_config = checkpoint.get("config")
            
            # Try to load from config file if not in checkpoint
            if not checkpoint_config:
                config_file = Path(projector_path).parent / "config.json"
                if config_file.exists():
                    import json
                    with open(config_file) as f:
                        checkpoint_config = json.load(f)
                    print(f"Loaded config from file: {config_file}")
            
            if checkpoint_config:
                print(f"Projector config: {checkpoint_config}")
                embed_dim = checkpoint_config.get("embed_dim", embed_dim)
                pooler = checkpoint_config.get("pooler", pooler)
                num_layers = checkpoint_config.get("num_layers", num_layers)
                dropout = checkpoint_config.get("dropout", dropout)
                projector_hidden_dim = checkpoint_config.get("projector_hidden_dim", projector_hidden_dim)
                oscar_hidden_dim = checkpoint_config.get("oscar_hidden_dim")
        
        # Detect if this is a DistillationProjector (has oscar_hidden_dim in config)
        use_distillation = oscar_hidden_dim is not None
        
        if use_distillation:
            print(f"Using DistillationProjector (oscar_hidden_dim={oscar_hidden_dim})")
            self._projector = DistillationProjector(
                oscar_hidden_dim=oscar_hidden_dim,
                embed_dim=embed_dim,
                hidden_dim=projector_hidden_dim or 8192,
            ).to(device=device, dtype=torch.bfloat16)
        else:
            self._projector = MEMProjector(
                hidden_dim=hidden_size,
                embed_dim=embed_dim,
                pooler=pooler,
                num_layers=num_layers,
                dropout=dropout,
                projector_hidden_dim=projector_hidden_dim,
            ).to(device=device, dtype=torch.bfloat16)
        
        if projector_path and checkpoint is not None:
            if "model_state_dict" in checkpoint:
                self._projector.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded projector weights from step {checkpoint.get('step', 'unknown')}")
            else:
                print("Warning: checkpoint does not contain model_state_dict, using random weights")
        
        self._projector.eval()
        
        self._embed_dim = embed_dim
        self._oscar_model_name = oscar_model_name
        self._projector_path = projector_path
    
    def _aggregate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Агрегация тензора [batch, num_tokens, hidden] -> [batch, hidden]"""
        return tensor.mean(dim=1)
    
    def encode(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Сжать документы в эмбеддинги для поиска через проектор.
        
        Query-independent: вопросы игнорируются, только документы.
        
        Args:
            documents: Список документов
            questions: Опционально (игнорируется для query-independent)
            
        Returns:
            Тензор [batch_size, embed_dim]
        """
        valid_docs = [doc for doc in documents if doc.strip()]
        
        if not valid_docs:
            return torch.zeros(len(documents), self._embed_dim, device=self._device)
        
        with torch.no_grad():
            compressed = self._oscar_model.compress_documents(
                documents=valid_docs,
                questions=None,
            )
            compressed = compressed.detach()
            embeddings = self._projector(compressed)
            embeddings = embeddings.detach()
        
        result = torch.zeros(len(documents), self._embed_dim, device=self._device)
        
        valid_indices = [i for i, doc in enumerate(documents) if doc.strip()]
        for i, idx in enumerate(valid_indices):
            if i < len(embeddings):
                result[idx] = embeddings[i]
        
        return result
    
    def encode_batch(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        """Сжать документы в список эмбеддингов.
        
        Args:
            documents: Список документов
            questions: Опционально (игнорируется)
            
        Returns:
            Список тензоров [embed_dim]
        """
        valid_docs = [doc for doc in documents if doc.strip()]
        
        if not valid_docs:
            return [torch.zeros(self._embed_dim, device=self._device) for _ in documents]
        
        with torch.no_grad():
            compressed = self._oscar_model.compress_documents(
                documents=valid_docs,
                questions=None,
            )
            compressed = compressed.detach()
            embeddings = self._projector(compressed)
            embeddings = embeddings.detach()
        
        results = []
        valid_idx = 0
        for doc in documents:
            if doc.strip() and valid_idx < len(embeddings):
                results.append(embeddings[valid_idx])
                valid_idx += 1
            else:
                results.append(torch.zeros(self._embed_dim, device=self._device))
        
        return results
    
    @property
    def latent_dim(self) -> int:
        return self._embed_dim
    
    @property
    def aggregation(self) -> str:
        return "projector"