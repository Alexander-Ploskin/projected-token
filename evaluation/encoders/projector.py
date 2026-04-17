import torch
import torch.nn as nn
from typing import Optional, Literal
from peft import LoraConfig, get_peft_model, PeftModel


PoolerType = Literal["mean", "first", "last", "max", "mean_max"]


class BaseMEMProjector(nn.Module):
    """Базовый класс для проектора MEM-токенов OSCAR в эмбеддинги для поиска."""

    def __init__(
        self,
        hidden_dim: int = 2048,
        embed_dim: int = 768,
        pooler: PoolerType = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.pooler = pooler

    def pool(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        """Pool MEM-токены в один вектор.

        Args:
            mem_hiddens: [batch, num_mem_tokens, hidden_dim]

        Returns:
            [batch, hidden_dim] или [batch, hidden_dim*2] для mean_max
        """
        if self.pooler == "mean":
            return mem_hiddens.mean(dim=1)
        elif self.pooler == "first":
            return mem_hiddens[:, 0, :]
        elif self.pooler == "last":
            return mem_hiddens[:, -1, :]
        elif self.pooler == "max":
            return mem_hiddens.max(dim=1).values
        elif self.pooler == "mean_max":
            mean_emb = mem_hiddens.mean(dim=1)
            max_emb = mem_hiddens.max(dim=1).values
            return torch.cat([mean_emb, max_emb], dim=-1)
        else:
            raise ValueError(f"Unknown pooler: {self.pooler}")

    @property
    def output_dim(self) -> int:
        if self.pooler == "mean_max":
            return self.hidden_dim * 2
        return self.hidden_dim

    def forward(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        """Преобразовать MEM-токены в эмбеддинги для поиска.

        Args:
            mem_hiddens: [batch, num_mem_tokens, hidden_dim]

        Returns:
            [batch, embed_dim] нормализованные эмбеддинги
        """
        raise NotImplementedError


class MEMProjector(BaseMEMProjector):
    """MLP проектор для преобразования MEM-токенов в search-friendly эмбеддинги.

    Вариант A: Только MLP проектор, OSCAR заморожен.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        embed_dim: int = 768,
        pooler: PoolerType = "mean",
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_dim, embed_dim, pooler)

        actual_input_dim = hidden_dim * 2 if pooler == "mean_max" else hidden_dim

        layers = []
        in_dim = actual_input_dim

        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        if num_layers > 1:
            layers.append(nn.LayerNorm(embed_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(mem_hiddens)
        embeddings = self.mlp(pooled)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings


class LoRAMEMProjector(BaseMEMProjector):
    """LoRA проектор - MLP с LoRA адаптером.

    Вариант B: MLP + LoRA адаптер, OSCAR заморожен.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        embed_dim: int = 768,
        pooler: PoolerType = "mean",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__(hidden_dim, embed_dim, pooler)

        actual_input_dim = hidden_dim * 2 if pooler == "mean_max" else hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(actual_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(lora_dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["0", "2"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.mlp = get_peft_model(self.mlp, lora_config)

    def forward(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(mem_hiddens)
        embeddings = self.mlp(pooled)
        return embeddings

    def print_trainable_parameters(self):
        """Вывести количество обучаемых параметров."""
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable params: {trainable_params:,} | All params: {all_params:,} | Ratio: {trainable_params / all_params:.2%}")


class FullFineTuneProjector(BaseMEMProjector):
    """Полное дообучение: OSCAR с LoRA + проектор.

    Вариант C: OSCAR (с LoRA) + проектор, все обучается.
    """

    def __init__(
        self,
        oscar_model: nn.Module,
        embed_dim: int = 768,
        pooler: PoolerType = "mean",
        oscar_lora_r: int = 8,
        oscar_lora_alpha: int = 16,
        projector_num_layers: int = 2,
        projector_dropout: float = 0.1,
    ):
        hidden_dim = oscar_model.config.hidden_size

        super().__init__(hidden_dim, embed_dim, pooler)
        self.oscar_model = oscar_model
        self.projector_num_layers = projector_num_layers

        lora_config = LoraConfig(
            r=oscar_lora_r,
            lora_alpha=oscar_lora_alpha,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.oscar_model = get_peft_model(self.oscar_model, lora_config)

        actual_input_dim = hidden_dim * 2 if pooler == "mean_max" else hidden_dim
        in_dim = actual_input_dim

        layers = []
        for i in range(projector_num_layers):
            out_dim = embed_dim if i == projector_num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < projector_num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(projector_dropout))
            in_dim = out_dim

        if projector_num_layers > 1:
            layers.append(nn.LayerNorm(embed_dim))

        self.projector = nn.Sequential(*layers)

    def get_compressed_embeddings(self, documents: list[str]) -> torch.Tensor:
        """Получить MEM-эмбеддинги от OSCAR.

        Args:
            documents: список документов

        Returns:
            [batch, num_mem_tokens, hidden_dim]
        """
        with torch.inference_mode():
            mem_embeddings = self.oscar_model.compress_documents(documents=documents)
        return mem_embeddings

    def forward(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(mem_hiddens)
        embeddings = self.mlp(pooled)
        return embeddings

    def forward_full(self, documents: list[str]) -> torch.Tensor:
        """Полный forward: документы -> эмбеддинги для поиска.

        Args:
            documents: список документов

        Returns:
            [batch, embed_dim] нормализованные эмбеддинги
        """
        mem_hiddens = self.get_compressed_embeddings(documents)
        return self.forward(mem_hiddens)

    def print_trainable_parameters(self):
        """Вывести количество обучаемых параметров."""
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Trainable params: {trainable_params:,} | All params: {all_params:,} | Ratio: {trainable_params / all_params:.2%}")