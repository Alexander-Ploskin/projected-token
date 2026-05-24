import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from peft import LoraConfig, get_peft_model


PoolerType = Literal["mean", "first", "last", "max", "mean_max", "first_last", "flatten"]


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
            [batch, hidden_dim] или [batch, hidden_dim*2] для mean_max/first_last,
            или [batch, num_mem_tokens * hidden_dim] для flatten
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
        elif self.pooler == "first_last":
            return torch.cat([mem_hiddens[:, 0, :], mem_hiddens[:, -1, :]], dim=-1)
        elif self.pooler == "flatten":
            return mem_hiddens.view(mem_hiddens.size(0), -1)
        else:
            raise ValueError(f"Unknown pooler: {self.pooler}")

    @property
    def output_dim(self) -> int:
        if self.pooler in {"mean_max", "first_last"}:
            return self.hidden_dim * 2
        elif self.pooler == "flatten":
            return self.hidden_dim * 8  # 8 mem tokens
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
        projector_hidden_dim: Optional[int] = None,
    ):
        super().__init__(hidden_dim, embed_dim, pooler)

        if projector_hidden_dim is None:
            projector_hidden_dim = hidden_dim

        # Calculate actual input dimension based on pooler
        if pooler in {"mean_max", "first_last"}:
            actual_input_dim = hidden_dim * 2
        elif pooler == "flatten":
            actual_input_dim = hidden_dim * 8  # 8 mem tokens
        else:
            actual_input_dim = hidden_dim

        layers = []
        in_dim = actual_input_dim

        for i in range(num_layers):
            if num_layers == 1:
                out_dim = embed_dim
            elif i == 0:
                out_dim = projector_hidden_dim
            elif i == num_layers - 1:
                out_dim = embed_dim
            else:
                out_dim = projector_hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        if num_layers > 1:
            layers.append(nn.LayerNorm(embed_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, mem_hiddens: torch.Tensor, mode: str = "doc") -> torch.Tensor:
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

        # Calculate actual input dimension based on pooler
        if pooler in {"mean_max", "first_last"}:
            actual_input_dim = hidden_dim * 2
        elif pooler == "flatten":
            actual_input_dim = hidden_dim * 8  # 8 mem tokens
        else:
            actual_input_dim = hidden_dim

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
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
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
        apply_lora: bool = True,
        oscar_lora_r: int = 8,
        oscar_lora_alpha: int = 16,
        oscar_lora_dropout: float = 0.1,
        projector_num_layers: int = 2,
        projector_dropout: float = 0.1,
    ):
        hidden_dim = oscar_model.config.hidden_size

        super().__init__(hidden_dim, embed_dim, pooler)
        self.oscar_model = oscar_model
        self.projector_num_layers = projector_num_layers

        if apply_lora:
            lora_config = LoraConfig(
                r=oscar_lora_r,
                lora_alpha=oscar_lora_alpha,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=oscar_lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.oscar_model = get_peft_model(self.oscar_model, lora_config)

        # Calculate actual input dimension based on pooler
        if pooler == "mean_max":
            actual_input_dim = hidden_dim * 2
        elif pooler == "flatten":
            actual_input_dim = hidden_dim * 8  # 8 mem tokens
        else:
            actual_input_dim = hidden_dim
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
        # During training we need gradients through OSCAR (LoRA adapters).
        if self.training:
            return self.oscar_model.compress_documents(documents=documents)
        with torch.inference_mode():
            return self.oscar_model.compress_documents(documents=documents)

    def forward(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(mem_hiddens)
        embeddings = self.projector(pooled)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
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


class DistillationProjector(nn.Module):
    """MLP проектор для дистилляции из SFR-Embedding-Mistral.
    
    Архитектура:
        - Input: flatten(8 * hidden_dim) = 28672 (для OSCAR flatten pooler)
        - Hidden: hidden_dim (например, 8192)
        - Output: embed_dim (4096 для SFR-Mistral)
        - Активация: GELU
        - Нормализация: LayerNorm
    """
    
    def __init__(
        self,
        oscar_hidden_dim: int = 3584,
        embed_dim: int = 4096,
        hidden_dim: int = 8192,
        num_layers: int = 2,
        use_normalize: bool = True,
    ):
        super().__init__()
        
        self.oscar_hidden_dim = oscar_hidden_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_normalize = use_normalize
        
        # Input: 8 * oscar_hidden_dim = 28672 (для flatten pooler)
        input_dim = 8 * oscar_hidden_dim
        
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = embed_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if mem_hiddens.dim() == 3:
            mem_hiddens = mem_hiddens.view(mem_hiddens.size(0), -1)
        
        embeddings = self.mlp(mem_hiddens)
        
        # Normalize output
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings


class DualHeadMEMProjector(BaseMEMProjector):
    """Conservative dual-head MLP projector with a shared pooled trunk."""

    def __init__(
        self,
        hidden_dim: int = 2048,
        embed_dim: int = 768,
        pooler: PoolerType = "flatten",
        trunk_hidden_dim: Optional[int] = None,
        head_hidden_dim: Optional[int] = None,
        trunk_layers: int = 1,
        head_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_dim, embed_dim, pooler)
        trunk_hidden_dim = trunk_hidden_dim or hidden_dim
        head_hidden_dim = head_hidden_dim or trunk_hidden_dim

        def _input_dim() -> int:
            if pooler in {"mean_max", "first_last"}:
                return hidden_dim * 2
            if pooler == "flatten":
                return hidden_dim * 8
            return hidden_dim

        layers: list[nn.Module] = []
        in_dim = _input_dim()
        for _ in range(max(1, int(trunk_layers))):
            layers.append(nn.Linear(in_dim, trunk_hidden_dim))
            layers.append(nn.LayerNorm(trunk_hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = trunk_hidden_dim
        self.trunk = nn.Sequential(*layers)

        def _make_head() -> nn.Sequential:
            head: list[nn.Module] = []
            in_head = trunk_hidden_dim
            for i in range(max(1, int(head_layers))):
                out_dim = embed_dim if i == max(1, int(head_layers)) - 1 else head_hidden_dim
                head.append(nn.Linear(in_head, out_dim))
                if out_dim != embed_dim:
                    head.append(nn.GELU())
                    head.append(nn.Dropout(dropout))
                    head.append(nn.LayerNorm(out_dim))
                in_head = out_dim
            head.append(nn.LayerNorm(embed_dim))
            return nn.Sequential(*head)

        self.query_head = _make_head()
        self.doc_head = _make_head()

    def forward(self, mem_hiddens: torch.Tensor, mode: str = "doc") -> torch.Tensor:
        pooled = self.pool(mem_hiddens)
        features = self.trunk(pooled)
        embeddings = self.query_head(features) if mode == "query" else self.doc_head(features)
        return F.normalize(embeddings, p=2, dim=-1)


class TokenAwareDualProjector(nn.Module):
    """Token-aware dual-head projector for asymmetric query/document encoding."""

    def __init__(
        self,
        hidden_dim: int = 3584,
        embed_dim: int = 768,
        num_mem_tokens: int = 8,
        attn_dim: int = 1536,
        num_attn_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        projector_hidden_dim: int = 4096,
        head_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embed_dim = int(embed_dim)
        self.num_mem_tokens = int(num_mem_tokens)
        self.attn_dim = int(attn_dim)
        self.num_attn_layers = int(num_attn_layers)
        self.num_heads = int(num_heads)

        self.input_ln = nn.LayerNorm(self.hidden_dim)
        self.token_stem = nn.Linear(self.hidden_dim, self.attn_dim)
        self.mem_positions = nn.Parameter(torch.zeros(1, self.num_mem_tokens, self.attn_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attn_dim,
            nhead=self.num_heads,
            dim_feedforward=max(self.attn_dim * 4, self.attn_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attn_trunk = nn.TransformerEncoder(encoder_layer, num_layers=self.num_attn_layers)
        self.pool_query = nn.Parameter(torch.zeros(self.attn_dim))
        self.pool_proj = nn.Linear(self.attn_dim * 3, self.attn_dim)

        def _make_head() -> nn.Sequential:
            layers: list[nn.Module] = []
            in_dim = self.attn_dim
            head_layers_local = max(1, int(head_layers))
            for i in range(head_layers_local):
                out_dim = self.embed_dim if i == head_layers_local - 1 else projector_hidden_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < head_layers_local - 1:
                    layers.append(nn.GELU())
                    layers.append(nn.Dropout(dropout))
                    layers.append(nn.LayerNorm(out_dim))
                in_dim = out_dim
            layers.append(nn.LayerNorm(self.embed_dim))
            return nn.Sequential(*layers)

        self.query_head = _make_head()
        self.doc_head = _make_head()
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mem_positions, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    def _pooled_features(self, mem_hiddens: torch.Tensor) -> torch.Tensor:
        x = self.input_ln(mem_hiddens)
        x = self.token_stem(x)
        if x.size(1) == self.num_mem_tokens:
            x = x + self.mem_positions
        x = self.attn_trunk(x)
        scores = torch.matmul(x, self.pool_query)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pooled = torch.sum(x * weights, dim=1)
        mean_pooled = x.mean(dim=1)
        max_pooled = x.max(dim=1).values
        pooled = torch.cat([attn_pooled, mean_pooled, max_pooled], dim=-1)
        return self.pool_proj(pooled)

    def forward(self, mem_hiddens: torch.Tensor, mode: str = "doc") -> torch.Tensor:
        pooled = self._pooled_features(mem_hiddens)
        if mode == "query":
            embeddings = self.query_head(pooled)
        else:
            embeddings = self.doc_head(pooled)
        embeddings = embeddings * self.logit_scale.clamp(min=0.01, max=100.0)
        return F.normalize(embeddings, p=2, dim=-1)