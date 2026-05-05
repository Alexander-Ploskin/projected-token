import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoModel, AutoTokenizer
import os

from projected_token.encoders.projector import LoRAMEMProjector
from projected_token.training.trainer import BaseTrainer
from projected_token.training.dataset import create_dataloaders
from projected_token.training.losses import get_loss_fn


class LoRATrainer(BaseTrainer):
    """Trainer для варианта B: LoRA проектор.

    OSCAR заморожен, MLP проектор с LoRA адаптером обучается.
    """

    def __init__(
        self,
        oscar_model_name: str,
        embed_dim: int = 768,
        pooler: str = "mean",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        batch_size: int = 64,
        dataset_path: str = "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
        dataset_config: str = "triplet",
        dataset_split: str = "train",
        lr: float = 1e-4,
        temperature: float = 0.02,
        val_split: float = 0.1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/lora",
        log_dir: str = "./logs/lora",
        run_root: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ):
        """Инициализация.

        Args:
            oscar_model_name: путь к OSCAR модели
            embed_dim: размерность выходных эмбеддингов
            pooler: стратегия pooling
            lora_r: rank LoRA
            lora_alpha: alpha LoRA
            lora_dropout: dropout LoRA
            batch_size: размер батча
            lr: learning rate
            temperature: температура для loss
            device: устройство
            output_dir: директория для чекпоинтов
            log_dir: директория для логов
            max_train_samples: лимит train samples
            max_val_samples: лимит val samples
        """
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        
        # Disable vocab expansion warning
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False

        hidden_size = self.oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")

        self.projector = LoRAMEMProjector(
            hidden_dim=hidden_size,
            embed_dim=embed_dim,
            pooler=pooler,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        ).to(device=device, dtype=torch.bfloat16)

        print("LoRA Projector architecture:")
        self.projector.print_trainable_parameters()

        train_loader, val_loader = create_dataloaders(
            oscar_model=self.oscar_model,
            batch_size=batch_size,
            dataset_path=dataset_path,
            dataset_config=dataset_config,
            dataset_split=dataset_split,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            val_split=val_split,
            device=device,
        )

        loss_fn = get_loss_fn("mnr", scale=1.0 / temperature)
        optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr)

        super().__init__(
            model=self.projector,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            output_dir=output_dir,
            log_dir=log_dir,
            run_root=run_root,
        )

    def encode_documents(self, texts: list[str]) -> torch.Tensor:
        """Закодировать тексты через OSCAR + LoRA проектор (query-independent).

        Args:
            texts: список текстов (документы или запросы)

        Returns:
            [batch, embed_dim] эмбеддинги
        """
        with torch.inference_mode():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)

        embeddings = self.projector(mem_embeddings)
        return embeddings


def create_lora_trainer(config: dict) -> LoRATrainer:
    """Создать LoRA trainer из конфига.

    Args:
        config: словарь с конфигом

    Returns:
        LoRATrainer instance
    """
    return LoRATrainer(
        oscar_model_name=config["oscar_model_name"],
        embed_dim=config.get("embed_dim", 768),
        pooler=config.get("pooler", "mean"),
        lora_r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.1),
        batch_size=config.get("batch_size", 64),
        dataset_path=config.get("dataset_path", "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3"),
        dataset_config=config.get("dataset_config", "triplet"),
        dataset_split=config.get("dataset_split", "train"),
        lr=config.get("lr", 1e-4),
        temperature=config.get("temperature", 0.02),
        val_split=float(config.get("val_split", 0.1)),
        device=config.get("device", "cuda:0"),
        output_dir=config.get("output_dir", "./checkpoints/lora"),
        log_dir=config.get("log_dir", "./logs/lora"),
        run_root=config.get("run_root"),
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
    )