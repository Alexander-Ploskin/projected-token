import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel

from projected_token.encoders.projector import FullFineTuneProjector
from projected_token.training.trainer import BaseTrainer
from projected_token.training.dataset import create_dataloaders
from projected_token.training.losses import get_loss_fn


class FullFineTuneTrainer(BaseTrainer):
    """Trainer для варианта C: Full fine-tune.

    OSCAR (с LoRA) + проектор обучаются вместе.
    """

    @staticmethod
    def _resolve_transformer_layers(module: nn.Module):
        candidates = [module]
        if hasattr(module, "base_model"):
            candidates.append(module.base_model)
        expanded = []
        for cand in candidates:
            expanded.append(cand)
            if hasattr(cand, "model"):
                expanded.append(cand.model)
        for cand in expanded:
            layers = getattr(cand, "layers", None)
            if layers is not None:
                return layers
            encoder = getattr(cand, "encoder", None)
            if encoder is not None and hasattr(encoder, "layer"):
                return encoder.layer
        return None

    def _apply_partial_unfreeze(self, unfreeze_last_n_layers: int) -> None:
        if unfreeze_last_n_layers <= 0:
            return
        for param in self.full_model.oscar_model.parameters():
            param.requires_grad = False
        layers = self._resolve_transformer_layers(self.full_model.oscar_model)
        if layers is None:
            print("Warning: could not resolve transformer layers for partial unfreeze.")
        else:
            for layer in list(layers)[-unfreeze_last_n_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        for param in self.full_model.projector.parameters():
            param.requires_grad = True

    def __init__(
        self,
        oscar_model_name: str,
        embed_dim: int = 768,
        pooler: str = "mean",
        apply_lora: bool = True,
        oscar_lora_r: int = 8,
        oscar_lora_alpha: int = 16,
        oscar_lora_dropout: float = 0.1,
        unfreeze_last_n_layers: int = 0,
        projector_num_layers: int = 2,
        projector_dropout: float = 0.1,
        batch_size: int = 64,
        dataset_path: str = "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
        dataset_config: str = "triplet",
        dataset_split: str = "train",
        lr: float = 1e-4,
        temperature: float = 0.02,
        val_split: float = 0.1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/full",
        log_dir: str = "./logs/full",
        run_root: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ):
        """Инициализация.

        Args:
            oscar_model_name: путь к OSCAR модели
            embed_dim: размерность выходных эмбеддингов
            pooler: стратегия pooling
            oscar_lora_r: rank LoRA для OSCAR
            oscar_lora_alpha: alpha LoRA для OSCAR
            projector_num_layers: количество слоев в проекторе
            projector_dropout: dropout проектора
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
        oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        
        # Disable vocab expansion warning
        if hasattr(oscar_model, 'compr') and hasattr(oscar_model.compr, 'config'):
            oscar_model.compr.config.mean_resizing = False

        hidden_size = oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")

        self.full_model = FullFineTuneProjector(
            oscar_model=oscar_model,
            embed_dim=embed_dim,
            pooler=pooler,
            apply_lora=apply_lora,
            oscar_lora_r=oscar_lora_r,
            oscar_lora_alpha=oscar_lora_alpha,
            oscar_lora_dropout=oscar_lora_dropout,
            projector_num_layers=projector_num_layers,
            projector_dropout=projector_dropout,
        ).to(device=device, dtype=torch.bfloat16)

        self._apply_partial_unfreeze(unfreeze_last_n_layers)

        print("Full Fine-Tune Model:")
        self.full_model.print_trainable_parameters()

        train_loader, val_loader = create_dataloaders(
            oscar_model=self.full_model.oscar_model,
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
        optimizer = torch.optim.AdamW(self.full_model.parameters(), lr=lr)

        super().__init__(
            model=self.full_model,
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
        """Закодировать тексты через OSCAR + проектор (query-independent).

        Args:
            texts: список текстов (документы или запросы)

        Returns:
            [batch, embed_dim] эмбеддинги
        """
        doc_embeddings = self.full_model.forward_full(texts)
        return doc_embeddings


def create_full_trainer(config: dict) -> FullFineTuneTrainer:
    """Создать Full fine-tune trainer из конфига.

    Args:
        config: словарь с конфигом

    Returns:
        FullFineTuneTrainer instance
    """
    return FullFineTuneTrainer(
        oscar_model_name=config["oscar_model_name"],
        embed_dim=config.get("embed_dim", 768),
        pooler=config.get("pooler", "mean"),
        oscar_lora_r=config.get("oscar_lora_r", 8),
        oscar_lora_alpha=config.get("oscar_lora_alpha", 16),
        oscar_lora_dropout=config.get("oscar_lora_dropout", 0.1),
        apply_lora=config.get("apply_lora", True),
        unfreeze_last_n_layers=config.get("unfreeze_last_n_layers", 0),
        projector_num_layers=config.get("projector_num_layers", 2),
        projector_dropout=config.get("projector_dropout", 0.1),
        batch_size=config.get("batch_size", 64),
        dataset_path=config.get("dataset_path", "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3"),
        dataset_config=config.get("dataset_config", "triplet"),
        dataset_split=config.get("dataset_split", "train"),
        lr=config.get("lr", 1e-4),
        temperature=config.get("temperature", 0.02),
        val_split=float(config.get("val_split", 0.1)),
        device=config.get("device", "cuda:0"),
        output_dir=config.get("output_dir", "./checkpoints/full"),
        log_dir=config.get("log_dir", "./logs/full"),
        run_root=config.get("run_root"),
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
    )