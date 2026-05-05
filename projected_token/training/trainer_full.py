import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModel

from projected_token.encoders.projector import FullFineTuneProjector
from projected_token.oscar_runtime import disable_transformers_allocator_warmup, configure_oscar_component_devices
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
        projector_lr: float | None = None,
        compressor_lr: float | None = None,
        temperature: float = 0.02,
        loss_name: str = "mnr",
        use_hard_negatives: bool = False,
        val_split: float = 0.1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/full",
        log_dir: str = "./logs/full",
        run_root: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        fallback_negative_strategy: str = "first_non_positive",
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
        disable_transformers_allocator_warmup()
        print(f"Loading OSCAR model: {oscar_model_name}")
        oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        configure_oscar_component_devices(oscar_model)
        
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
            fallback_negative_strategy=fallback_negative_strategy,
        )

        if loss_name == "mnr":
            loss_fn = get_loss_fn("mnr", scale=1.0 / temperature)
        elif loss_name == "infonce":
            loss_fn = get_loss_fn("infonce", temperature=temperature)
        elif loss_name == "triplet":
            loss_fn = get_loss_fn("triplet")
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")

        lr_projector = float(projector_lr) if projector_lr is not None else float(lr)
        lr_compressor = float(compressor_lr) if compressor_lr is not None else float(lr)
        projector_params = [p for p in self.full_model.projector.parameters() if p.requires_grad]
        compressor_params = [p for p in self.full_model.oscar_model.parameters() if p.requires_grad]
        param_groups = []
        if compressor_params:
            param_groups.append({"params": compressor_params, "lr": lr_compressor})
        if projector_params:
            param_groups.append({"params": projector_params, "lr": lr_projector})
        optimizer = torch.optim.AdamW(param_groups if param_groups else self.full_model.parameters(), lr=lr)

        super().__init__(
            model=self.full_model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            use_hard_negatives=use_hard_negatives,
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
        projector_lr=config.get("projector_lr"),
        compressor_lr=config.get("compressor_lr"),
        temperature=config.get("temperature", 0.02),
        loss_name=config.get("loss_name", "mnr"),
        use_hard_negatives=bool(config.get("use_hard_negatives", False)),
        val_split=float(config.get("val_split", 0.1)),
        device=config.get("device", "cuda:0"),
        output_dir=config.get("output_dir", "./checkpoints/full"),
        log_dir=config.get("log_dir", "./logs/full"),
        run_root=config.get("run_root"),
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
        fallback_negative_strategy=config.get("fallback_negative_strategy", "first_non_positive"),
    )