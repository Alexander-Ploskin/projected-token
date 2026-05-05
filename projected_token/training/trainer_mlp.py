import torch
from typing import Optional
from transformers import AutoModel

from projected_token.encoders.projector import MEMProjector
from projected_token.oscar_runtime import disable_transformers_allocator_warmup, configure_oscar_component_devices
from projected_token.training.trainer import BaseTrainer
from projected_token.training.dataset import create_dataloaders
from projected_token.training.losses import get_loss_fn


class MLPTrainer(BaseTrainer):
    """Trainer для варианта A: MLP проектор.

    OSCAR заморожен, только проектор обучается.
    """

    def __init__(
        self,
        oscar_model_name: str,
        embed_dim: int = 768,
        hidden_dim: int = 2048,
        pooler: str = "mean",
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 64,
        dataset_path: str = "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
        dataset_config: str = "triplet",
        dataset_split: str = "train",
        init_projector_checkpoint: Optional[str] = None,
        lr: float = 1e-4,
        temperature: float = 0.02,
        loss_name: str = "mnr",
        use_hard_negatives: bool = False,
        val_split: float = 0.1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/mlp",
        log_dir: str = "./logs/mlp",
        run_root: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        fallback_negative_strategy: str = "first_non_positive",
    ):
        """Инициализация.

        Args:
            oscar_model_name: путь к OSCAR модели
            embed_dim: размерность выходных эмбеддингов
            hidden_dim: скрытая размерность OSCAR
            pooler: стратегия pooling
            num_layers: количество слоев в MLP
            dropout: dropout
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
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        configure_oscar_component_devices(self.oscar_model)
        
        # Disable vocab expansion warning for compressor
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False

        hidden_size = self.oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")

        self.projector = MEMProjector(
            hidden_dim=hidden_size,
            embed_dim=embed_dim,
            pooler=pooler,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device=device, dtype=torch.bfloat16)

        if init_projector_checkpoint:
            state = torch.load(init_projector_checkpoint, map_location=device)
            state_dict = state.get("model_state_dict", state)
            self.projector.load_state_dict(state_dict, strict=False)
            print(f"Loaded initial projector checkpoint: {init_projector_checkpoint}")

        print("Projector architecture:")
        print(self.projector)

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
        optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr)

        self.projector_config = {
            "hidden_dim": hidden_size,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "num_layers": num_layers,
            "dropout": dropout,
            "dataset_path": dataset_path,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "init_projector_checkpoint": init_projector_checkpoint,
            "loss_name": loss_name,
            "use_hard_negatives": use_hard_negatives,
            "fallback_negative_strategy": fallback_negative_strategy,
        }

        super().__init__(
            model=self.projector,
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

        self.model_config = self.projector_config

    def encode_documents(self, texts: list[str]) -> torch.Tensor:
        """Закодировать тексты через OSCAR + проектор (query-independent).

        Args:
            texts: список текстов (документы или запросы)

        Returns:
            [batch, embed_dim] эмбеддинги
        """
        with torch.inference_mode():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)

        embeddings = self.projector(mem_embeddings)
        return embeddings


def create_mlp_trainer(config: dict) -> MLPTrainer:
    """Создать MLP trainer из конфига.

    Args:
        config: словарь с конфигом

    Returns:
        MLPTrainer instance
    """
    return MLPTrainer(
        oscar_model_name=config["oscar_model_name"],
        embed_dim=config.get("embed_dim", 768),
        hidden_dim=config.get("hidden_dim", 2048),
        pooler=config.get("pooler", "mean"),
        num_layers=config.get("num_layers", 2),
        dropout=float(config.get("dropout", 0.1)),
        batch_size=int(config.get("batch_size", 64)),
        dataset_path=config.get("dataset_path", "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3"),
        dataset_config=config.get("dataset_config", "triplet"),
        dataset_split=config.get("dataset_split", "train"),
        init_projector_checkpoint=config.get("init_projector_checkpoint"),
        lr=float(config.get("lr", 1e-4)),
        temperature=float(config.get("temperature", 0.02)),
        loss_name=config.get("loss_name", "mnr"),
        use_hard_negatives=bool(config.get("use_hard_negatives", False)),
        val_split=float(config.get("val_split", 0.1)),
        device=config.get("device", "cuda:0"),
        output_dir=config.get("output_dir", "./checkpoints/mlp"),
        log_dir=config.get("log_dir", "./logs/mlp"),
        run_root=config.get("run_root"),
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
        fallback_negative_strategy=config.get("fallback_negative_strategy", "first_non_positive"),
    )