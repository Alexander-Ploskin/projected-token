from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LoggingConfig:
    backends: list[str] = field(default_factory=lambda: ["console"])
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    log_every_steps: int = 50


@dataclass
class DistributedConfig:
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_flash_attn_2: bool = False
    xrag_token: str = "[XRAG]"
    freeze_llm: bool = True


@dataclass
class RetrieverConfig:
    retriever_name_or_path: Optional[str] = None
    freeze_retriever: bool = True
    max_length: int = 180


@dataclass
class ProjectorConfig:
    hidden_dim: int = 1024
    dropout: float = 0.0


@dataclass
class DataConfig:
    train_file: str = ""
    dev_file: Optional[str] = None
    use_summary_as_document: bool = False
    retriever_text_source: str = "text"  # "text" | "summary"
    max_train_samples: Optional[int] = None
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    max_seq_length: int = 336
    retrieval_embed_length: int = 1


@dataclass
class TrainConfig:
    seed: int = 1234
    output_dir: str = "./runs/pretrain"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2

    learning_rate: float = 6e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"  # mirror HF get_scheduler names
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None

    clip_grad_norm: float = 1.0
    checkpoint_every_steps: int = 500
    eval_every_steps: int = 500


@dataclass
class PretrainObjectiveConfig:
    alpha_nll: float = 1.0


@dataclass
class ExperimentConfig:
    task: str = "pretrain"

    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    objective: PretrainObjectiveConfig = field(default_factory=PretrainObjectiveConfig)

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    extra: Dict[str, Any] = field(default_factory=dict)
