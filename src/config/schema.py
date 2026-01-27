from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------- logging / distributed ----------------

@dataclass
class LoggingConfig:
    backends: list[str] = field(default_factory=lambda: ["console"])
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    log_every_steps: int = 50

@dataclass
class DistributedConfig:
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    gradient_accumulation_steps: int = 1

# ---------------- model / retriever / projector ----------------

@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    use_flash_attn_2: bool = False
    xrag_token: str = "[XRAG]"
    freeze_llm: bool = True
    torch_dtype: str = "bf16"  # "bf16" | "fp16" | "auto"

@dataclass
class RetrieverConfig:
    retriever_name_or_path: Optional[str] = None
    freeze_retriever: bool = True
    max_length: int = 180
    min_length: int = 30
    crop_strategy: str = "uniform"  # "uniform" | "log_uniform"

@dataclass
class ProjectorConfig:
    hidden_dim: int = 1024
    dropout: float = 0.0
    # NEW: load only projector weights from here (recommended path for finetune)
    checkpoint: Optional[str] = None

# ---------------- data ----------------

@dataclass
class DataConfig:
    train_file: str = ""
    dev_file: Optional[str] = None

    # HuggingFace dataset loading (if dataset_name is set, it takes precedence)
    dataset_name: Optional[str] = None  # e.g., "HuggingFaceFW/finewiki"
    dataset_subset: Optional[str] = None  # e.g., "en" for finewiki
    dataset_split_train: str = "train"  # HF dataset split name for training
    dataset_split_dev: Optional[str] = "validation"  # HF dataset split name for dev/validation
    streaming: bool = False  # Use streaming mode for large HF datasets
    dataset_cache_dir: Optional[str] = None  # Custom cache directory for HF datasets
    dataset_revision: Optional[str] = None  # Pin specific dataset version/commit
    
    # pretrain (document-based) knobs
    use_summary_as_document: bool = False
    retriever_text_source: str = "text"  # "text" | "summary"
    max_train_samples: Optional[int] = None
    preprocessing_num_workers: int = 4
    overwrite_cache: bool = False
    max_seq_length: int = 336
    retrieval_embed_length: int = 1

    # finetune (messages-only compression) knobs
    finetune_context_source: str = "all_user"  # "first_user" | "all_user"
    replace_user_with_xrag: bool = True
    xrag_user_prefix: str = "Please answer this question: "

# ---------------- train / objective ----------------

@dataclass
class TrainConfig:
    seed: int = 1234
    output_dir: str = "./runs/pretrain"
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2

    learning_rate: float = 6e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "linear"
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None

    clip_grad_norm: float = 1.0
    checkpoint_every_steps: int = 500
    eval_every_steps: int = 500
    save_total_limit: Optional[int] = None

    save_projector_only: bool = False
    projector_ckpt_name: str = "projector.pt"

@dataclass
class PretrainObjectiveConfig:
    alpha_nll: float = 1.0

@dataclass
class FinetuneObjectiveConfig:
    alpha_nll: float = 1.0
    alpha_kl: float = 0.0
    kl_temperature: float = 1.0

@dataclass
class GenerationConfig:
    max_new_tokens: int = 30
    do_sample: bool = True
    temperature: float = 0.1

@dataclass
class EvalConfig:
    task: str = "eval_popqa"  # "eval_popqa" | future tasks
    
    dataset_path: str = "data/popqa/popqa.parquet"
    output_path: str = "./runs/popqa_eval.json"
    limit: Optional[int] = None  # null = full dataset
    
    use_context: bool = True
    max_context_chars: int = 15000
    max_prompt_chars: int = 12000
    
    prompts_dir: str = "./prompts"
    system_prompt_basic: str = "basic.txt"
    system_prompt_context: str = "context.hbs"
    
    device: str = "cuda"
    torch_dtype: str = "fp16"
    
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class ExperimentConfig:
    task: str = "pretrain"  # "pretrain" | "finetune" | "eval_popqa"

    model: ModelConfig = field(default_factory=ModelConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    eval: EvalConfig = field(default_factory=EvalConfig)
    
    # Keep existing for compatibility
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    objective: Dict[str, Any] = field(default_factory=dict)

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    extra: Dict[str, Any] = field(default_factory=dict)
