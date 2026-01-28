from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader

from src.data.preprocessors.pretrain_encoder import encode_with_chat_format_pretrain
from src.data.preprocessors.finetune_encoder import encode_with_chat_format_finetune
from src.data.collators import PretrainCollator, FinetuneCollator
# Load dataset with auto-detection (HuggingFace or JSONl)
from src.data.utils import load_dataset_auto, validate_dataset_structure


def _cfg_get(cfg_section, key: str, default):
    return cfg_section.get(key, default) if isinstance(cfg_section, dict) else getattr(cfg_section, key, default)


def _keep_only(ds, keep: set[str]):
    to_remove = [c for c in ds["train"].column_names if c not in keep]
    return ds.remove_columns(to_remove)


def build_pretrain_dataloaders(cfg, llm_tokenizer, retriever_tokenizer=None):
    raw = load_dataset_auto(
        train_file=_cfg_get(cfg.data, "train_file", None) or "",
        dev_file=_cfg_get(cfg.data, "dev_file", None),
        dataset_name=_cfg_get(cfg.data, "dataset_name", None),
        dataset_subset=_cfg_get(cfg.data, "dataset_subset", None),
        dataset_split_train=_cfg_get(cfg.data, "dataset_split_train", "train"),
        dataset_split_dev=_cfg_get(cfg.data, "dataset_split_dev", "validation"),
        streaming=_cfg_get(cfg.data, "streaming", False),
        dataset_cache_dir=_cfg_get(cfg.data, "dataset_cache_dir", None),
        dataset_revision=_cfg_get(cfg.data, "dataset_revision", None),
    )
    
    # Validate dataset structure (will auto-detect streaming and skip detailed validation)
    validate_dataset_structure(raw)

    # Encoder uses per-example RNG; prefer stable "id" if present, else you can switch to with_indices=True.
    rng_seed = _cfg_get(cfg.train, "seed", 52)
    rng_key = _cfg_get(cfg.data, "rng_key", "id")

    encode_fn = partial(
        encode_with_chat_format_pretrain,
        tokenizer=llm_tokenizer,
        max_seq_length=cfg.data.max_seq_length,
        max_length=cfg.retriever.max_length,
        min_length=cfg.retriever.min_length,
        crop_strategy=cfg.retriever.crop_strategy,
        retrieval_embed_length=_cfg_get(cfg.data, "retrieval_embed_length", 1),
        xrag_token=_cfg_get(cfg.model, "xrag_token", "[XRAG]"),
        retriever_text_source=_cfg_get(cfg.data, "retriever_text_source", "text"),
        rng_seed=rng_seed,
        rng_key=rng_key,
    )

    # If your pretrain JSONL has no "id", you can set cfg.data.rng_key="__index__"
    # and use with_indices=True (see note below). [web:671]
    use_indices = (rng_key == "__index__")

    if use_indices:
        def encode_fn_with_idx(ex, idx):
            ex = dict(ex)
            ex["__index__"] = idx
            return encode_fn(ex)
        ds = raw.map(
            encode_fn_with_idx,
            with_indices=True,             # supported by datasets.map [web:671]
            batched=False,
            num_proc=cfg.data.preprocessing_num_workers,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Encoding pretrain data",
        )
    else:
        ds = raw.map(
            encode_fn,
            batched=False,
            num_proc=cfg.data.preprocessing_num_workers,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Encoding pretrain data",
        )

    keep = {"xrag_input_ids", "xrag_labels", "retriever_input_text"}
    ds = _keep_only(ds, keep)

    ds = ds.with_format("torch", columns=["xrag_input_ids", "xrag_labels"], output_all_columns=True)

    collate = PretrainCollator(
        llm_tokenizer=llm_tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retriever_max_length=cfg.retriever.max_length,
    )

    train_loader = DataLoader(
        ds["train"],
        shuffle=True,
        batch_size=cfg.train.per_device_train_batch_size,
        collate_fn=collate,
        num_workers=_cfg_get(cfg.data, "dataloader_num_workers", 0),
        pin_memory=True,
    )

    dev_loader = None
    if "dev" in ds:
        dev_loader = DataLoader(
            ds["dev"],
            shuffle=False,
            batch_size=cfg.train.per_device_eval_batch_size,
            collate_fn=collate,
            num_workers=_cfg_get(cfg.data, "dataloader_num_workers", 0),
            pin_memory=True,
        )

    return train_loader, dev_loader


def build_finetune_dataloaders(cfg, llm_tokenizer, retriever_tokenizer=None):
    """
    Finetune dataset format (as you showed):
      {"id": ..., "task_type": ..., "messages": [...]}

    Encoder will:
      - compress question/options into retriever_input_text (optional)
      - replace first user content with [XRAG]
      - train to generate assistant answer
    """
    raw = load_dataset_auto(
        train_file=_cfg_get(cfg.data, "train_file", None) or "",
        dev_file=_cfg_get(cfg.data, "dev_file", None),
        dataset_name=_cfg_get(cfg.data, "dataset_name", None),
        dataset_subset=_cfg_get(cfg.data, "dataset_subset", None),
        dataset_split_train=_cfg_get(cfg.data, "dataset_split_train", "train"),
        dataset_split_dev=_cfg_get(cfg.data, "dataset_split_dev", "validation"),
        streaming=_cfg_get(cfg.data, "streaming", False),
        dataset_cache_dir=_cfg_get(cfg.data, "dataset_cache_dir", None),
        dataset_revision=_cfg_get(cfg.data, "dataset_revision", None),
    )
    
    # Validate dataset structure (will auto-detect streaming and skip detailed validation)
    validate_dataset_structure(raw)

    rng_seed = _cfg_get(cfg.train, "seed", 13)
    rng_key = _cfg_get(cfg.data, "rng_key", "id")
    use_indices = (rng_key == "__index__")

    return_teacher = cfg.objective.get("alpha_kl", 0.0) > 0.0

    encode_fn = partial(
        encode_with_chat_format_finetune,
        tokenizer=llm_tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        max_seq_length=cfg.data.max_seq_length,
        retrieval_embed_length=_cfg_get(cfg.data, "retrieval_embed_length", 1),
        xrag_token=_cfg_get(cfg.model, "xrag_token", "[XRAG]"),
        context_source=_cfg_get(cfg.data, "finetune_context_source", "all_user"),
        replace_user_with_xrag=_cfg_get(cfg.data, "replace_user_with_xrag", True),
        xrag_user_prefix=_cfg_get(cfg.data, "xrag_user_prefix", "Please answer this question: "),
        use_retriever_embed=not (retriever_tokenizer is None),
        retriever_min_length=cfg.retriever.min_length,
        retriever_max_length=cfg.retriever.max_length,
        retriever_crop_strategy=cfg.retriever.crop_strategy,
        rng_seed=rng_seed,
        return_teacher=return_teacher,
        teacher_user_prefix="",
    )

    if use_indices:
        def encode_fn_with_idx(ex, idx):
            ex = dict(ex)
            ex["__index__"] = idx
            # finetune encoder uses example["id"] in RNG; if you want, you can teach it to use __index__
            if "id" not in ex or ex["id"] is None:
                ex["id"] = str(idx)
            return encode_fn(ex)

        ds = raw.map(
            encode_fn_with_idx,
            with_indices=True,            # supported by datasets.map [web:671]
            batched=False,
            num_proc=cfg.data.preprocessing_num_workers,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Encoding finetune data",
        )
    else:
        ds = raw.map(
            encode_fn,
            batched=False,
            num_proc=cfg.data.preprocessing_num_workers,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Encoding finetune data",
        )

    keep = {"xrag_input_ids", "xrag_labels"}
    if return_teacher:
        keep |= {"input_ids", "labels"}
    if retriever_tokenizer is not None:
        keep.add("retriever_input_text")

    ds = _keep_only(ds, keep)

    cols = ["xrag_input_ids", "xrag_labels"]
    if return_teacher:
        cols += ["input_ids", "labels"]
    ds = ds.with_format("torch", columns=cols, output_all_columns=True)

    collate = FinetuneCollator(
        llm_tokenizer=llm_tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retriever_max_length=cfg.retriever.max_length,
    )

    train_loader = DataLoader(
        ds["train"],
        shuffle=True,
        batch_size=cfg.train.per_device_train_batch_size,
        collate_fn=collate,
        num_workers=_cfg_get(cfg.data, "dataloader_num_workers", 0),
        pin_memory=True,
    )

    dev_loader = None
    if "dev" in ds:
        dev_loader = DataLoader(
            ds["dev"],
            shuffle=False,
            batch_size=cfg.train.per_device_eval_batch_size,
            collate_fn=collate,
            num_workers=_cfg_get(cfg.data, "dataloader_num_workers", 0),
            pin_memory=True,
        )

    return train_loader, dev_loader
