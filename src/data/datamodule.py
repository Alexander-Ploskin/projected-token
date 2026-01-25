from __future__ import annotations

from functools import partial
from torch.utils.data import DataLoader

from src.data.utils import load_json_dataset
from src.data.preprocessors.pretrain_encoder import encode_with_chat_format_pretrain
from src.data.collators import PretrainCollator


def _cfg_get(cfg_section, key: str, default):
    return cfg_section.get(key, default) if isinstance(cfg_section, dict) else getattr(cfg_section, key, default)


def build_pretrain_dataloaders(cfg, llm_tokenizer, retriever_tokenizer=None):
    raw = load_json_dataset(cfg.data.train_file, cfg.data.dev_file)

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
    )

    # 1) Map WITHOUT remove_columns (prevents losing newly-added columns on some dataset types). [web:258]
    ds = raw.map(
        encode_fn,
        batched=False,
        num_proc=cfg.data.preprocessing_num_workers,
        load_from_cache_file=False,  # keep for debugging; switch back later
        desc="Encoding pretrain data",
    )

    # 2) Now drop all original columns, keeping only what training needs.
    # DatasetDict.remove_columns applies to all splits.
    keep = {"xrag_input_ids", "xrag_labels", "retriever_input_text", "retriever_text", "retriever_inputs"}
    to_remove = [c for c in ds["train"].column_names if c not in keep]
    ds = ds.remove_columns(to_remove) 

    # 3) Only tensorize the tensor columns; keep retriever_input_text as Python objects.
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
    )

    dev_loader = None
    if "dev" in ds:
        dev_loader = DataLoader(
            ds["dev"],
            shuffle=False,
            batch_size=cfg.train.per_device_eval_batch_size,
            collate_fn=collate,
        )

    return train_loader, dev_loader
