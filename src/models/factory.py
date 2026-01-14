from __future__ import annotations

import inspect
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.projectors.mlp import TwoLayerMLPProjector
from src.models.xrag import XRAGForCausalLM


def build_tokenizer(model_name_or_path: str, xrag_token: str) -> tuple:
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<pad>"})
    tok.add_tokens([xrag_token])
    xrag_token_id = tok.convert_tokens_to_ids(xrag_token)
    return tok, xrag_token_id


def _flash_attn_kwargs(use_flash_attn_2: bool) -> dict:
    """
    Returns kwargs for AutoModelForCausalLM.from_pretrained depending on installed transformers.
    Prefers attn_implementation="flash_attention_2" when supported.
    """
    if not use_flash_attn_2:
        return {}

    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    params = sig.parameters

    if "attn_implementation" in params:
        return {"attn_implementation": "flash_attention_2"}  # preferred modern API [web:102]
    if "use_flash_attention_2" in params:
        return {"use_flash_attention_2": True}  # older API (not always supported)
    # transformers too old / doesn’t expose FA2 knobs => ignore
    return {}


def build_xrag_model(
    model_name_or_path: str,
    xrag_token_id: int,
    retriever_embed_dim: int,
    bridge_hidden_dim: int,
    bridge_dropout: float,
    use_flash_attn_2: bool,
    torch_dtype,
) -> XRAGForCausalLM:
    load_kwargs = {
        "torch_dtype": torch_dtype,
        **_flash_attn_kwargs(use_flash_attn_2),
    }

    llm = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)

    llm_hidden = int(llm.config.hidden_size)
    projector = TwoLayerMLPProjector(
        in_dim=retriever_embed_dim,
        out_dim=llm_hidden,
        hidden_dim=bridge_hidden_dim,
        dropout=bridge_dropout,
    )
    return XRAGForCausalLM(llm=llm, projector=projector, xrag_token_id=xrag_token_id)
