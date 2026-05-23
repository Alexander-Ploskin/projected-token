from __future__ import annotations

import os
from types import MethodType
from typing import Any

import torch


def disable_transformers_allocator_warmup() -> None:
    """Disable Transformers CUDA allocator warmup to avoid OSCAR adapter OOM spikes.

    OSCAR remote code loads decoder LoRA adapters via `load_adapter(...)`, which
    triggers `transformers.modeling_utils.caching_allocator_warmup` and can request
    very large temporary allocations on 24GB GPUs.
    """
    flag = os.getenv("OSCAR_DISABLE_ADAPTER_WARMUP", "1").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return

    try:
        import transformers.modeling_utils as modeling_utils
    except Exception:
        return

    if getattr(modeling_utils, "_projected_token_warmup_disabled", False):
        return

    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if original is None:
        return

    modeling_utils._projected_token_original_warmup = original
    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    modeling_utils._projected_token_warmup_disabled = True


def disable_resume_download_passthrough() -> None:
    """Patch AutoModel loaders to ignore legacy `resume_download` kwargs.

    Some remote OSCAR model code forwards `resume_download` to nested model
    loaders. Recent Transformers versions can propagate this keyword all the
    way into model constructors, causing:
      `TypeError: ... got an unexpected keyword argument 'resume_download'`.
    """
    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except Exception:
        return

    if getattr(AutoModel, "_projected_token_resume_download_patched", False):
        return

    orig_auto_from_pretrained = AutoModel.from_pretrained
    orig_causal_from_pretrained = AutoModelForCausalLM.from_pretrained

    @classmethod
    def _patched_auto_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.pop("resume_download", None)
        return orig_auto_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def _patched_causal_from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        kwargs.pop("resume_download", None)
        return orig_causal_from_pretrained.__func__(cls, pretrained_model_name_or_path, *model_args, **kwargs)

    AutoModel.from_pretrained = _patched_auto_from_pretrained
    AutoModelForCausalLM.from_pretrained = _patched_causal_from_pretrained
    AutoModel._projected_token_resume_download_patched = True


def _module_device(module: Any) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")


def _patch_compress_documents_to_compressor_device(model: Any) -> None:
    if getattr(model, "_projected_token_compress_patch", False):
        return

    def _compress_documents(self, documents: list[str], questions: list[str] | None = None) -> torch.Tensor:
        if questions is None:
            input_encoder = self.prepare_encoder_inputs(documents, max_length=128)
        else:
            input_encoder = self.prepare_encoder_inputs(documents, max_length=128, q_texts=questions)

        if getattr(self, "compr", None) is not None:
            target_device = _module_device(self.compr)
        else:
            target_device = self.decoder.device

        enc_input_ids = input_encoder["input_ids"].to(target_device)
        attention_mask = input_encoder["attention_mask"].to(target_device)
        return self.compress(enc_input_ids=enc_input_ids, enc_attention_mask=attention_mask)

    model._projected_token_original_compress_documents = model.compress_documents
    model.compress_documents = MethodType(_compress_documents, model)
    model._projected_token_compress_patch = True


def configure_oscar_component_devices(
    model: Any,
    *,
    decoder_device: str | None = None,
    compressor_device: str | None = None,
) -> None:
    """Optionally place decoder/compressor on different GPUs.

    Defaults can be provided via env vars:
    - `OSCAR_DECODER_DEVICE` (e.g. `cuda:1`)
    - `OSCAR_COMPRESSOR_DEVICE` (e.g. `cuda:0`)
    """
    decoder_target = decoder_device or os.getenv("OSCAR_DECODER_DEVICE")
    compressor_target = compressor_device or os.getenv("OSCAR_COMPRESSOR_DEVICE")

    if decoder_target and hasattr(model, "decoder"):
        model.decoder.to(decoder_target)

    if compressor_target and getattr(model, "compr", None) is not None:
        model.compr.to(compressor_target)
        _patch_compress_documents_to_compressor_device(model)
