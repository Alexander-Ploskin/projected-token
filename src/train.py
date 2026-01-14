from __future__ import annotations
import os
import math
import torch
import torch.nn.functional as F
from transformers import get_scheduler

from src.distributed.utils import mean_across_processes


def nll_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits: [B,T,V] predicts next token; shift so position t predicts labels at t+1
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

def build_scheduler(cfg, optimizer, num_training_steps: int):
    warmup_steps = int(num_training_steps * cfg.train.warmup_ratio)
    return get_scheduler(
        name=cfg.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

def save_checkpoint(accelerator, model, tokenizer, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)

    # Save HF model weights for LLM + projector (mock: assumes llm is HF model)
    unwrapped.llm.save_pretrained(out_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    tokenizer.save_pretrained(out_dir)

    # Save projector separately (handy for “projector-only” experiments later)
    accelerator.save(unwrapped.projector.state_dict(), os.path.join(out_dir, "projector.pt"))

@torch.no_grad()
def validate_pretrain_ppl(accelerator, xrag_model, retriever, dataloader) -> float:
    xrag_model.eval()

    total_nll = torch.tensor(0.0, device=accelerator.device)
    total_tokens = torch.tensor(0.0, device=accelerator.device)

    for batch in dataloader:
        retrieval_embeds = None
        if retriever is not None:
            retrieval_embeds = retriever.encode(
                batch["retriever_input_ids"], batch["retriever_attention_mask"]
            )

        out = xrag_model(
            input_ids=batch["xrag_input_ids"],
            attention_mask=batch["xrag_attention_mask"],
            retrieval_embeds=retrieval_embeds,
        )

        logits = out.logits          # [B,T,V]
        labels = batch["xrag_labels"]  # [B,T] with -100 ignore

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Sum NLL over valid tokens in this batch
        nll_sum = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )  # ignore_index behavior / reduction semantics

        num = (shift_labels != -100).sum()

        total_nll += nll_sum
        total_tokens += num

    # We need SUM across processes here (not mean), then compute ratio.
    total_nll = mean_across_processes(accelerator, total_nll) * accelerator.num_processes
    total_tokens = mean_across_processes(accelerator, total_tokens) * accelerator.num_processes

    xrag_model.train()
    return float(torch.exp(total_nll / total_tokens).item())  # perplexity definition

