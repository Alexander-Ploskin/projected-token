from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from transformers import get_scheduler, get_wsd_schedule

from src.distributed.utils import mean_across_processes


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

def build_scheduler(cfg, optimizer, num_training_steps: int):
    if cfg.train.num_warmup_steps is not None:
        warmup_steps = cfg.train.num_warmup_steps
    else:
        warmup_steps = int(num_training_steps * cfg.train.warmup_ratio)

    if cfg.train.lr_scheduler_type == "warmup_stable_decay":
        return get_wsd_schedule(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=cfg.train.num_decay_steps or 0,
            min_lr_ratio=cfg.train.min_lr_ratio or 0.1,
        )

    return get_scheduler(
        name=cfg.train.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

def save_projector_ckpt(model, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # model is unwrapped outside or inside caller; this only touches projector
    torch.save(model.projector.state_dict(), path)


def load_projector_checkpoint(model, path: str, *, map_location: str = "cpu") -> None:
    sd = torch.load(path, map_location=map_location)
    # strict=True here is good because you're loading exactly the projector module
    model.projector.load_state_dict(sd, strict=True)

def save_checkpoint(accelerator, model, tokenizer, output_dir: str, *, save_projector_only: bool = True, projector_ckpt_name: str = "projector.pt"):
    os.makedirs(output_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)

    if not save_projector_only:
        unwrapped.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # always (or conditional) save projector weights
    save_projector_ckpt(unwrapped, os.path.join(output_dir, projector_ckpt_name))

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
