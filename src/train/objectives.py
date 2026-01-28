from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class LossBundle:
    loss: "torch.Tensor"                 # total scalar used for backward
    logs: dict[str, float]               # already detached floats (per-step)

def nll_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # logits: [B,T,V] predicts next token; shift so position t predicts labels at t+1
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    
    # We need both mean (for backward) and sum (for exact PPL)
    l_flat = logits.view(-1, logits.size(-1))
    t_flat = labels.view(-1)
    
    loss_mean = F.cross_entropy(l_flat, t_flat, ignore_index=-100, reduction="mean")
    loss_sum = F.cross_entropy(l_flat, t_flat, ignore_index=-100, reduction="sum")
    ntokens = (t_flat != -100).sum()
    
    return loss_mean, loss_sum, ntokens

def kl_div_from_logits(
    *,
    teacher_logits: torch.Tensor,   # [B,Tt,V]
    student_logits: torch.Tensor,   # [B,Ts,V]
    labels_mask: torch.Tensor,      # [B,Ts] bool, True where we want KL (assistant)
    temperature: float,
) -> torch.Tensor:
    # Align to student time axis; teacher may differ in length, so crop to min
    T_common = min(teacher_logits.size(1), student_logits.size(1))
    teacher_logits = teacher_logits[:, :T_common, :]
    student_logits = student_logits[:, :T_common, :]
    labels_mask = labels_mask[:, :T_common]

    T = float(temperature)
    t_logp = F.log_softmax(teacher_logits / T, dim=-1)
    s_logp = F.log_softmax(student_logits / T, dim=-1)

    # Per-token KL: sum over vocab => [B,T]
    kl_tok = F.kl_div(s_logp, t_logp.exp(), reduction="none").sum(dim=-1) * (T * T)  # [B,T] [web:866]

    # Mask + normalize by number of selected tokens
    denom = labels_mask.float().sum().clamp_min(1.0)
    return (kl_tok * labels_mask.float()).sum() / denom

class BaseObjective:
    def compute(
        self,
        *,
        cfg,
        acc,
        model,
        batch: dict[str, Any],
        student_outputs,
        teacher_outputs=None,
    ) -> LossBundle:
        raise NotImplementedError

class PretrainObjective(BaseObjective):
    def compute(self, *, cfg, acc, model, batch, student_outputs, teacher_outputs=None) -> LossBundle:
        nll, nll_sum, ntokens = nll_loss(student_outputs.logits, batch["xrag_labels"])
        loss = nll
        return LossBundle(
            loss=loss, 
            logs={
                "nll": float(nll.detach().float().item()), 
                "loss": float(loss.detach().float().item()),
                "nll_sum": float(nll_sum.detach().float().item()),
                "ntokens": float(ntokens.detach().float().item()),
            }
        )

class FinetuneObjective(BaseObjective):
    def compute(self, *, cfg, acc, model, batch, student_outputs, teacher_outputs=None) -> LossBundle:
        loss = None
        logs: dict[str, float] = {}

        if cfg.objective.get("alpha_nll", 0.0) > 0:
            nll, nll_sum, ntokens = nll_loss(student_outputs.logits, batch["xrag_labels"])
            loss = cfg.objective["alpha_nll"] * nll
            logs["nll"] = float(nll.detach().float().item())
            logs["nll_sum"] = float(nll_sum.detach().float().item())
            logs["ntokens"] = float(ntokens.detach().float().item())

        if cfg.objective["alpha_kl"] and cfg.objective["alpha_kl"] > 0:
            assert teacher_outputs is not None
            mask = (batch["xrag_labels"] != -100)
            kl = kl_div_from_logits(
                teacher_logits=teacher_outputs.logits,
                student_logits=student_outputs.logits,
                labels_mask=mask,
                temperature=cfg.objective["kl_temperature"],
            )

            loss = (loss + cfg.objective["alpha_kl"] * kl) if loss is not None else (cfg.objective["alpha_kl"] * kl)
            logs["kl"] = float(kl.detach().float().item())

        assert loss is not None
        logs["loss"] = float(loss.detach().float().item())
        return LossBundle(loss=loss, logs=logs)
