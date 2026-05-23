#!/usr/bin/env python3
"""Train OSCAR+projector with explicit query/document BGE teacher targets."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel

from projected_token.artifacts import write_config_lock
from projected_token.config import load_yaml
from projected_token.encoders.projector import MEMProjector, DualHeadMEMProjector, TokenAwareDualProjector
from projected_token.retrieval.beir import evaluate_beir
from projected_token.oscar_runtime import (
    disable_resume_download_passthrough,
    disable_transformers_allocator_warmup,
    configure_oscar_component_devices,
)


def _resolve_h5_paths(paths_or_patterns: Union[str, Sequence[str]]) -> list[str]:
    raw_paths = [paths_or_patterns] if isinstance(paths_or_patterns, str) else list(paths_or_patterns)
    resolved: list[str] = []
    for raw in raw_paths:
        expanded = os.path.expandvars(os.path.expanduser(str(raw)))
        path = Path(expanded)
        if path.is_dir():
            matches = sorted(str(p) for p in path.glob("*.h5"))
        elif glob.has_magic(expanded):
            matches = sorted(glob.glob(expanded))
        else:
            matches = [expanded]
        resolved.extend(matches)
    deduped = list(dict.fromkeys(resolved))
    missing = [p for p in deduped if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Teacher embedding file(s) not found: {missing}")
    if not deduped:
        raise FileNotFoundError(f"No teacher embedding .h5 files matched: {paths_or_patterns}")
    return deduped


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class QueryDocH5Dataset(Dataset):
    def __init__(self, h5_paths: Sequence[str], split: str = "train", val_split: float = 0.05):
        self.h5_paths = [str(p) for p in h5_paths]
        self.files = [h5py.File(p, "r") for p in self.h5_paths]
        self.file_sources = []
        self.index_map: list[tuple[int, int]] = []
        self.file_candidate_mode: list[bool] = []
        for fi, h5 in enumerate(self.files):
            source_name = self._infer_file_source(Path(self.h5_paths[fi]))
            self.file_sources.append(source_name)
            self.file_candidate_mode.append("candidate_docs" in h5 and "teacher_scores" in h5)
            total = int(h5["query_embeddings"].shape[0])
            val_size = int(total * val_split)
            if val_split > 0 and total > 1:
                val_size = max(1, val_size)
            train_size = total - val_size
            index_range = range(0, train_size) if split == "train" else range(train_size, total)
            self.index_map.extend((fi, i) for i in index_range)
        print(f"Loaded query-doc {split} split: {len(self.index_map)} / files {[Path(p).name for p in self.h5_paths]}")

    @staticmethod
    def _infer_file_source(path: Path) -> str:
        stem = path.stem.lower()
        for candidate in ("msmarco", "fiqa", "quora", "nfcorpus", "scifact", "arguana"):
            if candidate in stem:
                return candidate
        return stem

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fi, row = self.index_map[idx]
        h5 = self.files[fi]
        if self.file_candidate_mode[fi]:
            candidate_docs = [_decode_text(value) for value in h5["candidate_docs"][row]]
            payload = {
                "query": _decode_text(h5["queries"][row]),
                "candidate_docs": candidate_docs,
                "teacher_scores": torch.tensor(h5["teacher_scores"][row], dtype=torch.float32),
                "query_target": torch.tensor(h5["query_embeddings"][row], dtype=torch.float32),
                "source": self.file_sources[fi],
            }
            if "candidate_embeddings" in h5:
                payload["candidate_targets"] = torch.tensor(h5["candidate_embeddings"][row], dtype=torch.float32)
            if "relevance" in h5:
                payload["relevance"] = torch.tensor(h5["relevance"][row], dtype=torch.float32)
            elif "labels" in h5:
                payload["relevance"] = torch.tensor(h5["labels"][row], dtype=torch.float32)
            if "candidate_source" in h5:
                payload["candidate_source"] = [_decode_text(value) for value in h5["candidate_source"][row]]
            return payload
        return {
            "query": _decode_text(h5["queries"][row]),
            "positive": _decode_text(h5["positive_docs"][row]),
            "negative": _decode_text(h5["negative_docs"][row]),
            "query_target": torch.tensor(h5["query_embeddings"][row], dtype=torch.float32),
            "positive_target": torch.tensor(h5["positive_embeddings"][row], dtype=torch.float32),
            "negative_target": torch.tensor(h5["negative_embeddings"][row], dtype=torch.float32),
            "source": self.file_sources[fi],
        }

    def close(self) -> None:
        for h5 in self.files:
            h5.close()


def collate_query_doc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if batch and "candidate_docs" in batch[0]:
        out: dict[str, Any] = {
            "queries": [x["query"] for x in batch],
            "candidate_docs": [x["candidate_docs"] for x in batch],
            "teacher_scores": torch.stack([x["teacher_scores"] for x in batch]),
            "query_targets": torch.stack([x["query_target"] for x in batch]),
            "sources": [x["source"] for x in batch],
        }
        if "candidate_targets" in batch[0]:
            out["candidate_targets"] = torch.stack([x["candidate_targets"] for x in batch])
        if "relevance" in batch[0]:
            out["relevance"] = torch.stack([x["relevance"] for x in batch])
        if "candidate_source" in batch[0]:
            out["candidate_sources"] = [x["candidate_source"] for x in batch]
        return out
    return {
        "queries": [x["query"] for x in batch],
        "positives": [x["positive"] for x in batch],
        "negatives": [x["negative"] for x in batch],
        "query_targets": torch.stack([x["query_target"] for x in batch]),
        "positive_targets": torch.stack([x["positive_target"] for x in batch]),
        "negative_targets": torch.stack([x["negative_target"] for x in batch]),
        "sources": [x["source"] for x in batch],
    }


class QueryDistillationTrainer:
    def __init__(
        self,
        *,
        oscar_model_name: str,
        teacher_embeddings_path: Union[str, Sequence[str]],
        embed_dim: int = 768,
        pooler: str = "flatten",
        projector_type: str = "mem",
        num_layers: int = 2,
        dropout: float = 0.0,
        projector_hidden_dim: int = 8192,
        mem_tokens: int = 8,
        attn_dim: int = 1536,
        attn_layers: int = 2,
        attn_heads: int = 8,
        stage1_steps: int = 0,
        stage1_query_mse_weight: float = 0.5,
        stage1_doc_mse_weight: float = 0.5,
        stage1_negative_mse_weight: float = 0.1,
        stage2_query_mse_weight: float = 0.2,
        stage2_doc_mse_weight: float = 0.2,
        stage2_negative_mse_weight: float = 0.0,
        stage2_teacher_listwise_kl_weight: float = 0.6,
        stage2_ranking_weight: float = 0.1,
        stage2_infonce_weight: float = 0.2,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        lr: float = 5e-5,
        min_lr: float = 1e-6,
        max_steps: int = 0,
        warmup_steps: int = 0,
        scheduler: str = "cosine",
        val_split: float = 0.02,
        query_mse_weight: float = 1.0,
        doc_mse_weight: float = 1.0,
        negative_mse_weight: float = 0.25,
        ranking_weight: float = 0.05,
        infonce_weight: float = 0.0,
        temperature: float = 0.07,
        margin: float = 0.05,
        beir_probe_config: str | None = None,
        beir_probe_samples: int = 20,
        beir_probe_samples_per_dataset: int | None = None,
        beir_probe_negatives: int = 20,
        beir_probe_batch_size: int = 8,
        beir_mini_eval_config: str | None = None,
        beir_mini_eval_every: int = 0,
        beir_full_eval_config: str | None = None,
        beir_full_eval_every: int = 0,
        selection_metric: str = "beir_proxy_ndcg@10",
        dataset_sampling_weights: dict[str, float] | None = None,
        resume_checkpoint: str | None = None,
        device: str = "cuda:0",
        output_dir: str = "checkpoints/query_distill",
        log_dir: str = "logs/query_distill",
    ):
        self.device = torch.device(device)
        self.oscar_model_name = oscar_model_name
        self.batch_size = int(batch_size)
        self.projector_type = str(projector_type)
        self.stage1_steps = max(0, int(stage1_steps))
        self.stage1_query_mse_weight = float(stage1_query_mse_weight)
        self.stage1_doc_mse_weight = float(stage1_doc_mse_weight)
        self.stage1_negative_mse_weight = float(stage1_negative_mse_weight)
        self.stage2_query_mse_weight = float(stage2_query_mse_weight)
        self.stage2_doc_mse_weight = float(stage2_doc_mse_weight)
        self.stage2_negative_mse_weight = float(stage2_negative_mse_weight)
        self.stage2_teacher_listwise_kl_weight = float(stage2_teacher_listwise_kl_weight)
        self.stage2_ranking_weight = float(stage2_ranking_weight)
        self.stage2_infonce_weight = float(stage2_infonce_weight)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.base_lr = float(lr)
        self.min_lr = float(min_lr)
        self.max_steps = int(max_steps or 0)
        self.warmup_steps = int(warmup_steps or 0)
        self.scheduler_name = scheduler
        self.val_split = float(val_split)
        self.query_mse_weight = float(query_mse_weight)
        self.doc_mse_weight = float(doc_mse_weight)
        self.negative_mse_weight = float(negative_mse_weight)
        self.ranking_weight = float(ranking_weight)
        self.infonce_weight = float(infonce_weight)
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.beir_probe_config = beir_probe_config
        self.beir_probe_samples = int(beir_probe_samples)
        self.beir_probe_samples_per_dataset = (
            int(beir_probe_samples_per_dataset) if beir_probe_samples_per_dataset is not None else None
        )
        self.beir_probe_negatives = int(beir_probe_negatives)
        self.beir_probe_batch_size = int(beir_probe_batch_size)
        self.beir_mini_eval_config = str(beir_mini_eval_config) if beir_mini_eval_config else None
        self.beir_mini_eval_every = int(beir_mini_eval_every)
        self.beir_full_eval_config = str(beir_full_eval_config) if beir_full_eval_config else None
        self.beir_full_eval_every = int(beir_full_eval_every)
        self.selection_metric = str(selection_metric)
        self.dataset_sampling_weights = {str(k): float(v) for k, v in (dataset_sampling_weights or {}).items()}
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint else None
        self.resume_step = 0
        self.resume_metrics: dict[str, float] = {}
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.run_root = self.output_dir.parent if self.output_dir.name == "checkpoints" else self.output_dir
        self.metrics_dir = self.run_root / "metrics"
        self.plots_dir = self.run_root / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.val_history_path = self.metrics_dir / "query_distill_val_history.json"
        self.checkpoint_val_history_path = self.output_dir / "val_metrics_history.json"
        self.val_history: list[dict[str, float]] = self._load_val_history()

        disable_transformers_allocator_warmup()
        disable_resume_download_passthrough()
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()
        configure_oscar_component_devices(self.oscar_model)
        if hasattr(self.oscar_model, "compr") and hasattr(self.oscar_model.compr, "config"):
            self.oscar_model.compr.config.mean_resizing = False
        hidden_size = int(self.oscar_model.compress_documents(["test"]).shape[-1])
        print(f"OSCAR hidden size: {hidden_size}")

        h5_paths = _resolve_h5_paths(teacher_embeddings_path)
        self.train_dataset = QueryDocH5Dataset(h5_paths, split="train", val_split=val_split)
        self.val_dataset = QueryDocH5Dataset(h5_paths, split="val", val_split=val_split)
        train_sampler = self._build_weighted_sampler(self.train_dataset, self.dataset_sampling_weights)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_query_doc,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_query_doc,
            num_workers=2,
        )

        if self.projector_type == "token_dual":
            self.projector = TokenAwareDualProjector(
                hidden_dim=hidden_size,
                embed_dim=embed_dim,
                num_mem_tokens=mem_tokens,
                attn_dim=attn_dim,
                num_attn_layers=attn_layers,
                num_heads=attn_heads,
                dropout=dropout,
                projector_hidden_dim=projector_hidden_dim,
                head_layers=max(1, int(num_layers)),
            ).to(device=self.device, dtype=torch.bfloat16)
        elif self.projector_type == "dual_head":
            self.projector = DualHeadMEMProjector(
                hidden_dim=hidden_size,
                embed_dim=embed_dim,
                pooler=pooler,
                trunk_hidden_dim=projector_hidden_dim,
                head_hidden_dim=max(embed_dim, projector_hidden_dim // 2),
                trunk_layers=1,
                head_layers=max(1, int(num_layers)),
                dropout=dropout,
            ).to(device=self.device, dtype=torch.bfloat16)
        else:
            self.projector = MEMProjector(
                hidden_dim=hidden_size,
                embed_dim=embed_dim,
                pooler=pooler,
                num_layers=num_layers,
                dropout=dropout,
                projector_hidden_dim=projector_hidden_dim,
            ).to(device=self.device, dtype=torch.bfloat16)
        print("Query distillation projector architecture:")
        print(self.projector)
        print(f"Projector parameters: {sum(p.numel() for p in self.projector.parameters()):,}")
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=self.base_lr, weight_decay=0.01)
        self.projector_config = {
            "oscar_hidden_dim": hidden_size,
            "oscar_model_name": self.oscar_model_name,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "projector_hidden_dim": projector_hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "mem_tokens": mem_tokens,
            "attn_dim": attn_dim,
            "attn_layers": attn_layers,
            "attn_heads": attn_heads,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "selection_metric": self.selection_metric,
            "projector_type": self.projector_type,
            "training_objective": "query_doc_bge_distill",
            "dataset_sampling_weights": self.dataset_sampling_weights,
            "resume_checkpoint": str(self.resume_checkpoint) if self.resume_checkpoint else None,
            "stage1_steps": self.stage1_steps,
            "stage1_query_mse_weight": self.stage1_query_mse_weight,
            "stage1_doc_mse_weight": self.stage1_doc_mse_weight,
            "stage1_negative_mse_weight": self.stage1_negative_mse_weight,
            "stage2_query_mse_weight": self.stage2_query_mse_weight,
            "stage2_doc_mse_weight": self.stage2_doc_mse_weight,
            "stage2_negative_mse_weight": self.stage2_negative_mse_weight,
            "stage2_teacher_listwise_kl_weight": self.stage2_teacher_listwise_kl_weight,
            "stage2_ranking_weight": self.stage2_ranking_weight,
            "stage2_infonce_weight": self.stage2_infonce_weight,
            "beir_mini_eval_config": self.beir_mini_eval_config,
            "beir_mini_eval_every": self.beir_mini_eval_every,
            "beir_full_eval_config": self.beir_full_eval_config,
            "beir_full_eval_every": self.beir_full_eval_every,
        }
        self._beir_probe_cases = self._load_beir_probe_cases()
        self.load_checkpoint()

    def _load_val_history(self) -> list[dict[str, float]]:
        for path in (self.val_history_path, self.checkpoint_val_history_path):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[val-history] cannot parse {path}: {exc}")
                continue
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
        return []

    def _write_val_history(self) -> None:
        payload = json.dumps(self.val_history, indent=2, ensure_ascii=False)
        self.val_history_path.write_text(payload, encoding="utf-8")
        self.checkpoint_val_history_path.write_text(payload, encoding="utf-8")

    def _record_val_metrics(self, step: int, metrics: dict[str, float]) -> None:
        row = {"step": int(step), **metrics}
        for idx, existing in enumerate(self.val_history):
            if int(existing.get("step", -1)) == int(step):
                self.val_history[idx] = row
                break
        else:
            self.val_history.append(row)
        self.val_history.sort(key=lambda item: int(item.get("step", -1)))
        self._write_val_history()

    def load_checkpoint(self) -> None:
        checkpoint_path: Path | None = None
        if self.resume_checkpoint is not None:
            checkpoint_path = self.resume_checkpoint
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_path}")
        else:
            best_path = self.output_dir / "best_model.pt"
            if best_path.exists():
                checkpoint_path = best_path
        if checkpoint_path is None:
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.projector.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.resume_step = int(checkpoint.get("step", 0) or 0)
        metrics = checkpoint.get("metrics", {})
        self.resume_metrics = dict(metrics) if isinstance(metrics, dict) else {}
        print(
            f"Loaded checkpoint from {checkpoint_path} at step {self.resume_step}",
            flush=True,
        )

    def _step_lr(self, step: int) -> float:
        if self.max_steps <= 0 and self.scheduler_name not in {"legacy", "none"}:
            if hasattr(self, "train_loader") and hasattr(self, "gradient_accumulation_steps"):
                epochs = self.projector_config.get("epochs", 1)
                self.max_steps = int(epochs * len(self.train_loader) / self.gradient_accumulation_steps)
                if self.resume_step > 0:
                    self.max_steps += self.resume_step
                print(f"[schedule] auto-calculated max_steps={self.max_steps} from {epochs} epochs", flush=True)

        if self.max_steps <= 0 or self.scheduler_name in {"legacy", "none"}:
            return float(self.optimizer.param_groups[0]["lr"])
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            lr = self.base_lr * step / max(1, self.warmup_steps)
        else:
            denom = max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / denom))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return float(lr)

    def _build_weighted_sampler(
        self,
        dataset: QueryDocH5Dataset,
        sampling_weights: dict[str, float],
    ) -> WeightedRandomSampler | None:
        if not sampling_weights:
            return None
        weights = np.zeros(len(dataset.index_map), dtype=np.float64)
        source_counts = collections.Counter()
        unknown_sources: set[str] = set()
        for idx, (fi, _row) in enumerate(dataset.index_map):
            source = dataset.file_sources[fi]
            source_counts[source] += 1
            if source in sampling_weights:
                weights[idx] = sampling_weights[source]
            else:
                unknown_sources.add(source)
        if np.all(weights == 0):
            print(
                f"[sampling] no dataset_sampling_weights matched sources={sorted(source_counts.keys())}; falling back to shuffle",
                flush=True,
            )
            return None
        if unknown_sources:
            print(
                f"[sampling] sources without explicit weights={sorted(unknown_sources)}; assigned zero probability",
                flush=True,
            )
        normalized = weights / max(weights.sum(), 1e-12)
        print(
            "[sampling] configured weights="
            + ", ".join(f"{k}:{v}" for k, v in sorted(sampling_weights.items())),
            flush=True,
        )
        print(
            "[sampling] source counts="
            + ", ".join(f"{k}:{source_counts[k]}" for k in sorted(source_counts.keys())),
            flush=True,
        )
        return WeightedRandomSampler(
            weights=torch.from_numpy(normalized),
            num_samples=len(dataset.index_map),
            replacement=True,
        )

    def encode_texts(self, texts: list[str], mode: str = "doc") -> torch.Tensor:
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
        if self.projector_type in {"token_dual", "dual_head"}:
            return self.projector(mem_embeddings, mode=mode)
        return self.projector(mem_embeddings)

    def encode_candidate_docs(self, candidate_docs: list[list[str]]) -> torch.Tensor:
        if not candidate_docs:
            return torch.empty(0, 0, device=self.device)
        batch_size = len(candidate_docs)
        num_candidates = len(candidate_docs[0])
        flat_docs = [doc for docs in candidate_docs for doc in docs]
        flat_emb = F.normalize(self.encode_texts(flat_docs, mode="doc").float(), dim=-1)
        return flat_emb.view(batch_size, num_candidates, -1)

    def _teacher_listwise_kl(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n: torch.Tensor,
        q_t: torch.Tensor,
        p_t: torch.Tensor,
        n_t: torch.Tensor,
    ) -> torch.Tensor:
        student_logits = torch.cat(
            [
                torch.sum(q * n, dim=-1, keepdim=True),
                torch.matmul(q, p.T),
            ],
            dim=1,
        ) / max(self.temperature, 1e-6)
        teacher_logits = torch.cat(
            [
                torch.sum(q_t * n_t, dim=-1, keepdim=True),
                torch.matmul(q_t, p_t.T),
            ],
            dim=1,
        ) / max(self.temperature, 1e-6)
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        student_log_probs = torch.log_softmax(student_logits, dim=1)
        return torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs), dim=1).mean()

    def _compute_candidate_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        q = F.normalize(self.encode_texts(batch["queries"], mode="query").float(), dim=-1)
        cand = self.encode_candidate_docs(batch["candidate_docs"])
        q_t = F.normalize(batch["query_targets"].to(self.device).float(), dim=-1)
        teacher_scores = batch["teacher_scores"].to(self.device).float()
        student_scores = torch.einsum("bd,bkd->bk", q, cand)
        student_logits = student_scores / max(self.temperature, 1e-6)
        teacher_logits = teacher_scores / max(self.temperature, 1e-6)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        teacher_score_kl = torch.sum(
            teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs),
            dim=-1,
        ).mean()

        labels = torch.argmax(teacher_scores, dim=-1)
        infonce_loss = F.cross_entropy(student_logits, labels)
        positive_scores = student_scores.gather(1, labels.unsqueeze(1)).squeeze(1)
        negative_mask = torch.ones_like(student_scores, dtype=torch.bool)
        negative_mask.scatter_(1, labels.unsqueeze(1), False)
        hardest_negative = student_scores.masked_fill(~negative_mask, -1e4).max(dim=-1).values
        hard_negative_margin = torch.relu(self.margin - positive_scores + hardest_negative).mean()

        if "candidate_targets" in batch:
            cand_t = F.normalize(batch["candidate_targets"].to(self.device).float(), dim=-1)
            target_doc = cand_t.gather(
                1,
                labels.view(-1, 1, 1).expand(-1, 1, cand_t.size(-1)),
            ).squeeze(1)
            q_mse = F.mse_loss(q, q_t)
            doc_mse = F.mse_loss(cand.gather(1, labels.view(-1, 1, 1).expand(-1, 1, cand.size(-1))).squeeze(1), target_doc)
            mse_anchor = q_mse + doc_mse
        else:
            q_mse = F.mse_loss(q, q_t)
            doc_mse = torch.zeros((), device=self.device)
            mse_anchor = q_mse

        current_step = int(getattr(self, "_current_step", 0))
        if self.stage1_steps > 0 and current_step < self.stage1_steps:
            total = 0.4 * q_mse + 0.2 * F.relu(1.0 - positive_scores).mean() + 0.4 * teacher_score_kl
            loss_stage = "stage1"
        else:
            total = (
                self.stage2_teacher_listwise_kl_weight * teacher_score_kl
                + self.stage2_infonce_weight * infonce_loss
                + self.stage2_ranking_weight * hard_negative_margin
                + max(self.stage2_query_mse_weight, 0.0) * mse_anchor
            )
            loss_stage = "stage2"

        student_rank = torch.argsort(torch.argsort(student_scores, dim=-1, descending=True), dim=-1)
        teacher_rank = torch.argsort(torch.argsort(teacher_scores, dim=-1, descending=True), dim=-1)
        centered_student = student_rank.float() - student_rank.float().mean(dim=-1, keepdim=True)
        centered_teacher = teacher_rank.float() - teacher_rank.float().mean(dim=-1, keepdim=True)
        spearman = (
            (centered_student * centered_teacher).sum(dim=-1)
            / (
                torch.linalg.norm(centered_student, dim=-1)
                * torch.linalg.norm(centered_teacher, dim=-1)
            ).clamp(min=1e-6)
        ).mean()
        positive_rank = student_rank.gather(1, labels.unsqueeze(1)).float().mean() + 1.0
        metrics = {
            "query_mse": float(q_mse.item()),
            "positive_mse": float(doc_mse.item()),
            "negative_mse": 0.0,
            "ranking_loss": float(hard_negative_margin.item()),
            "infonce_loss": float(infonce_loss.item()),
            "teacher_listwise_kl": float(teacher_score_kl.item()),
            "loss_stage": 1.0 if loss_stage == "stage1" else 2.0,
            "pos_sim": float(positive_scores.mean().item()),
            "neg_sim": float(hardest_negative.mean().item()),
            "gap": float((positive_scores - hardest_negative).mean().item()),
            "pos_sim_std": float(positive_scores.std(unbiased=False).item()),
            "neg_sim_std": float(hardest_negative.std(unbiased=False).item()),
            "gap_std": float((positive_scores - hardest_negative).std(unbiased=False).item()),
            "gap_min": float((positive_scores - hardest_negative).min().item()),
            "gap_max": float((positive_scores - hardest_negative).max().item()),
            "ranking_active_frac": float((self.margin - positive_scores + hardest_negative > 0).float().mean().item()),
            "teacher_top1_agreement": float((torch.argmax(student_scores, dim=-1) == labels).float().mean().item()),
            "teacher_student_spearman": float(spearman.item()),
            "positive_rank_in_candidates": float(positive_rank.item()),
            "embedding_uniformity": float(torch.pdist(q, p=2).pow(2).mul(-2.0).exp().mean().log().item()) if q.size(0) > 1 else 0.0,
        }
        return total, metrics

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        if "candidate_docs" in batch:
            return self._compute_candidate_loss(batch)
        q = F.normalize(self.encode_texts(batch["queries"], mode="query").float(), dim=-1)
        p = F.normalize(self.encode_texts(batch["positives"], mode="doc").float(), dim=-1)
        n = F.normalize(self.encode_texts(batch["negatives"], mode="doc").float(), dim=-1)
        q_t = F.normalize(batch["query_targets"].to(self.device).float(), dim=-1)
        p_t = F.normalize(batch["positive_targets"].to(self.device).float(), dim=-1)
        n_t = F.normalize(batch["negative_targets"].to(self.device).float(), dim=-1)

        q_mse = F.mse_loss(q, q_t)
        p_mse = F.mse_loss(p, p_t)
        n_mse = F.mse_loss(n, n_t)
        pos_sim = (q * p).sum(dim=-1)
        neg_sim = (q * n).sum(dim=-1)
        gap = pos_sim - neg_sim
        ranking_loss = torch.relu(self.margin - pos_sim + neg_sim).mean()
        logits = torch.matmul(q, p.T) / max(self.temperature, 1e-6)
        labels = torch.arange(q.size(0), device=q.device)
        infonce_loss = F.cross_entropy(logits, labels)
        teacher_listwise_kl = self._teacher_listwise_kl(q, p, n, q_t, p_t, n_t)
        current_step = int(getattr(self, "_current_step", 0))
        if self.stage1_steps > 0 and current_step < self.stage1_steps:
            total = (
                self.stage1_query_mse_weight * q_mse
                + self.stage1_doc_mse_weight * p_mse
                + self.stage1_negative_mse_weight * n_mse
            )
            loss_stage = "stage1"
        else:
            total = (
                self.stage2_query_mse_weight * q_mse
                + self.stage2_doc_mse_weight * p_mse
                + self.stage2_negative_mse_weight * n_mse
                + self.stage2_teacher_listwise_kl_weight * teacher_listwise_kl
                + self.stage2_ranking_weight * ranking_loss
                + self.stage2_infonce_weight * infonce_loss
            )
            loss_stage = "stage2"
        metrics = {
            "query_mse": float(q_mse.item()),
            "positive_mse": float(p_mse.item()),
            "negative_mse": float(n_mse.item()),
            "ranking_loss": float(ranking_loss.item()),
            "infonce_loss": float(infonce_loss.item()),
            "teacher_listwise_kl": float(teacher_listwise_kl.item()),
            "loss_stage": 1.0 if loss_stage == "stage1" else 2.0,
            "pos_sim": float(pos_sim.mean().item()),
            "neg_sim": float(neg_sim.mean().item()),
            "gap": float(gap.mean().item()),
            "pos_sim_std": float(pos_sim.std(unbiased=False).item()),
            "neg_sim_std": float(neg_sim.std(unbiased=False).item()),
            "gap_std": float(gap.std(unbiased=False).item()),
            "gap_min": float(gap.min().item()),
            "gap_max": float(gap.max().item()),
            "ranking_active_frac": float((self.margin - pos_sim + neg_sim > 0).float().mean().item()),
            "teacher_top1_agreement": float(
                (
                    torch.argmax(
                        torch.cat(
                            [
                                torch.sum(q * p, dim=-1, keepdim=True),
                                torch.sum(q * n, dim=-1, keepdim=True),
                            ],
                            dim=1,
                        ),
                        dim=1,
                    )
                    == torch.argmax(
                        torch.cat(
                            [
                                torch.sum(q_t * p_t, dim=-1, keepdim=True),
                                torch.sum(q_t * n_t, dim=-1, keepdim=True),
                            ],
                            dim=1,
                        ),
                        dim=1,
                    )
                ).float().mean().item()
            ),
        }
        return total, metrics

    def validate(self, step: int, *, force_mini_eval: bool = False, force_full_eval: bool = False) -> dict[str, float]:
        self.projector.eval()
        totals: dict[str, float] = {}
        per_batch: dict[str, list[float]] = {}
        count = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                loss, metrics = self.compute_loss(batch)
                metrics["loss_total"] = float(loss.item())
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                    per_batch.setdefault(key, []).append(float(value))
                count += 1
        out = {key: value / max(1, count) for key, value in totals.items()}
        for key, values in per_batch.items():
            arr = np.asarray(values, dtype=np.float32)
            out[f"{key}_batch_std"] = float(arr.std())
            out[f"{key}_batch_min"] = float(arr.min())
            out[f"{key}_batch_max"] = float(arr.max())
        out.update(self.run_beir_probe(step))
        should_run_mini = force_mini_eval or (
            self.beir_mini_eval_every > 0 and step > 0 and step % self.beir_mini_eval_every == 0
        )
        if should_run_mini:
            out.update(self.run_beir_eval(step, self.beir_mini_eval_config, "beir_mini"))
        should_run_full = force_full_eval or (
            self.beir_full_eval_every > 0 and step > 0 and step % self.beir_full_eval_every == 0
        )
        if should_run_full:
            out.update(self.run_beir_eval(step, self.beir_full_eval_config, "beir_full"))
        print(f"[val] step={step}", flush=True)
        print(
            "[val] losses "
            + " ".join(
                f"{key}={out[key]:.6f}"
                for key in ("loss_total", "query_mse", "positive_mse", "negative_mse", "ranking_loss", "infonce_loss")
                if key in out
            ),
            flush=True,
        )
        print(
            "[val] retrieval "
            + " ".join(
                f"{key}={out[key]:.6f}"
                for key in (
                    "pos_sim",
                    "neg_sim",
                    "gap",
                    "pos_sim_std",
                    "neg_sim_std",
                    "gap_std",
                    "gap_min",
                    "gap_max",
                    "ranking_active_frac",
                    "beir_probe_mrr@10",
                    "beir_probe_ndcg@10",
                    "beir_interquery_ndcg@10",
                    "beir_probe_recall@10",
                )
                if key in out
            ),
            flush=True,
        )
        batch_stat_keys = [key for key in out if key.endswith(("_batch_std", "_batch_min", "_batch_max"))]
        if batch_stat_keys:
            print("[val] batch_stats " + " ".join(f"{key}={out[key]:.6f}" for key in sorted(batch_stat_keys)), flush=True)
        probe_keys = [key for key in out if key.startswith("beir_probe_") and key not in {"beir_probe_mrr@10", "beir_probe_recall@10"}]
        if probe_keys:
            print("[val] beir_probe " + " ".join(f"{key}={out[key]:.6f}" for key in sorted(probe_keys)), flush=True)
        for key, value in out.items():
            self.writer.add_scalar(f"val/{key}", value, step)
        self.writer.add_scalar("val/meta/num_batches", count, step)
        self.writer.add_scalar("val/meta/num_examples", len(self.val_dataset), step)
        if per_batch:
            validation_detail = {
                "step": int(step),
                "metrics": out,
                "per_batch": per_batch,
            }
            (self.metrics_dir / f"validation_step_{step}.json").write_text(
                json.dumps(validation_detail, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.writer.flush()
        return out

    def save_checkpoint(self, step: int, metrics: dict[str, float], is_best: bool = False) -> None:
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "model_state_dict": self.projector.state_dict(),
            "config": self.projector_config,
        }
        torch.save(checkpoint, self.output_dir / f"checkpoint_step_{step}.pt")
        (self.output_dir / "config.json").write_text(json.dumps(self.projector_config, indent=2), encoding="utf-8")
        self._record_val_metrics(step, metrics)
        (self.output_dir / "metrics.json").write_text(
            json.dumps({"step": int(step), "val": metrics}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pt")
            (self.output_dir / "best_config.json").write_text(json.dumps(self.projector_config, indent=2), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _rank_bm25_hard_negatives(
        self,
        *,
        query: str,
        relevant_doc_ids: set[str],
        corpus_ids: list[str],
        corpus_texts: list[str],
        top_k: int,
    ) -> list[tuple[str, str]]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank-bm25 to use BM25 hard-negative BEIR probe") from exc
        bm25 = BM25Okapi([text.lower().split() for text in corpus_texts])
        scores = bm25.get_scores(query.lower().split())
        negatives: list[tuple[str, str]] = []
        for idx in np.argsort(-scores):
            doc_id = corpus_ids[int(idx)]
            if doc_id in relevant_doc_ids:
                continue
            text = corpus_texts[int(idx)]
            if text:
                negatives.append((doc_id, text))
            if len(negatives) >= top_k:
                break
        return negatives

    def _load_beir_probe_cases(self) -> list[dict[str, str]]:
        if not self.beir_probe_config:
            return []
        cfg = load_yaml(self.beir_probe_config)
        datasets = cfg.get("datasets", [])
        if not datasets:
            return []
        cases: list[dict[str, str]] = []
        samples_per_dataset = self.beir_probe_samples_per_dataset or self.beir_probe_samples
        for dataset_idx, dataset_item in enumerate(datasets):
            dataset_name = str(dataset_item.get("name", f"beir_{dataset_idx}"))
            dataset_dir = Path(dataset_item["path"])
            qrels_path = dataset_dir / "qrels" / f"{cfg.get('split', 'test')}.tsv"
            try:
                corpus_rows = self._read_jsonl(dataset_dir / "corpus.jsonl")
                query_rows = self._read_jsonl(dataset_dir / "queries.jsonl")
            except Exception as exc:
                print(f"[beir-probe] cannot load corpus/queries from {dataset_dir}: {exc}")
                continue
            corpus = {
                str(r["_id"]): " ".join(
                    p for p in [str(r.get("title", "")).strip(), str(r.get("text", "")).strip()] if p
                ).strip()
                for r in corpus_rows
            }
            corpus = {doc_id: text for doc_id, text in corpus.items() if text}
            corpus_ids = list(corpus.keys())
            corpus_texts = [corpus[doc_id] for doc_id in corpus_ids]
            queries = {str(r["_id"]): str(r.get("text", "")) for r in query_rows}
            relevant: dict[str, set[str]] = {}
            try:
                with qrels_path.open("r", encoding="utf-8") as fp:
                    _ = fp.readline()
                    for line in fp:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3 and int(parts[2]) > 0:
                            relevant.setdefault(str(parts[0]), set()).add(str(parts[1]))
            except Exception as exc:
                print(f"[beir-probe] cannot read qrels {qrels_path}: {exc}")
                continue
            candidate_qids = [qid for qid in relevant.keys() if qid in queries]
            rng = np.random.default_rng(42 + dataset_idx)
            rng.shuffle(candidate_qids)
            dataset_cases = 0
            for qid in candidate_qids:
                pos_ids = [doc_id for doc_id in sorted(relevant.get(qid, set())) if doc_id in corpus]
                if not pos_ids:
                    continue
                pos_text = corpus.get(pos_ids[0], "")
                if pos_text:
                    negatives = self._rank_bm25_hard_negatives(
                        query=queries[qid],
                        relevant_doc_ids=set(pos_ids),
                        corpus_ids=corpus_ids,
                        corpus_texts=corpus_texts,
                        top_k=self.beir_probe_negatives,
                    )
                    if not negatives:
                        continue
                    cases.append(
                        {
                            "dataset": dataset_name,
                            "query_id": qid,
                            "positive_doc_id": pos_ids[0],
                            "positive_doc_ids": pos_ids,
                            "query": queries[qid],
                            "positive": pos_text,
                            "negative": negatives[0][1],
                            "negative_doc_ids": [doc_id for doc_id, _text in negatives],
                            "negatives": [text for _doc_id, text in negatives],
                        }
                    )
                    dataset_cases += 1
                if dataset_cases >= samples_per_dataset:
                    break
            print(
                f"[beir-probe] loaded {dataset_cases} query cases from {dataset_name} "
                f"with bm25_hard_negatives={self.beir_probe_negatives} corpus_docs={len(corpus_ids)}",
                flush=True,
            )
        print(f"[beir-probe] loaded {len(cases)} total cases from {len(datasets)} configured datasets")
        return cases

    def _encode_probe_texts(self, texts: list[str], mode: str = "doc") -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), self.beir_probe_batch_size):
            batch = texts[start : start + self.beir_probe_batch_size]
            chunks.append(F.normalize(self.encode_texts(batch, mode=mode).float(), dim=-1))
        return torch.cat(chunks, dim=0)

    def _load_beir_eval_template(self, config_path: str | None) -> dict[str, Any] | None:
        if not config_path:
            return None
        cfg = load_yaml(config_path)
        return cfg if isinstance(cfg, dict) else None

    def run_beir_eval(self, step: int, config_path: str | None, prefix: str) -> dict[str, float]:
        template = self._load_beir_eval_template(config_path)
        if template is None:
            return {}
        checkpoint_path = self.output_dir / f"checkpoint_step_{step}.pt"
        if not checkpoint_path.exists():
            torch.save(
                {
                    "step": int(step),
                    "metrics": {},
                    "model_state_dict": self.projector.state_dict(),
                    "config": self.projector_config,
                },
                checkpoint_path,
            )
        start_time = time.perf_counter()
        cfg = json.loads(json.dumps(template))
        eval_encoder_device = os.getenv("BEIR_PIPELINE_A_DEVICE", str(self.device))
        eval_decoder_device = os.getenv("BEIR_PIPELINE_B_DEVICE", os.getenv("OSCAR_DECODER_DEVICE", eval_encoder_device))
        cfg.setdefault("encoder", {})
        cfg["encoder"]["name"] = "oscar_projector"
        cfg["encoder"]["kwargs"] = {
            "oscar_model_name": self.oscar_model_name,
            "projector_path": str(checkpoint_path),
            "device": str(eval_encoder_device),
            "embed_dim": int(self.projector_config.get("embed_dim", 768)),
            "pooler": str(self.projector_config.get("pooler", "flatten")),
            "num_layers": int(self.projector_config.get("num_layers", 2)),
            "dropout": float(self.projector_config.get("dropout", 0.0)),
            "projector_hidden_dim": int(self.projector_config.get("projector_hidden_dim", 8192)),
            "oscar_model_instance": self.oscar_model,
        }
        prev_decoder = os.environ.get("OSCAR_DECODER_DEVICE")
        prev_compressor = os.environ.get("OSCAR_COMPRESSOR_DEVICE")
        os.environ["OSCAR_DECODER_DEVICE"] = str(eval_decoder_device)
        os.environ["OSCAR_COMPRESSOR_DEVICE"] = str(eval_encoder_device)
        try:
            summary = evaluate_beir(cfg)
        finally:
            if prev_decoder is None:
                os.environ.pop("OSCAR_DECODER_DEVICE", None)
            else:
                os.environ["OSCAR_DECODER_DEVICE"] = prev_decoder
            if prev_compressor is None:
                os.environ.pop("OSCAR_COMPRESSOR_DEVICE", None)
            else:
                os.environ["OSCAR_COMPRESSOR_DEVICE"] = prev_compressor
        print(
            f"[{prefix}] devices encoder={eval_encoder_device} decoder={eval_decoder_device}",
            flush=True,
        )
        print(f"[{prefix}] step={step} elapsed_sec={time.perf_counter() - start_time:.1f}", flush=True)
        metrics: dict[str, float] = {}
        avg = summary.get("average", {})
        if isinstance(avg, dict):
            for key, value in avg.items():
                try:
                    metrics[f"{prefix}_{key}"] = float(value)
                except (TypeError, ValueError):
                    continue
        per_dataset = summary.get("per_dataset", {})
        if isinstance(per_dataset, dict):
            for dataset_name, ds_metrics in per_dataset.items():
                if not isinstance(ds_metrics, dict):
                    continue
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name))
                for key, value in ds_metrics.items():
                    try:
                        metrics[f"{prefix}_{safe_name}_{key}"] = float(value)
                    except (TypeError, ValueError):
                        continue
        return metrics

    def run_beir_probe(self, step: int) -> dict[str, float]:
        if not self._beir_probe_cases:
            return {}
        start_time = time.perf_counter()
        with torch.no_grad():
            queries = [c["query"] for c in self._beir_probe_cases]
            positives = [c["positive"] for c in self._beir_probe_cases]
            negatives = [c["negative"] for c in self._beir_probe_cases]
            q_emb = self._encode_probe_texts(queries, mode="query")
            p_emb = self._encode_probe_texts(positives, mode="doc")
            n_emb = self._encode_probe_texts(negatives, mode="doc")
            pos_cos = (q_emb * p_emb).sum(dim=-1).cpu().numpy()
            neg_cos = (q_emb * n_emb).sum(dim=-1).cpu().numpy()
            candidate_texts: list[str] = []
            offsets: list[tuple[int, int]] = []
            for case in self._beir_probe_cases:
                start = len(candidate_texts)
                candidate_texts.append(case["positive"])
                candidate_texts.extend(case.get("negatives", [case["negative"]]))
                offsets.append((start, len(candidate_texts)))
            cand_emb = self._encode_probe_texts(candidate_texts, mode="doc")
            similarities = torch.matmul(q_emb, cand_emb.T).detach().cpu().numpy()
        mrr_at_10 = 0.0
        recall_at_10 = 0.0
        ndcg_at_10 = 0.0
        interquery_mrr_at_10 = 0.0
        interquery_recall_at_10 = 0.0
        interquery_ndcg_at_10 = 0.0
        ranks: list[int] = []
        interquery_ranks: list[int] = []
        reciprocal_ranks: list[float] = []
        discounted_gains: list[float] = []
        for i, (start, stop) in enumerate(offsets):
            local_scores = similarities[i, start:stop]
            rank = int(np.where(np.argsort(-local_scores) == 0)[0][0]) + 1
            ranks.append(rank)
            interquery_ranks.append(rank)
            reciprocal_ranks.append(1.0 / rank)
            discounted_gain = 1.0 / float(np.log2(rank + 1.0)) if rank <= 10 else 0.0
            discounted_gains.append(discounted_gain)
            if rank <= 10:
                mrr_at_10 += 1.0 / rank
                recall_at_10 += 1.0
                ndcg_at_10 += discounted_gain
                interquery_mrr_at_10 += 1.0 / rank
                interquery_recall_at_10 += 1.0
                interquery_ndcg_at_10 += discounted_gain
        pos_mean = float(np.mean(pos_cos))
        neg_mean = float(np.mean(neg_cos))
        gaps = pos_cos - neg_cos
        gap_mean = float(np.mean(gaps))
        rank_arr = np.asarray(ranks, dtype=np.float32)
        interquery_rank_arr = np.asarray(interquery_ranks, dtype=np.float32)
        rr_arr = np.asarray(reciprocal_ranks, dtype=np.float32)
        ndcg_arr = np.asarray(discounted_gains, dtype=np.float32)
        denom = max(1, len(rank_arr))
        datasets = [str(case.get("dataset", "beir")) for case in self._beir_probe_cases]
        dataset_summaries: dict[str, dict[str, float]] = {}
        for dataset_name in sorted(set(datasets)):
            indices = np.asarray([i for i, name in enumerate(datasets) if name == dataset_name], dtype=np.int64)
            if indices.size == 0:
                continue
            ds_ranks = rank_arr[indices]
            ds_interquery_ranks = interquery_rank_arr[indices]
            ds_rr = rr_arr[indices]
            ds_ndcg = ndcg_arr[indices]
            ds_recall_at_10 = float(np.mean(ds_ranks <= 10))
            ds_interquery_recall_at_10 = float(np.mean(ds_interquery_ranks <= 10))
            dataset_summaries[dataset_name] = {
                "cases": float(indices.size),
                "pos_cosine_mean": float(np.mean(pos_cos[indices])),
                "neg_cosine_mean": float(np.mean(neg_cos[indices])),
                "gap_mean": float(np.mean(gaps[indices])),
                "gap_std": float(np.std(gaps[indices])),
                "rank_mean": float(np.mean(ds_ranks)),
                "rank_median": float(np.median(ds_ranks)),
                "interquery_rank_mean": float(np.mean(ds_interquery_ranks)),
                "interquery_rank_median": float(np.median(ds_interquery_ranks)),
                "mrr": float(np.mean(ds_rr)),
                "mrr@10": float(np.mean(np.where(ds_ranks <= 10, ds_rr, 0.0))),
                "ndcg@10": float(np.mean(ds_ndcg)),
                "recall@10": ds_recall_at_10,
                "interquery_ndcg@10": float(
                    np.mean(np.where(ds_interquery_ranks <= 10, 1.0 / np.log2(ds_interquery_ranks + 1.0), 0.0))
                ),
                "interquery_recall@10": ds_interquery_recall_at_10,
            }
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(pos_cos))
        ax.plot(x, pos_cos, marker="o", label="positive cosine")
        ax.plot(x, neg_cos, marker="x", label="negative cosine")
        ax.axhline(pos_mean, linestyle="--", linewidth=1.0, label=f"pos mean={pos_mean:.3f}")
        ax.axhline(neg_mean, linestyle="--", linewidth=1.0, label=f"neg mean={neg_mean:.3f}")
        ax.set_title(f"Query distill BEIR probe @ step {step}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig_path = self.plots_dir / f"beir_probe_step_{step}.png"
        fig.savefig(fig_path, dpi=120)
        self.writer.add_figure("beir_probe/cosine_plot", fig, step)
        plt.close(fig)
        for dataset_name, summary in dataset_summaries.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)
            indices = np.asarray([i for i, name in enumerate(datasets) if name == dataset_name], dtype=np.int64)
            ds_x = np.arange(indices.size)
            ds_fig, ds_ax = plt.subplots(figsize=(10, 4))
            ds_ax.plot(ds_x, pos_cos[indices], marker="o", label="positive cosine")
            ds_ax.plot(ds_x, neg_cos[indices], marker="x", label="negative cosine")
            ds_ax.axhline(summary["pos_cosine_mean"], linestyle="--", linewidth=1.0, label=f"pos mean={summary['pos_cosine_mean']:.3f}")
            ds_ax.axhline(summary["neg_cosine_mean"], linestyle="--", linewidth=1.0, label=f"neg mean={summary['neg_cosine_mean']:.3f}")
            ds_ax.set_title(f"BEIR probe {dataset_name} @ step {step}")
            ds_ax.grid(alpha=0.25)
            ds_ax.legend(loc="best")
            ds_fig.tight_layout()
            ds_fig_path = self.plots_dir / f"beir_probe_{safe_name}_step_{step}.png"
            ds_fig.savefig(ds_fig_path, dpi=120)
            self.writer.add_figure(f"beir_probe_by_dataset/{dataset_name}/cosine_plot", ds_fig, step)
            plt.close(ds_fig)
            dataset_detail = {
                "step": int(step),
                "dataset": dataset_name,
                "summary": summary,
                "cases": [
                    {
                        "idx": int(i),
                        "query_id": self._beir_probe_cases[i].get("query_id", ""),
                        "positive_doc_id": self._beir_probe_cases[i].get("positive_doc_id", ""),
                        "rank": int(ranks[i]),
                        "interquery_rank": int(interquery_ranks[i]),
                        "pos_cosine": float(pos_cos[i]),
                        "neg_cosine": float(neg_cos[i]),
                        "gap": float(gaps[i]),
                    }
                    for i in indices.tolist()
                ],
                "plot": str(ds_fig_path),
            }
            (self.metrics_dir / f"beir_probe_{safe_name}_step_{step}.json").write_text(
                json.dumps(dataset_detail, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.writer.add_histogram("beir_probe/pos_cosine", pos_cos, step)
        self.writer.add_histogram("beir_probe/neg_cosine", neg_cos, step)
        self.writer.add_histogram("beir_probe/gap", gaps, step)
        self.writer.add_histogram("beir_probe/rank", rank_arr, step)
        print(
            f"[beir-probe] step={step} mode=bm25_hard elapsed_sec={time.perf_counter() - start_time:.1f} "
            f"pos_mean={pos_mean:.6f} neg_mean={neg_mean:.6f} "
            f"gap={gap_mean:.6f} gap_std={float(np.std(gaps)):.6f} "
            f"rank_mean={float(rank_arr.mean()) if len(rank_arr) else 0.0:.6f} "
            f"rank_median={float(np.median(rank_arr)) if len(rank_arr) else 0.0:.6f} "
            f"mrr@10={mrr_at_10 / denom:.6f} ndcg@10={ndcg_at_10 / denom:.6f} "
            f"interquery_ndcg@10={interquery_ndcg_at_10 / denom:.6f} "
            f"r@10={recall_at_10 / denom:.6f} plot={fig_path}",
            flush=True,
        )
        probe_detail = {
            "step": int(step),
            "summary": {
                "pos_cosine_mean": pos_mean,
                "pos_cosine_std": float(np.std(pos_cos)),
                "pos_cosine_min": float(np.min(pos_cos)),
                "pos_cosine_max": float(np.max(pos_cos)),
                "neg_cosine_mean": neg_mean,
                "neg_cosine_std": float(np.std(neg_cos)),
                "neg_cosine_min": float(np.min(neg_cos)),
                "neg_cosine_max": float(np.max(neg_cos)),
                "gap_mean": gap_mean,
                "gap_std": float(np.std(gaps)),
                "gap_min": float(np.min(gaps)),
                "gap_max": float(np.max(gaps)),
                "rank_mean": float(rank_arr.mean()) if len(rank_arr) else 0.0,
                "rank_median": float(np.median(rank_arr)) if len(rank_arr) else 0.0,
                "rank_min": int(rank_arr.min()) if len(rank_arr) else 0,
                "rank_max": int(rank_arr.max()) if len(rank_arr) else 0,
                "interquery_rank_mean": float(interquery_rank_arr.mean()) if len(interquery_rank_arr) else 0.0,
                "interquery_rank_median": float(np.median(interquery_rank_arr)) if len(interquery_rank_arr) else 0.0,
                "interquery_rank_min": int(interquery_rank_arr.min()) if len(interquery_rank_arr) else 0,
                "interquery_rank_max": int(interquery_rank_arr.max()) if len(interquery_rank_arr) else 0,
                "mrr": float(rr_arr.mean()) if len(rr_arr) else 0.0,
                "mrr@10": mrr_at_10 / denom,
                "ndcg@10": ndcg_at_10 / denom,
                "recall@10": recall_at_10 / denom,
                "interquery_mrr@10": interquery_mrr_at_10 / denom,
                "interquery_ndcg@10": interquery_ndcg_at_10 / denom,
                "interquery_recall@10": interquery_recall_at_10 / denom,
            },
            "datasets": dataset_summaries,
            "cases": [
                {
                    "idx": int(i),
                    "dataset": self._beir_probe_cases[i].get("dataset", "beir"),
                    "rank": int(ranks[i]),
                    "interquery_rank": int(interquery_ranks[i]),
                    "pos_cosine": float(pos_cos[i]),
                    "neg_cosine": float(neg_cos[i]),
                    "gap": float(gaps[i]),
                }
                for i in range(len(ranks))
            ],
        }
        (self.metrics_dir / f"beir_probe_step_{step}.json").write_text(
            json.dumps(probe_detail, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.writer.flush()
        metrics = {
            "beir_probe_pos_cosine_mean": pos_mean,
            "beir_probe_pos_cosine_std": float(np.std(pos_cos)),
            "beir_probe_pos_cosine_min": float(np.min(pos_cos)),
            "beir_probe_pos_cosine_max": float(np.max(pos_cos)),
            "beir_probe_neg_cosine_mean": neg_mean,
            "beir_probe_neg_cosine_std": float(np.std(neg_cos)),
            "beir_probe_neg_cosine_min": float(np.min(neg_cos)),
            "beir_probe_neg_cosine_max": float(np.max(neg_cos)),
            "beir_probe_gap_mean": gap_mean,
            "beir_probe_gap_std": float(np.std(gaps)),
            "beir_probe_gap_min": float(np.min(gaps)),
            "beir_probe_gap_max": float(np.max(gaps)),
            "beir_probe_rank_mean": float(rank_arr.mean()) if len(rank_arr) else 0.0,
            "beir_probe_rank_median": float(np.median(rank_arr)) if len(rank_arr) else 0.0,
            "beir_probe_rank_min": float(rank_arr.min()) if len(rank_arr) else 0.0,
            "beir_probe_rank_max": float(rank_arr.max()) if len(rank_arr) else 0.0,
            "beir_probe_interquery_rank_mean": float(interquery_rank_arr.mean()) if len(interquery_rank_arr) else 0.0,
            "beir_probe_interquery_rank_median": float(np.median(interquery_rank_arr)) if len(interquery_rank_arr) else 0.0,
            "beir_probe_interquery_rank_min": float(interquery_rank_arr.min()) if len(interquery_rank_arr) else 0.0,
            "beir_probe_interquery_rank_max": float(interquery_rank_arr.max()) if len(interquery_rank_arr) else 0.0,
            "beir_probe_mrr@10": mrr_at_10 / denom,
            "beir_probe_ndcg@10": ndcg_at_10 / denom,
            "beir_probe_recall@10": recall_at_10 / denom,
            "beir_interquery_mrr@10": interquery_mrr_at_10 / denom,
            "beir_interquery_ndcg@10": interquery_ndcg_at_10 / denom,
            "beir_interquery_recall@10": interquery_recall_at_10 / denom,
            "beir_proxy_mrr@10": interquery_mrr_at_10 / denom,
            "beir_proxy_ndcg@10": interquery_ndcg_at_10 / denom,
            "beir_proxy_recall@10": interquery_recall_at_10 / denom,
        }
        for dataset_name, summary in dataset_summaries.items():
            metric_prefix = "beir_probe_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)
            for key, value in summary.items():
                metrics[f"{metric_prefix}_{key}"] = float(value)
                self.writer.add_scalar(f"beir_probe_by_dataset/{dataset_name}/{key}", float(value), step)
        return metrics

    def _selection_score(self, metrics: dict[str, float]) -> float:
        if self.selection_metric in metrics:
            return float(metrics[self.selection_metric])
        if self.selection_metric == "gap" and "gap" in metrics:
            return float(metrics["gap"])
        if self.selection_metric.startswith("beir_full_"):
            return float("-inf")
        fallback_keys = ("beir_probe_ndcg@10", "beir_probe_mrr@10", "beir_interquery_ndcg@10", "gap")
        for key in fallback_keys:
            if key in metrics:
                print(
                    f"[selection] metric {self.selection_metric!r} missing; falling back to {key}={metrics[key]:.6f}",
                    flush=True,
                )
                return float(metrics[key])
        return float("-inf")

    def _validate_and_checkpoint(
        self,
        *,
        step: int,
        best_score: float,
        validations_without_improvement: int,
        force_mini_eval: bool = False,
        force_full_eval: bool = False,
        early_stopping_patience: int = 0,
    ) -> tuple[float, int, bool]:
        val_metrics = self.validate(step, force_mini_eval=force_mini_eval, force_full_eval=force_full_eval)
        score = self._selection_score(val_metrics)
        is_best = score > best_score
        self.save_checkpoint(step, val_metrics, is_best=is_best)
        if is_best:
            return score, 0, False
        validations_without_improvement += 1
        if early_stopping_patience > 0 and validations_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at step {step}: best_score={best_score:.6f}, current_score={score:.6f}",
                flush=True,
            )
            return best_score, validations_without_improvement, True
        return best_score, validations_without_improvement, False

    def _run_mini_eval_only(self, step: int) -> None:
        metrics = self.run_beir_eval(step, self.beir_mini_eval_config, "beir_mini")
        if not metrics:
            return
        print(
            f"[mini-eval] step={step} "
            + " ".join(
                f"{k}={v:.6f}"
                for k, v in metrics.items()
                if k.endswith("ndcg@10") or k.endswith("mrr@10") or k.endswith("recall@10")
            ),
            flush=True,
        )
        for key, value in metrics.items():
            self.writer.add_scalar(f"val/{key}", value, step)
        self.writer.flush()

    def train(self, num_epochs: int, val_every_n_steps: int, early_stopping_patience: int = 0) -> None:
        self.projector_config["epochs"] = num_epochs
        print(f"\n{'=' * 60}")
        target = f"{self.max_steps} steps" if self.max_steps > 0 else f"{num_epochs} epochs"
        print(f"Starting query distillation training for {target}")
        print(f"Output directory: {self.output_dir}")
        print(f"Validation every {val_every_n_steps} steps")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Selection metric: {self.selection_metric}")
        if early_stopping_patience > 0:
            print(f"Early stopping patience: {early_stopping_patience} validations")
        print(f"{'=' * 60}\n")
        if self.resume_step > 0:
            global_step = int(self.resume_step)
            micro_step = int(self.resume_step * self.gradient_accumulation_steps)
            best_score = self._selection_score(self.resume_metrics) if self.resume_metrics else float("-inf")
            print(
                f"[resume] continuing from step={global_step} with best_score={best_score:.6f}",
                flush=True,
            )
        else:
            val_metrics = self.validate(0)
            self.save_checkpoint(0, val_metrics, True)
            global_step = 0
            micro_step = 0
            best_score = self._selection_score(val_metrics)
        validations_without_improvement = 0
        self.optimizer.zero_grad(set_to_none=True)
        reached_max_steps = False
        if self.max_steps > 0 and global_step >= self.max_steps:
            reached_max_steps = True
            print(f"[schedule] already reached max_steps={self.max_steps} at step={global_step} before loop", flush=True)

        for epoch in range(1, num_epochs + 1):
            if reached_max_steps:
                break
            self.projector.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                micro_step += 1
                self._current_step = int(global_step)
                loss, metrics = self.compute_loss({k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()})
                (loss / self.gradient_accumulation_steps).backward()
                should_step = micro_step % self.gradient_accumulation_steps == 0
                grad_norm = float("nan")
                if should_step:
                    global_step += 1
                    self._step_lr(global_step)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.projector.parameters(), max_norm=1.0).item()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "q_mse": f"{metrics['query_mse']:.6f}",
                    "d_mse": f"{metrics['positive_mse']:.6f}",
                    "rank": f"{metrics['ranking_loss']:.6f}",
                    "gap": f"{metrics['gap']:.4f}",
                    "grad": f"{grad_norm:.4f}" if should_step else "accum",
                    "accum": f"{micro_step % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
                if should_step:
                    for key, value in metrics.items():
                        self.writer.add_scalar(f"train/{key}", value, global_step)
                    self.writer.add_scalar("train/loss_total", float(loss.item()), global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], global_step)
                    self.writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    self.writer.add_scalar("train/micro_step", micro_step, global_step)
                    self.writer.add_scalar("train/gradient_accumulation_steps", self.gradient_accumulation_steps, global_step)
                    if "sources" in batch:
                        source_counts = collections.Counter(batch["sources"])
                        total_in_batch = max(1, len(batch["sources"]))
                        for source_name, count in source_counts.items():
                            self.writer.add_scalar(
                                f"train/source_fraction/{source_name}",
                                float(count) / float(total_in_batch),
                                global_step,
                            )
                if should_step and val_every_n_steps > 0 and global_step % val_every_n_steps == 0:
                    print(f"[schedule] step={global_step} running validation checkpoint", flush=True)
                    best_score, validations_without_improvement, should_stop = self._validate_and_checkpoint(
                        step=global_step,
                        best_score=best_score,
                        validations_without_improvement=validations_without_improvement,
                        early_stopping_patience=early_stopping_patience,
                    )
                    if should_stop:
                        self.writer.close()
                        self.train_dataset.close()
                        self.val_dataset.close()
                        return
                    self.projector.train()
                elif (
                    should_step
                    and self.beir_mini_eval_every > 0
                    and global_step > 0
                    and global_step % self.beir_mini_eval_every == 0
                ):
                    print(f"[schedule] step={global_step} running mini BEIR only", flush=True)
                    self._run_mini_eval_only(global_step)
                    self.projector.train()
                if self.max_steps > 0 and global_step >= self.max_steps:
                    reached_max_steps = True
                    print(f"[schedule] reached max_steps={self.max_steps} at epoch={epoch} step={global_step}", flush=True)
                    break
            if global_step > 0:
                print(f"[epoch-end] epoch={epoch} step={global_step} running forced BEIR full eval", flush=True)
                best_score, validations_without_improvement, should_stop = self._validate_and_checkpoint(
                    step=global_step,
                    best_score=best_score,
                    validations_without_improvement=validations_without_improvement,
                    force_full_eval=True,
                    early_stopping_patience=early_stopping_patience,
                )
                if should_stop:
                    self.writer.close()
                    self.train_dataset.close()
                    self.val_dataset.close()
                    return
                self.projector.train()
            if reached_max_steps:
                print(f"[schedule] stopping after epoch-end eval at step={global_step}", flush=True)
                break
        self.writer.close()
        self.train_dataset.close()
        self.val_dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train query/document BGE distillation projector")
    parser.add_argument("--oscar-model", required=True)
    parser.add_argument("--teacher-embeddings", nargs="+", required=True)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--pooler", default="flatten")
    parser.add_argument("--projector-type", default="mem", choices=["mem", "dual_head", "token_dual"])
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--projector-hidden-dim", type=int, default=8192)
    parser.add_argument("--mem-tokens", type=int, default=8)
    parser.add_argument("--attn-dim", type=int, default=1536)
    parser.add_argument("--attn-layers", type=int, default=2)
    parser.add_argument("--attn-heads", type=int, default=8)
    parser.add_argument("--stage1-steps", type=int, default=0)
    parser.add_argument("--stage1-query-mse-weight", type=float, default=0.5)
    parser.add_argument("--stage1-doc-mse-weight", type=float, default=0.5)
    parser.add_argument("--stage1-negative-mse-weight", type=float, default=0.1)
    parser.add_argument("--stage2-query-mse-weight", type=float, default=0.2)
    parser.add_argument("--stage2-doc-mse-weight", type=float, default=0.2)
    parser.add_argument("--stage2-negative-mse-weight", type=float, default=0.0)
    parser.add_argument("--stage2-teacher-listwise-kl-weight", type=float, default=0.6)
    parser.add_argument("--stage2-ranking-weight", type=float, default=0.1)
    parser.add_argument("--stage2-infonce-weight", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "none", "legacy"])
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--query-mse-weight", type=float, default=1.0)
    parser.add_argument("--doc-mse-weight", type=float, default=1.0)
    parser.add_argument("--negative-mse-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.05)
    parser.add_argument("--infonce-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--beir-probe-config", default=None)
    parser.add_argument("--beir-probe-samples", type=int, default=20)
    parser.add_argument("--beir-probe-samples-per-dataset", type=int, default=None)
    parser.add_argument("--beir-probe-negatives", type=int, default=20)
    parser.add_argument("--beir-probe-batch-size", type=int, default=8)
    parser.add_argument("--beir-mini-eval-config", default=None)
    parser.add_argument("--beir-mini-eval-every", type=int, default=0)
    parser.add_argument("--beir-full-eval-config", default=None)
    parser.add_argument("--beir-full-eval-every", type=int, default=0)
    parser.add_argument("--selection-metric", default="beir_proxy_ndcg@10")
    parser.add_argument("--dataset-sampling-weights", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_lock = vars(args).copy()
    if args.dataset_sampling_weights:
        config_lock["dataset_sampling_weights"] = json.loads(args.dataset_sampling_weights)
    config_lock["teacher_embeddings"] = list(args.teacher_embeddings)
    config_lock["recipe"] = "query_distill"
    write_config_lock(config_lock, Path(args.output_dir).parent / "config.lock.yaml")
    trainer = QueryDistillationTrainer(
        oscar_model_name=args.oscar_model,
        teacher_embeddings_path=list(args.teacher_embeddings),
        embed_dim=args.embed_dim,
        pooler=args.pooler,
        projector_type=args.projector_type,
        num_layers=args.num_layers,
        dropout=args.dropout,
        projector_hidden_dim=args.projector_hidden_dim,
        mem_tokens=args.mem_tokens,
        attn_dim=args.attn_dim,
        attn_layers=args.attn_layers,
        attn_heads=args.attn_heads,
        stage1_steps=args.stage1_steps,
        stage1_query_mse_weight=args.stage1_query_mse_weight,
        stage1_doc_mse_weight=args.stage1_doc_mse_weight,
        stage1_negative_mse_weight=args.stage1_negative_mse_weight,
        stage2_query_mse_weight=args.stage2_query_mse_weight,
        stage2_doc_mse_weight=args.stage2_doc_mse_weight,
        stage2_negative_mse_weight=args.stage2_negative_mse_weight,
        stage2_teacher_listwise_kl_weight=args.stage2_teacher_listwise_kl_weight,
        stage2_ranking_weight=args.stage2_ranking_weight,
        stage2_infonce_weight=args.stage2_infonce_weight,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        scheduler=args.scheduler,
        val_split=args.val_split,
        query_mse_weight=args.query_mse_weight,
        doc_mse_weight=args.doc_mse_weight,
        negative_mse_weight=args.negative_mse_weight,
        ranking_weight=args.ranking_weight,
        infonce_weight=args.infonce_weight,
        temperature=args.temperature,
        margin=args.margin,
        beir_probe_config=args.beir_probe_config,
        beir_probe_samples=args.beir_probe_samples,
        beir_probe_samples_per_dataset=args.beir_probe_samples_per_dataset,
        beir_probe_negatives=args.beir_probe_negatives,
        beir_probe_batch_size=args.beir_probe_batch_size,
        beir_mini_eval_config=args.beir_mini_eval_config,
        beir_mini_eval_every=args.beir_mini_eval_every,
        beir_full_eval_config=args.beir_full_eval_config,
        beir_full_eval_every=args.beir_full_eval_every,
        selection_metric=args.selection_metric,
        dataset_sampling_weights=json.loads(args.dataset_sampling_weights) if args.dataset_sampling_weights else None,
        resume_checkpoint=args.resume_checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    trainer.train(args.epochs, args.val_every, args.early_stopping_patience)


if __name__ == "__main__":
    main()
