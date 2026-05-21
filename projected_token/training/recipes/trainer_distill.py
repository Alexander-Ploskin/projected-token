#!/usr/bin/env python3
"""Trainer for蒸馏 (Distillation) from SFR-Embedding-Mistral.

This trainer:
1. Loads pre-computed teacher embeddings from HDF5
2. Trains projector to match teacher embeddings using true MSE loss
3. Uses OSCAR as student model, SFR-Embedding-Mistral as teacher
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import h5py
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from projected_token.encoders.projector import DistillationProjector, MEMProjector
from projected_token.io import write_json, write_csv
from projected_token.artifacts import metrics_to_rows
from projected_token.plotting import plot_training_curves
from projected_token.artifacts import create_run_layout, write_config_lock
from projected_token.oscar_runtime import disable_transformers_allocator_warmup, configure_oscar_component_devices


def _resolve_h5_paths(paths_or_patterns: Union[str, Sequence[str]]) -> list[str]:
    """Resolve explicit HDF5 paths, directories and glob patterns."""
    raw_paths = [paths_or_patterns] if isinstance(paths_or_patterns, str) else list(paths_or_patterns)
    resolved: list[str] = []
    for raw in raw_paths:
        expanded = os.path.expandvars(os.path.expanduser(str(raw)))
        path = Path(expanded)
        matches: list[str]
        if path.is_dir():
            matches = sorted(str(p) for p in path.glob("*.h5"))
        elif glob.has_magic(expanded):
            matches = sorted(glob.glob(expanded))
        else:
            matches = [expanded]
        resolved.extend(matches)

    # Preserve order while removing duplicates.
    deduped = list(dict.fromkeys(resolved))
    missing = [p for p in deduped if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Teacher embedding file(s) not found: {missing}")
    if not deduped:
        raise FileNotFoundError(f"No teacher embedding .h5 files matched: {paths_or_patterns}")
    return deduped


def _load_distillation_checkpoint(trainer: "DistillationTrainer", checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.projector.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if trainer.teacher_adapter is not None and "teacher_adapter_state_dict" in checkpoint:
        trainer.teacher_adapter.load_state_dict(checkpoint["teacher_adapter_state_dict"], strict=False)


class H5DistillationDataset(Dataset):
    """Dataset that reads from HDF5 file with teacher embeddings."""
    
    def __init__(self, h5_path: str, split: str = "train", val_split: float = 0.05):
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, 'r')
        
        total_samples = self.file['embeddings'].shape[0]
        val_size = int(total_samples * val_split)
        if val_split > 0 and total_samples > 1:
            val_size = max(1, val_size)
        train_size = total_samples - val_size
        
        if split == "train":
            self.indices = list(range(0, train_size))
        else:
            self.indices = list(range(train_size, total_samples))
        
        print(f"Loaded {split} split: {len(self.indices)} / {total_samples} samples")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Lazy load text from HDF5
        text = self.file['texts'][real_idx]
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        
        # Load teacher embedding
        target_embed = self.file['embeddings'][real_idx]
        
        return {
            "text": text,
            "target": torch.tensor(target_embed, dtype=torch.float32),
        }
    
    def close(self):
        self.file.close()


class ConcatH5DistillationDataset(Dataset):
    """Multiple HDF5 teacher files; each file is split train/val independently (same val_split)."""

    def __init__(self, h5_paths: Sequence[str], split: str = "train", val_split: float = 0.05):
        self.h5_paths = [str(p) for p in h5_paths]
        self.files = [h5py.File(p, "r") for p in self.h5_paths]
        self.index_map: list[tuple[int, int]] = []

        for fi, f in enumerate(self.files):
            total = int(f["embeddings"].shape[0])
            val_size = int(total * val_split)
            if val_split > 0 and total > 1:
                val_size = max(1, val_size)
            train_size = total - val_size
            if split == "train":
                for j in range(0, train_size):
                    self.index_map.append((fi, j))
            else:
                for j in range(train_size, total):
                    self.index_map.append((fi, j))

        names = [Path(p).name for p in self.h5_paths]
        print(f"Loaded concat {split} split: {len(self.index_map)} / files {names}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        fi, real_idx = self.index_map[idx]
        f = self.files[fi]
        text = f["texts"][real_idx]
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        target_embed = f["embeddings"][real_idx]
        return {
            "text": text,
            "target": torch.tensor(target_embed, dtype=torch.float32),
        }

    def close(self):
        for fh in self.files:
            fh.close()


def _train_val_h5_datasets(
    teacher_embeddings_path: Union[str, Sequence[str]],
    val_split: float,
) -> tuple[Dataset, Dataset]:
    paths = _resolve_h5_paths(teacher_embeddings_path)
    if len(paths) == 1:
        train_ds = H5DistillationDataset(paths[0], split="train", val_split=val_split)
        val_ds = H5DistillationDataset(paths[0], split="val", val_split=val_split)
    else:
        train_ds = ConcatH5DistillationDataset(paths, split="train", val_split=val_split)
        val_ds = ConcatH5DistillationDataset(paths, split="val", val_split=val_split)
    return train_ds, val_ds


def collate_fn_h5(batch):
    """Collate function for HDF5 dataset."""
    texts = [item["text"] for item in batch]
    targets = torch.stack([item["target"] for item in batch])
    return {"texts": texts, "targets": targets}


class DistillationTrainer:
    """Trainer for distillation from SFR-Embedding-Mistral."""
    
    def __init__(
        self,
        oscar_model_name: str,
        teacher_embeddings_path: Union[str, Sequence[str]],
        embed_dim: int = 768,
        pooler: str = "mean",
        num_layers: int = 2,
        dropout: float = 0.1,
        projector_hidden_dim: int = 8192,
        projector_type: str = "mem",
        batch_size: int = 128,
        lr: float = 1e-3,
        val_split: float = 0.05,
        infonce_weight: float = 1.0,
        margin_mse_weight: float = 0.0,
        mse_weight: float = 0.1,
        temperature: float = 0.02,
        margin: float = 0.1,
        hard_negative_weight: float = 0.0,
        hard_negative_margin: float = 0.05,
        selection_metric: str = "mse_loss",
        max_steps: int = 0,
        warmup_steps: int = 0,
        min_lr: float = 1e-5,
        scheduler: str = "legacy",
        proxy_eval_every_n_steps: int = 0,
        proxy_eval_config: Optional[str] = None,
        beir_probe_config: Optional[str] = None,
        beir_probe_samples: int = 20,
        beir_probe_negatives: int = 20,
        beir_probe_batch_size: int = 16,
        async_validation: bool = False,
        async_validation_device: Optional[str] = None,
        async_validation_max_pending: int = 1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/projector_distill",
        log_dir: str = "./logs/projector_distill",
    ):
        self.device = torch.device(device)
        self.oscar_model_name = str(oscar_model_name)
        self.val_split = float(val_split)
        
        disable_transformers_allocator_warmup()
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()
        configure_oscar_component_devices(self.oscar_model)
        
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False
        
        hidden_size = self.oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")
        
        self.batch_size = batch_size
        self.infonce_weight = float(infonce_weight)
        self.margin_mse_weight = float(margin_mse_weight)
        self.mse_weight = float(mse_weight)
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.hard_negative_weight = float(hard_negative_weight)
        self.hard_negative_margin = float(hard_negative_margin)
        self.selection_metric = str(selection_metric)
        self.max_steps = int(max_steps or 0)
        self.warmup_steps = int(warmup_steps or 0)
        self.min_lr = float(min_lr)
        self.scheduler_name = str(scheduler)
        self.proxy_eval_every_n_steps = int(proxy_eval_every_n_steps)
        self.proxy_eval_config = proxy_eval_config
        self.beir_probe_config = beir_probe_config or proxy_eval_config
        self.beir_probe_samples = int(beir_probe_samples)
        self.beir_probe_negatives = int(beir_probe_negatives)
        self.beir_probe_batch_size = int(beir_probe_batch_size)
        self.async_validation = bool(async_validation)
        self.async_validation_device = async_validation_device
        self.async_validation_max_pending = int(async_validation_max_pending)
        self._async_validation_jobs: list[dict[str, Any]] = []
        self._beir_probe_cases: list[dict[str, str]] = []
        
        # Load datasets (one or more HDF5 files for multi-domain distillation)
        print(f"Loading teacher embeddings from: {teacher_embeddings_path}")
        self.train_dataset, self.val_dataset = _train_val_h5_datasets(
            teacher_embeddings_path, val_split=val_split
        )
        teacher_dim = int(self.train_dataset[0]["target"].shape[-1])

        if projector_type == "mem":
            self.projector = MEMProjector(
                hidden_dim=hidden_size,
                embed_dim=embed_dim,
                pooler=pooler,
                num_layers=num_layers,
                dropout=dropout,
                projector_hidden_dim=projector_hidden_dim,
            ).to(device=self.device, dtype=torch.bfloat16)
        elif projector_type == "distillation":
            self.projector = DistillationProjector(
                oscar_hidden_dim=hidden_size,
                embed_dim=embed_dim,
                hidden_dim=projector_hidden_dim,
                num_layers=num_layers,
                use_normalize=True,
            ).to(device=self.device, dtype=torch.bfloat16)
        else:
            raise ValueError(f"Unknown projector_type: {projector_type}")

        # If dimensions already match, avoid free adapter that can cause collapse.
        self.teacher_adapter: nn.Module | None = None
        if teacher_dim != embed_dim:
            self.teacher_adapter = nn.Linear(teacher_dim, embed_dim).to(device=self.device, dtype=torch.bfloat16)
            print(f"Teacher dim ({teacher_dim}) != embed dim ({embed_dim}); using trainable teacher adapter.")
        else:
            print(f"Teacher dim matches embed dim ({embed_dim}); using raw teacher embeddings (no adapter).")
        print("Distillation projector architecture:")
        print(self.projector)
        total_params = sum(p.numel() for p in self.projector.parameters())
        adapter_params = sum(p.numel() for p in self.teacher_adapter.parameters()) if self.teacher_adapter is not None else 0
        print(f"Projector parameters: {total_params:,}, teacher adapter: {adapter_params:,}")
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn_h5,
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn_h5,
            num_workers=2,
        )
        
        optim_params = list(self.projector.parameters())
        if self.teacher_adapter is not None:
            optim_params.extend(self.teacher_adapter.parameters())
        self.base_lr = float(lr)
        self.optimizer = torch.optim.AdamW(optim_params, lr=self.base_lr, weight_decay=0.01)
        
        self.scheduler = None
        if self.max_steps <= 0:
            # Backward-compatible epoch-level scheduler for existing configs.
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10, eta_min=1e-5
            )
        
        self.projector_config = {
            "oscar_hidden_dim": hidden_size,
            "oscar_model_name": self.oscar_model_name,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "projector_hidden_dim": projector_hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "projector_type": projector_type,
            "teacher_dim": teacher_dim,
            "teacher_adapter_enabled": self.teacher_adapter is not None,
            "infonce_weight": self.infonce_weight,
            "margin_mse_weight": self.margin_mse_weight,
            "mse_weight": self.mse_weight,
            "temperature": self.temperature,
            "margin": self.margin,
            "hard_negative_weight": self.hard_negative_weight,
            "hard_negative_margin": self.hard_negative_margin,
            "selection_metric": self.selection_metric,
            "max_steps": self.max_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "scheduler": self.scheduler_name,
            "proxy_eval_every_n_steps": self.proxy_eval_every_n_steps,
            "proxy_eval_config": self.proxy_eval_config,
            "beir_probe_config": self.beir_probe_config,
            "beir_probe_samples": self.beir_probe_samples,
            "beir_probe_negatives": self.beir_probe_negatives,
            "beir_probe_batch_size": self.beir_probe_batch_size,
            "async_validation": self.async_validation,
            "async_validation_device": self.async_validation_device,
            "async_validation_max_pending": self.async_validation_max_pending,
            "use_normalize": True,
        }
        
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.run_root = Path(output_dir).parent if Path(output_dir).name == "checkpoints" else Path(output_dir)
        self.metrics_dir = self.run_root / "metrics"
        self.plots_dir = self.run_root / "plots"
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float('inf')
        self.best_selection_score: float | None = None
        self.train_history: list[dict[str, float]] = []
        self.val_history: list[dict[str, float]] = []

        self._beir_probe_cases = self._load_beir_probe_cases()
        
        self.load_checkpoint(output_dir)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load best checkpoint if exists."""
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            print(f"Loading checkpoint from {best_path}")
            checkpoint = torch.load(best_path, map_location=self.device)
            ckpt_has_adapter = "teacher_adapter_state_dict" in checkpoint
            if self.teacher_adapter is None and ckpt_has_adapter:
                print(
                    "Checkpoint was created with teacher_adapter enabled, but current run "
                    "uses raw teacher targets. Skipping checkpoint resume."
                )
                return
            self.projector.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if self.teacher_adapter is not None and "teacher_adapter_state_dict" in checkpoint:
                self.teacher_adapter.load_state_dict(checkpoint["teacher_adapter_state_dict"], strict=False)
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    
    def encode_documents(self, texts):
        """Encode documents using OSCAR + projector."""
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
        embeddings = self.projector(mem_embeddings)
        return embeddings
    
    def _teacher_project(self, teacher_embeds: torch.Tensor) -> torch.Tensor:
        teacher_f = teacher_embeds.float()
        if self.teacher_adapter is not None:
            teacher_proj = self.teacher_adapter(teacher_f.to(dtype=torch.bfloat16)).float()
        else:
            teacher_proj = teacher_f
        return teacher_proj

    def _infonce_loss(self, student_f: torch.Tensor, teacher_normed: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(student_f, teacher_normed.T) / max(self.temperature, 1e-6)
        labels = torch.arange(student_f.size(0), device=student_f.device)
        return F.cross_entropy(logits, labels)

    def _margin_mse_loss(self, student_f: torch.Tensor, teacher_normed: torch.Tensor) -> torch.Tensor:
        if student_f.size(0) < 2:
            return torch.tensor(0.0, device=student_f.device)
        s_sim = torch.matmul(student_f, student_f.T)
        t_sim = torch.matmul(teacher_normed, teacher_normed.T)
        mask = ~torch.eye(student_f.size(0), dtype=torch.bool, device=student_f.device)
        return F.mse_loss(s_sim[mask], t_sim[mask])

    def compute_distillation_loss(self, student_embeds, teacher_embeds):
        """Compute retrieval-aware objective + diagnostics."""
        student_f = F.normalize(student_embeds.float(), p=2, dim=-1)
        teacher_proj = self._teacher_project(teacher_embeds)

        # Track teacher adapter scale before normalization for debugging.
        teacher_raw_norm = teacher_proj.norm(p=2, dim=-1).mean()

        # Projector output is L2-normalized, so compare in the same space.
        teacher_normed = F.normalize(teacher_proj, p=2, dim=-1)

        mse_loss = F.mse_loss(student_f, teacher_normed)
        infonce_loss = self._infonce_loss(student_f, teacher_normed)
        margin_mse = self._margin_mse_loss(student_f, teacher_normed)
        cosine_sim = (student_f * teacher_normed).sum(dim=-1).mean()
        cosine_loss = 1.0 - cosine_sim

        student_norm = student_f.norm(p=2, dim=-1).mean()
        hardneg_loss = torch.tensor(0.0, device=student_f.device)
        if student_f.size(0) > 1:
            perm = torch.randperm(teacher_normed.size(0), device=teacher_normed.device)
            teacher_shuffled = teacher_normed[perm]
            cosine_shuffled = (student_f * teacher_shuffled).sum(dim=-1).mean()
            cosine_shuffled_val = cosine_shuffled.item()
            hardneg_loss = torch.relu(self.hard_negative_margin - cosine_sim + cosine_shuffled)
        else:
            cosine_shuffled_val = 0.0

        total_loss = (
            self.infonce_weight * infonce_loss
            + self.margin_mse_weight * margin_mse
            + self.mse_weight * mse_loss
            + self.hard_negative_weight * hardneg_loss
        )

        return (
            total_loss,
            mse_loss.item(),
            infonce_loss.item(),
            margin_mse.item(),
            hardneg_loss.item(),
            cosine_sim.item(),
            cosine_loss.item(),
            cosine_shuffled_val,
            student_norm.item(),
            teacher_raw_norm.item(),
        )

    def compute_retrieval_metrics(self, student_embeds: torch.Tensor, teacher_embeds: torch.Tensor) -> Dict[str, float]:
        """Compute in-batch retrieval metrics to compare distill vs contrastive."""
        student_f = F.normalize(student_embeds.float(), p=2, dim=-1)
        teacher_proj = self._teacher_project(teacher_embeds)
        teacher_normed = F.normalize(teacher_proj, p=2, dim=-1)

        similarities = torch.matmul(student_f, teacher_normed.T)
        ranks = torch.argsort(torch.argsort(similarities, dim=1, descending=True), dim=1)

        mrr = 0.0
        mrr_at_10 = 0.0
        ndcg_at_10 = 0.0
        recall_at_1 = 0.0
        recall_at_5 = 0.0
        recall_at_10 = 0.0
        batch_size = student_f.size(0)

        for i in range(batch_size):
            rank = ranks[i, i].item() + 1
            mrr += 1.0 / rank
            if rank <= 10:
                mrr_at_10 += 1.0 / rank
                ndcg_at_10 += 1.0 / float(np.log2(rank + 1.0))
            if rank == 1:
                recall_at_1 += 1.0
            if rank <= 5:
                recall_at_5 += 1.0
            if rank <= 10:
                recall_at_10 += 1.0

        mrr /= batch_size
        mrr_at_10 /= batch_size
        ndcg_at_10 /= batch_size
        recall_at_1 /= batch_size
        recall_at_5 /= batch_size
        recall_at_10 /= batch_size

        return {
            "mrr": mrr,
            "mrr@10": mrr_at_10,
            "ndcg@10": ndcg_at_10,
            "recall@1": recall_at_1,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
        }

    def _step_lr(self, step: int) -> float:
        if self.max_steps <= 0 or self.scheduler_name in {"legacy", "none"}:
            return float(self.optimizer.param_groups[0]["lr"])
        if self.scheduler_name != "cosine":
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            lr = self.base_lr * step / max(1, self.warmup_steps)
        else:
            denom = max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / denom))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return float(lr)

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _load_beir_probe_cases(self) -> list[dict[str, str]]:
        if not self.beir_probe_config:
            return []
        try:
            from projected_token.config import load_yaml
        except Exception as exc:
            print(f"[beir-probe] cannot import load_yaml: {exc}")
            return []
        cfg = load_yaml(self.beir_probe_config)
        datasets = cfg.get("datasets", [])
        if not datasets:
            print("[beir-probe] no datasets configured")
            return []
        dataset_item = datasets[0]
        dataset_name = str(dataset_item.get("name", "beir"))
        dataset_dir = Path(dataset_item["path"])
        qrels_path = dataset_dir / "qrels" / f"{cfg.get('split', 'test')}.tsv"
        try:
            corpus_rows = self._read_jsonl(dataset_dir / "corpus.jsonl")
            query_rows = self._read_jsonl(dataset_dir / "queries.jsonl")
        except Exception as exc:
            print(f"[beir-probe] cannot load corpus/queries from {dataset_dir}: {exc}")
            return []

        corpus = {
            str(r["_id"]): " ".join(
                p for p in [str(r.get("title", "")).strip(), str(r.get("text", "")).strip()] if p
            ).strip()
            for r in corpus_rows
        }
        queries = {str(r["_id"]): str(r.get("text", "")) for r in query_rows}
        relevant: dict[str, set[str]] = {}
        try:
            with qrels_path.open("r", encoding="utf-8") as f:
                header = f.readline()
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    qid, cid, score = parts[0], parts[1], parts[2]
                    if int(score) <= 0:
                        continue
                    relevant.setdefault(str(qid), set()).add(str(cid))
        except Exception as exc:
            print(f"[beir-probe] cannot read qrels {qrels_path}: {exc}")
            return []

        corpus_ids = list(corpus.keys())
        candidate_qids = [qid for qid in relevant.keys() if qid in queries]
        if not candidate_qids or not corpus_ids:
            return []
        rng = np.random.default_rng(42)
        rng.shuffle(candidate_qids)
        cases: list[dict[str, str]] = []
        for qid in candidate_qids:
            pos_ids = list(relevant.get(qid, set()))
            if not pos_ids:
                continue
            pos_id = pos_ids[0]
            pos_text = corpus.get(pos_id, "")
            if not pos_text:
                continue
            neg_texts: list[str] = []
            for cand in corpus_ids:
                if cand not in relevant[qid]:
                    neg_text = corpus.get(cand, "")
                    if neg_text:
                        neg_texts.append(neg_text)
                    if len(neg_texts) >= self.beir_probe_negatives:
                        break
            if not neg_texts:
                continue
            cases.append(
                {
                    "dataset": dataset_name,
                    "qid": qid,
                    "query": queries[qid],
                    "positive": pos_text,
                    "negative": neg_texts[0],
                    "negatives": neg_texts,
                }
            )
            if len(cases) >= self.beir_probe_samples:
                break
        print(f"[beir-probe] loaded {len(cases)} cases from {dataset_name}")
        return cases

    def run_beir_probe(self, step: int) -> Dict[str, float]:
        if not self._beir_probe_cases:
            return {}
        def encode_probe_texts(texts: list[str]) -> torch.Tensor:
            chunks: list[torch.Tensor] = []
            batch_size = max(1, self.beir_probe_batch_size)
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                chunks.append(F.normalize(self.encode_documents(batch).float(), p=2, dim=-1))
            return torch.cat(chunks, dim=0)

        with torch.no_grad():
            queries = [c["query"] for c in self._beir_probe_cases]
            positives = [c["positive"] for c in self._beir_probe_cases]
            negatives = [c["negative"] for c in self._beir_probe_cases]
            q_emb = encode_probe_texts(queries)
            p_emb = encode_probe_texts(positives)
            n_emb = encode_probe_texts(negatives)
            pos_cos = (q_emb * p_emb).sum(dim=-1).cpu().numpy()
            neg_cos = (q_emb * n_emb).sum(dim=-1).cpu().numpy()
            candidate_texts: list[str] = []
            candidate_offsets: list[tuple[int, int]] = []
            for case in self._beir_probe_cases:
                start = len(candidate_texts)
                candidate_texts.append(case["positive"])
                candidate_texts.extend(case.get("negatives", [case["negative"]]))
                candidate_offsets.append((start, len(candidate_texts)))
            cand_emb = encode_probe_texts(candidate_texts)
            similarities = torch.matmul(q_emb, cand_emb.T).cpu().numpy()
        gap = pos_cos - neg_cos
        pos_mean = float(np.mean(pos_cos))
        neg_mean = float(np.mean(neg_cos))
        gap_mean = float(np.mean(gap))
        ranks: list[int] = []
        recall_at_1 = 0.0
        recall_at_5 = 0.0
        recall_at_10 = 0.0
        mrr = 0.0
        mrr_at_10 = 0.0
        for idx, (start, stop) in enumerate(candidate_offsets):
            local_scores = similarities[idx, start:stop]
            # The positive document is always the first local candidate.
            pos_rank = int(np.where(np.argsort(-local_scores) == 0)[0][0]) + 1
            ranks.append(pos_rank)
            mrr += 1.0 / pos_rank
            if pos_rank <= 10:
                mrr_at_10 += 1.0 / pos_rank
            if pos_rank <= 1:
                recall_at_1 += 1.0
            if pos_rank <= 5:
                recall_at_5 += 1.0
            if pos_rank <= 10:
                recall_at_10 += 1.0
        denom = max(1, len(candidate_offsets))
        mrr /= denom
        mrr_at_10 /= denom
        recall_at_1 /= denom
        recall_at_5 /= denom
        recall_at_10 /= denom
        self.writer.add_scalar("beir_probe/pos_cosine_mean", pos_mean, step)
        self.writer.add_scalar("beir_probe/neg_cosine_mean", neg_mean, step)
        self.writer.add_scalar("beir_probe/gap_mean", gap_mean, step)
        self.writer.add_scalar("beir_probe/mrr", mrr, step)
        self.writer.add_scalar("beir_probe/mrr@10", mrr_at_10, step)
        self.writer.add_scalar("beir_probe/recall@1", recall_at_1, step)
        self.writer.add_scalar("beir_probe/recall@5", recall_at_5, step)
        self.writer.add_scalar("beir_probe/recall@10", recall_at_10, step)

        x = np.arange(len(pos_cos))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, pos_cos, marker="o", label="positive cosine")
        ax.plot(x, neg_cos, marker="x", label="negative cosine")
        ax.axhline(pos_mean, linestyle="--", linewidth=1.0, label=f"pos mean={pos_mean:.3f}")
        ax.axhline(neg_mean, linestyle="--", linewidth=1.0, label=f"neg mean={neg_mean:.3f}")
        ax.set_title(f"BEIR probe cosine @ step {step}")
        ax.set_xlabel("probe example idx")
        ax.set_ylabel("cosine similarity")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        self.writer.add_figure("beir_probe/cosine_plot", fig, global_step=step)
        fig_path = self.plots_dir / f"beir_probe_step_{step}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        print(
            f"[beir-probe] step={step} pos_mean={pos_mean:.6f} neg_mean={neg_mean:.6f} gap={gap_mean:.6f} "
            f"mrr@10={mrr_at_10:.6f} r@1={recall_at_1:.6f} r@5={recall_at_5:.6f} r@10={recall_at_10:.6f} "
            f"plot={fig_path}"
        )
        return {
            "beir_probe_pos_cosine_mean": pos_mean,
            "beir_probe_neg_cosine_mean": neg_mean,
            "beir_probe_gap_mean": gap_mean,
            "beir_probe_mrr": mrr,
            "beir_probe_mrr@10": mrr_at_10,
            "beir_probe_recall@1": recall_at_1,
            "beir_probe_recall@5": recall_at_5,
            "beir_probe_recall@10": recall_at_10,
        }
    
    def train_epoch(
        self,
        epoch: int,
        *,
        start_step: int = 0,
        stop_after_steps: int | None = None,
        val_every_n_steps: int = 0,
        best_val_loss: float | None = None,
        epochs_without_improvement: int = 0,
        early_stopping_patience: int = 0,
    ) -> Dict[str, float]:
        """Train one epoch."""
        self.projector.train()
        if self.teacher_adapter is not None:
            self.teacher_adapter.train()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_infonce = 0.0
        total_margin_mse = 0.0
        total_hardneg = 0.0
        total_cosine = 0.0
        num_batches = 0
        best_val_loss_value = float("inf") if best_val_loss is None else best_val_loss
        stop_requested = False
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for batch in pbar:
            if stop_after_steps is not None and num_batches >= stop_after_steps:
                break
            global_step = start_step + num_batches + 1
            if self.max_steps > 0:
                self._step_lr(global_step)
            texts = batch["texts"]
            teacher_embeds = batch["targets"].to(self.device, dtype=torch.bfloat16)
            
            student_embeds = self.encode_documents(texts)
            
            (
                loss,
                mse_loss,
                infonce_loss,
                margin_mse,
                hardneg_loss,
                cosine_sim,
                cosine_loss,
                cosine_shuffled,
                s_norm,
                t_norm,
            ) = self.compute_distillation_loss(student_embeds, teacher_embeds)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Debug: check if gradients exist
            grad_norm = 0.0
            for p in self.projector.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            torch.nn.utils.clip_grad_norm_(self.projector.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Debug: track weight changes
            if num_batches == 1 and epoch == 1:
                self._initial_weights = {n: p.clone() for n, p in self.projector.named_parameters()}
            
            loss_val = loss.item() if hasattr(loss, 'item') else loss
            total_loss += loss_val
            total_mse_loss += mse_loss
            total_infonce += infonce_loss
            total_margin_mse += margin_mse
            total_hardneg += hardneg_loss
            total_cosine += cosine_sim
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss_val:.6f}",
                "mse": f"{mse_loss:.6f}",
                "nce": f"{infonce_loss:.6f}",
                "mmse": f"{margin_mse:.6f}",
                "hn": f"{hardneg_loss:.6f}",
                "cos": f"{cosine_sim:.4f}",
                "cos_loss": f"{cosine_loss:.6f}",
                "cos_shuf": f"{cosine_shuffled:.4f}",
                "grad": f"{grad_norm:.4f}",
                "s_norm": f"{s_norm:.4f}",
                "t_norm": f"{t_norm:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Keep generic loss tag for backward-compatible dashboards.
            self.writer.add_scalar("train/loss", loss_val, global_step)
            self.writer.add_scalar("train/loss_total", loss_val, global_step)
            self.writer.add_scalar("train/loss_mse", mse_loss, global_step)
            self.writer.add_scalar("train/loss_infonce", infonce_loss, global_step)
            self.writer.add_scalar("train/loss_margin_mse", margin_mse, global_step)
            self.writer.add_scalar("train/loss_hardneg", hardneg_loss, global_step)
            self.writer.add_scalar("train/mse_loss", mse_loss, global_step)
            self.writer.add_scalar("train/cosine_sim", cosine_sim, global_step)
            self.writer.add_scalar("train/cosine_loss", cosine_loss, global_step)
            self.writer.add_scalar("train/cosine_sim_shuffled", cosine_shuffled, global_step)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], global_step)

            if (
                self.max_steps > 0
                and val_every_n_steps > 0
                and global_step % val_every_n_steps == 0
                and global_step < self.max_steps
            ):
                best_val_loss_value, epochs_without_improvement, _ = self._run_validation_checkpoint(
                    global_step, best_val_loss_value, epochs_without_improvement
                )
                self.projector.train()
                if self.teacher_adapter is not None:
                    self.teacher_adapter.train()
                if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                    stop_requested = True
                    break
        
        if num_batches == 0:
            return {
                "loss_total": 0.0,
                "mse_loss": 0.0,
                "infonce_loss": 0.0,
                "margin_mse_loss": 0.0,
                "hardneg_loss": 0.0,
                "mse_component": 0.0,
                "cosine_sim": 0.0,
                "cosine_loss": 1.0,
                "steps": 0,
            "best_val_loss": best_val_loss_value,
            "epochs_without_improvement": epochs_without_improvement,
            "stop_requested": stop_requested,
            }

        avg_loss = total_loss / num_batches
        avg_mse = total_mse_loss / num_batches
        avg_infonce = total_infonce / num_batches
        avg_margin_mse = total_margin_mse / num_batches
        avg_hardneg = total_hardneg / num_batches
        avg_cosine = total_cosine / num_batches
        return {
            "loss_total": avg_loss,
            "mse_loss": avg_mse,
            "infonce_loss": avg_infonce,
            "margin_mse_loss": avg_margin_mse,
            "hardneg_loss": avg_hardneg,
            "mse_component": avg_mse,
            "cosine_sim": avg_cosine,
            "cosine_loss": 1.0 - avg_cosine,
            "steps": num_batches,
            "best_val_loss": best_val_loss_value,
            "epochs_without_improvement": epochs_without_improvement,
            "stop_requested": stop_requested,
        }
    
    def validate(self, step: int) -> Dict[str, float]:
        """Validate on validation set."""
        self.projector.eval()
        if self.teacher_adapter is not None:
            self.teacher_adapter.eval()
        total_loss = 0.0
        total_mse_loss = 0.0
        total_infonce = 0.0
        total_margin_mse = 0.0
        total_hardneg = 0.0
        total_cosine = 0.0
        total_cosine_shuffled = 0.0
        total_s_norm = 0.0
        total_t_norm = 0.0
        total_mrr = 0.0
        total_mrr_at_10 = 0.0
        total_ndcg_at_10 = 0.0
        total_recall_at_1 = 0.0
        total_recall_at_5 = 0.0
        total_recall_at_10 = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", 
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                texts = batch["texts"]
                teacher_embeds = batch["targets"].to(self.device, dtype=torch.bfloat16)
                
                student_embeds = self.encode_documents(texts)
                (
                    loss,
                    mse_loss,
                    infonce_loss,
                    margin_mse,
                    hardneg_loss,
                    cosine_sim,
                    cosine_loss,
                    cosine_shuffled,
                    s_norm,
                    t_norm,
                ) = self.compute_distillation_loss(student_embeds, teacher_embeds)
                retrieval_metrics = self.compute_retrieval_metrics(student_embeds, teacher_embeds)
                
                total_loss += loss.item() if hasattr(loss, 'item') else loss
                total_mse_loss += mse_loss
                total_infonce += infonce_loss
                total_margin_mse += margin_mse
                total_hardneg += hardneg_loss
                total_cosine += cosine_sim
                total_cosine_shuffled += cosine_shuffled
                total_s_norm += s_norm
                total_t_norm += t_norm
                total_mrr += retrieval_metrics["mrr"]
                total_mrr_at_10 += retrieval_metrics["mrr@10"]
                total_ndcg_at_10 += retrieval_metrics["ndcg@10"]
                total_recall_at_1 += retrieval_metrics["recall@1"]
                total_recall_at_5 += retrieval_metrics["recall@5"]
                total_recall_at_10 += retrieval_metrics["recall@10"]
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_mse = total_mse_loss / num_batches
        avg_infonce = total_infonce / num_batches
        avg_margin_mse = total_margin_mse / num_batches
        avg_hardneg = total_hardneg / num_batches
        avg_cosine = total_cosine / num_batches
        avg_cosine_shuffled = total_cosine_shuffled / num_batches
        avg_s_norm = total_s_norm / num_batches
        avg_t_norm = total_t_norm / num_batches
        avg_mrr = total_mrr / num_batches
        avg_mrr_at_10 = total_mrr_at_10 / num_batches
        avg_ndcg_at_10 = total_ndcg_at_10 / num_batches
        avg_recall_at_1 = total_recall_at_1 / num_batches
        avg_recall_at_5 = total_recall_at_5 / num_batches
        avg_recall_at_10 = total_recall_at_10 / num_batches
        
        print(f"\nValidation @ step {step}:")
        print(f"  Loss total: {avg_loss:.6f}")
        print(f"  MSE:        {avg_mse:.6f}")
        print(f"  InfoNCE:    {avg_infonce:.6f}")
        print(f"  MarginMSE:  {avg_margin_mse:.6f}")
        print(f"  HardNeg:    {avg_hardneg:.6f}")
        print(f"  Cosine Sim: {avg_cosine:.6f}")
        print(f"  Cosine Shuffled: {avg_cosine_shuffled:.6f}")
        print(f"  MRR: {avg_mrr:.6f}, MRR@10: {avg_mrr_at_10:.6f}, NDCG@10: {avg_ndcg_at_10:.6f}")
        print(f"  R@1: {avg_recall_at_1:.6f}, R@5: {avg_recall_at_5:.6f}, R@10: {avg_recall_at_10:.6f}")
        print(f"  Student norm: {avg_s_norm:.4f}, Teacher norm: {avg_t_norm:.4f}")
        
        # Keep generic loss tag for backward-compatible dashboards.
        self.writer.add_scalar("val/loss", avg_loss, step)
        self.writer.add_scalar("val/loss_total", avg_loss, step)
        self.writer.add_scalar("val/loss_mse", avg_mse, step)
        self.writer.add_scalar("val/loss_infonce", avg_infonce, step)
        self.writer.add_scalar("val/loss_margin_mse", avg_margin_mse, step)
        self.writer.add_scalar("val/loss_hardneg", avg_hardneg, step)
        self.writer.add_scalar("val/mse_loss", avg_mse, step)
        self.writer.add_scalar("val/cosine_sim", avg_cosine, step)
        self.writer.add_scalar("val/cosine_loss", 1.0 - avg_cosine, step)
        self.writer.add_scalar("val/cosine_sim_shuffled", avg_cosine_shuffled, step)
        self.writer.add_scalar("val/mrr", avg_mrr, step)
        self.writer.add_scalar("val/mrr@10", avg_mrr_at_10, step)
        self.writer.add_scalar("val/ndcg@10", avg_ndcg_at_10, step)
        self.writer.add_scalar("val/recall@1", avg_recall_at_1, step)
        self.writer.add_scalar("val/recall@5", avg_recall_at_5, step)
        self.writer.add_scalar("val/recall@10", avg_recall_at_10, step)
        probe_metrics = self.run_beir_probe(step)

        out = {
            "loss_total": avg_loss,
            "mse_loss": avg_mse,
            "mse_component": avg_mse,
            "infonce_loss": avg_infonce,
            "margin_mse_loss": avg_margin_mse,
            "hardneg_loss": avg_hardneg,
            "cosine_sim": avg_cosine,
            "cosine_loss": 1.0 - avg_cosine,
            "cosine_sim_shuffled": avg_cosine_shuffled,
            "mrr": avg_mrr,
            "mrr@10": avg_mrr_at_10,
            "ndcg@10": avg_ndcg_at_10,
            "recall@1": avg_recall_at_1,
            "recall@5": avg_recall_at_5,
            "recall@10": avg_recall_at_10,
        }
        out.update(probe_metrics)
        return out
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "model_state_dict": self.projector.state_dict(),
            "config": self.projector_config,
        }
        if self.teacher_adapter is not None:
            checkpoint["teacher_adapter_state_dict"] = self.teacher_adapter.state_dict()
        
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.projector_config, f, indent=2)

        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"val": metrics}, f, indent=2)
        
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            
            best_config_path = os.path.join(self.output_dir, "best_config.json")
            with open(best_config_path, 'w') as f:
                json.dump(self.projector_config, f, indent=2)
            print(f"Saved best model to {best_path}")

    def _poll_async_validation_jobs(self, *, wait: bool = False) -> list[dict[str, Any]]:
        finished: list[dict[str, Any]] = []
        still_running: list[dict[str, Any]] = []
        for job in self._async_validation_jobs:
            proc: subprocess.Popen[str] = job["process"]
            if wait:
                proc.wait()
            if proc.poll() is None:
                still_running.append(job)
                continue
            metrics_path = Path(job["metrics_path"])
            metrics: dict[str, Any] = {}
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[async-val] failed to read metrics for step={job['step']}: {exc}")
            else:
                print(f"[async-val] missing metrics for step={job['step']} at {metrics_path}")
            if proc.returncode != 0:
                print(f"[async-val] step={job['step']} exited with code {proc.returncode}; log={job['log_path']}")
            else:
                print(f"[async-val] step={job['step']} complete; log={job['log_path']}")
            finished.append({"step": job["step"], "metrics": metrics, **job})
        self._async_validation_jobs = still_running
        return finished

    def _handle_finished_async_validation_jobs(
        self,
        best_val_loss: float,
        epochs_without_improvement: int,
    ) -> tuple[float, int, bool]:
        any_finished = False
        for result in self._poll_async_validation_jobs():
            metrics = result.get("metrics", {})
            if not metrics:
                continue
            any_finished = True
            result_step = int(result["step"])
            self.val_history.append({"step": result_step, **metrics})
            current_selection = self._selection_score(metrics)
            is_best = self.best_selection_score is None or current_selection > self.best_selection_score
            if is_best:
                best_val_loss = float(metrics.get("mse_loss", best_val_loss))
                self.best_selection_score = current_selection
                epochs_without_improvement = 0
                source = Path(self.output_dir) / f"checkpoint_step_{result_step}.pt"
                target = Path(self.output_dir) / "best_model.pt"
                if source.exists():
                    shutil.copy2(source, target)
                with (Path(self.output_dir) / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump({"val": metrics}, f, indent=2)
                print(
                    "  >> New async best! "
                    f"Val MSE: {best_val_loss:.6f}, "
                    f"MRR@10: {metrics.get('mrr@10', 0.0):.6f}, R@10: {metrics.get('recall@10', 0.0):.6f}, "
                    f"select={self.selection_metric}:{current_selection:.6f}"
                )
            else:
                epochs_without_improvement += 1
                print(f"  Async validation no improvement for {epochs_without_improvement} validation(s)")
        return best_val_loss, epochs_without_improvement, any_finished

    def _launch_async_validation(self, step: int) -> None:
        if not self.async_validation:
            return
        if len(self._async_validation_jobs) >= max(1, self.async_validation_max_pending):
            print(f"[async-val] skip step={step}; pending={len(self._async_validation_jobs)}")
            return
        ckpt_metrics = {"async_validation_launched": True, "step": step}
        self.save_checkpoint(step, ckpt_metrics, False)
        checkpoint_path = Path(self.output_dir) / f"checkpoint_step_{step}.pt"
        async_dir = self.run_root / "async_validation" / f"step_{step}"
        async_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = async_dir / "metrics.json"
        log_path = async_dir / "validate.log"
        teacher_paths = self.train_dataset.h5_paths if hasattr(self.train_dataset, "h5_paths") else [str(getattr(self.train_dataset, "h5_path"))]
        command = [
            sys.executable,
            "-m",
            "projected_token.training.recipes.trainer_distill",
            "validate-checkpoint",
            "--checkpoint",
            str(checkpoint_path),
            "--metrics-output",
            str(metrics_path),
            "--oscar-model",
            str(getattr(self, "oscar_model_name", "naver/oscar-qwen2-7B")),
            "--teacher-embeddings",
            *teacher_paths,
            "--embed-dim",
            str(self.projector_config["embed_dim"]),
            "--pooler",
            str(self.projector_config["pooler"]),
            "--dropout",
            str(self.projector_config["dropout"]),
            "--projector-hidden-dim",
            str(self.projector_config["projector_hidden_dim"]),
            "--num-layers",
            str(self.projector_config["num_layers"]),
            "--projector-type",
            str(self.projector_config["projector_type"]),
            "--batch-size",
            str(self.batch_size),
            "--val-split",
            str(self.val_split),
            "--device",
            self.async_validation_device or str(self.device),
            "--beir-probe-config",
            str(self.beir_probe_config or ""),
            "--beir-probe-samples",
            str(self.beir_probe_samples),
            "--beir-probe-negatives",
            str(self.beir_probe_negatives),
            "--beir-probe-batch-size",
            str(self.beir_probe_batch_size),
            "--step",
            str(step),
        ]
        env = os.environ.copy()
        if self.async_validation_device:
            env["CUDA_VISIBLE_DEVICES"] = self.async_validation_device.split(":", 1)[-1]
            # The spawned process sees only one GPU, so address it as cuda:0.
            command[command.index("--device") + 1] = "cuda:0"
        with log_path.open("w", encoding="utf-8") as log_fp:
            proc = subprocess.Popen(command, stdout=log_fp, stderr=subprocess.STDOUT, text=True, env=env)
        self._async_validation_jobs.append(
            {"step": step, "process": proc, "metrics_path": str(metrics_path), "log_path": str(log_path)}
        )
        print(f"[async-val] launched step={step} pid={proc.pid} device={self.async_validation_device} log={log_path}")

    def _export_reports(self) -> None:
        run_id = self.run_root.name
        write_json(self.metrics_dir / "distill_train_history.json", self.train_history)
        write_json(self.metrics_dir / "distill_val_history.json", self.val_history)
        train_rows = []
        for row in self.train_history:
            train_rows.extend(
                metrics_to_rows(
                    {
                        "mse_loss": row["mse_loss"],
                        "cosine_sim": row["cosine_sim"],
                        "cosine_loss": row.get("cosine_loss", 1.0 - row["cosine_sim"]),
                        "cosine_sim_shuffled": row.get("cosine_sim_shuffled", 0.0),
                    },
                    run_id=run_id,
                    dataset="distill_train",
                    split=f"epoch_{row['epoch']}",
                )
            )
        val_rows = []
        for row in self.val_history:
            val_rows.extend(
                metrics_to_rows(
                    {
                        "mse_loss": row["mse_loss"],
                        "cosine_sim": row["cosine_sim"],
                        "cosine_loss": row.get("cosine_loss", 1.0 - row["cosine_sim"]),
                        "cosine_sim_shuffled": row.get("cosine_sim_shuffled", 0.0),
                        "mrr": row.get("mrr", 0.0),
                        "mrr@10": row.get("mrr@10", 0.0),
                        "ndcg@10": row.get("ndcg@10", 0.0),
                        "recall@1": row.get("recall@1", 0.0),
                        "recall@5": row.get("recall@5", 0.0),
                        "recall@10": row.get("recall@10", 0.0),
                    },
                    run_id=run_id,
                    dataset="distill_val",
                    split=f"step_{row['step']}",
                )
            )
        write_csv(self.metrics_dir / "distill_train_history.csv", train_rows)
        write_csv(self.metrics_dir / "distill_val_history.csv", val_rows)
        plot_training_curves(
            self.train_history,
            x_key="epoch",
            output_path=self.plots_dir / "distill_training_curves.png",
            title="Distillation Train Curves",
            metric_keys=["mse_loss", "cosine_sim", "cosine_loss"],
        )
        plot_training_curves(
            self.val_history,
            x_key="step",
            output_path=self.plots_dir / "distill_validation_curves.png",
            title="Distillation Validation Curves",
            metric_keys=[
                "mse_loss",
                "cosine_sim",
                "cosine_loss",
                "cosine_sim_shuffled",
                "mrr",
                "mrr@10",
                "ndcg@10",
                "recall@1",
                "recall@5",
                "recall@10",
            ],
        )
    
    def _run_validation_checkpoint(self, step: int, best_val_loss: float, epochs_without_improvement: int) -> tuple[float, int, bool]:
        if self.async_validation:
            best_val_loss, epochs_without_improvement, _ = self._handle_finished_async_validation_jobs(
                best_val_loss, epochs_without_improvement
            )
            self._launch_async_validation(step)
            return best_val_loss, epochs_without_improvement, False
        val_metrics = self.validate(step)
        self.val_history.append({"step": step, **val_metrics})

        if self.proxy_eval_every_n_steps > 0 and self.proxy_eval_config and (step % self.proxy_eval_every_n_steps == 0):
            self.save_checkpoint(step, val_metrics, False)
            proxy_metrics = self.run_proxy_eval(step)
            if proxy_metrics:
                val_metrics.update(proxy_metrics)
                self.val_history[-1].update(proxy_metrics)

        current_selection = self._selection_score(val_metrics)
        is_best = self.best_selection_score is None or current_selection > self.best_selection_score
        if is_best:
            best_val_loss = val_metrics["mse_loss"]
            self.best_selection_score = current_selection
            epochs_without_improvement = 0
            print(
                "  >> New best! "
                f"Val MSE: {best_val_loss:.6f}, Cosine: {val_metrics['cosine_sim']:.6f}, "
                f"MRR@10: {val_metrics['mrr@10']:.6f}, R@10: {val_metrics['recall@10']:.6f}, "
                f"select={self.selection_metric}:{current_selection:.6f}"
            )
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} validation(s)")

        self.save_checkpoint(step, val_metrics, is_best)
        return best_val_loss, epochs_without_improvement, is_best

    def train(self, num_epochs: int, val_every_n_steps: int = 500, early_stopping_patience: int = 3):
        """Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            val_every_n_steps: Validate every N steps
            early_stopping_patience: Stop if no improvement for this many validation checks
        """
        print(f"\n{'='*60}")
        target = f"{self.max_steps} steps" if self.max_steps > 0 else f"{num_epochs} epochs"
        print(f"Starting distillation training for {target}")
        print(f"Output directory: {self.output_dir}")
        print(f"Validation every {val_every_n_steps} steps")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*60}\n")
        
        step = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Initial validation
        print("--- Initial Validation ---")
        if self.async_validation:
            self._launch_async_validation(0)
        else:
            val_metrics = self.validate(0)
            self.val_history.append({"step": 0, **val_metrics})
            self.save_checkpoint(0, val_metrics, True)
            best_val_loss = val_metrics['mse_loss']
            self.best_selection_score = self._selection_score(val_metrics)
        
        epoch = 0
        while True:
            if self.max_steps > 0 and step >= self.max_steps:
                break
            if self.max_steps <= 0 and epoch >= num_epochs:
                break
            best_val_loss, epochs_without_improvement, _ = self._handle_finished_async_validation_jobs(
                best_val_loss, epochs_without_improvement
            )

            epoch += 1
            remaining = self.max_steps - step if self.max_steps > 0 else None
            train_metrics = self.train_epoch(
                epoch,
                start_step=step,
                stop_after_steps=remaining,
                val_every_n_steps=val_every_n_steps,
                best_val_loss=best_val_loss,
                epochs_without_improvement=epochs_without_improvement,
                early_stopping_patience=early_stopping_patience,
            )
            steps_done = int(train_metrics.pop("steps", 0))
            best_val_loss = float(train_metrics.pop("best_val_loss", best_val_loss))
            epochs_without_improvement = int(train_metrics.pop("epochs_without_improvement", epochs_without_improvement))
            stop_requested = bool(train_metrics.pop("stop_requested", False))
            if steps_done <= 0:
                break
            step += steps_done
            self.train_history.append({"epoch": epoch, "step": step, **train_metrics})
            print(f"\nEpoch {epoch} summary:")
            print(f"  Train MSE: {train_metrics['mse_loss']:.6f}, Cosine: {train_metrics['cosine_sim']:.6f}")
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            should_validate = self.max_steps <= 0 or step >= self.max_steps or (val_every_n_steps > 0 and step % val_every_n_steps == 0)
            if should_validate:
                best_val_loss, epochs_without_improvement, _ = self._run_validation_checkpoint(
                    step, best_val_loss, epochs_without_improvement
                )
            
            # Early stopping check
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"\n*** Early stopping: no improvement for {early_stopping_patience} validations ***")
                break
            if stop_requested:
                print("\n*** Early stopping requested during epoch ***")
                break
            
            self.projector.train()
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best val MSE: {best_val_loss:.6f}")
        if self.best_selection_score is not None:
            print(f"Best selection score ({self.selection_metric}): {self.best_selection_score:.6f}")
        print(f"{'='*60}")
        self._poll_async_validation_jobs(wait=True)
        best_val_loss, epochs_without_improvement, _ = self._handle_finished_async_validation_jobs(
            best_val_loss, epochs_without_improvement
        )
        self._export_reports()
        self.writer.close()
        
        # Close HDF5 files
        self.train_dataset.close()
        self.val_dataset.close()

    def _selection_score(self, metrics: Dict[str, float]) -> float:
        metric = self.selection_metric
        if metric == "proxy_ndcg@10":
            return float(metrics.get("proxy_ndcg@10", float("-inf")))
        if metric == "proxy_mrr@10":
            return float(metrics.get("proxy_mrr@10", float("-inf")))
        if metric == "ndcg@10":
            return float(metrics.get("ndcg@10", float("-inf")))
        if metric == "mrr@10":
            return float(metrics.get("mrr@10", float("-inf")))
        if metric == "mse_loss":
            return -float(metrics.get("mse_loss", float("inf")))
        return float(metrics.get(metric, float("-inf")))

    def run_proxy_eval(self, step: int) -> Dict[str, float]:
        if not self.proxy_eval_config:
            return {}
        try:
            from projected_token.config import load_yaml
            from projected_token.retrieval.beir import evaluate_beir
        except Exception as exc:
            print(f"[proxy-eval] import failed: {exc}")
            return {}

        cfg = load_yaml(self.proxy_eval_config)
        encoder_cfg = cfg.get("encoder", {})
        encoder_kwargs = dict(encoder_cfg.get("kwargs", {}))
        current_checkpoint = Path(self.output_dir) / f"checkpoint_step_{step}.pt"
        encoder_kwargs["projector_path"] = str(current_checkpoint if current_checkpoint.exists() else Path(self.output_dir) / "best_model.pt")
        encoder_cfg["kwargs"] = encoder_kwargs
        cfg["encoder"] = encoder_cfg
        cfg["max_queries_per_dataset"] = int(cfg.get("max_queries_per_dataset", 100))
        try:
            proxy_summary = evaluate_beir(cfg)
        except Exception as exc:
            print(f"[proxy-eval] skipped at step={step}: {exc}")
            return {}
        avg = proxy_summary.get("average", {})
        proxy_ndcg10 = float(avg.get("ndcg@10", 0.0))
        proxy_mrr10 = float(avg.get("mrr@10", 0.0))
        self.writer.add_scalar("proxy/ndcg@10", proxy_ndcg10, step)
        self.writer.add_scalar("proxy/mrr@10", proxy_mrr10, step)
        print(f"[proxy-eval] step={step} ndcg@10={proxy_ndcg10:.6f} mrr@10={proxy_mrr10:.6f}")
        return {"proxy_ndcg@10": proxy_ndcg10, "proxy_mrr@10": proxy_mrr10}


def validate_checkpoint_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Validate a distillation checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--oscar-model", required=True)
    parser.add_argument("--teacher-embeddings", nargs="+", required=True)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--pooler", default="mean")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--projector-hidden-dim", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--projector-type", choices=["mem", "distillation"], default="mem")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--beir-probe-config", default=None)
    parser.add_argument("--beir-probe-samples", type=int, default=20)
    parser.add_argument("--beir-probe-negatives", type=int, default=20)
    parser.add_argument("--beir-probe-batch-size", type=int, default=16)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args(argv)

    trainer = DistillationTrainer(
        oscar_model_name=args.oscar_model,
        teacher_embeddings_path=list(args.teacher_embeddings),
        embed_dim=args.embed_dim,
        pooler=args.pooler,
        dropout=args.dropout,
        projector_hidden_dim=args.projector_hidden_dim,
        num_layers=args.num_layers,
        projector_type=args.projector_type,
        batch_size=args.batch_size,
        val_split=args.val_split,
        infonce_weight=0.0,
        margin_mse_weight=0.0,
        mse_weight=1.0,
        hard_negative_weight=0.0,
        selection_metric="mse_loss",
        proxy_eval_every_n_steps=0,
        proxy_eval_config=None,
        beir_probe_config=(args.beir_probe_config or None),
        beir_probe_samples=args.beir_probe_samples,
        beir_probe_negatives=args.beir_probe_negatives,
        beir_probe_batch_size=args.beir_probe_batch_size,
        device=args.device,
        output_dir=str(Path(args.metrics_output).parent / "checkpoints"),
        log_dir=str(Path(args.metrics_output).parent / "logs"),
    )
    _load_distillation_checkpoint(trainer, args.checkpoint)
    metrics = trainer.validate(args.step)
    output_path = Path(args.metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trainer.writer.close()
    trainer.train_dataset.close()
    trainer.val_dataset.close()


def main():
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == "validate-checkpoint":
        validate_checkpoint_main(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="Train projector via distillation from SFR")
    
    parser.add_argument("--oscar-model", type=str,
                        default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument(
        "--teacher-embeddings",
        nargs="+",
        default=["/data/teacher-embeddings/teacher_embeddings_100k.h5"],
        help="One or more HDF5 files with keys texts, embeddings (same teacher dim).",
    )
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--pooler", type=str, default="mean")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--projector-hidden-dim", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--projector-type", type=str, default="mem", choices=["mem", "distillation"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--infonce-weight", type=float, default=1.0)
    parser.add_argument("--margin-mse-weight", type=float, default=0.0)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--selection-metric", type=str, default="mse_loss")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="legacy", choices=["legacy", "none", "cosine"])
    parser.add_argument("--proxy-eval-every-n-steps", type=int, default=0)
    parser.add_argument("--proxy-eval-config", type=str, default=None)
    parser.add_argument("--beir-probe-config", type=str, default=None)
    parser.add_argument("--beir-probe-samples", type=int, default=20)
    parser.add_argument("--beir-probe-negatives", type=int, default=20)
    parser.add_argument("--beir-probe-batch-size", type=int, default=16)
    parser.add_argument("--async-validation", action="store_true")
    parser.add_argument("--async-validation-device", type=str, default=None)
    parser.add_argument("--async-validation-max-pending", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Stop if no improvement for this many validations")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str,
                        default="./checkpoints/projector_distill")
    parser.add_argument("--log-dir", type=str,
                        default="./logs/projector_distill")
    parser.add_argument("--run-id", type=str, default=None)
    
    args = parser.parse_args()
    if args.output_dir and args.log_dir:
        output_dir = Path(args.output_dir)
        log_dir = Path(args.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        run_root = output_dir.parent if output_dir.name == "checkpoints" else output_dir
    else:
        layout = create_run_layout("distill", run_id=args.run_id)
        output_dir = layout.checkpoints_dir
        log_dir = layout.logs_dir
        run_root = layout.root

    config_lock = {
        "recipe": "distill",
        "oscar_model": args.oscar_model,
        "teacher_embeddings": list(args.teacher_embeddings),
        "embed_dim": args.embed_dim,
        "pooler": args.pooler,
        "dropout": args.dropout,
        "projector_hidden_dim": args.projector_hidden_dim,
        "num_layers": args.num_layers,
        "projector_type": args.projector_type,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_split": args.val_split,
        "infonce_weight": args.infonce_weight,
        "margin_mse_weight": args.margin_mse_weight,
        "mse_weight": args.mse_weight,
        "temperature": args.temperature,
        "margin": args.margin,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_negative_margin": args.hard_negative_margin,
        "selection_metric": args.selection_metric,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "min_lr": args.min_lr,
        "scheduler": args.scheduler,
        "proxy_eval_every_n_steps": args.proxy_eval_every_n_steps,
        "proxy_eval_config": args.proxy_eval_config,
        "beir_probe_config": args.beir_probe_config,
        "beir_probe_samples": args.beir_probe_samples,
        "beir_probe_negatives": args.beir_probe_negatives,
        "beir_probe_batch_size": args.beir_probe_batch_size,
        "async_validation": args.async_validation,
        "async_validation_device": args.async_validation_device,
        "async_validation_max_pending": args.async_validation_max_pending,
        "epochs": args.epochs,
        "val_every": args.val_every,
        "early_stopping_patience": args.early_stopping_patience,
        "device": args.device,
    }
    write_config_lock(config_lock, run_root / "config.lock.yaml")
    
    trainer = DistillationTrainer(
        oscar_model_name=args.oscar_model,
        teacher_embeddings_path=list(args.teacher_embeddings),
        embed_dim=args.embed_dim,
        pooler=args.pooler,
        dropout=args.dropout,
        projector_hidden_dim=args.projector_hidden_dim,
        num_layers=args.num_layers,
        projector_type=args.projector_type,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        infonce_weight=args.infonce_weight,
        margin_mse_weight=args.margin_mse_weight,
        mse_weight=args.mse_weight,
        temperature=args.temperature,
        margin=args.margin,
        hard_negative_weight=args.hard_negative_weight,
        hard_negative_margin=args.hard_negative_margin,
        selection_metric=args.selection_metric,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        min_lr=args.min_lr,
        scheduler=args.scheduler,
        proxy_eval_every_n_steps=args.proxy_eval_every_n_steps,
        proxy_eval_config=args.proxy_eval_config,
        beir_probe_config=args.beir_probe_config,
        beir_probe_samples=args.beir_probe_samples,
        beir_probe_negatives=args.beir_probe_negatives,
        beir_probe_batch_size=args.beir_probe_batch_size,
        async_validation=args.async_validation,
        async_validation_device=args.async_validation_device,
        async_validation_max_pending=args.async_validation_max_pending,
        device=args.device,
        output_dir=str(output_dir),
        log_dir=str(log_dir),
    )
    
    trainer.train(num_epochs=args.epochs, val_every_n_steps=args.val_every, early_stopping_patience=args.early_stopping_patience)


if __name__ == "__main__":
    main()