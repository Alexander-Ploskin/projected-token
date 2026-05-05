#!/usr/bin/env python3
"""Trainer for蒸馏 (Distillation) from SFR-Embedding-Mistral.

This trainer:
1. Loads pre-computed teacher embeddings from HDF5
2. Trains projector to match teacher embeddings using true MSE loss
3. Uses OSCAR as student model, SFR-Embedding-Mistral as teacher
"""

import argparse
import json
import os
import h5py
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Union

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


class H5DistillationDataset(Dataset):
    """Dataset that reads from HDF5 file with teacher embeddings."""
    
    def __init__(self, h5_path: str, split: str = "train", val_split: float = 0.05):
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, 'r')
        
        total_samples = self.file['embeddings'].shape[0]
        val_size = int(total_samples * val_split)
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
    paths = [teacher_embeddings_path] if isinstance(teacher_embeddings_path, str) else list(teacher_embeddings_path)
    paths = [str(p) for p in paths]
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
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/projector_distill",
        log_dir: str = "./logs/projector_distill",
    ):
        self.device = torch.device(device)
        
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

        self.teacher_adapter = nn.Linear(teacher_dim, embed_dim).to(device=self.device, dtype=torch.bfloat16)
        print("Distillation projector architecture:")
        print(self.projector)
        total_params = sum(p.numel() for p in self.projector.parameters())
        adapter_params = sum(p.numel() for p in self.teacher_adapter.parameters())
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
        
        self.optimizer = torch.optim.AdamW(
            list(self.projector.parameters()) + list(self.teacher_adapter.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        
        # Cosine scheduler with warmup
        warmup_steps = len(self.train_loader)  # 1 epoch warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-5
        )
        
        self.projector_config = {
            "oscar_hidden_dim": hidden_size,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "projector_hidden_dim": projector_hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "projector_type": projector_type,
            "teacher_dim": teacher_dim,
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
        self.train_history: list[dict[str, float]] = []
        self.val_history: list[dict[str, float]] = []
        
        self.load_checkpoint(output_dir)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load best checkpoint if exists."""
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            print(f"Loading checkpoint from {best_path}")
            checkpoint = torch.load(best_path, map_location=self.device)
            self.projector.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "teacher_adapter_state_dict" in checkpoint:
                self.teacher_adapter.load_state_dict(checkpoint["teacher_adapter_state_dict"], strict=False)
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    
    def encode_documents(self, texts):
        """Encode documents using OSCAR + projector."""
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
        embeddings = self.projector(mem_embeddings)
        return embeddings
    
    def compute_distillation_loss(self, student_embeds, teacher_embeds):
        """Compute true MSE distillation loss + cosine diagnostics."""
        student_f = F.normalize(student_embeds.float(), p=2, dim=-1)
        teacher_f = teacher_embeds.float()
        teacher_proj = self.teacher_adapter(teacher_f.to(dtype=torch.bfloat16)).float()

        # Track teacher adapter scale before normalization for debugging.
        teacher_raw_norm = teacher_proj.norm(p=2, dim=-1).mean()

        # Projector output is L2-normalized, so compare in the same space.
        teacher_normed = F.normalize(teacher_proj, p=2, dim=-1)

        mse_loss = F.mse_loss(student_f, teacher_normed)
        cosine_sim = (student_f * teacher_normed).sum(dim=-1).mean()
        cosine_loss = 1.0 - cosine_sim

        student_norm = student_f.norm(p=2, dim=-1).mean()
        if student_f.size(0) > 1:
            perm = torch.randperm(teacher_normed.size(0), device=teacher_normed.device)
            teacher_shuffled = teacher_normed[perm]
            cosine_shuffled = (student_f * teacher_shuffled).sum(dim=-1).mean()
            cosine_shuffled_val = cosine_shuffled.item()
        else:
            cosine_shuffled_val = 0.0

        return (
            mse_loss,
            cosine_sim.item(),
            cosine_loss.item(),
            cosine_shuffled_val,
            student_norm.item(),
            teacher_raw_norm.item(),
        )

    def compute_retrieval_metrics(self, student_embeds: torch.Tensor, teacher_embeds: torch.Tensor) -> Dict[str, float]:
        """Compute in-batch retrieval metrics to compare distill vs contrastive."""
        student_f = F.normalize(student_embeds.float(), p=2, dim=-1)
        teacher_proj = self.teacher_adapter(teacher_embeds.float().to(dtype=torch.bfloat16)).float()
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.projector.train()
        self.teacher_adapter.train()
        total_loss = 0.0
        total_cosine = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for batch in pbar:
            texts = batch["texts"]
            teacher_embeds = batch["targets"].to(self.device, dtype=torch.bfloat16)
            
            student_embeds = self.encode_documents(texts)
            
            loss, cosine_sim, cosine_loss, cosine_shuffled, s_norm, t_norm = self.compute_distillation_loss(student_embeds, teacher_embeds)
            
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
            total_cosine += cosine_sim
            num_batches += 1
            
            pbar.set_postfix({
                "mse": f"{loss_val:.6f}",
                "cos": f"{cosine_sim:.4f}",
                "cos_loss": f"{cosine_loss:.6f}",
                "cos_shuf": f"{cosine_shuffled:.4f}",
                "grad": f"{grad_norm:.4f}",
                "s_norm": f"{s_norm:.4f}",
                "t_norm": f"{t_norm:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            global_step = (epoch - 1) * len(self.train_loader) + num_batches
            # Keep generic loss tag for backward-compatible dashboards.
            self.writer.add_scalar("train/loss", loss_val, global_step)
            self.writer.add_scalar("train/mse_loss", loss_val, global_step)
            self.writer.add_scalar("train/cosine_sim", cosine_sim, global_step)
            self.writer.add_scalar("train/cosine_loss", cosine_loss, global_step)
            self.writer.add_scalar("train/cosine_sim_shuffled", cosine_shuffled, global_step)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        avg_cosine = total_cosine / num_batches
        return {
            "mse_loss": avg_loss,
            "cosine_sim": avg_cosine,
            "cosine_loss": 1.0 - avg_cosine,
        }
    
    def validate(self, step: int) -> Dict[str, float]:
        """Validate on validation set."""
        self.projector.eval()
        self.teacher_adapter.eval()
        total_loss = 0.0
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
                loss, cosine_sim, cosine_loss, cosine_shuffled, s_norm, t_norm = self.compute_distillation_loss(student_embeds, teacher_embeds)
                retrieval_metrics = self.compute_retrieval_metrics(student_embeds, teacher_embeds)
                
                total_loss += loss.item() if hasattr(loss, 'item') else loss
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
        print(f"  MSE Loss:   {avg_loss:.6f}")
        print(f"  Cosine Sim: {avg_cosine:.6f}")
        print(f"  Cosine Shuffled: {avg_cosine_shuffled:.6f}")
        print(f"  MRR: {avg_mrr:.6f}, MRR@10: {avg_mrr_at_10:.6f}, NDCG@10: {avg_ndcg_at_10:.6f}")
        print(f"  R@1: {avg_recall_at_1:.6f}, R@5: {avg_recall_at_5:.6f}, R@10: {avg_recall_at_10:.6f}")
        print(f"  Student norm: {avg_s_norm:.4f}, Teacher norm: {avg_t_norm:.4f}")
        
        # Keep generic loss tag for backward-compatible dashboards.
        self.writer.add_scalar("val/loss", avg_loss, step)
        self.writer.add_scalar("val/mse_loss", avg_loss, step)
        self.writer.add_scalar("val/cosine_sim", avg_cosine, step)
        self.writer.add_scalar("val/cosine_loss", 1.0 - avg_cosine, step)
        self.writer.add_scalar("val/cosine_sim_shuffled", avg_cosine_shuffled, step)
        self.writer.add_scalar("val/mrr", avg_mrr, step)
        self.writer.add_scalar("val/mrr@10", avg_mrr_at_10, step)
        self.writer.add_scalar("val/ndcg@10", avg_ndcg_at_10, step)
        self.writer.add_scalar("val/recall@1", avg_recall_at_1, step)
        self.writer.add_scalar("val/recall@5", avg_recall_at_5, step)
        self.writer.add_scalar("val/recall@10", avg_recall_at_10, step)

        return {
            "mse_loss": avg_loss,
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
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "model_state_dict": self.projector.state_dict(),
            "teacher_adapter_state_dict": self.teacher_adapter.state_dict(),
            "config": self.projector_config,
        }
        
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
    
    def train(self, num_epochs: int, val_every_n_steps: int = 500, early_stopping_patience: int = 3):
        """Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            val_every_n_steps: Validate every N steps
            early_stopping_patience: Stop if no improvement for this many validation checks
        """
        print(f"\n{'='*60}")
        print(f"Starting distillation training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Validation every {val_every_n_steps} steps")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"{'='*60}\n")
        
        step = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Initial validation
        print("--- Initial Validation ---")
        val_metrics = self.validate(0)
        self.val_history.append({"step": 0, **val_metrics})
        self.save_checkpoint(0, val_metrics, True)
        best_val_loss = val_metrics['mse_loss']
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            self.train_history.append({"epoch": epoch, **train_metrics})
            print(f"\nEpoch {epoch} summary:")
            print(f"  Train MSE: {train_metrics['mse_loss']:.6f}, Cosine: {train_metrics['cosine_sim']:.6f}")
            
            self.scheduler.step()
            
            # Validate at end of epoch
            step += len(self.train_loader)
            val_metrics = self.validate(step)
            self.val_history.append({"step": step, **val_metrics})
            
            is_best = val_metrics['mse_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['mse_loss']
                epochs_without_improvement = 0
                print(
                    "  >> New best! "
                    f"Val MSE: {best_val_loss:.6f}, Cosine: {val_metrics['cosine_sim']:.6f}, "
                    f"MRR@10: {val_metrics['mrr@10']:.6f}, R@10: {val_metrics['recall@10']:.6f}"
                )
            else:
                epochs_without_improvement += 1
                print(f"  No improvement for {epochs_without_improvement} validation(s)")
            
            self.save_checkpoint(step, val_metrics, is_best)
            
            # Early stopping check
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n*** Early stopping: no improvement for {early_stopping_patience} validations ***")
                break
            
            self.projector.train()
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best val MSE: {best_val_loss:.6f}")
        print(f"{'='*60}")
        self._export_reports()
        self.writer.close()
        
        # Close HDF5 files
        self.train_dataset.close()
        self.val_dataset.close()


def main():
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
        device=args.device,
        output_dir=str(output_dir),
        log_dir=str(log_dir),
    )
    
    trainer.train(num_epochs=args.epochs, val_every_n_steps=args.val_every, early_stopping_patience=args.early_stopping_patience)


if __name__ == "__main__":
    main()