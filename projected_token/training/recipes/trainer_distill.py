#!/usr/bin/env python3
"""Trainer for蒸馏 (Distillation) from SFR-Embedding-Mistral.

This trainer:
1. Loads pre-computed teacher embeddings from HDF5
2. Trains projector to match teacher embeddings using MSE loss
3. Uses OSCAR as student model, SFR-Embedding-Mistral as teacher
"""

import argparse
import json
import os
import h5py
from pathlib import Path
from typing import Optional, Dict, Any

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

from projected_token.encoders.projector import DistillationProjector


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
        teacher_embeddings_path: str,
        oscar_hidden_dim: int = 3584,
        embed_dim: int = 4096,
        projector_hidden_dim: int = 8192,
        batch_size: int = 128,
        lr: float = 1e-3,
        val_split: float = 0.05,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/projector_distill",
        log_dir: str = "./logs/projector_distill",
    ):
        self.device = torch.device(device)
        
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()
        
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False
        
        hidden_size = self.oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")
        
        # Distillation projector: flatten(8 * hidden_dim) -> embed_dim
        self.projector = DistillationProjector(
            oscar_hidden_dim=hidden_size,
            embed_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            use_normalize=True,
        ).to(device=self.device, dtype=torch.bfloat16)
        
        print("DistillationProjector architecture:")
        print(self.projector)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.projector.parameters())
        print(f"Projector parameters: {total_params:,}")
        
        self.batch_size = batch_size
        
        # Load datasets
        print(f"Loading teacher embeddings from: {teacher_embeddings_path}")
        self.train_dataset = H5DistillationDataset(teacher_embeddings_path, split="train", val_split=val_split)
        self.val_dataset = H5DistillationDataset(teacher_embeddings_path, split="val", val_split=val_split)
        
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
        
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine scheduler with warmup
        warmup_steps = len(self.train_loader)  # 1 epoch warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-5
        )
        
        self.projector_config = {
            "oscar_hidden_dim": hidden_size,
            "embed_dim": embed_dim,
            "projector_hidden_dim": projector_hidden_dim,
            "use_normalize": True,
        }
        
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float('inf')
        
        self.load_checkpoint(output_dir)
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load best checkpoint if exists."""
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_path):
            print(f"Loading checkpoint from {best_path}")
            checkpoint = torch.load(best_path, map_location=self.device)
            self.projector.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    
    def encode_documents(self, texts):
        """Encode documents using OSCAR + projector."""
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
        embeddings = self.projector(mem_embeddings)
        return embeddings
    
    def compute_distillation_loss(self, student_embeds, teacher_embeds):
        """Compute loss - directly maximize cosine similarity."""
        student_f = student_embeds.float()
        teacher_f = teacher_embeds.float()
        
        student_norm = student_f.norm(p=2, dim=-1).mean()
        teacher_norm = teacher_f.norm(p=2, dim=-1).mean()
        
        # Student is normalized by projector, normalize teacher
        teacher_normed = F.normalize(teacher_f, p=2, dim=-1)
        
        # Cosine loss = 1 - cos (we want to MAXIMIZE cos)
        # This is equivalent to minimizing negative cosine
        cosine_sim = (student_f * teacher_normed).sum(dim=-1).mean()
        
        loss = 1.0 - cosine_sim
        
        return loss, cosine_sim.item(), student_norm.item(), teacher_norm.item()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.projector.train()
        total_loss = 0.0
        total_cosine = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
        for batch in pbar:
            texts = batch["texts"]
            teacher_embeds = batch["targets"].to(self.device, dtype=torch.bfloat16)
            
            student_embeds = self.encode_documents(texts)
            
            loss, cosine_sim, s_norm, t_norm = self.compute_distillation_loss(student_embeds, teacher_embeds)
            
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
                "loss": f"{loss_val:.6f}",
                "cos": f"{cosine_sim:.4f}",
                "grad": f"{grad_norm:.4f}",
                "s_norm": f"{s_norm:.4f}",
                "t_norm": f"{t_norm:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            global_step = (epoch - 1) * len(self.train_loader) + num_batches
            self.writer.add_scalar("train/loss", loss_val, global_step)
            self.writer.add_scalar("train/cosine_sim", cosine_sim, global_step)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        avg_cosine = total_cosine / num_batches
        return {"mse_loss": avg_loss, "cosine_sim": avg_cosine}
    
    def validate(self, step: int) -> Dict[str, float]:
        """Validate on validation set."""
        self.projector.eval()
        total_loss = 0.0
        total_cosine = 0.0
        total_s_norm = 0.0
        total_t_norm = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", 
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                texts = batch["texts"]
                teacher_embeds = batch["targets"].to(self.device, dtype=torch.bfloat16)
                
                student_embeds = self.encode_documents(texts)
                loss, cosine_sim, s_norm, t_norm = self.compute_distillation_loss(student_embeds, teacher_embeds)
                
                total_loss += loss.item() if hasattr(loss, 'item') else loss
                total_cosine += cosine_sim
                total_s_norm += s_norm
                total_t_norm += t_norm
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_cosine = total_cosine / num_batches
        avg_s_norm = total_s_norm / num_batches
        avg_t_norm = total_t_norm / num_batches
        
        print(f"\nValidation @ step {step}:")
        print(f"  MSE Loss:   {avg_loss:.6f}")
        print(f"  Cosine Sim: {avg_cosine:.6f}")
        print(f"  Student norm: {avg_s_norm:.4f}, Teacher norm: {avg_t_norm:.4f}")
        
        self.writer.add_scalar("val/mse_loss", avg_loss, step)
        self.writer.add_scalar("val/cosine_sim", avg_cosine, step)
        
        return {"mse_loss": avg_loss, "cosine_sim": avg_cosine}
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "model_state_dict": self.projector.state_dict(),
            "config": self.projector_config,
        }
        
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.projector_config, f, indent=2)
        
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            
            best_config_path = os.path.join(self.output_dir, "best_config.json")
            with open(best_config_path, 'w') as f:
                json.dump(self.projector_config, f, indent=2)
            print(f"Saved best model to {best_path}")
    
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
        self.save_checkpoint(0, val_metrics, True)
        best_val_loss = val_metrics['mse_loss']
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} summary:")
            print(f"  Train MSE: {train_metrics['mse_loss']:.6f}, Cosine: {train_metrics['cosine_sim']:.6f}")
            
            self.scheduler.step()
            
            # Validate at end of epoch
            step += len(self.train_loader)
            val_metrics = self.validate(step)
            
            is_best = val_metrics['mse_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['mse_loss']
                epochs_without_improvement = 0
                print(f"  >> New best! Val MSE: {best_val_loss:.6f}, Cosine: {val_metrics['cosine_sim']:.6f}")
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
        self.writer.close()
        
        # Close HDF5 files
        self.train_dataset.close()
        self.val_dataset.close()


def main():
    parser = argparse.ArgumentParser(description="Train projector via distillation from SFR")
    
    parser.add_argument("--oscar-model", type=str,
                        default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--teacher-embeddings", type=str,
                        default="/data/teacher-embeddings/teacher_embeddings_100k.h5")
    parser.add_argument("--embed-dim", type=int, default=4096)
    parser.add_argument("--projector-hidden-dim", type=int, default=8192)
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
    
    args = parser.parse_args()
    
    trainer = DistillationTrainer(
        oscar_model_name=args.oscar_model,
        teacher_embeddings_path=args.teacher_embeddings,
        embed_dim=args.embed_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        device=args.device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    
    trainer.train(num_epochs=args.epochs, val_every_n_steps=args.val_every, early_stopping_patience=args.early_stopping_patience)


if __name__ == "__main__":
    main()