#!/usr/bin/env python3
"""Trainer for distillation using HotpotQA distractor dataset.

This trainer uses both question and context embeddings from HotpotQA:
- Questions are used as queries
- Context documents are used as documents
- The projector learns to map OSCAR embeddings to SFR space
"""

import argparse
import json
import os
import h5py
from pathlib import Path
from typing import Optional, Dict, Any, List

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


class HotpotQADataset(Dataset):
    """Dataset that reads from HDF5 file with HotpotQA teacher embeddings."""
    
    def __init__(self, h5_path: str, split: str = "train", use_contexts: bool = False):
        self.h5_path = h5_path
        self.file = h5py.File(h5_path, 'r')
        self.use_contexts = use_contexts
        
        if split not in self.file:
            raise ValueError(f"Split '{split}' not found. Available: {list(self.file.keys())}")
        
        self.split_grp = self.file[split]
        
        # Load questions
        self.questions = list(self.split_grp['questions'][:])
        self.question_embeddings = self.split_grp['question_embeddings'][:]
        self.questions = [q.decode('utf-8') if isinstance(q, bytes) else q for q in self.questions]
        
        # Load context documents
        self.context_docs = list(self.split_grp['context_docs'][:])
        self.context_embeddings = self.split_grp['context_embeddings'][:]
        self.context_docs = [d.decode('utf-8') if isinstance(d, bytes) else d for d in self.context_docs]
        
        # Combine questions and contexts for training
        if use_contexts:
            self.texts = self.questions + self.context_docs
            self.targets = np.vstack([self.question_embeddings, self.context_embeddings])
        else:
            self.texts = self.questions
            self.targets = self.question_embeddings
        
        print(f"Loaded {split} split: {len(self.texts)} texts (questions: {len(self.questions)}, contexts: {len(self.context_docs)})")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
        }
    
    def get_questions(self) -> List[str]:
        return self.questions
    
    def get_context_docs(self) -> List[str]:
        return self.context_docs
    
    def get_context_embeddings(self) -> torch.Tensor:
        return torch.tensor(self.context_embeddings, dtype=torch.float32)
    
    def close(self):
        self.file.close()


class HotpotQADistillationTrainer:
    """Trainer for HotpotQA distillation."""
    
    def __init__(
        self,
        oscar_model_name: str,
        embeddings_path: str,
        oscar_hidden_dim: int = 3584,
        embed_dim: int = 4096,
        projector_hidden_dim: int = 8192,
        batch_size: int = 128,
        lr: float = 1e-3,
        use_contexts: bool = True,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/hotpot_projector_distill",
        log_dir: str = "./logs/hotpot_projector_distill",
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
        
        # Distillation projector
        self.projector = DistillationProjector(
            oscar_hidden_dim=hidden_size,
            embed_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            use_normalize=True,
        ).to(device=self.device, dtype=torch.float32)
        
        print("DistillationProjector architecture:")
        print(self.projector)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.projector.parameters())
        print(f"Projector parameters: {total_params:,}")
        
        self.batch_size = batch_size
        
        # Load datasets with both questions and contexts
        print(f"Loading embeddings from: {embeddings_path}")
        self.train_dataset = HotpotQADataset(embeddings_path, split="train", use_contexts=use_contexts)
        self.val_dataset = HotpotQADataset(embeddings_path, split="val", use_contexts=use_contexts)
        
        # Also keep separate context data for evaluation
        self.train_context_docs = self.train_dataset.get_context_docs()
        self.train_context_embeds = self.train_dataset.get_context_embeddings().to(self.device)
        
        self.val_context_docs = self.val_dataset.get_context_docs()
        self.val_context_embeds = self.val_dataset.get_context_embeddings().to(self.device)
        
        print(f"Train contexts: {len(self.train_context_docs)}")
        print(f"Val contexts: {len(self.val_context_docs)}")
        
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-5
        )
        
        self.projector_config = {
            "oscar_hidden_dim": hidden_size,
            "embed_dim": embed_dim,
            "projector_hidden_dim": projector_hidden_dim,
            "use_normalize": True,
            "dataset": "hotpotqa",
            "use_contexts": use_contexts,
        }
        
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using OSCAR + projector."""
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
            mem_embeddings = mem_embeddings.float()
        embeddings = self.projector(mem_embeddings)
        return embeddings
    
    def compute_cosine_loss(
        self,
        student_embeds: torch.Tensor,
        teacher_embeds: torch.Tensor,
    ) -> tuple:
        """Compute cosine loss between student and teacher embeddings."""
        student_norm = F.normalize(student_embeds, p=2, dim=-1)
        teacher_norm = F.normalize(teacher_embeds, p=2, dim=-1)
        
        cosine_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
        loss = 1.0 - cosine_sim
        
        return loss, cosine_sim.item()
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        texts = [item["text"] for item in batch]
        targets = torch.stack([item["target"] for item in batch])
        return {"texts": texts, "targets": targets}
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with cosine loss."""
        self.projector.train()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        
        total_loss = 0.0
        total_cosine = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            texts = batch["texts"]
            teacher_embeds = batch["targets"].to(self.device)
            
            # Encode with OSCAR + projector
            student_embeds = self.encode_texts(texts)
            
            # Compute cosine loss
            loss, cosine_sim = self.compute_cosine_loss(student_embeds, teacher_embeds)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.projector.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_cosine += cosine_sim
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cos": f"{cosine_sim:.4f}",
            })
        
        return {
            "loss": total_loss / num_batches,
            "cosine_sim": total_cosine / num_batches,
        }
    
    def validate(self, step: int) -> Dict[str, float]:
        """Validate on validation set."""
        self.projector.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
        )
        
        total_loss = 0.0
        total_cosine = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                texts = batch["texts"]
                teacher_embeds = batch["targets"].to(self.device)
                
                student_embeds = self.encode_texts(texts)
                loss, cosine_sim = self.compute_cosine_loss(student_embeds, teacher_embeds)
                
                total_loss += loss.item()
                total_cosine += cosine_sim
                num_batches += 1
        
        metrics = {
            "cosine_loss": total_loss / num_batches,
            "cosine_sim": total_cosine / num_batches,
        }
        
        print(f"Validation: Cosine Loss={metrics['cosine_loss']:.4f}, Cosine Sim={metrics['cosine_sim']:.4f}")
        
        self.writer.add_scalar("val/cosine_loss", metrics['cosine_loss'], step)
        self.writer.add_scalar("val/cosine_sim", metrics['cosine_sim'], step)
        
        return metrics
    
    def save_checkpoint(self, step: int, metrics: Dict, is_best: bool = False):
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
    
    def train(self, num_epochs: int, val_every_n_steps: int = 500):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"Starting HotpotQA distillation training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        step = 0
        best_cosine_sim = -1.0
        
        # Initial validation
        print("--- Initial Validation ---")
        val_metrics = self.validate(0)
        self.save_checkpoint(0, val_metrics, True)
        best_cosine_sim = val_metrics['cosine_sim']
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} summary:")
            print(f"  Loss: {train_metrics['loss']:.4f}, Cosine: {train_metrics['cosine_sim']:.4f}")
            
            self.writer.add_scalar("train/loss", train_metrics['loss'], step)
            self.writer.add_scalar("train/cosine_sim", train_metrics['cosine_sim'], step)
            
            self.scheduler.step()
            
            # Validate
            step += len(self.train_dataset) // self.batch_size
            val_metrics = self.validate(step)
            
            is_best = val_metrics['cosine_sim'] > best_cosine_sim
            if is_best:
                best_cosine_sim = val_metrics['cosine_sim']
                print(f"  >> New best! Cosine: {best_cosine_sim:.4f}")
            
            self.save_checkpoint(step, {**train_metrics, **val_metrics}, is_best)
            
            self.projector.train()
        
        print(f"\n{'='*60}")
        print(f"Training complete! Best cosine sim: {best_cosine_sim:.4f}")
        print(f"{'='*60}")
        
        self.writer.close()
        
        self.train_dataset.close()
        self.val_dataset.close()


def main():
    parser = argparse.ArgumentParser(description="Train projector via HotpotQA distillation")
    
    parser.add_argument("--oscar-model", type=str,
                        default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--embeddings-path", type=str,
                        default="/data/teacher-embeddings/hotpotqa_teacher_embeddings.h5")
    parser.add_argument("--embed-dim", type=int, default=4096)
    parser.add_argument("--projector-hidden-dim", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use-contexts", action="store_true", default=True,
                        help="Use both questions and context documents for training")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str,
                        default="./checkpoints/hotpot_projector_distill")
    parser.add_argument("--log-dir", type=str,
                        default="./logs/hotpot_projector_distill")
    
    args = parser.parse_args()
    
    trainer = HotpotQADistillationTrainer(
        oscar_model_name=args.oscar_model,
        embeddings_path=args.embeddings_path,
        embed_dim=args.embed_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        use_contexts=args.use_contexts,
        device=args.device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()