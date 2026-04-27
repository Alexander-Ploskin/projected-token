#!/usr/bin/env python3
"""Trainer for OSCAR Projector - train on MS MARCO v2, validate on PopQA + NQ.

This trainer:
1. Trains on MS MARCO v2 dataset
2. Validates on PopQA and Natural Questions separately
3. Uses flatten pooler with hard negatives
"""

import argparse
import json
import os
import random
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

from projected_token.encoders.projector import MEMProjector
from projected_token.training.losses import get_loss_fn


class TrainDataset(Dataset):
    """Training dataset for MS MARCO v2."""
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        print(f"Loading training dataset from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} training samples")
        
        avg_negatives = np.mean([len(item.get("negatives", [])) for item in self.data])
        print(f"Average negatives per sample: {avg_negatives:.1f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        negatives = item.get("negatives", [])
        
        if len(negatives) == 0:
            negatives = [item["positive"]]
        
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negatives": negatives,
            "domain": item.get("domain", "msmarco-v2"),
        }


class ValDataset(Dataset):
    """Validation dataset that holds all queries and documents for ranking projected_token."""
    
    def __init__(self, data_path: str):
        print(f"Loading validation dataset from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} validation samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        negatives = item.get("negatives", [])
        
        if len(negatives) == 0:
            negatives = [item["positive"]]
        
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negatives": negatives,
            "domain": item.get("domain", "unknown"),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for training."""
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negatives"][0] for item in batch]
    all_negatives = [item["negatives"] for item in batch]
    domains = [item["domain"] for item in batch]
    
    return {
        "queries": queries,
        "positives": positives,
        "negatives": negatives,
        "all_negatives": all_negatives,
        "domains": domains,
    }


class MSMarcoV2Trainer:
    """Trainer for MS MARCO v2 with PopQA/NQ validation."""
    
    def __init__(
        self,
        oscar_model_name: str,
        train_dataset: Dataset,
        val_datasets: Dict[str, Dataset],  # name -> dataset
        embed_dim: int = 768,
        hidden_dim: int = 2048,
        pooler: str = "flatten",
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 32,
        lr: float = 1e-4,
        temperature: float = 0.02,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/projector_msmarco_v2",
        log_dir: str = "./logs/projector_msmarco_v2",
        use_hard_negatives: bool = True,
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
        
        self.projector = MEMProjector(
            hidden_dim=hidden_size,
            embed_dim=embed_dim,
            pooler=pooler,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device=self.device, dtype=torch.bfloat16)
        
        print("Projector architecture:")
        print(self.projector)
        
        self.use_hard_negatives = use_hard_negatives
        self.batch_size = batch_size
        
        self.loss_fn = get_loss_fn("infonce", temperature=temperature)
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
        
        self.val_loaders = {}
        for name, dataset in val_datasets.items():
            self.val_loaders[name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
            )
        
        self.projector_config = {
            "hidden_dim": hidden_size,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_hard_negatives": use_hard_negatives,
        }
        
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def encode_documents(self, texts: List[str]) -> torch.Tensor:
        """Encode documents using OSCAR + projector."""
        mem_embeddings = self.oscar_model.compress_documents(texts)
        embeddings = self.projector(mem_embeddings)
        return embeddings
    
    def compute_metrics(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute retrieval metrics."""
        q_norm = F.normalize(query_embeds, p=2, dim=-1)
        
        if neg_embeds is not None:
            p_norm = F.normalize(pos_embeds, p=2, dim=-1)
            n_norm = F.normalize(neg_embeds, p=2, dim=-1)
            all_docs_norm = torch.cat([p_norm, n_norm], dim=0)
            similarities = torch.matmul(q_norm, all_docs_norm.T)
            
            mrr = 0.0
            recall_at_1 = 0.0
            recall_at_5 = 0.0
            recall_at_10 = 0.0
            batch_size = query_embeds.size(0)
            
            for i in range(batch_size):
                pos_sim = similarities[i, i].item()
                rank = (similarities[i] > pos_sim).sum().item() + 1
                mrr += 1.0 / rank
                if rank == 1:
                    recall_at_1 += 1.0
                if rank <= 5:
                    recall_at_5 += 1.0
                if rank <= 10:
                    recall_at_10 += 1.0
        else:
            p_norm = F.normalize(pos_embeds, p=2, dim=-1)
            similarities = torch.matmul(q_norm, p_norm.T)
            ranks = torch.argsort(torch.argsort(similarities, dim=1, descending=True), dim=1)
            
            mrr = 0.0
            recall_at_1 = 0.0
            recall_at_5 = 0.0
            recall_at_10 = 0.0
            batch_size = query_embeds.size(0)
            
            for i in range(batch_size):
                rank = ranks[i, i].item() + 1
                mrr += 1.0 / rank
                if rank == 1:
                    recall_at_1 += 1.0
                if rank <= 5:
                    recall_at_5 += 1.0
                if rank <= 10:
                    recall_at_10 += 1.0
        
        mrr /= batch_size
        recall_at_1 /= batch_size
        recall_at_5 /= batch_size
        recall_at_10 /= batch_size
        
        return {
            "mrr": mrr,
            "recall@1": recall_at_1,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
        }
    
    def validate(self, step: int) -> Dict[str, Dict[str, float]]:
        """Validate on all validation datasets."""
        self.projector.eval()
        
        all_results = {}
        
        for val_name, val_loader in self.val_loaders.items():
            print(f"\n--- Validation on {val_name} ---")
            
            total_loss = 0.0
            total_mrr = 0.0
            total_recall_at_1 = 0.0
            total_recall_at_5 = 0.0
            total_recall_at_10 = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Val {val_name}")):
                    queries = batch["queries"]
                    positives = batch["positives"]
                    negatives = batch["negatives"]
                    
                    query_embeds = self.encode_documents(queries)
                    pos_embeds = self.encode_documents(positives)
                    
                    if self.use_hard_negatives:
                        neg_embeds = self.encode_documents(negatives)
                    
                    if self.use_hard_negatives:
                        loss = self.loss_fn(
                            query_embeddings=query_embeds,
                            positive_embeddings=pos_embeds,
                            negative_embeddings=neg_embeds,
                        )
                    else:
                        loss = self.loss_fn(
                            query_embeddings=query_embeds,
                            positive_embeddings=pos_embeds,
                        )
                    
                    total_loss += loss.item()
                    
                    metrics = self.compute_metrics(
                        query_embeds, pos_embeds,
                        neg_embeds if self.use_hard_negatives else None
                    )
                    total_mrr += metrics["mrr"]
                    total_recall_at_1 += metrics["recall@1"]
                    total_recall_at_5 += metrics["recall@5"]
                    total_recall_at_10 += metrics["recall@10"]
                    num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_mrr = total_mrr / num_batches
            avg_recall_at_1 = total_recall_at_1 / num_batches
            avg_recall_at_5 = total_recall_at_5 / num_batches
            avg_recall_at_10 = total_recall_at_10 / num_batches
            
            results = {
                "loss": avg_loss,
                "mrr": avg_mrr,
                "recall@1": avg_recall_at_1,
                "recall@5": avg_recall_at_5,
                "recall@10": avg_recall_at_10,
            }
            
            all_results[val_name] = results
            
            print(f"Val {val_name} - Loss: {avg_loss:.4f}, MRR: {avg_mrr:.4f}")
            print(f"Val {val_name} - R@1: {avg_recall_at_1:.4f}, R@5: {avg_recall_at_5:.4f}, R@10: {avg_recall_at_10:.4f}")
            
            self.writer.add_scalar(f"val_{val_name}/loss", avg_loss, step)
            self.writer.add_scalar(f"val_{val_name}/mrr", avg_mrr, step)
            self.writer.add_scalar(f"val_{val_name}/recall@1", avg_recall_at_1, step)
            self.writer.add_scalar(f"val_{val_name}/recall@5", avg_recall_at_5, step)
            self.writer.add_scalar(f"val_{val_name}/recall@10", avg_recall_at_10, step)
        
        return all_results
    
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
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Validation every {val_every_n_steps} steps")
        
        step = 0
        best_val_loss = float('inf')
        
        print("\n--- Step 0 - Initial Validation ---")
        val_metrics = self.validate(0)
        avg_mrr = sum(v["mrr"] for v in val_metrics.values()) / len(val_metrics)
        self.save_checkpoint(0, {"val": val_metrics, "avg_mrr": avg_mrr}, True)
        best_val_loss = sum(v["loss"] for v in val_metrics.values()) / len(val_metrics)
        
        for epoch in range(1, num_epochs + 1):
            self.projector.train()
            total_loss = 0.0
            total_mrr = 0.0
            num_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                queries = batch["queries"]
                positives = batch["positives"]
                
                query_embeds = self.encode_documents(queries)
                pos_embeds = self.encode_documents(positives)
                
                neg_embeds = None
                if self.use_hard_negatives:
                    neg_embeds = self.encode_documents(batch["negatives"])
                
                if self.use_hard_negatives:
                    loss = self.loss_fn(
                        query_embeddings=query_embeds,
                        positive_embeddings=pos_embeds,
                        negative_embeddings=neg_embeds,
                    )
                else:
                    loss = self.loss_fn(
                        query_embeddings=query_embeds,
                        positive_embeddings=pos_embeds,
                    )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                metrics = self.compute_metrics(
                    query_embeds, pos_embeds,
                    neg_embeds if self.use_hard_negatives else None
                )
                total_mrr += metrics["mrr"]
                num_batches += 1
                step += 1
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mrr": f"{metrics['mrr']:.4f}"
                })
                
                if step % val_every_n_steps == 0:
                    print(f"\n--- Step {step} - Validation ---")
                    val_metrics = self.validate(step)
                    
                    avg_loss = sum(v["loss"] for v in val_metrics.values()) / len(val_metrics)
                    avg_mrr = sum(v["mrr"] for v in val_metrics.values()) / len(val_metrics)
                    
                    is_best = avg_loss < best_val_loss
                    if is_best:
                        best_val_loss = avg_loss
                    
                    self.save_checkpoint(step, {"val": val_metrics, "avg_mrr": avg_mrr}, is_best)
                    
                    if is_best:
                        print(f"New best model! Val loss: {avg_loss:.4f}")
                    
                    self.projector.train()
            
            epoch_loss = total_loss / num_batches
            epoch_mrr = total_mrr / num_batches
            print(f"\nEpoch {epoch}: Loss: {epoch_loss:.4f}, MRR: {epoch_mrr:.4f}")
            
            self.writer.add_scalar("train/loss", epoch_loss, epoch)
            self.writer.add_scalar("train/mrr", epoch_mrr, epoch)
        
        print("\nTraining complete!")
        print(f"Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train on MS MARCO v2, validate on PopQA/NQ")
    
    parser.add_argument("--train-dataset", type=str, required=True,
                        help="Path to training dataset JSON")
    parser.add_argument("--popqa-dataset", type=str, required=True,
                        help="Path to PopQA validation dataset JSON")
    parser.add_argument("--nq-dataset", type=str, required=True,
                        help="Path to Natural Questions validation dataset JSON")
    parser.add_argument("--oscar-model", type=str,
                        default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--pooler", type=str, default="flatten")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str,
                        default="./checkpoints/projector_msmarco_v2")
    parser.add_argument("--log-dir", type=str,
                        default="./logs/projector_msmarco_v2")
    
    args = parser.parse_args()
    
    train_dataset = TrainDataset(args.train_dataset)
    
    popqa_dataset = ValDataset(args.popqa_dataset)
    nq_dataset = ValDataset(args.nq_dataset)
    
    val_datasets = {
        "popqa": popqa_dataset,
        "natural_questions": nq_dataset,
    }
    
    trainer = MSMarcoV2Trainer(
        oscar_model_name=args.oscar_model,
        train_dataset=train_dataset,
        val_datasets=val_datasets,
        pooler=args.pooler,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        device=args.device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    
    trainer.train(num_epochs=args.epochs, val_every_n_steps=args.val_every)


if __name__ == "__main__":
    main()