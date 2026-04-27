#!/usr/bin/env python3
"""Trainer for OSCAR Projector with Flatten Pooler and Hard Negatives.

This trainer implements:
1. Flatten pooler (instead of mean pooling)
2. Hard negatives support (explicit negatives in loss)
3. Mixed domain training (MS MARCO + PopQA + NQ)
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


class MixedDomainDataset(Dataset):
    """Mixed-domain dataset with hard negatives.
    
    Format:
        {
            "query": str,
            "positive": str,
            "negatives": [str, str, ...],  # List of negative documents
            "domain": str,  # "msmarco", "popqa", "natural_questions"
        }
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
    ):
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        # Domain statistics
        self.domain_counts = {}
        for item in self.data:
            domain = item.get("domain", "unknown")
            self.domain_counts[domain] = self.domain_counts.get(domain, 0) + 1
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Domain distribution: {self.domain_counts}")
        
        # Check negatives
        avg_negatives = np.mean([len(item.get("negatives", [])) for item in self.data])
        print(f"Average negatives per sample: {avg_negatives:.1f}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        negatives = item.get("negatives", [])
        
        # Ensure we have at least one negative
        if len(negatives) == 0:
            negatives = [item["positive"]]  # Placeholder
        
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negatives": negatives,
            "domain": item.get("domain", "unknown"),
        }


def mixed_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for mixed domain dataset.
    
    Handles variable number of negatives per sample.
    """
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negatives"][0] for item in batch]  # Take first negative
    all_negatives = [item["negatives"] for item in batch]  # All negatives
    domains = [item["domain"] for item in batch]
    
    return {
        "queries": queries,
        "positives": positives,
        "negatives": negatives,
        "all_negatives": all_negatives,
        "domains": domains,
    }


class FlattenProjectorTrainer:
    """Trainer for OSCAR Projector with Flatten pooler and Hard Negatives."""
    
    def __init__(
        self,
        oscar_model_name: str,
        embed_dim: int = 768,
        hidden_dim: int = 2048,
        pooler: str = "flatten",
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 32,
        lr: float = 1e-4,
        temperature: float = 0.1,
        val_split: float = 0.1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/projector_flat",
        log_dir: str = "./logs/projector_flat",
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        use_hard_negatives: bool = True,
        devices: Optional[List[str]] = None,
    ):
        # Support multiple GPUs
        if devices is None:
            self.devices = [device] if not device.startswith("cuda") else [device]
        else:
            self.devices = devices
        
        primary_device = self.devices[0]
        
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(primary_device).eval()
        
        # Disable vocab expansion warning for compressor
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False
        
        # Get hidden size from OSCAR
        hidden_size = self.oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")
        
        self.projector = MEMProjector(
            hidden_dim=hidden_size,
            embed_dim=embed_dim,
            pooler=pooler,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device=primary_device, dtype=torch.bfloat16)
        
        # Use DataParallel if multiple GPUs available (but OSCAR stays on primary)
        if len(self.devices) > 1:
            print(f"Using {len(self.devices)} GPUs: {self.devices}")
            print("Note: OSCAR encoder stays on primary device, only projector is parallelized")
            self.projector = nn.DataParallel(self.projector, device_ids=list(range(len(self.devices))))
        
        print("Projector architecture:")
        print(self.projector)
        
        self.use_hard_negatives = use_hard_negatives
        self.batch_size = batch_size
        self.device = torch.device(primary_device)
        
        # Use InfoNCE with hard negatives
        loss_fn = get_loss_fn("infonce", temperature=temperature)
        optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr)
        
        self.projector_config = {
            "hidden_dim": hidden_size,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_hard_negatives": use_hard_negatives,
        }
        
        self.train_loader = None
        self.val_loader = None
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float('inf')
    
    def load_dataset(self, data_path: str, val_split: float = 0.1):
        """Load and split dataset."""
        full_dataset = MixedDomainDataset(data_path)
        
        val_size = max(1, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=mixed_collate_fn,
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=mixed_collate_fn,
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def encode_documents(self, texts: List[str]) -> torch.Tensor:
        """Encode documents using OSCAR + projector."""
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
            # Clone to detach from inference mode
            mem_embeddings = mem_embeddings.clone()
        
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
            # Include negatives in ranking: positives + negatives
            p_norm = F.normalize(pos_embeds, p=2, dim=-1)
            n_norm = F.normalize(neg_embeds, p=2, dim=-1)
            all_docs_norm = torch.cat([p_norm, n_norm], dim=0)
            similarities = torch.matmul(q_norm, all_docs_norm.T)
            
            # For each query i: positive is at position i
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
            # Only positives
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
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.projector.train()
        total_loss = 0.0
        total_mrr = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            queries = batch["queries"]
            positives = batch["positives"]
            negatives = batch["negatives"]
            
            # Encode
            query_embeds = self.encode_documents(queries)
            pos_embeds = self.encode_documents(positives)
            
            # Hard negatives - keep as [batch, embed_dim]
            neg_embeds = None
            if self.use_hard_negatives:
                neg_embeds = self.encode_documents(negatives)
            
            # Loss
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
            
            # Compute metrics with negatives if available (consistent with validation)
            metrics = self.compute_metrics(query_embeds, pos_embeds, neg_embeds if self.use_hard_negatives else None)
            total_mrr += metrics["mrr"]
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mrr": f"{metrics['mrr']:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        avg_mrr = total_mrr / num_batches
        
        return {"loss": avg_loss, "mrr": avg_mrr}
    
    def compute_similarity_stats(
        self,
        query_embeds: torch.Tensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute similarity statistics for logging."""
        query_embeds = F.normalize(query_embeds, p=2, dim=-1)
        pos_embeds = F.normalize(pos_embeds, p=2, dim=-1)
        neg_embeds = F.normalize(neg_embeds, p=2, dim=-1)
        
        sim_pos = (query_embeds * pos_embeds).sum(dim=-1)
        sim_neg = (query_embeds * neg_embeds).sum(dim=-1)
        
        return {
            "sim_pos_mean": sim_pos.mean().item(),
            "sim_pos_std": sim_pos.std().item(),
            "sim_neg_mean": sim_neg.mean().item(),
            "sim_neg_std": sim_neg.std().item(),
            "sim_diff_mean": (sim_pos - sim_neg).mean().item(),
        }
    
    def validate(self, step: int) -> Dict[str, float]:
        """Validate on validation set."""
        self.projector.eval()
        total_loss = 0.0
        total_mrr = 0.0
        total_recall_at_1 = 0.0
        total_recall_at_5 = 0.0
        total_recall_at_10 = 0.0
        num_batches = 0
        
        # Per-domain metrics
        domain_metrics = {
            "popqa": {"mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "count": 0},
            "natural_questions": {"mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "count": 0},
            "msmarco": {"mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "count": 0},
            "unknown": {"mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@10": 0.0, "count": 0},
        }
        
        all_sim_pos = []
        all_sim_neg = []
        val_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                queries = batch["queries"]
                positives = batch["positives"]
                negatives = batch["negatives"]
                domains = batch.get("domains", ["unknown"] * len(queries))
                
                # Encode
                query_embeds = self.encode_documents(queries)
                pos_embeds = self.encode_documents(positives)
                
                # Hard negatives - keep as [batch, embed_dim]
                if self.use_hard_negatives:
                    neg_embeds = self.encode_documents(negatives)
                
                # Loss
                if self.use_hard_negatives:
                    loss = self.loss_fn(
                        query_embeddings=query_embeds,
                        positive_embeddings=pos_embeds,
                        negative_embeddings=neg_embeds,
                    )
                    
                    sim_stats = self.compute_similarity_stats(query_embeds, pos_embeds, neg_embeds)
                    all_sim_pos.append(sim_stats["sim_pos_mean"])
                    all_sim_neg.append(sim_stats["sim_neg_mean"])
                else:
                    loss = self.loss_fn(
                        query_embeddings=query_embeds,
                        positive_embeddings=pos_embeds,
                    )
                
                total_loss += loss.item()
                
                # Compute metrics with negatives if available (consistent with per-domain)
                metrics = self.compute_metrics(query_embeds, pos_embeds, neg_embeds if self.use_hard_negatives else None)
                total_mrr += metrics["mrr"]
                total_recall_at_1 += metrics["recall@1"]
                total_recall_at_5 += metrics["recall@5"]
                total_recall_at_10 += metrics["recall@10"]
                num_batches += 1
                
                # Per-domain metrics - compute correctly including negatives
                if self.use_hard_negatives:
                    # Build similarity matrix: queries x (positives + negatives)
                    q_norm = F.normalize(query_embeds, p=2, dim=-1)
                    p_norm = F.normalize(pos_embeds, p=2, dim=-1)
                    n_norm = F.normalize(neg_embeds, p=2, dim=-1)
                    
                    # Concatenate positives and negatives
                    all_docs_norm = torch.cat([p_norm, n_norm], dim=0)  # [2*batch, embed]
                    sim_matrix = torch.matmul(q_norm, all_docs_norm.T)  # [batch, 2*batch]
                    
                    # For each query i: positive is at position i, negatives at i+batch
                    for i, domain in enumerate(domains):
                        if i >= len(query_embeds):
                            break
                        
                        domain = domain.lower()
                        if domain not in domain_metrics:
                            domain = "unknown"
                        
                        # Get similarities for query i
                        sims = sim_matrix[i]  # [2*batch]
                        
                        # Positive is at position i, negatives at i + batch_size
                        pos_idx = i
                        neg_indices = [i + len(query_embeds)]
                        
                        # Count how many docs have higher similarity than the positive
                        pos_sim = sims[pos_idx].item()
                        rank = (sims > pos_sim).sum().item() + 1
                        
                        mrr = 1.0 / rank
                        r1 = 1.0 if rank == 1 else 0.0
                        r5 = 1.0 if rank <= 5 else 0.0
                        r10 = 1.0 if rank <= 10 else 0.0
                        
                        domain_metrics[domain]["mrr"] += mrr
                        domain_metrics[domain]["recall@1"] += r1
                        domain_metrics[domain]["recall@5"] += r5
                        domain_metrics[domain]["recall@10"] += r10
                        domain_metrics[domain]["count"] += 1
                else:
                    # Without hard negatives, use the batch positives
                    q_norm = F.normalize(query_embeds, p=2, dim=-1)
                    p_norm = F.normalize(pos_embeds, p=2, dim=-1)
                    sim_matrix = torch.matmul(q_norm, p_norm.T)
                    
                    for i, domain in enumerate(domains):
                        if i >= len(query_embeds):
                            break
                        
                        domain = domain.lower()
                        if domain not in domain_metrics:
                            domain = "unknown"
                        
                        sims = sim_matrix[i]
                        sorted_indices = torch.argsort(sims, descending=True)
                        rank = (sorted_indices == i).nonzero(as_tuple=True)[0][0].item() + 1
                        
                        mrr = 1.0 / rank
                        r1 = 1.0 if rank == 1 else 0.0
                        r5 = 1.0 if rank <= 5 else 0.0
                        r10 = 1.0 if rank <= 10 else 0.0
                        
                        domain_metrics[domain]["mrr"] += mrr
                        domain_metrics[domain]["recall@1"] += r1
                        domain_metrics[domain]["recall@5"] += r5
                        domain_metrics[domain]["recall@10"] += r10
                        domain_metrics[domain]["count"] += 1
                
                # Collect examples from multiple batches for better picture
                if batch_idx < 3 and self.use_hard_negatives:
                    start_idx = batch_idx * 8
                    for i in range(start_idx, min(start_idx + 8, len(queries))):
                        if i >= len(queries):
                            break
                        q_emb = F.normalize(query_embeds[i:i+1], p=2, dim=-1)
                        p_emb = F.normalize(pos_embeds[i:i+1], p=2, dim=-1)
                        n_emb = F.normalize(neg_embeds[i:i+1], p=2, dim=-1)
                        
                        sim_q_p = (q_emb * p_emb).sum().item()
                        sim_q_n = (q_emb * n_emb).sum().item()
                        
                        # Also compute similarity with all positives in batch (like in metrics)
                        all_pos_sims = (q_emb * F.normalize(pos_embeds, p=2, dim=-1)).squeeze(0)
                        pos_rank = (all_pos_sims > sim_q_p).sum().item() + 1
                        pos_rank_percentile = pos_rank / len(all_pos_sims) * 100
                        
                        # Get top 3 most similar other positives
                        top3_other = torch.topk(all_pos_sims, min(3, len(all_pos_sims)))
                        top3_sims = top3_other.values.tolist()
                        
                        val_examples.append({
                            "query": queries[i][:100],
                            "positive": positives[i][:200],
                            "negative": negatives[i][:200],
                            "domain": domains[i] if i < len(domains) else "unknown",
                            "sim_pos": sim_q_p,
                            "sim_neg": sim_q_n,
                            "sim_diff": sim_q_p - sim_q_n,
                            "pos_rank_in_batch": pos_rank,
                            "pos_rank_percentile": pos_rank_percentile,
                            "top3_other_sims": top3_sims,
                        })
        
        avg_loss = total_loss / num_batches
        avg_mrr = total_mrr / num_batches
        avg_recall_at_1 = total_recall_at_1 / num_batches
        avg_recall_at_5 = total_recall_at_5 / num_batches
        avg_recall_at_10 = total_recall_at_10 / num_batches
        
        # Compute per-domain averages
        domain_results = {}
        for domain, metrics in domain_metrics.items():
            if metrics["count"] > 0:
                domain_results[domain] = {
                    "mrr": metrics["mrr"] / metrics["count"],
                    "recall@1": metrics["recall@1"] / metrics["count"],
                    "recall@5": metrics["recall@5"] / metrics["count"],
                    "recall@10": metrics["recall@10"] / metrics["count"],
                    "count": metrics["count"],
                }
                # Log to tensorboard
                self.writer.add_scalar(f"val/{domain}/mrr", domain_results[domain]["mrr"], step)
                self.writer.add_scalar(f"val/{domain}/recall@1", domain_results[domain]["recall@1"], step)
                self.writer.add_scalar(f"val/{domain}/recall@5", domain_results[domain]["recall@5"], step)
                self.writer.add_scalar(f"val/{domain}/recall@10", domain_results[domain]["recall@10"], step)
        
        # Logging
        self.writer.add_scalar("val/loss", avg_loss, step)
        self.writer.add_scalar("val/mrr", avg_mrr, step)
        self.writer.add_scalar("val/recall@1", avg_recall_at_1, step)
        self.writer.add_scalar("val/recall@5", avg_recall_at_5, step)
        self.writer.add_scalar("val/recall@10", avg_recall_at_10, step)
        
        if self.use_hard_negatives and all_sim_pos:
            avg_sim_pos = np.mean(all_sim_pos)
            avg_sim_neg = np.mean(all_sim_neg)
            self.writer.add_scalar("val/sim_pos", avg_sim_pos, step)
            self.writer.add_scalar("val/sim_neg", avg_sim_neg, step)
            self.writer.add_scalar("val/sim_diff", avg_sim_pos - avg_sim_neg, step)
            
            if val_examples:
                import json
                examples_path = os.path.join(self.log_dir, f"val_examples_step_{step}.json")
                with open(examples_path, 'w') as f:
                    json.dump(val_examples, f, indent=2, ensure_ascii=False)
                print(f"Saved validation examples to {examples_path}")
        
        return {
            "loss": avg_loss,
            "mrr": avg_mrr,
            "recall@1": avg_recall_at_1,
            "recall@5": avg_recall_at_5,
            "recall@10": avg_recall_at_10,
            "domain_results": domain_results,
        }
    
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
        
        # Save config.json separately
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
    
    def train(
        self,
        num_epochs: int,
        val_every_n_steps: int = 500,
    ):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Validation every {val_every_n_steps} steps")
        
        step = 0
        best_val_loss = float('inf')
        
        # Initial validation
        print("\n--- Step 0 - Initial Validation ---")
        val_metrics = self.validate(0)
        self._print_validation_metrics(val_metrics)
        self.save_checkpoint(0, {"val": val_metrics}, True)
        best_val_loss = val_metrics["loss"]
        
        for epoch in range(1, num_epochs + 1):
            self.projector.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                queries = batch["queries"]
                positives = batch["positives"]
                negatives = batch["negatives"]
                
                # Encode
                query_embeds = self.encode_documents(queries)
                pos_embeds = self.encode_documents(positives)
                
                # Hard negatives - keep as [batch, embed_dim]
                if self.use_hard_negatives:
                    neg_embeds = self.encode_documents(negatives)
                
                # Loss & backward
                self.optimizer.zero_grad()
                
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
                
                loss.backward()
                self.optimizer.step()
                
                step += 1
                
                # Logging
                self.writer.add_scalar("train/loss", loss.item(), step)
                
                query_norm = query_embeds.norm(dim=-1).mean().item()
                self.writer.add_scalar("debug/query_norm", query_norm, step)
                
                metrics = self.compute_metrics(query_embeds, pos_embeds)
                self.writer.add_scalar("train/mrr", metrics["mrr"], step)
                
                if self.use_hard_negatives:
                    sim_stats = self.compute_similarity_stats(query_embeds, pos_embeds, neg_embeds)
                    self.writer.add_scalar("train/sim_pos", sim_stats["sim_pos_mean"], step)
                    self.writer.add_scalar("train/sim_neg", sim_stats["sim_neg_mean"], step)
                    self.writer.add_scalar("train/sim_diff", sim_stats["sim_diff_mean"], step)
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "step": step
                })
                
                # Validation & checkpoint
                if step % val_every_n_steps == 0:
                    print(f"\n--- Step {step} - Validation ---")
                    val_metrics = self.validate(step)
                    self._print_validation_metrics(val_metrics)
                    
                    is_best = val_metrics["loss"] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics["loss"]
                        print(f"New best model! Val loss: {best_val_loss:.4f}")
                    
                    self.save_checkpoint(step, {"val": val_metrics}, is_best)
                    self.projector.train()
        
        print("\nTraining completed!")
        self.writer.close()
    
    def _print_validation_metrics(self, val_metrics: Dict[str, float]):
        """Print validation metrics in formatted way."""
        print(f"Val - Loss: {val_metrics['loss']:.4f}, MRR: {val_metrics['mrr']:.4f}")
        print(f"Val - R@1: {val_metrics['recall@1']:.4f}, R@5: {val_metrics['recall@5']:.4f}, R@10: {val_metrics['recall@10']:.4f}")
        
        if "domain_results" in val_metrics:
            print("\nPer-domain metrics:")
            for domain, metrics in val_metrics["domain_results"].items():
                if metrics.get("count", 0) > 0:
                    print(f"  {domain}: MRR={metrics['mrr']:.4f}, R@1={metrics['recall@1']:.4f}, R@10={metrics['recall@10']:.4f} (n={metrics['count']})")


def main():
    parser = argparse.ArgumentParser(description="Train OSCAR Projector with Flatten Pooler")
    parser.add_argument("--oscar-model", type=str, default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--pooler", type=str, default="flatten")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated list of GPUs, e.g., 'cuda:0,cuda:1'")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/projector_flat")
    parser.add_argument("--log-dir", type=str, default="./logs/projector_flat")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--no-hard-negatives", action="store_true", help="Disable hard negatives")
    args = parser.parse_args()
    
    # Generate unique ID for this run
    import uuid
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    
    # Backup existing logs and checkpoints if they exist
    import shutil
    
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    
    if output_dir.exists() and any(output_dir.iterdir()):
        backup_output = output_dir.parent / f"{output_dir.name}_backup_{run_id}"
        print(f"Backing up existing checkpoints to: {backup_output}")
        shutil.copytree(output_dir, backup_output, dirs_exist_ok=True)
    
    if log_dir.exists() and any(log_dir.iterdir()):
        backup_log = log_dir.parent / f"{log_dir.name}_backup_{run_id}"
        print(f"Backing up existing logs to: {backup_log}")
        shutil.copytree(log_dir, backup_log, dirs_exist_ok=True)
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy this trainer script to output directory
    trainer_script = Path(__file__).resolve()
    dest_script = output_dir / "trainer_flat.py"
    shutil.copy2(trainer_script, dest_script)
    print(f"Saved trainer script to: {dest_script}")
    
    # Parse multiple devices
    devices = None
    if args.devices:
        devices = [d.strip() for d in args.devices.split(',')]
    
    # Create trainer
    trainer = FlattenProjectorTrainer(
        oscar_model_name=args.oscar_model,
        embed_dim=args.embed_dim,
        pooler=args.pooler,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        val_split=args.val_split,
        device=args.device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        use_hard_negatives=not args.no_hard_negatives,
        devices=devices,
    )
    
    # Load dataset
    trainer.load_dataset(args.dataset, args.val_split)
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        val_every_n_steps=args.val_every,
    )


if __name__ == "__main__":
    main()