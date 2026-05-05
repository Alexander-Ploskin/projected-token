#!/usr/bin/env python3
"""Advanced Trainer with Knowledge Distillation for OSCAR Projector.

This trainer implements the OSCAR-style approach with:
1. Mixed-domain training (MS MARCO + PopQA + Natural Questions)
2. Knowledge Distillation from LLM teacher
3. Query-dependent compression
4. Combined loss: MNR + distillation + rerank

Based on the OSCAR paper: https://arxiv.org/html/2504.07109v1
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Any
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

import sys
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from projected_token.encoders.projector import MEMProjector
from projected_token.training.losses import get_loss_fn


class MixedDomainDataset(Dataset):
    """Mixed-domain dataset for training.
    
    Loads from the generated mixed dataset JSON file.
    """
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        domain_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize dataset.
        
        Args:
            data_path: Path to mixed dataset JSON
            max_samples: Limit number of samples
            domain_weights: Sampling weights for each domain
        """
        print(f"Loading mixed dataset from {data_path}...")
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
        
        self.domain_weights = domain_weights or {
            "msmarco": 0.3,
            "popqa": 0.4,
            "natural_questions": 0.3,
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        negative = item.get("negative")
        if negative is None:
            negatives = item.get("negatives", [])
            if isinstance(negatives, list) and negatives:
                negative = negatives[0]
            else:
                negative = item["positive"]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": negative,
            "domain": item.get("domain", "unknown"),
        }


def mixed_collate_fn(batch):
    """Collate function for mixed domain dataset."""
    return {
        "queries": [item["query"] for item in batch],
        "positives": [item["positive"] for item in batch],
        "negatives": [item["negative"] for item in batch],
        "domains": [item["domain"] for item in batch],
    }


class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss.
    
    Computes KL divergence between teacher and student logits.
    For retrieval, we use contrastive distillation - student should match
    teacher's similarity rankings.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        alpha: float = 0.5,
        distillation_type: str = "kl",
    ):
        """Initialize distillation loss.
        
        Args:
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss
            distillation_type: "kl" (KL divergence) or "contrastive"
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.distillation_type = distillation_type
    
    def forward(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.
        
        Args:
            student_embeddings: [batch, dim] - student (projected) embeddings
            teacher_embeddings: [batch, dim] - teacher (OSCAR mem-tokens) embeddings
            
        Returns:
            Scalar loss
        """
        # Normalize embeddings
        student_emb = F.normalize(student_embeddings, p=2, dim=-1)
        teacher_emb = F.normalize(teacher_embeddings, p=2, dim=-1)
        
        # Compute similarity matrices (use transpose instead of .T for safety)
        student_sim = torch.matmul(student_emb, student_emb.transpose(-2, -1)) / self.temperature
        teacher_sim = torch.matmul(teacher_emb, teacher_emb.transpose(-2, -1)) / self.temperature
        
        # KL divergence between similarity distributions
        student_log_probs = F.log_softmax(student_sim, dim=-1)
        teacher_probs = F.softmax(teacher_sim, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        return kl_loss


class CombinedLoss(nn.Module):
    """Combined loss for training projector.
    
    Combines:
    1. Multiple Negatives Ranking (MNR) loss - contrastive learning
    2. Distillation loss - match teacher rankings
    3. Domain-specific weighting
    """
    
    def __init__(
        self,
        mnr_scale: float = 20.0,
        distillation_weight: float = 0.3,
        temperature: float = 0.1,
        teacher_projector: Optional[nn.Module] = None,
    ):
        """Initialize combined loss.
        
        Args:
            mnr_scale: Scale for MNR loss
            distillation_weight: Weight for distillation component
            temperature: Temperature for distillation
        """
        super().__init__()
        self.mnr_scale = mnr_scale
        self.distillation_weight = distillation_weight
        self.distillation_loss = DistillationLoss(temperature=temperature)
        self.teacher_projector = teacher_projector
        
        # MNR loss from existing implementation
        self.mnr_loss_fn = get_loss_fn("mnr", scale=mnr_scale)
    
    def forward(
        self,
        query_emb: torch.Tensor,
        positive_emb: torch.Tensor,
        negative_emb: torch.Tensor,
        teacher_query_emb: Optional[torch.Tensor] = None,
        teacher_positive_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.
        
        Args:
            query_emb: [batch, dim] - projected query embeddings
            positive_emb: [batch, dim] - projected positive embeddings
            negative_emb: [batch, dim] - projected negative embeddings
            teacher_query_emb: [batch, dim] - OSCAR query embeddings (for distillation)
            teacher_positive_emb: [batch, dim] - OSCAR positive embeddings (for distillation)
            
        Returns:
            Dict with total_loss, mnr_loss, distillation_loss
        """
        # MNR loss (contrastive) - uses in-batch negatives, so only query and positive
        mnr_loss = self.mnr_loss_fn(query_emb, positive_emb)
        
        # Distillation loss - project teacher to student dimension
        distillation_loss = torch.tensor(0.0, device=query_emb.device)
        if teacher_query_emb is not None and teacher_positive_emb is not None:
            # Project teacher embeddings from 3584 to 768 using a simple linear layer
            # First, average the teacher embeddings if they have sequence dimension
            if teacher_query_emb.ndim == 3:
                teacher_query_proj = teacher_query_emb.mean(dim=1)  # [batch, 3584]
            else:
                teacher_query_proj = teacher_query_emb
            
            if teacher_positive_emb.ndim == 3:
                teacher_pos_proj = teacher_positive_emb.mean(dim=1)
            else:
                teacher_pos_proj = teacher_positive_emb
            
            if self.teacher_projector is not None:
                teacher_query_proj = self.teacher_projector(teacher_query_proj)
                teacher_pos_proj = self.teacher_projector(teacher_pos_proj)
            
            # Distill from teacher: student should match teacher's relative rankings
            distillation_loss = self.distillation_loss(
                torch.cat([query_emb, positive_emb], dim=0),
                torch.cat([teacher_query_proj, teacher_pos_proj], dim=0),
            )
        
        # Combined loss
        total_loss = mnr_loss + self.distillation_weight * distillation_loss
        
        return {
            "total_loss": total_loss,
            "mnr_loss": mnr_loss,
            "distillation_loss": distillation_loss,
        }


class AdvancedProjectorTrainer:
    """Advanced trainer with Knowledge Distillation.
    
    This trainer implements the OSCAR-style approach:
    1. Mixed-domain training (MS MARCO + PopQA + NQ)
    2. Knowledge Distillation from OSCAR mem-tokens
    3. Query-dependent compression (concatenate query + document)
    4. Combined loss: MNR + distillation
    """
    
    def __init__(
        self,
        oscar_model_name: str,
        projector_path: Optional[str] = None,
        mixed_dataset_path: str = "data/mixed_train_dataset.json",
        embed_dim: int = 768,
        pooler: str = "mean",
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 64,
        lr: float = 1e-4,
        temperature: float = 0.1,
        distillation_weight: float = 0.3,
        mnr_scale: float = 20.0,
        val_split: float = 0.1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/advanced_projector",
        log_dir: str = "./logs/advanced_projector",
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        use_query_dependent: bool = True,
    ):
        """Initialize advanced trainer.
        
        Args:
            oscar_model_name: Path to OSCAR model
            projector_path: Path to existing projector checkpoint (for fine-tuning)
            mixed_dataset_path: Path to mixed domain dataset
            embed_dim: Output embedding dimension
            pooler: Pooling strategy
            num_layers: Number of MLP layers
            dropout: Dropout rate
            batch_size: Batch size
            lr: Learning rate
            temperature: Temperature for distillation
            distillation_weight: Weight for distillation loss
            mnr_scale: Scale for MNR loss
            val_split: Validation split ratio
            device: Device
            output_dir: Output directory for checkpoints
            log_dir: Log directory
            max_train_samples: Limit training samples
            max_val_samples: Limit validation samples
            use_query_dependent: Use query-dependent compression
        """
        self.device = torch.device(device)
        self.use_query_dependent = use_query_dependent
        
        # Load OSCAR model (frozen)
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        
        # Disable vocab expansion warning
        if hasattr(self.oscar_model, 'compr') and hasattr(self.oscar_model.compr, 'config'):
            self.oscar_model.compr.config.mean_resizing = False
        
        hidden_size = self.oscar_model.compress_documents(['test']).shape[-1]
        print(f"OSCAR hidden size: {hidden_size}")
        
        # Initialize or load projector
        self.projector = MEMProjector(
            hidden_dim=hidden_size,
            embed_dim=embed_dim,
            pooler=pooler,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device=device, dtype=torch.bfloat16)
        
        if projector_path:
            print(f"Loading projector from {projector_path}")
            state_dict = torch.load(projector_path, map_location=device)
            self.projector.load_state_dict(state_dict)
        
        print("Projector architecture:")
        print(self.projector)

        # Teacher projector: projects OSCAR hidden -> student embedding dim for distillation
        self._teacher_projector = nn.Linear(hidden_size, embed_dim).to(device=device, dtype=torch.bfloat16)
        
        # Create data loaders
        train_loader, val_loader = self._create_dataloaders(
            mixed_dataset_path=mixed_dataset_path,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            val_split=val_split,
        )
        
        # Combined loss
        self.loss_fn = CombinedLoss(
            mnr_scale=mnr_scale,
            distillation_weight=distillation_weight,
            temperature=temperature,
            teacher_projector=self._teacher_projector,
        )
        
        # Optimizer - train both projector and teacher_projector
        self.optimizer = torch.optim.AdamW(
            list(self.projector.parameters()) + list(self._teacher_projector.parameters()), 
            lr=lr
        )
        
        # Training state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Save config
        self.config = {
            "oscar_model_name": oscar_model_name,
            "projector_path": projector_path,
            "mixed_dataset_path": mixed_dataset_path,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "temperature": temperature,
            "distillation_weight": distillation_weight,
            "mnr_scale": mnr_scale,
            "use_query_dependent": use_query_dependent,
        }
        
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to {config_path}")
    
    def _create_dataloaders(
        self,
        mixed_dataset_path: str,
        batch_size: int,
        max_train_samples: Optional[int],
        max_val_samples: Optional[int],
        val_split: float,
    ) -> tuple:
        """Create data loaders."""
        full_dataset = MixedDomainDataset(
            data_path=mixed_dataset_path,
            max_samples=max_train_samples,
        )
        
        val_size = max(1, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        
        if max_val_samples:
            val_indices = list(range(min(max_val_samples, len(val_dataset))))
            val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=mixed_collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=mixed_collate_fn,
        )
        
        return train_loader, val_loader
    
    def _encode_with_oscar(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """Encode texts through OSCAR to get teacher embeddings.
        
        Args:
            texts: List of texts
            is_query: If True, encode as query (different method)
            
        Returns:
            [batch, hidden_dim] OSCAR embeddings
        """
        with torch.inference_mode():
            if is_query:
                # For queries, use a simple encoding
                # OSCAR is primarily designed for document compression
                # For queries, we use the model directly
                mem_emb = self.oscar_model.compress_documents(texts)
            else:
                mem_emb = self.oscar_model.compress_documents(texts)
        
        return mem_emb
    
    def _encode_with_projector(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """Encode texts through OSCAR + projector.
        
        Args:
            texts: List of texts
            is_query: If True, use query-dependent encoding
            
        Returns:
            [batch, embed_dim] projected embeddings
        """
        # Get OSCAR embeddings
        oscar_emb = self._encode_with_oscar(texts, is_query=is_query)
        
        # Project
        proj_emb = self.projector(oscar_emb)
        
        return proj_emb
    
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Dict with queries, positives, negatives, domains
            
        Returns:
            Dict with losses
        """
        queries = batch["queries"]
        positives = batch["positives"]
        negatives = batch["negatives"]
        
        # Get teacher embeddings from OSCAR (for distillation)
        with torch.inference_mode():
            teacher_query = self._encode_with_oscar(queries, is_query=True)
            teacher_pos = self._encode_with_oscar(positives, is_query=False)
        
        # Get student embeddings from projector
        query_emb = self._encode_with_projector(queries, is_query=True)
        pos_emb = self._encode_with_projector(positives, is_query=False)
        neg_emb = self._encode_with_projector(negatives, is_query=False)
        
        # Compute loss
        losses = self.loss_fn(
            query_emb=query_emb,
            positive_emb=pos_emb,
            negative_emb=neg_emb,
            teacher_query_emb=teacher_query,
            teacher_positive_emb=teacher_pos,
        )
        
        # Backward
        self.optimizer.zero_grad()
        losses["total_loss"].backward()
        self.optimizer.step()
        
        self.global_step += 1
        
        return {
            "total_loss": losses["total_loss"].item(),
            "mnr_loss": losses["mnr_loss"].item(),
            "distillation_loss": losses["distillation_loss"].item(),
        }
    
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single validation step with similarity logging per domain."""
        queries = batch["queries"]
        positives = batch["positives"]
        negatives = batch["negatives"]
        domains = batch["domains"]
        
        with torch.inference_mode():
            query_emb = self._encode_with_projector(queries, is_query=True)
            pos_emb = self._encode_with_projector(positives, is_query=False)
            neg_emb = self._encode_with_projector(negatives, is_query=False)
            
            losses = self.loss_fn(
                query_emb=query_emb,
                positive_emb=pos_emb,
                negative_emb=neg_emb,
            )
            
            # Compute cosine similarities per domain
            query_emb_np = query_emb.float().cpu().numpy()
            pos_emb_np = pos_emb.float().cpu().numpy()
            neg_emb_np = neg_emb.float().cpu().numpy()
            
            # Normalize for cosine similarity
            query_norm = query_emb_np / np.linalg.norm(query_emb_np, axis=1, keepdims=True)
            pos_norm = pos_emb_np / np.linalg.norm(pos_emb_np, axis=1, keepdims=True)
            neg_norm = neg_emb_np / np.linalg.norm(neg_emb_np, axis=1, keepdims=True)
            
            # Compute similarities
            pos_sim = np.sum(query_norm * pos_norm, axis=1)
            neg_sim = np.sum(query_norm * neg_norm, axis=1)
            
            # Aggregate by domain
            domain_stats = {}
            for i, domain in enumerate(domains):
                if domain not in domain_stats:
                    domain_stats[domain] = {"pos_sim": [], "neg_sim": []}
                domain_stats[domain]["pos_sim"].append(float(pos_sim[i]))
                domain_stats[domain]["neg_sim"].append(float(neg_sim[i]))
            
            # Store for epoch summary
            if not hasattr(self, 'val_domain_stats'):
                self.val_domain_stats = []
            
            self.val_domain_stats.append({
                "domain_stats": domain_stats,
                "total_loss": losses["total_loss"].item(),
            })
        
        return {
            "total_loss": losses["total_loss"].item(),
            "mnr_loss": losses["mnr_loss"].item(),
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.projector.train()
        
        epoch_losses = {"total_loss": [], "mnr_loss": [], "distillation_loss": []}
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
            losses = self.training_step(batch)
            
            for k, v in losses.items():
                epoch_losses[k].append(v)
        
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self) -> Dict[str, float]:
        """Run validation with domain-specific similarity logging."""
        self.projector.eval()
        self.val_domain_stats = []  # Reset for this validation run
        
        val_losses = {"total_loss": [], "mnr_loss": []}
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            losses = self.validation_step(batch)
            
            for k, v in losses.items():
                val_losses[k].append(v)
        
        avg_losses = {k: sum(v) / len(v) for k, v in val_losses.items()}
        
        # Aggregate domain-specific similarities
        print("\n--- Validation Similarities by Domain ---")
        
        all_domains = {}
        for stats in self.val_domain_stats:
            for domain, domain_data in stats["domain_stats"].items():
                if domain not in all_domains:
                    all_domains[domain] = {"pos_sim": [], "neg_sim": []}
                all_domains[domain]["pos_sim"].extend(domain_data["pos_sim"])
                all_domains[domain]["neg_sim"].extend(domain_data["neg_sim"])
        
        for domain, data in all_domains.items():
            pos_mean = np.mean(data["pos_sim"])
            neg_mean = np.mean(data["neg_sim"])
            pos_std = np.std(data["pos_sim"])
            neg_std = np.std(data["neg_sim"])
            
            print(f"{domain:20s}: pos_sim={pos_mean:.4f}±{pos_std:.4f}, neg_sim={neg_mean:.4f}±{neg_std:.4f}, diff={pos_mean-neg_mean:+.4f}")
        
        # Save validation examples
        self._save_validation_examples()
        
        return avg_losses
    
    def _save_validation_examples(self, num_examples: int = 5):
        """Save example validation cases with similarities."""
        examples = []
        
        for stats in self.val_domain_stats[:3]:  # First 3 batches
            for domain, domain_data in stats["domain_stats"].items():
                for i in range(min(num_examples, len(domain_data["pos_sim"]))):
                    examples.append({
                        "domain": domain,
                        "pos_similarity": domain_data["pos_sim"][i],
                        "neg_similarity": domain_data["neg_sim"][i],
                        "diff": domain_data["pos_sim"][i] - domain_data["neg_sim"][i],
                    })
                break  # One batch per domain
        
        if examples:
            examples_path = self.output_dir / "validation_examples.json"
            with open(examples_path, 'w') as f:
                json.dump(examples, f, indent=2)
            print(f"\nValidation examples saved to: {examples_path}")
    
    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
    ):
        """Full training loop.
        
        Args:
            num_epochs: Number of epochs
            save_every: Save checkpoint every N epochs
        """
        print("\n" + "=" * 60)
        print("STARTING ADVANCED TRAINING")
        print("=" * 60)
        print(f"Config: {json.dumps(self.config, indent=2)}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n--- Epoch {epoch}/{num_epochs} ---")
            
            # Train
            train_losses = self.train_epoch(epoch)
            print(f"Train losses: {train_losses}")
            
            # Validate
            val_losses = self.validate()
            print(f"Val losses: {val_losses}")
            
            # Save checkpoint
            if epoch % save_every == 0 or epoch == num_epochs:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save(self.projector.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Save best
                if val_losses["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total_loss"]
                    best_path = self.output_dir / "best_model.pt"
                    torch.save(self.projector.state_dict(), best_path)
                    print(f"New best model saved!")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation loss: {self.best_val_loss}")


def create_advanced_trainer(config: dict) -> AdvancedProjectorTrainer:
    """Create advanced trainer from config."""
    return AdvancedProjectorTrainer(
        oscar_model_name=config["oscar_model_name"],
        projector_path=config.get("projector_path"),
        mixed_dataset_path=config.get("mixed_dataset_path", "data/mixed_train_dataset.json"),
        embed_dim=config.get("embed_dim", 768),
        pooler=config.get("pooler", "mean"),
        num_layers=config.get("num_layers", 2),
        dropout=float(config.get("dropout", 0.1)),
        batch_size=int(config.get("batch_size", 64)),
        lr=float(config.get("lr", 1e-4)),
        temperature=float(config.get("temperature", 0.1)),
        distillation_weight=float(config.get("distillation_weight", 0.3)),
        mnr_scale=float(config.get("mnr_scale", 20.0)),
        val_split=float(config.get("val_split", 0.1)),
        device=config.get("device", "cuda:0"),
        output_dir=config.get("output_dir", "./checkpoints/advanced_projector"),
        log_dir=config.get("log_dir", "./logs/advanced_projector"),
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
        use_query_dependent=config.get("use_query_dependent", True),
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced projector training")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    trainer = create_advanced_trainer(config)
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)