import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class InfoNCELoss(nn.Module):
    """InfoNCE loss для contrastive learning.

    L = -log(exp(sim(q, d+) / tau) / (exp(sim(q, d+) / tau) + sum(exp(sim(q, d-)) / tau)))
    """

    def __init__(self, temperature: float = 0.02):
        """Инициализация.

        Args:
            temperature: температура для softmax (стандарт 0.02-0.05)
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Вычислить InfoNCE loss.

        Args:
            query_embeddings: [batch, embed_dim]
            positive_embeddings: [batch, embed_dim]
            negative_embeddings: [batch, embed_dim] (different negatives per query)
                                     or None для in-batch

        Returns:
            scalar loss
        """
        # Debug: print shapes
        if query_embeddings.dim() != 2 or positive_embeddings.dim() != 2:
            raise ValueError(
                f"Expected 2D tensors, got query: {query_embeddings.shape}, positive: {positive_embeddings.shape}"
            )
        
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)

        sim_pos = (query_embeddings * positive_embeddings).sum(dim=-1) / self.temperature

        if negative_embeddings is not None:
            # Debug: check negative_embeddings shape
            if negative_embeddings.dim() == 3:
                # [batch, num_negatives, embed_dim] - take first negative
                negative_embeddings = negative_embeddings[:, 0, :]
            elif negative_embeddings.dim() != 2:
                raise ValueError(
                    f"Expected 2D or 3D negative_embeddings, got: {negative_embeddings.shape}"
                )
            
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
            
            # negatives: [batch, embed_dim] - each query has its own negative
            # Compute similarity for each query to its corresponding negative only
            # sim_neg[i] = similarity(query[i], negative[i])
            sim_neg = (query_embeddings * negative_embeddings).sum(dim=-1, keepdim=True) / self.temperature
            
            logits = torch.cat([sim_pos.unsqueeze(-1), sim_neg], dim=-1)
            labels = torch.zeros(query_embeddings.size(0), dtype=torch.long, device=query_embeddings.device)
        else:
            # Standard in-batch InfoNCE over query-document similarities.
            logits = torch.matmul(
                query_embeddings,
                positive_embeddings.mT
            ) / self.temperature
            labels = torch.arange(query_embeddings.size(0), device=query_embeddings.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class MultipleNegativesRankingLoss(nn.Module):
    """Multiple Negatives Ranking Loss - упрощенная версия InfoNCE для in-batch negatives.

    Используется в sentence-transformers.
    """

    def __init__(self, scale: float = 20.0):
        """Инициализация.

        Args:
            scale: масштаб (эквивалент 1/temperature, стандарт 20 = 1/0.05)
        """
        super().__init__()
        self.scale = scale

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Вычислить MNR loss.

        Args:
            query_embeddings: [batch, embed_dim]
            positive_embeddings: [batch, embed_dim]

        Returns:
            scalar loss
        """
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)

        scores = torch.matmul(query_embeddings, positive_embeddings.T) * self.scale
        labels = torch.arange(query_embeddings.size(0), device=query_embeddings.device)

        loss = F.cross_entropy(scores, labels)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss с margin для contrastive learning."""

    def __init__(self, margin: float = 0.5):
        """Инициализация.

        Args:
            margin: margin для triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Вычислить triplet loss.

        Args:
            query_embeddings: [batch, embed_dim]
            positive_embeddings: [batch, embed_dim]
            negative_embeddings: [batch, embed_dim]

        Returns:
            scalar loss
        """
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)

        pos_sim = F.cosine_similarity(query_embeddings, positive_embeddings)
        neg_sim = F.cosine_similarity(query_embeddings, negative_embeddings)

        losses = F.relu(self.margin - pos_sim + neg_sim)
        return losses.mean()


def get_loss_fn(loss_name: str = "mnr", **kwargs) -> nn.Module:
    """Получить функцию потери по имени.

    Args:
        loss_name: 'infonce', 'mnr', или 'triplet'
        **kwargs: аргументы для loss

    Returns:
        loss функция
    """
    if loss_name == "infonce":
        return InfoNCELoss(temperature=kwargs.get("temperature", 0.02))
    elif loss_name == "mnr":
        return MultipleNegativesRankingLoss(scale=kwargs.get("scale", 20.0))
    elif loss_name == "triplet":
        return TripletLoss(margin=kwargs.get("margin", 0.5))
    else:
        raise ValueError(f"Unknown loss: {loss_name}")