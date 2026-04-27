Прочитай статью OSCAR -- [https://arxiv.org/abs/2504.07109](https://arxiv.org/abs/2504.07109). Это context compression метод. Компрессор выдает для документов эмбеддинги 8 me-токенов. Я хочу использовать эти эмбеддинги для ретрива, но из коробки они не работают. Поэтому я обучил проектор, из эмбеддингов 8 токенов в пространство эмбеддингов для ретрива. Обучал на [https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-v3). И теперь я имею хорошие метрики поиска на ms-marco, а еще на sentence-transformers/natural-questions. Однако, когда я попробвал провалидировать на PopQA (query -- вопрос (question), а в качестве контекста статья из s_wiki_url, то получил ужасные результаты, при этом Salesforce/SFR-Embedding-Mistral на этой задаче работает.

Смогу ли я как-то обучить такой проектор, чтобы на PopQA тоже работал? Если да, предложи step by step инструкцию.

Для справки, код, который я использовал для обучения:
```
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoModel, AutoTokenizer


from projected_token.encoders.projector import MEMProjector
from projected_token.training.trainer import BaseTrainer
from projected_token.training.dataset import create_dataloaders
from projected_token.training.losses import get_loss_fn



class MLPTrainer(BaseTrainer):
    """Trainer для варианта A: MLP проектор.


    OSCAR заморожен, только проектор обучается.
    """


    def __init__(
        self,
        oscar_model_name: str,
        embed_dim: int = 768,
        hidden_dim: int = 2048,
        pooler: str = "mean",
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_size: int = 64,
        lr: float = 1e-4,
        temperature: float = 0.02,
        val_split: float = 0.1,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints/mlp",
        log_dir: str = "./logs/mlp",
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
    ):
        """Инициализация.


        Args:
            oscar_model_name: путь к OSCAR модели
            embed_dim: размерность выходных эмбеддингов
            hidden_dim: скрытая размерность OSCAR
            pooler: стратегия pooling
            num_layers: количество слоев в MLP
            dropout: dropout
            batch_size: размер батча
            lr: learning rate
            temperature: температура для loss
            device: устройство
            output_dir: директория для чекпоинтов
            log_dir: директория для логов
            max_train_samples: лимит train samples
            max_val_samples: лимит val samples
        """
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device).eval()
        
        # Disable vocab expansion warning for compressor
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
        ).to(device=device, dtype=torch.bfloat16)


        print("Projector architecture:")
        print(self.projector)


        train_loader, val_loader = create_dataloaders(
            oscar_model=self.oscar_model,
            batch_size=batch_size,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            val_split=val_split,
            device=device,
        )


        loss_fn = get_loss_fn("mnr", scale=1.0 / temperature)
        optimizer = torch.optim.AdamW(self.projector.parameters(), lr=lr)


        self.projector_config = {
            "hidden_dim": hidden_size,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "num_layers": num_layers,
            "dropout": dropout,
        }


        super().__init__(
            model=self.projector,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            output_dir=output_dir,
            log_dir=log_dir,
        )


        self.model_config = self.projector_config


    def encode_documents(self, texts: list[str]) -> torch.Tensor:
        """Закодировать тексты через OSCAR + проектор (query-independent).


        Args:
            texts: список текстов (документы или запросы)


        Returns:
            [batch, embed_dim] эмбеддинги
        """
        with torch.inference_mode():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)


        embeddings = self.projector(mem_embeddings)
        return embeddings



def create_mlp_trainer(config: dict) -> MLPTrainer:
    """Создать MLP trainer из конфига.


    Args:
        config: словарь с конфигом


    Returns:
        MLPTrainer instance
    """
    return MLPTrainer(
        oscar_model_name=config["oscar_model_name"],
        embed_dim=config.get("embed_dim", 768),
        hidden_dim=config.get("hidden_dim", 2048),
        pooler=config.get("pooler", "mean"),
        num_layers=config.get("num_layers", 2),
        dropout=float(config.get("dropout", 0.1)),
        batch_size=int(config.get("batch_size", 64)),
        lr=float(config.get("lr", 1e-4)),
        temperature=float(config.get("temperature", 0.02)),
        val_split=float(config.get("val_split", 0.1)),
        device=config.get("device", "cuda:0"),
        output_dir=config.get("output_dir", "./checkpoints/mlp"),
        log_dir=config.get("log_dir", "./logs/mlp"),
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
    )

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
from tqdm import tqdm
import os
import json
from datetime import datetime


from projected_token.training.losses import get_loss_fn, MultipleNegativesRankingLoss



class BaseTrainer:
    """Базовый класс trainer для обучения проектора."""


    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda:0",
        output_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        """Инициализация.


        Args:
            model: модель для обучения
            train_loader: train DataLoader
            val_loader: validation DataLoader
            loss_fn: функция потерь
            optimizer: оптимизатор
            device: устройство
            output_dir: директория для сохранения чекпоинтов
            log_dir: директория для логов
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.log_dir = log_dir


        self.model.to(self.device)


        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)


        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.model_config = {}


    def compute_metrics(self, query_embeds: torch.Tensor, doc_embeds: torch.Tensor) -> Dict[str, float]:
        """Вычислить метрики качества.


        Args:
            query_embeds: [batch, embed_dim]
            doc_embeds: [batch, embed_dim]


        Returns:
            словарь с метриками
        """
        query_embeds = torch.nn.functional.normalize(query_embeds, p=2, dim=-1)
        doc_embeds = torch.nn.functional.normalize(doc_embeds, p=2, dim=-1)


        similarities = torch.matmul(query_embeds, doc_embeds.T)
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
        """Обучить одну эпоху.


        Args:
            epoch: номер эпохи


        Returns:
            словарь с метриками
        """
        self.model.train()
        total_loss = 0.0
        total_mrr = 0.0


        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            queries = batch["queries"]
            positives = batch["positives"]
            negatives = batch["negatives"]


            # Encode queries and docs using the same projector
            # (treat queries as short documents)
            query_embeds = self.encode_documents(queries)
            doc_embeds = self.encode_documents(positives)


            self.optimizer.zero_grad()
            loss = self.loss_fn(query_embeds, doc_embeds)
            loss.backward()
            self.optimizer.step()


            total_loss += loss.item()
            metrics = self.compute_metrics(query_embeds, doc_embeds)
            total_mrr += metrics["mrr"]


            self.global_step += 1


            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mrr": f"{metrics['mrr']:.4f}"})


            self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            self.writer.add_scalar("train/mrr", metrics["mrr"], self.global_step)


        avg_loss = total_loss / len(self.train_loader)
        avg_mrr = total_mrr / len(self.train_loader)


        return {"loss": avg_loss, "mrr": avg_mrr}


    def validate(self, step: int):
        """Валидация на validation set.


        Args:
            step: номер шага для логирования


        Returns:
            словарь с метриками
        """
        self.model.eval()
        total_loss = 0.0
        total_mrr = 0.0
        total_recall_at_1 = 0.0
        total_recall_at_5 = 0.0
        total_recall_at_10 = 0.0
        num_batches = 0


        with torch.inference_mode():
            for batch in tqdm(self.val_loader, desc="Validation"):
                queries = batch["queries"]
                positives = batch["positives"]


                query_embeds = self.encode_documents(queries)
                doc_embeds = self.encode_documents(positives)


                loss = self.loss_fn(query_embeds, doc_embeds)
                total_loss += loss.item()


                metrics = self.compute_metrics(query_embeds, doc_embeds)
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


        self.writer.add_scalar("val/loss", avg_loss, step)
        self.writer.add_scalar("val/mrr", avg_mrr, step)
        self.writer.add_scalar("val/recall@1", avg_recall_at_1, step)
        self.writer.add_scalar("val/recall@5", avg_recall_at_5, step)
        self.writer.add_scalar("val/recall@10", avg_recall_at_10, step)


        return {
            "loss": avg_loss,
            "mrr": avg_mrr,
            "recall@1": avg_recall_at_1,
            "recall@5": avg_recall_at_5,
            "recall@10": avg_recall_at_10,
        }


    def encode_documents(self, texts: list[str]) -> torch.Tensor:
        """Закодировать тексты через проектор (query-independent).


        Args:
            texts: список текстов (документы или запросы)


        Returns:
            [batch, embed_dim] эмбеддинги
        """
        raise NotImplementedError


    def save_checkpoint(self, step: int, metrics: Dict[str, float], is_best: bool = False, config: Dict = None):
        """Сохранить чекпоинт.


        Args:
            step: номер шага
            metrics: метрики
            is_best: лучший ли это чекпоинт
            config: словарь с конфигурацией модели
        """
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "model_state_dict": self.model.state_dict(),
        }
        
        if config is not None:
            checkpoint["config"] = config


        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step_{step}.pt")
        torch.save(checkpoint, checkpoint_path)


        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)


        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)


    def train(self, num_epochs: int, val_every_n_steps: int = 100):
        """Основной цикл обучения.


        Args:
            num_epochs: количество эпох
            val_every_n_steps: частота валидации в шагах
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"Log directory: {self.log_dir}")
        print(f"Validation every {val_every_n_steps} steps")


        step = 0
        best_val_loss = float('inf')


        print("\n--- Step 0 - Initial Validation ---")
        val_metrics = self.validate(0)
        print(f"Val - Loss: {val_metrics['loss']:.4f}, MRR: {val_metrics['mrr']:.4f}")
        print(f"Val - R@1: {val_metrics['recall@1']:.4f}, R@5: {val_metrics['recall@5']:.4f}, R@10: {val_metrics['recall@10']:.4f}")
        self.save_checkpoint(0, {"val": val_metrics}, True, self.model_config)
        best_val_loss = val_metrics["loss"]
        self.model.train()


        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")


            self.model.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            
            for batch in pbar:
                queries = batch["queries"]
                positives = batch["positives"]


                # Forward - queries as questions, positives as documents
                query_embeds = self.encode_documents(queries)
                doc_embeds = self.encode_documents(positives)


                # Loss & backward
                self.optimizer.zero_grad()
                loss = self.loss_fn(query_embeds, doc_embeds)
                loss.backward()
                self.optimizer.step()


                step += 1
                self.global_step = step


                # Debug: check for embedding collapse
                query_norm = query_embeds.norm(dim=-1).mean().item()
                doc_norm = doc_embeds.norm(dim=-1).mean().item()


                # Logging
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("debug/query_norm", query_norm, step)
                self.writer.add_scalar("debug/doc_norm", doc_norm, step)
                
                metrics = self.compute_metrics(query_embeds, doc_embeds)
                self.writer.add_scalar("train/mrr", metrics["mrr"], step)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": step})


                # Validation & checkpoint every N steps
                if step % val_every_n_steps == 0:
                    print(f"\n--- Step {step} - Validation ---")
                    val_metrics = self.validate(step)
                    print(f"Val - Loss: {val_metrics['loss']:.4f}, MRR: {val_metrics['mrr']:.4f}")
                    print(f"Val - R@1: {val_metrics['recall@1']:.4f}, R@5: {val_metrics['recall@5']:.4f}, R@10: {val_metrics['recall@10']:.4f}")


                    is_best = val_metrics["loss"] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics["loss"]
                        print(f"New best model! Val loss: {best_val_loss:.4f}")


                    self.save_checkpoint(step, {"val": val_metrics}, is_best, self.model_config)
                    self.model.train()  # Back to train mode


        print("\nTraining completed!")
        self.writer.close()

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer



class MSMarcoDataset(Dataset):
    """Датасет MS MARCO для обучения проектора.


    Загружает данные из локальной директории.
    """


    def __init__(
        self,
        data_path: str = "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
        config: str = "triplet",
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """Инициализация датасета.


        Args:
            data_path: путь к локальным данным
            config: имя конфига (triplet, triplet-hard, etc.)
            split: 'train' или 'validation'
            max_samples: ограничение количества samples (для отладки)
        """
        self.split = split


        print(f"Loading MS MARCO dataset: {data_path}, config={config}, split={split}...")
        self.dataset = load_dataset(
            data_path,
            name=config,
            split=split,
        )


        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))


        print(f"Loaded {len(self.dataset)} samples")


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "query": item["query"],
            "positive": item["positive"],
            "negative": item["negative"],
        }



def collate_fn(batch):
    """Collate функция для DataLoader - просто пакует данные без encoding.


    Returns:
        словарь с query, positive, negative текстами
    """
    queries = [item["query"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]


    return {
        "queries": queries,
        "positives": positives,
        "negatives": negatives,
    }



def create_dataloaders(
    oscar_model,
    batch_size: int = 64,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    num_workers: int = 0,
    device: str = "cuda:0",
    val_split: float = 0.1,
):
    """Создать train и validation dataloaders.


    Args:
        oscar_model: OSCAR модель
        batch_size: размер батча
        max_train_samples: лимит train samples
        max_val_samples: лимит val samples
        num_workers: количество workers
        device: устройство
        val_split: доля данных для валидации


    Returns:
        (train_loader, val_loader)
    """
    full_dataset = MSMarcoDataset(split="train", max_samples=max_train_samples)
    
    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size
    
    if train_size < 1:
        raise ValueError(f"Not enough samples for train split: {len(full_dataset)} samples, val_split={val_split}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    if max_val_samples is not None:
        val_indices = list(range(min(max_val_samples, len(val_dataset))))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)


    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


    return train_loader, val_loader

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
            negative_embeddings: [batch, num_negatives, embed_dim] или None для in-batch


        Returns:
            scalar loss
        """
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)


        sim_pos = (query_embeddings * positive_embeddings).sum(dim=-1) / self.temperature


        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
            sim_neg = torch.matmul(
                query_embeddings,
                negative_embeddings.transpose(-2, -1)
            ) / self.temperature
            logits = torch.cat([sim_pos.unsqueeze(-1), sim_neg], dim=-1)
        else:
            sim_neg = torch.matmul(
                query_embeddings,
                query_embeddings.T
            ) / self.temperature
            mask = torch.eye(query_embeddings.size(0), device=query_embeddings.device)
            sim_neg = sim_neg.masked_fill(mask.bool(), float('-inf'))
            logits = torch.cat([sim_pos.unsqueeze(-1), sim_neg], dim=-1)


        labels = torch.zeros(query_embeddings.size(0), dtype=torch.long, device=query_embeddings.device)
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
```

Да, вы сможете обучить проектор для успешной работы на PopQA, однако текущий подход требует нескольких концептуальных изменений в архитектуре и исправления логической ошибки в коде. 

Ваша модель показывает хорошие результаты на MS MARCO, поскольку этот датасет состоит из пользовательских запросов и коротких текстовых пассажей. Однако PopQA — это датасет для ответов на вопросы о редких (long-tail) сущностях, где в качестве контекста выступают полнотекстовые статьи из Википедии. Salesforce/SFR-Embedding-Mistral справляется с этой задачей, так как имеет большое контекстное окно и обучался на длинных документах. [huggingface](https://huggingface.co/datasets/akariasai/PopQA)

### Почему текущий код дает плохие метрики на PopQA

1. **Игнорирование Hard Negatives (критический баг):** В методе `train_epoch` вы успешно извлекаете `negatives = batch["negatives"]`, но передаете в `self.loss_fn` только позитивные и квери-эмбеддинги. В результате `MultipleNegativesRankingLoss` использует только in-batch negatives (учится отличать разные темы). В PopQA крайне важно отличать похожие друг на друга сущности, для чего модель обязана видеть жесткие негативные примеры (hard negatives). [arxiv](https://arxiv.org/html/2502.10996v3)
2. **Проблема длины документов:** Вы обучаете проектор на коротких пассажах MS MARCO (~60 слов). OSCAR по-разному компрессирует короткий текст и длинную статью Википедии. На инференсе PopQA проектор получает 8 me-токенов длинного документа, которые являются для него out-of-distribution данными (он их никогда не видел). [emergentmind](https://www.emergentmind.com/papers/1611.09268)
3. **Асимметрия запросов:** Вы сжимаете короткий вопрос (5–10 слов) через тот же механизм `compress_documents`, что и длинный контекст, что избыточно и может приводить к потере интента самого вопроса.

***

### Step-by-Step инструкция по адаптации

#### Шаг 1: Исправьте вычисление Loss-функции
PopQA строится вокруг редких сущностей, поэтому модель должна учиться на hard negatives. Переключите ваш лосс на `InfoNCELoss` из вашего файла и явно закодируйте негативные документы.

Измените метод `train_epoch`:
```python
queries = batch["queries"]
positives = batch["positives"]
negatives = batch["negatives"]

query_embeds = self.encode_documents(queries)
doc_embeds = self.encode_documents(positives)
neg_embeds = self.encode_documents(negatives) # Кодируем negatives

self.optimizer.zero_grad()
# Используем InfoNCE и передаем negative_embeddings (с размерностью [batch, 1, embed_dim])
loss = self.loss_fn(
    query_embeddings=query_embeds, 
    positive_embeddings=doc_embeds,
    negative_embeddings=neg_embeds.unsqueeze(1)
)
```

#### Шаг 2: Обучайтесь на длинных документах (Document-level Training)
Поскольку контекстом в PopQA выступают длинные статьи (`s_wiki_url`), проектор должен учиться маппингу токенов компрессии длинных текстов.
Замените датасет MS MARCO Passage на:
- **MS MARCO Document Ranking** (где в качестве `positive` выступают веб-страницы целиком).
- Или используйте полные статьи Википедии на основе датасетов Natural Questions (NQ) или самого PopQA для генерации обучающих триплетов.

#### Шаг 3: Асимметричный энкодинг (Dual Encoder)
Не используйте `OSCAR.compress_documents` для коротких запросов. Вместо этого извлекайте эмбеддинги запроса стандартным способом из базовой языковой модели, а `MEMProjector` применяйте только к сжатым документам.

Добавьте отдельный метод для вопросов:
```python
def encode_queries(self, queries: list[str]) -> torch.Tensor:
    # Пример: используем классический [EOS] или mean pooling с базовой модели 
    inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True).to(self.device)
    with torch.inference_mode():
        outputs = self.oscar_model(**inputs)
        # Получаем стандартный скрытый вектор (например, последнего токена)
        query_vectors = outputs.last_hidden_state[:, -1, :] 
    # Опционально: пропускаем через отдельный линейный слой
    return self.query_linear(query_vectors) 
```

#### Шаг 4: Откажитесь от усреднения (mean pooling) в проекторе
В `MEMProjector` вы используете `pooler="mean"`. Усреднение 8 me-токенов смешивает все извлеченные факты из статьи Википедии, уничтожая пространственную информацию. Поскольку OSCAR выдает фиксированное количество токенов, эффективнее сделать **Flatten** (развернуть их в один вектор) перед подачей в MLP проектора:

```python
# Вместо усреднения:
flattened = mem_embeddings.view(batch_size, -1) # Развертка в размерность [batch, 8 * hidden_dim]
embeddings = self.projector(flattened) # MLP слой: nn.Linear(8 * hidden_dim, embed_dim)
```
Это позволит проектору понять, за какую семантическую часть документа отвечает каждый из 8 токенов в отдельности.

Для обучения проектора, который будет работать со сжатыми документами (эмбеддингами OSCAR) и длинным контекстом, как в PopQA, вам нужен датасет в формате **Document Retrieval** (поиск целых документов) с правильно подобранными hard negatives (сложными негативными примерами). [habr](https://habr.com/ru/companies/magnus-tech/articles/954130/)

MS MARCO Passage, который вы использовали ранее, не подходит, так как средняя длина текстов в нем около 60 слов. Вам нужны полные статьи Википедии или длинные веб-страницы.

### 1. Какой датасет использовать?

Для вашей задачи лучше всего подойдут следующие датасеты:

**Вариант А: MS MARCO Document Ranking (Hard Negatives)**
Это версия MS MARCO, но в качестве контекста используются полные веб-документы. Существуют уже готовые версии датасета со сгенерированными hard negatives (например, на Hugging Face: `Tevatron/msmarco-doc-corpus` или аналоги).

**Вариант Б: Natural Questions (NQ) / PopQA Training Split**
PopQA фокусируется на редких сущностях (long-tail). Вы можете использовать тренировочный сплит Natural Questions или сам PopQA, сопоставив вопросы с полными статьями Википедии, которые выступают в роли `s_wiki_url`. [alimbekov](https://alimbekov.com/llms-full.txt)

### 2. Как должен выглядеть формат данных?

Для успешного contrastive learning (особенно с `InfoNCELoss`) каждая запись в датасете должна быть **триплетом**:
1. **Query (Запрос)**: Короткий вопрос (например, *"What is the capital of France?"*).
2. **Positive Document (Позитивный контекст)**: Длинный документ или статья Википедии, содержащая правильный ответ.
3. **Hard Negative Document (Сложный негативный контекст)**: Длинный документ, который лексически или семантически похож на запрос, но **не содержит** ответа (например, статья про историю Франции или про столицу Италии).

В коде (формат Hugging Face Dataset или JSONL) это должно выглядеть так:
```json
{
  "query": "who played the doctor in doctor who season 10",
  "positive": "Peter Capaldi is a Scottish actor... [Полная статья Википедии про актера]",
  "negatives": [
    "Jodie Whittaker is an English actress... [Статья про другого Доктора]",
    "Doctor Who is a British science fiction television... [Общая статья про сериал]"
  ]
}
```

### 3. Как генерировать Hard Negatives для статей Википедии?

Если вы решите собрать датасет из PopQA/NQ самостоятельно, вам придется сгенерировать hard negatives (поскольку "из коробки" их там может не быть). 

Лучший способ сделать это без ручной разметки — использовать метод **BM25 mining** или **Dense Retrieval mining**:
1. Проиндексируйте весь корпус статей Википедии с помощью BM25 (например, через ElasticSearch или библиотеку `pyserini`) или сильной эмбеддинг-модели (например, `Salesforce/SFR-Embedding-Mistral`).
2. Для каждого вопроса (`query`) из тренировочной выборки сделайте поиск по корпусу.
3. Отфильтруйте позитивный документ (тот, что отмечен как правильный `s_wiki_url`).
4. Возьмите топ-10..50 оставшихся документов — это и будут ваши hard negatives. Они релевантны по ключевым словам или семантике, но являются ложными.

### 4. Интеграция в ваш DataLoader

В вашем коде нужно обновить `__getitem__`, чтобы он возвращал список или массив негативных текстов (вместо одной строки, как было для `mnr` loss). 

```python
def __getitem__(self, idx):
    item = self.dataset[idx]
    # Убедитесь, что 'negatives' - это список строк
    return {
        "query": item["query"],
        "positive": item["positive"],
        "negatives": item["negatives"][:self.num_negatives] # Берем, например, 4 hard negatives
    }
```
Количество hard negatives на один запрос ограничено только видеопамятью вашей GPU. Обычно используют от 1 до 7 hard negatives на один батч.

Разве не существует уже готового датасета типа ms-marco, но с длинными текстами, чтобы я мог работать с PopQA? Можно ли дистиллироваться из уже готового энкодера?

Да, готовые датасеты существуют, и вы также можете использовать дистилляцию, чтобы упростить процесс обучения. Давайте разберем оба подхода.

### 1. Готовые датасеты с длинными документами

Вам не нужно вручную собирать и размечать тексты. В экосистеме MS MARCO уже есть готовый датасет для длинных текстов, который идеально подходит под вашу задачу: **MS MARCO Document Ranking Dataset**. [microsoft.github](https://microsoft.github.io/msmarco/Datasets.html)

В отличие от *Passage Ranking* (который вы использовали и где тексты по 50-60 слов), *Document Ranking* содержит полные веб-страницы (сотни и тысячи слов), что структурно очень близко к статьям Википедии из PopQA. 

**Как его использовать:**
Вы можете загрузить уже готовую, предобработанную версию с Hugging Face, в которой уже собраны позитивные и негативные примеры (hard negatives):
- `Tevatron/msmarco-doc-corpus`
- `mteb/msmarco-v2` (версия для длинных текстов) [huggingface](https://huggingface.co/datasets/mteb/msmarco-v2)
- `namespace-Pt/msmarco-document` (часто используется для обучения ретриверов на длинных текстах).

Пример структуры в таких датасетах:
```python
# Структура одного сэмпла из MS MARCO Document Ranking
{
    "query": "what is the boiling point of water",
    "positive_document": "Water is a transparent... [длинная статья на 1000+ слов]",
    "hard_negative_documents": [
        "Boiling points of various liquids... [длинная статья без прямого ответа]",
        "The chemical properties of H2O... [длинная статья без нужных цифров]"
    ]
}
```

### 2. Можно ли дистиллироваться из готового энкодера?

Да, **Knowledge Distillation (Дистилляция знаний)** — это отличная идея, и для вашей задачи (перевод 8 me-токенов OSCAR в пространство эмбеддингов для ретрива) она сработает даже лучше и стабильнее, чем классическое contrastive learning (InfoNCE). 

Поскольку вы уже упомянули, что `Salesforce/SFR-Embedding-Mistral` отлично справляется с PopQA, вы можете использовать его как "модель-учителя" (Teacher), а ваш MLP-проектор поверх OSCAR — как "ученика" (Student).

#### Как работает дистилляция в вашем случае:
Вам больше не нужны негативные примеры (negatives) и сложные лосс-функции типа InfoNCE или MNR. Вы просто заставляете ваш проектор выдавать такие же векторы, как SFR-Embedding-Mistral.

**Step-by-step для дистилляции:**

1. **Подготовка датасета:**
   Берете любой корпус длинных текстов (тот же MS MARCO Document, статьи из Википедии или просто тексты из PopQA). Вам даже не нужны вопросы, только сами тексты!

2. **Получение таргетов (Teacher Embeddings):**
   Прогоняете эти длинные тексты через `Salesforce/SFR-Embedding-Mistral` и сохраняете полученные эмбеддинги. Это будут ваши "идеальные" векторы.
   *Размерность SFR-Mistral обычно 4096. Если ваш ретрив-движок требует 768, вам нужно будет либо обрезать векторы учителя, либо обучать проектор на выход 4096.*

3. **Обучение проектора (Student):**
   Прогоняете те же тексты через `OSCAR.compress_documents`, получаете 8 me-токенов. Разворачиваете их (Flatten) и пропускаете через ваш MLP.

4. **Loss-функция:**
   Используете обычный **MSE Loss** (Mean Squared Error) или **Cosine Embedding Loss** между выходом вашего MLP и вектором от SFR-Mistral.

```python
# Пример логики дистилляции в train_epoch
def train_epoch(self, epoch: int):
    # ...
    for batch in dataloader:
        documents = batch["documents"] # Только тексты, без вопросов и негативов
        
        # 1. Teacher (SFR-Mistral) - можно предрасчитать заранее!
        with torch.no_grad():
            target_embeds = teacher_model.encode(documents)
            
        # 2. Student (OSCAR + MLP)
        oscar_tokens = self.oscar_model.compress_documents(documents)
        flattened_tokens = oscar_tokens.view(batch_size, -1)
        student_embeds = self.projector(flattened_tokens)
        
        # 3. Loss (учим студента повторять за учителем)
        # MSE Loss отлично работает для дистилляции эмбеддингов
        loss = torch.nn.functional.mse_loss(student_embeds, target_embeds)
        
        # backward...
```

**Почему дистилляция лучше для вашей задачи:**
- Не нужно собирать сложные триплеты и майнить hard negatives.
- Учитель (SFR-Mistral) уже умеет читать длинные статьи и понимать, какая информация в них важна. Ваш проектор просто выучит это распределение.
- Вы можете обучать проектор на вообще любых длинных текстах (без разметки), просто скачав дамп Википедии.

**Резюме:** Используйте датасет `msmarco-document`. Если есть ресурсы для инференса учителя, переключитесь на дистилляцию через MSE Loss от `Salesforce/SFR-Embedding-Mistral` — это значительно упростит пайплайн и даст метрики, близкие к самому учителю. [microsoft.github](https://microsoft.github.io/msmarco/Datasets.html)