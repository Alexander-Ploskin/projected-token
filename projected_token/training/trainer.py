import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any
from tqdm import tqdm
import os
import json
from datetime import datetime
from pathlib import Path

from projected_token.training.losses import get_loss_fn, MultipleNegativesRankingLoss
from projected_token.io import write_csv, write_json
from projected_token.artifacts import metrics_to_rows
from projected_token.plotting import plot_training_curves


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
        run_root: str | None = None,
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
        self.run_root = Path(run_root) if run_root else (
            Path(output_dir).parent if Path(output_dir).name == "checkpoints" else Path(output_dir)
        )
        self.metrics_dir = self.run_root / "metrics"
        self.plots_dir = self.run_root / "plots"

        self.model.to(self.device)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.model_config = {}
        self.train_history: list[dict[str, Any]] = []
        self.val_history: list[dict[str, Any]] = []

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
        mrr_at_10 = 0.0
        ndcg_at_10 = 0.0
        recall_at_1 = 0.0
        recall_at_5 = 0.0
        recall_at_10 = 0.0
        batch_size = query_embeds.size(0)

        for i in range(batch_size):
            rank = ranks[i, i].item() + 1
            mrr += 1.0 / rank
            if rank <= 10:
                mrr_at_10 += 1.0 / rank
                ndcg_at_10 += 1.0 / torch.log2(torch.tensor(rank + 1.0)).item()
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
        total_mrr_at_10 = 0.0
        total_ndcg_at_10 = 0.0
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
                total_mrr_at_10 += metrics["mrr@10"]
                total_ndcg_at_10 += metrics["ndcg@10"]
                total_recall_at_1 += metrics["recall@1"]
                total_recall_at_5 += metrics["recall@5"]
                total_recall_at_10 += metrics["recall@10"]
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mrr = total_mrr / num_batches
        avg_mrr_at_10 = total_mrr_at_10 / num_batches
        avg_ndcg_at_10 = total_ndcg_at_10 / num_batches
        avg_recall_at_1 = total_recall_at_1 / num_batches
        avg_recall_at_5 = total_recall_at_5 / num_batches
        avg_recall_at_10 = total_recall_at_10 / num_batches

        self.writer.add_scalar("val/loss", avg_loss, step)
        self.writer.add_scalar("val/mrr", avg_mrr, step)
        self.writer.add_scalar("val/mrr@10", avg_mrr_at_10, step)
        self.writer.add_scalar("val/ndcg@10", avg_ndcg_at_10, step)
        self.writer.add_scalar("val/recall@1", avg_recall_at_1, step)
        self.writer.add_scalar("val/recall@5", avg_recall_at_5, step)
        self.writer.add_scalar("val/recall@10", avg_recall_at_10, step)

        return {
            "loss": avg_loss,
            "mrr": avg_mrr,
            "mrr@10": avg_mrr_at_10,
            "ndcg@10": avg_ndcg_at_10,
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

    def _export_run_reports(self) -> None:
        run_id = self.run_root.name
        write_json(self.metrics_dir / "train_history.json", self.train_history)
        write_json(self.metrics_dir / "val_history.json", self.val_history)

        train_rows: list[dict[str, Any]] = []
        for row in self.train_history:
            step = row.get("step", 0)
            train_rows.extend(metrics_to_rows(
                {k: v for k, v in row.items() if k not in {"step", "epoch"}},
                run_id=run_id,
                dataset="train",
                split=f"epoch_{row.get('epoch', 0)}_step_{step}",
            ))
        val_rows: list[dict[str, Any]] = []
        for row in self.val_history:
            step = row.get("step", 0)
            val_rows.extend(metrics_to_rows(
                {k: v for k, v in row.items() if k not in {"step", "epoch"}},
                run_id=run_id,
                dataset="validation",
                split=f"step_{step}",
            ))

        write_csv(self.metrics_dir / "train_history.csv", train_rows)
        write_csv(self.metrics_dir / "val_history.csv", val_rows)

        plot_training_curves(
            self.train_history,
            x_key="step",
            output_path=self.plots_dir / "training_curves.png",
            title="Training Curves",
            metric_keys=["loss", "mrr"],
        )
        plot_training_curves(
            self.val_history,
            x_key="step",
            output_path=self.plots_dir / "validation_curves.png",
            title="Validation Curves",
            metric_keys=["loss", "mrr", "mrr@10", "ndcg@10", "recall@10"],
        )

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
        self.val_history.append({"step": 0, "epoch": 0, **val_metrics})
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
                self.writer.add_scalar("train/mrr@10", metrics["mrr@10"], step)
                self.writer.add_scalar("train/ndcg@10", metrics["ndcg@10"], step)
                self.writer.add_scalar("train/recall@10", metrics["recall@10"], step)
                self.train_history.append({"step": step, "epoch": epoch, "loss": loss.item(), **metrics})
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": step})

                # Validation & checkpoint every N steps
                if step % val_every_n_steps == 0:
                    print(f"\n--- Step {step} - Validation ---")
                    val_metrics = self.validate(step)
                    print(f"Val - Loss: {val_metrics['loss']:.4f}, MRR: {val_metrics['mrr']:.4f}")
                    print(f"Val - R@1: {val_metrics['recall@1']:.4f}, R@5: {val_metrics['recall@5']:.4f}, R@10: {val_metrics['recall@10']:.4f}")
                    self.val_history.append({"step": step, "epoch": epoch, **val_metrics})

                    is_best = val_metrics["loss"] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics["loss"]
                        print(f"New best model! Val loss: {best_val_loss:.4f}")

                    self.save_checkpoint(step, {"val": val_metrics}, is_best, self.model_config)
                    self.model.train()  # Back to train mode

        print("\nTraining completed!")
        self._export_run_reports()
        self.writer.close()