#!/usr/bin/env python3
"""Train OSCAR+projector with explicit query/document BGE teacher targets."""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModel

from projected_token.artifacts import write_config_lock
from projected_token.config import load_yaml
from projected_token.encoders.projector import MEMProjector
from projected_token.oscar_runtime import disable_transformers_allocator_warmup, configure_oscar_component_devices


def _resolve_h5_paths(paths_or_patterns: Union[str, Sequence[str]]) -> list[str]:
    raw_paths = [paths_or_patterns] if isinstance(paths_or_patterns, str) else list(paths_or_patterns)
    resolved: list[str] = []
    for raw in raw_paths:
        expanded = os.path.expandvars(os.path.expanduser(str(raw)))
        path = Path(expanded)
        if path.is_dir():
            matches = sorted(str(p) for p in path.glob("*.h5"))
        elif glob.has_magic(expanded):
            matches = sorted(glob.glob(expanded))
        else:
            matches = [expanded]
        resolved.extend(matches)
    deduped = list(dict.fromkeys(resolved))
    missing = [p for p in deduped if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Teacher embedding file(s) not found: {missing}")
    if not deduped:
        raise FileNotFoundError(f"No teacher embedding .h5 files matched: {paths_or_patterns}")
    return deduped


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


class QueryDocH5Dataset(Dataset):
    def __init__(self, h5_paths: Sequence[str], split: str = "train", val_split: float = 0.05):
        self.h5_paths = [str(p) for p in h5_paths]
        self.files = [h5py.File(p, "r") for p in self.h5_paths]
        self.file_sources = []
        self.index_map: list[tuple[int, int]] = []
        for fi, h5 in enumerate(self.files):
            source_name = self._infer_file_source(Path(self.h5_paths[fi]))
            self.file_sources.append(source_name)
            total = int(h5["query_embeddings"].shape[0])
            val_size = int(total * val_split)
            if val_split > 0 and total > 1:
                val_size = max(1, val_size)
            train_size = total - val_size
            index_range = range(0, train_size) if split == "train" else range(train_size, total)
            self.index_map.extend((fi, i) for i in index_range)
        print(f"Loaded query-doc {split} split: {len(self.index_map)} / files {[Path(p).name for p in self.h5_paths]}")

    @staticmethod
    def _infer_file_source(path: Path) -> str:
        stem = path.stem.lower()
        for candidate in ("msmarco", "fiqa", "quora", "nfcorpus", "scifact", "arguana"):
            if candidate in stem:
                return candidate
        return stem

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fi, row = self.index_map[idx]
        h5 = self.files[fi]
        return {
            "query": _decode_text(h5["queries"][row]),
            "positive": _decode_text(h5["positive_docs"][row]),
            "negative": _decode_text(h5["negative_docs"][row]),
            "query_target": torch.tensor(h5["query_embeddings"][row], dtype=torch.float32),
            "positive_target": torch.tensor(h5["positive_embeddings"][row], dtype=torch.float32),
            "negative_target": torch.tensor(h5["negative_embeddings"][row], dtype=torch.float32),
            "source": self.file_sources[fi],
        }

    def close(self) -> None:
        for h5 in self.files:
            h5.close()


def collate_query_doc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queries": [x["query"] for x in batch],
        "positives": [x["positive"] for x in batch],
        "negatives": [x["negative"] for x in batch],
        "query_targets": torch.stack([x["query_target"] for x in batch]),
        "positive_targets": torch.stack([x["positive_target"] for x in batch]),
        "negative_targets": torch.stack([x["negative_target"] for x in batch]),
        "sources": [x["source"] for x in batch],
    }


class QueryDistillationTrainer:
    def __init__(
        self,
        *,
        oscar_model_name: str,
        teacher_embeddings_path: Union[str, Sequence[str]],
        embed_dim: int = 768,
        pooler: str = "flatten",
        num_layers: int = 2,
        dropout: float = 0.0,
        projector_hidden_dim: int = 8192,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        lr: float = 5e-5,
        min_lr: float = 1e-6,
        max_steps: int = 0,
        warmup_steps: int = 0,
        scheduler: str = "cosine",
        val_split: float = 0.02,
        query_mse_weight: float = 1.0,
        doc_mse_weight: float = 1.0,
        negative_mse_weight: float = 0.25,
        ranking_weight: float = 0.05,
        infonce_weight: float = 0.0,
        temperature: float = 0.07,
        margin: float = 0.05,
        beir_probe_config: str | None = None,
        beir_probe_samples: int = 20,
        beir_probe_samples_per_dataset: int | None = None,
        beir_probe_negatives: int = 20,
        beir_probe_batch_size: int = 8,
        selection_metric: str = "beir_proxy_ndcg@10",
        dataset_sampling_weights: dict[str, float] | None = None,
        device: str = "cuda:0",
        output_dir: str = "checkpoints/query_distill",
        log_dir: str = "logs/query_distill",
    ):
        self.device = torch.device(device)
        self.oscar_model_name = oscar_model_name
        self.batch_size = int(batch_size)
        self.gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        self.base_lr = float(lr)
        self.min_lr = float(min_lr)
        self.max_steps = int(max_steps or 0)
        self.warmup_steps = int(warmup_steps or 0)
        self.scheduler_name = scheduler
        self.val_split = float(val_split)
        self.query_mse_weight = float(query_mse_weight)
        self.doc_mse_weight = float(doc_mse_weight)
        self.negative_mse_weight = float(negative_mse_weight)
        self.ranking_weight = float(ranking_weight)
        self.infonce_weight = float(infonce_weight)
        self.temperature = float(temperature)
        self.margin = float(margin)
        self.beir_probe_config = beir_probe_config
        self.beir_probe_samples = int(beir_probe_samples)
        self.beir_probe_samples_per_dataset = (
            int(beir_probe_samples_per_dataset) if beir_probe_samples_per_dataset is not None else None
        )
        self.beir_probe_negatives = int(beir_probe_negatives)
        self.beir_probe_batch_size = int(beir_probe_batch_size)
        self.selection_metric = str(selection_metric)
        self.dataset_sampling_weights = {str(k): float(v) for k, v in (dataset_sampling_weights or {}).items()}
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.run_root = self.output_dir.parent if self.output_dir.name == "checkpoints" else self.output_dir
        self.metrics_dir = self.run_root / "metrics"
        self.plots_dir = self.run_root / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.val_history_path = self.metrics_dir / "query_distill_val_history.json"
        self.checkpoint_val_history_path = self.output_dir / "val_metrics_history.json"
        self.val_history: list[dict[str, float]] = self._load_val_history()

        disable_transformers_allocator_warmup()
        print(f"Loading OSCAR model: {oscar_model_name}")
        self.oscar_model = AutoModel.from_pretrained(
            oscar_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device).eval()
        configure_oscar_component_devices(self.oscar_model)
        if hasattr(self.oscar_model, "compr") and hasattr(self.oscar_model.compr, "config"):
            self.oscar_model.compr.config.mean_resizing = False
        hidden_size = int(self.oscar_model.compress_documents(["test"]).shape[-1])
        print(f"OSCAR hidden size: {hidden_size}")

        h5_paths = _resolve_h5_paths(teacher_embeddings_path)
        self.train_dataset = QueryDocH5Dataset(h5_paths, split="train", val_split=val_split)
        self.val_dataset = QueryDocH5Dataset(h5_paths, split="val", val_split=val_split)
        train_sampler = self._build_weighted_sampler(self.train_dataset, self.dataset_sampling_weights)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_query_doc,
            num_workers=4,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_query_doc,
            num_workers=2,
        )

        self.projector = MEMProjector(
            hidden_dim=hidden_size,
            embed_dim=embed_dim,
            pooler=pooler,
            num_layers=num_layers,
            dropout=dropout,
            projector_hidden_dim=projector_hidden_dim,
        ).to(device=self.device, dtype=torch.bfloat16)
        print("Query distillation projector architecture:")
        print(self.projector)
        print(f"Projector parameters: {sum(p.numel() for p in self.projector.parameters()):,}")
        self.optimizer = torch.optim.AdamW(self.projector.parameters(), lr=self.base_lr, weight_decay=0.01)
        self.projector_config = {
            "oscar_hidden_dim": hidden_size,
            "oscar_model_name": self.oscar_model_name,
            "embed_dim": embed_dim,
            "pooler": pooler,
            "projector_hidden_dim": projector_hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "selection_metric": self.selection_metric,
            "projector_type": "mem",
            "training_objective": "query_doc_bge_distill",
            "dataset_sampling_weights": self.dataset_sampling_weights,
        }
        self._beir_probe_cases = self._load_beir_probe_cases()
        self.load_checkpoint()

    def _load_val_history(self) -> list[dict[str, float]]:
        for path in (self.val_history_path, self.checkpoint_val_history_path):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[val-history] cannot parse {path}: {exc}")
                continue
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
        return []

    def _write_val_history(self) -> None:
        payload = json.dumps(self.val_history, indent=2, ensure_ascii=False)
        self.val_history_path.write_text(payload, encoding="utf-8")
        self.checkpoint_val_history_path.write_text(payload, encoding="utf-8")

    def _record_val_metrics(self, step: int, metrics: dict[str, float]) -> None:
        row = {"step": int(step), **metrics}
        for idx, existing in enumerate(self.val_history):
            if int(existing.get("step", -1)) == int(step):
                self.val_history[idx] = row
                break
        else:
            self.val_history.append(row)
        self.val_history.sort(key=lambda item: int(item.get("step", -1)))
        self._write_val_history()

    def load_checkpoint(self) -> None:
        best_path = self.output_dir / "best_model.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device)
            self.projector.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print(f"Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")

    def _step_lr(self, step: int) -> float:
        if self.max_steps <= 0 or self.scheduler_name in {"legacy", "none"}:
            return float(self.optimizer.param_groups[0]["lr"])
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            lr = self.base_lr * step / max(1, self.warmup_steps)
        else:
            denom = max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, (step - self.warmup_steps) / denom))
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return float(lr)

    def _build_weighted_sampler(
        self,
        dataset: QueryDocH5Dataset,
        sampling_weights: dict[str, float],
    ) -> WeightedRandomSampler | None:
        if not sampling_weights:
            return None
        weights = np.zeros(len(dataset.index_map), dtype=np.float64)
        source_counts = collections.Counter()
        unknown_sources: set[str] = set()
        for idx, (fi, _row) in enumerate(dataset.index_map):
            source = dataset.file_sources[fi]
            source_counts[source] += 1
            if source in sampling_weights:
                weights[idx] = sampling_weights[source]
            else:
                unknown_sources.add(source)
        if np.all(weights == 0):
            print(
                f"[sampling] no dataset_sampling_weights matched sources={sorted(source_counts.keys())}; falling back to shuffle",
                flush=True,
            )
            return None
        if unknown_sources:
            print(
                f"[sampling] sources without explicit weights={sorted(unknown_sources)}; assigned zero probability",
                flush=True,
            )
        normalized = weights / max(weights.sum(), 1e-12)
        print(
            "[sampling] configured weights="
            + ", ".join(f"{k}:{v}" for k, v in sorted(sampling_weights.items())),
            flush=True,
        )
        print(
            "[sampling] source counts="
            + ", ".join(f"{k}:{source_counts[k]}" for k in sorted(source_counts.keys())),
            flush=True,
        )
        return WeightedRandomSampler(
            weights=torch.from_numpy(normalized),
            num_samples=len(dataset.index_map),
            replacement=True,
        )

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        with torch.no_grad():
            mem_embeddings = self.oscar_model.compress_documents(documents=texts)
        return self.projector(mem_embeddings)

    def compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        q = F.normalize(self.encode_texts(batch["queries"]).float(), dim=-1)
        p = F.normalize(self.encode_texts(batch["positives"]).float(), dim=-1)
        n = F.normalize(self.encode_texts(batch["negatives"]).float(), dim=-1)
        q_t = F.normalize(batch["query_targets"].to(self.device).float(), dim=-1)
        p_t = F.normalize(batch["positive_targets"].to(self.device).float(), dim=-1)
        n_t = F.normalize(batch["negative_targets"].to(self.device).float(), dim=-1)

        q_mse = F.mse_loss(q, q_t)
        p_mse = F.mse_loss(p, p_t)
        n_mse = F.mse_loss(n, n_t)
        pos_sim = (q * p).sum(dim=-1)
        neg_sim = (q * n).sum(dim=-1)
        gap = pos_sim - neg_sim
        ranking_loss = torch.relu(self.margin - pos_sim + neg_sim).mean()
        logits = torch.matmul(q, p.T) / max(self.temperature, 1e-6)
        labels = torch.arange(q.size(0), device=q.device)
        infonce_loss = F.cross_entropy(logits, labels)
        total = (
            self.query_mse_weight * q_mse
            + self.doc_mse_weight * p_mse
            + self.negative_mse_weight * n_mse
            + self.ranking_weight * ranking_loss
            + self.infonce_weight * infonce_loss
        )
        metrics = {
            "query_mse": float(q_mse.item()),
            "positive_mse": float(p_mse.item()),
            "negative_mse": float(n_mse.item()),
            "ranking_loss": float(ranking_loss.item()),
            "infonce_loss": float(infonce_loss.item()),
            "pos_sim": float(pos_sim.mean().item()),
            "neg_sim": float(neg_sim.mean().item()),
            "gap": float(gap.mean().item()),
            "pos_sim_std": float(pos_sim.std(unbiased=False).item()),
            "neg_sim_std": float(neg_sim.std(unbiased=False).item()),
            "gap_std": float(gap.std(unbiased=False).item()),
            "gap_min": float(gap.min().item()),
            "gap_max": float(gap.max().item()),
            "ranking_active_frac": float((self.margin - pos_sim + neg_sim > 0).float().mean().item()),
        }
        return total, metrics

    def validate(self, step: int) -> dict[str, float]:
        self.projector.eval()
        totals: dict[str, float] = {}
        per_batch: dict[str, list[float]] = {}
        count = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                loss, metrics = self.compute_loss(batch)
                metrics["loss_total"] = float(loss.item())
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value
                    per_batch.setdefault(key, []).append(float(value))
                count += 1
        out = {key: value / max(1, count) for key, value in totals.items()}
        for key, values in per_batch.items():
            arr = np.asarray(values, dtype=np.float32)
            out[f"{key}_batch_std"] = float(arr.std())
            out[f"{key}_batch_min"] = float(arr.min())
            out[f"{key}_batch_max"] = float(arr.max())
        out.update(self.run_beir_probe(step))
        print(f"[val] step={step}", flush=True)
        print(
            "[val] losses "
            + " ".join(
                f"{key}={out[key]:.6f}"
                for key in ("loss_total", "query_mse", "positive_mse", "negative_mse", "ranking_loss", "infonce_loss")
                if key in out
            ),
            flush=True,
        )
        print(
            "[val] retrieval "
            + " ".join(
                f"{key}={out[key]:.6f}"
                for key in (
                    "pos_sim",
                    "neg_sim",
                    "gap",
                    "pos_sim_std",
                    "neg_sim_std",
                    "gap_std",
                    "gap_min",
                    "gap_max",
                    "ranking_active_frac",
                    "beir_probe_mrr@10",
                    "beir_probe_ndcg@10",
                    "beir_interquery_ndcg@10",
                    "beir_probe_recall@10",
                )
                if key in out
            ),
            flush=True,
        )
        batch_stat_keys = [key for key in out if key.endswith(("_batch_std", "_batch_min", "_batch_max"))]
        if batch_stat_keys:
            print("[val] batch_stats " + " ".join(f"{key}={out[key]:.6f}" for key in sorted(batch_stat_keys)), flush=True)
        probe_keys = [key for key in out if key.startswith("beir_probe_") and key not in {"beir_probe_mrr@10", "beir_probe_recall@10"}]
        if probe_keys:
            print("[val] beir_probe " + " ".join(f"{key}={out[key]:.6f}" for key in sorted(probe_keys)), flush=True)
        for key, value in out.items():
            self.writer.add_scalar(f"val/{key}", value, step)
        self.writer.add_scalar("val/meta/num_batches", count, step)
        self.writer.add_scalar("val/meta/num_examples", len(self.val_dataset), step)
        if per_batch:
            validation_detail = {
                "step": int(step),
                "metrics": out,
                "per_batch": per_batch,
            }
            (self.metrics_dir / f"validation_step_{step}.json").write_text(
                json.dumps(validation_detail, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.writer.flush()
        return out

    def save_checkpoint(self, step: int, metrics: dict[str, float], is_best: bool = False) -> None:
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "model_state_dict": self.projector.state_dict(),
            "config": self.projector_config,
        }
        torch.save(checkpoint, self.output_dir / f"checkpoint_step_{step}.pt")
        (self.output_dir / "config.json").write_text(json.dumps(self.projector_config, indent=2), encoding="utf-8")
        self._record_val_metrics(step, metrics)
        (self.output_dir / "metrics.json").write_text(
            json.dumps({"step": int(step), "val": metrics}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pt")
            (self.output_dir / "best_config.json").write_text(json.dumps(self.projector_config, indent=2), encoding="utf-8")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _load_beir_probe_cases(self) -> list[dict[str, str]]:
        if not self.beir_probe_config:
            return []
        cfg = load_yaml(self.beir_probe_config)
        datasets = cfg.get("datasets", [])
        if not datasets:
            return []
        cases: list[dict[str, str]] = []
        samples_per_dataset = self.beir_probe_samples_per_dataset or self.beir_probe_samples
        for dataset_idx, dataset_item in enumerate(datasets):
            dataset_name = str(dataset_item.get("name", f"beir_{dataset_idx}"))
            dataset_dir = Path(dataset_item["path"])
            qrels_path = dataset_dir / "qrels" / f"{cfg.get('split', 'test')}.tsv"
            try:
                corpus_rows = self._read_jsonl(dataset_dir / "corpus.jsonl")
                query_rows = self._read_jsonl(dataset_dir / "queries.jsonl")
            except Exception as exc:
                print(f"[beir-probe] cannot load corpus/queries from {dataset_dir}: {exc}")
                continue
            corpus = {
                str(r["_id"]): " ".join(
                    p for p in [str(r.get("title", "")).strip(), str(r.get("text", "")).strip()] if p
                ).strip()
                for r in corpus_rows
            }
            queries = {str(r["_id"]): str(r.get("text", "")) for r in query_rows}
            relevant: dict[str, set[str]] = {}
            try:
                with qrels_path.open("r", encoding="utf-8") as fp:
                    _ = fp.readline()
                    for line in fp:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3 and int(parts[2]) > 0:
                            relevant.setdefault(str(parts[0]), set()).add(str(parts[1]))
            except Exception as exc:
                print(f"[beir-probe] cannot read qrels {qrels_path}: {exc}")
                continue
            corpus_ids = list(corpus.keys())
            candidate_qids = [qid for qid in relevant.keys() if qid in queries]
            rng = np.random.default_rng(42 + dataset_idx)
            rng.shuffle(candidate_qids)
            dataset_cases = 0
            for qid in candidate_qids:
                pos_ids = list(relevant.get(qid, set()))
                if not pos_ids:
                    continue
                pos_text = corpus.get(pos_ids[0], "")
                neg_texts = [corpus[cid] for cid in corpus_ids if cid not in relevant[qid] and corpus.get(cid, "")]
                if pos_text and neg_texts:
                    cases.append(
                        {
                            "dataset": dataset_name,
                            "query_id": qid,
                            "positive_doc_id": pos_ids[0],
                            "query": queries[qid],
                            "positive": pos_text,
                            "negative": neg_texts[0],
                            "negatives": neg_texts[: self.beir_probe_negatives],
                        }
                    )
                    dataset_cases += 1
                if dataset_cases >= samples_per_dataset:
                    break
            print(f"[beir-probe] loaded {dataset_cases} cases from {dataset_name}")
        print(f"[beir-probe] loaded {len(cases)} total cases from {len(datasets)} configured datasets")
        return cases

    def _encode_probe_texts(self, texts: list[str]) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), self.beir_probe_batch_size):
            batch = texts[start : start + self.beir_probe_batch_size]
            chunks.append(F.normalize(self.encode_texts(batch).float(), dim=-1))
        return torch.cat(chunks, dim=0)

    def run_beir_probe(self, step: int) -> dict[str, float]:
        if not self._beir_probe_cases:
            return {}
        with torch.no_grad():
            queries = [c["query"] for c in self._beir_probe_cases]
            positives = [c["positive"] for c in self._beir_probe_cases]
            negatives = [c["negative"] for c in self._beir_probe_cases]
            q_emb = self._encode_probe_texts(queries)
            p_emb = self._encode_probe_texts(positives)
            n_emb = self._encode_probe_texts(negatives)
            pos_cos = (q_emb * p_emb).sum(dim=-1).cpu().numpy()
            neg_cos = (q_emb * n_emb).sum(dim=-1).cpu().numpy()
            candidate_texts: list[str] = []
            offsets: list[tuple[int, int]] = []
            for case in self._beir_probe_cases:
                start = len(candidate_texts)
                candidate_texts.append(case["positive"])
                candidate_texts.extend(case.get("negatives", [case["negative"]]))
                offsets.append((start, len(candidate_texts)))
            cand_emb = self._encode_probe_texts(candidate_texts)
            similarities = torch.matmul(q_emb, cand_emb.T).cpu().numpy()
        mrr_at_10 = 0.0
        recall_at_10 = 0.0
        ndcg_at_10 = 0.0
        interquery_mrr_at_10 = 0.0
        interquery_recall_at_10 = 0.0
        interquery_ndcg_at_10 = 0.0
        ranks: list[int] = []
        interquery_ranks: list[int] = []
        reciprocal_ranks: list[float] = []
        discounted_gains: list[float] = []
        for i, (start, stop) in enumerate(offsets):
            local_scores = similarities[i, start:stop]
            rank = int(np.where(np.argsort(-local_scores) == 0)[0][0]) + 1
            interquery_rank = int(np.where(np.argsort(-similarities[i]) == start)[0][0]) + 1
            ranks.append(rank)
            interquery_ranks.append(interquery_rank)
            reciprocal_ranks.append(1.0 / rank)
            discounted_gain = 1.0 / float(np.log2(rank + 1.0)) if rank <= 10 else 0.0
            discounted_gains.append(discounted_gain)
            if rank <= 10:
                mrr_at_10 += 1.0 / rank
                recall_at_10 += 1.0
                ndcg_at_10 += discounted_gain
            if interquery_rank <= 10:
                interquery_mrr_at_10 += 1.0 / interquery_rank
                interquery_recall_at_10 += 1.0
                interquery_ndcg_at_10 += 1.0 / float(np.log2(interquery_rank + 1.0))
        denom = max(1, len(offsets))
        pos_mean = float(np.mean(pos_cos))
        neg_mean = float(np.mean(neg_cos))
        gaps = pos_cos - neg_cos
        gap_mean = float(np.mean(gaps))
        rank_arr = np.asarray(ranks, dtype=np.float32)
        interquery_rank_arr = np.asarray(interquery_ranks, dtype=np.float32)
        rr_arr = np.asarray(reciprocal_ranks, dtype=np.float32)
        ndcg_arr = np.asarray(discounted_gains, dtype=np.float32)
        datasets = [str(case.get("dataset", "beir")) for case in self._beir_probe_cases]
        dataset_summaries: dict[str, dict[str, float]] = {}
        for dataset_name in sorted(set(datasets)):
            indices = np.asarray([i for i, name in enumerate(datasets) if name == dataset_name], dtype=np.int64)
            if indices.size == 0:
                continue
            ds_ranks = rank_arr[indices]
            ds_interquery_ranks = interquery_rank_arr[indices]
            ds_rr = rr_arr[indices]
            ds_ndcg = ndcg_arr[indices]
            ds_recall_at_10 = float(np.mean(ds_ranks <= 10))
            ds_interquery_recall_at_10 = float(np.mean(ds_interquery_ranks <= 10))
            dataset_summaries[dataset_name] = {
                "cases": float(indices.size),
                "pos_cosine_mean": float(np.mean(pos_cos[indices])),
                "neg_cosine_mean": float(np.mean(neg_cos[indices])),
                "gap_mean": float(np.mean(gaps[indices])),
                "gap_std": float(np.std(gaps[indices])),
                "rank_mean": float(np.mean(ds_ranks)),
                "rank_median": float(np.median(ds_ranks)),
                "interquery_rank_mean": float(np.mean(ds_interquery_ranks)),
                "interquery_rank_median": float(np.median(ds_interquery_ranks)),
                "mrr": float(np.mean(ds_rr)),
                "mrr@10": float(np.mean(np.where(ds_ranks <= 10, ds_rr, 0.0))),
                "ndcg@10": float(np.mean(ds_ndcg)),
                "recall@10": ds_recall_at_10,
                "interquery_ndcg@10": float(
                    np.mean(np.where(ds_interquery_ranks <= 10, 1.0 / np.log2(ds_interquery_ranks + 1.0), 0.0))
                ),
                "interquery_recall@10": ds_interquery_recall_at_10,
            }
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(pos_cos))
        ax.plot(x, pos_cos, marker="o", label="positive cosine")
        ax.plot(x, neg_cos, marker="x", label="negative cosine")
        ax.axhline(pos_mean, linestyle="--", linewidth=1.0, label=f"pos mean={pos_mean:.3f}")
        ax.axhline(neg_mean, linestyle="--", linewidth=1.0, label=f"neg mean={neg_mean:.3f}")
        ax.set_title(f"Query distill BEIR probe @ step {step}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig_path = self.plots_dir / f"beir_probe_step_{step}.png"
        fig.savefig(fig_path, dpi=120)
        self.writer.add_figure("beir_probe/cosine_plot", fig, step)
        plt.close(fig)
        for dataset_name, summary in dataset_summaries.items():
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)
            indices = np.asarray([i for i, name in enumerate(datasets) if name == dataset_name], dtype=np.int64)
            ds_x = np.arange(indices.size)
            ds_fig, ds_ax = plt.subplots(figsize=(10, 4))
            ds_ax.plot(ds_x, pos_cos[indices], marker="o", label="positive cosine")
            ds_ax.plot(ds_x, neg_cos[indices], marker="x", label="negative cosine")
            ds_ax.axhline(summary["pos_cosine_mean"], linestyle="--", linewidth=1.0, label=f"pos mean={summary['pos_cosine_mean']:.3f}")
            ds_ax.axhline(summary["neg_cosine_mean"], linestyle="--", linewidth=1.0, label=f"neg mean={summary['neg_cosine_mean']:.3f}")
            ds_ax.set_title(f"BEIR probe {dataset_name} @ step {step}")
            ds_ax.grid(alpha=0.25)
            ds_ax.legend(loc="best")
            ds_fig.tight_layout()
            ds_fig_path = self.plots_dir / f"beir_probe_{safe_name}_step_{step}.png"
            ds_fig.savefig(ds_fig_path, dpi=120)
            self.writer.add_figure(f"beir_probe_by_dataset/{dataset_name}/cosine_plot", ds_fig, step)
            plt.close(ds_fig)
            dataset_detail = {
                "step": int(step),
                "dataset": dataset_name,
                "summary": summary,
                "cases": [
                    {
                        "idx": int(i),
                        "query_id": self._beir_probe_cases[i].get("query_id", ""),
                        "positive_doc_id": self._beir_probe_cases[i].get("positive_doc_id", ""),
                        "rank": int(ranks[i]),
                        "interquery_rank": int(interquery_ranks[i]),
                        "pos_cosine": float(pos_cos[i]),
                        "neg_cosine": float(neg_cos[i]),
                        "gap": float(gaps[i]),
                    }
                    for i in indices.tolist()
                ],
                "plot": str(ds_fig_path),
            }
            (self.metrics_dir / f"beir_probe_{safe_name}_step_{step}.json").write_text(
                json.dumps(dataset_detail, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        self.writer.add_histogram("beir_probe/pos_cosine", pos_cos, step)
        self.writer.add_histogram("beir_probe/neg_cosine", neg_cos, step)
        self.writer.add_histogram("beir_probe/gap", gaps, step)
        self.writer.add_histogram("beir_probe/rank", rank_arr, step)
        print(
            f"[beir-probe] step={step} pos_mean={pos_mean:.6f} neg_mean={neg_mean:.6f} "
            f"gap={gap_mean:.6f} gap_std={float(np.std(gaps)):.6f} "
            f"rank_mean={float(rank_arr.mean()) if len(rank_arr) else 0.0:.6f} "
            f"rank_median={float(np.median(rank_arr)) if len(rank_arr) else 0.0:.6f} "
            f"mrr@10={mrr_at_10 / denom:.6f} ndcg@10={ndcg_at_10 / denom:.6f} "
            f"interquery_ndcg@10={interquery_ndcg_at_10 / denom:.6f} "
            f"r@10={recall_at_10 / denom:.6f} plot={fig_path}",
            flush=True,
        )
        probe_detail = {
            "step": int(step),
            "summary": {
                "pos_cosine_mean": pos_mean,
                "pos_cosine_std": float(np.std(pos_cos)),
                "pos_cosine_min": float(np.min(pos_cos)),
                "pos_cosine_max": float(np.max(pos_cos)),
                "neg_cosine_mean": neg_mean,
                "neg_cosine_std": float(np.std(neg_cos)),
                "neg_cosine_min": float(np.min(neg_cos)),
                "neg_cosine_max": float(np.max(neg_cos)),
                "gap_mean": gap_mean,
                "gap_std": float(np.std(gaps)),
                "gap_min": float(np.min(gaps)),
                "gap_max": float(np.max(gaps)),
                "rank_mean": float(rank_arr.mean()) if len(rank_arr) else 0.0,
                "rank_median": float(np.median(rank_arr)) if len(rank_arr) else 0.0,
                "rank_min": int(rank_arr.min()) if len(rank_arr) else 0,
                "rank_max": int(rank_arr.max()) if len(rank_arr) else 0,
                "interquery_rank_mean": float(interquery_rank_arr.mean()) if len(interquery_rank_arr) else 0.0,
                "interquery_rank_median": float(np.median(interquery_rank_arr)) if len(interquery_rank_arr) else 0.0,
                "interquery_rank_min": int(interquery_rank_arr.min()) if len(interquery_rank_arr) else 0,
                "interquery_rank_max": int(interquery_rank_arr.max()) if len(interquery_rank_arr) else 0,
                "mrr": float(rr_arr.mean()) if len(rr_arr) else 0.0,
                "mrr@10": mrr_at_10 / denom,
                "ndcg@10": ndcg_at_10 / denom,
                "recall@10": recall_at_10 / denom,
                "interquery_mrr@10": interquery_mrr_at_10 / denom,
                "interquery_ndcg@10": interquery_ndcg_at_10 / denom,
                "interquery_recall@10": interquery_recall_at_10 / denom,
            },
            "datasets": dataset_summaries,
            "cases": [
                {
                    "idx": int(i),
                    "dataset": self._beir_probe_cases[i].get("dataset", "beir"),
                    "rank": int(ranks[i]),
                    "interquery_rank": int(interquery_ranks[i]),
                    "pos_cosine": float(pos_cos[i]),
                    "neg_cosine": float(neg_cos[i]),
                    "gap": float(gaps[i]),
                }
                for i in range(len(ranks))
            ],
        }
        (self.metrics_dir / f"beir_probe_step_{step}.json").write_text(
            json.dumps(probe_detail, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.writer.flush()
        metrics = {
            "beir_probe_pos_cosine_mean": pos_mean,
            "beir_probe_pos_cosine_std": float(np.std(pos_cos)),
            "beir_probe_pos_cosine_min": float(np.min(pos_cos)),
            "beir_probe_pos_cosine_max": float(np.max(pos_cos)),
            "beir_probe_neg_cosine_mean": neg_mean,
            "beir_probe_neg_cosine_std": float(np.std(neg_cos)),
            "beir_probe_neg_cosine_min": float(np.min(neg_cos)),
            "beir_probe_neg_cosine_max": float(np.max(neg_cos)),
            "beir_probe_gap_mean": gap_mean,
            "beir_probe_gap_std": float(np.std(gaps)),
            "beir_probe_gap_min": float(np.min(gaps)),
            "beir_probe_gap_max": float(np.max(gaps)),
            "beir_probe_rank_mean": float(rank_arr.mean()) if len(rank_arr) else 0.0,
            "beir_probe_rank_median": float(np.median(rank_arr)) if len(rank_arr) else 0.0,
            "beir_probe_rank_min": float(rank_arr.min()) if len(rank_arr) else 0.0,
            "beir_probe_rank_max": float(rank_arr.max()) if len(rank_arr) else 0.0,
            "beir_probe_interquery_rank_mean": float(interquery_rank_arr.mean()) if len(interquery_rank_arr) else 0.0,
            "beir_probe_interquery_rank_median": float(np.median(interquery_rank_arr)) if len(interquery_rank_arr) else 0.0,
            "beir_probe_interquery_rank_min": float(interquery_rank_arr.min()) if len(interquery_rank_arr) else 0.0,
            "beir_probe_interquery_rank_max": float(interquery_rank_arr.max()) if len(interquery_rank_arr) else 0.0,
            "beir_probe_mrr@10": mrr_at_10 / denom,
            "beir_probe_ndcg@10": ndcg_at_10 / denom,
            "beir_probe_recall@10": recall_at_10 / denom,
            "beir_interquery_mrr@10": interquery_mrr_at_10 / denom,
            "beir_interquery_ndcg@10": interquery_ndcg_at_10 / denom,
            "beir_interquery_recall@10": interquery_recall_at_10 / denom,
            # Backward-compatible alias used for selection. This is a sampled
            # cross-query candidate-pool metric, not full-corpus retrieval.
            "beir_proxy_mrr@10": interquery_mrr_at_10 / denom,
            "beir_proxy_ndcg@10": interquery_ndcg_at_10 / denom,
            "beir_proxy_recall@10": interquery_recall_at_10 / denom,
        }
        for dataset_name, summary in dataset_summaries.items():
            metric_prefix = "beir_probe_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)
            for key, value in summary.items():
                metrics[f"{metric_prefix}_{key}"] = float(value)
                self.writer.add_scalar(f"beir_probe_by_dataset/{dataset_name}/{key}", float(value), step)
        return metrics

    def _selection_score(self, metrics: dict[str, float]) -> float:
        if self.selection_metric in metrics:
            return float(metrics[self.selection_metric])
        if self.selection_metric == "gap" and "gap" in metrics:
            return float(metrics["gap"])
        fallback_keys = ("beir_probe_ndcg@10", "beir_probe_mrr@10", "beir_interquery_ndcg@10", "gap")
        for key in fallback_keys:
            if key in metrics:
                print(
                    f"[selection] metric {self.selection_metric!r} missing; falling back to {key}={metrics[key]:.6f}",
                    flush=True,
                )
                return float(metrics[key])
        return float("-inf")

    def train(self, num_epochs: int, val_every_n_steps: int, early_stopping_patience: int = 0) -> None:
        print(f"\n{'=' * 60}")
        target = f"{self.max_steps} steps" if self.max_steps > 0 else f"{num_epochs} epochs"
        print(f"Starting query distillation training for {target}")
        print(f"Output directory: {self.output_dir}")
        print(f"Validation every {val_every_n_steps} steps")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Selection metric: {self.selection_metric}")
        if early_stopping_patience > 0:
            print(f"Early stopping patience: {early_stopping_patience} validations")
        print(f"{'=' * 60}\n")
        val_metrics = self.validate(0)
        self.save_checkpoint(0, val_metrics, True)
        global_step = 0
        micro_step = 0
        best_score = self._selection_score(val_metrics)
        validations_without_improvement = 0
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, num_epochs + 1):
            self.projector.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            for batch in pbar:
                micro_step += 1
                loss, metrics = self.compute_loss({k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()})
                (loss / self.gradient_accumulation_steps).backward()
                should_step = micro_step % self.gradient_accumulation_steps == 0
                grad_norm = float("nan")
                if should_step:
                    global_step += 1
                    if self.max_steps > 0:
                        self._step_lr(global_step)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.projector.parameters(), max_norm=1.0).item()
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({
                    "loss": f"{loss.item():.6f}",
                    "q_mse": f"{metrics['query_mse']:.6f}",
                    "d_mse": f"{metrics['positive_mse']:.6f}",
                    "rank": f"{metrics['ranking_loss']:.6f}",
                    "gap": f"{metrics['gap']:.4f}",
                    "grad": f"{grad_norm:.4f}" if should_step else "accum",
                    "accum": f"{micro_step % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
                if should_step:
                    for key, value in metrics.items():
                        self.writer.add_scalar(f"train/{key}", value, global_step)
                    self.writer.add_scalar("train/loss_total", float(loss.item()), global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], global_step)
                    self.writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                    self.writer.add_scalar("train/micro_step", micro_step, global_step)
                    self.writer.add_scalar("train/gradient_accumulation_steps", self.gradient_accumulation_steps, global_step)
                    if "sources" in batch:
                        source_counts = collections.Counter(batch["sources"])
                        total_in_batch = max(1, len(batch["sources"]))
                        for source_name, count in source_counts.items():
                            self.writer.add_scalar(
                                f"train/source_fraction/{source_name}",
                                float(count) / float(total_in_batch),
                                global_step,
                            )
                if should_step and val_every_n_steps > 0 and global_step % val_every_n_steps == 0:
                    val_metrics = self.validate(global_step)
                    score = self._selection_score(val_metrics)
                    is_best = score > best_score
                    self.save_checkpoint(global_step, val_metrics, is_best=is_best)
                    if is_best:
                        best_score = score
                        validations_without_improvement = 0
                    else:
                        validations_without_improvement += 1
                        if early_stopping_patience > 0 and validations_without_improvement >= early_stopping_patience:
                            print(
                                f"Early stopping at step {global_step}: "
                                f"best_score={best_score:.6f}, current_score={score:.6f}"
                            )
                            self.writer.close()
                            self.train_dataset.close()
                            self.val_dataset.close()
                            return
                    self.projector.train()
                if self.max_steps > 0 and global_step >= self.max_steps:
                    self.writer.close()
                    self.train_dataset.close()
                    self.val_dataset.close()
                    return
        self.writer.close()
        self.train_dataset.close()
        self.val_dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train query/document BGE distillation projector")
    parser.add_argument("--oscar-model", required=True)
    parser.add_argument("--teacher-embeddings", nargs="+", required=True)
    parser.add_argument("--embed-dim", type=int, default=768)
    parser.add_argument("--pooler", default="flatten")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--projector-hidden-dim", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "none", "legacy"])
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--query-mse-weight", type=float, default=1.0)
    parser.add_argument("--doc-mse-weight", type=float, default=1.0)
    parser.add_argument("--negative-mse-weight", type=float, default=0.25)
    parser.add_argument("--ranking-weight", type=float, default=0.05)
    parser.add_argument("--infonce-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--beir-probe-config", default=None)
    parser.add_argument("--beir-probe-samples", type=int, default=20)
    parser.add_argument("--beir-probe-samples-per-dataset", type=int, default=None)
    parser.add_argument("--beir-probe-negatives", type=int, default=20)
    parser.add_argument("--beir-probe-batch-size", type=int, default=8)
    parser.add_argument("--selection-metric", default="beir_proxy_ndcg@10")
    parser.add_argument("--dataset-sampling-weights", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_lock = vars(args).copy()
    if args.dataset_sampling_weights:
        config_lock["dataset_sampling_weights"] = json.loads(args.dataset_sampling_weights)
    config_lock["teacher_embeddings"] = list(args.teacher_embeddings)
    config_lock["recipe"] = "query_distill"
    write_config_lock(config_lock, Path(args.output_dir).parent / "config.lock.yaml")
    trainer = QueryDistillationTrainer(
        oscar_model_name=args.oscar_model,
        teacher_embeddings_path=list(args.teacher_embeddings),
        embed_dim=args.embed_dim,
        pooler=args.pooler,
        num_layers=args.num_layers,
        dropout=args.dropout,
        projector_hidden_dim=args.projector_hidden_dim,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        scheduler=args.scheduler,
        val_split=args.val_split,
        query_mse_weight=args.query_mse_weight,
        doc_mse_weight=args.doc_mse_weight,
        negative_mse_weight=args.negative_mse_weight,
        ranking_weight=args.ranking_weight,
        infonce_weight=args.infonce_weight,
        temperature=args.temperature,
        margin=args.margin,
        beir_probe_config=args.beir_probe_config,
        beir_probe_samples=args.beir_probe_samples,
        beir_probe_samples_per_dataset=args.beir_probe_samples_per_dataset,
        beir_probe_negatives=args.beir_probe_negatives,
        beir_probe_batch_size=args.beir_probe_batch_size,
        selection_metric=args.selection_metric,
        dataset_sampling_weights=json.loads(args.dataset_sampling_weights) if args.dataset_sampling_weights else None,
        device=args.device,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    trainer.train(args.epochs, args.val_every, args.early_stopping_patience)


if __name__ == "__main__":
    main()
