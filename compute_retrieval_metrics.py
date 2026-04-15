#!/usr/bin/env python3
"""Вычисление метрик поиска для PopQA датасета."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Set

import numpy as np
import pyarrow.parquet as pq
import faiss
from tqdm import tqdm
import torch


def load_parquet(path: str) -> List[Dict[str, Any]]:
    """Загрузить датасет из Parquet."""
    table = pq.read_table(path)
    pydict = table.to_pydict()
    num_rows = table.num_rows
    return [{col: row[i] for col, row in pydict.items()} for i in range(num_rows)]


def check_relevance(document: str, answers: List[str]) -> bool:
    """Проверить, содержит ли документ хотя бы один ответ."""
    if not document:
        return False
    
    doc_lower = document.lower()
    for answer in answers:
        if answer.lower() in doc_lower:
            return True
    return False


def compute_recall_at_k(rel_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
    """Compute Recall@k."""
    retrieved_k = set(retrieved_docs[:k])
    if len(rel_docs) == 0:
        return 0.0
    return len(rel_docs & retrieved_k) / len(rel_docs)


def compute_mrr(rel_docs: Set[int], retrieved_docs: List[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved_docs, 1):
        if doc_id in rel_docs:
            return 1.0 / i
    return 0.0


def compute_ndcg_at_k(rel_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
    """Compute NDCG@k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k], 1):
        if doc_id in rel_docs:
            dcg += 1.0 / np.log2(i + 1)
    
    num_rel = min(len(rel_docs), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_rel + 1))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_precision_at_k(rel_docs: Set[int], retrieved_docs: List[int], k: int) -> float:
    """Compute Precision@k."""
    retrieved_k = set(retrieved_docs[:k])
    if k == 0:
        return 0.0
    return len(rel_docs & retrieved_k) / k


def main():
    parser = argparse.ArgumentParser(description="Вычисление метрик поиска для PopQA")
    
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--encoder", required=True, choices=["oscar", "salesforce"])
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                        help="Значения k для метрик")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-path", default="retrieval_metrics.json")
    parser.add_argument("--save-retrieval-details", default="/data/popqa_index_new/retrieval_details.json",
                        help="Путь для сохранения деталей поиска по всем вопросам")
    
    args = parser.parse_args()
    
    # Загрузка датасета
    print(f"Загрузка датасета из {args.dataset_path}...")
    dataset = load_parquet(args.dataset_path)
    print(f"Загружено {len(dataset)} записей")
    
    # Загрузка метаданных
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    
    valid_indices = metadata["valid_indices"]
    text_col = metadata["text_col"]
    aggregation = metadata.get("aggregation", "mean")
    
    print(f"Valid indices: {len(valid_indices)}")
    print(f"Text column: {text_col}")
    print(f"Aggregation: {aggregation}")
    
    # Загрузка FAISS индекса
    print(f"Загрузка индекса...")
    index = faiss.read_index(args.index_path)
    print(f"Index size: {index.ntotal}")
    
    # Проверка: сопоставление id документа в датасете с индексами
    # id_to_idx маппит original_id (из датасета) -> index position
    id_to_idx = {orig_id: i for i, orig_id in enumerate(valid_indices)}
    print(f"ID mapping size: {len(id_to_idx)}")
    
    # DEBUG: проверим несколько id
    print(f"\nSample valid_indices: {valid_indices[:5]}")
    print(f"Sample dataset ids: {[dataset[i].get('id') for i in range(5) if i in valid_indices]}")
    
    # Создаём mapping: вопрос -> данные
    # В PopQA каждый вопрос соответствует своей записи,
    # релевантный документ - это s_wiki_content с тем же индексом
    questions_data = []
    
    print("\nПодготовка данных вопросов...")
    
    # Создаём mapping: original_idx -> index_in_vector
    orig_idx_to_vector_idx = {orig: i for i, orig in enumerate(valid_indices)}
    
    for q_idx, item in enumerate(tqdm(dataset, desc="Подготовка вопросов")):
        question = item.get("question", "")
        
        # Релевантный документ - это документ с тем же индексом в датасете
        # Если индекс в valid_indices - это релевантный документ
        relevant_docs = set()
        if q_idx in orig_idx_to_vector_idx:
            relevant_docs.add(orig_idx_to_vector_idx[q_idx])
        
        questions_data.append({
            "question": question,
            "relevant_docs": relevant_docs,
            "q_idx": q_idx,
            "subj": item.get("subj"),
            "obj": item.get("obj"),
        })
    
    # Статистика по релевантности
    num_with_relevant = sum(1 for q in questions_data if len(q["relevant_docs"]) > 0)
    print(f"\nВопросов с релевантными документами: {num_with_relevant} / {len(questions_data)}")
    
    # Инициализация энкодера
    print(f"\nИнициализация энкодера {args.encoder}...")
    
    if args.encoder == "oscar":
        from evaluation.encoders import OscarEncoder
        encoder = OscarEncoder(
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            aggregation=aggregation,
        )
        print(f"Стратегия агрегации: {encoder.aggregation}")
    elif args.encoder == "salesforce":
        from evaluation.encoders import SalesforceEncoder
        encoder = SalesforceEncoder(
            model_name_or_path=args.model_name_or_path,
            device=args.device,
        )
        print(f"Размерность эмбеддинга: {encoder.latent_dim}")
    
    # Кодирование вопросов и поиск
    print("Кодирование вопросов и поиск...")
    
    all_queries = [q["question"] for q in questions_data]
    retrieved_results = []
    
    for i in tqdm(range(0, len(all_queries), args.batch_size), desc="Поиск"):
        batch_questions = all_queries[i:i + args.batch_size]
        
        with torch.inference_mode():
            if args.encoder == "salesforce":
                query_embeddings = encoder.encode(documents=[""] * len(batch_questions), questions=batch_questions)
            else:
                query_embeddings = encoder.encode(batch_questions)
        
        query_embeddings = query_embeddings.cpu().numpy().astype(np.float32)
        
        # Нормализация
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        query_embeddings = query_embeddings / norms
        
        max_k = max(args.top_k)
        distances, indices = index.search(query_embeddings, max_k)
        
        for j in range(len(batch_questions)):
            retrieved_results.append({
                "indices": indices[j].tolist(),
                "distances": distances[j].tolist(),
            })
    
    # Сохраняем информацию по всем вопросам
    print("\nСохранение деталей поиска...")
    all_retrieval_details = []
    for i in range(len(questions_data)):
        q = questions_data[i]
        r = retrieved_results[i]
        all_retrieval_details.append({
            "q_idx": i,
            "question": q["question"],
            "subj": q.get("subj"),
            "obj": q.get("obj"),
            "num_relevant_docs": len(q["relevant_docs"]),
            "relevant_doc_indices": list(q["relevant_docs"]),
            "retrieved_top20": r["indices"][:20],
            "retrieved_scores": r["distances"][:20],
            "hit_in_top1": 1 if r["indices"][0] in q["relevant_docs"] else 0,
            "hit_in_top5": 1 if len(set(r["indices"][:5]) & q["relevant_docs"]) > 0 else 0,
            "hit_in_top10": 1 if len(set(r["indices"][:10]) & q["relevant_docs"]) > 0 else 0,
        })
    
    with open(args.save_retrieval_details, "w") as f:
        json.dump(all_retrieval_details, f, indent=2)
    print(f"Сохранено: {args.save_retrieval_details}")
    
    # Вычисление метрик
    print("\nВычисление метрик...")
    
    metrics = {k: [] for k in [f"recall@{k}" for k in args.top_k] + 
                          [f"precision@{k}" for k in args.top_k] +
                          [f"ndcg@{k}" for k in args.top_k] + 
                          ["mrr"]}
    
    for q_data, r_data in zip(questions_data, retrieved_results):
        rel_docs = q_data["relevant_docs"]
        retrieved = r_data["indices"]
        
        if len(rel_docs) == 0:
            continue
        
        for k in args.top_k:
            metrics[f"recall@{k}"].append(compute_recall_at_k(rel_docs, retrieved, k))
            metrics[f"precision@{k}"].append(compute_precision_at_k(rel_docs, retrieved, k))
            metrics[f"ndcg@{k}"].append(compute_ndcg_at_k(rel_docs, retrieved, k))
        
        metrics["mrr"].append(compute_mrr(rel_docs, retrieved))
    
    # Агрегация
    results = {}
    for metric_name, values in metrics.items():
        if values:
            results[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "count": len(values)
            }
        else:
            results[metric_name] = {"mean": 0.0, "std": 0.0, "count": 0}
    
    # Сохранение
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n=== Результаты ===")
    for metric_name, values in sorted(results.items()):
        print(f"{metric_name}: {values['mean']:.4f} ± {values['std']:.4f} (n={values['count']})")
    
    print(f"\nСохранено: {output_path}")


if __name__ == "__main__":
    main()