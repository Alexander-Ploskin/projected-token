
"""Индексация датасета с использованием энкодера в векторную базу данных."""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional, Any, Dict

import numpy as np
import torch
from tqdm import tqdm
import pyarrow.parquet as pq


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Загрузить датасет из JSONL."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def load_parquet(path: str) -> List[Dict[str, Any]]:
    """Загрузить датасет из Parquet."""
    table = pq.read_table(path)
    pydict = table.to_pydict()
    num_rows = table.num_rows
    return [{col: row[i] for col, row in pydict.items()} for i in range(num_rows)]


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Автоматически определить формат и загрузить датасет."""
    if path.endswith('.jsonl'):
        return load_jsonl(path)
    elif path.endswith('.parquet'):
        return load_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path}")


def create_faiss_index(embeddings: np.ndarray, metric: str = "ip") -> Any:
    """Создать FAISS индекс.
    
    Args:
        embeddings: Матрица эмбеддингов [n, dim]
        metric: "ip" (inner product) для косинусной схожести или "l2" для Euclidean
    
    Returns:
        FAISS индекс
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu or faiss-gpu required. Install: pip install faiss-cpu")

    dim = embeddings.shape[1]
    
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    
    index.add(embeddings)
    return index


def create_sklearn_index(embeddings: np.ndarray, metric: str = "cosine") -> Any:
    """Создать sklearn NearestNeighbors индекс.
    
    Args:
        embeddings: Матрица эмбеддингов [n, dim]
        metric: "cosine" или "euclidean"
    
    Returns:
        NearestNeighbors объект
    """
    from sklearn.neighbors import NearestNeighbors
    
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    if metric == "cosine":
        # Для косинусной схожести нормализуем вектора
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
    
    nn = NearestNeighbors(n_neighbors=embeddings.shape[0], metric=metric)
    nn.fit(embeddings)
    return nn


def main():
    parser = argparse.ArgumentParser(
        description="Индексация датасета с использованием энкодера"
    )
    
    # Аргументы датасета
    parser.add_argument(
        "--input-path",
        required=True,
        help="Путь к датасету (parquet или jsonl)"
    )
    parser.add_argument(
        "--text-col",
        default="s_wiki_content",
        help="Название колонки с текстом документа"
    )
    parser.add_argument(
        "--id-col",
        default="id",
        help="Название колонки с ID документа (опционально)"
    )
    
    # Аргументы энкодера
    parser.add_argument(
        "--encoder",
        required=True,
        choices=["oscar", "salesforce"],
        help="Тип энкодера"
    )
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Путь к модели или HuggingFace model_id"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Устройство для инференса"
    )
    parser.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "first", "last", "max", "mean_max"],
        help="Стратегия агрегации mem-токенов: mean, first, last, max, mean_max"
    )
    
    # Аргументы индекса
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Директория для сохранения индекса"
    )
    parser.add_argument(
        "--index-type",
        default="faiss",
        choices=["faiss", "sklearn"],
        help="Тип векторного индекса"
    )
    parser.add_argument(
        "--metric",
        default="ip",
        choices=["ip", "l2", "cosine"],
        help="Метрика схожести (ip=cosine после нормировки, l2=euclidean)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Размер батча для кодирования"
    )
    
    args = parser.parse_args()
    
    print(f"Загрузка датасета из {args.input_path}...")
    dataset = load_dataset(args.input_path)
    print(f"Загружено {len(dataset)} документов")
    
    # Извлекаем тексты
    texts = [item.get(args.text_col, "") for item in dataset]
    valid_indices = [i for i, t in enumerate(texts) if t and isinstance(t, str)]
    valid_texts = [texts[i] for i in valid_indices]
    
    print(f"Валидных документов: {len(valid_texts)}")
    
    # Инициализируем энкодер
    print(f"Инициализация энкодера {args.encoder}...")
    
    if args.encoder == "oscar":
        from evaluation.encoders import OscarEncoder
        encoder = OscarEncoder(
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            aggregation=args.aggregation,
        )
    elif args.encoder == "salesforce":
        from evaluation.encoders import SalesforceEncoder
        encoder = SalesforceEncoder(
            model_name_or_path=args.model_name_or_path,
            device=args.device,
        )
    else:
        raise ValueError(f"Unknown encoder: {args.encoder}")
    
    print(f"Размерность латентного пространства: {encoder.latent_dim}")
    if hasattr(encoder, 'aggregation'):
        print(f"Стратегия агрегации: {encoder.aggregation}")
    
    # Кодируем документы батчами
    print(f"Кодирование документов (batch_size={args.batch_size})...")
    all_embeddings = []
    
    for i in tqdm(range(0, len(valid_texts), args.batch_size)):
        batch_texts = valid_texts[i:i + args.batch_size]
        batch_embeddings = encoder.encode(batch_texts)
        all_embeddings.append(batch_embeddings.cpu().numpy())
    
    embeddings = np.vstack(all_embeddings)
    print(f"Получено {embeddings.shape[0]} эмбеддингов размера {embeddings.shape[1]}")
    
    # Создаём индекс
    print(f"Создание {args.index_type} индекса (metric={args.metric})...")
    
    if args.index_type == "faiss":
        index = create_faiss_index(embeddings, metric=args.metric)
    else:
        index = create_sklearn_index(embeddings, metric=args.metric)
    
    # Сохраняем
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем индекс
    index_path = output_dir / "index"
    if args.index_type == "faiss":
        import faiss
        faiss.write_index(index, str(index_path))
    else:
        with open(index_path, 'wb') as f:
            pickle.dump(index, f)
    print(f"Индекс сохранён: {index_path}")
    
    # Сохраняем метаданные
    metadata = {
        "encoder": args.encoder,
        "model_name_or_path": args.model_name_or_path,
        "device": args.device,
        "latent_dim": encoder.latent_dim,
        "aggregation": args.aggregation,
        "num_documents": len(valid_texts),
        "index_type": args.index_type,
        "metric": args.metric,
        "text_col": args.text_col,
        "id_col": args.id_col,
        "valid_indices": valid_indices,
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Метаданные сохранены: {metadata_path}")
    
    # Сохраняем исходные данные (только валидные)
    if args.id_col and args.id_col != "id":
        id_mapping = {i: dataset[valid_indices[i]].get(args.id_col, i) for i in range(len(valid_indices))}
    else:
        id_mapping = {i: i for i in range(len(valid_indices))}
    
    ids_path = output_dir / "ids.json"
    with open(ids_path, 'w') as f:
        json.dump(id_mapping, f)
    
    print("\n=== Индексация завершена ===")
    print(f"Документов: {len(valid_texts)}")
    print(f"Размерность: {encoder.latent_dim}")
    print(f"Индекс: {args.index_type}")
    print(f"Выходная директория: {output_dir}")


if __name__ == "__main__":
    main()