import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Any
from datasets import load_dataset


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
        fallback_negative_strategy: str = "first_non_positive",
    ):
        """Инициализация датасета.

        Args:
            data_path: путь к локальным данным
            config: имя конфига (triplet, triplet-hard, etc.)
            split: 'train' или 'validation'
            max_samples: ограничение количества samples (для отладки)
        """
        self.split = split
        self.fallback_negative_strategy = fallback_negative_strategy

        print(f"Loading MS MARCO dataset: {data_path}, config={config}, split={split}...")
        self.dataset = load_dataset(
            data_path,
            name=config,
            split=split,
        )

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples")

    def _extract_from_passages(self, item: dict[str, Any]) -> dict[str, str]:
        passages = item.get("passages")
        if not isinstance(passages, dict):
            raise ValueError("Expected dict field 'passages' for MS MARCO raw sample.")

        texts = passages.get("passage_text", [])
        selected = passages.get("is_selected", [])
        if not isinstance(texts, list) or not texts:
            raise ValueError("Raw MS MARCO sample has no non-empty passages.")

        positive_idx = None
        for idx, flag in enumerate(selected):
            if idx < len(texts) and int(flag) > 0 and isinstance(texts[idx], str) and texts[idx].strip():
                positive_idx = idx
                break
        if positive_idx is None:
            for idx, text in enumerate(texts):
                if isinstance(text, str) and text.strip():
                    positive_idx = idx
                    break
        if positive_idx is None:
            raise ValueError("Could not identify positive passage in raw MS MARCO sample.")

        positive = texts[positive_idx].strip()
        negative_candidates = [
            text.strip()
            for idx, text in enumerate(texts)
            if idx != positive_idx and isinstance(text, str) and text.strip()
        ]
        if negative_candidates:
            if self.fallback_negative_strategy == "random":
                # PyTorch DataLoader worker RNG controls randomness here if enabled.
                negative = negative_candidates[torch.randint(0, len(negative_candidates), (1,)).item()]
            else:
                negative = negative_candidates[0]
        else:
            negative = positive

        query = item.get("query") or item.get("question") or ""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Could not identify query text in raw MS MARCO sample.")
        return {
            "query": query.strip(),
            "positive": positive,
            "negative": negative,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # sentence-transformers triplets and preprocessed json datasets
        if {"query", "positive", "negative"}.issubset(item.keys()):
            return {
                "query": item["query"],
                "positive": item["positive"],
                "negative": item["negative"],
            }
        # microsoft/ms_marco v1.x raw format with passages/is_selected
        if "passages" in item:
            return self._extract_from_passages(item)
        raise ValueError(
            "Unsupported dataset row format. "
            "Expected keys [query, positive, negative] or raw MS MARCO passages."
        )


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
    dataset_path: str = "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
    dataset_config: str = "triplet",
    dataset_split: str = "train",
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    num_workers: int = 0,
    device: str = "cuda:0",
    val_split: float = 0.1,
    fallback_negative_strategy: str = "first_non_positive",
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
    full_dataset = MSMarcoDataset(
        data_path=dataset_path,
        config=dataset_config,
        split=dataset_split,
        max_samples=max_train_samples,
        fallback_negative_strategy=fallback_negative_strategy,
    )
    
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