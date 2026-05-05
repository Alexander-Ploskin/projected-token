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
    dataset_path: str = "/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3",
    dataset_config: str = "triplet",
    dataset_split: str = "train",
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
    full_dataset = MSMarcoDataset(
        data_path=dataset_path,
        config=dataset_config,
        split=dataset_split,
        max_samples=max_train_samples,
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