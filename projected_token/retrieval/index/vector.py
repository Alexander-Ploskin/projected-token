from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.where(norms == 0, 1, norms)


def create_faiss_index(embeddings: np.ndarray, metric: str = "ip") -> Any:
    import faiss

    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if metric == "ip" else faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index: Any, path: str | Path) -> None:
    import faiss

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: str | Path) -> Any:
    import faiss

    return faiss.read_index(str(path))
