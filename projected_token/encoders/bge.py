from __future__ import annotations

from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from projected_token.encoders import Encoder


class BGEEncoder(Encoder):
    """BGE encoder for dense retrieval embeddings."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda:0",
        trust_remote_code: bool = True,
        normalize_embeddings: bool = False,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        document_prefix: str = "",
    ) -> None:
        self._device = torch.device(device)
        self._normalize_embeddings = normalize_embeddings
        self._query_prefix = str(query_prefix)
        self._document_prefix = str(document_prefix)
        self._model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self._latent_dim = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        texts = questions if questions is not None else documents
        is_query = questions is not None
        valid_indices = [idx for idx, text in enumerate(texts) if isinstance(text, str) and text.strip()]
        if not valid_indices:
            return torch.zeros(len(texts), self._latent_dim, device=self._device)

        prefix = self._query_prefix if is_query else self._document_prefix
        valid_texts = [prefix + texts[idx] for idx in valid_indices]
        embeddings = self._model.encode(
            valid_texts,
            convert_to_tensor=True,
            device=self._device,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )

        result = torch.zeros(len(texts), self._latent_dim, device=self._device)
        for emb_idx, src_idx in enumerate(valid_indices):
            result[src_idx] = embeddings[emb_idx]
        return result

    def encode_batch(
        self,
        documents: List[str],
        questions: Optional[List[str]] = None,
    ) -> List[torch.Tensor]:
        encoded = self.encode(documents=documents, questions=questions)
        return [encoded[idx] for idx in range(encoded.shape[0])]

    @property
    def latent_dim(self) -> int:
        return self._latent_dim
