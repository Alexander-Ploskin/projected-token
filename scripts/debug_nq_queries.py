import json
import numpy as np
import faiss
from pathlib import Path
import torch
from projected_token.retrieval.beir import _load_beir_dataset, _encode_query_batches
from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import l2_normalize
from transformers import AutoModel
from projected_token.oscar_runtime import configure_oscar_component_devices

print("Loading queries...")
_, _, query_ids, query_texts, relevant_map = _load_beir_dataset(Path("/data/beir/nq"), split="test")

print("Loading OSCAR...")
oscar_model = AutoModel.from_pretrained(
    "naver/oscar-qwen2-7B",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda:0",
).eval()

encoder_cfg = {
    "name": "oscar_projector",
    "kwargs": {
        "oscar_model_name": "naver/oscar-qwen2-7B",
        "projector_path": "artifacts/query_distill_runs/e9-flatten-kl-4096/checkpoints/checkpoint_step_6000.pt",
        "device": "cuda:0",
        "embed_dim": 768,
        "pooler": "flatten",
        "num_layers": 2,
        "dropout": 0.0,
        "oscar_model_instance": oscar_model,
    },
}
encoder = build_encoder(encoder_cfg)

print("Encoding first 5 queries...")
q_texts = query_texts[:5]
q_emb = _encode_query_batches(encoder, q_texts, batch_size=5)
q_emb = l2_normalize(q_emb)

print(q_emb.shape)
print(q_emb[:5, :5])
print(np.linalg.norm(q_emb, axis=1))

print("Loading shard 0...")
shards_dir = Path("artifacts/results/retrieval/beir_nq_e9_6000_shards")
emb = np.load(shards_dir / "embeddings_0.npy")

print("Computing dot products...")
scores = np.dot(q_emb, emb.T)
print("Max scores for each query:")
print(np.max(scores, axis=1))
print("Min scores for each query:")
print(np.min(scores, axis=1))
print("Mean scores for each query:")
print(np.mean(scores, axis=1))
