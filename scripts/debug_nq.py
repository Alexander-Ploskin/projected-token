import json
import numpy as np
import faiss
from pathlib import Path

out_dir = Path("artifacts/results/retrieval/beir_nq_e9_6000_fixed")
shards_dir = Path("artifacts/results/retrieval/beir_nq_e9_6000_shards")

print("Loading shard 0...")
emb = np.load(shards_dir / "embeddings_0.npy")
with open(shards_dir / "ids_0.json", "r") as f:
    ids = json.load(f)

print(emb.shape)
print("First 5 embeddings:")
print(emb[:5, :5])

print("Norms:")
print(np.linalg.norm(emb[:5], axis=1))
