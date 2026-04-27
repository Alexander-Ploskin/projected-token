#!/usr/bin/env python3
"""Download MS MARCO v1 corpus and queries."""

import json
import os
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Installing datasets...")
    import subprocess
    subprocess.run(['pip', 'install', 'datasets', 'sentencepiece'], check=True)
    from datasets import load_dataset

# Download corpus
print("Loading MS MARCO corpus...")
corpus_ds = load_dataset("microsoft/ms_marco", "v1.1", trust_remote_code=True)

print("Corpus keys:", corpus_ds.keys())
print("Corpus info:", corpus_ds)

# Save a small sample first to check format
output_dir = Path("/app/last_projected-token/data/msmarco_v1")
output_dir.mkdir(parents=True, exist_ok=True)

# Try to get corpus
if 'corpus' in corpus_ds:
    corpus = corpus_ds['corpus']
    print(f"Corpus size: {len(corpus)}")
    print("Sample:", corpus[0])
else:
    print("No corpus in dataset, checking other splits...")
    for split in corpus_ds.keys():
        print(f"{split}: {len(corpus_ds[split])}")
        if len(corpus_ds[split]) > 0:
            print("Sample:", corpus_ds[split][0])