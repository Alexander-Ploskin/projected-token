# Projected Token



### Evaluate KILT with Multiple Retrievers

Unified OpenQA evaluation now supports:

- `dense_faiss` (SFR/BGE/other dense HF encoders),
- `bm25` (bm25s index),
- `splade_csr` (SPLADE v3 sparse CSR index).

#### Compatibility Requirements

Your index directory must include:

- `metadata.json` with indexing metadata (or equivalent fields),
- backend-specific artifacts:
  - dense: `index.faiss` or `index_shards_manifest.json` + shard `.faiss`,
  - bm25: `bm25s_index/`, `chunks.jsonl`, `bm25s_doc_order.json`,
  - splade: `corpus_csr.npz`, `doc_row_ids.npy`,
- for dense/splade: IDs aligned with KILT passage row IDs in the text cache parquet.

If the method used a different query formatting or encoder, pass the matching options at eval time (model, prompt template, max length, normalization behavior).

#### How a Precomputed KILT Dense Index Should Look

A ready-to-evaluate index directory should look like one of these layouts.

Single-file FAISS layout:

```text
your_kilt_dense_index/
├── metadata.json
└── index.faiss
```

Sharded FAISS layout:

```text
your_kilt_dense_index/
├── metadata.json
├── index_shards_manifest.json
└── index_shards/
    ├── s0_0_8000.faiss
    ├── s0_8000_16000.faiss
    └── ...
```

Expected manifest fields (`index_shards_manifest.json`):

- `index_type` (for example `flat_fp16`)
- `dim` (embedding dimension)
- `num_shards`
- `entries[]` with:
  - `path` (absolute path to shard `.faiss`)
  - `start`, `end` (global vector-id interval)
  - `count` (usually `end - start`)
  - `shard_id`

Important consistency checks:

- `metadata.json` should reference the same embedding family and index settings used at build time.
- FAISS IDs must be row-aligned with the KILT text cache parquet used in evaluation.
- If `metadata.json` contains `prepare_cache_path`, it should point to a valid parquet; otherwise pass `--text-cache-path` explicitly during eval.

#### A) Index Already Exists (evaluate only)

Example: evaluate PopQA for an existing dense BGE index:

```bash
python -m projected_token retrieval eval-kilt-openqa \
  --dataset popqa \
  --index-dir /path/to/your_kilt_dense_index \
  --output-path artifacts/results/retrieval/bge_popqa_metrics.json \
  --retrieval-backend dense_faiss \
  --encoder-type bge \
  --model-name-or-path BAAI/bge-large-en-v1.5 \
  --disable-query-prefix \
  --pooling cls \
  --popqa-split test \
  --text-cache-path /path/to/kilt_prepare.parquet \
  --top-k 1,5,10 \
  --search-workers 24 \
  --search-query-batch-size 128
```

Example: evaluate BM25 index:

```bash
python -m projected_token retrieval eval-kilt-openqa \
  --dataset popqa \
  --index-dir artifacts/indexes/kilt_bm25 \
  --output-path artifacts/results/retrieval/kilt_bm25_popqa_metrics.json \
  --retrieval-backend bm25 \
  --encoder-type bm25 \
  --popqa-split test \
  --top-k 1,5,10
```

Example: evaluate SPLADE v3 CSR index:

```bash
python -m projected_token retrieval eval-kilt-openqa \
  --dataset popqa \
  --index-dir artifacts/indexes/kilt_splade_v3 \
  --output-path artifacts/results/retrieval/kilt_splade_popqa_metrics.json \
  --retrieval-backend splade_csr \
  --encoder-type splade_v3 \
  --model-name-or-path naver/splade-v3 \
  --disable-query-prefix \
  --popqa-split test \
  --text-cache-path /path/to/kilt_prepare.parquet \
  --splade-agg max \
  --splade-top-n-terms 128 \
  --top-k 1,5,10
```

#### B) Index Does Not Exist Yet (build then evaluate)

1) Build a KILT index (examples):

```bash
# Dense (SFR template)
python -m projected_token retrieval index-kilt-sfr \
  --config configs/retrieval/kilt_sfr.yaml

# BM25
python -m projected_token retrieval index-kilt-bm25 \
  --config configs/retrieval/kilt_bm25.yaml

# SPLADE v3 (CSR)
python -m projected_token retrieval index-kilt-splade \
  --config configs/retrieval/kilt_splade_v3.yaml
```

2) Evaluate on a target dataset:

```bash
python -m projected_token retrieval eval-kilt-openqa \
  --config configs/retrieval/eval_kilt_sfr_hotpot_distractor.yaml
```

Legacy command `eval-kilt-sfr-openqa` is still available and forwards to the unified evaluator.

#### Fast Re-scoring Without Re-running Search

If you already have saved top-k retrieval results (`*_topk.jsonl`), recompute metrics only:

```bash
python -m projected_token retrieval eval-kilt-openqa \
  --config configs/retrieval/eval_kilt_sfr_popqa.yaml \
  --search-results-jsonl-path artifacts/results/retrieval/kilt_sfr_popqa_metrics_topk.jsonl \
  --no-use-search-cache \
  --no-save-search-cache \
  --no-save-search-results-jsonl
```

This mode skips dense search and only re-runs relevance judging + metric aggregation.

## Training

Train a projector from a recipe:

```bash
python -m projected_token train --config configs/training/projector_mlp.yaml
python -m projected_token train --config configs/training/projector_lora.yaml
python -m projected_token train --config configs/training/projector_full.yaml
python -m projected_token train --config configs/training/projector_flat.yaml
python -m projected_token train --config configs/training/projector_distill.yaml
```

The unified `train` command dispatches to reusable trainers or internalized legacy recipes depending on the `recipe` field.

## Data Preparation

PopQA:

```bash
python -m projected_token data prepare-popqa \
  --output-dir data/eval/popqa \
  --workers 8 \
  --batch-size 50
```

MS MARCO and derived finetuning sets:

```bash
python -m projected_token data prepare-msmarco --help
python -m projected_token data prepare-msmarco-v2 --help
python -m projected_token data mixed-dataset --help
python -m projected_token data hard-negatives --help
python -m projected_token data teacher-embeddings --help
```

## Data And Artifacts

The refactor keeps all copied data and results in normalized locations:

- `data/msmarco/` - MS MARCO JSON files.
- `data/finetune/` - mixed training datasets and hard negatives.
- `data/eval/` - evaluation datasets.
- `artifacts/results/` - generated outputs, metrics, old result JSONs, visualizations, logs.
- `artifacts/indexes/` - FAISS indexes and metadata.
- `artifacts/checkpoints/` - projector checkpoints and adapters.
- `artifacts/teacher_embeddings/` - teacher embedding `.h5` files.
- `artifacts/manifest.json` - path, size, and SHA256 for copied data/artifact files.

## Migration From Old Scripts

| Old root script | New command |
| --- | --- |
| `evaluation/cli.py paraphrase` | `python -m projected_token generate --task paraphrase` |
| `evaluation/cli.py qa` | `python -m projected_token generate --task qa` |
| `evaluation/cli.py eval-gpt` | `python -m projected_token evaluate --task paraphrase` |
| `evaluation/cli.py eval-qa` | `python -m projected_token evaluate --task qa` |
| `scripts/train_projector.py ...` | `python -m projected_token train --config ...` |
| `index_dataset.py` | `python -m projected_token retrieval build-index --config ...` |
| `compute_retrieval_metrics.py` | `python -m projected_token retrieval evaluate --config ...` |
| `data/eval/cli.py prepare-popqa` | `python -m projected_token data prepare-popqa` |

## Docker

```bash
docker build -t projected-token .
docker run --gpus all --rm -it \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/artifacts:/workspace/artifacts" \
  -v "$HOME/.cache/huggingface:/workspace/data/hf_cache" \
  projected-token --help
```

## Notes

- Run commands from the repository root.
- Keep `PYTHONPATH=.` when using source checkout without installation.
- LLM judge commands expect an OpenAI-compatible endpoint.
- AlignScore requires an external checkpoint path in the metric config.
- Some internalized data/retrieval recipes preserve the original argparse options; pass extra options after the subcommand.
