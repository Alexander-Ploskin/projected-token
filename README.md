# Projected Token

Clean research repository for context-compression experiments in LLM/RAG systems. The code is organized around one package, `projected_token`, and one public CLI. It includes generation, evaluation, retrieval, data preparation, and projector-training workflows for OSCAR, xRAG, PISCO/COCOM, RAG, and no-context baselines.

## Repository Layout

```text
projected-token/
├── projected_token/             # Python package and all public code
│   ├── cli/                     # Unified Click CLI
│   ├── models/                  # OSCAR, PISCO, xRAG, RAG, SimpleLLM
│   ├── metrics/                 # GPT judge, QA judge, AlignScore, simple metrics
│   ├── generation.py            # Generation pipeline
│   ├── evaluation.py            # Evaluation pipeline
│   ├── pipeline.py              # End-to-end runs
│   ├── encoders/                # OSCAR/SFR/projector encoders
│   ├── training/                # Training recipes and reusable trainers
│   ├── retrieval/               # Indexing, retrieval tasks, ranking metrics
│   ├── datasets/                # Dataset interfaces and PopQA loader
│   └── data/                    # Data-preparation modules
├── configs/
│   ├── experiments/             # QA/paraphrase generation configs
│   ├── training/                # Projector training recipes
│   ├── retrieval/               # Index/eval configs
│   ├── data/                    # Data preparation configs
│   └── schemas/                 # Documented YAML schemas
├── data/                        # Raw and prepared datasets
├── artifacts/                   # Results, indexes, checkpoints, manifest
├── docs/                        # Research notes copied from the original project
├── Dockerfile
└── pyproject.toml
```

The old root-level scripts from `last_projected-token` were internalized under domain modules. They are no longer public entrypoints; use `python -m projected_token ...` instead.

## Installation

```bash
cd final/projected-token
poetry install
export PYTHONPATH=.
```

Or run directly from the source tree:

```bash
python -m projected_token --help
```

The project uses GPU-heavy dependencies (`torch`, `transformers`, `faiss`, `sentence-transformers`) and may download HuggingFace models at runtime.

## CLI

```bash
python -m projected_token --help
```

Top-level commands:

- `generate` - generate paraphrases or QA answers.
- `evaluate` - evaluate generated JSONL files.
- `run` - run generation followed by evaluation.
- `run-all` - run all matching experiment configs.
- `train` - train a projector from a YAML recipe.
- `train-matrix` - run a matrix of training recipes.
- `retrieval build-index` - build a FAISS index.
- `retrieval evaluate` - compute retrieval metrics.
- `retrieval evaluate-beir` - evaluate checkpoints on BEIR-format datasets.
- `data ...` - prepare datasets and teacher embeddings.

## Configs

Configs are split by workflow:

- `configs/experiments/` for model generation and QA/paraphrase experiments.
- `configs/training/` for projector recipes: `mlp`, `lora`, `full`, `flat`, `distill`, `hotpot_distill`.
- `configs/retrieval/` for indexing and retrieval evaluation.
- `configs/data/` for data preparation.
- `configs/schemas/` documents the accepted YAML fields.

Class paths use the new package namespace, for example:

```yaml
dataset:
  class: projected_token.datasets.PopqaDataset

model:
  class: projected_token.models.OscarModel
  kwargs:
    model_name_or_path: naver/oscar-qwen2-7B
    device: cuda:0
    trust_remote_code: true
```

## Generation

Paraphrase generation:

```bash
python -m projected_token generate \
  --task paraphrase \
  --config configs/experiments/oscar_7b_paraphrase.yaml \
  --input data/eval/popqa/test.jsonl \
  --output artifacts/results/paraphrase/oscar_7b.jsonl \
  --text-col s_wiki_content \
  --batch-size 4
```

QA generation:

```bash
python -m projected_token generate \
  --task qa \
  --config configs/experiments/oscar_7b_qa.yaml \
  --input data/eval/popqa/test.jsonl \
  --output artifacts/results/qa/oscar_7b.jsonl \
  --text-col s_wiki_content \
  --question-col question \
  --batch-size 4
```

## Evaluation

Paraphrase evaluation with simple metrics and optional GPT judge:

```bash
python -m projected_token evaluate \
  --task paraphrase \
  --input artifacts/results/paraphrase/oscar_7b.jsonl \
  --output artifacts/results/paraphrase/oscar_7b_metrics.json \
  --config configs/experiments/oscar_7b_paraphrase.yaml \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --model Qwen/Qwen3.5-27B
```

QA evaluation:

```bash
python -m projected_token evaluate \
  --task qa \
  --input artifacts/results/qa/oscar_7b.jsonl \
  --output artifacts/results/qa/oscar_7b_metrics.json \
  --config configs/experiments/oscar_7b_qa.yaml \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --model Qwen/Qwen3.5-27B
```

End-to-end run:

```bash
python -m projected_token run \
  --task qa \
  --config configs/experiments/oscar_7b_qa.yaml \
  --input data/eval/popqa/test.parquet \
  --output-dir artifacts/results/qa/oscar_7b \
  --batch-size 4
```

## Retrieval

Build a vector index:

```bash
python -m projected_token retrieval build-index \
  --config configs/retrieval/popqa_oscar_projector.yaml
```

Evaluate retrieval:

```bash
python -m projected_token retrieval evaluate \
  --config configs/retrieval/popqa_oscar_projector.yaml
```

Evaluate BEIR-3:

```bash
python -m projected_token retrieval evaluate-beir \
  --config configs/retrieval/beir3_oscar_projector.yaml
```

Ranking metrics include `recall@k`, `precision@k`, `ndcg@k`, `mrr`, and capped `mrr@k` (including `mrr@10`).

## Training

Train a projector from a recipe:

```bash
python -m projected_token train --config configs/training/projector_mlp.yaml
python -m projected_token train --config configs/training/projector_lora.yaml
python -m projected_token train --config configs/training/projector_full.yaml
python -m projected_token train --config configs/training/projector_flat.yaml
python -m projected_token train --config configs/training/projector_distill.yaml
```

Run experiment matrices:

```bash
python -m projected_token train-matrix --config configs/training/matrix_contrastive.yaml
python -m projected_token train-matrix --config configs/training/matrix_distill.yaml
python -m projected_token train-matrix --config configs/training/matrix_two_stage.yaml
python -m projected_token train-matrix --config configs/training/matrix_lora_unfreeze.yaml
```

The unified `train` command dispatches to reusable trainers or internalized legacy recipes depending on the `recipe` field.
Each training run is persisted under `artifacts/runs/<timestamp>_<name>/` with:
- `config.lock.yaml`
- `checkpoints/`
- `logs/tensorboard/`
- `metrics/*.json` + `metrics/*.csv`
- `plots/*.png`

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
  -v "/mnt/raid/a-ploskin/projected-token-hf-home:/workspace/data/hf_cache" \
  -e HF_HOME=/workspace/data/hf_cache \
  projected-token --help
```

## Notes

- Run commands from the repository root.
- Keep `PYTHONPATH=.` when using source checkout without installation.
- LLM judge commands expect an OpenAI-compatible endpoint.
- AlignScore requires an external checkpoint path in the metric config.
- Some internalized data/retrieval recipes preserve the original argparse options; pass extra options after the subcommand.
