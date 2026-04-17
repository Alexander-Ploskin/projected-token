#!/bin/bash
# Пересчет метрик для всех методов с сохранением деталей

cd /app/last_projected-token

MODELS=(
  "/data/huggingface/naver/oscar-qwen2-7B"
)

AGGREGATIONS=(
  "mean:first:last:max:mean_max"
)

# Для каждого метода запускаем compute_retrieval_metrics

# popqa_index_oscar_mean
poetry run python compute_retrieval_metrics.py \
  --dataset-path /data/popqa_enriched.parquet \
  --index-path /data/popqa_index_oscar_mean/index \
  --metadata-path /data/popqa_index_oscar_mean/metadata.json \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --top-k 1 3 5 10 20 \
  --output-path /data/popqa_index_oscar_mean/retrieval_metrics.json \
  --save-retrieval-details /data/popqa_index_oscar_mean/retrieval_details.json

# popqa_index_oscar_first
poetry run python compute_retrieval_metrics.py \
  --dataset-path /data/popqa_enriched.parquet \
  --index-path /data/popqa_index_oscar_first/index \
  --metadata-path /data/popqa_index_oscar_first/metadata.json \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --top-k 1 3 5 10 20 \
  --output-path /data/popqa_index_oscar_first/retrieval_metrics.json \
  --save-retrieval-details /data/popqa_index_oscar_first/retrieval_details.json

# popqa_index_oscar_last
poetry run python compute_retrieval_metrics.py \
  --dataset-path /data/popqa_enriched.parquet \
  --index-path /data/popqa_index_oscar_last/index \
  --metadata-path /data/popqa_index_oscar_last/metadata.json \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --top-k 1 3 5 10 20 \
  --output-path /data/popqa_index_oscar_last/retrieval_metrics.json \
  --save-retrieval-details /data/popqa_index_oscar_last/retrieval_details.json

# popqa_index_oscar_max
poetry run python compute_retrieval_metrics.py \
  --dataset-path /data/popqa_enriched.parquet \
  --index-path /data/popqa_index_oscar_max/index \
  --metadata-path /data/popqa_index_oscar_max/metadata.json \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --top-k 1 3 5 10 20 \
  --output-path /data/popqa_index_oscar_max/retrieval_metrics.json \
  --save-retrieval-details /data/popqa_index_oscar_max/retrieval_details.json

# popqa_index_oscar_mean_max
poetry run python compute_retrieval_metrics.py \
  --dataset-path /data/popqa_enriched.parquet \
  --index-path /data/popqa_index_oscar_mean_max/index \
  --metadata-path /data/popqa_index_oscar_mean_max/metadata.json \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --top-k 1 3 5 10 20 \
  --output-path /data/popqa_index_oscar_mean_max/retrieval_metrics.json \
  --save-retrieval-details /data/popqa_index_oscar_mean_max/retrieval_details.json

# popqa_index_salesforce
poetry run python compute_retrieval_metrics.py \
  --dataset-path /data/popqa_enriched.parquet \
  --index-path /data/popqa_index_salesforce/index \
  --metadata-path /data/popqa_index_salesforce/metadata.json \
  --encoder salesforce \
  --model-name-or-path /data/huggingface/Salesforce/SFR-Embedding-Mistral \
  --device cuda:0 \
  --top-k 1 3 5 10 20 \
  --output-path /data/popqa_index_salesforce/retrieval_metrics.json \
  --save-retrieval-details /data/popqa_index_salesforce/retrieval_details.json

echo "Метрики пересчитаны!"