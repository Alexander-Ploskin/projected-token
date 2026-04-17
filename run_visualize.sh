#!/bin/bash
# Визуализация результатов поиска для PopQA

cd /app/last_projected-token

# Запускаем визуализацию для всех методов
poetry run python visualize_results.py \
  --dataset-path /data/popqa_enriched.parquet \
  --output-dir /data/popqa_visualizations \
  --methods \
    /data/popqa_index_oscar_mean \
    /data/popqa_index_oscar_first \
    /data/popqa_index_oscar_last \
    /data/popqa_index_oscar_max \
    /data/popqa_index_oscar_mean_max \
    /data/popqa_index_salesforce

echo "Графики сохранены в /data/popqa_visualizations/"
ls -la /data/popqa_visualizations/