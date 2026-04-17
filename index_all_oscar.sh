poetry run python index_dataset.py \
  --input-path /data/popqa_enriched.parquet \
  --text-col s_wiki_content \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --aggregation mean \
  --output-dir /data/popqa_index_oscar_mean

poetry run python index_dataset.py \
  --input-path /data/popqa_enriched.parquet \
  --text-col s_wiki_content \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --aggregation first \
  --output-dir /data/popqa_index_oscar_first

poetry run python index_dataset.py \
  --input-path /data/popqa_enriched.parquet \
  --text-col s_wiki_content \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --aggregation last \
  --output-dir /data/popqa_index_oscar_last

poetry run python index_dataset.py \
  --input-path /data/popqa_enriched.parquet \
  --text-col s_wiki_content \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --aggregation max \
  --output-dir /data/popqa_index_oscar_max

poetry run python index_dataset.py \
  --input-path /data/popqa_enriched.parquet \
  --text-col s_wiki_content \
  --encoder oscar \
  --model-name-or-path /data/huggingface/naver/oscar-qwen2-7B \
  --device cuda:0 \
  --aggregation mean_max \
  --output-dir /data/popqa_index_oscar_mean_max
