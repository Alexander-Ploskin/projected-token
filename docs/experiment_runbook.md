# План экспериментов OSCAR retrieval (от простого к сложному)

Этот runbook полностью повторяет план A->E и добавляет команды для запуска каждого этапа.

## 0) Подготовка окружения

```bash
cd /home/a-ploskin/repos/ms-thesis/projected-token
poetry install
```

## 1) Точка отсчета и единый протокол оценки

Цель:
- зафиксировать один evaluation harness перед всеми train-ранами;
- использовать:
  - `projected_token/retrieval/pipeline.py`
  - `projected_token/retrieval/beir.py`
  - `projected_token/retrieval/bm25_baseline.py`
  - `projected_token/retrieval/metrics/ranking.py`
- считать метрики:
  - PopQA: `mrr@10`, `ndcg@10`, `recall@10`
  - BEIR3: per-dataset + average `ndcg@10`
  - delta к BM25 на тех же корпусах.

Зафиксированный протокол:
- `configs/retrieval/eval_protocol_oscar.yaml`

Команда запуска baseline-протокола:

```bash
poetry run python -m projected_token train-roadmap --stage freeze-eval
```

Артефакты этапа:
- `artifacts/results/retrieval/roadmap/bm25_popqa_metrics.json`
- `artifacts/results/retrieval/roadmap/bm25_beir_summary.json`
- `artifacts/results/retrieval/roadmap/bm25_baselines.json`

## 2) Этап A: усиленный contrastive без изменения компрессора

Идея:
- frozen compressor;
- абляции `pooler`, `loss`, negative strategy;
- датасеты:
  - `sentence-transformers/msmarco-msmarco-distilbert-base-v3`
  - `microsoft/ms_marco` (добавлен через поддержку raw формата в train pipeline).

### 2.1 Подготовка hard negatives для `microsoft/ms_marco` (опционально)

BM25-only mining:

```bash
poetry run python -m projected_token data prepare-msmarco \
  --corpus /data/huggingface/mteb/msmarco/corpus.jsonl \
  --queries /data/huggingface/mteb/msmarco/queries.jsonl \
  --qrels /data/huggingface/mteb/msmarco/qrels/train.tsv \
  --output-dir data/finetune/msmarco_v1 \
  --negative-type bm25 \
  --num-negatives 5
```

Dense-mining hard negatives (дальше в абляциях):
- `intfloat/e5-base-v2`
- `BAAI/bge-base-en-v1.5`

### 2.2 Запуск матрицы этапа A

```bash
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_a.yaml
```

Конфиги этапа A:
- `configs/training/stage_a_mlp_st_mnr_mean.yaml`
- `configs/training/stage_a_mlp_st_infonce_hardneg.yaml`
- `configs/training/stage_a_mlp_ms_marco_infonce.yaml`
- `configs/training/stage_a_lora_st_infonce.yaml`

Опционально запуск по одному:

```bash
poetry run python -m projected_token train --config configs/training/stage_a_mlp_st_mnr_mean.yaml
poetry run python -m projected_token train --config configs/training/stage_a_mlp_st_infonce_hardneg.yaml
poetry run python -m projected_token train --config configs/training/stage_a_mlp_ms_marco_infonce.yaml
poetry run python -m projected_token train --config configs/training/stage_a_lora_st_infonce.yaml
```

## 3) Этап B: distillation в проектор (компрессор frozen)

Teacher set:
- `intfloat/e5-base-v2`
- `BAAI/bge-base-en-v1.5`
- `intfloat/e5-large-v2` (опционально, тяжелее)

### 3.1 Генерация teacher embeddings

`e5-base-v2`:

```bash
poetry run python -m projected_token data teacher-embeddings \
  --corpus-path /data/huggingface/Tevatron/msmarco-passage-corpus/corpus.jsonl.gz \
  --num-samples 100000 \
  --output-dir /data/teacher-embeddings \
  --output-name e5-base-v2_teacher_embeddings_msmarco_100k.h5 \
  --teacher-model-name intfloat/e5-base-v2 \
  --pooling mean \
  --prompt-style e5
```

`bge-base-en-v1.5`:

```bash
poetry run python -m projected_token data teacher-embeddings \
  --corpus-path /data/huggingface/Tevatron/msmarco-passage-corpus/corpus.jsonl.gz \
  --num-samples 100000 \
  --output-dir /data/teacher-embeddings \
  --output-name bge-base-en-v1.5_teacher_embeddings_msmarco_100k.h5 \
  --teacher-model-name BAAI/bge-base-en-v1.5 \
  --pooling mean \
  --prompt-style none
```

`e5-large-v2`:

```bash
poetry run python -m projected_token data teacher-embeddings \
  --corpus-path /data/huggingface/Tevatron/msmarco-passage-corpus/corpus.jsonl.gz \
  --num-samples 100000 \
  --output-dir /data/teacher-embeddings \
  --output-name e5-large-v2_teacher_embeddings_msmarco_100k.h5 \
  --teacher-model-name intfloat/e5-large-v2 \
  --pooling mean \
  --prompt-style e5
```

### 3.2 Запуск матрицы этапа B

```bash
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_b_distill.yaml
```

Конфиги этапа B:
- `configs/training/stage_b_distill_e5_base.yaml`
- `configs/training/stage_b_distill_bge_base.yaml`
- `configs/training/stage_b_distill_e5_large.yaml`

## 4) Этап C: многоэтапные стратегии (distill -> contrastive)

Последовательности:
- C1: distill (multi-domain teacher targets) -> contrastive (MS MARCO)
- C2: distill -> contrastive mixed-domain (MS MARCO + NQ + PopQA-like)

Контроль деградации:
- после каждого этапа считать PopQA + BEIR3;
- если in-domain растет, а BEIR падает, сохранять checkpoint pre-overfit.

Запуск матрицы этапа C:

```bash
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_c_two_stage.yaml
```

Конфиги этапа C:
- `configs/training/stage_c_two_stage_from_e5_base.yaml`
- `configs/training/stage_c_two_stage_from_bge_base.yaml`

## 5) Этап D: retriever-first обучение с нуля под OSCAR-представления

Идея:
- новый recipe поверх текущего стека;
- joint loss: `alpha * contrastive + (1-alpha) * distill`;
- curriculum: сначала больше distill, потом больше contrastive.

Используемые файлы:
- `projected_token/encoders/projector.py`
- `projected_token/training/advanced_trainer.py`

Запуск этапа D:

```bash
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_d_joint_loss.yaml
```

Или один конфиг:

```bash
poetry run python -m projected_token train --config configs/training/stage_d_joint_loss.yaml
```

## 6) Этап E: fine-tune компрессора OSCAR

Используем:
- `recipe: full`
- частичный unfreeze / LoRA
- безопасные LR (`compressor_lr` в 5-10 раз ниже `projector_lr`).

Порядок повышения сложности:
- E1: unfreeze последние 2 слоя
- E2: unfreeze 4 слоя
- E3: LoRA на компрессор + проектор

Запуск матрицы этапа E:

```bash
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_e_full_ft.yaml
```

Конфиги этапа E:
- `configs/training/stage_e_full_unfreeze2_safe.yaml`
- `configs/training/stage_e_full_unfreeze4_safe.yaml`
- `configs/training/stage_e_full_lora_safe.yaml`

## 7) Рекомендуемый порядок запусков (go/no-go)

```bash
# 1) Freeze eval protocol + BM25 baselines
poetry run python -m projected_token train-roadmap --stage freeze-eval

# 2) Stage A
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_a.yaml

# 3) Stage B
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_b_distill.yaml

# 4) Stage C
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_c_two_stage.yaml

# 5) Stage D
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_d_joint_loss.yaml

# 6) Stage E
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_e_full_ft.yaml
```

Полный прогон через единый оркестратор:

```bash
poetry run python -m projected_token train-roadmap --stage all
```

## 8) Что считать успешным итогом

KPI:
- главный: рост BEIR average `ndcg@10` без существенной просадки PopQA `mrr@10`;
- второй: стабильность между доменами (меньшая дисперсия по `scifact` / `nfcorpus` / `fiqa`);
- практический: воспроизводимость лучшей конфигурации на 2-3 seed.

Финальные артефакты:
- матрицы: `artifacts/results/matrix/*.json`
- по стадиям: `artifacts/results/retrieval/roadmap/stage_*_best_eval.json`
- сводка: `artifacts/results/retrieval/roadmap/roadmap_summary.json`

## 9) Минимальный reproducible script (one-shot)

Ниже минимальный сценарий для запуска всего пайплайна подряд.  
Скрипт:
- останавливается при первой ошибке (`set -euo pipefail`);
- пишет лог в файл;
- выполняет этапы строго в порядке плана.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/a-ploskin/repos/ms-thesis/projected-token"
LOG_DIR="$REPO_DIR/artifacts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/roadmap_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[START] $(date -Is)"
cd "$REPO_DIR"

poetry install

# 1) Freeze eval protocol + BM25 baselines
poetry run python -m projected_token train-roadmap --stage freeze-eval

# 2) Stage A
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_a.yaml

# 3) Stage B
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_b_distill.yaml

# 4) Stage C
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_c_two_stage.yaml

# 5) Stage D
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_d_joint_loss.yaml

# 6) Stage E
poetry run python -m projected_token train-matrix --config configs/training/matrix_stage_e_full_ft.yaml

# Optional consolidated pass (re-evaluates best per stage via roadmap orchestrator)
poetry run python -m projected_token train-roadmap --stage all

echo "[DONE] $(date -Is)"
echo "Log saved to: $LOG_FILE"
```

Запуск в фоне (рекомендуется для долгих прогонов):

```bash
nohup bash run_roadmap.sh > /tmp/run_roadmap.out 2>&1 &
```
