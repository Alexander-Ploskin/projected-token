# OSCAR Projector Experiment Runbook

## 1) Blockers and correctness prerequisites

Before running heavy experiments, ensure these fixes are present:
- `trainer_full`: removed undefined `query_encoder`.
- `FullFineTuneProjector.forward`: uses `self.projector` and normalizes output.
- Full path gradients enabled through OSCAR in train mode.
- `advanced_trainer`: optimizer initialization order fixed; teacher projector wired into combined loss.
- Mixed dataset schema compatibility: supports `negative` and `negatives`.
- `InfoNCE`: in-batch branch uses `query x positive` logits.

## 2) Contrastive matrix (1/2/3-layer MLP)

Run:

```bash
python -m projected_token train-matrix --config configs/training/matrix_contrastive.yaml
```

Configs included:
- `configs/training/projector_mlp_1layer_msmarco.yaml`
- `configs/training/projector_mlp_2layer_msmarco.yaml`
- `configs/training/projector_mlp_3layer_msmarco.yaml`
- `configs/training/projector_mlp_1layer_msmarco_v2.yaml`
- `configs/training/projector_mlp_2layer_msmarco_v2.yaml`
- `configs/training/projector_mlp_3layer_msmarco_v2.yaml`
- `configs/training/projector_flat_mixed_1layer.yaml`
- `configs/training/projector_flat_mixed_2layer.yaml`
- `configs/training/projector_flat_mixed_3layer.yaml`

## 3) Distillation matrix (SFR teacher, 1/2/3-layer)

Run:

```bash
python -m projected_token train-matrix --config configs/training/matrix_distill.yaml
```

Configs:
- `configs/training/projector_distill_1layer.yaml`
- `configs/training/projector_distill_2layer.yaml`
- `configs/training/projector_distill_3layer.yaml`

## 4) Two-stage training (distill -> contrastive)

1. Run distillation matrix and select best checkpoints.
2. Run:

```bash
python -m projected_token train-matrix --config configs/training/matrix_two_stage.yaml
```

Configs:
- `configs/training/two_stage_distill_to_contrastive_1layer.yaml`
- `configs/training/two_stage_distill_to_contrastive_2layer.yaml`
- `configs/training/two_stage_distill_to_contrastive_3layer.yaml`

`init_projector_checkpoint` points to stage-A outputs.

## 5) LoRA / partial-unfreeze ablations

Run:

```bash
python -m projected_token train-matrix --config configs/training/matrix_lora_unfreeze.yaml
```

Configs:
- `configs/training/projector_full_lora.yaml`
- `configs/training/projector_full_unfreeze2.yaml`
- `configs/training/projector_full_unfreeze4.yaml`

## 6) Retrieval and BEIR-3 evaluation

PopQA retrieval:

```bash
python -m projected_token retrieval build-index --config configs/retrieval/popqa_oscar_projector.yaml
python -m projected_token retrieval evaluate --config configs/retrieval/popqa_oscar_projector.yaml
```

BEIR-3:

```bash
python -m projected_token retrieval evaluate-beir --config configs/retrieval/beir3_oscar_projector.yaml
```

Datasets:
- SciFact
- NFCorpus
- FiQA-2018

## 7) Metrics and artifacts

Primary metrics:
- `mrr@10`
- `ndcg@10`
- `recall@k` for `k in [1,3,5,10,20]`

All runs produce structured artifacts under:
- `artifacts/runs/<run_id>/checkpoints`
- `artifacts/runs/<run_id>/logs/tensorboard`
- `artifacts/runs/<run_id>/metrics/*.json`
- `artifacts/runs/<run_id>/metrics/*.csv`
- `artifacts/runs/<run_id>/plots/*.png`

Matrix summaries:
- `artifacts/results/matrix/*.json`
- `artifacts/results/matrix/*.csv`
- `artifacts/results/matrix/*.png`

Retrieval summaries:
- `artifacts/results/retrieval/*.json`
- `artifacts/results/retrieval/*.csv`
- `artifacts/results/retrieval/*.png`
