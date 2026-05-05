# OSCAR Projector Experiments Journal

Consolidated report of all currently available artifacts for OSCAR projector experiments.
Goal: build strong question-to-document embeddings on top of OSCAR.

## Snapshot (2026-04-29, current workspace state)

- Distillation matrix artifacts are available for 1/2/3-layer MLP.
- MS MARCO validation comparisons with baselines and OSCAR no-projector baselines are available for `val_split=0.02` and `val_split=0.1`.
- PopQA enriched comparisons are available for `BM25`, `SFR`, and one projector checkpoint (`contrastive_3layer`).
- BEIR-3 has complete `BM25` results and complete `projector_best` results; `SFR` has completed `scifact` and `nfcorpus` artifacts, `fiqa-2018` is still pending in the running job.
- t-SNE figures are generated for 3 datasets (`msmarco`, `popqa_enriched`, `scifact`) x 7 models.

## Environment / Repro Notes

- Runtime: docker container `aploskin-opencode`.
- Hardware: 2x H100.
- Package policy: installs through internal Artifactory only.
- All experiment outputs are stored under `artifacts/`.

## Run Status Log (latest relevant)

| Timestamp (UTC) | Stage | Status | Key point |
|---|---|---|---|
| 2026-04-28 15:38 | Distill matrix (1/2/3 layers) | completed | Summary exists: `artifacts/results/matrix/distill_summary.json`. |
| 2026-04-28 20:24 | Contrastive matrix relaunch | failed | OOM from overlapping heavy jobs on same GPU. |
| 2026-04-28 20:24 | Two-stage matrix relaunch | failed | OOM from overlapping heavy jobs on same GPU. |
| 2026-04-28 21:22 | MS MARCO val compare (`0.02`) | completed | Baselines + projector checkpoints exported. |
| 2026-04-28 21:30 | MS MARCO val compare (`0.1`) | completed | Baselines + projector checkpoints exported. |
| 2026-04-28 21:02-21:03 | PopQA enriched compare | completed | `BM25`, `SFR`, `projector_contrastive3` exported. |
| 2026-04-28 22:48+ | BEIR targeted methods | partial | `BM25` and `projector_best` complete; `SFR`/`OSCAR flatten` final summaries still pending in active runs. |

## Training / Checkpoint Metrics (available)

Source: `artifacts/analysis/checkpoint_snapshot/checkpoint_eval_summary.json`.

| Run | Loss | MRR | MRR@10 | NDCG@10 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| contrastive_1layer | 1.5446 | 0.7246 | 0.7210 | 0.7720 | 0.6112 | 0.8689 | 0.9306 |
| contrastive_2layer | 1.2374 | 0.7555 | 0.7534 | 0.8044 | 0.6370 | 0.9121 | 0.9615 |
| contrastive_3layer | 1.2046 | 0.7595 | 0.7572 | 0.8073 | 0.6411 | 0.9128 | 0.9610 |
| two_stage_1layer | 2.4111 | 0.5309 | 0.5222 | 0.5943 | 0.3860 | 0.7101 | 0.8233 |
| distill_2layer | - | 0.0313 | 0.0174 | 0.0259 | 0.0077 | 0.0272 | 0.0548 |

### Distillation depth snapshot

Source: same checkpoint snapshot (`distill_depth` section).

| Distill variant | Best step | MSE | Cosine |
|---|---|---:|---:|
| distill_1layer | step_2232 | 0.003195 | 0.996805 |
| distill_3layer | step_2976 | 0.001491 | 0.998509 |
| distill_2layer (latest parsed) | latest | 0.000008 | 0.996993 |

## MS MARCO Validation: Baselines vs OSCAR / Projector

### Setup A: `val_split=0.02`, `n=512`

Source: `artifacts/results/retrieval/msmarco_val_projector_vs_baselines_v002_with_oscar_pooling.json`.

| Method | MRR | MRR@10 | NDCG@10 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 baseline | 0.9531 | 0.9522 | 0.9584 | 0.9336 | 0.9766 | 0.9766 |
| SFR baseline | 0.9990 | 0.9990 | 0.9993 | 0.9980 | 1.0000 | 1.0000 |
| OSCAR baseline first | 0.1681 | 0.1436 | 0.1886 | 0.0859 | 0.2109 | 0.3398 |
| OSCAR baseline last | 0.0764 | 0.0479 | 0.0718 | 0.0195 | 0.0781 | 0.1523 |
| OSCAR baseline flatten | 0.0819 | 0.0532 | 0.0799 | 0.0195 | 0.0801 | 0.1699 |
| Projector contrastive 1-layer | 0.8420 | 0.8404 | 0.8732 | 0.7578 | 0.9473 | 0.9727 |
| Projector contrastive 2-layer | 0.8201 | 0.8183 | 0.8555 | 0.7285 | 0.9453 | 0.9688 |
| Projector contrastive 3-layer | 0.8215 | 0.8201 | 0.8592 | 0.7246 | 0.9375 | 0.9785 |
| Projector two-stage 1-layer | 0.7163 | 0.7129 | 0.7686 | 0.5938 | 0.8789 | 0.9414 |

### Setup B: `val_split=0.1`, `n=2048`

Source: `artifacts/results/retrieval/msmarco_val_projector_vs_baselines_v010_with_oscar_pooling.json`.

| Method | MRR | MRR@10 | NDCG@10 | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 baseline | 0.9430 | 0.9424 | 0.9521 | 0.9146 | 0.9751 | 0.9810 |
| SFR baseline | 0.9993 | 0.9993 | 0.9995 | 0.9985 | 1.0000 | 1.0000 |
| OSCAR baseline first | 0.1633 | 0.1384 | 0.1842 | 0.0781 | 0.2080 | 0.3374 |
| OSCAR baseline last | 0.0754 | 0.0472 | 0.0720 | 0.0176 | 0.0796 | 0.1558 |
| OSCAR baseline flatten | 0.0794 | 0.0506 | 0.0773 | 0.0186 | 0.0840 | 0.1675 |
| Projector contrastive 1-layer | 0.8689 | 0.8680 | 0.8971 | 0.7939 | 0.9644 | 0.9854 |
| Projector contrastive 2-layer | 0.8294 | 0.8285 | 0.8674 | 0.7295 | 0.9614 | 0.9849 |
| Projector contrastive 3-layer | 0.8423 | 0.8415 | 0.8778 | 0.7476 | 0.9624 | 0.9873 |
| Projector two-stage 1-layer | 0.7305 | 0.7278 | 0.7842 | 0.6050 | 0.8975 | 0.9590 |

## PopQA Enriched (`/data/popqa_enriched.parquet`)

| Method | MRR | NDCG@10 | R@10 |
|---|---:|---:|---:|
| BM25 baseline | 0.6182 | 0.6634 | 0.8170 |
| SFR baseline | 0.7668 | 0.8098 | 0.9458 |
| Projector contrastive 3-layer | 0.0085 | 0.0107 | 0.0211 |

Sources:
- `artifacts/results/retrieval/popqa_enriched_bm25_baseline_metrics.json`
- `artifacts/results/retrieval/popqa_enriched_sfr_baseline_metrics.json`
- `artifacts/results/retrieval/popqa_enriched_contrastive3_metrics.json`

## BEIR-3 (SciFact / NFCorpus / FiQA-2018)

### BM25 complete results

Source: `artifacts/results/retrieval/comparison_with_oscar_pooling/bm25_beir3_summary.json`.

| Dataset | MRR | NDCG@10 | R@10 |
|---|---:|---:|---:|
| scifact | 0.6372 | 0.6647 | 0.7849 |
| nfcorpus | 0.5239 | 0.3105 | 0.1525 |
| fiqa-2018 | 0.2958 | 0.2322 | 0.2979 |
| average | 0.4856 | 0.4024 | 0.4118 |

### SFR current available BEIR results (partial)

Sources:
- `artifacts/results/retrieval/beir3_comparison/sfr/scifact_metrics.json`
- `artifacts/results/retrieval/beir3_comparison/sfr/nfcorpus_metrics.json`

| Dataset | MRR | NDCG@10 | R@10 | Status |
|---|---:|---:|---:|---|
| scifact | 0.7262 | 0.7541 | 0.8722 | completed |
| nfcorpus | 0.5968 | 0.3852 | 0.1887 | completed |
| fiqa-2018 | - | - | - | pending in running job |

### Projector best BEIR-3 (complete)

Source: `artifacts/results/retrieval/beir_targeted_methods/projector_best_beir3_summary.json`.

| Dataset | MRR | NDCG@10 | R@10 |
|---|---:|---:|---:|
| scifact | 0.0832 | 0.0877 | 0.1403 |
| nfcorpus | 0.1398 | 0.0527 | 0.0160 |
| fiqa-2018 | 0.0299 | 0.0206 | 0.0352 |
| average | 0.0843 | 0.0537 | 0.0638 |

### OSCAR baseline flatten BEIR-3 (complete)

Source: `artifacts/results/retrieval/beir_targeted_methods/oscar_flatten_beir3_summary.json`.

| Dataset | MRR | NDCG@10 | R@10 |
|---|---:|---:|---:|
| scifact | 0.0011 | 0.0000 | 0.0000 |
| nfcorpus | 0.0514 | 0.0168 | 0.0022 |
| fiqa-2018 | 0.0044 | 0.0017 | 0.0013 |
| average | 0.0190 | 0.0061 | 0.0012 |

## t-SNE Analysis Assets

Source: `artifacts/analysis/tsne_retrieval_models_v3/summary.json`.

- Datasets: `msmarco`, `popqa_enriched`, `scifact`.
- Models per dataset (7): `sfr_baseline`, `oscar_first`, `oscar_last`, `oscar_flatten`, `projector_contrastive_1layer`, `projector_contrastive_3layer`, `projector_two_stage_1layer`.
- Sample pairs per dataset: `160`.
- Total generated plots: `21` PNG.

## Additional Historical Retrieval Artifacts (legacy context)

These artifacts are present and numerically valid, but protocol/config is older and not fully aligned with the latest comparison scripts:

| File | MRR | NDCG@10 | R@10 | Count |
|---|---:|---:|---:|---:|
| `artifacts/results/retrieval_metrics.json` | 0.7621 | 0.8053 | 0.9420 | 14232 |
| `artifacts/results/short_popqa_retrieval_metrics.json` | 0.3353 | 0.3594 | 0.4498 | 5000 |
| `artifacts/results/short_msmarco_retrieval_metrics.json` | 0.0085 | 0.0100 | 0.0200 | 500 |
| `artifacts/results/short_nq_retrieval_metrics.json` | 0.0079 | 0.0090 | 0.0180 | 500 |
| `artifacts/results/popqa_content_retrieval_metrics.json` | 0.0520 | 0.0596 | 0.1000 | 500 |
| `artifacts/results/popqa_title_retrieval_metrics.json` | 0.3492 | 0.3837 | 0.5120 | 500 |

## Key Findings (presentation-ready)

1. On MS MARCO validation, projector checkpoints are much stronger than raw OSCAR no-projector pooling, but still behind SFR and BM25 under current protocol.
2. Best projector on MS MARCO (`contrastive_1layer`, `val_split=0.1`) reaches `MRR=0.8689`, `R@10=0.9854`, showing retrieval signal is strong in-domain.
3. Cross-domain transfer is currently the main weakness: on PopQA and BEIR, projector results lag far behind both `SFR` and `BM25`.
4. Distillation aligns embedding space very tightly (low MSE / high cosine), but alignment alone does not guarantee retrieval quality on out-of-domain benchmarks.
5. SFR remains strongest and most stable baseline across available datasets; BM25 remains a robust lexical baseline that projector must beat in BEIR/PopQA before claiming generalization.

## Risks / Gaps Before Final Defense

- BEIR `SFR` and `OSCAR flatten` targeted summary jobs are still running and final JSON/CSV not yet materialized.
- Full MS MARCO evaluation output files (`*_full.json/csv/png`) are not present in artifacts yet.
- Distill-2 checkpoint could not be loaded in projector-eval path due state-dict mismatch and was skipped in those comparisons.

## Recommended Finalization Steps (for final slide deck)

1. Wait for active BEIR jobs to finish and append final `SFR` + `OSCAR flatten` rows.
2. Rebuild a single BEIR master table with four methods: `BM25`, `SFR`, `OSCAR flatten`, `projector_best`.
3. Add one final comparison figure per benchmark (`MS MARCO`, `PopQA`, `BEIR`) from existing CSV/JSON artifacts.
4. Keep paused training tracks (`contrastive`, `two-stage`, `lora/unfreeze`) explicitly marked as pending scope after current evaluation phase.
