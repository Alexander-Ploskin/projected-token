#!/usr/bin/env python3
"""Generate the full teacher-embedding bundle for query distillation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from projected_token.data.recipes.generate_query_doc_teacher_embeddings import generate_query_doc_teacher_embeddings


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    output_name: str
    max_samples: int
    negative_strategy: str
    beir_dataset: str | None = None
    preferred_split: str = "train"
    bm25_rank_start: int = 10
    bm25_rank_end: int = 60
    dense_rank_start: int = 10
    dense_rank_end: int = 60
    use_score_band: bool = True
    use_cross_encoder: bool = True


def _require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path: {path}")


def _resolve_split(beir_root: Path, dataset: str, preferred: str) -> str:
    preferred_path = beir_root / dataset / "qrels" / f"{preferred}.tsv"
    if preferred_path.exists():
        return preferred
    for candidate in ("dev", "test", "train"):
        candidate_path = beir_root / dataset / "qrels" / f"{candidate}.tsv"
        if candidate_path.exists():
            return candidate
    raise FileNotFoundError(f"No qrels split available for {dataset} in {beir_root / dataset / 'qrels'}")


def _download_beir_dataset_if_missing(beir_root: Path, dataset: str) -> None:
    corpus_path = beir_root / dataset / "corpus.jsonl"
    if corpus_path.exists():
        return
    beir_root.mkdir(parents=True, exist_ok=True)
    try:
        from beir import util
    except ImportError as exc:
        raise ImportError("Install beir package to auto-download missing BEIR datasets") from exc
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    print(f"[download] missing BEIR dataset {dataset}; downloading from {url}")
    util.download_and_unzip(url, str(beir_root))


def _default_specs(
    *,
    msmarco_samples: int = 500000,
    nf_samples: int = 3200,
    fiqa_samples: int = 14000,
    arguana_samples: int = 1400,
    quora_samples: int = 15000,
    scifact_samples: int = 800,
) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="msmarco",
            output_name="msmarco-hard.h5",
            max_samples=msmarco_samples,
            negative_strategy="random",
            beir_dataset=None,
            preferred_split="train",
            use_score_band=False,
            use_cross_encoder=False,
            bm25_rank_start=10,
            bm25_rank_end=50,
            dense_rank_start=10,
            dense_rank_end=200,
        ),
        DatasetSpec(
            name="nfcorpus",
            output_name="nfcorpus-train-hard.h5",
            max_samples=nf_samples,
            negative_strategy="hybrid",
            beir_dataset="nfcorpus",
            preferred_split="train",
            bm25_rank_start=10,
            bm25_rank_end=60,
            dense_rank_start=10,
            dense_rank_end=60,
        ),
        DatasetSpec(
            name="fiqa",
            output_name="fiqa-train-hard.h5",
            max_samples=fiqa_samples,
            negative_strategy="bge_dense",
            beir_dataset="fiqa",
            preferred_split="train",
            bm25_rank_start=10,
            bm25_rank_end=60,
            dense_rank_start=10,
            dense_rank_end=100,
        ),
        DatasetSpec(
            name="arguana",
            output_name="arguana-train-hard.h5",
            max_samples=arguana_samples,
            negative_strategy="bge_dense",
            beir_dataset="arguana",
            preferred_split="train",
            bm25_rank_start=10,
            bm25_rank_end=60,
            dense_rank_start=10,
            dense_rank_end=100,
        ),
        DatasetSpec(
            name="quora",
            output_name="quora-train-hard.h5",
            max_samples=quora_samples,
            negative_strategy="bge_dense",
            beir_dataset="quora",
            preferred_split="train",
            bm25_rank_start=10,
            bm25_rank_end=60,
            dense_rank_start=10,
            dense_rank_end=100,
        ),
        DatasetSpec(
            name="scifact",
            output_name="scifact-train-hard.h5",
            max_samples=scifact_samples,
            negative_strategy="hybrid",
            beir_dataset="scifact",
            preferred_split="train",
            bm25_rank_start=10,
            bm25_rank_end=60,
            dense_rank_start=10,
            dense_rank_end=60,
        ),
    ]


def _apply_overrides(specs: list[DatasetSpec], args: argparse.Namespace) -> list[DatasetSpec]:
    by_name = {spec.name: spec for spec in specs}
    if args.msmarco_negative_strategy:
        spec = by_name["msmarco"]
        by_name["msmarco"] = DatasetSpec(
            name=spec.name,
            output_name=spec.output_name,
            max_samples=spec.max_samples,
            negative_strategy=args.msmarco_negative_strategy,
            beir_dataset=spec.beir_dataset,
            preferred_split=spec.preferred_split,
            bm25_rank_start=args.msmarco_bm25_rank_start if args.msmarco_bm25_rank_start is not None else spec.bm25_rank_start,
            bm25_rank_end=args.msmarco_bm25_rank_end if args.msmarco_bm25_rank_end is not None else spec.bm25_rank_end,
            dense_rank_start=args.msmarco_dense_rank_start if args.msmarco_dense_rank_start is not None else spec.dense_rank_start,
            dense_rank_end=args.msmarco_dense_rank_end if args.msmarco_dense_rank_end is not None else spec.dense_rank_end,
            use_score_band=spec.use_score_band,
            use_cross_encoder=spec.use_cross_encoder,
        )
    return list(by_name.values())


def _is_h5_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    summary_path = path.with_suffix(".summary.json")
    return summary_path.exists() and summary_path.stat().st_size > 0


def generate_full_query_distill_teacher_embeddings(
    *,
    out_dir: Path,
    beir_root: Path,
    model_name: str,
    device: str,
    batch_size: int,
    max_seq_length: int,
    score_band_min_margin: float,
    score_band_max_margin: float,
    cross_encoder_model_name: str | None,
    cross_encoder_threshold: float,
    cross_encoder_batch_size: int,
    force: bool,
    download_beir_missing: bool,
    specs: list[DatasetSpec],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for spec in specs:
        output_path = out_dir / spec.output_name
        if not force and _is_h5_complete(output_path):
            summary_path = output_path.with_suffix(".summary.json")
            results[spec.name] = {
                "status": "cached",
                "output_path": str(output_path),
                "summary_path": str(summary_path),
            }
            print(f"[cache] {spec.name}: using existing {output_path}")
            continue

        beir_dir = None
        beir_split = "train"
        if spec.beir_dataset:
            if download_beir_missing:
                _download_beir_dataset_if_missing(beir_root, spec.beir_dataset)
            beir_dir = beir_root / spec.beir_dataset
            _require_path(beir_dir / "corpus.jsonl")
            _require_path(beir_dir / "queries.jsonl")
            beir_split = _resolve_split(beir_root, spec.beir_dataset, spec.preferred_split)
            _require_path(beir_dir / "qrels" / f"{beir_split}.tsv")

        summary = generate_query_doc_teacher_embeddings(
            output_path=output_path,
            input_json=None,
            beir_dir=beir_dir,
            beir_split=beir_split,
            max_samples=spec.max_samples,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            negative_strategy=spec.negative_strategy,
            bm25_rank_start=spec.bm25_rank_start,
            bm25_rank_end=spec.bm25_rank_end,
            bm25_seed=42,
            dense_rank_start=spec.dense_rank_start,
            dense_rank_end=spec.dense_rank_end,
            dense_query_batch_size=64,
            score_band_min_margin=score_band_min_margin if spec.use_score_band else None,
            score_band_max_margin=score_band_max_margin if spec.use_score_band else None,
            cross_encoder_model_name=cross_encoder_model_name if spec.use_cross_encoder else None,
            cross_encoder_threshold=cross_encoder_threshold,
            cross_encoder_batch_size=cross_encoder_batch_size,
        )
        results[spec.name] = {
            "status": "generated",
            "output_path": str(output_path),
            "summary": summary,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate query-distill teacher embeddings bundle")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/teacher-embeddings/query-doc-bge-base-full"))
    parser.add_argument("--beir-root", type=Path, default=Path("/data/beir"))
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--score-band-min-margin", type=float, default=0.05)
    parser.add_argument("--score-band-max-margin", type=float, default=0.30)
    parser.add_argument("--cross-encoder-model-name", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-encoder-threshold", type=float, default=0.5)
    parser.add_argument("--cross-encoder-batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-beir-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--msmarco-samples", type=int, default=500000)
    parser.add_argument("--nf-samples", type=int, default=3200)
    parser.add_argument("--fiqa-samples", type=int, default=14000)
    parser.add_argument("--arguana-samples", type=int, default=1400)
    parser.add_argument("--quora-samples", type=int, default=15000)
    parser.add_argument("--scifact-samples", type=int, default=800)
    parser.add_argument("--msmarco-negative-strategy", choices=("random", "bm25", "bge_dense", "hybrid"), default=None)
    parser.add_argument("--msmarco-bm25-rank-start", type=int, default=None)
    parser.add_argument("--msmarco-bm25-rank-end", type=int, default=None)
    parser.add_argument("--msmarco-dense-rank-start", type=int, default=None)
    parser.add_argument("--msmarco-dense-rank-end", type=int, default=None)
    args = parser.parse_args()

    specs = _apply_overrides(
        _default_specs(
            msmarco_samples=int(args.msmarco_samples),
            nf_samples=int(args.nf_samples),
            fiqa_samples=int(args.fiqa_samples),
            arguana_samples=int(args.arguana_samples),
            quora_samples=int(args.quora_samples),
            scifact_samples=int(args.scifact_samples),
        ),
        args,
    )
    result = generate_full_query_distill_teacher_embeddings(
        out_dir=args.out_dir,
        beir_root=args.beir_root,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        score_band_min_margin=args.score_band_min_margin,
        score_band_max_margin=args.score_band_max_margin,
        cross_encoder_model_name=args.cross_encoder_model_name,
        cross_encoder_threshold=args.cross_encoder_threshold,
        cross_encoder_batch_size=args.cross_encoder_batch_size,
        force=args.force,
        download_beir_missing=bool(args.download_beir_missing),
        specs=specs,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
