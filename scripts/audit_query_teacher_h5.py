#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import h5py


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def audit_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        query_ds = h5.get("queries")
        candidate_docs = h5.get("candidate_docs")
        teacher_scores = h5.get("teacher_scores")
        candidate_embeddings = h5.get("candidate_embeddings")
        relevance = h5.get("relevance")
        if relevance is None:
            relevance = h5.get("labels")
        candidate_source = h5.get("candidate_source")
        q_emb = h5.get("query_embeddings")
        if candidate_docs is not None or teacher_scores is not None:
            required = {
                "queries": query_ds,
                "candidate_docs": candidate_docs,
                "teacher_scores": teacher_scores,
                "query_embeddings": q_emb,
            }
            missing = [name for name, ds in required.items() if ds is None]
            if missing:
                return {
                    "path": str(path),
                    "valid": False,
                    "format": "candidate",
                    "error": f"missing datasets: {missing}",
                }
            rows = int(query_ds.shape[0])
            candidate_shape = tuple(int(x) for x in candidate_docs.shape)
            teacher_shape = tuple(int(x) for x in teacher_scores.shape)
            candidate_k = int(candidate_docs.shape[1]) if len(candidate_docs.shape) > 1 else 0
            shape_consistent = (
                int(candidate_docs.shape[0]) == rows
                and int(teacher_scores.shape[0]) == rows
                and tuple(candidate_docs.shape[:2]) == tuple(teacher_scores.shape[:2])
                and int(q_emb.shape[0]) == rows
            )
            if candidate_embeddings is not None:
                shape_consistent = shape_consistent and tuple(candidate_embeddings.shape[:2]) == tuple(candidate_docs.shape[:2])
            if relevance is not None:
                shape_consistent = shape_consistent and tuple(relevance.shape[:2]) == tuple(candidate_docs.shape[:2])
            if candidate_source is not None:
                shape_consistent = shape_consistent and tuple(candidate_source.shape[:2]) == tuple(candidate_docs.shape[:2])
            split_guess = "train"
            stem = path.stem.lower()
            if "dev" in stem:
                split_guess = "dev"
            elif "test" in stem:
                split_guess = "test"
            teacher_min = float(teacher_scores[:].min()) if rows > 0 else 0.0
            teacher_max = float(teacher_scores[:].max()) if rows > 0 else 0.0
            return {
                "path": str(path),
                "valid": bool(shape_consistent),
                "format": "candidate",
                "rows": rows,
                "candidate_k": candidate_k,
                "candidate_shape": candidate_shape,
                "teacher_scores_shape": teacher_shape,
                "candidate_embeddings_shape": tuple(int(x) for x in candidate_embeddings.shape) if candidate_embeddings is not None else None,
                "has_relevance": relevance is not None,
                "has_candidate_source": candidate_source is not None,
                "teacher_score_min": teacher_min,
                "teacher_score_max": teacher_max,
                "query_example": _decode_text(query_ds[0]) if rows > 0 else "",
                "candidate_example": _decode_text(candidate_docs[0, 0]) if rows > 0 and candidate_k > 0 else "",
                "split_guess": split_guess,
                "has_test_marker": "test" in stem,
            }
        pos_ds = h5.get("positive_docs")
        neg_ds = h5.get("negative_docs")
        p_emb = h5.get("positive_embeddings")
        n_emb = h5.get("negative_embeddings")
        required = {
            "queries": query_ds,
            "positive_docs": pos_ds,
            "negative_docs": neg_ds,
            "query_embeddings": q_emb,
            "positive_embeddings": p_emb,
            "negative_embeddings": n_emb,
        }
        missing = [name for name, ds in required.items() if ds is None]
        if missing:
            return {
                "path": str(path),
                "valid": False,
                "error": f"missing datasets: {missing}",
            }

        rows = int(query_ds.shape[0])
        shape_consistent = (
            int(pos_ds.shape[0]) == rows
            and int(neg_ds.shape[0]) == rows
            and int(q_emb.shape[0]) == rows
            and int(p_emb.shape[0]) == rows
            and int(n_emb.shape[0]) == rows
        )
        emb_dims = {
            "query_embeddings": int(q_emb.shape[-1]),
            "positive_embeddings": int(p_emb.shape[-1]),
            "negative_embeddings": int(n_emb.shape[-1]),
        }
        multi_negative = len(n_emb.shape) == 3
        split_guess = "train"
        stem = path.stem.lower()
        if "dev" in stem:
            split_guess = "dev"
        elif "test" in stem:
            split_guess = "test"

        return {
            "path": str(path),
            "valid": bool(shape_consistent),
            "format": "triple",
            "rows": rows,
            "query_example": _decode_text(query_ds[0]) if rows > 0 else "",
            "positive_example": _decode_text(pos_ds[0]) if rows > 0 else "",
            "negative_example": _decode_text(neg_ds[0]) if rows > 0 else "",
            "emb_dims": emb_dims,
            "negative_mode": "multi" if multi_negative else "single",
            "split_guess": split_guess,
            "has_test_marker": "test" in stem,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit query distill H5 teacher bundles.")
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="H5 paths or glob patterns.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/results/retrieval/query_distill_h5_audit.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    resolved: list[Path] = []
    for raw in args.paths:
        matches = sorted(Path(p) for p in glob.glob(raw) if Path(p).is_file())
        if matches:
            resolved.extend(matches)
        else:
            candidate = Path(raw)
            if candidate.exists() and candidate.is_file():
                resolved.append(candidate)
    resolved = list(dict.fromkeys(resolved))
    if not resolved:
        raise FileNotFoundError("No H5 files found for provided --paths.")

    reports = [audit_h5(path) for path in resolved]
    total_rows = sum(int(item.get("rows", 0)) for item in reports if item.get("valid"))
    format_counts: dict[str, int] = {}
    for item in reports:
        format_counts[str(item.get("format", "unknown"))] = format_counts.get(str(item.get("format", "unknown")), 0) + 1
    has_test_files = [item["path"] for item in reports if item.get("has_test_marker")]
    summary = {
        "files": reports,
        "num_files": len(reports),
        "num_valid": sum(1 for item in reports if item.get("valid")),
        "total_rows": total_rows,
        "format_counts": format_counts,
        "has_test_files": has_test_files,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
