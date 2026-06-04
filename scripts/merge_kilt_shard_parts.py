#!/usr/bin/env python3
"""Merge FAISS sub-parts of a KILT shard into a single index file."""

import argparse
import json
from pathlib import Path

import faiss


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge KILT shard sub-parts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-parts", type=int, required=True)
    args = parser.parse_args()

    merged_index = faiss.IndexFlatIP(768)
    merged_ids: list[str] = []

    for part_id in range(args.num_parts):
        suffix = f"_part{part_id}"
        faiss_path = args.output_dir / f"kilt_shard_{args.shard_id}{suffix}.faiss"
        ids_path = args.output_dir / f"kilt_shard_{args.shard_id}{suffix}_ids.json"
        if not faiss_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"Missing part {part_id}: {faiss_path}, {ids_path}")

        part_index = faiss.read_index(str(faiss_path))
        part_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if part_index.ntotal != len(part_ids):
            raise ValueError(
                f"Part {part_id} mismatch: faiss={part_index.ntotal}, ids={len(part_ids)}"
            )

        vectors = part_index.reconstruct_n(0, part_index.ntotal)
        merged_index.add(vectors)
        merged_ids.extend(part_ids)
        print(f"Merged part {part_id}: {part_index.ntotal:,} vectors")

    out_faiss = args.output_dir / f"kilt_shard_{args.shard_id}.faiss"
    out_ids = args.output_dir / f"kilt_shard_{args.shard_id}_ids.json"
    faiss.write_index(merged_index, str(out_faiss))
    out_ids.write_text(json.dumps(merged_ids), encoding="utf-8")

    progress = {
        "processed": merged_index.ntotal,
        "shard_id": args.shard_id,
        "num_shards": 3,
        "merged_from_parts": args.num_parts,
    }
    (args.output_dir / f"kilt_shard_{args.shard_id}_progress.json").write_text(
        json.dumps(progress),
        encoding="utf-8",
    )
    print(f"Wrote {out_faiss} and {out_ids} ({merged_index.ntotal:,} vectors total)")


if __name__ == "__main__":
    main()
