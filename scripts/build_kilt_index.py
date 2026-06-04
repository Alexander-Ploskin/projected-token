#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import faiss

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.retrieval.encoders import build_encoder
from projected_token.retrieval.index.vector import l2_normalize


def shard_paths(
    out_dir: Path,
    shard_id: int,
    part_id: int = 0,
    num_parts: int = 1,
) -> dict[str, Path]:
    suffix = f"_part{part_id}" if num_parts > 1 else ""
    return {
        "faiss": out_dir / f"kilt_shard_{shard_id}{suffix}.faiss",
        "ids": out_dir / f"kilt_shard_{shard_id}{suffix}_ids.json",
        "progress": out_dir / f"kilt_shard_{shard_id}{suffix}_progress.json",
    }


def part_range(shard_len: int, part_id: int, num_parts: int) -> tuple[int, int]:
    if num_parts <= 1:
        return 0, shard_len
    part_size = (shard_len + num_parts - 1) // num_parts
    start = part_id * part_size
    end = min(start + part_size, shard_len)
    return start, end


def load_resume_state(paths: dict[str, Path]) -> tuple[faiss.IndexFlatIP, list[str], int]:
    progress_path = paths["progress"]
    faiss_path = paths["faiss"]
    ids_path = paths["ids"]

    if not progress_path.exists():
        return faiss.IndexFlatIP(768), [], 0

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    processed = int(progress.get("processed", 0))
    if processed <= 0:
        return faiss.IndexFlatIP(768), [], 0

    if not faiss_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"Progress checkpoint exists ({processed} docs) but index files are missing: "
            f"{faiss_path}, {ids_path}"
        )

    index = faiss.read_index(str(faiss_path))
    doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if index.ntotal != processed or len(doc_ids) != processed:
        raise ValueError(
            f"Checkpoint mismatch for shard: processed={processed}, "
            f"faiss={index.ntotal}, ids={len(doc_ids)}"
        )

    print(
        f"[KILT Indexer] Resuming from checkpoint: {processed} documents already encoded.",
        flush=True,
    )
    return index, doc_ids, processed


def save_checkpoint(
    paths: dict[str, Path],
    index: faiss.IndexFlatIP,
    doc_ids: list[str],
    processed: int,
    shard_id: int,
    num_shards: int,
    part_id: int = 0,
    num_parts: int = 1,
    range_start: int = 0,
    range_end: int = 0,
) -> None:
    paths["faiss"].parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(paths["faiss"]))
    with open(paths["ids"], "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)
    with open(paths["progress"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed": processed,
                "shard_id": shard_id,
                "num_shards": num_shards,
                "part_id": part_id,
                "num_parts": num_parts,
                "range_start": range_start,
                "range_end": range_end,
            },
            f,
        )


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for KILT using OSCAR E9")
    parser.add_argument("--output-dir", type=str, default="/mnt/raid/a-ploskin/kilt_faiss_indexes", help="Directory to save the index")
    parser.add_argument("--shard-id", type=int, required=True, help="Shard ID (0-indexed)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--compressor-device", type=str, default="cuda:0", help="Device for compressor and projector")
    parser.add_argument("--decoder-device", type=str, default="cuda:1", help="Device for decoder")
    parser.add_argument("--projector-path", type=str, default="artifacts/query_distill_runs/e11-pertoken-gated/checkpoints/best_model.pt", help="Path to projector checkpoint")
    parser.add_argument("--pooler", type=str, default=None, help="Pooler override (defaults to checkpoint config)")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500_000,
        help="Save partial FAISS checkpoint every N encoded documents (0 disables)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing partial checkpoints and start from scratch",
    )
    parser.add_argument("--part-id", type=int, default=0, help="Sub-part ID within the shard (0-indexed)")
    parser.add_argument(
        "--num-parts",
        type=int,
        default=1,
        help="Split the shard into this many contiguous sub-parts for parallel workers",
    )
    args = parser.parse_args()

    if args.part_id < 0 or args.num_parts <= 0 or args.part_id >= args.num_parts:
        raise ValueError(
            f"Invalid part split: part_id={args.part_id}, num_parts={args.num_parts}"
        )

    out_dir = Path(args.output_dir)
    print(f"[KILT Indexer] Output directory: {out_dir}", flush=True)
    part_label = (
        f"part {args.part_id}/{args.num_parts} of "
        if args.num_parts > 1
        else ""
    )
    print(
        f"[KILT Indexer] Processing {part_label}shard {args.shard_id} of {args.num_shards}",
        flush=True,
    )

    # Set env vars for OSCAR component placement
    os.environ["OSCAR_COMPRESSOR_DEVICE"] = args.compressor_device
    os.environ["OSCAR_DECODER_DEVICE"] = args.decoder_device

    print(f"Loading OSCAR model...")
    print(f"Compressor/Projector device: {args.compressor_device}")
    print(f"Decoder device: {args.decoder_device}")
    
    encoder_cfg = {
        "name": "oscar_projector",
        "kwargs": {
            "oscar_model_name": "naver/oscar-qwen2-7B",
            "projector_path": args.projector_path,
            "device": args.compressor_device,
            "embed_dim": 768,
            "pooler": args.pooler or "flatten",
            "num_layers": 2,
            "dropout": 0.0,
        }
    }
    encoder = build_encoder(encoder_cfg)

    print(f"[KILT Indexer] Loading s-nlp/kilt dataset (shard {args.shard_id}/{args.num_shards})...", flush=True)
    ds = load_dataset("s-nlp/kilt", split="train")
    ds = ds.shard(num_shards=args.num_shards, index=args.shard_id)
    
    range_start, range_end = part_range(len(ds), args.part_id, args.num_parts)
    part_len = range_end - range_start
    print(
        f"[KILT Indexer] Shard contains {len(ds)} documents; "
        f"this worker covers [{range_start}, {range_end}) = {part_len} documents.",
        flush=True,
    )

    paths = shard_paths(out_dir, args.shard_id, args.part_id, args.num_parts)
    if args.no_resume:
        index = faiss.IndexFlatIP(768)
        doc_ids: list[str] = []
        resume_local = 0
    else:
        index, doc_ids, resume_local = load_resume_state(paths)

    batch_texts = []
    batch_ids = []
    processed = resume_local
    next_checkpoint = (
        ((resume_local // args.checkpoint_every) + 1) * args.checkpoint_every
        if args.checkpoint_every > 0
        else None
    )

    def process_batch(texts, ids):
        nonlocal processed, next_checkpoint
        if not texts:
            return
        embeddings = encoder.encode(texts, None)
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        embeddings = l2_normalize(embeddings)
        index.add(embeddings)
        doc_ids.extend(ids)
        processed += len(ids)

        if (
            args.checkpoint_every > 0
            and next_checkpoint is not None
            and processed >= next_checkpoint
        ):
            save_checkpoint(
                paths,
                index,
                doc_ids,
                processed,
                args.shard_id,
                args.num_shards,
                args.part_id,
                args.num_parts,
                range_start,
                range_end,
            )
            print(
                f"[KILT Indexer] Saved checkpoint at {processed}/{part_len} documents.",
                flush=True,
            )
            while next_checkpoint is not None and processed >= next_checkpoint:
                next_checkpoint += args.checkpoint_every

    iter_start = range_start + resume_local
    iterator = range(iter_start, range_end)
    if resume_local:
        iterator = tqdm(iterator, desc="Encoding", initial=resume_local, total=part_len)
    else:
        iterator = tqdm(iterator, desc="Encoding", total=part_len)

    for i in iterator:
        row = ds[i]
        title = str(row.get("wikipedia_title", "")).strip()
        text = str(row.get("text", "")).strip()
        doc_id = str(row.get("_id", ""))

        full_text = f"{title} {text}".strip()
        batch_texts.append(full_text)
        batch_ids.append(doc_id)

        if len(batch_texts) >= args.batch_size:
            process_batch(batch_texts, batch_ids)
            batch_texts = []
            batch_ids = []

    if batch_texts:
        process_batch(batch_texts, batch_ids)

    print(f"Saving FAISS index to {paths['faiss']}...")
    save_checkpoint(
        paths,
        index,
        doc_ids,
        processed,
        args.shard_id,
        args.num_shards,
        args.part_id,
        args.num_parts,
        range_start,
        range_end,
    )

    print("Done!")

if __name__ == "__main__":
    main()
