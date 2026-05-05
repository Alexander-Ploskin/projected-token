#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compact many flat shard .faiss files into fewer larger files."
    )
    parser.add_argument("--shard-dir", required=True, help="Directory with s{shard}_{start}_{end}.faiss files.")
    parser.add_argument(
        "--files-per-merged",
        type=int,
        default=1000,
        help="How many source shard files to merge into one output shard file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be merged.",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Do not delete source files after successful merge.",
    )
    args = parser.parse_args()
    if int(args.files_per_merged) < 2:
        raise ValueError("--files-per-merged must be >= 2")
    return args


def scan_shards(shard_dir: Path) -> dict[int, list[tuple[int, int, Path]]]:
    pattern = re.compile(r"^s(?P<shard>\d+)_(?P<start>\d+)_(?P<end>\d+)\.faiss$")
    grouped: dict[int, list[tuple[int, int, Path]]] = {}
    for item in shard_dir.iterdir():
        if not item.is_file():
            continue
        match = pattern.match(item.name)
        if not match:
            continue
        shard_id = int(match.group("shard"))
        start = int(match.group("start"))
        end = int(match.group("end"))
        grouped.setdefault(shard_id, []).append((start, end, item))
    for shard_id in grouped:
        grouped[shard_id].sort(key=lambda row: (row[0], row[1]))
    return grouped


def merge_group(entries: list[tuple[int, int, Path]], output_path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    import faiss

    first = faiss.read_index(str(entries[0][2]))
    merged = first
    for _, _, src_path in entries[1:]:
        src = faiss.read_index(str(src_path))
        merged.merge_from(src, 0)
    tmp_path = output_path.with_suffix(".tmp.faiss")
    faiss.write_index(merged, str(tmp_path))
    tmp_path.replace(output_path)


def main() -> None:
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    grouped = scan_shards(shard_dir)
    if not grouped:
        print("No shard files found, nothing to compact.")
        return

    files_per_merged = int(args.files_per_merged)
    total_in = 0
    total_out = 0
    total_merged_groups = 0

    for shard_id in sorted(grouped):
        entries = grouped[shard_id]
        total_in += len(entries)
        idx = 0
        while idx < len(entries):
            group = entries[idx:idx + files_per_merged]
            if len(group) == 1:
                total_out += 1
                idx += files_per_merged
                continue

            start = group[0][0]
            end = group[-1][1]
            output_path = shard_dir / f"s{shard_id}_{start}_{end}.faiss"
            print(
                f"[shard {shard_id}] merge {len(group)} files "
                f"({group[0][2].name} .. {group[-1][2].name}) -> {output_path.name}"
            )
            merge_group(group, output_path, bool(args.dry_run))
            total_out += 1
            total_merged_groups += 1

            if not args.dry_run and not args.keep_source:
                for _, _, src_path in group:
                    if src_path != output_path:
                        src_path.unlink(missing_ok=True)

            idx += files_per_merged

    if total_merged_groups == 0:
        print("No groups large enough to merge; layout already compact enough.")
    print(
        "Compaction summary:",
        {
            "input_files": total_in,
            "output_files_estimate": total_out,
            "merged_groups": total_merged_groups,
            "files_per_merged": files_per_merged,
            "dry_run": bool(args.dry_run),
        },
    )


if __name__ == "__main__":
    main()
