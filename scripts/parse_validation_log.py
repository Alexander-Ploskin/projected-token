#!/usr/bin/env python3
import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any


VAL_STEP_RE = re.compile(r"^\[val\]\s+step=(\d+)")
BEIR_STEP_RE = re.compile(r"^\[beir-probe\]\s+step=(\d+)\s+(.*)$")
EPOCH_RE = re.compile(r"Epoch\s+(\d+):")
TRAIN_STEP_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[(.*)\]\s*$")


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip(",")
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_kv_blob(blob: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for token in blob.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        metrics[key] = _parse_scalar(value)
    return metrics


def _parse_train_line(line: str) -> dict[str, Any] | None:
    if "Epoch " not in line or "loss=" not in line:
        return None

    epoch_match = EPOCH_RE.search(line)
    step_match = TRAIN_STEP_RE.search(line)
    if not epoch_match or not step_match:
        return None

    epoch = int(epoch_match.group(1))
    step_in_epoch = int(step_match.group(1))
    step_total = int(step_match.group(2))
    metrics_blob = step_match.group(3)

    # metrics are comma-separated inside tqdm's brackets
    metrics: dict[str, Any] = {}
    for token in metrics_blob.split(","):
        token = token.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        metrics[key.strip()] = _parse_scalar(value)

    return {
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "step_total": step_total,
        "metrics": metrics,
        "raw": line,
    }


def parse_log(log_path: Path, history_size: int) -> list[dict[str, Any]]:
    train_history: deque[dict[str, Any]] = deque(maxlen=max(history_size + 5, 10))
    val_records: list[dict[str, Any]] = []
    last_beir_probe_by_step: dict[int, dict[str, Any]] = {}
    current_val: dict[str, Any] | None = None
    last_train_signature: tuple[int, int] | None = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            train_step = _parse_train_line(line)
            if train_step is not None:
                signature = (train_step["epoch"], train_step["step_in_epoch"])
                if last_train_signature == signature and len(train_history) > 0:
                    train_history[-1] = train_step
                else:
                    train_history.append(train_step)
                    last_train_signature = signature
                continue

            beir_match = BEIR_STEP_RE.match(line)
            if beir_match:
                step = int(beir_match.group(1))
                last_beir_probe_by_step[step] = {
                    "step": step,
                    "metrics": _parse_kv_blob(beir_match.group(2)),
                    "raw": line,
                }
                continue

            val_match = VAL_STEP_RE.match(line)
            if val_match:
                if current_val is not None:
                    val_records.append(current_val)

                step = int(val_match.group(1))
                current_val = {
                    "val_step": step,
                    "train_steps_before_validation": list(train_history)[-history_size:],
                    "val_metrics": {},
                    "val_lines": [line],
                    "beir_probe": last_beir_probe_by_step.get(step),
                }
                continue

            if current_val is not None and line.startswith("[val] "):
                content = line[len("[val] ") :]
                group, _, tail = content.partition(" ")
                current_val["val_metrics"][group] = _parse_kv_blob(tail)
                current_val["val_lines"].append(line)
                continue

            if current_val is not None:
                val_records.append(current_val)
                current_val = None

    if current_val is not None:
        val_records.append(current_val)

    for idx, rec in enumerate(val_records, start=1):
        rec["validation_index"] = idx

    return val_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse training log and save validation metrics with the previous "
            "N training steps before each validation."
        )
    )
    parser.add_argument("log_path", type=Path, help="Path to training log file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--history-size",
        type=int,
        default=5,
        help="How many train steps to keep before each validation (default: 5)",
    )
    args = parser.parse_args()

    if not args.log_path.exists():
        raise FileNotFoundError(f"Log file not found: {args.log_path}")

    output = args.output
    if output is None:
        output = args.log_path.with_suffix(args.log_path.suffix + ".val_with_train5.json")

    records = parse_log(args.log_path, args.history_size)

    with output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(records)} validation blocks to: {output}")


if __name__ == "__main__":
    main()
