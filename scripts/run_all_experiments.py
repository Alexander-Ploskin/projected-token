#!/usr/bin/env python3
"""Run all experiments using CLI.

This is a wrapper script that uses the unified CLI (pt) to run all experiments.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("/home/jovyan/rpt/results")
# Full PopQA dataset for complete experiments
INPUT_FILE = Path("/home/jovyan/rpt/data/popqa_enriched.parquet")
# Test dataset for quick validation
TEST_INPUT_FILE = Path("/home/jovyan/rpt/data/popqa_test_100.parquet")
CONFIGS_DIR = Path("/home/jovyan/rpt/research-proj-token/configs")


def run_experiments(input_file: Path, output_dir: Path, full_run: bool = True):
    """Run all experiments via CLI."""
    mode = "FULL" if full_run else "TEST"
    print("=" * 60)
    print(f"RUNNING ALL EXPERIMENTS VIA CLI ({mode} MODE)")
    print("=" * 60)

    # Check if input file exists
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run all experiments via CLI
    cmd = [
        sys.executable, "-m", "evaluation.cli", "run-all",
        "--configs-dir", str(CONFIGS_DIR),
        "--input-path", str(input_file),
        "--output-dir", str(output_dir),
        "--gpt-base-url", "http://localhost:8000/v1",
        "--gpt-model", "Qwen/Qwen3.5-27B",
        "--batch-size", "5",
        "--gpt-batch-size", "10",
        "--verbose",
    ]

    print(f"\nInput file: {input_file}")
    print(f"Output dir: {output_dir}")
    print(f"\nRunning command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd="/home/jovyan/rpt/research-proj-token")

    return result.returncode


def main():
    """Run all experiments via CLI."""
    import argparse

    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--test", action="store_true", help="Run on test dataset only")
    parser.add_argument("--input", type=str, help="Custom input file path")
    parser.add_argument("--output", type=str, help="Custom output directory")

    args = parser.parse_args()

    # Determine input file and output directory
    if args.input:
        input_file = Path(args.input)
    elif args.test:
        input_file = TEST_INPUT_FILE
    else:
        input_file = INPUT_FILE

    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    return run_experiments(input_file, output_dir, full_run=not args.test)


if __name__ == "__main__":
    sys.exit(main())
