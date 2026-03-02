#!/usr/bin/env python3
"""Run all experiments from scratch for all models."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml

# Configuration
OUTPUT_DIR = Path("/home/jovyan/rpt/results")
INPUT_FILE = Path("/home/jovyan/rpt/results/simple_qwen_paraphrase_test.jsonl")

# Models to test
MODELS = [
    {
        "name": "qwen2_1_5b",
        "class": "evaluation.models.SimpleLLM",
        "kwargs": {
            "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
            "device": "cuda:0",
            "use_chat_template": True,
        },
        "generation": {"do_sample": True, "temperature": 0.3, "max_new_tokens": 512},
    },
    {
        "name": "rag_qwen2_7b",
        "class": "evaluation.models.SimpleLLM",
        "kwargs": {
            "model_name_or_path": "Qwen/Qwen2-7B-Instruct",
            "device": "cuda:0",
            "use_chat_template": True,
        },
        "generation": {"do_sample": True, "temperature": 0.3, "max_new_tokens": 512},
    },
    {
        "name": "oscar_7b",
        "class": "evaluation.models.OscarModel",
        "kwargs": {
            "model_name_or_path": "naver/oscar-qwen2-7B",
            "device": "cuda:0",
            "trust_remote_code": True,
        },
        "generation": {"max_new_tokens": 512},
    },
    {
        "name": "pisco_mistral",
        "class": "evaluation.models.PiscoModel",
        "kwargs": {
            "model_name_or_path": "naver/pisco-mistral",
            "device": "cuda:0",
            "trust_remote_code": True,
        },
        "generation": {"max_new_tokens": 512},
    },
]


def create_config(model: Dict, output_name: str) -> Path:
    """Create a config file for a model."""
    config = {
        "dataset": {"class": "evaluation.datasets.PopqaDataset"},
        "model": {"class": model["class"], "kwargs": model["kwargs"]},
        "generation": model["generation"],
        "experiment": {"sample_count": 1},
    }

    config_path = Path(f"/home/jovyan/rpt/research-proj-token/config_{output_name}.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    return config_path


def run_paraphrase(config_path: Path, output_path: Path, model_name: str) -> bool:
    """Run paraphrase generation for a model."""
    print(f"\n{'='*60}")
    print(f"Running {model_name} paraphrase...")
    print(f"{'='*60}")

    cmd = [
        "python3",
        "-m",
        "evaluation.cli",
        "paraphrase",
        "--config",
        str(config_path),
        "--input-path",
        str(INPUT_FILE),
        "--output-path",
        str(output_path),
        "--text-col",
        "s_wiki_content",
        "--batch-size",
        "5",
        "-v",
    ]

    result = subprocess.run(cmd, cwd="/home/jovyan/rpt/research-proj-token")
    return result.returncode == 0


def run_gpt_judge(input_path: Path, output_path: Path, model_name: str) -> bool:
    """Run GPT Judge evaluation."""
    print(f"\n{'='*60}")
    print(f"Running GPT Judge for {model_name}...")
    print(f"{'='*60}")

    cmd = [
        "python3",
        "evaluate_gpt_judge_only.py",
        str(input_path),
        "--output",
        str(output_path),
        "--base-url",
        "http://localhost:8000/v1",
        "--api-key",
        "dummy",
        "--model",
        "Qwen/Qwen3.5-27B",
        "--batch-size",
        "10",
    ]

    result = subprocess.run(cmd, cwd="/home/jovyan/rpt/research-proj-token")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("RE-RUNNING ALL EXPERIMENTS FROM SCRATCH")
    print("=" * 60)

    results = {}

    for model in MODELS:
        model_name = model["name"]
        print(f"\n{'#'*60}")
        print(f"# Processing: {model_name}")
        print(f"{'#'*60}")

        # Create config
        config_path = create_config(model, model_name)
        print(f"Created config: {config_path}")

        # Run paraphrase
        paraphrase_output = OUTPUT_DIR / f"{model_name}_paraphrase_test.jsonl"
        success = run_paraphrase(config_path, paraphrase_output, model_name)

        if not success:
            print(f"ERROR: Paraphrase failed for {model_name}")
            results[model_name] = {"paraphrase": False, "gpt_judge": False}
            continue

        print(f"Paraphrase output: {paraphrase_output}")

        # Run GPT Judge
        metrics_output = OUTPUT_DIR / f"{model_name}_metrics.json"
        success = run_gpt_judge(paraphrase_output, metrics_output, model_name)

        if not success:
            print(f"ERROR: GPT Judge failed for {model_name}")
            results[model_name] = {"paraphrase": True, "gpt_judge": False}
            continue

        print(f"Metrics output: {metrics_output}")
        results[model_name] = {"paraphrase": True, "gpt_judge": True}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for model_name, status in results.items():
        paraphrase_status = "✅" if status["paraphrase"] else "❌"
        gpt_status = "✅" if status["gpt_judge"] else "❌"
        print(f"{model_name}: Paraphrase {paraphrase_status}, GPT Judge {gpt_status}")

    all_success = all(v["paraphrase"] and v["gpt_judge"] for v in results.values())
    print(f"\nOverall: {'✅ ALL SUCCESS' if all_success else '❌ SOME FAILED'}")

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
