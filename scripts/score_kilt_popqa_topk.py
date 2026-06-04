#!/usr/bin/env python3
"""Backward-compatible wrapper: score KILT PopQA top-k JSONL."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    argv = list(sys.argv)
    if "--dataset" not in argv and "-h" not in argv and "--help" not in argv:
        argv = [argv[0], "--dataset", "popqa", *argv[1:]]
    if "--popqa-path" in argv:
        idx = argv.index("--popqa-path")
        argv[idx] = "--queries-path"
    sys.argv = argv
    runpy.run_path(str(Path(__file__).resolve().parent / "score_kilt_openqa_topk.py"), run_name="__main__")
