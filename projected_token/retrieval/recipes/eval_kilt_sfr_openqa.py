#!/usr/bin/env python3
"""Backward-compatible SFR wrapper around eval_kilt_openqa."""

from __future__ import annotations

from projected_token.retrieval.recipes.eval_kilt_openqa import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
