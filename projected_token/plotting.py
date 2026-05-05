from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def plot_training_curves(
    history: list[dict[str, Any]],
    *,
    x_key: str,
    output_path: str | Path,
    title: str,
    metric_keys: list[str] | None = None,
) -> None:
    if not history:
        return
    target = _ensure_parent(output_path)
    keys = metric_keys or [k for k in history[0].keys() if k != x_key and isinstance(history[0].get(k), (int, float))]
    if not keys:
        return
    plt.figure(figsize=(10, 6))
    xs = [row[x_key] for row in history if x_key in row]
    for key in keys:
        ys = [row.get(key) for row in history]
        plt.plot(xs, ys, label=key)
    plt.xlabel(x_key)
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close()


def plot_metric_comparison(
    labels: list[str],
    values: list[float],
    *,
    output_path: str | Path,
    title: str,
    y_label: str,
) -> None:
    if not labels or not values or len(labels) != len(values):
        return
    target = _ensure_parent(output_path)
    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(target, dpi=160)
    plt.close()
