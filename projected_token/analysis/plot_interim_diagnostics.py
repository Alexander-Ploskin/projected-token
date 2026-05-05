from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def main() -> None:
    base = Path("artifacts/analysis/interim")
    base.mkdir(parents=True, exist_ok=True)

    contrastive_rows = _load_rows(base / "contrastive_validation_curve.csv")
    distill_rows = _load_rows(base / "distill_validation_curve.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if contrastive_rows:
        xs = [row["step"] for row in contrastive_rows]
        axes[0].plot(xs, [row["mrr"] for row in contrastive_rows], marker="o", label="MRR")
        axes[0].plot(xs, [row["recall@10"] for row in contrastive_rows], marker="o", label="Recall@10")
        axes[0].set_title("Contrastive Validation")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("score")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    if distill_rows:
        xs = [row["step"] for row in distill_rows]
        axes[1].plot(xs, [row["cosine_sim"] for row in distill_rows], marker="o", label="Cosine sim")
        axes[1].plot(xs, [row["mse_loss"] for row in distill_rows], marker="o", label="MSE loss")
        axes[1].set_title("Distillation Validation")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("value")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

    fig.tight_layout()
    output_path = base / "training_diagnostics.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
