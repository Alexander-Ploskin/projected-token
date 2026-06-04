#!/usr/bin/env python3
"""Plot training (TensorBoard) and validation curves for a query_distill run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_VAL_HISTORY_CANDIDATES = (
    "metrics/query_distill_val_history.json",
    "checkpoints/val_metrics_history.json",
)


def _dedupe_val_by_step(rows: list[dict[str, Any]], *, max_step: int | None) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        if "step" not in row:
            continue
        by_step[int(row["step"])] = row
    out = [by_step[s] for s in sorted(by_step)]
    if max_step is not None:
        out = [r for r in out if int(r["step"]) <= max_step]
    return out


def _load_val_history(
    run_dir: Path,
    *,
    prefer_metrics: bool = True,
    max_step: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    order = list(_VAL_HISTORY_CANDIDATES)
    if not prefer_metrics:
        order = list(reversed(order))
    for rel in order:
        path = run_dir / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        history = _dedupe_val_by_step(payload, max_step=max_step)
        return history, rel
    raise FileNotFoundError(f"No validation history JSON found under {run_dir}")


def _trim_tb_resume_segment(steps: np.ndarray, values: np.ndarray, *, dedupe_resume: bool) -> tuple[np.ndarray, np.ndarray]:
    if not dedupe_resume or steps.size <= 1:
        return steps, values
    restart_at = 0
    for i in range(1, steps.size):
        if steps[i] < steps[i - 1]:
            restart_at = i
    return steps[restart_at:], values[restart_at:]


def _dedupe_tb_duplicate_steps(steps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if steps.size == 0:
        return steps, values
    by_step: dict[int, float] = {}
    for s, v in zip(steps.tolist(), values.tolist()):
        by_step[int(s)] = float(v)
    keys = sorted(by_step)
    return np.array(keys, dtype=np.int64), np.array([by_step[k] for k in keys], dtype=np.float64)


def _load_tb_scalars(
    tb_dir: Path,
    tag: str,
    *,
    dedupe_resume: bool = True,
    max_step: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    steps = np.array([int(e.step) for e in events], dtype=np.int64)
    values = np.array([float(e.value) for e in events], dtype=np.float64)
    steps, values = _trim_tb_resume_segment(steps, values, dedupe_resume=dedupe_resume)
    steps, values = _dedupe_tb_duplicate_steps(steps, values)
    if max_step is not None and steps.size:
        mask = steps <= max_step
        steps, values = steps[mask], values[mask]
    return steps, values


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _add_stage_marker(ax: plt.Axes, stage1_steps: int, *, label: bool = True) -> None:
    if stage1_steps <= 0:
        return
    ax.axvline(stage1_steps, color="#c44e52", linestyle="--", linewidth=1.2, alpha=0.85)
    if label:
        ax.text(
            stage1_steps,
            0.98,
            "stage 2",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=9,
            color="#c44e52",
        )


def _plot_panel(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    ylabel: str,
    stage1_steps: int,
    logy: bool = False,
) -> plt.Figure:
    n = len(series)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (name, (xs, ys)) in zip(axes_flat, series.items()):
        if xs.size == 0:
            ax.set_visible(False)
            continue
        ax.plot(xs, ys, linewidth=1.0, alpha=0.9)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        _add_stage_marker(ax, stage1_steps, label=False)
        if logy:
            ax.set_yscale("log")
    for ax in axes_flat[len(series) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def _val_series(history: list[dict[str, Any]], keys: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not history:
        return {}
    steps = np.array([int(row["step"]) for row in history], dtype=np.int64)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key in keys:
        if key not in history[0]:
            continue
        values = np.array([float(row[key]) for row in history if key in row], dtype=np.float64)
        if values.size == len(steps):
            out[key] = (steps, values)
    return out


def _style_ax(ax: plt.Axes, *, title: str, stage1_steps: int, ylabel: str = "value") -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    _add_stage_marker(ax, stage1_steps, label=False)


def _plot_line(
    ax: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    label: str | None = None,
    marker: str | None = None,
    linewidth: float = 1.0,
    alpha: float = 0.9,
    color: str | None = None,
) -> None:
    if xs.size == 0:
        return
    ax.plot(xs, ys, linewidth=linewidth, alpha=alpha, label=label, marker=marker, color=color)


def _plot_summary_1x4(
    *,
    run_label: str,
    stage1_steps: int,
    out_dir: Path,
    train_loss_total: tuple[np.ndarray, np.ndarray],
    val_loss_total: tuple[np.ndarray, np.ndarray],
    val_sims: dict[str, tuple[np.ndarray, np.ndarray]],
    beir_mrr: tuple[np.ndarray, np.ndarray],
) -> list[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    xs, ys = train_loss_total
    _plot_line(axes[0], xs, ys, linewidth=0.8, alpha=0.55)
    _style_ax(axes[0], title="loss_total (train)", stage1_steps=stage1_steps)

    xs, ys = val_loss_total
    _plot_line(axes[1], xs, ys, marker="o", linewidth=1.5)
    _style_ax(axes[1], title="loss_total (val)", stage1_steps=stage1_steps)

    colors = {"pos_sim": "#1f77b4", "neg_sim": "#ff7f0e", "gap": "#2ca02c"}
    sim_ax = axes[2]
    gap_ax = sim_ax.twinx()
    for key in ("pos_sim", "neg_sim"):
        if key not in val_sims:
            continue
        xs, ys = val_sims[key]
        _plot_line(
            sim_ax,
            xs,
            ys,
            marker="o",
            linewidth=1.2,
            label=key,
            alpha=0.85,
            color=colors[key],
        )
    if "gap" in val_sims:
        xs, ys = val_sims["gap"]
        _plot_line(
            gap_ax,
            xs,
            ys,
            marker="o",
            linewidth=1.2,
            label="gap",
            alpha=0.85,
            color=colors["gap"],
        )
        gap_ax.set_ylim(0.0, 0.2)
        gap_ax.set_ylabel("gap")
    _style_ax(sim_ax, title="pos_sim / neg_sim / gap (val)", stage1_steps=stage1_steps)
    sim_lines, sim_labels = sim_ax.get_legend_handles_labels()
    gap_lines, gap_labels = gap_ax.get_legend_handles_labels()
    sim_ax.legend(sim_lines + gap_lines, sim_labels + gap_labels, fontsize=8)

    xs, ys = beir_mrr
    _plot_line(axes[3], xs, ys, marker="s", linewidth=1.5, color=colors["neg_sim"])
    _style_ax(axes[3], title="beir_mrr@10", stage1_steps=stage1_steps, ylabel="score")

    fig.suptitle(f"{run_label} — summary", fontsize=13, y=1.02)
    fig.tight_layout()
    stem = out_dir / "summary_1x4"
    _save_figure(fig, stem)
    return [stem.with_suffix(".png"), stem.with_suffix(".pdf")]


def _plot_grid_4x4(
    *,
    run_label: str,
    stage1_steps: int,
    out_dir: Path,
    panels: list[tuple[str, np.ndarray, np.ndarray, bool]],
) -> list[Path]:
    fig, axes = plt.subplots(4, 4, figsize=(21, 14), squeeze=False)
    for ax, (title, xs, ys, use_marker) in zip(axes.ravel(), panels):
        _plot_line(
            ax,
            xs,
            ys,
            marker="o" if use_marker else None,
            linewidth=1.5 if use_marker else 0.8,
            alpha=0.9 if use_marker else 0.55,
        )
        _style_ax(ax, title=title, stage1_steps=stage1_steps)
        if "gap" in title.lower():
            ax.set_ylim(0.0, 0.2)
    fig.suptitle(f"{run_label} — all metrics", fontsize=13, y=1.01)
    fig.tight_layout()
    stem = out_dir / "grid_4x4"
    _save_figure(fig, stem)
    return [stem.with_suffix(".png"), stem.with_suffix(".pdf")]


def plot_run(
    run_dir: Path,
    *,
    run_label: str,
    stage1_steps: int,
    output_subdir: str = "training_validation",
    output_dir: Path | None = None,
    prefer_metrics_val: bool = True,
    dedupe_resume: bool = True,
    max_step: int | None = None,
) -> list[Path]:
    run_dir = run_dir.resolve()
    out_dir = output_dir or (run_dir / "plots" / output_subdir)
    val_history, val_source = _load_val_history(
        run_dir,
        prefer_metrics=prefer_metrics_val,
        max_step=max_step,
    )
    tb_dir = run_dir / "logs" / "tensorboard"
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard log directory not found: {tb_dir}")

    tb_kw = {"dedupe_resume": dedupe_resume, "max_step": max_step}
    saved: list[Path] = []

    train_loss = {
        "loss_total": _load_tb_scalars(tb_dir, "train/loss_total", **tb_kw),
        "query_mse": _load_tb_scalars(tb_dir, "train/query_mse", **tb_kw),
        "positive_mse": _load_tb_scalars(tb_dir, "train/positive_mse", **tb_kw),
        "ranking_loss": _load_tb_scalars(tb_dir, "train/ranking_loss", **tb_kw),
    }
    fig = _plot_panel(
        train_loss,
        title=f"{run_label} — training losses",
        ylabel="value",
        stage1_steps=stage1_steps,
    )
    stem = out_dir / "training_loss_components"
    _save_figure(fig, stem)
    saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    train_distill = {
        "teacher_listwise_kl": _load_tb_scalars(tb_dir, "train/teacher_listwise_kl", **tb_kw),
        "infonce_loss": _load_tb_scalars(tb_dir, "train/infonce_loss", **tb_kw),
        "gap (pos - hardest neg)": _load_tb_scalars(tb_dir, "train/gap", **tb_kw),
        "teacher_top1_agreement": _load_tb_scalars(tb_dir, "train/teacher_top1_agreement", **tb_kw),
    }
    fig = _plot_panel(
        train_distill,
        title=f"{run_label} — training distillation signals",
        ylabel="value",
        stage1_steps=stage1_steps,
    )
    stem = out_dir / "training_distillation_signals"
    _save_figure(fig, stem)
    saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    train_opt = {
        "learning_rate": _load_tb_scalars(tb_dir, "train/lr", **tb_kw),
        "grad_norm": _load_tb_scalars(tb_dir, "train/grad_norm", **tb_kw),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, (xs, ys)) in zip(axes, train_opt.items()):
        ax.plot(xs, ys, linewidth=1.0)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        _add_stage_marker(ax, stage1_steps)
    fig.suptitle(f"{run_label} — optimizer", fontsize=13)
    fig.tight_layout()
    stem = out_dir / "training_optimizer"
    _save_figure(fig, stem)
    saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    val_keys = [
        "loss_total",
        "query_mse",
        "positive_mse",
        "ranking_loss",
        "teacher_listwise_kl",
        "infonce_loss",
    ]
    fig = _plot_panel(
        _val_series(val_history, val_keys),
        title=f"{run_label} — validation losses",
        ylabel="value",
        stage1_steps=stage1_steps,
    )
    stem = out_dir / "validation_loss_components"
    _save_figure(fig, stem)
    saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    val_retrieval = _val_series(
        val_history,
        ["pos_sim", "neg_sim", "gap", "teacher_top1_agreement", "teacher_student_spearman"],
    )
    if val_retrieval:
        fig = _plot_panel(
            val_retrieval,
            title=f"{run_label} — validation retrieval diagnostics",
            ylabel="value",
            stage1_steps=stage1_steps,
        )
        stem = out_dir / "validation_retrieval_diagnostics"
        _save_figure(fig, stem)
        saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    probe_rows = [r for r in val_history if "beir_probe_ndcg@10" in r]
    if probe_rows:
        steps = np.array([int(r["step"]) for r in probe_rows])
        ndcg = np.array([float(r["beir_probe_ndcg@10"]) for r in probe_rows])
        mrr = np.array([float(r.get("beir_probe_mrr@10", np.nan)) for r in probe_rows])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(steps, ndcg, marker="o", label="ndcg@10")
        if np.isfinite(mrr).any():
            ax.plot(steps, mrr, marker="s", label="mrr@10")
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.set_title(f"{run_label} — BEIR probe (in-validation)")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        _add_stage_marker(ax, stage1_steps)
        fig.tight_layout()
        stem = out_dir / "validation_beir_probe"
        _save_figure(fig, stem)
        saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    full_rows = [r for r in val_history if r.get("beir_full_ndcg@10") is not None]
    if full_rows:
        steps = [int(r["step"]) for r in full_rows]
        ndcg = [float(r["beir_full_ndcg@10"]) for r in full_rows]
        mrr = [float(r.get("beir_full_mrr@10", np.nan)) for r in full_rows]
        x = np.arange(len(steps))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.35
        ax.bar(x - width / 2, ndcg, width=width, label="ndcg@10")
        if np.isfinite(mrr).all():
            ax.bar(x + width / 2, mrr, width=width, label="mrr@10")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in steps], rotation=20, ha="right")
        ax.set_ylabel("score")
        ax.set_title(f"{run_label} — BEIR full eval (epoch end)")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        stem = out_dir / "validation_beir_full_epoch_end"
        _save_figure(fig, stem)
        saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    train_steps, train_loss_total = _load_tb_scalars(tb_dir, "train/loss_total", **tb_kw)
    val_steps, val_loss = _val_series(val_history, ["loss_total"]).get("loss_total", (np.array([]), np.array([])))
    if train_steps.size and val_steps.size:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(train_steps, train_loss_total, linewidth=0.8, alpha=0.55, label="train (per step)")
        ax.plot(val_steps, val_loss, marker="o", linewidth=1.5, label="validation")
        ax.set_xlabel("step")
        ax.set_ylabel("loss_total")
        ax.set_title(f"{run_label} — train vs validation total loss")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        _add_stage_marker(ax, stage1_steps)
        fig.tight_layout()
        stem = out_dir / "train_vs_validation_loss_total"
        _save_figure(fig, stem)
        saved.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])

    val_retrieval_series = _val_series(val_history, ["pos_sim", "neg_sim", "gap"])
    beir_mrr = _val_series(val_history, ["beir_probe_mrr@10"]).get(
        "beir_probe_mrr@10", (np.array([]), np.array([]))
    )
    saved.extend(
        _plot_summary_1x4(
            run_label=run_label,
            stage1_steps=stage1_steps,
            out_dir=out_dir,
            train_loss_total=_load_tb_scalars(tb_dir, "train/loss_total", **tb_kw),
            val_loss_total=_val_series(val_history, ["loss_total"]).get(
                "loss_total", (np.array([]), np.array([]))
            ),
            val_sims=val_retrieval_series,
            beir_mrr=beir_mrr,
        )
    )

    grid_specs: list[tuple[str, str, str, bool]] = [
        ("loss_total (train)", "train", "train/loss_total", False),
        ("query_mse (train)", "train", "train/query_mse", False),
        ("positive_mse (train)", "train", "train/positive_mse", False),
        ("ranking_loss (train)", "train", "train/ranking_loss", False),
        ("infonce_loss (train)", "train", "train/infonce_loss", False),
        ("gap (train)", "train", "train/gap", False),
        ("lr (train)", "train", "train/lr", False),
        ("grad_norm (train)", "train", "train/grad_norm", False),
        ("loss_total (val)", "val", "loss_total", True),
        ("query_mse (val)", "val", "query_mse", True),
        ("positive_mse (val)", "val", "positive_mse", True),
        ("ranking_loss (val)", "val", "ranking_loss", True),
        ("infonce_loss (val)", "val", "infonce_loss", True),
        ("pos_sim (val)", "val", "pos_sim", True),
        ("neg_sim (val)", "val", "neg_sim", True),
        ("gap (val)", "val", "gap", True),
    ]
    val_all = _val_series(
        val_history,
        ["loss_total", "query_mse", "positive_mse", "ranking_loss", "infonce_loss", "pos_sim", "neg_sim", "gap"],
    )
    panels: list[tuple[str, np.ndarray, np.ndarray, bool]] = []
    for title, source, key, use_marker in grid_specs:
        if source == "train":
            xs, ys = _load_tb_scalars(tb_dir, key, **tb_kw)
        else:
            xs, ys = val_all.get(key, (np.array([]), np.array([])))
        panels.append((title, xs, ys, use_marker))
    saved.extend(
        _plot_grid_4x4(
            run_label=run_label,
            stage1_steps=stage1_steps,
            out_dir=out_dir,
            panels=panels,
        )
    )

    manifest = {
        "run_dir": str(run_dir),
        "run_label": run_label,
        "stage1_steps": stage1_steps,
        "val_source": val_source,
        "dedupe_resume": dedupe_resume,
        "max_step": max_step,
        "n_val_points": len(val_history),
        "n_train_steps": int(train_steps[-1]) if train_steps.size else 0,
        "outputs": [str(p) for p in saved],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    saved.append(manifest_path)
    return saved


def _infer_stage1_steps(run_dir: Path) -> int:
    for rel in ("e11_pertoken_gated.yaml", "config.lock.yaml"):
        path = run_dir / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^stage1_steps:\s*(\d+)\s*$", text, re.MULTILINE)
        if match:
            return int(match.group(1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/query_distill_runs/e11-pertoken-gated"),
        help="Query distillation run directory",
    )
    parser.add_argument("--label", default=None, help="Plot title label (default: run dir name)")
    parser.add_argument("--stage1-steps", type=int, default=None, help="Vertical marker for stage-2 start")
    parser.add_argument("--output-subdir", default="training_validation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures (default: <run-dir>/plots/<output-subdir>)",
    )
    parser.add_argument(
        "--prefer-checkpoints-val",
        action="store_true",
        help="Use checkpoints/val_metrics_history.json instead of metrics/query_distill_val_history.json",
    )
    parser.add_argument(
        "--no-dedupe-resume",
        action="store_true",
        help="Do not drop earlier TensorBoard segments after resume restarts",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Drop validation points and clip plots above this global step",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    label = args.label or run_dir.name
    stage1_steps = args.stage1_steps if args.stage1_steps is not None else _infer_stage1_steps(run_dir)
    out_dir = args.output_dir or (run_dir / "plots" / args.output_subdir)

    saved = plot_run(
        run_dir,
        run_label=label,
        stage1_steps=stage1_steps,
        output_subdir=args.output_subdir,
        output_dir=out_dir,
        prefer_metrics_val=not args.prefer_checkpoints_val,
        dedupe_resume=not args.no_dedupe_resume,
        max_step=args.max_step,
    )
    print(f"Wrote {len(saved)} files under {out_dir}")
    for path in saved:
        if path.suffix in {".png", ".pdf"}:
            print(f"  {path}")


if __name__ == "__main__":
    main()
