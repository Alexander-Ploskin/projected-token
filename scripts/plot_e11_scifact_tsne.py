#!/usr/bin/env python3
"""t-SNE for query vs document pools (unpaired by default): BGE teacher vs E11 checkpoints.

Paired train samples make t-SNE look artificially good — use --sampling-mode unpaired (default).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from projected_token.encoders.bge import BGEEncoder
from projected_token.encoders.oscar import OscarProjectorEncoder

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "font.family": "sans-serif",
        "legend.fontsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)
QUERY_COLOR = "#1f77b4"
DOC_COLOR = "#d62728"
NEG_COLOR = "#7f7f7f"
QUERY_MARKER = "o"
DOC_MARKER = "^"
NEG_MARKER = "x"


def _decode_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sample_from_h5(
    path: Path,
    sample_size: int,
    seed: int,
    *,
    mode: str,
    include_negatives: bool,
) -> dict[str, Any]:
    """Sample texts (+ teacher emb slices) for t-SNE.

    unpaired: independent query / positive-doc / negative pools (honest retrieval view).
    paired: aligned train pairs (can look deceptively good — for ablation only).
    """
    with h5py.File(path, "r") as h5:
        n = int(h5["queries"].shape[0])
        rng = np.random.default_rng(seed)
        k = min(sample_size, n)

        if mode == "paired":
            idx = np.sort(rng.choice(n, size=k, replace=False))
            q_idx = d_idx = n_idx = idx
        else:
            q_idx = np.sort(rng.choice(n, size=k, replace=False))
            d_idx = np.sort(rng.choice(n, size=k, replace=False))
            n_idx = np.sort(rng.choice(n, size=k, replace=False))

        def rows(name: str, indices: np.ndarray) -> list[str]:
            return [_decode_str(h5[name][int(i)]) for i in indices]

        out: dict[str, Any] = {
            "mode": mode,
            "n": k,
            "queries": rows("queries", q_idx),
            "positives": rows("positive_docs", d_idx),
            "negatives": rows("negative_docs", n_idx) if include_negatives else [],
            "query_teacher": np.asarray(h5["query_embeddings"][q_idx], dtype=np.float32),
            "positive_teacher": np.asarray(h5["positive_embeddings"][d_idx], dtype=np.float32),
        }
        if include_negatives:
            out["negative_teacher"] = np.asarray(h5["negative_embeddings"][n_idx], dtype=np.float32)
        if mode == "paired":
            out["pair_cos_teacher"] = np.sum(out["query_teacher"] * out["positive_teacher"], axis=1)
    return out


def _encode_normalized(encoder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        emb = encoder.encode(batch, None)
        if hasattr(emb, "detach"):
            emb = F.normalize(emb.float(), dim=-1).detach().cpu().numpy()
        else:
            emb = F.normalize(torch.tensor(emb).float(), dim=-1).numpy()
        chunks.append(np.asarray(emb, dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _encode_bge_split(bge: BGEEncoder, queries: list[str], docs: list[str], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """BGE uses different prefixes for queries vs documents."""
    q_chunks, d_chunks = [], []
    for start in range(0, len(queries), batch_size):
        qb = queries[start : start + batch_size]
        qe = bge.encode([""] * len(qb), qb)
        q_chunks.append(F.normalize(qe.float(), dim=-1).detach().cpu().numpy())
    for start in range(0, len(docs), batch_size):
        db = docs[start : start + batch_size]
        de = bge.encode(db, None)
        d_chunks.append(F.normalize(de.float(), dim=-1).detach().cpu().numpy())
    q_emb = np.concatenate(q_chunks, axis=0) if q_chunks else np.zeros((0, 768), dtype=np.float32)
    d_emb = np.concatenate(d_chunks, axis=0) if d_chunks else np.zeros((0, 768), dtype=np.float32)
    return q_emb.astype(np.float32), d_emb.astype(np.float32)


def fit_tsne_2d(stack: np.ndarray, seed: int, perplexity: float, max_iter: int) -> np.ndarray:
    n = stack.shape[0]
    perp = max(5.0, min(perplexity, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        perplexity=perp,
        learning_rate="auto",
        max_iter=max_iter,
    )
    return tsne.fit_transform(stack)


def _cosine_stats(q: np.ndarray, p: np.ndarray, n: np.ndarray | None) -> dict[str, float]:
    # Random query–doc pairs (min length across pools)
    m = min(len(q), len(p))
    qp = np.sum(q[:m] * p[:m], axis=1) if m else np.array([])
    stats = {
        "cos_query_doc_mean": float(np.mean(qp)) if len(qp) else float("nan"),
        "cos_query_doc_std": float(np.std(qp)) if len(qp) else float("nan"),
    }
    if n is not None and len(n):
        mn = min(len(q), len(n))
        qn = np.sum(q[:mn] * n[:mn], axis=1)
        stats["cos_query_neg_mean"] = float(np.mean(qn))
        stats["cos_query_neg_std"] = float(np.std(qn))
    return stats


def plot_tsne_panel(
    emb_q: np.ndarray,
    emb_p: np.ndarray,
    emb_n: np.ndarray | None,
    *,
    title: str,
    subtitle: str,
    out_path: Path,
    seed: int,
    perplexity: float,
    max_iter: int,
    pair_lines: bool,
) -> Path:
    parts = [emb_q, emb_p]
    labels: list[str] = ["query"] * len(emb_q) + ["doc"] * len(emb_p)
    if emb_n is not None and len(emb_n):
        parts.append(emb_n)
        labels.extend(["neg"] * len(emb_n))
    stack = np.concatenate(parts, axis=0)
    labels_arr = np.array(labels)
    emb_2d = fit_tsne_2d(stack, seed=seed, perplexity=perplexity, max_iter=max_iter)

    nq, np_ = len(emb_q), len(emb_p)
    q2d = emb_2d[:nq]
    p2d = emb_2d[nq : nq + np_]
    n2d = emb_2d[nq + np_ :] if emb_n is not None and len(emb_n) else None

    fig, ax = plt.subplots(figsize=(8.5, 7))
    if pair_lines and nq == np_:
        for i in range(nq):
            ax.plot([q2d[i, 0], p2d[i, 0]], [q2d[i, 1], p2d[i, 1]], c="#bbbbbb", alpha=0.35, linewidth=0.8, zorder=1)

    ax.scatter(q2d[:, 0], q2d[:, 1], c=QUERY_COLOR, marker=QUERY_MARKER, s=42, alpha=0.75, edgecolors="white", linewidths=0.4, label="Query", zorder=2)
    ax.scatter(p2d[:, 0], p2d[:, 1], c=DOC_COLOR, marker=DOC_MARKER, s=52, alpha=0.75, edgecolors="white", linewidths=0.4, label="Document", zorder=2)
    if n2d is not None:
        ax.scatter(n2d[:, 0], n2d[:, 1], c=NEG_COLOR, marker=NEG_MARKER, s=36, alpha=0.55, label="Hard negative", zorder=2)

    ax.set_title(title, fontweight="semibold", pad=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="best", framealpha=0.92)
    fig.text(0.5, 0.02, subtitle, ha="center", fontsize=9.5, color="#444444")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".png")


def plot_combined_grid(
    panels: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None, bool]],
    *,
    out_path: Path,
    seed: int,
    perplexity: float,
    max_iter: int,
    dataset_label: str,
    sampling_note: str,
) -> None:
    ncols = 2
    nrows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6.5 * nrows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (title, subtitle, emb_q, emb_p, emb_n, pair_lines) in zip(axes_flat, panels):
        parts = [emb_q, emb_p]
        if emb_n is not None and len(emb_n):
            parts.append(emb_n)
        stack = np.concatenate(parts, axis=0)
        emb_2d = fit_tsne_2d(stack, seed=seed, perplexity=perplexity, max_iter=max_iter)
        nq, np_ = len(emb_q), len(emb_p)
        q2d = emb_2d[:nq]
        p2d = emb_2d[nq : nq + np_]
        off = nq + np_
        n2d = emb_2d[off:] if emb_n is not None and len(emb_n) else None
        if pair_lines and nq == np_:
            for i in range(nq):
                ax.plot([q2d[i, 0], p2d[i, 0]], [q2d[i, 1], p2d[i, 1]], c="#bbbbbb", alpha=0.35, linewidth=0.6, zorder=1)
        ax.scatter(q2d[:, 0], q2d[:, 1], c=QUERY_COLOR, marker=QUERY_MARKER, s=28, alpha=0.72, edgecolors="white", linewidths=0.3)
        ax.scatter(p2d[:, 0], p2d[:, 1], c=DOC_COLOR, marker=DOC_MARKER, s=34, alpha=0.72, edgecolors="white", linewidths=0.3)
        if n2d is not None:
            ax.scatter(n2d[:, 0], n2d[:, 1], c=NEG_COLOR, marker=NEG_MARKER, s=22, alpha=0.55)
        ax.set_title(title, fontweight="semibold", fontsize=12)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.grid(True, alpha=0.22, linestyle="--")
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=8, color="#555555")

    for ax in axes_flat[len(panels) :]:
        ax.axis("off")

    legend_elems = [
        Line2D([0], [0], marker=QUERY_MARKER, color="w", markerfacecolor=QUERY_COLOR, markersize=9, label="Query"),
        Line2D([0], [0], marker=DOC_MARKER, color="w", markerfacecolor=DOC_COLOR, markersize=9, label="Document"),
        Line2D([0], [0], marker=NEG_MARKER, color="w", markerfacecolor=NEG_COLOR, markersize=8, label="Hard negative"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"{dataset_label} t-SNE — {sampling_note}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_oscar_encoder(*, projector_path: Path, oscar_model: str, device: str) -> OscarProjectorEncoder:
    return OscarProjectorEncoder(
        oscar_model_name=oscar_model,
        projector_path=str(projector_path),
        device=device,
        embed_dim=768,
        pooler="per_token_gated",
        num_layers=2,
        dropout=0.0,
        projector_hidden_dim=1024,
    )


def reload_projector(encoder: OscarProjectorEncoder, projector_path: Path) -> None:
    checkpoint = torch.load(projector_path, weights_only=False, map_location=encoder._device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"No model_state_dict in {projector_path}")
    encoder._projector.load_state_dict(checkpoint["model_state_dict"])
    encoder._projector.eval()
    print(f"[tsne] loaded projector {projector_path.name} (step={checkpoint.get('step', '?')})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="E11 query-doc t-SNE vs BGE teacher")
    parser.add_argument("--teacher-h5", type=Path, required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/query_distill_runs/e11-pertoken-gated/checkpoints"))
    parser.add_argument("--oscar-model", default="naver/oscar-qwen2-7B")
    parser.add_argument("--sample-size", type=int, default=80)
    parser.add_argument("--sampling-mode", choices=("unpaired", "paired"), default="unpaired")
    parser.add_argument("--include-negatives", action="store_true", default=True)
    parser.add_argument("--no-negatives", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=25.0)
    parser.add_argument("--tsne-max-iter", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reencode-teacher", action="store_true", help="BGE encode with query/doc prefixes (recommended for unpaired)")
    parser.add_argument("--grid-name", default=None)
    args = parser.parse_args()

    include_neg = args.include_negatives and not args.no_negatives
    dataset_label = args.dataset_name or args.teacher_h5.stem.replace("-train-hard", "").replace("-", " ").title()
    data = sample_from_h5(args.teacher_h5, args.sample_size, args.seed, mode=args.sampling_mode, include_negatives=include_neg)
    queries, positives, negatives = data["queries"], data["positives"], data.get("negatives", [])
    n = data["n"]
    pair_lines = args.sampling_mode == "paired"
    sampling_note = (
        f"paired train pairs (n={n}) — can look optimistic"
        if pair_lines
        else f"unpaired query/doc/neg pools (n={n} each)"
    )
    print(f"[tsne] {dataset_label}: {sampling_note}", flush=True)

    specs = [
        ("e11_step0", args.checkpoint_dir / "checkpoint_step_0.pt", "E11 — start (step 0)", "Before training"),
        ("e11_after_stage1", args.checkpoint_dir / "checkpoint_step_1500.pt", "E11 — after stage 1", "MSE-only (step 1500)"),
        ("e11_best", args.checkpoint_dir / "best_model.pt", "E11 — best", "Final checkpoint"),
        ("bge_teacher", None, "BGE-base-en-v1.5", "Teacher"),
    ]

    embeddings: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
    cosine_stats: dict[str, dict[str, float]] = {}

    if args.reencode_teacher or args.sampling_mode == "unpaired":
        print("[tsne] encoding teacher with BGE (query vs doc prefixes)...", flush=True)
        bge = BGEEncoder(model_name_or_path="BAAI/bge-base-en-v1.5", device=args.device)
        q_t, p_t = _encode_bge_split(bge, queries, positives, args.batch_size)
        n_t = _encode_normalized(bge, negatives, args.batch_size) if include_neg and negatives else None
        embeddings["bge_teacher"] = (q_t, p_t, n_t)
        del bge
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        n_t = data.get("negative_teacher") if include_neg else None
        embeddings["bge_teacher"] = (data["query_teacher"], data["positive_teacher"], n_t)

    cosine_stats["bge_teacher"] = _cosine_stats(*embeddings["bge_teacher"])

    oscar_ckpts = [s for s in specs if s[1] is not None]
    if oscar_ckpts:
        encoder = load_oscar_encoder(projector_path=oscar_ckpts[0][1], oscar_model=args.oscar_model, device=args.device)
        for key, path, _, _ in oscar_ckpts:
            if path != oscar_ckpts[0][1]:
                reload_projector(encoder, path)
            print(f"[tsne] encoding {key}...", flush=True)
            q_e = _encode_normalized(encoder, queries, args.batch_size)
            p_e = _encode_normalized(encoder, positives, args.batch_size)
            n_e = _encode_normalized(encoder, negatives, args.batch_size) if include_neg and negatives else None
            embeddings[key] = (q_e, p_e, n_e)
            cosine_stats[key] = _cosine_stats(q_e, p_e, n_e)
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_dir = args.output_dir
    summary: dict[str, Any] = {
        "dataset": dataset_label,
        "teacher_h5": str(args.teacher_h5),
        "sampling_mode": args.sampling_mode,
        "n_per_pool": n,
        "include_negatives": include_neg,
        "cosine_stats": cosine_stats,
        "plots": {},
    }
    if pair_lines and "pair_cos_teacher" in data:
        summary["paired_cos_teacher_mean"] = float(np.mean(data["pair_cos_teacher"]))

    grid_panels: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray | None, bool]] = []
    for key, _path, title, subtitle in specs:
        emb_q, emb_p, emb_n = embeddings[key]
        sub = f"{subtitle} · cos(q,d)={cosine_stats[key]['cos_query_doc_mean']:.3f}"
        if "cos_query_neg_mean" in cosine_stats[key]:
            sub += f", cos(q,n)={cosine_stats[key]['cos_query_neg_mean']:.3f}"
        png = plot_tsne_panel(
            emb_q, emb_p, emb_n,
            title=title, subtitle=sub,
            out_path=out_dir / f"{key}_tsne",
            seed=args.seed, perplexity=args.perplexity, max_iter=args.tsne_max_iter,
            pair_lines=pair_lines,
        )
        summary["plots"][key] = {"png": str(png), "cosine": cosine_stats[key]}
        grid_panels.append((title, sub, emb_q, emb_p, emb_n, pair_lines))

    grid_path = out_dir / (args.grid_name or f"e11_{dataset_label.lower()}_grid")
    plot_combined_grid(
        grid_panels, out_path=grid_path, seed=args.seed,
        perplexity=args.perplexity, max_iter=args.tsne_max_iter,
        dataset_label=dataset_label, sampling_note=sampling_note,
    )
    summary["grid_png"] = str(grid_path.with_suffix(".png"))

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
