#!/usr/bin/env python3
"""Clear embedding diagnostic: match vs random doc vs mined hard neg (violin + simple t-SNE)."""

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
from sklearn.manifold import TSNE

from projected_token.encoders.bge import BGEEncoder
from projected_token.encoders.oscar import OscarProjectorEncoder

plt.rcParams.update({"figure.facecolor": "white", "font.size": 11})

CATS = [
    ("match", "Match (query↔doc)", "#2ca02c"),
    ("random", "Random doc", "#ff7f0e"),
    ("hard", "Hard neg (mined)", "#7f7f7f"),
]
TSNE_COLORS = {"query": "#1f77b4", "match_doc": "#2ca02c", "random_doc": "#ff7f0e"}


def _decode_str(v: Any) -> str:
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def sample_paired_h5(path: Path, n: int, seed: int) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        total = int(h5["queries"].shape[0])
        rng = np.random.default_rng(seed)
        k = min(n, total)
        idx = rng.choice(total, size=k, replace=False)
        perm = idx.copy()
        for _ in range(20):
            rng.shuffle(perm)
            if not np.any(perm == idx):
                break
        return {
            "n": k,
            "queries": [_decode_str(h5["queries"][i]) for i in idx],
            "positives": [_decode_str(h5["positive_docs"][i]) for i in idx],
            "positives_shuffled": [_decode_str(h5["positive_docs"][i]) for i in perm],
            "hard_negs": [_decode_str(h5["negative_docs"][i]) for i in idx],
        }


def _encode_norm(encoder: Any, texts: list[str], bs: int) -> np.ndarray:
    out = []
    for s in range(0, len(texts), bs):
        b = texts[s : s + bs]
        e = encoder.encode(b, None)
        if hasattr(e, "detach"):
            e = F.normalize(e.float(), dim=-1).detach().cpu().numpy()
        out.append(np.asarray(e, dtype=np.float32))
    return np.concatenate(out, axis=0)


def _encode_bge_queries(bge: BGEEncoder, queries: list[str], bs: int) -> np.ndarray:
    parts = []
    for s in range(0, len(queries), bs):
        qb = queries[s : s + bs]
        parts.append(F.normalize(bge.encode([""] * len(qb), qb).float(), dim=-1).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros((0, 768), dtype=np.float32)


def _encode_bge_docs(bge: BGEEncoder, docs: list[str], bs: int) -> np.ndarray:
    parts = []
    for s in range(0, len(docs), bs):
        db = docs[s : s + bs]
        parts.append(F.normalize(bge.encode(db, None).float(), dim=-1).detach().cpu().numpy())
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros((0, 768), dtype=np.float32)


def _cos_diag(q: np.ndarray, match_d: np.ndarray, rand_d: np.ndarray, hard_d: np.ndarray) -> dict[str, np.ndarray]:
    match = np.sum(q * match_d, axis=1)
    random = np.sum(q * rand_d, axis=1)
    hard = np.sum(q * hard_d, axis=1)
    return {
        "match": match,
        "random": random,
        "hard": hard,
        "margin_random": match - random,
        "margin_hard": match - hard,
        "rank_ok": (match > random).astype(np.float32),
    }


def plot_violin_grid(
    all_scores: dict[str, dict[str, np.ndarray]],
    *,
    dataset: str,
    out_path: Path,
) -> None:
    models = list(all_scores.keys())
    titles = {
        "bge_teacher": "BGE teacher",
        "e11_step0": "E11 · step 0",
        "e11_after_stage1": "E11 · after stage 1",
        "e11_best": "E11 · best",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, key in zip(axes.ravel(), models):
        scores = all_scores[key]
        data = [scores[c] for c, _, _ in CATS]
        labels = [lab for _, lab, _ in CATS]
        colors = [col for _, _, col in CATS]
        parts = ax.violinplot(data, positions=range(3), showmeans=True, showmedians=False, widths=0.85)
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col)
            body.set_alpha(0.55)
            body.set_edgecolor(col)
        parts["cmeans"].set_color("#222222")
        ax.axhline(0, color="#ccc", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("cosine similarity")
        ax.set_title(titles.get(key, key), fontweight="semibold")
        m_match = float(np.mean(scores["match"]))
        m_rand = float(np.mean(scores["random"]))
        m_hard = float(np.mean(scores["hard"]))
        margin = m_match - m_hard
        ax.text(
            0.02, 0.98,
            f"Δ(match−hard)={margin:+.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            color="#2ca02c" if margin > 0.05 else "#d62728",
        )
    fig.suptitle(
        f"{dataset}: query vs match / random / hard negative (cosine)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_margin_probe_grid(
    all_scores: dict[str, dict[str, np.ndarray]],
    *,
    dataset: str,
    out_path: Path,
) -> None:
    """Main slide: margin distributions — step 0 near zero, best large."""
    titles = {
        "bge_teacher": "BGE teacher",
        "e11_step0": "E11 · step 0 (init)",
        "e11_after_stage1": "E11 · after stage 1",
        "e11_best": "E11 · best",
    }
    margin_cats = [
        ("margin_random", "Δ: match − random", "#2ca02c"),
        ("margin_hard", "Δ: match − hard neg", "#1f77b4"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, key in zip(axes.ravel(), all_scores.keys()):
        scores = all_scores[key]
        data = [scores[c] for c, _, _ in margin_cats]
        colors = [col for _, _, col in margin_cats]
        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, widths=0.8)
        for body, col in zip(parts["bodies"], colors):
            body.set_facecolor(col)
            body.set_alpha(0.6)
            body.set_edgecolor(col)
        parts["cmeans"].set_color("#222")
        ax.axhline(0, color="#d62728", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([lab for _, lab, _ in margin_cats], fontsize=9)
        ax.set_ylim(-0.15, 0.85)
        ax.set_ylabel("margin (cosine gap)")
        mr = float(scores["margin_random"].mean())
        mh = float(scores["margin_hard"].mean())
        acc_strict = float((scores["margin_random"] > 0.15).mean())
        good = mr > 0.15 and acc_strict > 0.7
        ax.set_title(titles.get(key, key), fontweight="semibold", color="#2ca02c" if good else "#d62728")
        ax.text(
            0.02, 0.98,
            f"mean Δrand={mr:+.2f}  Δhard={mh:+.2f}\nfrac Δrand>0.15: {acc_strict:.0%}",
            transform=ax.transAxes, va="top", fontsize=8.5,
        )
        if key.startswith("e11_step") and mr < 0.05:
            ax.text(0.5, 0.5, "collapse\n(poor)", transform=ax.transAxes, ha="center", va="center",
                    fontsize=14, color="#d62728", alpha=0.35, fontweight="bold")
    fig.suptitle(
        f"{dataset}: retrieval quality = margin (gap), not raw cosine\n"
        "step 0 ≈ 0 (poor); best = large (good)",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_progress(
    all_scores: dict[str, dict[str, np.ndarray]],
    *,
    dataset: str,
    out_path: Path,
) -> None:
    order = ["e11_step0", "e11_after_stage1", "e11_best", "bge_teacher"]
    labels = ["E11 step 0", "E11 stage 1", "E11 best", "BGE teacher"]
    mr = [all_scores[k]["margin_random"].mean() for k in order]
    mh = [all_scores[k]["margin_hard"].mean() for k in order]
    acc = [(all_scores[k]["margin_random"] > 0.15).mean() for k in order]

    x = np.arange(len(order))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w / 2, mr, w, label="Δ (match − random)", color="#2ca02c", alpha=0.85)
    ax.bar(x + w / 2, mh, w, label="Δ (match − hard neg)", color="#1f77b4", alpha=0.85)
    ax.plot(x, acc, "o-", color="#d62728", linewidth=2, markersize=8, label="frac. Δ(match−random) > 0.15")
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("margin / accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(f"{dataset}: E11 training progress", fontweight="semibold")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_simple_tsne(
    q: np.ndarray,
    match_d: np.ndarray,
    rand_d: np.ndarray,
    *,
    title: str,
    out_path: Path,
    seed: int,
) -> None:
    n = len(q)
    stack = np.concatenate([q, match_d, rand_d], axis=0)
    xy = TSNE(n_components=2, random_state=seed, init="pca", perplexity=min(30, max(5, n // 2)), max_iter=400).fit_transform(stack)
    q2, m2, r2 = xy[:n], xy[n : 2 * n], xy[2 * n :]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(r2[:, 0], r2[:, 1], c=TSNE_COLORS["random_doc"], s=40, alpha=0.5, label="Random doc", marker="^")
    ax.scatter(m2[:, 0], m2[:, 1], c=TSNE_COLORS["match_doc"], s=45, alpha=0.75, label="Matched doc", marker="s")
    ax.scatter(q2[:, 0], q2[:, 1], c=TSNE_COLORS["query"], s=40, alpha=0.85, label="Query", marker="o", zorder=3)
    for i in range(min(n, 40)):
        ax.plot([q2[i, 0], m2[i, 0]], [q2[i, 1], m2[i, 1]], c="#2ca02c", alpha=0.25, linewidth=0.7, zorder=1)
    ax.set_title(title, fontweight="semibold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_oscar(projector: Path, oscar: str, device: str) -> OscarProjectorEncoder:
    return OscarProjectorEncoder(
        oscar_model_name=oscar,
        projector_path=str(projector),
        device=device,
        embed_dim=768,
        pooler="per_token_gated",
        num_layers=2,
        dropout=0.0,
        projector_hidden_dim=1024,
    )


def reload_proj(enc: OscarProjectorEncoder, path: Path) -> None:
    ckpt = torch.load(path, weights_only=False, map_location=enc._device)
    enc._projector.load_state_dict(ckpt["model_state_dict"])
    enc._projector.eval()


def _save_scores_cache(path: Path, all_scores: dict[str, dict[str, np.ndarray]], emb_cache: dict) -> None:
    arrays = {}
    for model, scores in all_scores.items():
        for name, arr in scores.items():
            arrays[f"{model}__{name}"] = arr
    for model, (q, m, r) in emb_cache.items():
        arrays[f"{model}__emb_q"] = q
        arrays[f"{model}__emb_match"] = m
        arrays[f"{model}__emb_random"] = r
    np.savez_compressed(path, **arrays)


def _load_scores_cache(path: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    raw = np.load(path)
    all_scores: dict[str, dict[str, np.ndarray]] = {}
    emb_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    score_keys = ("match", "random", "hard", "margin_random", "margin_hard", "rank_ok")
    models = {k.split("__", 1)[0] for k in raw.files if "__" in k}
    for model in models:
        all_scores[model] = {sk: raw[f"{model}__{sk}"] for sk in score_keys if f"{model}__{sk}" in raw.files}
        if f"{model}__emb_q" in raw.files:
            emb_cache[model] = (raw[f"{model}__emb_q"], raw[f"{model}__emb_match"], raw[f"{model}__emb_random"])
    return all_scores, emb_cache


def _render_plots(
    all_scores: dict[str, dict[str, np.ndarray]],
    emb_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    out: Path,
    dataset_name: str,
    seed: int,
    n_pairs: int,
) -> dict[str, Any]:
    slug = dataset_name.lower().replace(" ", "_")
    plot_margin_probe_grid(all_scores, dataset=dataset_name, out_path=out / f"{slug}_margin_probe_grid")
    plot_training_progress(all_scores, dataset=dataset_name, out_path=out / "training_progress")
    plot_violin_grid(all_scores, dataset=dataset_name, out_path=out / "cosine_raw_violin")
    for key, title in [
        ("e11_step0", "E11 step 0 — collapse"),
        ("e11_best", "E11 best"),
        ("bge_teacher", "BGE teacher"),
    ]:
        if key in emb_cache:
            q_e, m_e, r_e = emb_cache[key]
            plot_simple_tsne(q_e, m_e, r_e, title=title, out_path=out / f"tsne_simple_{key}", seed=seed)
    summary = {
        "dataset": dataset_name,
        "n_pairs": n_pairs,
        "means": {
            k: {c: float(v.mean()) for c, v in s.items() if c in ("match", "random", "hard", "margin_random", "margin_hard", "rank_ok")}
            for k, s in all_scores.items()
        },
        "margin_match_minus_hard": {k: float(s["margin_hard"].mean()) for k, s in all_scores.items()},
        "margin_match_minus_random": {k: float(s["margin_random"].mean()) for k, s in all_scores.items()},
        "fraction_margin_random_gt_0.15": {
            k: float((s["margin_random"] > 0.15).mean()) for k, s in all_scores.items()
        },
        "main_plot": str((out / f"{slug}_margin_probe_grid").with_suffix(".png")),
    }
    (out / "margin_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-h5", type=Path, required=True)
    p.add_argument("--dataset-name", default="Quora")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/query_distill_runs/e11-pertoken-gated/checkpoints"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-pairs", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--oscar-model", default="naver/oscar-qwen2-7B")
    p.add_argument("--plots-only", action="store_true", help="Re-render PNGs from scores_cache.npz (no GPU)")
    args = p.parse_args()

    out = args.output_dir
    cache_path = out / "scores_cache.npz"
    if args.plots_only:
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing {cache_path}; run without --plots-only first")
        all_scores, emb_cache = _load_scores_cache(cache_path)
        summary = _render_plots(
            all_scores, emb_cache, out=out, dataset_name=args.dataset_name, seed=args.seed, n_pairs=args.n_pairs
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return

    data = sample_paired_h5(args.teacher_h5, args.n_pairs, args.seed)
    q_t, p_t, ps_t, hn_t = data["queries"], data["positives"], data["positives_shuffled"], data["hard_negs"]

    specs = [
        ("bge_teacher", None),
        ("e11_step0", args.checkpoint_dir / "checkpoint_step_0.pt"),
        ("e11_after_stage1", args.checkpoint_dir / "checkpoint_step_1500.pt"),
        ("e11_best", args.checkpoint_dir / "best_model.pt"),
    ]

    all_scores: dict[str, dict[str, np.ndarray]] = {}
    emb_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    print("[margin] BGE teacher...", flush=True)
    bge = BGEEncoder(model_name_or_path="BAAI/bge-base-en-v1.5", device=args.device)
    q_e = _encode_bge_queries(bge, q_t, args.batch_size)
    m_e = _encode_bge_docs(bge, p_t, args.batch_size)
    r_e = _encode_bge_docs(bge, ps_t, args.batch_size)
    h_e = _encode_bge_docs(bge, hn_t, args.batch_size)
    all_scores["bge_teacher"] = _cos_diag(q_e, m_e, r_e, h_e)
    emb_cache["bge_teacher"] = (q_e, m_e, r_e)
    del bge
    torch.cuda.empty_cache()

    oscar_specs = [(k, pt) for k, pt in specs if pt is not None]
    enc = load_oscar(oscar_specs[0][1], args.oscar_model, args.device)
    for key, path in oscar_specs:
        if path != oscar_specs[0][1]:
            reload_proj(enc, path)
        print(f"[margin] {key}...", flush=True)
        q_e = _encode_norm(enc, q_t, args.batch_size)
        m_e = _encode_norm(enc, p_t, args.batch_size)
        r_e = _encode_norm(enc, ps_t, args.batch_size)
        h_e = _encode_norm(enc, hn_t, args.batch_size)
        all_scores[key] = _cos_diag(q_e, m_e, r_e, h_e)
        emb_cache[key] = (q_e, m_e, r_e)
    del enc
    torch.cuda.empty_cache()

    out.mkdir(parents=True, exist_ok=True)
    _save_scores_cache(cache_path, all_scores, emb_cache)
    summary = _render_plots(
        all_scores, emb_cache, out=out, dataset_name=args.dataset_name, seed=args.seed, n_pairs=data["n"]
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
