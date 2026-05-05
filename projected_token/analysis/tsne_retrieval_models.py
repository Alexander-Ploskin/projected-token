from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonlines
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from sklearn.manifold import TSNE

from projected_token.encoders.oscar import OscarEncoder, OscarProjectorEncoder
from projected_token.encoders.salesforce import SalesforceEncoder


def _encode(encoder: Any, texts: list[str], batch_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        emb = encoder.encode(batch)
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        chunks.append(np.asarray(emb, dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def _sample_msmarco(sample_size: int, seed: int) -> tuple[list[str], list[str]]:
    ds = load_dataset("/data/huggingface/sentence-transformers/msmarco-msmarco-distilbert-base-v3", name="triplet", split="train")
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(ds))
    idx = rng.choice(len(ds), size=n, replace=False)
    rows = [ds[int(i)] for i in idx]
    return [str(r["query"]) for r in rows], [str(r["positive"]) for r in rows]


def _sample_popqa(sample_size: int, seed: int) -> tuple[list[str], list[str]]:
    table = pq.read_table("/data/popqa_enriched.parquet")
    data = table.to_pydict()
    n_all = table.num_rows
    rng = np.random.default_rng(seed)
    n = min(sample_size, n_all)
    idx = rng.choice(n_all, size=n, replace=False)
    queries = [str(data["question"][int(i)]) for i in idx]
    docs = [str(data["s_wiki_content"][int(i)]) for i in idx]
    return queries, docs


def _sample_beir_scifact(sample_size: int, seed: int) -> tuple[list[str], list[str]]:
    q_rows = []
    c_rows = []
    with jsonlines.open("/data/huggingface/BeIR/scifact/queries.jsonl", "r") as reader:
        for row in reader:
            q_rows.append(str(row.get("text", "")))
    with jsonlines.open("/data/huggingface/BeIR/scifact/corpus.jsonl", "r") as reader:
        for row in reader:
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            c_rows.append(" ".join([x for x in [title, text] if x]).strip())
    rng = np.random.default_rng(seed)
    qn = min(sample_size, len(q_rows))
    dn = min(sample_size, len(c_rows))
    q_idx = rng.choice(len(q_rows), size=qn, replace=False)
    d_idx = rng.choice(len(c_rows), size=dn, replace=False)
    queries = [q_rows[int(i)] for i in q_idx]
    docs = [c_rows[int(i)] for i in d_idx]
    n = min(len(queries), len(docs))
    return queries[:n], docs[:n]


def _plot_tsne(path: Path, emb_q: np.ndarray, emb_d: np.ndarray, title: str, seed: int) -> None:
    n = min(len(emb_q), len(emb_d))
    if n < 10:
        return
    stack = np.concatenate([emb_q[:n], emb_d[:n]], axis=0)
    labels = np.array(["query"] * n + ["doc"] * n)
    perplexity = max(5, min(30, (2 * n - 1) // 3))
    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity)
    emb_2d = tsne.fit_transform(stack)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    qmask = labels == "query"
    dmask = labels == "doc"
    plt.scatter(emb_2d[qmask, 0], emb_2d[qmask, 1], s=14, alpha=0.7, label="query")
    plt.scatter(emb_2d[dmask, 0], emb_2d[dmask, 1], s=14, alpha=0.7, label="doc")
    plt.title(title)
    plt.xlabel("tsne-1")
    plt.ylabel("tsne-2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run t-SNE for OSCAR/SFR/projector models on multiple datasets.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="artifacts/analysis/tsne_retrieval_models")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "msmarco": _sample_msmarco(args.sample_size, args.seed),
        "popqa_enriched": _sample_popqa(args.sample_size, args.seed),
        "scifact": _sample_beir_scifact(args.sample_size, args.seed),
    }

    oscar_model = "/data/huggingface/naver/oscar-qwen2-7B"
    model_specs: list[tuple[str, callable]] = [
        ("sfr_baseline", lambda: SalesforceEncoder(model_name_or_path="/data/huggingface/Salesforce/SFR-Embedding-Mistral", device=args.device)),
        ("oscar_first", lambda: OscarEncoder(model_name_or_path=oscar_model, device=args.device, aggregation="first")),
        ("oscar_last", lambda: OscarEncoder(model_name_or_path=oscar_model, device=args.device, aggregation="last")),
        ("oscar_flatten", lambda: OscarEncoder(model_name_or_path=oscar_model, device=args.device, aggregation="flatten")),
        (
            "projector_contrastive_1layer",
            lambda: OscarProjectorEncoder(
                oscar_model_name=oscar_model,
                projector_path="artifacts/runs/20260428_142514_mlp-projector-mlp-1layer-msmarco/checkpoints/best_model.pt",
                device=args.device,
            ),
        ),
        (
            "projector_contrastive_3layer",
            lambda: OscarProjectorEncoder(
                oscar_model_name=oscar_model,
                projector_path="artifacts/runs/20260428_173204_mlp-projector-mlp-3layer-msmarco/checkpoints/best_model.pt",
                device=args.device,
            ),
        ),
        (
            "projector_two_stage_1layer",
            lambda: OscarProjectorEncoder(
                oscar_model_name=oscar_model,
                projector_path="artifacts/runs/20260428_175038_mlp-two-stage-distill-to-contrastive-1layer/checkpoints/best_model.pt",
                device=args.device,
            ),
        ),
    ]

    summary: dict[str, Any] = {"datasets": {}, "models": [name for name, _ in model_specs]}
    for dataset_name, (queries, docs) in datasets.items():
        summary["datasets"][dataset_name] = {"pairs": len(queries), "plots": {}}
        for model_name, build_encoder in model_specs:
            encoder = None
            model_out = out_dir / dataset_name / f"{model_name}_tsne.png"
            try:
                encoder = build_encoder()
                q_emb = _encode(encoder, queries, args.batch_size)
                d_emb = _encode(encoder, docs, args.batch_size)
                _plot_tsne(
                    model_out,
                    q_emb,
                    d_emb,
                    title=f"{dataset_name}: {model_name}",
                    seed=args.seed,
                )
                summary["datasets"][dataset_name]["plots"][model_name] = str(model_out)
            except Exception as exc:
                summary["datasets"][dataset_name]["plots"][model_name] = f"error: {exc}"
            finally:
                del encoder
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
