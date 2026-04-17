
"""Simple paraphrase evaluation without NLTK."""

import json
import argparse
import numpy as np
from difflib import SequenceMatcher


def load_jsonl(path: str):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def simple_tokenize(text: str):
    return text.lower().split()


def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def calculate_metrics(originals, paraphrases):
    """Calculate simple metrics without NLTK."""
    results = []

    for orig, para in zip(originals, paraphrases):
        if not orig.strip() or not para.strip():
            continue

        orig_tokens = set(simple_tokenize(orig))
        para_tokens = set(simple_tokenize(para))

        # Jaccard similarity
        jaccard = jaccard_similarity(orig_tokens, para_tokens)

        # Character-level similarity
        char_sim = SequenceMatcher(None, orig.lower(), para.lower()).ratio()

        # Word overlap
        word_overlap = len(orig_tokens & para_tokens) / max(len(orig_tokens), 1)

        # Length ratio
        orig_len = len(orig.split())
        para_len = len(para.split())
        length_ratio = para_len / max(orig_len, 1)

        results.append({
            'jaccard': jaccard,
            'char_sim': char_sim,
            'word_overlap': word_overlap,
            'length_ratio': length_ratio
        })

    if not results:
        return {}

    # Average metrics
    avg_metrics = {}
    for key in ['jaccard', 'char_sim', 'word_overlap', 'length_ratio']:
        avg_metrics[f'avg_{key}'] = np.mean([r[key] for r in results])

    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path')
    parser.add_argument('--output', '-o')
    args = parser.parse_args()

    print(f"Loading {args.input_path}...")
    data = load_jsonl(args.input_path)

    originals = [d.get('s_wiki_content', '') for d in data]
    paraphrases = [d.get('rephrased_text', '') for d in data]

    print(f"Calculating metrics for {len(originals)} samples...")
    metrics = calculate_metrics(originals, paraphrases)

    print("\n=== PARAPHRASE METRICS ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
