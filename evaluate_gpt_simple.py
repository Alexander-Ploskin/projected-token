
"""Evaluate paraphrases using GPT Judge metric and simple similarity metrics."""

import argparse
import json
from difflib import SequenceMatcher
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = """You are a strict factual consistency judge.

You will be given cases with:
- candidate (paraphrased text)
- reference (original text)

Task: Compare candidate against reference and judge factual consistency.

Rubric:
- supported: All factual claims in candidate are supported by reference
- partially_supported: Most claims supported, but minor unsupported/ambiguous parts
- contradicted: Any clear factual contradiction between candidate and reference
- unknown: Reference lacks enough info to assess most claims

Return JSON with this exact format:
{
  "verdicts": [
    {
      "id": 0,
      "score": 0.9,
      "label": "supported"
    }
  ]
}

Score: 0.0-1.0 (1.0 = perfect factual consistency)
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file as list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization by lowercasing and splitting on whitespace."""
    return text.lower().split()


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def calculate_simple_metrics(
    originals: List[str], paraphrases: List[str]
) -> Dict[str, float]:
    """Calculate similarity metrics between original and paraphrased texts."""
    results = []
    for orig, para in zip(originals, paraphrases):
        if not orig.strip() or not para.strip():
            continue

        orig_tokens = set(simple_tokenize(orig))
        para_tokens = set(simple_tokenize(para))

        results.append({
            "jaccard": jaccard_similarity(orig_tokens, para_tokens),
            "char_sim": SequenceMatcher(None, orig.lower(), para.lower()).ratio(),
            "word_overlap": len(orig_tokens & para_tokens) / max(len(orig_tokens), 1),
            "length_ratio": len(para.split()) / max(len(orig.split()), 1),
        })

    if not results:
        return {}

    return {
        f"avg_{key}": float(np.mean([r[key] for r in results]))
        for key in ["jaccard", "char_sim", "word_overlap", "length_ratio"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to JSONL file with paraphrases')
    parser.add_argument('--output', '-o', help='Output path for metrics JSON')
    parser.add_argument('--base-url', default='http://localhost:8000/v1', help='vLLM API base URL')
    parser.add_argument('--api-key', default='dummy', help='API key')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct', help='Model name')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
    args = parser.parse_args()

    print(f"Loading {args.input_path}...")
    data = load_jsonl(args.input_path)

    originals = [d.get('s_wiki_content', '') for d in data]
    paraphrases = [d.get('rephrased_text', '') for d in data]

    valid_indices = [i for i in range(len(data)) if originals[i].strip() and paraphrases[i].strip()]
    originals = [originals[i] for i in valid_indices]
    paraphrases = [paraphrases[i] for i in valid_indices]

    print(f"Evaluating {len(originals)} valid paraphrases...")

    # Simple metrics
    print("\n=== Calculating simple metrics ===")
    simple_metrics = calculate_simple_metrics(originals, paraphrases)
    print(f"avg_jaccard: {simple_metrics.get('avg_jaccard', 0):.4f}")
    print(f"avg_char_sim: {simple_metrics.get('avg_char_sim', 0):.4f}")
    print(f"avg_word_overlap: {simple_metrics.get('avg_word_overlap', 0):.4f}")
    print(f"avg_length_ratio: {simple_metrics.get('avg_length_ratio', 0):.4f}")

    # GPT Judge
    print(f"\n=== Initializing GPT Judge (model: {args.model}) ===")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"=== Running GPT Judge (batch_size={args.batch_size}) ===")
    all_scores = []
    all_labels = []

    for i in tqdm(range(0, len(originals), args.batch_size)):
        batch_end = min(i + args.batch_size, len(originals))
        cases = [
            {"id": idx, "candidate": paraphrases[idx], "reference": originals[idx]}
            for idx in range(i, batch_end)
        ]

        try:
            response = client.chat.completions.create(
                model=args.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({"cases": cases}, ensure_ascii=False)},
                ],
            )

            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                for verdict in result.get('verdicts', []):
                    score = verdict.get('score', -1)
                    if 0 <= score <= 1:
                        all_scores.append(score)
                        all_labels.append(verdict.get('label', 'unknown'))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse response: {content[:200]}")

        except Exception as e:
            print(f"Error in batch {i}: {e}")

    if all_scores:
        print(f"\n=== GPT Judge Results ===")
        print(f"avg_gpt_judge_score: {np.mean(all_scores):.4f}")
        print(f"std_gpt_judge_score: {np.std(all_scores):.4f}")
        print(f"min_gpt_judge_score: {np.min(all_scores):.4f}")
        print(f"max_gpt_judge_score: {np.max(all_scores):.4f}")

        label_counts = {}
        for label in all_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} ({100*count/len(all_labels):.1f}%)")

        gpt_metrics = {
            'avg_gpt_judge_score': float(np.mean(all_scores)),
            'std_gpt_judge_score': float(np.std(all_scores)),
            'min_gpt_judge_score': float(np.min(all_scores)),
            'max_gpt_judge_score': float(np.max(all_scores)),
            'label_distribution': label_counts
        }
    else:
        print("\n=== GPT Judge Results ===")
        print("ERROR: No valid scores obtained")
        gpt_metrics = {'error': 'No valid scores'}

    all_metrics = {**simple_metrics, **gpt_metrics}

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved to {args.output}")

    return all_metrics


if __name__ == '__main__':
    main()
