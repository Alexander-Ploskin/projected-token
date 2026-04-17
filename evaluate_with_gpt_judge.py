
"""Evaluate paraphrase with GPT Judge metric using local vllm."""

import json
import argparse
import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm

# Import GPT Judge metric
from evaluation.metrics.gpt_score.gpt_score import GPTScoreMetric


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


def calculate_simple_metrics(originals, paraphrases):
    """Calculate simple metrics without NLTK."""
    results = []

    for orig, para in zip(originals, paraphrases):
        if not orig.strip() or not para.strip():
            continue

        orig_tokens = set(simple_tokenize(orig))
        para_tokens = set(simple_tokenize(para))

        jaccard = jaccard_similarity(orig_tokens, para_tokens)
        char_sim = SequenceMatcher(None, orig.lower(), para.lower()).ratio()
        word_overlap = len(orig_tokens & para_tokens) / max(len(orig_tokens), 1)
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

    avg_metrics = {}
    for key in ['jaccard', 'char_sim', 'word_overlap', 'length_ratio']:
        avg_metrics[f'avg_{key}'] = np.mean([r[key] for r in results])

    return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='Path to JSONL file with paraphrases')
    parser.add_argument('--output', '-o', help='Output path for metrics JSON')
    parser.add_argument('--base-url', default='http://localhost:8000/v1', help='vLLM API base URL')
    parser.add_argument('--api-key', default='dummy', help='API key (can be dummy for local)')
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct', help='Model name for judge')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for GPT Judge')
    args = parser.parse_args()

    print(f"Loading {args.input_path}...")
    data = load_jsonl(args.input_path)

    originals = [d.get('s_wiki_content', '') for d in data]
    paraphrases = [d.get('rephrased_text', '') for d in data]

    # Filter valid pairs
    valid_indices = [i for i in range(len(data)) if originals[i].strip() and paraphrases[i].strip()]
    originals = [originals[i] for i in valid_indices]
    paraphrases = [paraphrases[i] for i in valid_indices]

    print(f"Evaluating {len(originals)} valid paraphrases...")

    # Calculate simple metrics
    print("\n=== Calculating simple metrics ===")
    simple_metrics = calculate_simple_metrics(originals, paraphrases)
    print(f"avg_jaccard: {simple_metrics.get('avg_jaccard', 0):.4f}")
    print(f"avg_char_sim: {simple_metrics.get('avg_char_sim', 0):.4f}")
    print(f"avg_word_overlap: {simple_metrics.get('avg_word_overlap', 0):.4f}")
    print(f"avg_length_ratio: {simple_metrics.get('avg_length_ratio', 0):.4f}")

    # Initialize GPT Judge
    print(f"\n=== Initializing GPT Judge (model: {args.model}) ===")
    gpt_judge = GPTScoreMetric({
        'base_url': args.base_url,
        'api_key': args.api_key,
        'model': args.model,
        'temperature': 0.0
    })

    # Evaluate in batches
    print(f"=== Running GPT Judge (batch_size={args.batch_size}) ===")
    all_scores = []
    all_labels = []

    for i in tqdm(range(0, len(originals), args.batch_size)):
        batch_end = min(i + args.batch_size, len(originals))
        batch_refs = originals[i:batch_end]
        batch_cands = paraphrases[i:batch_end]

        verdicts = gpt_judge.judge_batch(batch_cands, batch_refs)

        for v in verdicts:
            score = v.get('llm_judge_score', -1.0)
            if score >= 0:
                all_scores.append(score)
                all_labels.append(v.get('llm_judge_label', 'unknown'))

    # Calculate GPT Judge stats
    if all_scores:
        print(f"\n=== GPT Judge Results ===")
        print(f"avg_gpt_judge_score: {np.mean(all_scores):.4f}")
        print(f"std_gpt_judge_score: {np.std(all_scores):.4f}")
        print(f"min_gpt_judge_score: {np.min(all_scores):.4f}")
        print(f"max_gpt_judge_score: {np.max(all_scores):.4f}")

        # Label distribution
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

    # Combine all metrics
    all_metrics = {**simple_metrics, **gpt_metrics}

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nSaved to {args.output}")

    return all_metrics


if __name__ == '__main__':
    main()
