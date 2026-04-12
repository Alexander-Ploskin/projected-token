#!/usr/bin/env python3
"""Evaluate paraphrase quality using BLEU, ROUGE, and other metrics."""

import json
import argparse
from typing import List, Dict
import numpy as np

try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available, skipping BLEU/ROUGE metrics")

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("Warning: rouge-score not available, skipping ROUGE metrics")


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def simple_tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return text.lower().split()


def calculate_bleu(originals: List[str], paraphrases: List[str]) -> float:
    """Calculate BLEU score."""
    if not NLTK_AVAILABLE:
        return 0.0

    try:
        # Tokenize
        references = [[simple_tokenize(doc)] for doc in originals]
        hypotheses = [simple_tokenize(para) for para in paraphrases]

        # Calculate BLEU with smoothing
        smoother = SmoothingFunction().method1
        bleu = corpus_bleu(references, hypotheses, smoothing_function=smoother)
        return bleu
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0


def calculate_rouge(originals: List[str], paraphrases: List[str]) -> Dict[str, float]:
    """Calculate ROUGE scores."""
    if not ROUGE_AVAILABLE:
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}

    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = {
            'rouge-1': [],
            'rouge-2': [],
            'rouge-l': []
        }

        for orig, para in zip(originals, paraphrases):
            if orig.strip() and para.strip():
                s = scorer.score(orig, para)
                scores['rouge-1'].append(s['rouge1'].fmeasure)
                scores['rouge-2'].append(s['rouge2'].fmeasure)
                scores['rouge-l'].append(s['rougeL'].fmeasure)

        return {k: np.mean(v) if v else 0.0 for k, v in scores.items()}
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}


def calculate_edit_similarity(originals: List[str], paraphrases: List[str]) -> float:
    """Calculate average character-level edit similarity."""
    try:
        from difflib import SequenceMatcher
        similarities = []
        for orig, para in zip(originals, paraphrases):
            if orig.strip() and para.strip():
                sim = SequenceMatcher(None, orig.lower(), para.lower()).ratio()
                similarities.append(sim)
        return np.mean(similarities) if similarities else 0.0
    except Exception as e:
        print(f"Edit similarity calculation error: {e}")
        return 0.0


def calculate_length_stats(originals: List[str], paraphrases: List[str]) -> Dict[str, float]:
    """Calculate length statistics."""
    orig_lengths = [len(doc.split()) for doc in originals]
    para_lengths = [len(para.split()) for para in paraphrases]

    return {
        'avg_original_length': np.mean(orig_lengths),
        'avg_paraphrase_length': np.mean(para_lengths),
        'length_ratio': np.mean(para_lengths) / np.mean(orig_lengths) if np.mean(orig_lengths) > 0 else 0
    }


def evaluate_paraphrase(input_path: str, output_path: str = None):
    """Main evaluation function."""
    print(f"Loading data from {input_path}...")
    data = load_jsonl(input_path)

    # Extract originals and paraphrases
    originals = [d.get('s_wiki_content', '') for d in data]
    paraphrases = [d.get('rephrased_text', '') for d in data]

    # Filter empty entries
    valid_indices = [i for i in range(len(data)) if originals[i].strip() and paraphrases[i].strip()]
    originals = [originals[i] for i in valid_indices]
    paraphrases = [paraphrases[i] for i in valid_indices]

    print(f"Evaluating {len(originals)} valid paraphrases...")

    # Calculate metrics
    results = {}

    # BLEU
    bleu = calculate_bleu(originals, paraphrases)
    results['bleu'] = bleu
    print(f"BLEU: {bleu:.4f}")

    # ROUGE
    rouge_scores = calculate_rouge(originals, paraphrases)
    results.update(rouge_scores)
    print(f"ROUGE-1: {rouge_scores['rouge-1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge-2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rouge-l']:.4f}")

    # Edit similarity
    edit_sim = calculate_edit_similarity(originals, paraphrases)
    results['edit_similarity'] = edit_sim
    print(f"Edit Similarity: {edit_sim:.4f}")

    # Length stats
    length_stats = calculate_length_stats(originals, paraphrases)
    results.update(length_stats)
    print(f"Length Ratio: {length_stats['length_ratio']:.4f}")

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate paraphrase quality')
    parser.add_argument('input_path', help='Path to JSONL file with paraphrases')
    parser.add_argument('--output', '-o', help='Output path for metrics JSON')

    args = parser.parse_args()
    evaluate_paraphrase(args.input_path, args.output)
