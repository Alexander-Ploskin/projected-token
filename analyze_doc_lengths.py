
"""Анализ распределения длины документов в токенах в PopQA датасете."""

import argparse
from collections import Counter
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Анализ длины документов в токенах")
    parser.add_argument("--dataset-path", default="/data/popqa_enriched.parquet")
    parser.add_argument("--text-col", default="s_wiki_content")
    parser.add_argument("--max-docs", type=int, default=None, help="Ограничить число документов")
    args = parser.parse_args()
    
    # Загрузка датасета
    print(f"Загрузка датасета из {args.dataset_path}...")
    table = pq.read_table(args.dataset_path)
    texts = table.column(args.text_col).to_pylist()
    
    if args.max_docs:
        texts = texts[:args.max_docs]
    
    print(f"Документов: {len(texts)}")
    
    # Используем токенизатор Qwen (от OSCAR модели)
    from transformers import AutoTokenizer
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/huggingface/Qwen/Qwen2-7B-Instruct",
        trust_remote_code=True
    )
    
    # Токенизация
    print("Токенизация...")
    lengths = []
    for text in tqdm(texts, desc="Подсчет токенов"):
        if text and isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    
    # Статистика
    print("\n=== Статистика длин в токенах ===")
    print(f"Документов: {len(lengths)}")
    print(f"Мин: {lengths.min()}")
    print(f"Макс: {lengths.max()}")
    print(f"Среднее: {lengths.mean():.1f}")
    print(f"Медиана: {np.median(lengths):.1f}")
    print(f"Стандартное отклонение: {lengths.std():.1f}")
    
    # Перцентили
    print("\n=== Перцентили ===")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"P{p}: {np.percentile(lengths, p):.0f}")
    
    # Распределение
    print("\n=== Распределение ===")
    ranges = [
        (0, 32), (33, 64), (65, 96), (97, 128), 
        (129, 256), (257, 512), (513, 1024), (1025, float('inf'))
    ]
    
    for r_min, r_max in ranges:
        if r_max == float('inf'):
            count = np.sum(lengths >= r_min)
            pct = 100 * count / len(lengths)
            print(f">= {r_min}: {count} ({pct:.1f}%)")
        else:
            count = np.sum((lengths >= r_min) & (lengths <= r_max))
            pct = 100 * count / len(lengths)
            print(f"{r_min}-{r_max}: {count} ({pct:.1f}%)")
    
    # Сколько документов > 128 (лимит OSCAR)
    over_limit = np.sum(lengths > 128)
    print(f"\nДокументов > 128 токенов: {over_limit} ({100*over_limit/len(lengths):.1f}%)")
    
    # Сколько нужно mem-tokens при разных compression rates
    print("\n=== Требуемое число mem-tokens при разных compression rates ===")
    for rate in [8, 16, 32, 64]:
        mem_tokens_needed = np.ceil(lengths / rate).astype(int)
        avg_mem_tokens = mem_tokens_needed.mean()
        max_mem_tokens = mem_tokens_needed.max()
        print(f"Rate x{rate}: в среднем {avg_mem_tokens:.1f} mem-tokens, макс {max_mem_tokens}")


if __name__ == "__main__":
    main()