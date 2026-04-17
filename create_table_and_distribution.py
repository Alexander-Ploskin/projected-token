
"""Создание визуализаций: таблица метрик и распределение длин документов."""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq


def load_metrics(index_dir: str) -> dict:
    """Загрузка метрик из директории индекса."""
    metrics_path = Path(index_dir) / "retrieval_metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, 'r') as f:
        return json.load(f)


def create_metrics_table(index_dirs: list, output_path: str):
    """Создание таблицы с метриками для всех методов."""
    methods = []
    all_metrics = []
    
    for idx_dir in index_dirs:
        method_name = Path(idx_dir).name.replace('popqa_index_', '')
        methods.append(method_name)
        
        metrics = load_metrics(idx_dir)
        if metrics:
            all_metrics.append(metrics)
        else:
            all_metrics.append({})
    
    # Ключевые метрики для отображения
    key_metrics = ['recall@1', 'recall@5', 'recall@10', 'recall@20', 
                   'precision@1', 'ndcg@1', 'ndcg@10', 'mrr']
    
    # Создание таблицы
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')
    
    # Заголовок таблицы
    table_data = [['Method'] + key_metrics]
    
    for method, metrics in zip(methods, all_metrics):
        row = [method]
        for metric in key_metrics:
            if metric in metrics:
                val = metrics[metric]['mean']
                row.append(f"{val:.4f}")
            else:
                row.append('-')
        table_data.append(row)
    
    table = ax.table(cellText=table_data[1:], 
                     colLabels=table_data[0],
                     cellLoc='center',
                     loc='center',
                     colColours=['#4472C4'] * (len(key_metrics) + 1))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Стилизация заголовка
    for i in range(len(key_metrics) + 1):
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_facecolor('#4472C4')
    
    # Чередование цветов строк
    for i in range(1, len(methods) + 1):
        for j in range(len(key_metrics) + 1):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E6F2FF')
            else:
                table[(i, j)].set_facecolor('#FFFFFF')
    
    ax.set_title('Retrieval Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def create_length_distribution(dataset_path: str, output_path: str):
    """Создание гистограммы распределения длин документов в токенах."""
    from transformers import AutoTokenizer
    
    print("Загрузка датасета...")
    table = pq.read_table(dataset_path)
    texts = table.column('s_wiki_content').to_pylist()
    print(f"Документов: {len(texts)}")
    
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/huggingface/Qwen/Qwen2-7B-Instruct",
        trust_remote_code=True
    )
    
    print("Токенизация...")
    lengths = []
    for text in texts:
        if text and isinstance(text, str):
            tokens = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    
    # Создание графика
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Гистограмма
    ax1 = axes[0]
    ax1.hist(lengths, bins=50, edgecolor='black', alpha=0.7, color='#4472C4')
    ax1.axvline(lengths.mean(), color='red', linestyle='--', label=f'Mean: {lengths.mean():.0f}')
    ax1.axvline(np.median(lengths), color='orange', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
    ax1.axvline(128, color='green', linestyle='--', label='OSCAR limit (128)')
    ax1.set_xlabel('Token count')
    ax1.set_ylabel('Number of documents')
    ax1.set_title('Document Length Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Кумулятивное распределение
    ax2 = axes[1]
    sorted_lengths = np.sort(lengths)
    cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths) * 100
    ax2.plot(sorted_lengths, cumulative, color='#4472C4', linewidth=2)
    ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(90, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(95, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(128, color='green', linestyle='--', label='OSCAR limit (128)')
    ax2.set_xlabel('Token count')
    ax2.set_ylabel('Cumulative %')
    ax2.set_title('Cumulative Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 2000)
    
    # Добавить перцентили
    p50 = np.percentile(lengths, 50)
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    ax2.annotate(f'P50={p50:.0f}', xy=(p50, 50), xytext=(p50+100, 55),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.annotate(f'P90={p90:.0f}', xy=(p90, 90), xytext=(p90+100, 85),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    ax2.annotate(f'P95={p95:.0f}', xy=(p95, 95), xytext=(p95+100, 88),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Статистика в консоль
    print("\n=== Статистика длин ===")
    print(f"Mean: {lengths.mean():.1f}, Median: {np.median(lengths):.1f}")
    print(f"P50: {p50:.0f}, P90: {p90:.0f}, P95: {p95:.0f}")
    print(f"Max: {lengths.max()}")
    print(f"Documents > 128 tokens: {np.sum(lengths > 128)} ({100*np.sum(lengths > 128)/len(lengths):.1f}%)")


def main():
    # Index directories
    index_dirs = [
        "/data/popqa_index_oscar_mean",
        "/data/popqa_index_oscar_first",
        "/data/popqa_index_oscar_last",
        "/data/popqa_index_oscar_max",
        "/data/popqa_index_oscar_mean_max",
        "/data/popqa_index_salesforce",
    ]
    
    output_dir = Path("/data/popqa_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Таблица метрик
    create_metrics_table(index_dirs, str(output_dir / "metrics_table.png"))
    
    # Распределение длин
    create_length_distribution("/data/popqa_enriched.parquet", 
                               str(output_dir / "doc_length_distribution.png"))


if __name__ == "__main__":
    main()