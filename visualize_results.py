
"""Визуализация результатов поиска для PopQA."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_retrieval_details(path: str) -> list:
    with open(path, 'r') as f:
        return json.load(f)


def load_dataset(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


def load_embeddings(index_path: str) -> np.ndarray:
    """Загрузка эмбеддингов из numpy файла."""
    # Ищем embeddings рядом с индексом
    index_dir = Path(index_path).parent
    emb_path = index_dir / "embeddings.npy"
    if emb_path.exists():
        return np.load(emb_path)
    
    # Пробуем other name
    emb_path = index_dir / "documents.npy"
    if emb_path.exists():
        return np.load(emb_path)
    
    return None


def plot_tsne(embeddings: np.ndarray, df: pd.DataFrame, title: str, output_path: str):
    """t-SNE визуализация эмбеддингов с раскраской по pop категориям."""
    print(f"t-SNE for {title}...")
    
    # Выбираем подвыборку если слишком много
    n_samples = min(5000, len(embeddings))
    indices = np.random.choice(len(embeddings), n_samples, replace=False)
    
    emb_sample = embeddings[indices]
    df_sample = df.iloc[indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d = tsne.fit_transform(emb_sample)
    
    # Категории по pop
    pops = df_sample['s_pop'].values
    pop_bins = [0, 10, 100, 1000, 10000, float('inf')]
    pop_labels = ['<10', '10-100', '100-1K', '1K-10K', '>10K']
    pop_cats = pd.cut(pops, bins=pop_bins, labels=pop_labels)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(pop_labels)))
    for i, label in enumerate(pop_labels):
        mask = pop_cats == label
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.6, s=10)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE: {title}')
    ax.legend(title='s_pop', markerscale=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_comparison(results: dict, output_path: str):
    """Bar chart с метриками для разных методов."""
    methods = list(results.keys())
    
    metrics_to_plot = ['recall@1', 'recall@5', 'recall@10', 'mrr']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        values = [results[method].get(m, {}).get('mean', 0) for m in metrics_to_plot]
        bars = ax.bar(x + i * width - 0.4 + width/2, values, width, label=method)
        # Add value labels
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Retrieval Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('@', '\n@') for m in metrics_to_plot])
    ax.legend()
    ax.set_ylim(0, max(0.5, max(
        results[m].get('recall@1', {}).get('mean', 0) for m in methods
    ) * 1.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_metrics_by_pop(df: pd.DataFrame, details: list, title: str, output_path: str):
    """График Recall@K по категориям популярности."""
    # Добавляем pop и hit к details
    for d, row in zip(details, df.itertuples()):
        d['s_pop'] = row.s_pop
        d['o_pop'] = row.o_pop
    
    # Категории pop
    pop_bins = [0, 10, 100, 1000, 10000, float('inf')]
    pop_labels = ['<10', '10-100', '100-1K', '1K-10K', '>10K']
    
    pop_recall1 = defaultdict(list)
    pop_recall5 = defaultdict(list)
    pop_recall10 = defaultdict(list)
    
    for d in details:
        cat = pd.cut([d['s_pop']], bins=pop_bins, labels=pop_labels)[0]
        pop_recall1[cat].append(d.get('hit_in_top1', 0))
        pop_recall5[cat].append(d.get('hit_in_top5', 0))
        pop_recall10[cat].append(d.get('hit_in_top10', 0))
    
    # Средние
    pop_labels_list = [l for l in pop_labels if l in pop_recall1]
    recall1_means = [np.mean(pop_recall1[l]) for l in pop_labels_list]
    recall5_means = [np.mean(pop_recall5[l]) for l in pop_labels_list]
    recall10_means = [np.mean(pop_recall10[l]) for l in pop_labels_list]
    
    x = np.arange(len(pop_labels_list))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, recall1_means, width, label='Recall@1')
    ax.bar(x, recall5_means, width, label='Recall@5')
    ax.bar(x + width, recall10_means, width, label='Recall@10')
    
    ax.set_xlabel('Subject Popularity')
    ax.set_ylabel('Recall')
    ax.set_title(f'Recall by Popularity: {title}')
    ax.set_xticks(x)
    ax.set_xticklabels(pop_labels_list)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_topk_hit_rate(details: list, title: str, output_path: str):
    """Hit rate при разных k."""
    k_values = list(range(1, 21))
    hit_rates = []
    
    for k in k_values:
        hits = sum(1 for d in details if len(set(d['retrieved_top20'][:k]) & set(d['relevant_doc_indices'])) > 0)
        hit_rates.append(hits / len(details))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, hit_rates, 'b-o', markersize=4)
    ax.set_xlabel('K')
    ax.set_ylabel('Hit Rate')
    ax.set_title(f'Hit Rate@K: {title}')
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(hit_rates) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_pop_distribution(df: pd.DataFrame, output_path: str):
    """Распределение популярности субъектов."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Гистограмма
    ax = axes[0]
    pops = df['s_pop'].values
    ax.hist(np.log10(pops + 1), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('log10(s_pop + 1)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Subject Popularity (log scale)')
    
    # По категориям
    ax = axes[1]
    pop_bins = [0, 10, 100, 1000, 10000, float('inf')]
    pop_labels = ['<10', '10-100', '100-1K', '1K-10K', '>10K']
    counts = pd.cut(df['s_pop'], bins=pop_bins, labels=pop_labels).value_counts()
    ax.bar(pop_labels, counts.values, color='steelblue', edgecolor='black')
    ax.set_xlabel('Popularity Category')
    ax.set_ylabel('Count')
    ax.set_title('Documents by Popularity Category')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Визуализация результатов")
    parser.add_argument("--dataset-path", default="/data/popqa_enriched.parquet")
    parser.add_argument("--output-dir", default="/data/popqa_visualizations")
    parser.add_argument("--methods", nargs="+", help="Список методов (названия директорий)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка датасета
    print("Загрузка датасета...")
    df = load_dataset(args.dataset_path)
    print(f"Датасет: {len(df)} записей")
    
    # Визуализация распределения pop
    plot_pop_distribution(df, str(output_dir / "pop_distribution.png"))
    
    # Если указаны методы - обрабатываем каждый
    if args.methods:
        metrics_results = {}
        
        for method in args.methods:
            method_dir = Path(method)
            details_path = method_dir / "retrieval_details.json"
            
            if not details_path.exists():
                print(f"Skipping {method}: no retrieval_details.json")
                continue
            
            print(f"\nОбработка {method}...")
            details = load_retrieval_details(str(details_path))
            
            # Используем basename для имени файла
            method_name = Path(method).name
            
            # Метрики по pop
            plot_metrics_by_pop(df, details, method_name, 
                               str(output_dir / f"recall_by_pop_{method_name}.png"))
            
            # Hit rate по k
            plot_topk_hit_rate(details, method_name, 
                              str(output_dir / f"hit_rate_{method_name}.png"))
            
            # Загружаем метрики
            metrics_path = method_dir / "retrieval_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics_results[method_name] = json.load(f)
            
            # t-SNE если есть эмбеддинги
            emb_path = method_dir / "embeddings.npy"
            if emb_path.exists():
                embeddings = np.load(emb_path)
                plot_tsne(embeddings, df, method_name, 
                         str(output_dir / f"tsne_{method_name}.png"))
        
        # Сравнение методов
        if metrics_results:
            plot_metrics_comparison(metrics_results, 
                                   str(output_dir / "metrics_comparison.png"))
    
    print(f"\nВсе визуализации сохранены в: {output_dir}")


if __name__ == "__main__":
    main()