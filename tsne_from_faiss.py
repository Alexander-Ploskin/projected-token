
"""Извлечение эмбеддингов из FAISS индекса и построение t-SNE."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import faiss


def load_faiss_index(index_path: str) -> np.ndarray:
    """Загрузка FAISS индекса и извлечение векторов."""
    index = faiss.read_index(index_path)
    # Извлекаем все вектора из индекса
    vectors = index.reconstruct_n(0, index.ntotal)
    return vectors.astype(np.float32)


def plot_tsne(embeddings: np.ndarray, df: pd.DataFrame, method_name: str, output_path: str, 
              top_titles: list, n_samples: int = 5000):
    """t-SNE визуализация."""
    print(f"t-SNE for {method_name} (samples: {n_samples})...")
    
    # Выбираем подвыборку
    n = min(n_samples, len(embeddings))
    indices = np.random.RandomState(42).choice(len(embeddings), n, replace=False)
    
    emb_sample = embeddings[indices]
    df_sample = df.iloc[indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_sample)
    
    # Раскраска по o_wiki_title (топ 5 из всей базы)
    titles = df_sample['o_wiki_title'].fillna('unknown').values
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.colormaps.get_cmap('tab10').resampled(len(top_titles) + 1)
    for i, title in enumerate(top_titles):
        mask = titles == title
        if mask.sum() > 0:
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                       c=[colors(i)], label=title[:30], alpha=0.7, s=15)
    
    # Остальные - серым
    other_mask = ~np.isin(titles, top_titles)
    if other_mask.sum() > 0:
        ax.scatter(emb_2d[other_mask, 0], emb_2d[other_mask, 1], 
                   c='lightgray', label='other', alpha=0.3, s=5)
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE: {method_name}')
    ax.legend(title='o_wiki_title', markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_tsne_combined(embeddings_dict: dict, df: pd.DataFrame, output_path: str, top_titles: list, n_samples: int = 3000):
    """t-SNE для всех методов на одном графике с PCA для выравнивания размерностей."""
    from sklearn.decomposition import PCA
    
    print(f"Combined t-SNE for {len(embeddings_dict)} methods...")
    
    n_methods = len(embeddings_dict)
    n = min(n_samples // n_methods, 800)
    
    # Находим минимальную размерность (но не больше чем n_samples)
    min_dim = min(e.shape[1] for e in embeddings_dict.values())
    min_samples = n
    pca_dim = min(min_dim, min_samples - 1)  # PCA не может использовать больше компонент чем samples
    
    all_embeddings = []
    all_titles = []
    method_indices = []
    
    np.random.seed(42)
    
    for i, (method_name, embeddings) in enumerate(embeddings_dict.items()):
        indices = np.random.choice(len(embeddings), min(n, len(embeddings)), replace=False)
        emb = embeddings[indices]
        
        # PCA до pca_dim
        if emb.shape[1] > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=42)
            emb = pca.fit_transform(emb)
        
        all_embeddings.append(emb)
        all_titles.extend(df.iloc[indices]['o_wiki_title'].fillna('unknown').values)
        method_indices.extend([i] * len(indices))
    
    all_emb = np.vstack(all_embeddings)
    method_indices = np.array(method_indices)
    all_titles = np.array(all_titles)
    
    # Один t-SNE для всех
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    all_emb_2d = tsne.fit_transform(all_emb)
    
    # График 1: по методам
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    colors_methods = plt.colormaps.get_cmap('tab10').resampled(n_methods)
    
    for i, method_name in enumerate(embeddings_dict.keys()):
        mask = method_indices == i
        short_name = method_name.replace('popqa_index_', '')
        ax1.scatter(all_emb_2d[mask, 0], all_emb_2d[mask, 1],
                   c=[colors_methods(i)], label=short_name, alpha=0.6, s=15)
    
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('t-SNE: All Methods Combined')
    ax1.legend(title='Method', markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_by_method.png'), dpi=150)
    plt.close()
    
    # График 2: по o_wiki_title (топ 5 + other)
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    
    colors_titles = plt.colormaps.get_cmap('tab10').resampled(len(top_titles) + 1)
    for i, title in enumerate(top_titles):
        mask = all_titles == title
        if mask.sum() > 0:
            ax2.scatter(all_emb_2d[mask, 0], all_emb_2d[mask, 1],
                       c=[colors_titles(i)], label=title[:30], alpha=0.7, s=15)
    
    other_mask = ~np.isin(all_titles, top_titles)
    if other_mask.sum() > 0:
        ax2.scatter(all_emb_2d[other_mask, 0], all_emb_2d[other_mask, 1],
                   c='lightgray', label='other', alpha=0.3, s=5)
    
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE: All Methods Combined')
    ax2.legend(title='o_wiki_title', markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_by_title.png'), dpi=150)
    plt.close()
    print(f"Saved: {output_path.replace('.png', '_by_method.png')} and {output_path.replace('.png', '_by_title.png')}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE из FAISS индексов")
    parser.add_argument("--dataset-path", default="/data/popqa_enriched.parquet")
    parser.add_argument("--output-dir", default="/data/popqa_visualizations")
    parser.add_argument("--methods", nargs="+", help="Список директорий с индексами")
    parser.add_argument("--n-samples", type=int, default=3000)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка датасета
    print("Загрузка датасета...")
    df = load_dataset(args.dataset_path)
    
    embeddings_dict = {}
    
    for method_dir in args.methods:
        method_path = Path(method_dir)
        index_path = method_path / "index"
        method_name = method_path.name
        
        if not index_path.exists():
            print(f"Skipping {method_name}: no index")
            continue
        
        print(f"Загрузка {method_name}...")
        embeddings = load_faiss_index(str(index_path))
        embeddings_dict[method_name] = embeddings
        print(f"  loaded {embeddings.shape}")
    
    if not embeddings_dict:
        print("No embeddings loaded!")
        return
    
    # Вычисляем топ-5 o_wiki_title по всей базе
    title_counts = df['o_wiki_title'].fillna('unknown').value_counts()
    top_titles = title_counts.head(5).index.tolist()
    print(f"Top 5 titles: {top_titles}")
    
    # Индивидуальные t-SNE
    for method_name, embeddings in embeddings_dict.items():
        plot_tsne(embeddings, df, method_name, 
                 str(output_dir / f"tsne_{method_name}.png"),
                 top_titles=top_titles,
                 n_samples=args.n_samples)
    
    # Объединенный t-SNE (с PCA для выравнивания размерностей)
    if len(embeddings_dict) > 1:
        plot_tsne_combined(embeddings_dict, df, 
                          str(output_dir / "tsne_combined.png"),
                          top_titles=top_titles,
                          n_samples=args.n_samples)
    
    print(f"\nГотово! Файлы в: {output_dir}")


def load_dataset(path: str) -> pd.DataFrame:
    table = pq.read_table(path)
    return table.to_pandas()


if __name__ == "__main__":
    main()