import sys
import multiprocessing
from pathlib import Path
from projected_token.retrieval.bm25_baseline import SimpleBM25, _load_beir_dataset
from projected_token.retrieval.metrics.ranking import aggregate_rankings
from projected_token.io import write_json
from projected_token.artifacts import write_metrics_bundle

def process_queries(args):
    bm25, query_texts, query_ids, relevant_map, corpus_ids, search_k = args
    results = []
    for idx, qid in enumerate(query_ids):
        relevant = relevant_map.get(qid, set())
        if not relevant:
            continue
        doc_indices = bm25.search(query_texts[idx], top_k=min(search_k, len(corpus_ids)))
        retrieved = [corpus_ids[i] for i in doc_indices if 0 <= i < len(corpus_ids)]
        results.append((relevant, retrieved))
    return results

def main():
    out_dir = Path("artifacts/results/retrieval/beir_nq_bm25")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading dataset...")
    corpus_ids, corpus_texts, query_ids, query_texts, relevant_map = _load_beir_dataset(Path("/data/beir/nq"), split="test")
    
    print("Building BM25 index...")
    bm25 = SimpleBM25.from_texts(corpus_texts)
    
    print("Searching...")
    search_k = 100
    top_k = [1, 3, 5, 10, 20]
    
    num_processes = multiprocessing.cpu_count()
    chunk_size = len(query_ids) // num_processes + 1
    
    tasks = []
    for i in range(0, len(query_ids), chunk_size):
        q_ids_chunk = query_ids[i:i+chunk_size]
        q_texts_chunk = query_texts[i:i+chunk_size]
        tasks.append((bm25, q_texts_chunk, q_ids_chunk, relevant_map, corpus_ids, search_k))
        
    with multiprocessing.Pool(num_processes) as pool:
        results_chunks = pool.map(process_queries, tasks)
        
    ranking_cases = []
    for chunk in results_chunks:
        ranking_cases.extend(chunk)
        
    print("Aggregating metrics...")
    metrics = aggregate_rankings(ranking_cases, top_k)
    
    per_dataset = {"nq": metrics}
    
    summary = {
        "run_id": "beir_nq_bm25",
        "method": "bm25",
        "split": "test",
        "datasets": ["nq"],
        "top_k": top_k,
        "search_k": search_k,
        "per_dataset": per_dataset,
        "average": metrics,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "job_summary.json", summary)
    print("Done!")

if __name__ == "__main__":
    main()
