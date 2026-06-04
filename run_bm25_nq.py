import sys
from pathlib import Path
from projected_token.retrieval.bm25_baseline import evaluate_beir3_bm25
from projected_token.io import write_json

out_dir = Path("artifacts/results/retrieval/beir_nq_bm25")
out_dir.mkdir(parents=True, exist_ok=True)

result = evaluate_beir3_bm25(
    datasets=[{"name": "nq", "path": "/data/beir/nq"}],
    split="test",
    top_k=[1, 3, 5, 10, 20],
    search_k=100,
    output_path=out_dir / "summary.json",
    output_csv_path=out_dir / "summary.csv",
    run_id="beir_nq_bm25",
)
write_json(out_dir / "job_summary.json", result)
