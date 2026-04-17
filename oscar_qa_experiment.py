
"""QA experiment with OSCAR on PopQA dataset - calculate InAcc metric."""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import torch


def load_parquet(path: str, max_rows: int = None):
    """Load dataset from parquet."""
    table = pq.read_table(path)
    if max_rows:
        table = table.slice(0, max_rows)
    df = table.to_pandas()
    return df


def check_answer_in_response(response: str, possible_answers: list) -> bool:
    """Check if any possible answer is in the response."""
    if not response:
        return False
    response_lower = response.lower()
    for answer in possible_answers:
        if answer and isinstance(answer, str) and answer.lower() in response_lower:
            return True
    return False


def run_oscar_qa(df, oscar_model, max_new_tokens: int = 32, batch_size: int = 8):
    """Run OSCAR QA on dataset."""
    results = []
    
    questions = df['question'].tolist()
    documents = df['s_wiki_content'].tolist()
    objs = df['obj'].tolist()
    possible_answers_list = df['possible_answers'].tolist()
    
    # Filter out None/empty values
    valid_data = []
    for i in range(len(df)):
        q = questions[i] if questions[i] else ""
        d = documents[i] if documents[i] else ""
        o = objs[i] if objs[i] else ""
        pa = possible_answers_list[i] if possible_answers_list[i] else [o]
        
        if q.strip() and d.strip():
            valid_data.append({
                'question': q,
                'document': d,
                'obj': o,
                'possible_answers': pa,
            })
    
    print(f"Valid samples: {len(valid_data)} / {len(df)}")
    print(f"Running OSCAR QA, batch_size={batch_size}")
    
    for i in tqdm(range(0, len(valid_data), batch_size), desc="OSCAR QA"):
        batch = valid_data[i:i+batch_size]
        batch_questions = [b['question'] for b in batch]
        batch_docs = [b['document'] for b in batch]
        batch_objs = [b['obj'] for b in batch]
        batch_possible = [b['possible_answers'] for b in batch]
        
        with torch.inference_mode():
            # OSCAR compresses documents and generates answers
            emb = oscar_model._model.compress_documents(
                documents=batch_docs, 
                questions=batch_questions
            )
            answers = oscar_model._model.generate_from_compressed_documents_and_questions(
                questions=batch_questions,
                compressed_documents=emb,
                max_new_tokens=max_new_tokens
            )
        
        for j in range(len(batch_questions)):
            response = answers[j] if j < len(answers) else ""
            possible_answers = batch_possible[j]
            
            # Check if any possible answer is in response
            has_correct = check_answer_in_response(response, possible_answers)
            
            results.append({
                'question': batch_questions[j],
                'document': batch_docs[j][:200] + "..." if len(batch_docs[j]) > 200 else batch_docs[j],
                'obj': batch_objs[j],
                'possible_answers': possible_answers,
                'response': response,
                'has_correct_answer': has_correct,
            })
    
    return results


def calculate_metrics(results):
    """Calculate InAcc and other metrics."""
    total = len(results)
    correct = sum(1 for r in results if r['has_correct_answer'])
    inaccurate = total - correct
    
    accuracy = correct / total if total > 0 else 0
    inacc = inaccurate / total if total > 0 else 0
    
    return {
        'total': total,
        'correct': correct,
        'inaccurate': inaccurate,
        'accuracy': accuracy,
        'inacc': inacc,
    }


def main():
    parser = argparse.ArgumentParser(description="OSCAR QA experiment on PopQA")
    parser.add_argument("--dataset-path", default="/data/popqa_enriched.parquet")
    parser.add_argument("--model-path", default="/data/huggingface/naver/oscar-qwen2-7B")
    parser.add_argument("--max-rows", type=int, default=500, help="Max rows to process")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="/data/popqa_qa_results")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    df = load_parquet(args.dataset_path, args.max_rows)
    print(f"Loaded {len(df)} rows")
    
    # Check for necessary columns
    required_cols = ['question', 's_wiki_content', 'obj', 'possible_answers']
    for col in required_cols:
        if col not in df.columns:
            # Try alternative column names
            if col == 'possible_answers' and 'possible_answers' not in df.columns:
                print(f"Warning: {col} not found, will use obj as possible answer")
                df['possible_answers'] = df['obj'].apply(lambda x: [x] if x else [])
    
    # Initialize OSCAR model
    print(f"Loading OSCAR model from {args.model_path}...")
    from evaluation.models.oscar import OscarModel
    oscar_model = OscarModel(
        model_name_or_path=args.model_path,
        device=args.device,
    )
    
    # Run QA experiment
    output_file = output_dir / f"qa_results_oscar_{args.max_rows}.json"
    
    print(f"\nRunning OSCAR QA...")
    results = run_oscar_qa(
        df, oscar_model, 
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    print("\n=== QA Results ===")
    print(f"Total questions: {metrics['total']}")
    print(f"Correct answers: {metrics['correct']}")
    print(f"Inaccurate answers: {metrics['inaccurate']}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"InAcc: {metrics['inacc']:.4f} ({metrics['inacc']*100:.2f}%)")
    
    # Save metrics
    metrics_file = output_dir / f"metrics_oscar_{args.max_rows}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")
    
    # Show some examples
    print("\n=== Sample Results ===")
    for i, r in enumerate(results[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Q: {r['question'][:100]}...")
        print(f"A: {r['response'][:100]}...")
        print(f"Expected: {r['obj']}")
        print(f"Correct: {r['has_correct_answer']}")


if __name__ == "__main__":
    main()