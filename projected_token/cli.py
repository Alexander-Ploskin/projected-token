import click
import logging

from projected_token.dataset import PopQADatasetPreparer
from projected_token.evaluation import PopQAEvaluator


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


@click.group()
def cli():
    """Projected Token CLI tools."""
    pass

@cli.command(name="prepare-popqa")
@click.option('--output-dir', '-o', default='./data/popqa_processed', help='Directory to save the processed dataset')
@click.option('--workers', '-w', default=8, help='Number of parallel workers for downloading')
@click.option('--batch-size', '-b', default=50, help='Batch size for Wikidata resolution')
def prepare_popqa(output_dir, workers, batch_size):
    """
    Prepares the PopQA dataset by resolving Wikidata URIs to Wikipedia content.
    """
    click.echo(f"Starting PopQA preparation with {workers} workers...")
    preparer = PopQADatasetPreparer(output_dir=output_dir, max_workers=workers, batch_size=batch_size)
    preparer.run()
    click.echo("Dataset preparation complete.")

@cli.command(name="evaluate-popqa")
@click.option('--input-file', '-i', required=True, help='Path to the input dataset file (e.g., parquet, csv)')
@click.option('--output-file', '-o', default='./data/evaluation_results_no_context.json', help='Path to save evaluation results')
@click.option('--model', '-m', default=None, help='HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)')
@click.option('--limit', '-n', default=None, type=int, help='Limit number of samples for quick testing')
def evaluate_popqa(input_file, output_file, model, limit):
    """
    Evaluates the LLM on the PopQA dataset (Closed-Book / No Context).
    """
    click.echo(f"Starting NO-CONTEXT evaluation using model {model or 'Qwen/Qwen2.5-1.5B-Instruct'}...")
    evaluator = PopQAEvaluator(model_id=model)
    evaluator.evaluate(input_file=input_file, output_file=output_file, limit=limit, use_context=False)

@cli.command(name="evaluate-popqa-context")
@click.option('--input-file', '-i', required=True, help='Path to the enriched dataset file containing wiki content')
@click.option('--output-file', '-o', default='./data/evaluation_results_context.json', help='Path to save evaluation results')
@click.option('--model', '-m', default=None, help='HuggingFace model ID (default: Qwen/Qwen2.5-1.5B-Instruct)')
@click.option('--limit', '-n', default=None, type=int, help='Limit number of samples for quick testing')
def evaluate_popqa_context(input_file, output_file, model, limit):
    """
    Evaluates the LLM on the PopQA dataset using Retrieved Context (RAG).
    Input file must be the output of 'prepare-popqa' containing 's_wiki_content'.
    """
    click.echo(f"Starting CONTEXT-AWARE evaluation using model {model or 'Qwen/Qwen2.5-1.5B-Instruct'}...")
    evaluator = PopQAEvaluator(model_id=model)
    evaluator.evaluate(input_file=input_file, output_file=output_file, limit=limit, use_context=True)

if __name__ == '__main__':
    cli()
