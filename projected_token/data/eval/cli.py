import click
import logging

from projected_token.data.eval.popqa import PopQADatasetPreparer


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


if __name__ == '__main__':
    cli()
