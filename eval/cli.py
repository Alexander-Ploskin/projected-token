from __future__ import annotations

import logging
import sys
from pathlib import Path
import click

from src.config.loader import load_experiment_config
from eval.runners.factory import EvalRunnerFactory


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """xrag-lab: experiments for xRAG-style embedding-to-token compression."""
    pass


@cli.command("eval")
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def eval_cmd(config: Path) -> None:
    """Run evaluation using settings from CONFIG (YAML). TASK=popqa|..."""
    cfg = load_experiment_config(str(config))  # Dict from YAML
    factory = EvalRunnerFactory(cfg)
    runner = factory.create()
    runner.run()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def main() -> None:
    setup_logging()
    cli()


if __name__ == "__main__":
    main()
