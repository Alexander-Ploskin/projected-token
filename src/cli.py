from __future__ import annotations

import dataclasses
from pathlib import Path
import logging
import sys

import click

from src.config.loader import load_experiment_config
from src.distributed.accelerator import create_accelerator
from src.distributed.utils import seed_everything
from src.tracker.factory import build_tracker
from src.runners.pretrain import PretrainRunner
from src.runners.finetune import FinetuneRunner  # NEW


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """xrag-lab: experiments for xRAG-style embedding-to-token compression."""
    pass


@cli.command("pretrain")
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def pretrain_cmd(config: Path) -> None:
    """Run pretraining using settings from CONFIG (YAML)."""
    cfg = load_experiment_config(str(config))

    seed_everything(cfg.train.seed)

    accelerator = create_accelerator(
        mixed_precision=cfg.distributed.mixed_precision,
        gradient_accumulation_steps=cfg.distributed.gradient_accumulation_steps,
    )

    tracker = build_tracker(dataclasses.asdict(cfg.logging))
    if accelerator.is_main_process:
        tracker.log_config(dataclasses.asdict(cfg))

    PretrainRunner(cfg, accelerator, tracker).run()
    tracker.close()


@cli.command("finetune")  # NEW
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def finetune_cmd(config: Path) -> None:
    """Run finetuning using settings from CONFIG (YAML)."""
    cfg = load_experiment_config(str(config))

    seed_everything(cfg.train.seed)

    accelerator = create_accelerator(
        mixed_precision=cfg.distributed.mixed_precision,
        gradient_accumulation_steps=cfg.distributed.gradient_accumulation_steps,
    )

    tracker = build_tracker(dataclasses.asdict(cfg.logging))
    if accelerator.is_main_process:
        tracker.log_config(dataclasses.asdict(cfg))

    FinetuneRunner(cfg, accelerator, tracker).run()
    tracker.close()


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
