
"""CLI для обучения проектора OSCAR эмбеддингов.

Usage:
    python scripts/train_projector.py mlp --config configs/projector_mlp.yaml
    python scripts/train_projector.py lora --config configs/projector_lora.yaml
    python scripts/train_projector.py full --config configs/projector_full.yaml
"""

import click
import yaml
from pathlib import Path

from evaluation.training.trainer_mlp import MLPTrainer, create_mlp_trainer
from evaluation.training.trainer_lora import LoRATrainer, create_lora_trainer
from evaluation.training.trainer_full import FullFineTuneTrainer, create_full_trainer


@click.group()
def cli():
    """Train OSCAR projector for retrieval."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--epochs", "-e", type=int, default=None, help="Override number of epochs")
def mlp(config_path: str, epochs: int | None):
    """Train MLP projector (Variant A).

    Only the projector is trained, OSCAR is frozen.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if epochs is not None:
        config["epochs"] = epochs

    print("=" * 60)
    print("Variant A: MLP Projector Training")
    print("=" * 60)

    trainer = create_mlp_trainer(config)
    trainer.train(
        num_epochs=config.get("epochs", 3),
        val_every_n_steps=config.get("val_every_n_steps", 100),
    )


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--epochs", "-e", type=int, default=None, help="Override number of epochs")
def lora(config_path: str, epochs: int | None):
    """Train LoRA projector (Variant B).

    MLP projector with LoRA adapter is trained, OSCAR is frozen.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if epochs is not None:
        config["epochs"] = epochs

    print("=" * 60)
    print("Variant B: LoRA Projector Training")
    print("=" * 60)

    trainer = create_lora_trainer(config)
    trainer.train(
        num_epochs=config.get("epochs", 3),
        val_every_n_steps=config.get("val_every_n_steps", 100),
    )


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--epochs", "-e", type=int, default=None, help="Override number of epochs")
def full(config_path: str, epochs: int | None):
    """Train full model (Variant C).

    OSCAR with LoRA + projector are trained together.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if epochs is not None:
        config["epochs"] = epochs

    print("=" * 60)
    print("Variant C: Full Fine-Tune Training")
    print("=" * 60)

    trainer = create_full_trainer(config)
    trainer.train(
        num_epochs=config.get("epochs", 3),
        val_every_n_steps=config.get("val_every_n_steps", 100),
    )


if __name__ == "__main__":
    cli()