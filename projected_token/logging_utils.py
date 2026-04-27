from __future__ import annotations

from datetime import datetime

import click


def log_step(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    click.echo(click.style(f"[{timestamp}] ", fg="blue") + message)


def log_warning(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    click.echo(click.style(f"[{timestamp}] WARNING ", fg="yellow") + message)


def log_error(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    click.echo(click.style(f"[{timestamp}] ERROR ", fg="red") + message)
