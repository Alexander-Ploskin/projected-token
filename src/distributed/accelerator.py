from __future__ import annotations
from accelerate import Accelerator


def create_accelerator(mixed_precision: str, gradient_accumulation_steps: int) -> Accelerator:
    # Keep this centralized so every runner is consistent.
    return Accelerator(
        mixed_precision=mixed_precision if mixed_precision != "no" else "no",
        gradient_accumulation_steps=gradient_accumulation_steps,

    )
