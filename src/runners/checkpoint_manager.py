from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import List, Optional
from accelerate.logging import get_logger

logger = get_logger(__name__)

class CheckpointManager:
    """
    Manages saving, rotating, and identifying best checkpoints.
    """
    def __init__(
        self, 
        output_dir: str, 
        save_total_limit: Optional[int] = None,
        best_metric_higher_better: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.best_metric_higher_better = best_metric_higher_better
        self.best_metric = float("-inf") if best_metric_higher_better else float("inf")
        
        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize history from disk
        self.checkpoint_history = self._discover_checkpoints()

    def _discover_checkpoints(self) -> List[str]:
        """Discovers existing checkpoints in the output directory."""
        ckpts = sorted(
            [str(p) for p in self.output_dir.glob("checkpoint-*")],
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0
        )
        return ckpts

    def save(
        self,
        accelerator,
        model,
        tokenizer,
        save_func: callable,
        step: Optional[int] = None,
        metric: Optional[float] = None,
        is_final: bool = False
    ):
        """
        Single method to handle saving:
        - Periodic checkpoints (if step provided)
        - Best checkpoint (if metric provided)
        - Final checkpoint (if is_final is True)
        """
        # 1. Final checkpoint
        if is_final:
            last_dir = self.output_dir / "last"
            logger.info(f"Saving final checkpoint: {last_dir}", main_process_only=True)
            save_func(accelerator, model, tokenizer, str(last_dir))

        # 2. Periodic checkpoint (last N)
        if step is not None:
            ckpt_dir = self.output_dir / f"checkpoint-{step}"
            logger.info(f"Saving periodic checkpoint: {ckpt_dir}", main_process_only=True)
            save_func(accelerator, model, tokenizer, str(ckpt_dir))
            
            if accelerator.is_main_process:
                self.checkpoint_history.append(str(ckpt_dir))
                self._rotate_checkpoints()

        # 3. Best checkpoint
        if metric is not None:
            is_better = (
                (metric > self.best_metric) if self.best_metric_higher_better 
                else (metric < self.best_metric)
            )
            if is_better:
                self.best_metric = metric
                best_dir = self.output_dir / "best"
                logger.info(f"New best metric={metric:.4f}! Saving to {best_dir}", main_process_only=True)
                save_func(accelerator, model, tokenizer, str(best_dir))

    def _rotate_checkpoints(self):
        """Removes old checkpoints if the limit is exceeded."""
        if self.save_total_limit is None:
            return
            
        while len(self.checkpoint_history) > self.save_total_limit:
            to_delete = self.checkpoint_history.pop(0)
            if os.path.exists(to_delete):
                shutil.rmtree(to_delete)
                logger.info(f"Removed old checkpoint: {to_delete}", main_process_only=True)
