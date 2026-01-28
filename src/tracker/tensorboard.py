from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional

from .base import Tracker

logger = logging.getLogger(__name__)

class TensorboardTracker(Tracker):
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorboardTracker initialized at {log_dir}")
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorboardTracker (tensorboardX) initialized at {log_dir}")
            except ImportError:
                logger.warning("Neither torch.utils.tensorboard nor tensorboardX is installed. Tensorboard logging will be disabled.")

    def log_config(self, cfg: Dict[str, Any]) -> None:
        if self.writer is None:
            return
        # Log config as text
        cfg_str = json.dumps(cfg, indent=2, ensure_ascii=False)
        self.writer.add_text("config", f"```json\n{cfg_str}\n```", global_step=0)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self.writer is None:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, global_step=step)

    def log_text(self, text: str) -> None:
        if self.writer is None:
            return
        # Use a generic tag for random log text
        self.writer.add_text("logs", text)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
