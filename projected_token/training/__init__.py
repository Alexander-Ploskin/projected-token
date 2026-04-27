"""Training package.

Heavy trainer classes are imported lazily by ``projected_token.training.recipes``
so that CLI help and config validation do not import torch/transformers.
"""