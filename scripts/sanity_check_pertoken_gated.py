#!/usr/bin/env python3
"""Quick sanity check for MEMProjectorGated before E11 training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from projected_token.encoders.projector import MEMProjectorGated


def main() -> None:
    proj = MEMProjectorGated(28672, 1024, 768, 0.15)
    proj.eval()
    param_count = sum(p.numel() for p in proj.parameters())
    print(f"Parameter count: {param_count:,}")

    x_flat = torch.randn(4, 28672)
    out_flat = proj(x_flat)
    assert out_flat.shape == (4, 768), f"Wrong flat shape: {out_flat.shape}"
    assert not torch.isnan(out_flat).any(), "NaN in flat output"

    x_3d = torch.randn(4, 8, 3584)
    out_3d = proj(x_3d)
    assert out_3d.shape == (4, 768), f"Wrong 3D shape: {out_3d.shape}"
    assert not torch.isnan(out_3d).any(), "NaN in 3D output"

    with torch.no_grad():
        h0 = torch.relu(proj.token_proj(x_3d[0:1]))
        h1 = torch.relu(proj.token_proj(x_3d[1:2]))
        gates_0 = torch.softmax(proj.gate(h0), dim=1)
        gates_1 = torch.softmax(proj.gate(h1), dim=1)
    assert not torch.allclose(gates_0, gates_1), "Gates should differ for different inputs"

    print("Architecture OK")


if __name__ == "__main__":
    main()
