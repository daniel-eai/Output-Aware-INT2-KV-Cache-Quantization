#!/usr/bin/env python3
"""Create identity and normalized Hadamard rotation checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rotation.compute_kv_rotation import (
    add_layer,
    build_hadamard,
    empty_result,
)


def write_rotation(
    output_dir: Path, name: str, rotation: torch.Tensor, layers: int
) -> None:
    eigvals = torch.ones(rotation.shape[-1], dtype=torch.float32)
    for side in ("k", "v"):
        state = empty_result(name)
        for layer_id in range(layers):
            add_layer(state, layer_id, rotation, eigvals)
        torch.save(state, output_dir / f"{side}_rotation_{name}.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rotation(
        args.output_dir,
        "identity",
        torch.eye(args.head_dim, dtype=torch.float32),
        args.num_layers,
    )
    write_rotation(
        args.output_dir,
        "hadamard",
        build_hadamard(args.head_dim).float(),
        args.num_layers,
    )
    print(f"wrote fixed rotations to {args.output_dir}")


if __name__ == "__main__":
    main()
