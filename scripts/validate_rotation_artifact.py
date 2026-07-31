#!/usr/bin/env python3
"""Validate finite and orthogonal K/V rotation checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def layer_entries(state: dict) -> list[tuple[int, dict]]:
    layers = state.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("checkpoint has no non-empty 'layers' mapping")
    return sorted((int(key), value) for key, value in layers.items())


def validate(path: Path, expected_layers: int | None, atol: float) -> None:
    state = torch.load(path, map_location="cpu")
    entries = layer_entries(state)
    if expected_layers is not None and len(entries) != expected_layers:
        raise ValueError(
            f"{path}: found {len(entries)} layers, expected {expected_layers}"
        )
    worst = 0.0
    for layer_id, entry in entries:
        rotation = entry["rotation"].float()
        if not torch.isfinite(rotation).all():
            raise ValueError(f"{path}: non-finite rotation at layer {layer_id}")
        if rotation.ndim not in (2, 3) or rotation.shape[-1] != rotation.shape[-2]:
            raise ValueError(
                f"{path}: invalid rotation shape {tuple(rotation.shape)} at layer {layer_id}"
            )
        eye = torch.eye(rotation.shape[-1], dtype=rotation.dtype)
        error = (rotation.transpose(-1, -2) @ rotation - eye).abs().max().item()
        worst = max(worst, error)
        if error > atol:
            raise ValueError(
                f"{path}: orthogonality error {error:.3e} at layer {layer_id}"
            )
    print(f"{path}: layers={len(entries)}, max_orthogonality_error={worst:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-rotation", type=Path, required=True)
    parser.add_argument("--v-rotation", type=Path, required=True)
    parser.add_argument("--expected-layers", type=int)
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()
    validate(args.k_rotation, args.expected_layers, args.atol)
    validate(args.v_rotation, args.expected_layers, args.atol)


if __name__ == "__main__":
    main()
