#!/usr/bin/env python3
"""Build K/V rotation checkpoints from Q/K/V calibration dumps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def build_hadamard(size: int) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {size}")
    if size == 1:
        return torch.ones(1, 1, dtype=torch.float64)
    half = build_hadamard(size // 2)
    return torch.cat(
        [torch.cat([half, half], dim=1), torch.cat([half, -half], dim=1)],
        dim=0,
    ) / math.sqrt(2)


def bit_reversal_permutation(size: int) -> torch.Tensor:
    if size < 1 or size & (size - 1):
        raise ValueError(f"Bit-reversal size must be a power of two, got {size}")
    bits = int(math.log2(size))
    return torch.tensor([int(f"{index:0{bits}b}"[::-1], 2) for index in range(size)])


def permutation_matrix(eigenvalues: torch.Tensor) -> torch.Tensor:
    size = len(eigenvalues)
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    bit_reversed = bit_reversal_permutation(size)
    permutation = torch.zeros(size, dtype=torch.long)
    for index in range(size):
        permutation[bit_reversed[index]] = sorted_indices[index]
    return torch.eye(size, dtype=torch.float64)[:, permutation]


def chunk_paths(tensor_dir: Path, chunk_id: str) -> list[Path]:
    if chunk_id != "all":
        path = tensor_dir / f"{chunk_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing calibration tensor: {path}")
        return [path]

    paths = sorted(tensor_dir.glob("*.pt"), key=lambda path: int(path.stem))
    paths = [path for path in paths if int(path.stem) != 0]
    if not paths:
        raise FileNotFoundError(f"No calibration tensors in {tensor_dir}")
    return paths


def load_tensor(layer_dir: Path, name: str, chunk_id: str) -> torch.Tensor:
    tensors = [
        torch.load(path, map_location="cpu").float().double()
        for path in chunk_paths(layer_dir / name, chunk_id)
    ]
    return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)


def symmetric_eigendecomposition(
    covariance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    covariance = (covariance + covariance.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors, eigenvalues


def compute_key_rotation(
    layer_dir: Path, chunk_id: str, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = load_tensor(layer_dir, "q", chunk_id)
    keys = load_tensor(layer_dir, "k", chunk_id)
    query_heads = queries.shape[1] if queries.ndim >= 3 else queries.shape[0]
    key_heads = keys.shape[1] if keys.ndim >= 3 else keys.shape[0]
    group_size = query_heads // key_heads
    queries = queries.reshape(-1, query_heads, head_dim)

    covariance = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    for head in range(key_heads):
        grouped = queries[:, head * group_size : (head + 1) * group_size, :].reshape(
            -1, head_dim
        )
        covariance += grouped.T @ grouped / grouped.shape[0]
    return symmetric_eigendecomposition(covariance / key_heads)


def compute_value_rotation(
    layer_dir: Path, chunk_id: str, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    queries = load_tensor(layer_dir, "q", chunk_id)
    keys = load_tensor(layer_dir, "k", chunk_id)
    values = load_tensor(layer_dir, "v", chunk_id)
    query_heads = queries.shape[1] if queries.ndim >= 3 else queries.shape[0]
    key_heads = keys.shape[1] if keys.ndim >= 3 else keys.shape[0]
    group_size = query_heads // key_heads
    queries = queries.reshape(-1, query_heads, head_dim)
    keys = keys.reshape(-1, key_heads, head_dim)
    values = values.reshape(-1, key_heads, head_dim)
    token_count = queries.shape[0]

    covariance = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    for head in range(key_heads):
        grouped_queries = queries[
            :, head * group_size : (head + 1) * group_size, :
        ].reshape(-1, head_dim)
        head_keys = keys[:, head, :]
        head_values = values[:, head, :]
        query_covariance = (
            grouped_queries.T @ grouped_queries / grouped_queries.shape[0]
        )
        weights = (head_keys @ query_covariance * head_keys).sum(dim=1)
        weights = weights / weights.sum().clamp(min=1e-12) * token_count
        weighted_values = head_values * weights.unsqueeze(1).sqrt()
        covariance += weighted_values.T @ weighted_values / token_count
    return symmetric_eigendecomposition(covariance / key_heads)


def compose_rotation(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    hadamard: torch.Tensor,
) -> torch.Tensor:
    return eigenvectors @ hadamard @ permutation_matrix(eigenvalues)


def layer_directories(dump_path: Path) -> list[Path]:
    directories = [
        path
        for path in dump_path.iterdir()
        if path.is_dir() and path.name.startswith("layer_")
    ]
    return sorted(
        directories, key=lambda path: int(path.name.split("_", maxsplit=1)[1])
    )


def empty_result(objective: str) -> dict:
    return {
        "format_version": 1,
        "objective": objective,
        "source_grouping": "layer",
        "layers": {},
    }


def add_layer(
    result: dict,
    layer_id: int,
    rotation: torch.Tensor,
    eigenvalues: torch.Tensor,
) -> None:
    result["layers"][layer_id] = {
        "layer_id": layer_id,
        "rotation": rotation.float().contiguous(),
        "eigenvalues": eigenvalues.float().contiguous(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-id", default="all")
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hadamard = build_hadamard(args.head_dim)
    key_result = empty_result("qqt_r_h_pbr")
    value_result = empty_result("sst_r_h_pbr")

    directories = layer_directories(args.dump_path)
    if not directories:
        raise ValueError(f"No layer directories found in {args.dump_path}")

    for layer_dir in directories:
        layer_id = int(layer_dir.name.split("_", maxsplit=1)[1])
        key_vectors, key_values = compute_key_rotation(
            layer_dir, args.chunk_id, args.head_dim
        )
        value_vectors, value_values = compute_value_rotation(
            layer_dir, args.chunk_id, args.head_dim
        )
        key_rotation = compose_rotation(key_vectors, key_values, hadamard)
        value_rotation = compose_rotation(value_vectors, value_values, hadamard)
        add_layer(key_result, layer_id, key_rotation, key_values)
        add_layer(value_result, layer_id, value_rotation, value_values)

    key_path = args.output_dir / "k_rotation_qqt_r_h_pbr.pt"
    value_path = args.output_dir / "v_rotation_sst_r_h_pbr.pt"
    torch.save(key_result, key_path)
    torch.save(value_result, value_path)
    print(f"saved {key_path}")
    print(f"saved {value_path}")


if __name__ == "__main__":
    main()
