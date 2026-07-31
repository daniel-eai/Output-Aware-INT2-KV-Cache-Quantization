#!/usr/bin/env python3
"""Utilities shared by rotation calibration scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

DT = torch.float32
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FPSeq:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def parse_chunks(spec: str) -> list[str]:
    return [value.strip() for value in spec.split(",") if value.strip()]


def quant_int2_ste(
    tensor: torch.Tensor, clip_ratio: float, group_size: int
) -> torch.Tensor:
    shape = tensor.shape
    if shape[-1] % group_size != 0:
        raise ValueError(
            f"last dim {shape[-1]} must be divisible by group_size={group_size}"
        )
    grouped = tensor.reshape(*shape[:-1], shape[-1] // group_size, group_size)
    if clip_ratio and clip_ratio > 0:
        index = min(max(int(clip_ratio * group_size), 0), group_size - 1)
        threshold = grouped.detach().abs().sort(dim=-1).values[..., index : index + 1]
        grouped = torch.minimum(torch.maximum(grouped, -threshold), threshold)
    row_min = grouped.amin(dim=-1, keepdim=True)
    row_max = grouped.amax(dim=-1, keepdim=True)
    scale = (row_max - row_min).clamp(min=1e-8) / 3.0
    zero = -row_min / scale
    unrounded = grouped / scale + zero
    quantized = (unrounded + 0.5).floor().clamp(0, 3)
    straight_through = unrounded + (quantized - unrounded).detach()
    return ((straight_through - zero) * scale).reshape(shape)


def load_layer_entry(state: dict, layer_id: int) -> dict:
    layers = state["layers"]
    entry = layers.get(layer_id, layers.get(str(layer_id)))
    if entry is None:
        raise ValueError(f"missing layer {layer_id} in artifact")
    return entry


def load_headwise_rotation(
    state: dict,
    layer_id: int,
    kv_heads: int,
    head_dim: int,
    device: str = DEV,
) -> torch.Tensor:
    rotation = load_layer_entry(state, layer_id)["rotation"].to(DT).to(device)
    if rotation.shape == (head_dim, head_dim):
        return rotation.unsqueeze(0).expand(kv_heads, -1, -1).contiguous()
    if rotation.shape == (kv_heads, head_dim, head_dim):
        return rotation.contiguous()
    raise ValueError(
        f"layer {layer_id} rotation shape {tuple(rotation.shape)} is "
        f"incompatible with kv_heads={kv_heads}, head_dim={head_dim}"
    )


def load_layer_mu(mean_state: dict, layer_id: int, device: str = DEV) -> torch.Tensor:
    return load_layer_entry(mean_state, layer_id)["mu"].to(DT).to(device).contiguous()


def rotate_headwise(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "lhd,hde->lhe", tensor.to(rotation.dtype), rotation
    ).contiguous()


def derotate_headwise(tensor: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.einsum(
        "lhd,hde->lhe",
        tensor.to(rotation.dtype),
        rotation.transpose(-1, -2),
    ).contiguous()


def restore_exact_window(
    approx: torch.Tensor,
    reference: torch.Tensor,
    prefix: int,
    recent: int,
) -> torch.Tensor:
    output = approx.clone()
    length = output.shape[0]
    if prefix > 0:
        output[: min(prefix, length)] = reference[: min(prefix, length)]
    if recent > 0:
        output[max(0, length - recent) :] = reference[max(0, length - recent) :]
    return output


def load_chunk_tensor(root: Path, name: str, chunk: str) -> torch.Tensor:
    path = root / name / f"{chunk}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu").to(DT)


def load_fp_sequences(
    dump_path: str,
    layer_id: int,
    chunks: Sequence[str],
    prefix: int,
    recent: int,
    device: str = DEV,
) -> list[FPSeq]:
    layer_dir = Path(dump_path) / f"layer_{layer_id}"
    minimum_length = prefix + recent + 8
    sequences = []
    for chunk in chunks:
        queries = load_chunk_tensor(layer_dir, "q", chunk)
        keys = load_chunk_tensor(layer_dir, "k", chunk)
        values = load_chunk_tensor(layer_dir, "v", chunk)
        sequence_lengths = load_chunk_tensor(layer_dir, "seq_lens", chunk).to(
            torch.long
        )
        offset = 0
        for length in sequence_lengths.tolist():
            if length >= minimum_length:
                sequences.append(
                    FPSeq(
                        q=queries[offset : offset + length].to(device),
                        k=keys[offset : offset + length].to(device),
                        v=values[offset : offset + length].to(device),
                    )
                )
            offset += length
    return sequences


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def summarize_ortho(rotation: torch.Tensor) -> float:
    identity = torch.eye(
        rotation.shape[-1],
        device=rotation.device,
        dtype=rotation.dtype,
    )
    return float(
        (rotation.transpose(-1, -2) @ rotation - identity).abs().max().detach().cpu()
    )
