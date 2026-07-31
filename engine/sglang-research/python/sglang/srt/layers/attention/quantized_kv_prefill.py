"""Shared helpers for int2 quantized KV prefill.

These utilities were originally inline methods on :class:`TritonAttnBackend`.
They're factored out here so other attention backends (notably FA3) can reuse
the same rotation (Hadamard for MHA-int2, Oscar for unified HP+int2) and
HP+int2 aware dequantization pipeline, keeping a single source of truth for
int2 prefill semantics.

Callers are expected to drive the higher level flow themselves:

  1. Call :func:`prepare_quantized_extend_qkv` before writing KV to the pool.
     Pass the returned ``pre_rotated_k`` / ``pre_rotated_v`` to
     ``set_kv_buffer(..., already_hadamard_transformed=True)`` so the pool does
     not rotate again.
  2. Call :func:`dequantize_prefix_kv` to materialize contiguous per-token K/V
     for the cached prefix (any mix of HP and int2 tiers is handled).
  3. Concatenate the dequantized prefix with the newly rotated extend K/V and
     run ``flash_attn_varlen_func`` (or equivalent).
  4. Call :func:`apply_inverse_v_rotation` on the attention output if the V
     path was rotated (Hadamard self-inverse, or ``result @ R_v.T`` for
     Oscar).

The module is intentionally framework-light: it only depends on torch +
the JIT Hadamard kernel (``sglang.jit_kernel.hadamard``, with an optional
``fast_hadamard_transform`` fast path) and on the pool surface
(``get_raw_key_buffer`` / ``get_key_scales_zeros`` / ...) that both
``MHATokenToKVPool`` and ``UnifiedInt2HPKVPool`` already implement.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional, Tuple

from sglang.srt.environ import envs
import torch
import triton
import triton.language as tl

try:
    from fast_hadamard_transform import hadamard_transform
except ImportError:
    from sglang.jit_kernel.hadamard import hadamard_transform

from sglang.srt.mem_cache.kv_quant_kernels import (
    _get_num_scale_groups,
    dequantize_kv_int2_triton,
)

_OSCAR_TRITON_Q_ROTATION_ENABLED = envs.SGLANG_OSCAR_TRITON_Q_ROTATION.get()
_OSCAR_TRITON_Q_ROTATION_MAX_ROWS = envs.SGLANG_OSCAR_TRITON_Q_ROTATION_MAX_ROWS.get()
_OSCAR_TRITON_Q_ROTATION_BLOCK_M = envs.SGLANG_OSCAR_TRITON_Q_ROTATION_BLOCK_M.get()
_OSCAR_TRITON_Q_ROTATION_DIRECT_KV = envs.SGLANG_OSCAR_TRITON_Q_ROTATION_DIRECT_KV.get()
_OSCAR_TRITON_Q_ROTATION_INPLACE = envs.SGLANG_OSCAR_TRITON_Q_ROTATION_INPLACE.get()

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


def apply_segmented_hadamard_transform(
    tensor: torch.Tensor,
    hadamard_order: Optional[int] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Segmented (blockwise) FWHT along the last dim, matching the fused int2
    write / decode kernels.

    ``hadamard_order`` defaults to ``envs.HADAMARD_ORDER.get()`` when
    unspecified. ``out_dtype``, when set, casts the input before transforming
    (used by callers that want to upcast bf16 → fp32 internally before the
    rotation, then cast back). The returned tensor has the same shape as
    ``tensor`` and a dtype determined by ``hadamard_transform``'s rules
    (typically the input dtype after the optional cast). Self-inverse with
    the ``1/sqrt(order)`` pre-normalization.

    Single canonical implementation; previously duplicated in three places
    (memory_pool.py, this file, and inline in triton_backend.py).
    """
    if hadamard_order is None:
        hadamard_order = envs.HADAMARD_ORDER.get()
    if out_dtype is not None:
        tensor = tensor.to(out_dtype)
    return hadamard_transform(
        tensor.view(
            *tensor.shape[:-1],
            tensor.shape[-1] // hadamard_order,
            hadamard_order,
        )
        / math.sqrt(hadamard_order)
    ).view_as(tensor)


# Backward-compatible alias for module-internal callers.
_apply_segmented_hadamard_transform = apply_segmented_hadamard_transform


@triton.jit
def _oscar_q_rotation_dot_kernel(
    q,
    R,
    out,
    n_rows: tl.constexpr,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    r_stride_head: tl.constexpr,
    r_stride_in: tl.constexpr,
    r_stride_out: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    head = tl.program_id(1)
    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = tl.arange(0, BLOCK_D)
    offs_o = tl.arange(0, BLOCK_D)

    q_tile = tl.load(
        q
        + rows[:, None] * q_stride_row
        + head * q_stride_head
        + offs_i[None, :] * q_stride_dim,
        mask=rows[:, None] < n_rows,
        other=0.0,
    )
    kv_head = head // Q_PER_KV
    r_base = R + kv_head * r_stride_head
    r_mat = tl.load(
        r_base + offs_i[:, None] * r_stride_in + offs_o[None, :] * r_stride_out
    )
    acc = tl.dot(q_tile, r_mat, out_dtype=tl.float32)
    tl.store(
        out
        + rows[:, None] * out_stride_row
        + head * out_stride_head
        + offs_o[None, :] * out_stride_dim,
        acc,
        mask=rows[:, None] < n_rows,
    )


@triton.jit
def _oscar_q_rotation_kernel(
    q,
    R,
    out,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    r_stride_head: tl.constexpr,
    r_stride_in: tl.constexpr,
    r_stride_out: tl.constexpr,
    out_stride_row: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offs_i = tl.arange(0, BLOCK_D)
    offs_o = tl.arange(0, BLOCK_D)

    q_vec = tl.load(
        q + row * q_stride_row + head * q_stride_head + offs_i * q_stride_dim
    )
    kv_head = head // Q_PER_KV
    r_base = R + kv_head * r_stride_head
    r_mat = tl.load(
        r_base + offs_i[:, None] * r_stride_in + offs_o[None, :] * r_stride_out
    )
    acc = tl.sum(q_vec[:, None].to(tl.float32) * r_mat.to(tl.float32), axis=0)
    tl.store(
        out + row * out_stride_row + head * out_stride_head + offs_o * out_stride_dim,
        acc,
    )


def _apply_oscar_q_rotation_triton(
    q: torch.Tensor,
    R: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not _OSCAR_TRITON_Q_ROTATION_ENABLED:
        return None
    if not q.is_cuda or not R.is_cuda:
        return None
    if q.dim() != 3 or R.dim() != 3:
        return None
    if q.shape[-1] != 128 or R.shape[-2:] != (128, 128):
        return None
    if q.dtype != R.dtype:
        return None
    q_heads = q.shape[-2]
    kv_heads = R.shape[0]
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        return None
    max_rows = _OSCAR_TRITON_Q_ROTATION_MAX_ROWS
    if max_rows > 0 and q.shape[0] > max_rows:
        return None

    if out is q and q.stride(-1) != 1:
        return None
    q_contig = q if q.stride(-1) == 1 else q.contiguous()
    R_contig = R if R.stride(-1) == 1 else R.contiguous()
    if out is None:
        out = torch.empty(q_contig.shape, device=q_contig.device, dtype=R_contig.dtype)
    elif (
        out.shape != q_contig.shape
        or out.device != q_contig.device
        or out.dtype != R_contig.dtype
        or out.stride(-1) != 1
    ):
        return None
    r_stride_head, r_stride_in, r_stride_out = R_contig.stride()

    n_rows, q_heads, _ = q_contig.shape
    q_per_kv = q_heads // R_contig.shape[0]
    block_m = max(1, _OSCAR_TRITON_Q_ROTATION_BLOCK_M)
    if block_m > 1 and n_rows >= block_m:
        _oscar_q_rotation_dot_kernel[((n_rows + block_m - 1) // block_m, q_heads)](
            q_contig,
            R_contig,
            out,
            n_rows,
            q_contig.stride(0),
            q_contig.stride(1),
            q_contig.stride(2),
            r_stride_head,
            r_stride_in,
            r_stride_out,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            Q_PER_KV=q_per_kv,
            BLOCK_M=block_m,
            BLOCK_D=128,
        )
    else:
        _oscar_q_rotation_kernel[(n_rows, q_heads)](
            q_contig,
            R_contig,
            out,
            q_contig.stride(0),
            q_contig.stride(1),
            q_contig.stride(2),
            r_stride_head,
            r_stride_in,
            r_stride_out,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            Q_PER_KV=q_per_kv,
            BLOCK_D=128,
        )
    return out


def _apply_oscar_q_rotation(
    q: torch.Tensor,
    R: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q_triton = _apply_oscar_q_rotation_triton(q, R, out=out)
    if q_triton is not None:
        return q_triton
    R_q = R
    if R.dim() == 3 and q.shape[-2] != R.shape[0]:
        R_q = _expand_oscar_rotation_for_q(R, q.shape[-2], "K")
    return _apply_oscar_rotation(q, R_q)


def _pool_uses_oscar_rotation(kv_pool) -> bool:
    """True for the mixed HP+int2 unified pool that loads per-layer Oscar
    rotation matrices. Non-oscar int2 pools (MHA without mixed HP+int2
    storage) implicitly use the legacy segmented Hadamard rotation.
    """
    return getattr(kv_pool, "_R_k", None) is not None


def _apply_oscar_rotation(tensor: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Apply Oscar rotation along the last dim.

    ``R`` can be either the legacy layerwise matrix ``[D, D]`` or a headwise
    matrix table ``[H, D, D]``. In the headwise case the tensor must have shape
    ``[..., H, D]`` and each head receives its own rotation.
    """
    if R.dim() == 2:
        return (tensor.to(R.dtype) @ R).contiguous()
    if R.dim() != 3:
        raise ValueError(f"Oscar rotation must have rank 2 or 3, got {R.dim()}")
    if tensor.dim() < 2:
        raise ValueError(
            f"Headwise Oscar rotation needs tensor rank >= 2, got {tensor.dim()}"
        )
    if tensor.shape[-2] != R.shape[0]:
        raise ValueError(
            f"Headwise Oscar rotation head mismatch: tensor has {tensor.shape[-2]} "
            f"heads but rotation has {R.shape[0]}"
        )
    if tensor.shape[-1] != R.shape[-2] or R.shape[-2] != R.shape[-1]:
        raise ValueError(
            f"Headwise Oscar rotation dim mismatch: tensor last dim "
            f"{tensor.shape[-1]}, rotation shape {tuple(R.shape)}"
        )
    return torch.einsum("...hd,hde->...he", tensor.to(R.dtype), R).contiguous()


def _expand_oscar_rotation_for_q(
    R: torch.Tensor, q_head_num: int, name: str
) -> torch.Tensor:
    """Expand a KV-headwise rotation table to query heads for GQA.

    Layerwise rotations are returned unchanged. For headwise rotations, each
    KV head rotation is repeated for its query-head group.
    """
    if R.dim() == 2:
        return R
    if R.dim() != 3:
        raise ValueError(f"Oscar {name} rotation must have rank 2 or 3, got {R.dim()}")
    kv_head_num = R.shape[0]
    if q_head_num % kv_head_num != 0:
        raise ValueError(
            f"Cannot map {q_head_num} query heads to {kv_head_num} KV-head "
            f"Oscar {name} rotations"
        )
    group = q_head_num // kv_head_num
    if group == 1:
        return R
    return R.repeat_interleave(group, dim=0)


def _expand_oscar_k_rotation_for_q(R_k: torch.Tensor, q_head_num: int) -> torch.Tensor:
    """Expand a KV-headwise K rotation table to query heads for GQA."""
    return _expand_oscar_rotation_for_q(R_k, q_head_num, "K")


def _get_oscar_k_rotation_for_q(
    kv_pool, layer_idx: int, q_head_num: int
) -> torch.Tensor:
    """Return the original K rotation after validating its GQA mapping.

    Headwise OptR rotations remain KV-headwise. The Triton Q kernels map each
    query head to ``q_head // q_per_kv`` directly, avoiding a repeated
    Q-headwise rotation table and improving reuse when GQA heads share R_k.
    """
    R_k = kv_pool._R_k[layer_idx]
    if R_k.dim() == 2:
        return R_k
    if R_k.dim() != 3 or q_head_num % R_k.shape[0] != 0:
        raise ValueError(
            f"Cannot map {q_head_num} query heads to K rotation shape "
            f"{tuple(R_k.shape)}"
        )
    if not _OSCAR_TRITON_Q_ROTATION_DIRECT_KV:
        cache = getattr(kv_pool, "_R_q_cache", None)
        if cache is None:
            cache = {}
            setattr(kv_pool, "_R_q_cache", cache)
        key = (int(layer_idx), int(q_head_num))
        R_q = cache.get(key)
        if R_q is None or R_q.device != R_k.device or R_q.dtype != R_k.dtype:
            R_q = _expand_oscar_k_rotation_for_q(R_k, q_head_num).contiguous()
            cache[key] = R_q
        return R_q
    return R_k


def _apply_pool_oscar_k_rotation_to_q(
    q: torch.Tensor,
    kv_pool,
    layer_idx: int,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply a pool K rotation, optionally reusing Q storage for decode."""
    R_k = _get_oscar_k_rotation_for_q(kv_pool, layer_idx, q.shape[-2])
    out = q if inplace and _OSCAR_TRITON_Q_ROTATION_INPLACE and R_k.dim() == 3 else None
    return _apply_oscar_q_rotation(
        q,
        R_k,
        out=out,
    )


def _apply_oscar_k_rotation_to_q(q: torch.Tensor, R_k: torch.Tensor) -> torch.Tensor:
    """Apply layerwise/headwise K rotation to query heads."""
    return _apply_oscar_q_rotation(q, R_k)


def _apply_oscar_v_matrix_to_q(result: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Apply a layerwise/headwise V-space matrix to query-head outputs."""
    return _apply_oscar_rotation(
        result,
        _expand_oscar_rotation_for_q(M, result.shape[-2], "V"),
    )


def _apply_oscar_v_inverse_to_q(
    result: torch.Tensor, R_v: torch.Tensor
) -> torch.Tensor:
    """Apply the orthogonal inverse of a layerwise/headwise V rotation."""
    if R_v.dim() == 2:
        M = R_v.T
    elif R_v.dim() == 3:
        M = R_v.transpose(-1, -2)
    else:
        raise ValueError(f"Oscar V rotation must have rank 2 or 3, got {R_v.dim()}")
    return _apply_oscar_v_matrix_to_q(result, M)


def prepare_quantized_extend_qkv(
    kv_pool,
    layer: "RadixAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_already_hadamard_transformed: bool = False,
    kv_already_hadamard_transformed: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Apply the pool's active rotation to Q/K/V for int2 extend.

    Returns the (possibly) rotated ``q, k, v`` tensors and a flag indicating
    whether the attention output must be inverse-rotated afterwards. For
    non-int2 pools this is a no-op.
    """
    need_v_inverse = False
    kv_dtype = kv_pool.dtype
    if kv_dtype != "int2":
        return q, k, v, need_v_inverse

    if _pool_uses_oscar_rotation(kv_pool):
        layer_idx = layer.layer_id - kv_pool.start_layer
        R_k = kv_pool._R_k[layer_idx]
        R_v = kv_pool._R_v[layer_idx]
        v_rotation_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))
        if not q_already_hadamard_transformed:
            q = _apply_pool_oscar_k_rotation_to_q(q, kv_pool, layer_idx)
        if not kv_already_hadamard_transformed:
            if hasattr(kv_pool, "apply_oscar_k_cache_transform"):
                k = kv_pool.apply_oscar_k_cache_transform(layer.layer_id, k)
            else:
                k = _apply_oscar_rotation(k, R_k)
            if v_rotation_absorbed:
                v = v.to(R_v.dtype).contiguous()
            else:
                v = _apply_oscar_rotation(v, R_v)
        need_v_inverse = True
        return q, k, v, need_v_inverse

    if not q_already_hadamard_transformed:
        q = _apply_segmented_hadamard_transform(q)
    if not kv_already_hadamard_transformed:
        k = _apply_segmented_hadamard_transform(k)
        v = _apply_segmented_hadamard_transform(v)
    need_v_inverse = True
    return q, k, v, need_v_inverse


@triton.jit
def _mixed_prefix_dequant_kernel(
    prefix_indices_ptr,
    quant_ptr,
    scales_zeros_ptr,
    hp_ptr,
    out_ptr,
    num_tokens,
    num_heads,
    head_dim: tl.constexpr,
    quant_stride_token: tl.constexpr,
    quant_stride_head: tl.constexpr,
    quant_stride_dim: tl.constexpr,
    sz_stride_token: tl.constexpr,
    sz_stride_head: tl.constexpr,
    sz_stride_dim: tl.constexpr,
    hp_stride_token: tl.constexpr,
    hp_stride_head: tl.constexpr,
    hp_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    offs = tl.arange(0, BLOCK_DIM)
    dim_mask = offs < head_dim

    slot = tl.load(prefix_indices_ptr + token_idx)
    is_hp = slot >= HP_OFFSET
    quarter_dim = head_dim // 4

    byte_offsets = offs % quarter_dim
    packed = tl.load(
        quant_ptr
        + slot * quant_stride_token
        + head_idx * quant_stride_head
        + byte_offsets * quant_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=0,
    )
    shift = (offs // quarter_dim) * 2
    q = ((packed >> shift) & 0x03).to(tl.float32)

    group_ids = offs // GROUP_SIZE
    scale = tl.load(
        scales_zeros_ptr
        + slot * sz_stride_token
        + head_idx * sz_stride_head
        + (group_ids * 2) * sz_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=1.0,
    ).to(tl.float32)
    zero = tl.load(
        scales_zeros_ptr
        + slot * sz_stride_token
        + head_idx * sz_stride_head
        + (group_ids * 2 + 1) * sz_stride_dim,
        mask=(~is_hp) & dim_mask,
        other=0.0,
    ).to(tl.float32)
    quant_val = (q - zero) * scale

    hp_slot = slot - HP_OFFSET
    hp_val = tl.load(
        hp_ptr
        + hp_slot * hp_stride_token
        + head_idx * hp_stride_head
        + offs * hp_stride_dim,
        mask=is_hp & dim_mask,
        other=0.0,
    )
    out_val = tl.where(is_hp, hp_val, quant_val)
    tl.store(
        out_ptr
        + token_idx * out_stride_token
        + head_idx * out_stride_head
        + offs * out_stride_dim,
        out_val,
        mask=(token_idx < num_tokens) & (head_idx < num_heads) & dim_mask,
    )


def _mixed_prefix_dequantize_tensor(
    prefix_indices: torch.Tensor,
    quantized: torch.Tensor,
    scales_zeros: torch.Tensor,
    hp: torch.Tensor,
    hp_offset: int,
    head_dim: int,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    num_tokens = prefix_indices.shape[0]
    num_heads = quantized.shape[1]
    out = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=model_dtype,
        device=prefix_indices.device,
    )
    if num_tokens == 0:
        return out
    num_groups = _get_num_scale_groups(scales_zeros)
    group_size = head_dim // num_groups
    grid = (num_tokens, num_heads)
    _mixed_prefix_dequant_kernel[grid](
        prefix_indices,
        quantized,
        scales_zeros,
        hp,
        out,
        num_tokens,
        num_heads,
        head_dim,
        quantized.stride(0),
        quantized.stride(1),
        quantized.stride(2),
        scales_zeros.stride(0),
        scales_zeros.stride(1),
        scales_zeros.stride(2),
        hp.stride(0),
        hp.stride(1),
        hp.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HP_OFFSET=int(hp_offset),
        GROUP_SIZE=group_size,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )
    return out


def dequantize_prefix_kv(
    kv_pool,
    layer_id: int,
    prefix_indices: torch.Tensor,
    model_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dequantize the prefix slots referenced by ``prefix_indices`` into dense
    ``[num_tokens, head_num, head_dim]`` tensors in ``model_dtype``.

    Supports both ``MHATokenToKVPool`` (all slots are int2) and
    ``UnifiedInt2HPKVPool`` (some slots are HP, others are int2; classified
    by ``slot >= hp_global_offset`` -- HP slot ids start at exactly
    ``hp_global_offset``).

    Grouped scales (``scales.shape[-1] > 2``) are handled by
    ``dequantize_kv_int2_triton`` internally.
    """
    device = prefix_indices.device
    raw_k_buffer = kv_pool.get_raw_key_buffer(layer_id)
    raw_v_buffer = kv_pool.get_raw_value_buffer(layer_id)
    head_num = raw_k_buffer.shape[1]
    head_dim = (
        raw_k_buffer.shape[-1] * 4
        if kv_pool.dtype == "int2"
        else raw_k_buffer.shape[-1]
    )
    v_head_dim = (
        raw_v_buffer.shape[-1] * 4
        if kv_pool.dtype == "int2"
        else raw_v_buffer.shape[-1]
    )
    if prefix_indices.numel() == 0:
        return (
            torch.empty(
                (0, head_num, head_dim),
                dtype=model_dtype,
                device=device,
            ),
            torch.empty(
                (0, head_num, v_head_dim),
                dtype=model_dtype,
                device=device,
            ),
        )

    prefix_indices = prefix_indices.to(torch.int64)
    if (
        getattr(kv_pool, "mixed_kv_enabled", None) is not None
        and kv_pool.mixed_kv_enabled()
    ):
        assert (
            kv_pool.dtype == "int2"
        ), f"Unsupported quantized KV dtype: {kv_pool.dtype}"
        return (
            _mixed_prefix_dequantize_tensor(
                prefix_indices,
                raw_k_buffer,
                kv_pool.get_key_scales_zeros(layer_id),
                kv_pool.get_hp_key_buffer(layer_id),
                kv_pool.hp_global_offset,
                head_dim,
                model_dtype,
            ),
            _mixed_prefix_dequantize_tensor(
                prefix_indices,
                raw_v_buffer,
                kv_pool.get_value_scales_zeros(layer_id),
                kv_pool.get_hp_value_buffer(layer_id),
                kv_pool.hp_global_offset,
                v_head_dim,
                model_dtype,
            ),
        )

    raw_k = raw_k_buffer[prefix_indices]
    raw_v = raw_v_buffer[prefix_indices]
    scales_k = kv_pool.get_key_scales_zeros(layer_id)[prefix_indices]
    scales_v = kv_pool.get_value_scales_zeros(layer_id)[prefix_indices]
    assert kv_pool.dtype == "int2", f"Unsupported quantized KV dtype: {kv_pool.dtype}"
    return (
        dequantize_kv_int2_triton(raw_k, scales_k, head_dim, model_dtype),
        dequantize_kv_int2_triton(raw_v, scales_v, v_head_dim, model_dtype),
    )


def apply_inverse_v_rotation(
    result: torch.Tensor,
    kv_pool,
    layer: "RadixAttention",
    need_v_inverse: bool,
) -> torch.Tensor:
    """Apply the inverse V rotation on an attention output tensor, when
    required. Hadamard is self-inverse (segmented FWHT), Oscar inverts as
    ``result @ R_v.T`` in ``R_v``'s dtype for layerwise V, or the equivalent
    per-head transform for KV-headwise V.

    For KV-headwise V, ``result`` must have shape ``[..., q_heads, v_head_dim]``;
    callers should reshape beforehand if their output is stored flattened.
    """
    if not need_v_inverse or kv_pool.dtype != "int2":
        return result
    if _pool_uses_oscar_rotation(kv_pool):
        layer_idx = layer.layer_id - kv_pool.start_layer
        R_v = kv_pool._R_v[layer_idx]
        return _apply_oscar_v_inverse_to_q(result, R_v).contiguous()
    return _apply_segmented_hadamard_transform(result)


@triton.jit
def _build_prefix_indices_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    prefix_lens_ptr,
    prefix_indptr_ptr,
    out_ptr,
    req_to_token_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per request; walk the prefix in fixed BLOCK_SIZE chunks.
    # A single ``tl.arange(0, next_pow2(max_prefix_len))`` silently corrupts the
    # gather once the block exceeds ~32768 elements (per-program register/block
    # limit on this Triton/sm90 build): with a 64k prefix the block jumps to
    # 65536 and the long-context prefix slot ids come back garbled, so the int2
    # prefill attention reads the wrong rows and decoding derails past
    # ``32768 + chunked_prefill_size`` tokens. Looping over a small fixed block
    # removes the dependence on prefix length (mirrors _count_mixed_hp_lens /
    # _scatter_mixed_kv_indices, which already loop).
    req_idx = tl.program_id(0)
    req_pool_idx = tl.load(req_pool_indices_ptr + req_idx)
    prefix_len = tl.load(prefix_lens_ptr + req_idx)
    out_start = tl.load(prefix_indptr_ptr + req_idx)
    row_base = req_pool_idx * req_to_token_stride
    num_loops = tl.cdiv(prefix_len, BLOCK_SIZE)
    for i in range(num_loops):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < prefix_len
        slots = tl.load(
            req_to_token_ptr + row_base + offs,
            mask=mask,
            other=0,
        ).to(tl.int64)
        tl.store(out_ptr + out_start + offs, slots, mask=mask)


def _cpu_int_list(values) -> Optional[list[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def build_prefix_indices_from_req_to_token(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cache_seqlens_cpu=None,
) -> torch.Tensor:
    """Gather valid prefix slot ids per request as a flat 1-D tensor by
    reading ``req_to_token`` directly (no page striding / page_size division).

    Required by the unified mixed HP+int2 prefill path. HP slot ids in that
    pool are encoded ``HP_OFFSET + hp_page_id`` and adjacent positions can
    map to arbitrary HP page ids, so the page-table-based reconstruction
    used by :func:`build_prefix_indices_from_page_table` returns garbled HP
    slot ids and reads the wrong rows out of ``hp_k_buffer`` / ``hp_v_buffer``.

    Inputs:
        req_to_token       : int32 [max_req_slots, max_context_len]
        req_pool_indices   : int64 [bs] -- per-request row index
        cache_seqlens      : int32 or int64 [bs] -- valid prefix length

    Returns:
        flat int64 1-D tensor of slot ids, in (request, position) order
        consistent with what the FA varlen kernel expects when concatenated
        with the freshly written extend slots.
    """
    device = req_to_token.device
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    cache_seqlens_cpu = _cpu_int_list(cache_seqlens_cpu)
    if cache_seqlens_cpu is None:
        raise ValueError(
            "build_prefix_indices_from_req_to_token requires CPU prefix lengths "
            "to avoid CUDA boolean-index synchronization"
        )
    total_prefix = sum(cache_seqlens_cpu)
    out = torch.empty((total_prefix,), dtype=torch.int64, device=device)
    if total_prefix == 0:
        return out
    prefix_lens_cpu = torch.tensor(cache_seqlens_cpu, dtype=torch.int32)
    prefix_indptr_cpu = torch.empty((bs + 1,), dtype=torch.int32)
    prefix_indptr_cpu[0] = 0
    prefix_indptr_cpu[1:] = torch.cumsum(prefix_lens_cpu, dim=0)
    prefix_lens = prefix_lens_cpu.to(device, non_blocking=True)
    prefix_indptr = prefix_indptr_cpu.to(device, non_blocking=True)
    # Fixed block: the kernel loops over the prefix, so BLOCK_SIZE no longer
    # has to cover ``max_prefix_len`` in one shot. A single oversized block
    # (next_power_of_2 of a 64k prefix -> 65536) silently corrupted the gather
    # on this Triton build, garbling long-context prefixes past 32768 tokens.
    _build_prefix_indices_kernel[(bs,)](
        req_to_token,
        req_pool_indices.to(torch.int64),
        prefix_lens,
        prefix_indptr,
        out,
        req_to_token.stride(0),
        BLOCK_SIZE=2048,
        num_warps=8,
        num_stages=1,
    )
    return out
