"""Per-row clipping and INT2 KV-cache packing kernels."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.kv_quant_kernels import (
    _get_num_scale_groups,
    _is_power_of_two,
)


@triton.jit
def _pretransformed_int2_set_kv_clip_single_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    HP_OFFSET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_QUARTER: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_tokens
    cache_loc = tl.load(loc_ptr + tok_offs, mask=tok_mask, other=0)
    if HP_OFFSET >= 0:
        active = tok_mask & (cache_loc < HP_OFFSET)
    else:
        active = tok_mask

    full_offs = tl.arange(0, HEAD_DIM)
    dim_offs_q = tl.arange(0, BLOCK_QUARTER)
    base = (
        tok_offs[:, None] * input_stride_token
        + head_idx * input_stride_head
        + full_offs[None, :] * input_stride_dim
    )
    rows = tl.load(
        input_ptr + base,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if CLIP_INDEX >= 0:
        sorted_rows = tl.sort(tl.abs(rows))
        pick = (full_offs == CLIP_INDEX)[None, :]
        threshold = tl.sum(tl.where(pick, sorted_rows, 0.0), axis=1)
        rows = tl.minimum(
            tl.maximum(rows, -threshold[:, None]),
            threshold[:, None],
        )

    row_min = tl.min(rows, axis=1)
    row_max = tl.max(rows, axis=1)
    scale = tl.maximum(row_max - row_min, 1e-8) / 3.0
    zero = -row_min / scale

    reshaped = tl.reshape(rows, (BLOCK_TOK, 4, BLOCK_QUARTER))
    permuted = tl.permute(reshaped, (0, 2, 1))
    split_input = tl.reshape(permuted, (BLOCK_TOK, BLOCK_QUARTER, 2, 2))
    even, odd = tl.split(split_input)
    value0, value2 = tl.split(even)
    value1, value3 = tl.split(odd)
    quant0 = (value0 / scale[:, None] + zero[:, None] + 0.5).to(tl.uint8)
    quant1 = (value1 / scale[:, None] + zero[:, None] + 0.5).to(tl.uint8)
    quant2 = (value2 / scale[:, None] + zero[:, None] + 0.5).to(tl.uint8)
    quant3 = (value3 / scale[:, None] + zero[:, None] + 0.5).to(tl.uint8)
    packed = quant0 | (quant1 << 2) | (quant2 << 4) | (quant3 << 6)

    cache_offset = (
        cache_loc[:, None] * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offs_q[None, :] * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, packed, mask=active[:, None])

    sz_offset = cache_loc * sz_stride_loc + head_idx * sz_stride_head
    tl.store(
        scales_zeros_ptr + sz_offset,
        scale,
        mask=active,
    )
    tl.store(
        scales_zeros_ptr + sz_offset + sz_stride_dim,
        zero,
        mask=active,
    )


@triton.jit
def _pretransformed_int2_set_kv_clip_grouped_kernel(
    input_ptr,
    loc_ptr,
    cache_ptr,
    scales_zeros_ptr,
    num_tokens,
    num_heads,
    input_stride_token,
    input_stride_head,
    input_stride_dim,
    cache_stride_loc,
    cache_stride_head,
    cache_stride_dim,
    sz_stride_loc,
    sz_stride_head,
    sz_stride_dim,
    HEAD_DIM: tl.constexpr,
    BLOCK_QUARTER: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_tokens
    cache_loc = tl.load(loc_ptr + tok_offs, mask=tok_mask, other=0)
    if HP_OFFSET >= 0:
        active = tok_mask & (cache_loc < HP_OFFSET)
    else:
        active = tok_mask

    full_offs = tl.arange(0, HEAD_DIM)
    base = (
        tok_offs[:, None] * input_stride_token
        + head_idx * input_stride_head
        + full_offs[None, :] * input_stride_dim
    )
    rows = tl.load(
        input_ptr + base,
        mask=tok_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    if CLIP_INDEX >= 0:
        sorted_rows = tl.sort(tl.abs(rows))
        pick = (full_offs == CLIP_INDEX)[None, :]
        threshold = tl.sum(tl.where(pick, sorted_rows, 0.0), axis=1)
        rows = tl.minimum(
            tl.maximum(rows, -threshold[:, None]),
            threshold[:, None],
        )

    grouped = tl.reshape(rows, (BLOCK_TOK, NUM_GROUPS, GROUP_SIZE))
    value_min = tl.min(grouped, axis=2)
    value_max = tl.max(grouped, axis=2)
    scale = tl.maximum(value_max - value_min, 1e-8) / 3.0
    zero = tl.math.div_rn(-value_min, scale)
    quantized = (
        tl.math.div_rn(grouped, scale[:, :, None]) + zero[:, :, None] + 0.5
    ).to(tl.uint8)

    quantized = tl.reshape(quantized, (BLOCK_TOK, HEAD_DIM))
    reshaped = tl.reshape(quantized, (BLOCK_TOK, 4, BLOCK_QUARTER))
    permuted = tl.permute(reshaped, (0, 2, 1))
    split_input = tl.reshape(permuted, (BLOCK_TOK, BLOCK_QUARTER, 2, 2))
    even, odd = tl.split(split_input)
    quant0, quant2 = tl.split(even)
    quant1, quant3 = tl.split(odd)
    packed = quant0 | (quant1 << 2) | (quant2 << 4) | (quant3 << 6)

    dim_offs_q = tl.arange(0, BLOCK_QUARTER)
    cache_offset = (
        cache_loc[:, None] * cache_stride_loc
        + head_idx * cache_stride_head
        + dim_offs_q[None, :] * cache_stride_dim
    )
    tl.store(cache_ptr + cache_offset, packed, mask=active[:, None])

    group_ids = tl.arange(0, NUM_GROUPS)
    sz_offset = cache_loc[:, None] * sz_stride_loc + head_idx * sz_stride_head
    tl.store(
        scales_zeros_ptr + sz_offset + (group_ids[None, :] * 2) * sz_stride_dim,
        scale,
        mask=active[:, None],
    )
    tl.store(
        scales_zeros_ptr + sz_offset + (group_ids[None, :] * 2 + 1) * sz_stride_dim,
        zero,
        mask=active[:, None],
    )


def _can_use_grouped_clip_kernel(
    head_dim: int, scales_zeros_buffer: torch.Tensor
) -> bool:
    num_groups = _get_num_scale_groups(scales_zeros_buffer)
    if num_groups == 1:
        return True
    if head_dim % num_groups != 0:
        return False
    group_size = head_dim // num_groups
    return _is_power_of_two(num_groups) and _is_power_of_two(group_size)


def _clip_index(clip_ratio: float, head_dim: int) -> int:
    if clip_ratio <= 0.0:
        return -1
    return min(max(int(clip_ratio * head_dim), 0), head_dim - 1)


def _vectorized_elems_per_thread(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 8
    if dtype.is_floating_point and dtype.itemsize == 1:
        return 16
    raise AssertionError(
        f"clip int2 kernel requires bf16 or fp8 input dtype, got {dtype}"
    )


def _pick_block_tok_and_num_warps(
    head_dim: int, elements_per_thread: int
) -> Tuple[int, int]:
    block_tok = 4
    while block_tok * head_dim < 32 * elements_per_thread:
        block_tok *= 2
    total_elems = block_tok * head_dim
    assert total_elems % (32 * elements_per_thread) == 0
    return block_tok, total_elems // (32 * elements_per_thread)


def _launch_single_clip_int2(
    data: torch.Tensor,
    loc: torch.Tensor,
    buffer: torch.Tensor,
    scales_zeros_buffer: torch.Tensor,
    clip_ratio: float,
    hp_global_offset=None,
) -> None:
    num_tokens, num_heads, head_dim = data.shape
    if num_tokens == 0:
        return
    assert _is_power_of_two(
        head_dim
    ), f"clip int2 kernel requires power-of-two head_dim, got {head_dim}"
    elements_per_thread = _vectorized_elems_per_thread(data.dtype)
    block_tok, num_warps = _pick_block_tok_and_num_warps(head_dim, elements_per_thread)
    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pretransformed_int2_set_kv_clip_single_kernel[grid](
        data,
        loc,
        buffer,
        scales_zeros_buffer,
        num_tokens,
        num_heads,
        data.stride(0),
        data.stride(1),
        data.stride(2),
        buffer.stride(0),
        buffer.stride(1),
        buffer.stride(2),
        scales_zeros_buffer.stride(0),
        scales_zeros_buffer.stride(1),
        scales_zeros_buffer.stride(2),
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        HEAD_DIM=head_dim,
        BLOCK_QUARTER=head_dim // 4,
        BLOCK_TOK=block_tok,
        CLIP_INDEX=_clip_index(clip_ratio, head_dim),
        num_warps=num_warps,
        num_stages=1,
    )


def _launch_grouped_clip_int2(
    data: torch.Tensor,
    loc: torch.Tensor,
    buffer: torch.Tensor,
    scales_zeros_buffer: torch.Tensor,
    clip_ratio: float,
    hp_global_offset=None,
) -> None:
    num_tokens, num_heads, head_dim = data.shape
    if num_tokens == 0:
        return
    num_groups = _get_num_scale_groups(scales_zeros_buffer)
    group_size = head_dim // num_groups
    elements_per_thread = _vectorized_elems_per_thread(data.dtype)
    block_tok, num_warps = _pick_block_tok_and_num_warps(head_dim, elements_per_thread)
    assert _is_power_of_two(head_dim)
    assert _is_power_of_two(num_groups) and head_dim % num_groups == 0

    grid = (triton.cdiv(num_tokens, block_tok), num_heads)
    _pretransformed_int2_set_kv_clip_grouped_kernel[grid](
        data,
        loc,
        buffer,
        scales_zeros_buffer,
        num_tokens,
        num_heads,
        data.stride(0),
        data.stride(1),
        data.stride(2),
        buffer.stride(0),
        buffer.stride(1),
        buffer.stride(2),
        scales_zeros_buffer.stride(0),
        scales_zeros_buffer.stride(1),
        scales_zeros_buffer.stride(2),
        HEAD_DIM=head_dim,
        BLOCK_QUARTER=triton.next_power_of_2(head_dim // 4),
        NUM_GROUPS=num_groups,
        GROUP_SIZE=group_size,
        HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
        BLOCK_TOK=block_tok,
        CLIP_INDEX=_clip_index(clip_ratio, head_dim),
        num_warps=num_warps,
        num_stages=1,
    )


def quantized_set_kv_int2_pretransformed_clip_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    clip_ratio_k: float,
    clip_ratio_v: float,
    hp_global_offset=None,
) -> None:
    """Clip already-rotated K/V rows and write their packed INT2 values."""
    assert (
        cache_k.shape == cache_v.shape
    ), f"K/V shape mismatch: {cache_k.shape} vs {cache_v.shape}"
    num_tokens, _num_heads, head_dim = cache_k.shape
    assert (
        head_dim % 4 == 0
    ), f"head_dim must be divisible by 4 for INT2, got {head_dim}"
    if num_tokens == 0:
        return

    k_grouped_ok = _can_use_grouped_clip_kernel(head_dim, k_scales_zeros_buffer)
    v_grouped_ok = _can_use_grouped_clip_kernel(head_dim, v_scales_zeros_buffer)
    if not (k_grouped_ok and v_grouped_ok):
        raise NotImplementedError(
            "clip int2 kernel requires power-of-two group configurations"
        )

    launch_k = (
        _launch_single_clip_int2
        if _get_num_scale_groups(k_scales_zeros_buffer) == 1
        else _launch_grouped_clip_int2
    )
    launch_v = (
        _launch_single_clip_int2
        if _get_num_scale_groups(v_scales_zeros_buffer) == 1
        else _launch_grouped_clip_int2
    )
    launch_k(
        cache_k,
        loc,
        k_cache_buffer,
        k_scales_zeros_buffer,
        clip_ratio_k,
        hp_global_offset,
    )
    launch_v(
        cache_v,
        loc,
        v_cache_buffer,
        v_scales_zeros_buffer,
        clip_ratio_v,
        hp_global_offset,
    )
