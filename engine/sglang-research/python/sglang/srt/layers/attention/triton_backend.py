from __future__ import annotations
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.quantized_kv_prefill import (
    _apply_pool_oscar_k_rotation_to_q,
    _apply_oscar_v_inverse_to_q,
    _pool_uses_oscar_rotation,
    apply_inverse_v_rotation,
    apply_segmented_hadamard_transform,
    dequantize_prefix_kv,
    prepare_quantized_extend_qkv,
)
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_core_count,
    get_int_env_var,
    next_power_of_2,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput


@triton.jit
def _count_mixed_hp_lens_kernel(
    req_to_token_ptr,  # int32 [num_req_slots, max_ctx]
    req_pool_indices_ptr,  # int64 [bs]
    seq_lens_ptr,  # int32 [bs]
    hp_lens_ptr,  # int32 [bs]
    rtt_stride_row,
    HP_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Count per-request HP lengths without dense mask materialization.

    This keeps the req-pool indirection fused with the tier classification so
    ``_build_mixed_kv_indices`` never has to materialize a gathered ``rows``
    tensor or per-token boolean masks. The quant tier length is derived from
    ``seq_len - hp_len`` on the Python side.
    """
    req = tl.program_id(0)
    req_pool_idx = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)

    hp_count = tl.zeros((), dtype=tl.int32)
    num_loops = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loops):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < seq_len
        slot = tl.load(
            req_to_token_ptr + req_pool_idx * rtt_stride_row + offs.to(tl.int64),
            mask=valid,
            other=0,
        ).to(tl.int64)
        hp_count += tl.sum((valid & (slot >= HP_OFFSET)).to(tl.int32), axis=0)

    tl.store(hp_lens_ptr + req, hp_count)


@triton.jit
def _scatter_mixed_kv_indices_kernel(
    req_to_token_ptr,  # int32 [num_req_slots, max_ctx]
    req_pool_indices_ptr,  # int64 [bs]
    seq_lens_ptr,  # int32 or int64 [bs] -- cast inside
    hp_kv_indptr_ptr,  # int32 [bs + 1]   already cumsum'd
    quant_kv_indptr_ptr,  # int32 [bs + 1]   already cumsum'd
    hp_kv_indices_ptr,  # int64 [*] destination, pre-sized
    quant_kv_indices_ptr,  # int64 [*] destination, pre-sized
    rtt_stride_row,
    HP_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Slot-id-classified scatter into hp/quant index buffers, one block per req.

    For each request i we walk ``req_to_token[req_pool_indices[i], 0..seq_len)``
    in ``BLOCK_SIZE`` chunks. Each lane decides whether its slot id is HP
    (``slot >= HP_OFFSET``) or quant, then contributes to within-block exclusive
    prefix sums that act as scatter offsets into the pre-cumsum'd
    ``hp_kv_indptr`` / ``quant_kv_indptr`` tier-local layout. No masked-select,
    no Python bs-loop, and no D2H sync: stride and offset arithmetic is all on
    device with shapes known statically.
    """
    req = tl.program_id(0)
    req_pool_idx = tl.load(req_pool_indices_ptr + req).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int32)
    hp_base = tl.load(hp_kv_indptr_ptr + req).to(tl.int64)
    quant_base = tl.load(quant_kv_indptr_ptr + req).to(tl.int64)

    # Running counters for the chunked scatter. Triton tracks these as scalar
    # SSA values that accumulate across the Python-side for loop below.
    hp_running = tl.zeros((), dtype=tl.int32)
    quant_running = tl.zeros((), dtype=tl.int32)

    num_loops = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loops):
        offs = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offs < seq_len
        slot = tl.load(
            req_to_token_ptr + req_pool_idx * rtt_stride_row + offs.to(tl.int64),
            mask=valid,
            other=0,
        ).to(tl.int64)
        # HP slot ids start at exactly ``HP_OFFSET`` (page 0 is a valid HP
        # page), so the boundary is ``>=`` not ``>``. The unified pool /
        # allocator (``unified_kv_pool._split_global_locs``,
        # ``unified_kv_allocator.free``) and the GPU flush kernel
        # (``gpu_flush_int2``) all classify by ``>=``; using ``>`` here would
        # misclassify HP slot id ``HP_OFFSET`` as quant and read OOB from
        # the quant buffer.
        is_hp = valid & (slot >= HP_OFFSET)
        is_quant = valid & (
            slot < HP_OFFSET
        )  # == valid & ~is_hp; explicit to avoid ~bool dtype quirks

        hp_inc = is_hp.to(tl.int32)
        quant_inc = is_quant.to(tl.int32)

        # tl.cumsum gives an inclusive prefix; subtract the lane value to get
        # the exclusive prefix (= rank of this lane among HP/quant entries
        # within this block).
        hp_rank = tl.cumsum(hp_inc, axis=0) - hp_inc
        quant_rank = tl.cumsum(quant_inc, axis=0) - quant_inc

        tl.store(
            hp_kv_indices_ptr + hp_base + (hp_running + hp_rank).to(tl.int64),
            slot - HP_OFFSET,
            mask=is_hp,
        )
        tl.store(
            quant_kv_indices_ptr
            + quant_base
            + (quant_running + quant_rank).to(tl.int64),
            slot,
            mask=is_quant,
        )

        hp_running += tl.sum(hp_inc, axis=0)
        quant_running += tl.sum(quant_inc, axis=0)


def logit_capping_mod(logit_capping_method, logit_cap):
    # positive logit_cap -> tanh cap
    if logit_capping_method == "tanh":
        return logit_cap
    else:
        raise ValueError()


@dataclass
class ForwardMetadata:
    attn_logits: torch.Tensor
    attn_lse: torch.Tensor
    max_extend_len: int
    num_kv_splits: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: torch.Tensor
    custom_mask: torch.Tensor
    mask_indptr: torch.Tensor
    # Sliding window
    window_kv_indptr: torch.Tensor
    window_kv_indices: torch.Tensor
    window_num_kv_splits: torch.Tensor
    window_kv_offsets: torch.Tensor
    # Separate attn_logits for SWA layers when v_head_dim differs
    swa_attn_logits: Optional[torch.Tensor] = None
    # Per-tier indptr/indices for the unified single-launch mixed int2 path.
    mixed_hp_kv_indptr: Optional[torch.Tensor] = None
    mixed_hp_kv_indices: Optional[torch.Tensor] = None
    mixed_quant_kv_indptr: Optional[torch.Tensor] = None
    mixed_quant_kv_indices: Optional[torch.Tensor] = None
    # Single combined stage-1 scratch: HP splits in the first hp_max slots,
    # quant splits in the next quant_max slots. Stage-2 reduces both in one
    # launch.
    mixed_attn_logits: Optional[torch.Tensor] = None
    mixed_attn_lse: Optional[torch.Tensor] = None
    # Per-tier split counts populated by get_num_kv_splits_triton.
    mixed_hp_num_kv_splits: Optional[torch.Tensor] = None
    mixed_quant_num_kv_splits: Optional[torch.Tensor] = None


class TritonAttnBackend(AttentionBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
    ):
        # Lazy import to avoid the initialization of cuda context
        from sglang.srt.layers.attention.triton_ops.decode_attention import (
            decode_attention_fwd,
            decode_attention_fwd_int2_unified,
            decode_attention_fwd_quantized,
        )
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            build_unified_kv_indices,
            extend_attention_fwd,
            extend_attention_fwd_mixed_int2,
            extend_attention_fwd_unified,
        )

        super().__init__()

        self.decode_attention_fwd = torch.compiler.disable(decode_attention_fwd)
        self.decode_attention_fwd_quantized = torch.compiler.disable(
            decode_attention_fwd_quantized
        )
        self.decode_attention_fwd_int2_unified = torch.compiler.disable(
            decode_attention_fwd_int2_unified
        )
        self.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
        self.extend_attention_fwd_mixed_int2 = torch.compiler.disable(
            extend_attention_fwd_mixed_int2
        )
        self.extend_attention_fwd_unified = torch.compiler.disable(
            extend_attention_fwd_unified
        )
        self.build_unified_kv_indices = torch.compiler.disable(build_unified_kv_indices)
        self._dump_kv_done_layers = set()
        self._dump_saved_tokens = {}
        self._dump_chunk_idx = {}
        self._mixed_tiled_prefix_logged = False

        # Parse args
        self.skip_prefill = skip_prefill
        max_bs = model_runner.req_to_token_pool.size
        self.sliding_window_size = model_runner.sliding_window_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.num_draft_tokens = model_runner.server_args.speculative_num_draft_tokens
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        # The decode triton kernel derives attn_lse offsets from attn_logits
        # strides via integer division by v_head_dim (the "// Lv" trick in
        # _fwd_kernel_stage1/stage2), so attn_logits.shape[-1] must exactly
        # match the layer's v_head_dim. For hybrid SWA models where SWA and
        # full-attention layers use different v_head_dim (e.g. Gemma 4:
        # swa=256, full=512), we allocate a second buffer for SWA layers.
        full_v_head_dim = model_runner.model_config.v_head_dim
        swa_v_head_dim = model_runner.model_config.swa_v_head_dim
        if self.sliding_window_size is not None and swa_v_head_dim != full_v_head_dim:
            self.v_head_dim = full_v_head_dim
            self.swa_v_head_dim = swa_v_head_dim
        elif (
            model_runner.hybrid_gdn_config is not None
            or model_runner.kimi_linear_config is not None
            or model_runner.linear_attn_model_spec is not None
        ):
            # For hybrid linear models, layer_id = 0 may not be full attention
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
            self.swa_v_head_dim = None
        else:
            self.v_head_dim = getattr(
                model_runner.token_to_kv_pool,
                "v_head_dim",
                model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1],
            )
            self.swa_v_head_dim = None
        self.max_context_len = model_runner.model_config.context_len
        self.enable_mixed_kv = (
            getattr(model_runner.token_to_kv_pool, "mixed_kv_enabled", None) is not None
            and model_runner.token_to_kv_pool.mixed_kv_enabled()
            and not self.use_mla
            and self.sliding_window_size is None
            and self.swa_v_head_dim is None
        )
        self.mixed_hp_prefix_tokens = (
            model_runner.token_to_kv_pool.hp_prefix_tokens
            if self.enable_mixed_kv
            else 0
        )
        self.mixed_hp_recent_tokens = (
            model_runner.token_to_kv_pool.hp_recent_tokens
            if self.enable_mixed_kv
            else 0
        )
        self.mixed_hp_global_offset = (
            model_runner.token_to_kv_pool.hp_global_offset
            if self.enable_mixed_kv
            else 0
        )
        # Mixed-KV decode uses a fixed HP split count because the HP window is
        # bounded by ``hp_prefix + hp_recent + flush_interval - 1`` tokens.
        # ``SGLANG_MIXED_KV_HP_MAX_SPLITS`` is therefore the direct per-request
        # HP cap for the unified int2 decode path.
        self.max_hp_kv_splits = (
            envs.SGLANG_MIXED_KV_HP_MAX_SPLITS.get() if self.enable_mixed_kv else 0
        )
        # Output dtype for per-tier intermediate buffers in the mixed-KV path.
        self.model_dtype = model_runner.dtype
        self.device = model_runner.device
        self.use_mixed_tiled_prefix_extend = (
            envs.SGLANG_MIXED_KV_TILED_PREFIX_EXTEND.get()
            and torch.cuda.get_device_capability(self.device)[0] < 9
        )
        self.device_core_count = get_device_core_count(model_runner.gpu_id)
        self.static_kv_splits = get_bool_env_var(
            "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
        )
        self.max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        self.use_a100_mixed_kv_split_schedule = (
            self.enable_mixed_kv
            and get_bool_env_var("SGLANG_MIXED_KV_A100_SPLIT_SCHEDULE", "false")
            and torch.cuda.get_device_capability(self.device) == (8, 0)
        )
        if self.use_a100_mixed_kv_split_schedule:
            if self.max_kv_splits < 24:
                raise ValueError(
                    "SGLANG_MIXED_KV_A100_SPLIT_SCHEDULE requires "
                    "--triton-attention-num-kv-splits >= 24"
                )
            print(
                "[MixedKV] A100 quant-history split schedule enabled: "
                "<=1024:8, <=4096:16, >4096:24"
            )

        self.allow_bidirectional_attention_in_extend = (
            model_runner.server_args.disable_cuda_graph
            and (model_runner.server_args.chunked_prefill_size == -1)
        )

        # Decide whether enable deterministic inference with batch-invariant operations
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )

        # Configure deterministic inference settings
        if self.enable_deterministic:
            # Use fixed split tile size for batch invariance
            self.split_tile_size = get_int_env_var(
                "SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE", 256
            )
            # Set static_kv_splits to False to use deterministic logic instead
            self.static_kv_splits = False
        else:
            self.split_tile_size = (
                model_runner.server_args.triton_attention_split_tile_size
            )

        if self.split_tile_size is not None:
            self.max_kv_splits = (
                self.max_context_len + self.split_tile_size - 1
            ) // self.split_tile_size

        # Check arguments
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        # Initialize buffers
        # TODO(Jianan Ji): Make sure it behaves as expected when kv_indptr_buf is provided and sliding window is enabled
        if kv_indptr_buf is None:
            self.kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            )
        else:
            self.kv_indptr = kv_indptr_buf

        # If sliding window is enabled, we might need two sets of buffers
        # because of interleaved attention types (e.g. for Gemma3)
        self.window_kv_indptr = None
        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indptr_buf is None:
                self.window_kv_indptr = torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
            else:
                # When provided a buffer, create a clone for the second buffer
                self.window_kv_indptr = torch.zeros_like(kv_indptr_buf)

        if not self.skip_prefill:
            self.qo_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

            self.mask_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int64, device=model_runner.device
            )

        # Initialize forward metadata
        self.forward_metadata: ForwardMetadata = None

        self.cuda_graph_custom_mask = None

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
        max_kv_splits: Optional[int] = None,
    ):
        """Fill ``num_kv_splits`` with a per-sequence split count.

        ``max_kv_splits`` overrides the per-call upper bound (defaults to
        ``self.max_kv_splits``). The mixed-KV path uses the override to cap
        the HP-side split count independently of the quant/primary side.
        """
        if max_kv_splits is None:
            max_kv_splits = self.max_kv_splits
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert (
            num_group * num_seq == num_token
        ), f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"

        # Legacy dynamic splitting logic (non-deterministic)
        if (
            self.static_kv_splits or self.device_core_count <= 0
        ) and not self.enable_deterministic:
            num_kv_splits.fill_(max_kv_splits)
            return

        # deterministic
        if self.split_tile_size is not None and self.enable_deterministic:
            # expand seq_lens to match num_token
            if num_group > 1:
                expanded_seq_lens = seq_lens.repeat_interleave(num_group)
            else:
                expanded_seq_lens = seq_lens

            num_kv_splits[:] = torch.clamp(
                (expanded_seq_lens + self.split_tile_size - 1) // self.split_tile_size,
                max=max_kv_splits,
            )
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def get_mixed_quant_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        quant_lens: torch.Tensor,
    ):
        """Plan only the INT2-history tier, with an optional SM80 schedule."""
        if not self.use_a100_mixed_kv_split_schedule:
            self.get_num_kv_splits(num_kv_splits, quant_lens)
            return

        num_token, num_seq = num_kv_splits.shape[0], quant_lens.shape[0]
        num_group = num_token // num_seq
        assert num_group * num_seq == num_token
        schedule_seq = 256 if num_seq < 256 else triton.next_power_of_2(num_seq)
        get_a100_mixed_quant_kv_splits_triton[(1,)](
            num_kv_splits,
            quant_lens,
            num_seq,
            num_group,
            self.max_kv_splits,
            MAX_NUM_SEQ=schedule_seq,
        )

    def _build_mixed_kv_indices(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        hp_kv_indptr: torch.Tensor,
        hp_kv_indices: torch.Tensor,
        quant_kv_indptr: torch.Tensor,
        quant_kv_indices: torch.Tensor,
        bs: int,
    ):
        """Classify each token's slot id as HP vs quant and scatter into the
        caller-provided per-tier index buffers.

        Sync-free on the decode hot path. Previously this routine ran a
        ``for i in range(bs)`` Python loop with ``rows[hp_mask[i]]``
        masked-selects whose output shape is data-dependent -- each
        masked-select forces a cudaStreamSynchronize so PyTorch can learn the
        size. That was the single biggest CPU-critical-path blocker in
        mixed-KV decode after the flush pipeline was fused.

        The replacement:
          * ``hp_kv_indptr`` is built from a Triton HP-length counting kernel
            that streams ``req_to_token`` through the req-pool indirection --
            no dense gather/mask materialization, no sync.
          * ``quant_kv_indptr`` is derived from the full sequence lengths minus
            the HP prefix sum, so there is no separate quant-length pass.
          * The per-(req, pos) scatter into ``hp_kv_indices`` /
            ``quant_kv_indices`` happens inside a single triton kernel
            (``_scatter_mixed_kv_indices_kernel``) that walks each request's
            ``[0, seq_len)`` range in ``BLOCK_SIZE`` chunks and uses
            ``tl.cumsum`` for within-block ranks. No Python bs-loop, no
            masked-select, no sync.
        """
        seq_lens = seq_lens[:bs]
        req_pool_indices = req_pool_indices[:bs].to(torch.int64)
        # Cast seq_lens to int32 once; both mixed-KV Triton kernels want
        # int32. Keeps the conversion off the hot path's per-step alloc trail.
        seq_lens_i32 = seq_lens.to(torch.int32)
        hp_lens = torch.empty_like(seq_lens_i32)
        # Count directly from ``req_to_token`` so the hot path no longer
        # materializes a dense gathered ``rows`` tensor or boolean masks.
        _count_mixed_hp_lens_kernel[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens_i32,
            hp_lens,
            self.req_to_token.stride(0),
            HP_OFFSET=int(self.mixed_hp_global_offset),
            BLOCK_SIZE=512,
            num_warps=2,
            num_stages=1,
        )

        # indptr = exclusive prefix sum of per-req lengths. ``cumsum`` + slice
        # assignment are shape-static so no D2H read is forced. The leading
        # ``[0]`` element stays at zero from the buffer's ``torch.zeros``
        # allocation; assigning a Python scalar there would force a sync H2D
        # copy that blocks the CPU on prior decode work, recreating the
        # ~1.5 ms inter-step bubble.
        hp_kv_indptr[1 : bs + 1] = torch.cumsum(hp_lens, dim=0)
        quant_kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens_i32, dim=0)
        quant_kv_indptr[1 : bs + 1] -= hp_kv_indptr[1 : bs + 1]

        # Single triton launch scatters the tier-classified slot ids directly
        # into the pre-sized destination buffers. BLOCK_SIZE here is the
        # per-request chunk size; picking 512 matches
        # ``create_flashinfer_kv_indices_triton`` and balances occupancy
        # against the ``tl.cumsum`` reduction depth.
        _scatter_mixed_kv_indices_kernel[(bs,)](
            self.req_to_token,
            req_pool_indices,
            seq_lens_i32,
            hp_kv_indptr,
            quant_kv_indptr,
            hp_kv_indices,
            quant_kv_indices,
            self.req_to_token.stride(0),
            HP_OFFSET=int(self.mixed_hp_global_offset),
            BLOCK_SIZE=512,
            num_warps=2,
            num_stages=1,
        )
        return seq_lens_i32 - hp_lens

    def _forward_extend_quantized_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        pre_rotated_q: Optional[torch.Tensor] = None,
        pre_rotated_k: Optional[torch.Tensor] = None,
        pre_rotated_v: Optional[torch.Tensor] = None,
        need_v_inverse_override: Optional[bool] = None,
        prefix_indptr: Optional[torch.Tensor] = None,
        prefix_indices: Optional[torch.Tensor] = None,
        window_size: tuple[int, int] = (-1, -1),
    ):
        kv_pool = forward_batch.token_to_kv_pool
        q3 = (
            pre_rotated_q
            if pre_rotated_q is not None
            else q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        )
        k3 = pre_rotated_k if pre_rotated_k is not None else k.contiguous()
        v3 = pre_rotated_v if pre_rotated_v is not None else v.contiguous()
        if need_v_inverse_override is None:
            q3, k3, v3, need_v_inverse = prepare_quantized_extend_qkv(
                kv_pool,
                layer,
                q3,
                k3,
                v3,
                q_already_hadamard_transformed=pre_rotated_q is not None,
                kv_already_hadamard_transformed=(
                    pre_rotated_k is not None and pre_rotated_v is not None
                ),
            )
        else:
            need_v_inverse = need_v_inverse_override

        prefix_indptr = (
            prefix_indptr
            if prefix_indptr is not None
            else self.forward_metadata.kv_indptr
        )
        prefix_indices = (
            prefix_indices
            if prefix_indices is not None
            else self.forward_metadata.kv_indices
        )
        softcap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        use_tiled_prefix = (
            self.use_mixed_tiled_prefix_extend
            and getattr(kv_pool, "mixed_kv_enabled", lambda: False)()
            and prefix_indices.numel() > 0
            and window_size == (-1, -1)
            and q3.shape[-1] == 128
            and k3.shape[-1] == 128
            and v3.shape[-1] == 128
            and q3.dtype == k3.dtype == v3.dtype
            and not softcap
            and kv_pool.get_key_scales_zeros(layer.layer_id).shape[-1] == 2
            and kv_pool.get_value_scales_zeros(layer.layer_id).shape[-1] == 2
        )
        if use_tiled_prefix:
            if not self._mixed_tiled_prefix_logged:
                print(
                    "[MIXED-TILED-PREFIX] inline HP/INT2 dequantization active "
                    "for extend attention"
                )
                self._mixed_tiled_prefix_logged = True
            result = torch.empty(
                (q3.shape[0], q3.shape[1], v3.shape[-1]),
                dtype=q3.dtype,
                device=q3.device,
            )
            self.extend_attention_fwd_mixed_int2(
                q3.contiguous(),
                k3.contiguous(),
                v3.contiguous(),
                result,
                kv_pool.get_raw_key_buffer(layer.layer_id),
                kv_pool.get_raw_value_buffer(layer.layer_id),
                kv_pool.get_key_scales_zeros(layer.layer_id),
                kv_pool.get_value_scales_zeros(layer.layer_id),
                kv_pool.get_hp_key_buffer(layer.layer_id),
                kv_pool.get_hp_value_buffer(layer.layer_id),
                kv_pool.hp_global_offset,
                self.forward_metadata.qo_indptr,
                prefix_indptr,
                prefix_indices,
                causal,
                self.forward_metadata.max_extend_len,
                sm_scale=layer.scaling,
                logit_cap=softcap or 0.0,
                xai_temperature_len=layer.xai_temperature_len,
            )
            result = apply_inverse_v_rotation(result, kv_pool, layer, need_v_inverse)
            o.copy_(result.view_as(o))
            return o

        prefix_k, prefix_v = dequantize_prefix_kv(
            kv_pool,
            layer.layer_id,
            prefix_indices,
            q3.dtype,
        )

        unified_k_parts = []
        unified_v_parts = []
        unified_k_lens = []
        extend_start_loc = forward_batch.extend_start_loc
        for i, extend_len in enumerate(forward_batch.extend_seq_lens_cpu):
            prefix_start = int(prefix_indptr[i].item())
            prefix_end = int(prefix_indptr[i + 1].item())
            extend_start = int(extend_start_loc[i].item())
            extend_end = extend_start + int(extend_len)
            req_k = torch.cat(
                [prefix_k[prefix_start:prefix_end], k3[extend_start:extend_end]], dim=0
            )
            req_v = torch.cat(
                [prefix_v[prefix_start:prefix_end], v3[extend_start:extend_end]], dim=0
            )
            unified_k_parts.append(req_k)
            unified_v_parts.append(req_v)
            unified_k_lens.append(req_k.shape[0])

        unified_k = torch.cat(unified_k_parts, dim=0) if unified_k_parts else k3[:0]
        unified_v = torch.cat(unified_v_parts, dim=0) if unified_v_parts else v3[:0]
        cu_seqlens_q = self.forward_metadata.qo_indptr.to(torch.int32)
        cu_seqlens_k = torch.empty(
            (len(unified_k_lens) + 1,), dtype=torch.int32, device=self.device
        )
        cu_seqlens_k[0] = 0
        cu_seqlens_k[1:] = torch.cumsum(
            torch.tensor(unified_k_lens, dtype=torch.int32, device=self.device), dim=0
        )

        if os.environ.get("SGLANG_MIXED_KV_PREFILL_TORCH") == "1":
            result_parts = []
            left_window, right_window = window_size
            for i, _ in enumerate(forward_batch.extend_seq_lens_cpu):
                q_start = int(cu_seqlens_q[i].item())
                q_end = int(cu_seqlens_q[i + 1].item())
                k_start = int(cu_seqlens_k[i].item())
                k_end = int(cu_seqlens_k[i + 1].item())

                q_i = q3[q_start:q_end]
                k_i = unified_k[k_start:k_end]
                v_i = unified_v[k_start:k_end]
                q_len = q_i.shape[0]
                k_len = k_i.shape[0]

                if q_i.shape[1] != k_i.shape[1]:
                    if q_i.shape[1] % k_i.shape[1] != 0:
                        raise RuntimeError(
                            "GQA torch fallback requires q heads to be a "
                            "multiple of kv heads"
                        )
                    repeat = q_i.shape[1] // k_i.shape[1]
                    k_i = k_i.repeat_interleave(repeat, dim=1)
                    v_i = v_i.repeat_interleave(repeat, dim=1)

                logits = (
                    torch.einsum("qhd,khd->hqk", q_i.float(), k_i.float())
                    * layer.scaling
                )
                if softcap is not None and softcap > 0:
                    logits = softcap * torch.tanh(logits / softcap)

                if causal or left_window >= 0 or right_window >= 0:
                    prefix_len = k_len - q_len
                    q_pos = prefix_len + torch.arange(
                        q_len, device=q_i.device, dtype=torch.int64
                    )
                    k_pos = torch.arange(k_len, device=q_i.device, dtype=torch.int64)
                    mask = torch.zeros(
                        (q_len, k_len), device=q_i.device, dtype=torch.bool
                    )
                    if causal:
                        mask |= k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
                    if left_window >= 0:
                        mask |= k_pos.unsqueeze(0) < (q_pos.unsqueeze(1) - left_window)
                    if right_window >= 0:
                        mask |= k_pos.unsqueeze(0) > (q_pos.unsqueeze(1) + right_window)
                    logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))

                probs = torch.softmax(logits, dim=-1).to(v_i.dtype)
                result_parts.append(
                    torch.einsum("hqk,khd->qhd", probs, v_i).to(q3.dtype)
                )
            result = torch.cat(result_parts, dim=0) if result_parts else q3[:0]
        elif q3.shape[-1] > 256 or torch.cuda.get_device_capability(q3.device)[0] < 9:
            # FA2/FA3 cap head_dim at 256 and the jit flash_attn dense path
            # dispatches to a Hopper (sm90) FA3 kernel that fails on non-Hopper
            # GPUs with "FlashAttention forward only supports head dimension at
            # most 256" or "invalid argument". Route the mixed-KV prefill through
            # per-request SDPA on the already-dequantized dense K/V instead.
            result = self._sdpa_varlen_prefill(
                q3,
                unified_k_parts,
                unified_v_parts,
                cu_seqlens_q,
                forward_batch.extend_seq_lens_cpu,
                unified_k_lens,
                layer.scaling,
                causal,
                window_size[0] if window_size[0] >= 0 else -1,
            )
        else:
            result = flash_attn_varlen_func(
                q=q3,
                k=unified_k,
                v=unified_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max(forward_batch.extend_seq_lens_cpu),
                max_seqlen_k=max(unified_k_lens) if unified_k_lens else 0,
                softmax_scale=layer.scaling,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                ver=(
                    4
                    if os.environ.get("SGLANG_FORCE_FLASH_ATTN_VARLEN_V4") == "1"
                    else 3
                ),
            )
        result = apply_inverse_v_rotation(result, kv_pool, layer, need_v_inverse)
        o.copy_(result.view_as(o))
        return o

    def _sdpa_varlen_prefill(
        self,
        q3: torch.Tensor,  # [total_q, num_q_heads, head_dim]
        k_parts: list,  # per-req [k_len_i, num_kv_heads, head_dim]
        v_parts: list,  # per-req [k_len_i, num_kv_heads, v_head_dim]
        cu_seqlens_q: torch.Tensor,  # int32 [bs+1]
        extend_seq_lens_cpu,  # list[int] per req (query lengths)
        k_lens: list,  # list[int] per req (full kv lengths)
        sm_scale: float,
        causal: bool,
        sliding_window: int,  # >=0 window size (w-1 left); -1 disabled
    ) -> torch.Tensor:
        """Varlen prefill via per-request SDPA, for head_dim > 256 (FA caps at
        256) or non-Hopper GPUs. Operates on the already-dequantized dense K/V.
        Builds an additive mask combining causality + (optional) sliding window.
        MQA/GQA handled by ``enable_gqa=True``. Returns
        ``[total_q, num_q_heads, v_head_dim]``.
        """
        num_q_heads = q3.shape[1]
        v_head_dim = v_parts[0].shape[-1] if v_parts else q3.shape[-1]
        out = q3.new_empty((q3.shape[0], num_q_heads, v_head_dim))
        q_starts = cu_seqlens_q.tolist()
        for i, q_len in enumerate(extend_seq_lens_cpu):
            q_len = int(q_len)
            if q_len == 0:
                continue
            k_len = int(k_lens[i])
            qs = int(q_starts[i])
            # [1, H, q_len, hd] / [1, Hkv, k_len, hd]
            qi = q3[qs : qs + q_len].transpose(0, 1).unsqueeze(0)
            ki = k_parts[i].transpose(0, 1).unsqueeze(0)
            vi = v_parts[i].transpose(0, 1).unsqueeze(0)
            # Query position p (0-based within request) corresponds to absolute
            # key index (k_len - q_len + p), since the last q_len keys are the
            # extend tokens and the leading (k_len - q_len) are the prefix.
            # The initial prefill has no cached prefix, so q_len == k_len. Let
            # SDPA express that standard causal case directly; constructing an
            # explicit [q_len, k_len] mask would consume 16 GiB at 128K even
            # though the backend can apply the same mask implicitly.
            native_causal = causal and sliding_window < 0 and q_len == k_len
            native_noncausal = not causal and sliding_window < 0
            if native_causal or native_noncausal:
                attn_mask = None
            else:
                q_abs = torch.arange(k_len - q_len, k_len, device=q3.device).unsqueeze(
                    1
                )  # [q_len, 1]
                k_abs = torch.arange(k_len, device=q3.device).unsqueeze(0)  # [1, k_len]
                allowed = torch.ones((q_len, k_len), dtype=torch.bool, device=q3.device)
                if causal:
                    allowed &= k_abs <= q_abs
                if sliding_window >= 0:
                    # window covers keys [q_abs - sliding_window, q_abs]
                    allowed &= k_abs >= (q_abs - sliding_window)
                attn_mask = torch.zeros(
                    (q_len, k_len), dtype=qi.dtype, device=q3.device
                )
                attn_mask.masked_fill_(~allowed, float("-inf"))
            oi = torch.nn.functional.scaled_dot_product_attention(
                qi,
                ki,
                vi,
                attn_mask=attn_mask,
                is_causal=native_causal,
                scale=sm_scale,
                enable_gqa=(num_q_heads != ki.shape[1]),
            )
            out[qs : qs + q_len] = oi.squeeze(0).transpose(0, 1)
        return out

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init auxiliary variables for triton attention backend."""

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None
        mixed_hp_kv_indptr = None
        mixed_hp_kv_indices = None
        mixed_quant_kv_indptr = None
        mixed_quant_kv_indices = None
        mixed_attn_logits = None
        mixed_attn_lse = None
        mixed_hp_num_kv_splits = None
        mixed_quant_num_kv_splits = None
        spec_info = forward_batch.spec_info

        if forward_batch.forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = torch.empty(
                    forward_batch.seq_lens_sum, dtype=torch.int64, device=self.device
                )
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                # Sliding window
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indptr, window_kv_indices, window_kv_lens, _ = (
                        update_sliding_window_buffer(
                            self.window_kv_indptr,
                            self.req_to_token,
                            self.sliding_window_size,
                            forward_batch.seq_lens,
                            forward_batch.req_pool_indices,
                            bs,
                            self.device,
                            self.token_to_kv_pool_allocator,
                        )
                    )
                    window_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    self.get_num_kv_splits(window_num_kv_splits, window_kv_lens)
                if self.enable_mixed_kv:
                    mixed_hp_kv_indptr = torch.zeros(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    mixed_quant_kv_indptr = torch.zeros(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    mixed_hp_kv_indices = torch.empty(
                        forward_batch.seq_lens_sum,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    mixed_quant_kv_indices = torch.empty(
                        forward_batch.seq_lens_sum,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    total_splits = self.max_kv_splits + self.max_hp_kv_splits
                    # Single combined stage-1 scratch. LSE is pre-filled with
                    # -inf so the tier-agnostic stage-2 can skip unused splits.
                    mixed_attn_logits = torch.empty(
                        (bs, self.num_head, total_splits, self.v_head_dim),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    mixed_attn_lse = torch.full(
                        (bs, self.num_head, total_splits),
                        float("-inf"),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    mixed_hp_num_kv_splits = torch.full(
                        (bs,),
                        self.max_hp_kv_splits,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    mixed_quant_num_kv_splits = torch.empty(
                        (bs,), dtype=torch.int32, device=self.device
                    )
                    mixed_quant_lens = self._build_mixed_kv_indices(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        mixed_hp_kv_indptr,
                        mixed_hp_kv_indices,
                        mixed_quant_kv_indptr,
                        mixed_quant_kv_indices,
                        bs,
                    )
            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
                bs = kv_indptr.shape[0] - 1

            attn_logits = torch.empty(
                (bs, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if self.swa_v_head_dim is not None:
                swa_attn_logits = torch.empty(
                    (bs, self.num_head, self.max_kv_splits, self.swa_v_head_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                swa_attn_logits = None
            attn_lse = torch.empty(
                (bs, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            num_kv_splits = torch.empty((bs,), dtype=torch.int32, device=self.device)
            if self.enable_mixed_kv:
                # HP uses the fixed cap above; only the quant tier is
                # right-sized, and it uses the full sequence length as a cheap
                # planning proxy instead of per-tier mixed-KV counts.
                self.get_mixed_quant_num_kv_splits(
                    mixed_quant_num_kv_splits, mixed_quant_lens
                )
            else:
                self.get_num_kv_splits(num_kv_splits, forward_batch.seq_lens)

            qo_indptr = None
            custom_mask = None
            mask_indptr = None
            max_extend_len = None
        elif forward_batch.forward_mode.is_target_verify():
            bs = len(forward_batch.req_pool_indices)
            qo_indptr = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            # Different with flashinfer kv_indptr and kv_indices construction
            kv_indptr[1 : bs + 1] = torch.cumsum(forward_batch.seq_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                kv_indptr[-1], dtype=torch.int64, device=self.device
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                # window_kv_offsets is used to calculate the start position in custom mask
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool_allocator,
                )

            custom_mask = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (
                forward_batch.seq_lens + self.num_draft_tokens
            )
            mask_indptr = self.mask_indptr
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len[:bs], dim=0)
            mask_indptr = mask_indptr[: bs + 1]
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None

        elif forward_batch.forward_mode.is_draft_extend():
            kv_indices, kv_indptr, qo_indptr, custom_mask = (
                spec_info.generate_attn_arg_prefill(
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    None,
                    self.req_to_token,
                )
            )
            kv_indices = kv_indices.to(torch.int64)
            mask_indptr = None
            # TODO(FIXME): This will trigger an invalid Eagle tree when using
            # `max(spec_info.accept_length_cpu)`.
            # It might have been forgotten to update somewhere.
            max_extend_len = torch.max(spec_info.accept_length).item()
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            kv_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_prefix_lens, dim=0
            )
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                sum(forward_batch.extend_prefix_lens_cpu),
                dtype=torch.int64,
                device=self.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.extend_prefix_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            # Sliding window
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                (
                    window_kv_indptr,
                    window_kv_indices,
                    window_kv_lens,
                    window_kv_offsets,
                ) = update_sliding_window_buffer(
                    self.window_kv_indptr,
                    self.req_to_token,
                    self.sliding_window_size,
                    forward_batch.extend_prefix_lens,
                    forward_batch.req_pool_indices,
                    bs,
                    self.device,
                    self.token_to_kv_pool_allocator,
                )

            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
            mask_indptr = None
            attn_logits = None
            attn_lse = None
            max_extend_len = max(forward_batch.extend_seq_lens_cpu)
            num_kv_splits = None

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
            mixed_hp_kv_indptr=mixed_hp_kv_indptr,
            mixed_hp_kv_indices=mixed_hp_kv_indices,
            mixed_quant_kv_indptr=mixed_quant_kv_indptr,
            mixed_quant_kv_indices=mixed_quant_kv_indices,
            mixed_attn_logits=mixed_attn_logits,
            mixed_attn_lse=mixed_attn_lse,
            mixed_hp_num_kv_splits=mixed_hp_num_kv_splits,
            mixed_quant_num_kv_splits=mixed_quant_num_kv_splits,
        )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
        cuda_graph_num_kv_splits_buf: Optional[torch.Tensor] = None,
    ):
        self.cuda_graph_attn_logits = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        if self.swa_v_head_dim is not None:
            self.cuda_graph_swa_attn_logits = torch.zeros(
                (
                    max_num_tokens,
                    self.num_head,
                    self.max_kv_splits,
                    self.swa_v_head_dim,
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.cuda_graph_swa_attn_logits = None
        self.cuda_graph_attn_lse = torch.zeros(
            (max_num_tokens, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )

        if cuda_graph_num_kv_splits_buf is None:
            self.cuda_graph_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.cuda_graph_num_kv_splits = cuda_graph_num_kv_splits_buf

        if kv_indices_buf is None:
            self.cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.cuda_graph_kv_indices = kv_indices_buf
        if self.enable_mixed_kv:
            self.cuda_graph_mixed_hp_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.cuda_graph_mixed_quant_kv_indptr = torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=self.device
            )
            self.cuda_graph_mixed_hp_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
            self.cuda_graph_mixed_quant_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.int64,
                device=self.device,
            )
            total_splits = self.max_kv_splits + self.max_hp_kv_splits
            # Single combined stage-1 scratch. LSE pre-filled to -inf so the
            # tier-agnostic stage-2 skips unused splits.
            self.cuda_graph_mixed_attn_logits = torch.zeros(
                (max_num_tokens, self.num_head, total_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.cuda_graph_mixed_attn_lse = torch.full(
                (max_num_tokens, self.num_head, total_splits),
                float("-inf"),
                dtype=torch.float32,
                device=self.device,
            )
            self.cuda_graph_mixed_hp_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_hp_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )
            self.cuda_graph_mixed_quant_num_kv_splits = torch.zeros(
                (max_num_tokens,), dtype=torch.int32, device=self.device
            )

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )

        if self.sliding_window_size is not None and self.sliding_window_size > 0:
            if kv_indices_buf is None:
                self.cuda_graph_window_kv_indices = torch.zeros(
                    (max_num_tokens * self.sliding_window_size),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                self.cuda_graph_window_kv_indices = torch.zeros_like(kv_indices_buf)

            self.cuda_graph_window_num_kv_splits = torch.full(
                (max_num_tokens,),
                self.max_kv_splits,
                dtype=torch.int32,
                device=self.device,
            )

            self.cuda_graph_window_kv_offsets = torch.zeros(
                (max_bs,),
                dtype=torch.int32,
                device=self.device,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        assert encoder_lens is None, "Not supported"
        window_kv_indptr = self.window_kv_indptr
        window_kv_indices = None
        window_num_kv_splits = None
        window_kv_offsets = None
        swa_attn_logits = None
        mixed_hp_kv_indptr = None
        mixed_hp_kv_indices = None
        mixed_quant_kv_indptr = None
        mixed_quant_kv_indices = None
        mixed_attn_logits = None
        mixed_attn_lse = None
        mixed_hp_num_kv_splits = None
        mixed_quant_num_kv_splits = None

        if forward_mode.is_decode_or_idle():
            if spec_info is None:
                kv_indptr = self.kv_indptr
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                kv_indices = self.cuda_graph_kv_indices
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices,
                    seq_lens,
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indptr, window_kv_indices, _, _ = (
                        update_sliding_window_buffer_cuda_graph(
                            self.window_kv_indptr,
                            window_kv_indices,
                            self.req_to_token,
                            self.sliding_window_size,
                            seq_lens[:bs],
                            req_pool_indices,
                            bs,
                            self.token_to_kv_pool_allocator,
                        )
                    )
                if self.enable_mixed_kv:
                    mixed_hp_kv_indptr = self.cuda_graph_mixed_hp_kv_indptr
                    mixed_hp_kv_indices = self.cuda_graph_mixed_hp_kv_indices
                    mixed_quant_kv_indptr = self.cuda_graph_mixed_quant_kv_indptr
                    mixed_quant_kv_indices = self.cuda_graph_mixed_quant_kv_indices
                    mixed_attn_logits = self.cuda_graph_mixed_attn_logits
                    mixed_attn_lse = self.cuda_graph_mixed_attn_lse
                    mixed_hp_num_kv_splits = self.cuda_graph_mixed_hp_num_kv_splits
                    mixed_quant_num_kv_splits = (
                        self.cuda_graph_mixed_quant_num_kv_splits
                    )
                    mixed_quant_lens = self._build_mixed_kv_indices(
                        req_pool_indices,
                        seq_lens,
                        mixed_hp_kv_indptr,
                        mixed_hp_kv_indices,
                        mixed_quant_kv_indptr,
                        mixed_quant_kv_indices,
                        bs,
                    )
                    mixed_hp_num_kv_splits[:bs] = self.max_hp_kv_splits
                    self.get_mixed_quant_num_kv_splits(
                        mixed_quant_num_kv_splits[:bs], mixed_quant_lens[:bs]
                    )

            else:
                kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices

            attn_logits = self.cuda_graph_attn_logits
            swa_attn_logits = self.cuda_graph_swa_attn_logits
            attn_lse = self.cuda_graph_attn_lse
            max_extend_len = None
            num_kv_splits = self.cuda_graph_num_kv_splits
            qo_indptr = None
            custom_mask = None
            mask_indptr = None
        elif forward_mode.is_target_verify():
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )

            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                window_kv_indptr, window_kv_indices, _, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )

            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
            max_extend_len = self.num_draft_tokens
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        elif forward_mode.is_draft_extend(include_v2=True):
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            custom_mask = None
            mask_indptr = None
            max_extend_len = num_tokens_per_bs
            num_kv_splits = None
            attn_logits = None
            attn_lse = None
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph capture."
            )

        self.forward_metadata = ForwardMetadata(
            attn_logits,
            attn_lse,
            max_extend_len,
            num_kv_splits,
            kv_indptr,
            kv_indices,
            qo_indptr,
            custom_mask,
            mask_indptr,
            window_kv_indptr,
            window_kv_indices,
            window_num_kv_splits,
            window_kv_offsets,
            swa_attn_logits=swa_attn_logits,
            mixed_hp_kv_indptr=mixed_hp_kv_indptr,
            mixed_hp_kv_indices=mixed_hp_kv_indices,
            mixed_quant_kv_indptr=mixed_quant_kv_indptr,
            mixed_quant_kv_indices=mixed_quant_kv_indices,
            mixed_attn_logits=mixed_attn_logits,
            mixed_attn_lse=mixed_attn_lse,
            mixed_hp_num_kv_splits=mixed_hp_num_kv_splits,
            mixed_quant_num_kv_splits=mixed_quant_num_kv_splits,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        # NOTE: encoder_lens expected to be zeros or None
        if forward_mode.is_decode_or_idle():
            # Update kv_indptr, kv_indices
            kv_indptr = self.kv_indptr
            kv_indices = self.cuda_graph_kv_indices
            num_kv_splits = self.cuda_graph_num_kv_splits
            mixed_hp_num_kv_splits = None
            mixed_quant_num_kv_splits = None
            if self.enable_mixed_kv:
                mixed_hp_num_kv_splits = self.cuda_graph_mixed_hp_num_kv_splits
                mixed_quant_num_kv_splits = self.cuda_graph_mixed_quant_num_kv_splits
            if spec_info is None:
                kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
                kv_indptr = kv_indptr[: bs + 1]
                create_flashinfer_kv_indices_triton[(bs,)](
                    self.req_to_token,
                    req_pool_indices[:bs],
                    seq_lens[:bs],
                    kv_indptr,
                    None,
                    kv_indices,
                    self.req_to_token.stride(0),
                )
                num_token = bs
                if (
                    self.sliding_window_size is not None
                    and self.sliding_window_size > 0
                ):
                    window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                    window_kv_indices = self.cuda_graph_window_kv_indices
                    _, _, window_kv_lens, _ = update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices[:bs],
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                    self.get_num_kv_splits(
                        window_num_kv_splits[:num_token], window_kv_lens[:bs]
                    )
                if self.enable_mixed_kv:
                    mixed_quant_lens = self._build_mixed_kv_indices(
                        req_pool_indices,
                        seq_lens,
                        self.cuda_graph_mixed_hp_kv_indptr,
                        self.cuda_graph_mixed_hp_kv_indices,
                        self.cuda_graph_mixed_quant_kv_indptr,
                        self.cuda_graph_mixed_quant_kv_indices,
                        bs,
                    )
                    mixed_hp_num_kv_splits[:bs] = self.max_hp_kv_splits
                    # The unified attention wrapper fills LSE with -inf every
                    # call, so the shared scratch is always in a known state
                    # entering stage-2. No extra reset needed here.

            else:
                assert False, "Multi-step cuda graph init is not done here."
            if self.enable_mixed_kv:
                self.get_mixed_quant_num_kv_splits(
                    mixed_quant_num_kv_splits[:num_token], mixed_quant_lens[:bs]
                )
            else:
                self.get_num_kv_splits(num_kv_splits[:num_token], seq_lens[:bs])

        elif forward_mode.is_target_verify():
            # Update qo_indptr, kv_indptr, kv_indices, custom_mask, mask_indptr
            bs = len(req_pool_indices)
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                (1 + bs) * self.num_draft_tokens,
                step=self.num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
            if self.sliding_window_size is not None and self.sliding_window_size > 0:
                window_num_kv_splits = self.cuda_graph_window_num_kv_splits
                window_kv_indices = self.cuda_graph_window_kv_indices
                window_kv_offsets = self.cuda_graph_window_kv_offsets
                _, _, window_kv_lens, window_kv_offsets[:bs] = (
                    update_sliding_window_buffer_cuda_graph(
                        self.window_kv_indptr,
                        window_kv_indices,
                        self.req_to_token,
                        self.sliding_window_size,
                        seq_lens[:bs],
                        req_pool_indices,
                        bs,
                        self.token_to_kv_pool_allocator,
                    )
                )
            custom_mask = self.cuda_graph_custom_mask
            custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask
            seq_mask_len = self.num_draft_tokens * (seq_lens + self.num_draft_tokens)
            mask_indptr = self.mask_indptr[: bs + 1]
            mask_indptr[1 : bs + 1] = torch.cumsum(seq_mask_len, dim=0)
        elif forward_mode.is_draft_extend(include_v2=True):
            seq_lens = seq_lens[:bs]
            num_tokens_per_bs = self.speculative_num_steps + 1
            qo_indptr = self.qo_indptr[: bs + 1]
            qo_indptr[: bs + 1] = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                step=num_tokens_per_bs,
                dtype=torch.int32,
                device=self.device,
            )
            kv_indptr = self.kv_indptr[: bs + 1]
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indices = self.cuda_graph_kv_indices
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                seq_lens,
                kv_indptr,
                None,
                kv_indices,
                self.req_to_token.stride(0),
            )
        else:
            raise ValueError(
                f"Invalid forward mode: {forward_mode=} for CUDA Graph replay."
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Return buffers for verify attention kernels that needs to be filled after draft.

        Typically, these are tree mask and position buffers.
        """
        return [self.cuda_graph_custom_mask, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        pass

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if k is None and v is None:
            pool = forward_batch.token_to_kv_pool
            cache_loc = forward_batch.out_cache_loc
            if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
                cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)
            k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
            k = k_buffer[cache_loc]
            v = v_buffer[cache_loc]
        elif k is None or v is None:
            raise ValueError("Both k and v should be None or not None")

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or (
                layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
                and self.allow_bidirectional_attention_in_extend
            )
        ):
            causal = False

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = (
                layer.sliding_window_size
            )  # Needed for sliding window mask
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            window_kv_offsets = self.forward_metadata.window_kv_offsets
        else:
            sliding_window_size = -1
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            window_kv_offsets = None

        kv_pool = forward_batch.token_to_kv_pool
        use_quantized_dense_prefill = (
            hasattr(kv_pool, "dtype")
            and kv_pool.dtype == "int2"
            and self.forward_metadata.custom_mask is None
        )
        pre_rotated_q = None
        pre_rotated_k = None
        pre_rotated_v = None
        need_v_inverse = None
        if (
            not self.enable_deterministic
            and use_quantized_dense_prefill
            and getattr(kv_pool, "dtype", None) == "int2"
            and k is not None
            and v is not None
        ):
            # Int2 prefill used to rotate K/V once for attention and again when
            # writing the KV cache. Pre-rotate them here so both consumers can
            # share the same tensors.
            pre_rotated_q, pre_rotated_k, pre_rotated_v, need_v_inverse = (
                prepare_quantized_extend_qkv(
                    kv_pool,
                    layer,
                    q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                    k.contiguous(),
                    v.contiguous(),
                )
            )

        # Save KV cache first (must do this before unified kernel)
        if save_kv_cache and k is not None and v is not None:
            if (
                pre_rotated_k is not None
                and pre_rotated_v is not None
                and getattr(kv_pool, "dtype", None) == "int2"
            ):
                kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    pre_rotated_k,
                    pre_rotated_v,
                    layer.k_scale,
                    layer.v_scale,
                    already_hadamard_transformed=True,
                    is_decode=False,
                )
            elif (
                self.use_mla or layer.k_scale is None
            ):  # Triton MLA currently doesn't support quantized kv cache
                kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                )
            else:
                kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k.clone(),  # cloned to protect k,v from in-place mutation in set_kv_buffer
                    v.clone(),
                    layer.k_scale,
                    layer.v_scale,
                )

        layer_id = layer.layer_id
        if (
            k is not None
            and v is not None
            and layer_id not in self._dump_kv_done_layers
            and get_bool_env_var("DUMP_KVCACHE", "false")
            and (
                not os.environ.get("DUMP_KVCACHE_TRIGGER_FILE")
                or os.path.exists(os.environ["DUMP_KVCACHE_TRIGGER_FILE"])
            )
        ):
            dump_tokens = get_int_env_var("DUMP_KVCACHE_TOKENS", 100)
            saved_so_far = self._dump_saved_tokens.get(layer_id, 0)
            chunk_idx = self._dump_chunk_idx.get(layer_id, 0)
            remaining = dump_tokens - saved_so_far

            if remaining > 0:
                num_tokens = q.shape[0]
                tokens_to_save = min(num_tokens, remaining)

                if str(
                    getattr(forward_batch.token_to_kv_pool, "device", "cuda")
                ).startswith("cuda"):
                    torch.cuda.synchronize()

                q_dump = (
                    q[:tokens_to_save]
                    .view(-1, layer.tp_q_head_num, layer.qk_head_dim)
                    .contiguous()
                    .detach()
                )
                k_dump = k[:tokens_to_save].contiguous().detach()
                v_dump = v[:tokens_to_save].contiguous().detach()

                chunk_seq_lens = []
                if forward_batch.extend_seq_lens is not None:
                    remain = tokens_to_save
                    for slen in forward_batch.extend_seq_lens.tolist():
                        if remain <= 0:
                            break
                        take = min(slen, remain)
                        chunk_seq_lens.append(take)
                        remain -= take
                else:
                    chunk_seq_lens = [tokens_to_save]
                chunk_seq_lens_t = torch.tensor(chunk_seq_lens, dtype=torch.int32)

                tp_size = get_attention_tp_size()
                tp_rank = get_attention_tp_rank()
                if tp_size > 1:
                    attn_tp_group = get_attention_tp_group()
                    q_dump = attn_tp_group.all_gather(q_dump, dim=1)
                    k_dump = attn_tp_group.all_gather(k_dump, dim=1)
                    v_dump = attn_tp_group.all_gather(v_dump, dim=1)

                if tp_rank == 0:
                    save_dir = os.environ.get("DUMP_KVCACHE_DIR", ".")
                    for name, tensor in (
                        ("q", q_dump),
                        ("k", k_dump),
                        ("v", v_dump),
                    ):
                        chunk_dir = os.path.join(save_dir, f"layer_{layer_id}", name)
                        os.makedirs(chunk_dir, exist_ok=True)
                        torch.save(
                            tensor.cpu(), os.path.join(chunk_dir, f"{chunk_idx}.pt")
                        )
                    seq_dir = os.path.join(save_dir, f"layer_{layer_id}", "seq_lens")
                    os.makedirs(seq_dir, exist_ok=True)
                    torch.save(
                        chunk_seq_lens_t, os.path.join(seq_dir, f"{chunk_idx}.pt")
                    )
                    print(
                        f"Dumped QKV chunk {chunk_idx} for layer {layer_id} "
                        f"({tokens_to_save} tokens, {len(chunk_seq_lens)} seqs, "
                        f"total {saved_so_far + tokens_to_save}/{dump_tokens})"
                    )

                self._dump_saved_tokens[layer_id] = saved_so_far + tokens_to_save
                self._dump_chunk_idx[layer_id] = chunk_idx + 1

                if saved_so_far + tokens_to_save >= dump_tokens:
                    self._dump_kv_done_layers.add(layer_id)

        # Deterministic mode: use unified 1-stage kernel
        if self.enable_deterministic:
            return self._forward_extend_unified(
                q, o, layer, forward_batch, causal, logits_soft_cap, sinks
            )

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        if use_quantized_dense_prefill:
            prefix_indptr = self.forward_metadata.kv_indptr
            prefix_indices = self.forward_metadata.kv_indices
            flash_window_size = (-1, -1)
            if sliding_window_size > -1:
                prefix_indptr = self.forward_metadata.window_kv_indptr
                prefix_indices = self.forward_metadata.window_kv_indices
                flash_window_size = (sliding_window_size, 0)
                if prefix_indptr is None or prefix_indices is None:
                    raise RuntimeError(
                        "INT2 sliding-window prefill requires window KV metadata"
                    )
            return self._forward_extend_quantized_dense(
                q,
                k,
                v,
                o,
                layer,
                forward_batch,
                causal,
                pre_rotated_q=pre_rotated_q,
                pre_rotated_k=pre_rotated_k,
                pre_rotated_v=pre_rotated_v,
                need_v_inverse_override=need_v_inverse,
                prefix_indptr=prefix_indptr,
                prefix_indices=prefix_indices,
                window_size=flash_window_size,
            )

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            k_descale,
            v_descale,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
        )
        return o

    def _forward_extend_unified(
        self,
        q: torch.Tensor,
        o: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        causal: bool,
        logits_soft_cap: float,
        sinks: Optional[torch.Tensor],
    ):
        """
        Unified 1-stage extend attention for deterministic inference.
        Both prefix and extend KV are accessed through unified kv_indices.
        """
        bs = forward_batch.batch_size

        # Determine sliding window settings
        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            sliding_window_size = layer.sliding_window_size
            # Note: for unified kernel, we use full kv_indptr (not window)
            prefix_kv_indptr = self.forward_metadata.window_kv_indptr
            prefix_kv_indices = self.forward_metadata.window_kv_indices
            # Compute window start positions (absolute position of first key in window)
            # window_start_pos = seq_len - window_len
            window_kv_lens = prefix_kv_indptr[1 : bs + 1] - prefix_kv_indptr[:bs]
            # Handle TARGET_VERIFY mode where extend_prefix_lens might not be set
            if forward_batch.extend_prefix_lens is not None:
                window_start_pos = (
                    forward_batch.extend_prefix_lens[:bs] - window_kv_lens
                )
            else:
                # Infer from spec_info: prefix_len = seq_len - draft_token_num
                if forward_batch.spec_info is not None and hasattr(
                    forward_batch.spec_info, "draft_token_num"
                ):
                    extend_prefix_lens = (
                        forward_batch.seq_lens[:bs]
                        - forward_batch.spec_info.draft_token_num
                    )
                    window_start_pos = extend_prefix_lens - window_kv_lens
                else:
                    window_start_pos = None
        else:
            sliding_window_size = -1
            prefix_kv_indptr = self.forward_metadata.kv_indptr
            prefix_kv_indices = self.forward_metadata.kv_indices
            window_start_pos = None

        # Build unified kv_indices using fused Triton kernel
        extend_kv_indices = forward_batch.out_cache_loc

        # Handle cases where extend_seq_lens or extend_start_loc might not be set
        # In speculative decoding, we can infer these from spec_info or compute them
        if forward_batch.extend_seq_lens is None:
            # TARGET_VERIFY mode: infer extend_seq_lens from spec_info
            if forward_batch.spec_info is not None and hasattr(
                forward_batch.spec_info, "draft_token_num"
            ):
                draft_token_num = forward_batch.spec_info.draft_token_num
                extend_seq_lens = torch.full(
                    (bs,), draft_token_num, dtype=torch.int32, device=self.device
                )
            else:
                raise RuntimeError(
                    "extend_seq_lens is None but cannot infer from spec_info. "
                    "This should not happen in TARGET_VERIFY mode."
                )
        else:
            extend_seq_lens = forward_batch.extend_seq_lens

        # Check extend_start_loc separately - it might be None even when extend_seq_lens is set
        if forward_batch.extend_start_loc is None:
            # Compute extend_start_loc from extend_seq_lens
            # extend_start_loc[i] = sum(extend_seq_lens[0:i])
            extend_start_loc = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=self.device),
                    torch.cumsum(extend_seq_lens[:-1], dim=0),
                ]
            )
        else:
            extend_start_loc = forward_batch.extend_start_loc

        unified_kv_indptr, unified_kv_indices, prefix_lens = (
            self.build_unified_kv_indices(
                prefix_kv_indptr,
                prefix_kv_indices,
                extend_start_loc,
                extend_seq_lens,
                extend_kv_indices,
                bs,
            )
        )

        # Convert prefix_lens to int32 for the kernel
        prefix_lens = prefix_lens.to(torch.int32)

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Call unified kernel
        self.extend_attention_fwd_unified(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            k_descale,
            v_descale,
            self.forward_metadata.qo_indptr,
            unified_kv_indptr,
            unified_kv_indices,
            prefix_lens,
            self.forward_metadata.max_extend_len,
            custom_mask=self.forward_metadata.custom_mask,
            mask_indptr=self.forward_metadata.mask_indptr,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            is_causal=causal,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_start_pos=window_start_pos,
            xai_temperature_len=layer.xai_temperature_len,
        )

        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks=None,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        # TODO: reuse the buffer across layers
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

        if save_kv_cache:
            if self.use_mla:  # Triton MLA currently doesn't support quantized kv cache
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                )
            else:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer,
                    forward_batch.out_cache_loc,
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                    is_decode=True,
                )

        if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            kv_indptr = self.forward_metadata.window_kv_indptr
            kv_indices = self.forward_metadata.window_kv_indices
            num_kv_splits = self.forward_metadata.window_num_kv_splits
        else:
            kv_indptr = self.forward_metadata.kv_indptr
            kv_indices = self.forward_metadata.kv_indices
            num_kv_splits = self.forward_metadata.num_kv_splits

        if layer.k_scale is not None and layer.v_scale is not None:
            k_descale = layer.k_scale_float
            v_descale = layer.v_scale_float
        else:
            k_descale = 1.0
            v_descale = 1.0

        # Select the correctly-sized attn_logits buffer for this layer.
        # The triton kernel's // Lv stride trick requires attn_logits.shape[-1]
        # to exactly match the layer's v_head_dim.
        attn_logits = self.forward_metadata.attn_logits
        if (
            self.forward_metadata.swa_attn_logits is not None
            and layer.v_head_dim == self.swa_v_head_dim
        ):
            attn_logits = self.forward_metadata.swa_attn_logits

        # Int2 quantized KV cache path (the only supported quant tier).
        kv_pool = forward_batch.token_to_kv_pool
        if hasattr(kv_pool, "dtype") and kv_pool.dtype == "int2":
            uses_oscar = _pool_uses_oscar_rotation(kv_pool)

            q_for_decode = q.contiguous().view(
                -1, layer.tp_q_head_num, layer.qk_head_dim
            )
            mixed_hp_kv_indptr = self.forward_metadata.mixed_hp_kv_indptr
            mixed_hp_kv_indices = self.forward_metadata.mixed_hp_kv_indices
            mixed_quant_kv_indptr = self.forward_metadata.mixed_quant_kv_indptr
            mixed_quant_kv_indices = self.forward_metadata.mixed_quant_kv_indices
            mixed_hp_num_kv_splits = self.forward_metadata.mixed_hp_num_kv_splits
            mixed_quant_num_kv_splits = self.forward_metadata.mixed_quant_num_kv_splits
            mixed_decode_metadata_available = (
                mixed_hp_kv_indptr is not None
                and mixed_quant_kv_indptr is not None
                and mixed_hp_num_kv_splits is not None
                and mixed_quant_num_kv_splits is not None
            )
            mixed_decode_enabled = (
                self.enable_mixed_kv
                and kv_pool.dtype == "int2"
                and sinks is None
                and mixed_decode_metadata_available
            )
            if (
                self.enable_mixed_kv
                and kv_pool.dtype == "int2"
                and mixed_decode_metadata_available
                and sinks is not None
            ):
                raise NotImplementedError(
                    "Mixed KV windows do not support sink tokens in Triton decode."
                )

            # Hard guarantee that the upstream gating actually held: if mixed
            # KV is enabled with an int2 pool, ``init_forward_metadata`` must
            # have built the per-tier indices. Falling through to the
            # non-mixed ``decode_attention_fwd_quantized`` path would treat
            # HP slot ids (>= HP_OFFSET) as quant slot ids and read OOB
            # garbage from the quant buffer. The known offenders are the
            # ``spec_info != None`` decode-or-idle paths (currently gated out
            # at server-args / model-runner level); this assertion makes the
            # gating load-bearing at the kernel boundary so any future
            # widening of those upstream gates surfaces here loudly instead
            # of silently corrupting attention output.
            if self.enable_mixed_kv and kv_pool.dtype == "int2":
                assert mixed_decode_metadata_available, (
                    "Mixed-KV pool active but mixed decode metadata not built. "
                    "spec_info / non-decode-or-idle paths must not reach the "
                    "mixed-KV decode dispatch -- check upstream gating in "
                    "ServerArgs._unified_mixed_kv_active and "
                    "model_runner_kv_cache_mixin._init_pools."
                )

            oscar_layer_idx = layer.layer_id - kv_pool.start_layer

            if uses_oscar:
                q_for_decode = _apply_pool_oscar_k_rotation_to_q(
                    q_for_decode,
                    kv_pool,
                    oscar_layer_idx,
                    inplace=True,
                )
            else:
                q_for_decode = apply_segmented_hadamard_transform(q_for_decode)
            if mixed_decode_enabled:
                bs = q_for_decode.shape[0]
                self.decode_attention_fwd_int2_unified(
                    q_for_decode,
                    kv_pool.get_hp_key_buffer(layer.layer_id),
                    kv_pool.get_hp_value_buffer(layer.layer_id),
                    kv_pool.get_raw_key_buffer(layer.layer_id),
                    kv_pool.get_raw_value_buffer(layer.layer_id),
                    kv_pool.get_key_scales_zeros(layer.layer_id),
                    kv_pool.get_value_scales_zeros(layer.layer_id),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    mixed_hp_kv_indptr,
                    mixed_hp_kv_indices,
                    mixed_quant_kv_indptr,
                    mixed_quant_kv_indices,
                    self.forward_metadata.mixed_attn_logits[:bs],
                    self.forward_metadata.mixed_attn_lse[:bs],
                    mixed_hp_num_kv_splits[:bs],
                    mixed_quant_num_kv_splits[:bs],
                    self.max_hp_kv_splits,
                    self.max_kv_splits,
                    layer.scaling,
                    logit_cap=logits_soft_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
            else:
                # Use optimized quantized attention kernel
                self.decode_attention_fwd_quantized(
                    q_for_decode,
                    kv_pool.get_raw_key_buffer(layer.layer_id),
                    kv_pool.get_raw_value_buffer(layer.layer_id),
                    kv_pool.get_key_scales_zeros(layer.layer_id),
                    kv_pool.get_value_scales_zeros(layer.layer_id),
                    o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                    kv_indptr,
                    kv_indices,
                    attn_logits,
                    self.forward_metadata.attn_lse,
                    num_kv_splits,
                    self.max_kv_splits,
                    layer.scaling,
                    kv_pool.dtype,
                    logit_cap=logits_soft_cap,
                    sinks=sinks,
                    xai_temperature_len=layer.xai_temperature_len,
                )
            # int2: V is always rotated, so apply the inverse rotation to the
            # output. Oscar mode uses ``o @ R_v.T``; Hadamard mode re-applies
            # the segmented FWHT (self-inverse with 1/sqrt(N)).
            if uses_oscar:
                o3 = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                R_v = kv_pool._R_v[oscar_layer_idx]
                o3.copy_(_apply_oscar_v_inverse_to_q(o3, R_v).to(o3.dtype))
            else:
                o = apply_segmented_hadamard_transform(o)
        else:
            # Standard attention with dequantized or non-quantized KV cache
            self.decode_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                kv_indptr,
                kv_indices,
                attn_logits,
                self.forward_metadata.attn_lse,
                num_kv_splits,
                self.max_kv_splits,
                layer.scaling,
                k_descale,
                v_descale,
                logit_cap=logits_soft_cap,
                sinks=sinks,
                xai_temperature_len=layer.xai_temperature_len,
            )
        return o


class TritonMultiStepDraftBackend:
    """
    Wrap multiple triton attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        max_bs = model_runner.req_to_token_pool.size * self.topk
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.attn_backends: List[TritonAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                TritonAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                )
            )
        self.max_context_len = self.attn_backends[0].max_context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.device = model_runner.device
        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.page_size = model_runner.server_args.page_size

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: Optional[torch.Tensor],
        call_fn: int,
    ):
        if kv_indices_buffer is None:
            kv_indices_buffer = self.cuda_graph_kv_indices

        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        if call_fn is None:
            return

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : seq_lens_sum * self.topk + bs * (i + 1)
            ]
            call_fn(i, forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices = torch.empty(
            (
                self.speculative_num_steps,
                forward_batch.batch_size * self.topk * self.max_context_len,
            ),
            dtype=torch.int64,
            device=self.device,
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, max_num_tokens * self.max_context_len),
            dtype=torch.int64,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (max_num_tokens,),
            self.attn_backends[0].max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs,
                max_num_tokens,
                kv_indices_buf=self.cuda_graph_kv_indices[i],
                cuda_graph_num_kv_splits_buf=self.cuda_graph_num_kv_splits,
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, None, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        self.common_template(forward_batch, None, None)

        # NOTE: Multi-step's attention backends use the slice of
        # - kv_indptr buffer (cuda graph and non-cuda graph)
        # - kv_indices buffer (cuda graph only)
        # So we don't need to assign the KV indices inside the attention backend.

        # Compute num_kv_splits only once
        num_token = forward_batch.batch_size * self.topk
        self.attn_backends[-1].get_num_kv_splits(
            self.attn_backends[-1].cuda_graph_num_kv_splits[:num_token],
            forward_batch.seq_lens[:bs],
        )


@triton.jit
def get_num_kv_splits_triton(
    num_kv_splits_ptr,
    seq_lens_ptr,
    num_seq,
    num_group,
    num_head,
    num_kv_head,
    max_kv_splits,
    device_core_count,
    MAX_NUM_SEQ: tl.constexpr,
):
    # TODO: this method is tunable, we need more online serving data to tune it
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq

    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=0)
    max_seq_len = tl.max(seq_lens)
    seq_lens = tl.load(seq_lens_ptr + offs_seq, mask=mask_seq, other=max_seq_len)
    min_seq_len = tl.min(seq_lens)
    if max_seq_len * 8 < min_seq_len * 10:
        min_seq_len = max_seq_len
    max_kv_splits_1 = tl.minimum(tl.cdiv(max_seq_len, min_seq_len), max_kv_splits)
    kv_chunk_size_1 = tl.cdiv(max_seq_len, max_kv_splits_1)

    # NOTE: this is a hack to let num_kv_split grows up with seqlen gradually
    ext_seq_len = tl.cast(max_seq_len, tl.float32) / 64.0
    ext_device_core_count = tl.cast(
        device_core_count * tl.maximum(tl.log2(ext_seq_len), 1.0), tl.int32
    )
    block_h, num_kv_group = 16, num_head // num_kv_head
    if num_kv_group == 1:
        token_grid = num_seq * num_group * num_head
    else:
        # from triton_ops/decode_attention.py:_decode_grouped_att_m_fwd
        block_h = tl.minimum(block_h, num_kv_group)
        token_grid = num_seq * num_group * tl.cdiv(num_head, block_h)
    max_kv_splits_2 = tl.minimum(
        tl.cdiv(ext_device_core_count, token_grid), max_kv_splits
    )
    kv_chunk_size_2 = tl.cdiv(max_seq_len, max_kv_splits_2)

    num_kv_splits = tl.maximum(
        tl.cdiv(seq_lens, kv_chunk_size_1), tl.cdiv(seq_lens, kv_chunk_size_2)
    )

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


@triton.jit
def get_a100_mixed_quant_kv_splits_triton(
    num_kv_splits_ptr,
    quant_lens_ptr,
    num_seq,
    num_group,
    max_kv_splits,
    MAX_NUM_SEQ: tl.constexpr,
):
    """A100 batch-1 schedule measured for the mixed INT2-history kernel.

    Thresholds use quantized-history length, excluding the 64-token prefix and
    256-token recent BF16 tiers. The cap keeps the helper safe for explicitly
    smaller scratch allocations, although production enables it with cap 24.
    """
    offs_seq = tl.arange(0, MAX_NUM_SEQ)
    mask_seq = offs_seq < num_seq
    quant_lens = tl.load(quant_lens_ptr + offs_seq, mask=mask_seq, other=0)
    num_kv_splits = tl.where(
        quant_lens <= 0,
        1,
        tl.where(quant_lens <= 1024, 8, tl.where(quant_lens <= 4096, 16, 24)),
    )
    num_kv_splits = tl.minimum(num_kv_splits, max_kv_splits)

    offs_token = offs_seq * num_group
    mask_token = offs_token < num_seq * num_group
    for i in range(0, num_group):
        tl.store(num_kv_splits_ptr + i + offs_token, num_kv_splits, mask=mask_token)


def update_sliding_window_buffer(
    window_kv_indptr,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    device,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_indices = torch.empty(
        window_kv_indptr[-1], dtype=torch.int64, device=device
    )
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx


def update_sliding_window_buffer_cuda_graph(
    window_kv_indptr,
    window_kv_indices,
    req_to_token,
    sliding_window_size,
    seq_lens,
    req_pool_indices,
    bs,
    token_to_kv_pool_allocator=None,
):
    window_kv_lens = torch.minimum(
        seq_lens,
        torch.tensor(sliding_window_size),
    )
    window_kv_indptr[1 : bs + 1] = torch.cumsum(window_kv_lens, dim=0)
    window_kv_indptr = window_kv_indptr[: bs + 1]
    window_kv_start_idx = seq_lens - window_kv_lens
    create_flashinfer_kv_indices_triton[(bs,)](
        req_to_token,
        req_pool_indices,
        window_kv_lens,
        window_kv_indptr,
        window_kv_start_idx,
        window_kv_indices,
        req_to_token.stride(0),
    )
    # full to swa index mapping
    if hasattr(token_to_kv_pool_allocator, "translate_loc_from_full_to_swa"):
        kv_last_index = window_kv_indptr[-1]
        window_kv_indices[:kv_last_index] = (
            token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
                window_kv_indices[:kv_last_index]
            )
        )
    return window_kv_indptr, window_kv_indices, window_kv_lens, window_kv_start_idx
