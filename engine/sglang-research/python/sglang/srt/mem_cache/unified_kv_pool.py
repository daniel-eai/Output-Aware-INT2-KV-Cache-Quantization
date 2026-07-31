"""
Unified HP + int2 KV cache pool.

Quant arena: paged with ``N_Q`` slots per page. HP arena: shared HP-prefix
pool (paged) followed by per-request HP-recent ring slabs. Slot id namespace
is flat (``[0, num_quant_pages*N_Q)`` quant, ``[HP_OFFSET, ...)`` HP), and
kernels dispatch by ``slot >= HP_OFFSET``.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.QuantKernel.fused_hadamard_int2_kv import (
    quantized_set_kv_int2_pretransformed_triton,
)
from sglang.QuantKernel.oscar_rotation_clip_int2_kv import (
    quantized_set_kv_int2_pretransformed_clip_triton,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.attention.quantized_kv_prefill import _apply_oscar_rotation
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    OscarRotationConfig,
    _set_kv_buffer_impl,
    get_tensor_size_bytes,
    load_oscar_rotation_config,
    load_oscar_rotations,
)

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


@triton.jit
def _set_mixed_hp_buffer_kernel(
    src_ptr,
    dst_ptr,
    loc_ptr,
    num_tokens,
    row_dim: tl.constexpr,
    src_stride_token: tl.constexpr,
    src_stride_dim: tl.constexpr,
    dst_stride_loc: tl.constexpr,
    dst_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offs = block_idx * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    loc = tl.load(loc_ptr + token_idx)
    is_hp = loc >= HP_OFFSET
    hp_loc = loc - HP_OFFSET
    mask = is_hp & (token_idx < num_tokens) & (offs < row_dim)
    vals = tl.load(
        src_ptr + token_idx * src_stride_token + offs * src_stride_dim,
        mask=mask,
        other=0.0,
    )
    tl.store(
        dst_ptr + hp_loc * dst_stride_loc + offs * dst_stride_dim,
        vals,
        mask=mask,
    )


@triton.jit
def _set_hp_kv_oscar_k_write_kernel(
    k_input_ptr,
    v_input_ptr,
    R_ptr,
    mu_rot_ptr,
    hp_k_ptr,
    hp_v_ptr,
    loc_ptr,
    num_tokens,
    num_heads,
    k_input_stride_token: tl.constexpr,
    k_input_stride_head: tl.constexpr,
    k_input_stride_dim: tl.constexpr,
    v_input_stride_token: tl.constexpr,
    v_input_stride_head: tl.constexpr,
    v_input_stride_dim: tl.constexpr,
    R_stride_head: tl.constexpr,
    R_stride_in: tl.constexpr,
    R_stride_out: tl.constexpr,
    mu_stride_head: tl.constexpr,
    mu_stride_dim: tl.constexpr,
    hp_k_stride_loc: tl.constexpr,
    hp_k_stride_head: tl.constexpr,
    hp_k_stride_dim: tl.constexpr,
    hp_v_stride_loc: tl.constexpr,
    hp_v_stride_head: tl.constexpr,
    hp_v_stride_dim: tl.constexpr,
    HP_OFFSET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    HEADWISE_R: tl.constexpr,
):
    """Fused HP-tier write for OSCAR/OptR K centering.

    K is transformed as ``K @ R_k - mu_rot`` and written directly to the HP
    cache. V is copied as-is; this kernel is only used when V rotation has
    already been absorbed into the model projection weights.
    """
    pid_tok = tl.program_id(0)
    head_idx = tl.program_id(1)
    if head_idx >= num_heads:
        return

    tok_offs = pid_tok * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    tok_mask = tok_offs < num_tokens
    loc = tl.load(loc_ptr + tok_offs, mask=tok_mask, other=0).to(tl.int64)
    active = tok_mask & (loc >= HP_OFFSET)
    hp_loc = tl.where(active, loc - HP_OFFSET, 0)

    dim_offs = tl.arange(0, HEAD_DIM)
    k_base = (
        tok_offs[:, None] * k_input_stride_token
        + head_idx * k_input_stride_head
        + dim_offs[None, :] * k_input_stride_dim
    )
    k_tile = tl.load(
        k_input_ptr + k_base,
        mask=tok_mask[:, None],
        other=0.0,
    )

    r_in = tl.arange(0, HEAD_DIM)
    r_out = tl.arange(0, HEAD_DIM)
    if HEADWISE_R:
        R_offs = (
            head_idx * R_stride_head
            + r_in[:, None] * R_stride_in
            + r_out[None, :] * R_stride_out
        )
    else:
        R_offs = r_in[:, None] * R_stride_in + r_out[None, :] * R_stride_out
    R_tile = tl.load(R_ptr + R_offs)

    k_rows = tl.dot(k_tile, R_tile, out_dtype=tl.float32)
    mu_vals = tl.load(
        mu_rot_ptr + head_idx * mu_stride_head + dim_offs * mu_stride_dim
    ).to(tl.float32)
    k_rows = k_rows - mu_vals[None, :]

    hp_k_offs = (
        hp_loc[:, None] * hp_k_stride_loc
        + head_idx * hp_k_stride_head
        + dim_offs[None, :] * hp_k_stride_dim
    )
    tl.store(hp_k_ptr + hp_k_offs, k_rows, mask=active[:, None])

    v_base = (
        tok_offs[:, None] * v_input_stride_token
        + head_idx * v_input_stride_head
        + dim_offs[None, :] * v_input_stride_dim
    )
    v_vals = tl.load(
        v_input_ptr + v_base,
        mask=tok_mask[:, None],
        other=0.0,
    )
    hp_v_offs = (
        hp_loc[:, None] * hp_v_stride_loc
        + head_idx * hp_v_stride_head
        + dim_offs[None, :] * hp_v_stride_dim
    )
    tl.store(hp_v_ptr + hp_v_offs, v_vals, mask=active[:, None])


def _is_power_of_two_int(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _pick_hp_oscar_k_write_tile(head_dim: int) -> Tuple[int, int]:
    # tl.dot needs a reasonably sized M dimension; keep this fixed so decode
    # one-token writes still compile through the same path.
    block_tok = 16
    num_warps = 8 if head_dim >= 128 else 4
    return block_tok, num_warps


def _resolve_torch_dtype(name: str, *, kind: str) -> torch.dtype:
    """Map a friendly dtype name (``bf16``/``bfloat16``/``fp16``/``half``/``fp32``)
    to the corresponding ``torch.dtype``. ``kind`` is only used in the error
    message ("scale" / "HP" / etc.) so the caller's intent surfaces in the
    failure mode.
    """
    n = name.lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported {kind} dtype: {name}. Expected bf16/fp16/fp32.")


def resolve_scale_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="scale")


def resolve_hp_dtype(name: str) -> torch.dtype:
    return _resolve_torch_dtype(name, kind="HP")


def compute_page_geometry(hp_dtype: torch.dtype) -> Tuple[int, int]:
    """Return ``(N_H, N_Q)`` for int2 + ``hp_dtype``.

    ``N_Q`` is the int2 page size used by the paged quant allocator (and
    ``--page-size``). ``N_H`` is retained as ``1`` for documentation and for
    legacy callers, but no longer carries the LCM byte-equivalence invariant —
    HP and quant arenas are decoupled allocations under the slab design.
    """
    hp_itemsize = torch.empty(0, dtype=hp_dtype).element_size()
    return 1, 4 * hp_itemsize


def compute_recent_ring_size(hp_recent_tokens: int, n_q: int) -> int:
    # Max HP-recent occupancy between flushes is hp_recent + (N_Q - 1); the
    # ring reuses slots after the oldest N_Q have been demoted to quant.
    return int(hp_recent_tokens) + int(n_q) - 1


def _slice_local_k_mean(
    mu: torch.Tensor, local_num_heads: int, global_lid: int
) -> torch.Tensor:
    total_heads = int(mu.shape[0])
    if total_heads == local_num_heads:
        return mu.contiguous()
    tp_rank = int(get_attention_tp_rank())
    tp_size = int(get_attention_tp_size())
    if total_heads % tp_size == 0 and total_heads // tp_size == local_num_heads:
        head_start = tp_rank * local_num_heads
    elif total_heads < tp_size and local_num_heads == 1:
        head_start = tp_rank % total_heads
    else:
        raise ValueError(
            f"K-mean layer {global_lid} has {total_heads} heads, cannot map to "
            f"local_num_heads={local_num_heads} with attention TP size {tp_size}"
        )
    return mu.narrow(0, head_start, local_num_heads).contiguous()


def _load_oscar_k_means(
    path: str,
    layer_num: int,
    start_layer: int,
    head_dim: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
):
    state = torch.load(path, map_location="cpu")
    if "layers" not in state:
        raise ValueError(f"K-mean checkpoint at {path} missing 'layers' key")
    layers = state["layers"]
    loaded = []
    for local in range(layer_num):
        global_lid = start_layer + local
        entry = layers.get(global_lid, layers.get(str(global_lid)))
        if entry is None:
            raise ValueError(f"K-mean checkpoint at {path} missing layer {global_lid}")
        mu = entry["mu"].float()
        if mu.dim() != 2 or int(mu.shape[1]) != head_dim:
            raise ValueError(
                f"K-mean layer {global_lid} has shape {tuple(mu.shape)}, "
                f"expected [num_kv_heads, {head_dim}]"
            )
        loaded.append(_slice_local_k_mean(mu, num_heads, global_lid).to(dtype))

    return torch.stack(loaded, dim=0).to(device).contiguous()


def _rotate_one_k_mean(mu: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    if R.dim() == 2:
        return (mu.to(R.dtype) @ R).contiguous()
    if R.dim() == 3:
        return torch.einsum("hd,hde->he", mu.to(R.dtype), R).contiguous()
    raise ValueError(f"Oscar K rotation must have rank 2 or 3, got {R.dim()}")


def _precompute_rotated_k_means(k_means, R_k):
    if k_means is None:
        return None
    rotated = [_rotate_one_k_mean(k_means[i], R_k[i]) for i in range(len(k_means))]
    return torch.stack(rotated, dim=0).contiguous()


class UnifiedInt2HPKVPool(KVCache):
    """Unified HP + int2 MHA KV cache.

    The pool exposes:
      * ``k_buffer[l]``, ``v_buffer[l]``           – quant (int2 packed uint8) views
      * ``hp_k_buffer[l]``, ``hp_v_buffer[l]``     – HP (``hp_dtype``) views
      * ``k_scales_zeros[l]``, ``v_scales_zeros[l]`` – per-group scales+zeros in
        ``scale_dtype`` (bf16/fp16/fp32)

    The quant and HP views alias the same byte arena. Callers must treat a
    physical page as homogeneous (either tier) at any given time; this invariant
    is enforced by the ``UnifiedInt2HPKVAllocator`` that hands out slot ids into
    these views.
    """

    def __init__(
        self,
        num_quant_pages: int,
        hp_dtype: torch.dtype,
        hp_prefix_tokens: int,
        hp_recent_tokens: int,
        dtype: str,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        max_req_slots: int,
        v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        model_dtype: Optional[torch.dtype] = None,
        kv_cache_quant_group_size: Optional[int] = None,
        scale_dtype: torch.dtype = torch.bfloat16,
        num_hp_prefix_slots: int = 0,
    ):
        assert dtype == "int2", (
            "UnifiedInt2HPKVPool supports only int2 quant tier; got %s" % dtype
        )
        # Work around KVCache.__init__ dtype validation: it stores ``dtype`` as
        # a string and sets ``store_dtype=torch.uint8`` for int2.
        super().__init__(
            size=num_quant_pages,  # used by base class for sizing heuristics only
            page_size=1,
            dtype=dtype,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
            model_dtype=model_dtype,
        )

        self.num_quant_pages = int(num_quant_pages)
        self.hp_dtype = hp_dtype
        self.scale_dtype = scale_dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim if v_head_dim is not None else head_dim
        self.hp_prefix_tokens = int(hp_prefix_tokens)
        self.hp_recent_tokens = int(hp_recent_tokens)
        self.kv_cache_quant_group_size = kv_cache_quant_group_size

        self.N_H, self.N_Q = compute_page_geometry(hp_dtype)
        self.hp_recent_ring_size = compute_recent_ring_size(
            self.hp_recent_tokens, self.N_Q
        )
        if num_hp_prefix_slots > 0 and num_hp_prefix_slots % self.N_Q != 0:
            num_hp_prefix_slots = (
                (num_hp_prefix_slots + self.N_Q - 1) // self.N_Q * self.N_Q
            )
        self.num_hp_prefix_slots = int(num_hp_prefix_slots)
        self.slab_size = self.hp_recent_ring_size  # back-compat alias
        self._hp_offset = self.num_quant_pages * self.N_Q
        self._hp_recent_base = self.num_hp_prefix_slots

        # Window sizes must be N_Q-aligned so radix tree (page_size=N_Q) and
        # the flush kernel land on page boundaries.
        if self.hp_prefix_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_PREFIX_TOKENS ({self.hp_prefix_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        if self.hp_recent_tokens % self.N_Q != 0:
            raise ValueError(
                f"SGLANG_MIXED_KV_RECENT_TOKENS ({self.hp_recent_tokens}) "
                f"must be a multiple of N_Q ({self.N_Q})."
            )
        # Flush every N_Q decode steps demotes exactly N_Q HP-recent slots
        # into one quant page. Per-request counter (initialized at admission
        # to (hp_recent+N_Q-1)-H_0) keeps every flush whole-page.
        self.flush_interval = self.N_Q
        self.max_req_slots = int(max_req_slots)
        self._flush_counter = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )
        self._next_slab_offset = torch.zeros(
            (self.max_req_slots,), dtype=torch.int32, device=self.device
        )

        # Forward-done event stashed each iteration; consumed by
        # ``wait_pending_forward`` inside the flush apply phase.
        self._pending_forward_done = None

        self.k_quant_group_size, self.k_num_scale_groups = self._resolve_quant_grouping(
            self.head_dim, "K"
        )
        self.v_quant_group_size, self.v_num_scale_groups = self._resolve_quant_grouping(
            self.v_head_dim, "V"
        )
        assert self.head_dim % 4 == 0, (
            f"head_dim={self.head_dim} must be divisible by 4 for int2 packing"
        )
        assert self.v_head_dim % 4 == 0, (
            f"v_head_dim={self.v_head_dim} must be divisible by 4 for int2 packing"
        )

        self._create_arenas()

        # Cached attributes used by the rest of the stack.
        self.device_module = torch.get_device_module(self.device)
        self.alt_stream = None
        self.row_dim = self.head_num * self.head_dim  # for store_cache helpers
        self.same_kv_dim = self.head_dim == self.v_head_dim

        # Oscar rotation + clip. Per-layer or per-KV-head orthogonal matrices
        # are loaded in ``hp_dtype`` so write-side and read-side transforms are
        # plain bf16 GEMMs/einsums.
        self._oscar_cfg: OscarRotationConfig = load_oscar_rotation_config()
        self._k_clip_ratio: float = self._oscar_cfg.k_clip_ratio
        self._v_clip_ratio: float = self._oscar_cfg.v_clip_ratio
        self._R_k = load_oscar_rotations(
            self._oscar_cfg.k_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
            num_heads=self.head_num,
        )
        self._R_v = load_oscar_rotations(
            self._oscar_cfg.v_rotation_path,
            layer_num=self.layer_num,
            start_layer=self.start_layer,
            head_dim=self.v_head_dim,
            device=torch.device(self.device),
            dtype=self.hp_dtype,
            num_heads=self.head_num,
        )
        self._warned_headwise_fused_fallback = False

        self._K_mu_rot = None
        _k_mean_path = envs.SGLANG_OSCAR_K_MEAN_PATH.get()
        if envs.SGLANG_OSCAR_K_MEAN_IN_POOL.get() and _k_mean_path:
            k_means = _load_oscar_k_means(
                _k_mean_path,
                layer_num=self.layer_num,
                start_layer=self.start_layer,
                head_dim=self.head_dim,
                num_heads=self.head_num,
                device=torch.device(self.device),
                dtype=self.hp_dtype,
            )
            self._K_mu_rot = _precompute_rotated_k_means(k_means, self._R_k)
            logger.info(
                "UnifiedInt2HPKVPool: K-mean centering active in pool "
                "(path=%s, rotated-space subtract)",
                _k_mean_path,
            )
        logger.info(
            "UnifiedInt2HPKVPool: Oscar rotation enabled " "(k_clip=%.4f v_clip=%.4f)",
            self._k_clip_ratio,
            self._v_clip_ratio,
        )

        hp_total_slots = (
            self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size
        )
        self._finalize_allocation_log(hp_total_slots)
        hp_itemsize = torch.empty(0, dtype=self.hp_dtype).element_size()
        per_layer_hp_elems = (
            self.layer_num * self.head_num * (self.head_dim + self.v_head_dim)
        )
        hp_bytes = hp_total_slots * per_layer_hp_elems * hp_itemsize
        logger.info(
            "UnifiedInt2HPKVPool: HP arena reserves %.2f GB "
            "(hp_prefix_pool_slots=%d, max_req_slots=%d, recent_ring=%d "
            "= R=%d + N_Q-1=%d, P=%d, layers=%d, head_num=%d, "
            "head_dim+v_head_dim=%d, hp_dtype=%s)",
            hp_bytes / GB,
            self.num_hp_prefix_slots,
            self.max_req_slots,
            self.hp_recent_ring_size,
            self.hp_recent_tokens,
            self.N_Q - 1,
            self.hp_prefix_tokens,
            self.layer_num,
            self.head_num,
            self.head_dim + self.v_head_dim,
            str(self.hp_dtype),
        )

    # -- Configuration accessors -------------------------------------------

    def mixed_kv_enabled(self) -> bool:
        return True

    def stash_pending_forward(self, event) -> None:
        """Record the most recent forward-stream completion event.

        Called once per iteration from the scheduler. The event is consumed
        by :meth:`wait_pending_forward` at the apply boundary inside
        ``_alloc_for_decode_mixed``.
        """
        self._pending_forward_done = event

    def wait_pending_forward(self) -> None:
        """Order the current stream after the stashed forward-done event.

        Must be issued *before* the apply phase of the flush
        (``gpu_flush_int2_apply``), since the remap kernel writes
        ``req_to_token`` at positions the previous forward's attention is
        concurrently reading. Pre-apply work (allocator free, plan kernel)
        runs ahead of this wait so its host syncs don't block on the
        previous forward.
        """
        if self._pending_forward_done is None:
            return
        torch.cuda.current_stream().wait_event(self._pending_forward_done)
        self._pending_forward_done = None

    @property
    def hp_global_offset(self) -> int:
        return self._hp_offset

    @property
    def hp_size(self) -> int:
        return self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size

    @property
    def quant_size(self) -> int:
        return self.num_quant_pages * self.N_Q

    @property
    def hp_prefix_pool_slots(self) -> int:
        return self.num_hp_prefix_slots

    @property
    def hp_recent_base(self) -> int:
        """First HP-buffer index reserved for per-req recent slabs."""
        return self._hp_recent_base

    def release_req_slab(self, req_pool_idx) -> None:
        # Reset the per-req HP-recent cursor and flush counter so the next
        # request taking over ``req_pool_idx`` starts clean.
        if isinstance(req_pool_idx, torch.Tensor):
            idx = req_pool_idx.to(self._next_slab_offset.device).to(torch.int64)
            if idx.numel() == 0:
                return
            self._next_slab_offset[idx] = 0
            self._flush_counter[idx] = 0
        else:
            i = int(req_pool_idx)
            self._next_slab_offset[i] = 0
            self._flush_counter[i] = 0

    def _resolve_quant_grouping(
        self, head_dim: int, tensor_name: str
    ) -> tuple[int, int]:
        group_size = (
            head_dim
            if self.kv_cache_quant_group_size is None
            else self.kv_cache_quant_group_size
        )
        if group_size <= 0:
            raise ValueError(
                f"{tensor_name} kv_cache_quant_group_size must be positive, got {group_size}"
            )
        if head_dim % group_size != 0:
            raise ValueError(
                f"{tensor_name} head_dim ({head_dim}) must be divisible by "
                f"kv_cache_quant_group_size ({group_size})"
            )
        return group_size, head_dim // group_size

    # -- Arena construction ------------------------------------------------

    def _create_arenas(self):
        # HP arena layout: [shared prefix pool] [per-req recent slab 0]
        # [per-req recent slab 1] ... Quant arena is paged with N_Q slots
        # per page; scales/zeros are quant-only.
        hp_total_slots = (
            self.num_hp_prefix_slots + self.max_req_slots * self.hp_recent_ring_size
        )
        nq = self.num_quant_pages * self.N_Q
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                self.k_buffer = [
                    torch.zeros(
                        (nq, self.head_num, self.head_dim // 4),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_buffer = [
                    torch.zeros(
                        (nq, self.head_num, self.v_head_dim // 4),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.k_scales_zeros = [
                    torch.zeros(
                        (
                            nq,
                            self.head_num,
                            2 * self.k_num_scale_groups,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.v_scales_zeros = [
                    torch.zeros(
                        (
                            nq,
                            self.head_num,
                            2 * self.v_num_scale_groups,
                        ),
                        dtype=self.scale_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.hp_k_buffer = [
                    torch.zeros(
                        (
                            hp_total_slots,
                            self.head_num,
                            self.head_dim,
                        ),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
                self.hp_v_buffer = [
                    torch.zeros(
                        (
                            hp_total_slots,
                            self.head_num,
                            self.v_head_dim,
                        ),
                        dtype=self.hp_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]

        def _base_ptrs(tensors) -> torch.Tensor:
            return torch.tensor(
                [tensor.data_ptr() for tensor in tensors],
                dtype=torch.int64,
                device=self.device,
            )

        def _strides(t: torch.Tensor) -> tuple:
            return (int(t.stride(0)), int(t.stride(1)), int(t.stride(2)))

        self._flush_hp_k_ptrs = _base_ptrs(self.hp_k_buffer)
        self._flush_hp_v_ptrs = _base_ptrs(self.hp_v_buffer)
        self._flush_quant_k_ptrs = _base_ptrs(self.k_buffer)
        self._flush_quant_v_ptrs = _base_ptrs(self.v_buffer)
        self._flush_k_sz_ptrs = _base_ptrs(self.k_scales_zeros)
        self._flush_v_sz_ptrs = _base_ptrs(self.v_scales_zeros)
        self._flush_hp_k_stride = _strides(self.hp_k_buffer[0])
        self._flush_hp_v_stride = _strides(self.hp_v_buffer[0])
        self._flush_quant_k_stride = _strides(self.k_buffer[0])
        self._flush_quant_v_stride = _strides(self.v_buffer[0])
        self._flush_k_sz_stride = _strides(self.k_scales_zeros[0])
        self._flush_v_sz_stride = _strides(self.v_scales_zeros[0])

    # -- KVCache interface -------------------------------------------------

    def get_kv_size_bytes(self):
        k = sum(get_tensor_size_bytes(t) for t in self.k_buffer)
        k += sum(get_tensor_size_bytes(s) for s in self.k_scales_zeros)
        k += sum(get_tensor_size_bytes(t) for t in self.hp_k_buffer)
        v = sum(get_tensor_size_bytes(t) for t in self.v_buffer)
        v += sum(get_tensor_size_bytes(s) for s in self.v_scales_zeros)
        v += sum(get_tensor_size_bytes(t) for t in self.hp_v_buffer)
        return k, v

    def _layer_index(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def get_layer_head_num(self, layer_id: int) -> int:
        return self.head_num

    def get_layer_head_dim(self, layer_id: int) -> int:
        return self.head_dim

    def get_layer_v_head_dim(self, layer_id: int) -> int:
        return self.v_head_dim

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        # Triton backend asks for the quant view in the mixed path; HP view is
        # accessed via ``get_hp_key_buffer``.
        return self.k_buffer[self._layer_index(layer_id)]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_raw_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[self._layer_index(layer_id)]

    def get_raw_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[self._layer_index(layer_id)]

    def get_key_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.k_scales_zeros[self._layer_index(layer_id)]

    def get_value_scales_zeros(self, layer_id: int) -> torch.Tensor:
        return self.v_scales_zeros[self._layer_index(layer_id)]

    def get_hp_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_k_buffer[self._layer_index(layer_id)]

    def get_hp_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.hp_v_buffer[self._layer_index(layer_id)]

    def get_raw_kv_buffer(self, layer_id: int):
        idx = self._layer_index(layer_id)
        return {
            "k_buffer": self.k_buffer[idx],
            "v_buffer": self.v_buffer[idx],
            "k_scales_zeros": self.k_scales_zeros[idx],
            "v_scales_zeros": self.v_scales_zeros[idx],
            "dtype": "int2",
        }

    def _split_global_locs(self, loc: torch.Tensor):
        loc64 = loc.to(torch.int64)
        hp_mask = loc64 >= self._hp_offset
        quant_loc = loc64[~hp_mask]
        hp_loc_global = loc64[hp_mask] - self._hp_offset
        return quant_loc, hp_loc_global, hp_mask

    def apply_oscar_k_cache_transform(
        self, layer_id: int, cache_k: torch.Tensor
    ) -> torch.Tensor:
        idx = self._layer_index(layer_id)
        k_hp = _apply_oscar_rotation(cache_k, self._R_k[idx])
        if self._K_mu_rot is not None:
            k_hp.sub_(self._K_mu_rot[idx])
        return k_hp

    def _rotate_kv_inplace(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the per-layer Oscar rotation ``rows @ R`` to HP K/V tiles.

        Returns tensors in ``self.hp_dtype`` ready to be stored or packed.
        ``R_k`` / ``R_v`` are ``[head_dim, head_dim]`` bf16 on the KV device,
        loaded in ``__init__``.
        """
        idx = self._layer_index(layer_id)
        k_hp = self.apply_oscar_k_cache_transform(layer_id, cache_k)
        if v_rotation_absorbed:
            v_hp = cache_v.to(self.hp_dtype)
        else:
            v_hp = _apply_oscar_rotation(cache_v, self._R_v[idx])
        return k_hp, v_hp

    def _prepare_hp_kv_tensors(
        self,
        layer_id: int,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_rotated: bool,
        v_rotation_absorbed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the Oscar rotation to HP K/V and cast to ``hp_dtype``.
        ``already_rotated`` skips the rotation pre-pass.
        """
        if already_rotated:
            return cache_k.to(self.hp_dtype), cache_v.to(self.hp_dtype)
        return self._rotate_kv_inplace(layer_id, cache_k, cache_v, v_rotation_absorbed)

    def _set_hp_kv_buffer(
        self,
        layer_id: int,
        hp_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)
        _set_kv_buffer_impl(
            cache_k,
            cache_v,
            self.hp_k_buffer[idx],
            self.hp_v_buffer[idx],
            hp_loc,
            row_dim=self.row_dim,
            store_dtype=self.hp_dtype,
            device_module=self.device_module,
            alt_stream=self.alt_stream,
            same_kv_dim=self.same_kv_dim,
        )

    def _set_quant_kv_buffer_extend(
        self,
        layer_id: int,
        quant_loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_hadamard_transformed: bool,
        mixed_hp_offset: Optional[int] = None,
        v_rotation_absorbed: bool = False,
    ):
        """Prefill/extend-only: rotate (oscar R) + optional per-row clip +
        int2-pack + write quant slots.

        Decode-time flushes go through the dedicated GPU flush kernel
        (see ``gpu_flush_int2``); this method is *not* used for those.
        """
        idx = self._layer_index(layer_id)
        clip_on = self._k_clip_ratio > 0.0 or self._v_clip_ratio > 0.0

        if not already_hadamard_transformed:
            cache_k, cache_v = self._rotate_kv_inplace(
                layer_id, cache_k, cache_v, v_rotation_absorbed
            )
        else:
            cache_k = cache_k.to(self.hp_dtype)
            cache_v = cache_v.to(self.hp_dtype)

        if not clip_on:
            quantized_set_kv_int2_pretransformed_triton(
                cache_k,
                cache_v,
                quant_loc,
                self.k_buffer[idx],
                self.v_buffer[idx],
                self.k_scales_zeros[idx],
                self.v_scales_zeros[idx],
                hp_global_offset=mixed_hp_offset,
            )
            return

        quantized_set_kv_int2_pretransformed_clip_triton(
            cache_k,
            cache_v,
            quant_loc,
            self.k_buffer[idx],
            self.v_buffer[idx],
            self.k_scales_zeros[idx],
            self.v_scales_zeros[idx],
            self._k_clip_ratio,
            self._v_clip_ratio,
            hp_global_offset=mixed_hp_offset,
        )

    def _set_hp_kv_buffer_oscar_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        already_rotated: bool,
        v_rotation_absorbed: bool,
    ) -> bool:
        """Try fused HP write for ``K @ R_k - mu_rot`` plus V copy.

        Returns False when the existing generic rotate/copy path should be used.
        """
        if loc.numel() == 0:
            return True
        if already_rotated or not v_rotation_absorbed:
            return False
        if getattr(self, "_R_k", None) is None or self._K_mu_rot is None:
            return False
        if cache_k.dim() != 3 or cache_v.dim() != 3 or cache_k.shape != cache_v.shape:
            return False

        idx = self._layer_index(layer_id)
        num_tokens, num_heads, head_dim = cache_k.shape
        if num_tokens == 0:
            return True
        if not _is_power_of_two_int(int(head_dim)):
            return False

        R_k = self._R_k[idx]
        mu_rot = self._K_mu_rot[idx]
        if cache_k.dtype != R_k.dtype:
            return False
        if mu_rot.shape != (num_heads, head_dim):
            return False

        if R_k.dim() == 2:
            if R_k.shape != (head_dim, head_dim):
                return False
            headwise_r = False
            r_stride_head = 0
            r_stride_in, r_stride_out = R_k.stride(0), R_k.stride(1)
        elif R_k.dim() == 3:
            if R_k.shape != (num_heads, head_dim, head_dim):
                return False
            headwise_r = True
            r_stride_head, r_stride_in, r_stride_out = R_k.stride()
        else:
            return False

        hp_k = self.hp_k_buffer[idx]
        hp_v = self.hp_v_buffer[idx]
        if hp_k.shape[1] != num_heads or hp_v.shape[1] != num_heads:
            return False
        if hp_k.shape[2] != head_dim or hp_v.shape[2] != head_dim:
            return False

        block_tok, num_warps = _pick_hp_oscar_k_write_tile(int(head_dim))
        grid = (triton.cdiv(num_tokens, block_tok), num_heads)
        _set_hp_kv_oscar_k_write_kernel[grid](
            cache_k,
            cache_v,
            R_k,
            mu_rot,
            hp_k,
            hp_v,
            loc,
            num_tokens,
            num_heads,
            cache_k.stride(0),
            cache_k.stride(1),
            cache_k.stride(2),
            cache_v.stride(0),
            cache_v.stride(1),
            cache_v.stride(2),
            r_stride_head,
            r_stride_in,
            r_stride_out,
            mu_rot.stride(0),
            mu_rot.stride(1),
            hp_k.stride(0),
            hp_k.stride(1),
            hp_k.stride(2),
            hp_v.stride(0),
            hp_v.stride(1),
            hp_v.stride(2),
            HP_OFFSET=int(self._hp_offset),
            HEAD_DIM=head_dim,
            BLOCK_TOK=block_tok,
            HEADWISE_R=headwise_r,
            num_warps=num_warps,
            num_stages=1,
        )
        return True

    def _set_mixed_hp_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        idx = self._layer_index(layer_id)

        def _launch(src: torch.Tensor, dst: torch.Tensor):
            src2 = src.reshape(src.shape[0], -1)
            dst2 = dst.reshape(dst.shape[0], -1)
            row_dim = src2.shape[1]
            if row_dim == 0 or src2.shape[0] == 0:
                return
            block_row = min(1024, triton.next_power_of_2(row_dim))
            grid = (src2.shape[0], triton.cdiv(row_dim, block_row))
            _set_mixed_hp_buffer_kernel[grid](
                src2,
                dst2,
                loc,
                src2.shape[0],
                row_dim,
                src2.stride(0),
                src2.stride(1),
                dst2.stride(0),
                dst2.stride(1),
                HP_OFFSET=int(self._hp_offset),
                BLOCK_ROW=block_row,
                num_warps=4,
                num_stages=1,
            )

        _launch(cache_k, self.hp_k_buffer[idx])
        _launch(cache_v, self.hp_v_buffer[idx])

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
        already_hadamard_transformed: bool = False,
        is_decode: bool = False,
    ):
        """Write K/V to the unified pool.

        ``is_decode`` selects the path:
            * ``True`` -- single-token decode write. Caller fills ``loc`` with
              valid HP slot ids (from ``allocator.alloc_hp_recent``); we
              write only the HP buffer, with no boolean masking (safe under
              CUDA-graph capture).
            * ``False`` (extend / prefill) -- mixed write: int2 quant slots get
              the rotated+clipped pack, HP slots get the bf16 row.

        Callers must pass ``is_decode`` based on
        ``forward_batch.forward_mode.is_decode_or_idle()``. Capture state is
        *not* a reliable proxy: piecewise CUDA graph captures parts of prefill
        (would mis-route to the HP-only branch) and ``--disable-cuda-graph``
        runs decode eagerly (would mis-route to the quant+HP branch).
        """
        if loc.numel() == 0:
            return

        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        v_rotation_absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))

        if is_decode:
            if self._set_hp_kv_buffer_oscar_k_fused(
                layer_id,
                loc,
                cache_k,
                cache_v,
                already_hadamard_transformed,
                v_rotation_absorbed,
            ):
                return
            hp_local = loc.to(torch.int64) - self._hp_offset
            cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
                layer_id,
                cache_k,
                cache_v,
                already_hadamard_transformed,
                v_rotation_absorbed,
            )
            self._set_hp_kv_buffer(layer_id, hp_local, cache_k_hp, cache_v_hp)
            return

        self._set_quant_kv_buffer_extend(
            layer_id,
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            mixed_hp_offset=int(self._hp_offset),
            v_rotation_absorbed=v_rotation_absorbed,
        )
        if not self._set_hp_kv_buffer_oscar_k_fused(
            layer_id,
            loc,
            cache_k,
            cache_v,
            already_hadamard_transformed,
            v_rotation_absorbed,
        ):
            cache_k_hp, cache_v_hp = self._prepare_hp_kv_tensors(
                layer_id,
                cache_k,
                cache_v,
                already_hadamard_transformed,
                v_rotation_absorbed,
            )
            self._set_mixed_hp_kv_buffer(layer_id, loc, cache_k_hp, cache_v_hp)

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        # Moves only make sense within the same tier. The allocator ensures
        # that; callers should split by tier before calling here.
        tgt_q, tgt_hp, tgt_mask = self._split_global_locs(tgt_loc)
        src_q, src_hp, src_mask = self._split_global_locs(src_loc)
        assert torch.equal(
            tgt_mask, src_mask
        ), "move_kv_cache requires src/tgt tiers to match"
        for l in range(self.layer_num):
            if tgt_q.numel() > 0:
                self.k_buffer[l][tgt_q] = self.k_buffer[l][src_q]
                self.v_buffer[l][tgt_q] = self.v_buffer[l][src_q]
                self.k_scales_zeros[l][tgt_q] = self.k_scales_zeros[l][src_q]
                self.v_scales_zeros[l][tgt_q] = self.v_scales_zeros[l][src_q]
            if tgt_hp.numel() > 0:
                self.hp_k_buffer[l][tgt_hp] = self.hp_k_buffer[l][src_hp]
                self.hp_v_buffer[l][tgt_hp] = self.hp_v_buffer[l][src_hp]

    def get_cpu_copy(self, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")

    def load_cpu_copy(self, kv_cache_cpu, indices):
        raise NotImplementedError("CPU offload is not supported by UnifiedInt2HPKVPool")
