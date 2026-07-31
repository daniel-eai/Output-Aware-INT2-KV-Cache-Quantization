#!/usr/bin/env python3
"""Shared utilities for OptR calibration."""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import torch

# Repo root = parent of this package.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "rotation"))

from kv_calib_utils import (  # noqa: E402
    DEV,
    DT,
    ensure_parent,
    load_fp_sequences,
    load_headwise_rotation,
    load_layer_entry,
    load_layer_mu,
    parse_chunks,
    quant_int2_ste,
    summarize_ortho,
)

DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B-Thinking-2507"


def _model_cache_name(model_path):
    return "models--" + model_path.replace("/", "--")


def _candidate_cache_roots():
    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.extend([Path(hf_home) / "hub", Path(hf_home)])
    roots.extend(
        [
            Path(REPO) / "caches" / "hf" / "hub",
            Path("/shared/huggingface/hub"),
        ]
    )
    out = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def resolve_snapshot(model_path=None, snap_dir=None):
    """Resolve a local HF snapshot directory containing config + safetensors."""
    if snap_dir:
        path = Path(snap_dir).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"HF snapshot does not exist: {path}")
        return str(path)

    glob_override = os.environ.get("OPTR_HF_SNAPSHOT_GLOB")
    if glob_override:
        cands = sorted(Path(p) for p in glob.glob(os.path.expanduser(glob_override)))
        if not cands:
            raise FileNotFoundError(f"no snapshot matching {glob_override}")
        return str(cands[-1])

    model_path = model_path or DEFAULT_MODEL_PATH
    local = Path(model_path).expanduser()
    if local.exists():
        if (local / "config.json").exists():
            return str(local)
        snaps = sorted((local / "snapshots").glob("*/config.json"))
        if snaps:
            return str(snaps[-1].parent)

    cache_name = _model_cache_name(model_path)
    cands = []
    for root in _candidate_cache_roots():
        cands.extend(sorted((root / cache_name / "snapshots").glob("*/config.json")))
    if cands:
        return str(cands[-1].parent)

    raise FileNotFoundError(
        f"could not find local HF snapshot for {model_path}; pass --hf-snapshot "
        "or set OPTR_HF_SNAPSHOT_GLOB"
    )


def load_model_geometry(model_path=None, snap_dir=None):
    snap = resolve_snapshot(model_path=model_path, snap_dir=snap_dir)
    cfg = json.load(open(Path(snap) / "config.json"))
    hidden = int(cfg["hidden_size"])
    layers = int(cfg["num_hidden_layers"])
    q_heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", q_heads))
    head_dim = int(cfg.get("head_dim") or (hidden // q_heads))
    return {
        "snapshot": snap,
        "hidden": hidden,
        "layers": layers,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }


class WOCache:
    """Lazily load per-layer o_proj head slices from the HF safetensors."""

    def __init__(
        self, snap_dir=None, model_path=None, q_heads=None, head_dim=None, hidden=None
    ):
        geom = load_model_geometry(model_path=model_path, snap_dir=snap_dir)
        self.snap = geom["snapshot"]
        self.q_heads = int(q_heads or geom["q_heads"])
        self.head_dim = int(head_dim or geom["head_dim"])
        self.hidden = int(hidden or geom["hidden"])
        idx = sorted(Path(self.snap).glob("*.safetensors.index.json"))
        if idx:
            wmap = json.load(open(idx[0]))["weight_map"]
            self.weight_map = {k: str(Path(self.snap) / v) for k, v in wmap.items()}
        else:
            # single shard
            shards = sorted(Path(self.snap).glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"no safetensors files found in {self.snap}")
            shard = str(shards[0])
            from safetensors import safe_open

            with safe_open(shard, framework="pt") as t:
                self.weight_map = {k: shard for k in t.keys()}
        self._cache = {}

    def heads(self, layer_id, device=DEV):
        """Return W_O slices for a layer: [q_heads, hidden, head_dim] (float32)."""
        if layer_id in self._cache:
            return self._cache[layer_id].to(device)
        name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        path = self.weight_map[name]
        from safetensors import safe_open

        with safe_open(path, framework="pt") as t:
            W = t.get_tensor(name).to(DT)  # [hidden, q_heads*head_dim]
        hd, qh = self.head_dim, self.q_heads
        # [hidden, qh, hd] -> [qh, hidden, hd]
        Wh = W.reshape(self.hidden, qh, hd).permute(1, 0, 2).contiguous()
        self._cache[layer_id] = Wh.cpu()
        return Wh.to(device)


def _causal_mask(lo, L, device):
    pos = torch.arange(lo, L, device=device).unsqueeze(1)
    return torch.arange(L, device=device).unsqueeze(0) > pos


def softmax_probs(q_tail, kh, mask, scale):
    logits = ((q_tail @ kh.T) * scale).masked_fill(mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def head_layout(seq_q, kvh):
    qh = seq_q.shape[1]
    if qh % kvh != 0:
        raise ValueError(f"q_heads={qh} not divisible by kv_heads={kvh}")
    return qh // kvh
