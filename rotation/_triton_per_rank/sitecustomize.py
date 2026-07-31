"""Redirect each tensor-parallel rank to a separate Triton cache."""

import os


def _apply():
    base = os.environ.get("OSCAR_TRITON_PER_RANK_BASE")
    if not base:
        return
    rank = os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
    target = os.path.join(base, f"rank{rank}")
    try:
        os.makedirs(target, exist_ok=True)
    except OSError:
        return
    os.environ["TRITON_CACHE_DIR"] = target


_apply()
