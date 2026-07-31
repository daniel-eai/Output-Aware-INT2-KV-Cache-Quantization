"""Compatibility shims for the calibration dump environment."""

import types


def _pytorch_sampling_fallback(probs, *args, **kwargs):
    """Return a deterministic token for one-token dump requests."""
    import torch

    return torch.argmax(probs, dim=-1)


def _install_sgl_kernel_compat():
    try:
        import sgl_kernel
    except ImportError:
        return

    runtime_sampling_fallbacks = (
        "top_k_top_p_sampling_from_probs",
        "top_p_sampling_from_probs",
        "min_p_sampling_from_probs",
        "top_k_top_p_sampling_from_logits",
        "top_k_mask_logits",
    )

    class _SglKernelProxy(types.ModuleType):
        def __getattr__(self, name):
            if name in runtime_sampling_fallbacks:
                return _pytorch_sampling_fallback

            def _stub(*args, **kwargs):
                raise NotImplementedError(
                    f"sgl_kernel.{name} is unavailable in this build"
                )

            return _stub

    sgl_kernel.__class__ = _SglKernelProxy


def _install_sgl_kernel_version_shim():
    import importlib.metadata as _md

    _orig_version = _md.version

    def _patched_version(name):
        try:
            return _orig_version(name)
        except _md.PackageNotFoundError:
            if name in ("sgl-kernel", "sgl_kernel"):
                return "99.0.0"
            raise

    _md.version = _patched_version


_install_sgl_kernel_compat()
_install_sgl_kernel_version_shim()
