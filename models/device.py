"""Cross-platform compute-device selection.

Resolves the best available device so the same codebase runs on:
  - Ubuntu PC with NVIDIA GPU  -> "cuda"
  - Apple Silicon M1           -> "mps"
  - anything else              -> "cpu"

A non-empty DEVICE environment variable overrides auto-detection.
"""

import os

# Apple's MPS backend does not implement every PyTorch op; fall back to CPU
# for the unsupported ones instead of crashing. Must be set before torch use.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# PyTorch 2.14+ routes some ordinary ops through a "native" DSL (Triton) that JIT-compiles
# a C shim at runtime and therefore needs a host C compiler (gcc/clang). On machines without
# a compiler this raises "Failed to find C compiler" on the first CUDA op (e.g. Qwen3 RoPE).
# We use eager kernels instead so GPU inference works without a build toolchain.
# See docs/MILESTONE_2.md for the full investigation.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

from config import DEVICE_OVERRIDE


def get_device() -> str:
    """Return the best available compute device: cuda > mps > cpu.

    Honours the ``DEVICE`` env var override when set.
    """
    if DEVICE_OVERRIDE:
        return DEVICE_OVERRIDE

    import torch

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def describe_device() -> str:
    """Human-readable one-line device summary for logs and the UI."""
    device = get_device()
    try:
        import torch

        if device == "cuda":
            name = torch.cuda.get_device_name(0)
            return f"cuda ({name})"
        if device == "mps":
            return "mps (Apple Silicon)"
    except Exception:
        # torch missing or driver issue — still report the resolved device.
        pass
    return device
