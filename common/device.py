"""Device and dtype resolution.

This is a CUDA-first codebase, but free-tier GPUs are not uniform: Kaggle/Colab hand
out T4 (Turing, sm_75) most of the time and A10G/A100 occasionally. T4 has fp16 tensor
cores but **no bf16**, so hardcoding bf16 autocast either errors out or silently falls
back to fp32 and blows the 16GB budget. Dtype is therefore resolved from the actual
compute capability, and training code must stay correct on both paths.

MPS and CPU are kept only so unit tests and shape-level smoke tests can run off-GPU.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Accelerator:
    device: torch.device
    amp_dtype: torch.dtype | None  # None => run in fp32, no autocast
    name: str
    total_memory_gb: float

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def autocast(self):
        """Autocast context for this accelerator, or a no-op if fp32."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def grad_scaler(self) -> torch.amp.GradScaler | None:
        """fp16 needs loss scaling; bf16 and fp32 do not."""
        if self.amp_dtype is torch.float16:
            return torch.amp.GradScaler(self.device.type)
        return None


def pick_device(prefer: str | None = None) -> Accelerator:
    """Resolve the accelerator and the dtype it can actually train in.

    `prefer` forces a backend ("cuda", "mps", "cpu") for A/B testing; otherwise
    CUDA > MPS > CPU.
    """
    if prefer == "cpu" or (prefer is None and not _cuda() and not _mps()):
        return Accelerator(torch.device("cpu"), None, "cpu", 0.0)

    if prefer == "cuda" or (prefer is None and _cuda()):
        props = torch.cuda.get_device_properties(0)
        major = props.major
        # bf16 needs Ampere (sm_80) or newer. T4/V100 are sm_75/sm_70 -> fp16.
        amp = torch.bfloat16 if major >= 8 else torch.float16
        return Accelerator(
            torch.device("cuda"), amp, props.name, props.total_memory / 1024**3
        )

    # MPS is a smoke-test path only. Keep it in fp32: autocast coverage is partial,
    # and a dtype artifact here would masquerade as a porting bug during diffing.
    return Accelerator(torch.device("mps"), None, "mps", 0.0)


def _cuda() -> bool:
    return torch.cuda.is_available()


def _mps() -> bool:
    return torch.backends.mps.is_available()


def describe(acc: Accelerator) -> str:
    dt = "fp32" if acc.amp_dtype is None else str(acc.amp_dtype).replace("torch.", "")
    mem = f", {acc.total_memory_gb:.1f}GB" if acc.total_memory_gb else ""
    return f"{acc.name} [{acc.device.type}, {dt}{mem}]"
