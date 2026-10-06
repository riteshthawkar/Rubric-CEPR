from __future__ import annotations

from typing import Any


def resolve_torch_device(requested_device: str = "auto") -> str:
    if requested_device != "auto":
        return requested_device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_dtype(torch_module: Any, requested_dtype: str | None, device: str) -> Any:
    if requested_dtype is None or requested_dtype == "auto":
        return torch_module.float32 if device == "cpu" else torch_module.bfloat16

    aliases = {
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    normalized = aliases.get(requested_dtype.lower())
    if normalized is None:
        raise ValueError(f"Unsupported torch dtype: {requested_dtype}")
    return getattr(torch_module, normalized)
