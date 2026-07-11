"""Core DSP helpers used by the DDSP guitar inference stack."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import einops
import numpy as np
import torch

DB_RANGE = -80.0
EPS = 1e-7


def scale_db(db: torch.Tensor) -> torch.Tensor:
    """Scale decibels from [-80, 0] to [0, 1]."""
    return (db / DB_RANGE) + 1.0


def inv_scale_db(db_scaled: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`scale_db`."""
    return (db_scaled - 1.0) * DB_RANGE


def hz_to_midi(f: torch.Tensor | np.ndarray | float) -> torch.Tensor | np.ndarray:
    if isinstance(f, (int, float, np.ndarray)):
        f = np.clip(f, EPS, np.inf)
        return 12 * np.log2(f / 440.0) + 69.0
    return 12 * torch.log2(torch.clamp(f, min=EPS) / 440.0) + 69.0


def midi_to_hz(m: torch.Tensor) -> torch.Tensor:
    f = 440.0 * torch.pow(2.0, (m - 69.0) / 12.0)
    return torch.clamp(f, min=EPS)


def midi_to_unit(m: torch.Tensor, midi_min: float, midi_max: float, clip: bool = True) -> torch.Tensor:
    midi = torch.clamp(m, midi_min, midi_max) if clip else m
    return (midi - midi_min) / (midi_max - midi_min)


def unit_to_midi(u: torch.Tensor, midi_min: float, midi_max: float, clip: bool = True) -> torch.Tensor:
    unit = torch.clamp(u, 0.0, 1.0) if clip else u
    return unit * (midi_max - midi_min) + midi_min


def hz_to_unit(f: torch.Tensor, hz_min: float, hz_max: float, clip: bool = True) -> torch.Tensor:
    return midi_to_unit(hz_to_midi(f), hz_to_midi(hz_min), hz_to_midi(hz_max), clip)


def unit_to_hz(u: torch.Tensor, hz_min: float, hz_max: float, clip: bool = True) -> torch.Tensor:
    return midi_to_hz(unit_to_midi(u, hz_to_midi(hz_min), hz_to_midi(hz_max), clip))


def fold(x: torch.Tensor) -> torch.Tensor:
    """Fold the channel dimension into the batch dimension."""
    if x.dim() == 3:
        return einops.rearrange(x, "batch channel time -> (batch channel) time")
    if x.dim() == 4:
        return einops.rearrange(x, "batch channel time feat -> (batch channel) time feat")
    raise ValueError("fold expects a 3D or 4D tensor")


def unfold(x: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Inverse of :func:`fold`."""
    if x.dim() == 2:
        return einops.rearrange(x, "(batch channel) time -> batch channel time", channel=n_channels)
    if x.dim() == 3:
        return einops.rearrange(x, "(batch channel) time feat -> batch channel time feat", channel=n_channels)
    raise ValueError("unfold expects a 2D or 3D tensor")


def convert_dtype(tensors: Dict[str, torch.Tensor], dtype_map: Dict[torch.dtype, torch.dtype]):
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor) and value.dtype in dtype_map:
            tensors[key] = value.to(dtype_map[value.dtype])
    return tensors


def resample_feature(x: torch.Tensor, out_samples: int, mode: str) -> torch.Tensor:
    """Resample the last dimension to ``out_samples`` using interpolation."""
    if x.dim() < 4:
        tensor = x
        added_channel = False
        if tensor.dim() < 3:
            tensor = tensor.unsqueeze(-1)
            added_channel = True
        tensor = tensor.transpose(1, -1)
        tensor = torch.nn.functional.interpolate(tensor, out_samples, mode=mode)
        tensor = tensor.transpose(1, -1)
        if added_channel:
            tensor = tensor.squeeze(-1)
        return tensor

    batch, channel, _, feature = x.shape
    tensor = einops.rearrange(x, "b c t f -> (b c f) t")
    tensor = torch.nn.functional.interpolate(tensor, out_samples, mode=mode)
    return einops.rearrange(tensor, "(b c f) t -> b c t f", b=batch, c=channel)


@dataclass
class Quantizer:
    """Simple linear quantizer used for classification targets."""

    value_range: tuple[float, float]
    n_bins: int

    def __post_init__(self):
        self.min_value, self.max_value = self.value_range

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, self.min_value, self.max_value)
        return linear_quantize(x, self.min_value, self.max_value, self.n_bins)

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return linear_dequantize(x, self.min_value, self.max_value, self.n_bins)


def linear_quantize(value: torch.Tensor, min_value: float, max_value: float, n_bins: int) -> torch.Tensor:
    value = value.squeeze(-1)
    scale = (value - min_value) / (max_value - min_value)
    return torch.round(scale * (n_bins - 1)).long()


def linear_dequantize(bin_idx: torch.Tensor, min_value: float, max_value: float, n_bins: int) -> torch.Tensor:
    return (bin_idx.float() / (n_bins - 1) * (max_value - min_value) + min_value).unsqueeze(-1)


__all__ = [
    "DB_RANGE",
    "EPS",
    "Quantizer",
    "convert_dtype",
    "fold",
    "hz_to_midi",
    "hz_to_unit",
    "inv_scale_db",
    "linear_dequantize",
    "linear_quantize",
    "midi_to_hz",
    "midi_to_unit",
    "resample_feature",
    "scale_db",
    "unit_to_hz",
    "unfold",
]
