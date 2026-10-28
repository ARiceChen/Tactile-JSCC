"""Additive white Gaussian noise for real-valued channel symbols."""

from __future__ import annotations

import torch
from torch import nn


class WirelessChannel(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, symbols: torch.Tensor, snr_db: float = 10.0,
                generator: torch.Generator | None = None) -> torch.Tensor:
        if symbols.ndim != 2 or symbols.shape[1] < 1:
            raise ValueError("symbols must have shape [batch, real_symbol_count]")
        power = symbols.detach().square().mean().clamp_min(self.eps)
        noise_std = power.sqrt() * 10.0 ** (-float(snr_db) / 20.0)
        noise = torch.randn(symbols.shape, device=symbols.device,
                            dtype=symbols.dtype, generator=generator)
        return symbols + noise_std * noise
