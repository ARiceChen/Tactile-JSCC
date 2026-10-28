"""Tactile-JSCC encoder and decoders."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from models.tactile_channel import WirelessChannel


BLOCK_LENGTH = 256


def _dct_basis(symbols: int, length: int = BLOCK_LENGTH) -> torch.Tensor:
    """Orthonormal low-frequency basis for initializing learned signal paths."""
    sample = torch.arange(length, dtype=torch.float32)
    frequency = torch.arange(symbols, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi / length * (sample + 0.5) * frequency)
    basis *= math.sqrt(2.0 / length)
    basis[0] *= 1.0 / math.sqrt(2.0)
    return basis


def symbols_for_ratio(compression_ratio: int, block_length: int = BLOCK_LENGTH) -> int:
    """Number of real channel symbols produced by the final encoder layer."""
    if compression_ratio < 1 or compression_ratio > block_length:
        raise ValueError("compression_ratio must be in [1, block_length]")
    return max(1, block_length // compression_ratio)


class ESProjection(nn.Module):
    """Feature entropy times learned stability, normalized over time."""

    def __init__(self, features: int, eps: float = 1e-8) -> None:
        super().__init__()
        if features < 2:
            raise ValueError("feature-space entropy requires at least two features")
        self.stability = nn.Linear(features, features)
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, features]")
        probability = torch.softmax(features, dim=-1)
        entropy = -(probability * (probability + self.eps).log()).sum(dim=-1)
        stability = F.relu(self.stability(features)).sum(dim=-1)
        score = entropy * stability
        return (score / (score.amax(dim=1, keepdim=True) + self.eps)).unsqueeze(-1)


class ESAAttention(nn.Module):
    """Entropy-modulated keys and values with temporal projections."""

    def __init__(self, length: int, width: int, heads: int = 8, projected_length: int = 64) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if projected_length < 1 or projected_length > length:
            raise ValueError("projected_length must be in [1, length]")
        self.length = length
        self.heads = heads
        self.head_width = width // heads
        self.esa = ESProjection(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.key_projection = nn.Linear(length, projected_length, bias=False)
        self.value_projection = nn.Linear(length, projected_length, bias=False)
        self.output = nn.Linear(width, width)

    def forward(self, features: torch.Tensor, modulation_features: torch.Tensor | None = None) -> torch.Tensor:
        batch, length, width = features.shape
        if length != self.length:
            raise ValueError(f"expected {self.length} temporal tokens, got {length}")
        alpha = self.esa(features if modulation_features is None else modulation_features)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)

        query = heads(self.query(features))
        key = heads(alpha * self.key(features))
        value = heads(alpha * self.value(features))
        key = self.key_projection(key.transpose(-1, -2)).transpose(-1, -2)
        value = self.value_projection(value.transpose(-1, -2)).transpose(-1, -2)
        attention = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(self.head_width), dim=-1)
        mixed = attention @ value
        mixed = mixed.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(mixed)


class ESABlock(nn.Module):
    """One attention layer with modulation before feature normalization."""

    def __init__(self, length: int, width: int, heads: int,
                 projected_length: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = ESAAttention(length, width, heads, projected_length)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(features)
        features = features + self.attention(normalized, modulation_features=features)
        return features + self.feedforward(self.feedforward_norm(features))


class ESAEncoder(nn.Module):
    """Entropy-adaptive encoder with a learned linear signal path."""

    def __init__(self, compression_ratio: int, width: int = 280, layers: int = 1,
                 heads: int = 8, projected_length: int = 64) -> None:
        super().__init__()
        self.symbol_count = symbols_for_ratio(compression_ratio)
        self.esa_residual_cap = 0.25
        self.input_projection = nn.Linear(1, width, bias=False)
        nn.init.normal_(self.input_projection.weight, std=4.0)
        self.blocks = nn.Sequential(*(
            ESABlock(BLOCK_LENGTH, width, heads, projected_length) for _ in range(layers)
        ))
        self.token_projection = nn.Linear(width, 1)
        self.compression = nn.Linear(BLOCK_LENGTH, self.symbol_count)
        self.esa_output_norm = nn.LayerNorm(BLOCK_LENGTH, elementwise_affine=False)
        self.signal_skip = nn.Linear(BLOCK_LENGTH, self.symbol_count, bias=False)
        with torch.no_grad():
            self.signal_skip.weight.copy_(_dct_basis(self.symbol_count))
            self.compression.weight.zero_()
            self.compression.bias.zero_()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 2 or signal.shape[1] != BLOCK_LENGTH:
            raise ValueError(f"signal must have shape [batch, {BLOCK_LENGTH}]")
        features = self.input_projection(signal.unsqueeze(-1))
        encoded = self.compression(self.esa_output_norm(
            self.token_projection(self.blocks(features)).squeeze(-1)
        ))
        encoded = self.esa_residual_cap * torch.tanh(encoded / self.esa_residual_cap)
        return encoded + self.signal_skip(signal)


class ESADecoder(nn.Module):
    """Entropy-adaptive coarse decoder."""

    def __init__(self, compression_ratio: int, width: int = 280, layers: int = 1,
                 heads: int = 8, projected_length: int = 64) -> None:
        super().__init__()
        self.symbol_count = symbols_for_ratio(compression_ratio)
        self.expansion = nn.Linear(self.symbol_count, BLOCK_LENGTH)
        self.input_projection = nn.Linear(1, width, bias=False)
        nn.init.normal_(self.input_projection.weight, std=4.0)
        self.blocks = nn.Sequential(*(
            ESABlock(BLOCK_LENGTH, width, heads, projected_length) for _ in range(layers)
        ))
        self.token_projection = nn.Linear(width, 1)
        with torch.no_grad():
            self.expansion.weight.copy_(_dct_basis(self.symbol_count).T)
            self.expansion.bias.zero_()
            self.token_projection.weight.zero_()
            self.token_projection.bias.zero_()

    def forward(self, received: torch.Tensor) -> torch.Tensor:
        if received.ndim != 2 or received.shape[1] != self.symbol_count:
            raise ValueError(f"received must have shape [batch, {self.symbol_count}]")
        expanded = self.expansion(received)
        features = self.input_projection(expanded.unsqueeze(-1))
        residual = self.token_projection(self.blocks(features)).squeeze(-1)
        return expanded + residual


class DeformableConv1d(nn.Module):
    """One-dimensional deformable convolution with differentiable linear sampling."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.offsets = nn.Conv1d(in_channels, kernel_size, kernel_size=3, padding=1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        length = features.shape[-1]
        if length < 2:
            raise ValueError("deformable convolution requires at least two time steps")
        offsets = self.offsets(features)
        locations = torch.arange(length, device=features.device, dtype=features.dtype)[None, None, :]
        taps = torch.arange(self.kernel_size, device=features.device, dtype=features.dtype)[None, :, None]
        position = locations + taps - self.kernel_size // 2 + offsets
        x_coordinate = 2 * position / (length - 1) - 1
        grid = torch.stack((x_coordinate, torch.zeros_like(x_coordinate)), dim=-1)
        sampled = F.grid_sample(features.unsqueeze(2), grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        return torch.einsum("bckl,ock->bol", sampled, self.weight) + self.bias[None, :, None]


class TFDecoder(nn.Module):
    """Local deformable denoising and frequency-domain refinement."""

    def __init__(self, frequency_width: int = 64, local_width: int = 32) -> None:
        super().__init__()
        self.local = nn.Sequential(
            DeformableConv1d(1, local_width), nn.GELU(),
            nn.Conv1d(local_width, local_width, kernel_size=3, padding=1), nn.GELU()
        )
        self.frequency_input = nn.Linear(2, frequency_width)
        self.frequency_transformer = nn.TransformerEncoderLayer(
            d_model=frequency_width, nhead=4, dim_feedforward=2 * frequency_width,
            dropout=0.0, batch_first=True, activation="gelu"
        )
        self.frequency_output = nn.Linear(frequency_width, 2)
        self.global_projection = nn.Conv1d(1, local_width, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv1d(2 * local_width, local_width, kernel_size=1), nn.GELU(),
            nn.Conv1d(local_width, 1, kernel_size=1)
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        if coarse.ndim != 2 or coarse.shape[1] != BLOCK_LENGTH:
            raise ValueError(f"coarse signal must have shape [batch, {BLOCK_LENGTH}]")
        local = self.local(coarse.unsqueeze(1))
        spectrum = torch.fft.rfft(coarse, dim=-1, norm="ortho")
        complex_features = torch.stack((spectrum.real, spectrum.imag), dim=-1)
        refined = self.frequency_output(
            self.frequency_transformer(self.frequency_input(complex_features))
        )
        refined_spectrum = torch.complex(refined[..., 0], refined[..., 1])
        global_signal = torch.fft.irfft(refined_spectrum, n=BLOCK_LENGTH, dim=-1, norm="ortho")
        global_features = self.global_projection(global_signal.unsqueeze(1))
        refined = self.fusion(torch.cat((local, global_features), dim=1)).squeeze(1)
        return coarse + refined


class TactileJSCC(nn.Module):
    """End-to-end Tactile-JSCC: ESA encoder, wireless channel, ESA and TF decoders."""

    def __init__(self, compression_ratio: int = 32, width: int = 280, esa_layers: int = 1,
                 heads: int = 8, projected_length: int = 64) -> None:
        super().__init__()
        self.compression_ratio = compression_ratio
        self.encoder = ESAEncoder(compression_ratio, width, esa_layers, heads, projected_length)
        self.channel = WirelessChannel()
        self.coarse_decoder = ESADecoder(compression_ratio, width, esa_layers, heads,
                                         projected_length)
        self.fine_decoder = TFDecoder()

    @property
    def esa_encoder(self) -> ESAEncoder:
        return self.encoder

    @property
    def esa_decoder(self) -> ESADecoder:
        return self.coarse_decoder

    @property
    def tf_decoder(self) -> TFDecoder:
        return self.fine_decoder

    @property
    def symbol_count(self) -> int:
        return self.encoder.symbol_count

    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        return self.esa_encoder(signal)

    def decode(self, received: torch.Tensor, return_coarse: bool = False):
        coarse = self.esa_decoder(received)
        fine = self.tf_decoder(coarse)
        return (coarse, fine) if return_coarse else fine

    def forward(self, signal: torch.Tensor, snr_db: float = 10.0,
                return_coarse: bool = False, generator: torch.Generator | None = None):
        transmitted = self.encode(signal)
        received = self.channel(transmitted, snr_db=snr_db, generator=generator)
        return self.decode(received, return_coarse=return_coarse)


