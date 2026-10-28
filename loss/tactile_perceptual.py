"""Perceptual tactile reconstruction loss."""

from __future__ import annotations

import torch
from torch import nn


class TactilePerceptualLoss(nn.Module):
    """Mean ((x - x_hat)^2 / (1 + (I + eps)^beta)), I=(|x|+|x_hat|)/2."""

    def __init__(self, beta: float = 1.75, eps: float = 1e-6,
                 reduction: str = "mean") -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError("beta must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean, sum, or none")
        self.beta = beta
        self.eps = eps
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if prediction.shape != reference.shape:
            raise ValueError("prediction and reference must have matching shapes")
        intensity = (prediction.abs() + reference.abs()) / 2
        distortion = (prediction - reference).square() / (1 + (intensity + self.eps).pow(self.beta))
        if self.reduction == "mean":
            return distortion.mean()
        if self.reduction == "sum":
            return distortion.sum()
        return distortion
