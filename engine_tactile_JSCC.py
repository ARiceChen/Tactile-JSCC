"""Training and ST-SIM scoring for tactile JSCC."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from vibromaf.metrics.stsim import st_sim


def train_one_epoch(loader, model: nn.Module, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device,
                    snr_db: float = 10.0, gradient_clip: float = 1.0) -> dict:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch_index, signal in enumerate(loader):
        signal = signal.to(device, non_blocking=True)
        reconstructed = model(signal, snr_db=snr_db)
        loss = criterion(reconstructed, signal)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(signal)
        sample_count += len(signal)
    if not sample_count:
        raise ValueError("empty training loader")
    return {"loss": loss_sum / sample_count, "samples": sample_count}


@torch.no_grad()
def evaluate_loss(loader, model: nn.Module, criterion: nn.Module,
                  device: torch.device, snr_db: float = 10.0,
                  eval_seed: int = 2026) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(eval_seed)
    loss_sum = 0.0
    sample_count = 0
    for signal, _, _ in loader:
        signal = signal.to(device, non_blocking=True)
        reconstructed = model(signal, snr_db=snr_db, generator=generator)
        loss_sum += float(criterion(reconstructed, signal)) * len(signal)
        sample_count += len(signal)
    if not sample_count:
        raise ValueError("empty validation loader")
    return loss_sum / sample_count


@torch.no_grad()
def evaluate_stsim(loader, model: nn.Module, device: torch.device,
                    snr_db: float = 10.0, eval_seed: int = 2026) -> dict:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(eval_seed)
    segments = defaultdict(list)
    sample_count = 0
    for signal, recording_indices, starts in loader:
        signal = signal.to(device, non_blocking=True)
        reconstructed = model(signal, snr_db=snr_db, generator=generator)
        for original, estimate, index, start in zip(
            signal.cpu().numpy(), reconstructed.cpu().numpy(),
            recording_indices.tolist(), starts.tolist()
        ):
            segments[index].append((start, original, estimate))
        sample_count += len(signal)
    if not sample_count:
        raise ValueError("empty test loader")

    scores = []
    for recording in segments.values():
        recording.sort(key=lambda item: item[0])
        original = np.concatenate([item[1] for item in recording])
        estimate = np.concatenate([item[2] for item in recording])
        if len(original) < 512:
            continue
        score = st_sim(estimate, original)
        score = abs(score) if isinstance(score, complex) else float(score)
        if math.isfinite(score):
            scores.append(score)
    return {"st_sim": float(np.mean(scores)) if scores else None,
            "recordings_scored": len(scores), "samples": sample_count}
