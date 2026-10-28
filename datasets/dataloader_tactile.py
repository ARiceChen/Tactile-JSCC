"""LMT_108 vibrotactile recordings with recording-level stratified splitting."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


BLOCK_LENGTH = 256


@dataclass(frozen=True)
class Recording:
    path: Path
    material: str


def discover_lmt108_recordings(data_root: str | Path) -> list[Recording]:
    """Read tapping Z recordings from the local LMT_108 directory tree."""
    root = Path(data_root).expanduser().resolve()
    tapping = root / "AccelScansComponents" / "Tapping"
    if root.name == "Tapping":
        tapping = root
    if not tapping.is_dir():
        raise FileNotFoundError(f"LMT_108 Tapping directory not found under {root}")
    files = sorted((*tapping.glob("Training/*.txt"), *tapping.glob("Testing/*.txt")))
    if not files:
        raise FileNotFoundError(f"no LMT_108 tapping .txt recordings found under {tapping}")
    recordings = []
    for path in files:
        if "_Tapping_" not in path.stem:
            raise ValueError(f"cannot read material name from {path.name}")
        recordings.append(Recording(path=path, material=path.stem.split("_Tapping_", 1)[0]))
    return recordings


def stratified_recording_split(recordings: list[Recording], seed: int = 42,
                               test_fraction: float = 0.30,
                               validation_fraction: float = 0.0) -> dict[str, list[Recording]]:
    """Split complete recordings before any 256-sample windows are generated."""
    if not 0 < test_fraction < 1 or not 0 <= validation_fraction < 1:
        raise ValueError("fractions must be in [0, 1], with a positive test fraction")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must sum to less than 1")
    grouped: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        grouped[recording.material].append(recording)
    randomizer = random.Random(seed)
    split: dict[str, list[Recording]] = {"train": [], "val": [], "test": []}
    for material in sorted(grouped):
        members = sorted(grouped[material], key=lambda item: item.path.as_posix())
        randomizer.shuffle(members)
        test_count = max(1, round(len(members) * test_fraction))
        val_count = round(len(members) * validation_fraction)
        if len(members) - test_count - val_count < 1:
            raise ValueError(f"material {material} has too few recordings for this split")
        split["test"].extend(members[:test_count])
        split["val"].extend(members[test_count:test_count + val_count])
        split["train"].extend(members[test_count + val_count:])
    for name in split:
        split[name].sort(key=lambda item: item.path.as_posix())
    return split


class LMT108Dataset(Dataset):
    """Load 1D recordings once; sample training windows or enumerate test blocks."""

    def __init__(self, recordings: list[Recording], training: bool,
                 windows_per_recording: int = 1, eval_stride: int = BLOCK_LENGTH,
                 train_window_policy: str = "random") -> None:
        if not recordings:
            raise ValueError("recordings cannot be empty")
        if windows_per_recording < 1 or eval_stride < 1:
            raise ValueError("windows_per_recording and eval_stride must be positive")
        if train_window_policy not in ("random", "grid"):
            raise ValueError("train_window_policy must be 'random' or 'grid'")
        self.recordings = tuple(recordings)
        self.training = training
        self.windows_per_recording = windows_per_recording
        self.train_window_policy = train_window_policy
        self.signals: list[torch.Tensor] = []
        self.eval_windows: list[tuple[int, int]] = []
        for index, recording in enumerate(self.recordings):
            values = np.loadtxt(recording.path, dtype=np.float32)
            if values.ndim != 1 or len(values) < BLOCK_LENGTH or not np.isfinite(values).all():
                raise ValueError(f"invalid 1D tactile signal in {recording.path}")
            signal = torch.from_numpy(values)
            self.signals.append(signal)
            if not training or train_window_policy == "grid":
                for start in range(0, len(signal) - BLOCK_LENGTH + 1, eval_stride):
                    self.eval_windows.append((index, start))

    def __len__(self) -> int:
        if self.training and self.train_window_policy == "random":
            return len(self.recordings) * self.windows_per_recording
        return len(self.eval_windows)

    def __getitem__(self, index: int):
        if self.training and self.train_window_policy == "random":
            recording_index = index // self.windows_per_recording
            signal = self.signals[recording_index]
            start = int(torch.randint(len(signal) - BLOCK_LENGTH + 1, (1,)))
        else:
            recording_index, start = self.eval_windows[index]
            signal = self.signals[recording_index]
            if not self.training:
                return signal[start:start + BLOCK_LENGTH], recording_index, start
        return signal[start:start + BLOCK_LENGTH]
