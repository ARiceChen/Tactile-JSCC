"""Train Tactile-JSCC on LMT_108 recordings."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataloader_tactile import (
    LMT108Dataset, Recording, discover_lmt108_recordings, stratified_recording_split,
)
from engine_tactile_JSCC import evaluate_loss, evaluate_stsim, train_one_epoch
from loss.tactile_perceptual import TactilePerceptualLoss
from models.tactile_JSCC import BLOCK_LENGTH, TactileJSCC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tactile-JSCC on LMT_108")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/tactile_jscc"))
    parser.add_argument("--compression-ratio", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--step-size", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.75)
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--train-window-policy", choices=("random", "grid"), default="random")
    parser.add_argument("--train-windows-per-recording", type=int, default=1)
    parser.add_argument("--eval-stride", type=int, default=BLOCK_LENGTH)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--esa-layers", type=int, default=1)
    parser.add_argument("--projected-length", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def make_loader(recordings: list[Recording], training: bool, args: argparse.Namespace,
                generator: torch.Generator | None = None) -> DataLoader:
    dataset = LMT108Dataset(
        recordings, training=training,
        windows_per_recording=args.train_windows_per_recording if training else 1,
        eval_stride=args.eval_stride, train_window_policy=args.train_window_policy,
    )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                      generator=generator, num_workers=args.num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=args.num_workers > 0)


def build_model(args: argparse.Namespace, device: torch.device) -> TactileJSCC:
    model = TactileJSCC(
        compression_ratio=args.compression_ratio, width=args.width,
        esa_layers=args.esa_layers, projected_length=args.projected_length,
    ).to(device)
    count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={count:,} real_symbols={model.symbol_count} "
          f"effective_cr={BLOCK_LENGTH / model.symbol_count:.3f}")
    return model


def initialize_pca(model: TactileJSCC, dataset: LMT108Dataset) -> None:
    windows = torch.stack([
        signal[start:start + BLOCK_LENGTH]
        for signal in dataset.signals
        for start in range(0, len(signal) - BLOCK_LENGTH + 1, BLOCK_LENGTH)
    ]).double()
    mean = windows.mean(dim=0)
    covariance = (windows - mean).T @ (windows - mean) / (len(windows) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -model.symbol_count:].T.flip(0).float()
    largest = basis.abs().argmax(dim=1)
    basis *= torch.sign(basis[torch.arange(len(basis)), largest]).unsqueeze(1)
    mean = mean.float()
    with torch.no_grad():
        model.encoder.signal_skip.weight.copy_(basis.to(model.encoder.signal_skip.weight.device))
        model.coarse_decoder.expansion.weight.copy_(basis.T.to(model.coarse_decoder.expansion.weight.device))
        residual_mean = mean - (mean @ basis.T) @ basis
        model.coarse_decoder.expansion.bias.copy_(residual_mean.to(model.coarse_decoder.expansion.bias.device))
    variance = float(eigenvalues[-model.symbol_count:].sum() / eigenvalues.sum())
    print(f"PCA initialization: {len(windows)} windows, {variance:.3%} variance")


def run(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    run_dir = args.output_root / f"cr_{args.compression_ratio}"
    checkpoint_path = run_dir / "last.pt"
    if checkpoint_path.exists() and not args.resume:
        raise FileExistsError(f"{checkpoint_path} exists; use --resume or another output directory")
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        saved = json.loads((run_dir / "config.json").read_text())
        for key in ("compression_ratio", "batch_size", "learning_rate", "step_size",
                    "beta", "snr_db", "seed", "eval_seed", "validation_fraction",
                    "train_window_policy", "train_windows_per_recording", "eval_stride",
                    "width", "esa_layers", "projected_length"):
            if getattr(args, key) != saved[key]:
                raise ValueError(f"cannot resume with changed {key}")
        split_data = json.loads((run_dir / "split.json").read_text())["splits"]
        split = {name: [Recording(Path(item["path"]), item["material"]) for item in entries]
                 for name, entries in split_data.items()}
    else:
        split = stratified_recording_split(
            discover_lmt108_recordings(args.data_root), seed=args.seed,
            test_fraction=0.30, validation_fraction=args.validation_fraction,
        )
        write_json(run_dir / "split.json", {
            "splits": {name: [{"path": str(item.path), "material": item.material}
                              for item in entries] for name, entries in split.items()}
        })
        write_json(run_dir / "config.json", {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        })

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(split["train"], True, args, generator)
    val_loader = make_loader(split["val"], False, args) if split["val"] else None
    model = build_model(args, device)
    if not args.resume:
        initialize_pca(model, train_loader.dataset)
    criterion = TactilePerceptualLoss(beta=args.beta)
    linear = list(model.encoder.signal_skip.parameters()) + list(model.coarse_decoder.expansion.parameters())
    linear_ids = {id(parameter) for parameter in linear}
    other = [parameter for parameter in model.parameters() if id(parameter) not in linear_ids]
    optimizer = torch.optim.AdamW([
        {"params": other, "lr": args.learning_rate},
        {"params": linear, "lr": args.learning_rate * 0.1},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["best_val_loss"])
        generator.set_state(checkpoint["loader_rng"].cpu())
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        train = train_one_epoch(train_loader, model, optimizer, criterion, device, args.snr_db)
        scheduler.step()
        val_loss = evaluate_loss(val_loader, model, criterion, device, args.snr_db,
                                 args.eval_seed) if val_loader else None
        with (run_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps({"epoch": epoch + 1, "train": train,
                                     "val_loss": val_loss,
                                     "learning_rate": optimizer.param_groups[0]["lr"],
                                     "elapsed_seconds": time.perf_counter() - started}) + "\n")
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "val_loss": val_loss}, run_dir / "best.pt")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
            "best_val_loss": best_val_loss, "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, checkpoint_path)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={train['loss']:.6g} "
              + (f"val_loss={val_loss:.6g} " if val_loss is not None else "")
              + f"time={time.perf_counter() - started:.1f}s", flush=True)

    selected = run_dir / "best.pt" if val_loader else checkpoint_path
    model.load_state_dict(torch.load(selected, map_location=device, weights_only=True)["model"])
    test = evaluate_stsim(make_loader(split["test"], False, args), model, device,
                           args.snr_db, args.eval_seed)
    write_json(run_dir / "test.json", {"checkpoint": selected.name, **test})
    print(f"test ST-SIM: {test['st_sim']}")


if __name__ == "__main__":
    run(parse_args())
