<div align="center">


# Task-Aware Joint Source-Channel Coding for Robust Vibrotactile Transmission



[![Journal](https://img.shields.io/badge/IEEE-TMC%202026-blue.svg)](https://doi.org/10.1109/TMC.2026.3711711)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)

</div>

---

## 📖 Overview

**Tactile-JSCC** provides a training pipeline for 1D vibrotactile signal transmission using the LMT_108 dataset. Each run trains a model at a specified compression ratio over an additive white Gaussian noise (AWGN) channel and evaluates perceptual similarity using the **ST-SIM** metric on held-out recordings.

## 🛠️ Requirements

Use Python 3.10 or newer and install a compatible version of PyTorch for your system. GPU support is recommended for accelerated training.

Install the remaining dependencies with:

```bash
pip install -r requirements.txt
```

## 📂 Dataset

### Dataset Structure

Set `--data-root` to the root directory of the processed **LMT_108** dataset. The loader expects the following directory structure:

```text
LMT_108/
└── AccelScansComponents/
    └── Tapping/
        ├── Training/
        │   └── *.txt
        └── Testing/
            └── *.txt
```

### Data Processing

The data pipeline follows these steps:

1. **Signal selection:** Load tapping-Z recordings from both the `Training` and `Testing` directories.
2. **Recording-level split:** Create a seeded, material-stratified 70/30 train/test split by recording.
3. **Window generation:** Divide recordings into 256-sample windows, ignoring incomplete trailing samples.

The recording-level split is performed before window generation to keep windows from the same recording within the same split.

## 🚀 Training

### Single Compression Ratio

Run the following command to train a model with a compression ratio of 32:

```bash
python main_tactile_JSCC.py \
    --data-root /path/to/LMT_108 \
    --output-root runs/lmt108 \
    --compression-ratio 32
```

Replace `/path/to/LMT_108` with the actual path to your dataset.

### Default Configuration

| Parameter | Default |
|:---|:---|
| Training Epochs | 100 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate | `1e-5` |
| Learning Rate Scheduler | StepLR |
| Scheduler Step Size | 100 |
| Scheduler Decay Factor | 0.5 |
| Channel Model | AWGN |
| Channel SNR | 10 dB |
| Perceptual Loss Exponent | 1.75 |
| Signal Window Length | 256 samples |
| Evaluation Metric | ST-SIM |

The linear signal path is initialized using PCA fitted exclusively on training recordings.

### Training Window Policy

By default, training uses one randomly selected window per recording per epoch. To train on every nonoverlapping window instead, set the window policy to `grid`:

```bash
python main_tactile_JSCC.py \
    --data-root /path/to/LMT_108 \
    --output-root runs/lmt108 \
    --compression-ratio 32 \
    --train-window-policy grid
```

### Batch Training Across Compression Ratios

To run experiments with compression ratios of **2, 4, 8, 16, 32, and 48**, execute:

```bash
TACTILE_DATA_ROOT=/path/to/LMT_108 \
TACTILE_OUTPUT_ROOT=runs/lmt108 \
bash scripts/run_lmt108_ratios.sh
```

To use grid-window training for the batch experiments:

```bash
TACTILE_TRAIN_WINDOW_POLICY=grid \
TACTILE_DATA_ROOT=/path/to/LMT_108 \
TACTILE_OUTPUT_ROOT=runs/lmt108 \
bash scripts/run_lmt108_ratios.sh
```

Ensure the environment variables are set correctly before launching the script.

## 💾 Checkpoints and Outputs

Each run saves its configuration, data split, training history, checkpoint, and test results under the specified output directory.

```text
runs/lmt108/
└── cr_<ratio>/
    ├── config.json
    ├── split.json
    ├── history.jsonl
    ├── last.pt
    ├── best.pt        # Saved when validation is enabled
    └── test.json
```

| File | Description |
|:---|:---|
| `config.json` | Training configuration |
| `split.json` | Recording-level train/test split |
| `history.jsonl` | Training history |
| `last.pt` | Latest model checkpoint |
| `best.pt` | Best validation-loss checkpoint, when enabled |
| `test.json` | Test evaluation results |

### Resume Training

To resume an interrupted run, pass `--resume` with the same training settings:

```bash
python main_tactile_JSCC.py \
    --data-root /path/to/LMT_108 \
    --output-root runs/lmt108 \
    --compression-ratio 32 \
    --resume
```

When `--validation-fraction` is greater than zero, the best validation-loss checkpoint is also saved as `best.pt`.

## 📄 Citation

If you use this code in your research, please cite the corresponding paper:

> *Task-Aware Joint Source-Channel Coding for Robust Vibrotactile Transmission.* IEEE Transactions on Mobile Computing (TMC), 2026.

[Paper DOI: 10.1109/TMC.2026.3711711](https://doi.org/10.1109/TMC.2026.3711711)

