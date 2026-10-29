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



## 📄 Citation

If you use this code in your research, please cite the corresponding paper:

> *Task-Aware Joint Source-Channel Coding for Robust Vibrotactile Transmission.* IEEE Transactions on Mobile Computing (TMC), 2026.

[Paper DOI: 10.1109/TMC.2026.3711711](https://doi.org/10.1109/TMC.2026.3711711)

