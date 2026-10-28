# Tactile-JSCC

Training code for 1D tactile signal transmission on LMT_108. Each run trains
one compression ratio with an AWGN channel and reports ST-SIM on a held-out
recording split.

## Install

Use Python 3.10 or newer and install a PyTorch build for your system, then:

```bash
pip install -r requirements.txt
```

## Data

Point `--data-root` to the LMT_108 directory containing:

```text
AccelScansComponents/Tapping/Training/*.txt
AccelScansComponents/Tapping/Testing/*.txt
```

The loader uses tapping-Z recordings from both folders. It creates a seeded,
material-stratified 70/30 train/test split by recording before forming
256-sample windows. Incomplete trailing samples are ignored.

## Train

```bash
python main_tactile_JSCC.py \
  --data-root /path/to/LMT_108 \
  --output-root runs/lmt108 \
  --compression-ratio 32
```

Defaults: 100 epochs, batch size 8, AdamW learning rate `1e-5`, StepLR step
100 with factor 0.5, AWGN at 10 dB, and perceptual loss exponent 1.75. The
linear signal path is initialized with PCA fitted only to training recordings.
Use `--train-window-policy grid` to train on every nonoverlapping window rather
than one random window per recording and epoch.

To train compression ratios 2, 4, 8, 16, 32, and 48:

```bash
TACTILE_DATA_ROOT=/path/to/LMT_108 \
TACTILE_OUTPUT_ROOT=runs/lmt108 \
bash scripts/run_lmt108_ratios.sh
```

Set `TACTILE_TRAIN_WINDOW_POLICY=grid` for grid-window training. Pass
`--resume` to continue an interrupted run with the same settings. Each run
writes `config.json`, `split.json`, `history.jsonl`, `last.pt`, and `test.json`
under `--output-root/cr_<ratio>/`. With `--validation-fraction` greater than
zero, the best validation-loss checkpoint is also saved as `best.pt`.
