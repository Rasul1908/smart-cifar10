# Smart CIFAR-10: Dynamic Weighted-Branch CNN

A compact, reproducible PyTorch project for CIFAR-10 image classification using **parallel convolutional branches** combined by **learned, input-dependent weights**. Trains end-to-end on a single GPU (e.g., Colab T4) and reaches strong accuracy with a small footprint.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#)
[![Open in Colab](https://img.shields.io/badge/Colab-Open-yellow.svg)](#)


---

## Highlights
- **Dynamic fusion:** multiple conv branches, **softmax weights** predicted from per-channel means of the input.
- **Simple & fast:** runs in Colab; minimal dependencies.
- **Reproducible:** fixed seeds, CLI config, saved metrics/plots.
- **Clear results:** loss-by-iteration and accuracy-by-epoch plots; best test accuracy logged.

---

## Results (example)
| Metric | Value |
|---|---|
| Best Test Accuracy | **88.2%** |
| Epoch (best) | 26 |
| Params | ~X.XX M |
| Hardware | Colab T4 |

> Replace with your exact numbers once you run. Plots saved in `reports/figs/`.

---

## Quickstart

### Option A — Notebook (Colab friendly)
1. Open `notebooks/cifar10.ipynb`.
2. Run all cells (dataset auto-downloads via `torchvision`).

### Option B — CLI (local)
```bash
# 1) Create env
conda env create -f environment.yml
conda activate smart-cifar10

# 2) Train
python -m src.train \
  --epochs 30 \
  --batch-size 128 \
  --lr 0.1 \
  --optimizer sgd \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --scheduler cosine \
  --seed 42 \
  --outdir runs/exp1

# 3) Evaluate (best checkpoint)
python -m src.eval --ckpt runs/exp1/best.ckpt
