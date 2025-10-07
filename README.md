# Smart CIFAR-10: Dynamic Weighted-Branch CNN

A compact, reproducible PyTorch project for CIFAR-10 image classification using **parallel convolutional branches** combined by **input-dependent softmax weights** derived from per-channel means. Trains end-to-end on a single GPU (e.g., Colab T4) with standard data augmentation.

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](#)
[![Open in Colab](https://img.shields.io/badge/Colab-Open-yellow.svg)](https://colab.research.google.com/github/<your-username>/<your-repo>/blob/main/notebooks/cifar10.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](#)

---

## Highlights
- **Dynamic fusion:** Multiple conv branches; **softmax weights** predicted from the input’s channel statistics (per-channel means).
- **Lightweight & fast:** Minimal code; runs smoothly on a single GPU (e.g., Colab T4).
- **Reproducible:** Fixed seeds, logged metrics, one-command training.
- **Clear results:** Loss-by-iteration and accuracy-by-epoch plots; best test accuracy reported.

---

## Project structure

---

## Results (example)
| Metric | Value |
|---|---|
| Best Test Accuracy | **88.2%** |


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
