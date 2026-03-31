# Multi-View 3D Object Reconstruction Under Limited Visual Observations

This project benchmarks two learning-based methods for reconstructing 3D objects from a limited number of 2D input views (1–5). We implement **3D-R2N2** and **Pix2Vox** in PyTorch, evaluate them on the ShapeNet dataset across three object categories (airplane, car, chair), and propose an adaptive method selection strategy.

## Overview

Reconstructing 3D geometry from sparse 2D observations is inherently ill-posed due to depth ambiguity and occlusions. This project systematically analyses how reconstruction quality evolves as the number of input views increases, comparing two multi-view fusion strategies:

| Method | Fusion Strategy | Key Idea |
|--------|----------------|----------|
| **3D-R2N2** | Recurrent (3D-GRU) | Sequentially refines a voxel grid as each view arrives |
| **Pix2Vox** | Context-aware | Adaptively weights each view's contribution via a scoring network |

## Pipeline

```
Input Images (1–5 views) → 2D Encoder → 3D Reconstruction → 32³ Voxel Grid
                                                ↓
                              Evaluate IoU / F-score vs. Ground Truth
```

## Key Features

- Full PyTorch implementations of 3D-R2N2 and Pix2Vox
- Synthetic data generator (procedural airplane/car/chair shapes with multi-view rendering) — runs out-of-the-box, no downloads needed
- Also supports real ShapeNet rendered images + voxel ground truth
- Evaluation with varying input views (1 → 5) measuring IoU and F-score
- Per-category failure mode analysis
- Adaptive method selection: routes each (category, view count) to the best-performing model

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 3d_reconstruction.ipynb
```

Run all cells. Training takes ~30–40 minutes on CPU (checkpoints are saved so re-runs skip training). GPU/MPS is detected and used automatically if available.

## Results

The notebook produces:
- IoU and F-score vs. number of views plots
- Per-category performance breakdown and heatmaps
- Qualitative 3D voxel visualisations comparing both methods against ground truth
- Adaptive selection analysis showing it outperforms any single method

## References

1. C. B. Choy et al., "3D-R2N2: A Unified Approach for Single and Multi-View 3D Object Reconstruction," ECCV 2016
2. H. Xie et al., "Pix2Vox: Context-Aware 3D Reconstruction from Single and Multi-View Images," ICCV 2019
