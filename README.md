# Multi-View 3D Object Reconstruction Under Limited Visual Observations

**MM 805 — Final Project | University of Alberta, Winter 2026**
**Authors:** Md Fatinfaiaz Isty · Azmal Awasaf

---

## Overview

This project benchmarks three learning-based methods for reconstructing 3D objects from a limited number of 2D input views (1–5). We implement **3D-R2N2** and **Pix2Vox** in PyTorch, evaluate them on synthetic ShapeNet-style data across three object categories (airplane, car, chair), and add **TripoSR** as a modern zero-shot baseline for single-image reconstruction.

| Method | Year | Strategy | Parameters |
|--------|------|----------|------------|
| **3D-R2N2** | ECCV 2016 | Recurrent 3D-GRU — sequential view fusion | ~17.5 M |
| **Pix2Vox** | ICCV 2019 | Context-aware scoring — adaptive view weighting | ~12.7 M |
| **TripoSR** | 2024 | Triplane NeRF — zero-shot from a single image | ~1 GB weights |

---

## Pipeline

```
Input Images (1–5 views)
        │
        ▼
  2D Encoder (CNN)
        │
        ▼
3D Reconstruction (R2N2 GRU  /  Pix2Vox fusion  /  TripoSR triplane)
        │
        ▼
  32³ Voxel Grid  ──►  IoU / F-score vs. Ground Truth
                  ──►  .obj Mesh (TripoSR)
                  ──►  Voxel preview PNG
```

---

## Repository Structure

```
multiview-3d-reconstruction/
├── 3d_reconstruction.ipynb   # Main notebook (Sections 1–10)
├── inference.py              # Web-ready Reconstructor class (Flask/FastAPI backend)
├── requirements.txt          # All dependencies
├── presentation.html         # 8-slide HTML slideshow
├── results/                  # Saved plots and metrics
├── reconstruction_output/    # TripoSR .obj meshes + preview PNGs
└── TripoSR/                  # Cloned TripoSR repo (not pip-installable)
    └── torchmcubes.py        # scikit-image shim (replaces C++ torchmcubes ext)
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** tested with PyTorch 2.5.1+cu121. For CPU-only, replace `rembg[gpu]` with `rembg` and remove the `onnxruntime-gpu` line.

### 2. Clone TripoSR

TripoSR is not a pip package — clone it into the project root:

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
```

The notebook and `inference.py` automatically add `TripoSR/` to `sys.path`.

### 3. torchmcubes shim

TripoSR normally requires a compiled CUDA C++ extension (`torchmcubes`). If compilation fails (common on shared servers), the repo includes a pure-Python fallback at `TripoSR/torchmcubes.py` that wraps `scikit-image`'s marching cubes — no action needed, it's already in place.

### 4. Run the notebook

```bash
jupyter notebook 3d_reconstruction.ipynb
```

Run all cells in order:

| Section | Content |
|---------|---------|
| 1–3 | Dataset generation, model definitions (R2N2 / Pix2Vox) |
| 4–6 | Training, evaluation, per-category analysis |
| 7 | TripoSR setup and model loading |
| 8 | TripoSR benchmark on synthetic data |
| 9 | Cross-method performance comparison |
| 10 | Real-image demo + GPU memory cleanup |

Training (R2N2 + Pix2Vox) takes ~30–40 minutes on CPU; checkpoints are saved so re-runs skip training. TripoSR weights (~1 GB) are downloaded automatically from HuggingFace on first run.

---

## Key Features

- Full PyTorch implementations of 3D-R2N2 and Pix2Vox
- Synthetic data generator (procedural airplane/car/chair shapes with multi-view rendering) — runs out-of-the-box, no downloads needed
- TripoSR integration as a zero-shot single-image baseline
- Direct density-field voxelisation (avoids `rtree`/`trimesh.contains` dependency)
- Evaluation with varying input views (1 → 5) measuring IoU and F-score
- Per-category failure mode analysis and heatmaps
- Adaptive method selection: routes each (category, view count) to the best-performing model

---

## Web Inference Module

[inference.py](inference.py) provides a framework-agnostic `Reconstructor` class ready to serve as the backend for a Flask or FastAPI web platform:

```python
from inference import Reconstructor

rec = Reconstructor(device="cuda")   # or "cpu"
result = rec.reconstruct("photo.jpg")

# result keys:
#   mesh_path, preview_path, voxels, n_occupied,
#   occupancy_pct, inference_time_ms, total_time_ms
```

Features:
- Auto device detection (CUDA → MPS → CPU)
- Background removal via rembg (forced to CPU to avoid GPU OOM)
- Official TripoSR preprocessing: foreground resize + gray (0.5) background composite
- Lazy model loading — model stays in memory across calls
- `batch_reconstruct()` and `warmup()` helpers

---

## Results

The notebook produces:
- IoU and F-score vs. number of input views (1–5) for all three categories
- Per-category performance heatmaps
- Qualitative 3D voxel visualisations (R2N2 vs Pix2Vox vs TripoSR)
- Adaptive method selection analysis

### TripoSR demo parameters

| Parameter | Value |
|-----------|-------|
| Input resolution | 512 × 512 |
| Voxel grid | 32³ |
| Mesh resolution (marching cubes) | 64 |
| Density threshold | 25.0 |
| Chunk size | 131 072 |
| Background removal | rembg U2Net (CPU) |
| Foreground ratio | 0.85 |

---

## Presentation

Open [presentation.html](presentation.html) in any browser for the 8-slide project presentation. Navigate with arrow keys or on-screen buttons; press `F` for fullscreen.

---

## References

1. C. B. Choy et al., "3D-R2N2: A Unified Approach for Single and Multi-View 3D Object Reconstruction," ECCV 2016
2. H. Xie et al., "Pix2Vox: Context-Aware 3D Reconstruction from Single and Multi-View Images," ICCV 2019
3. Z. Zou et al., "TripoSR: Fast 3D Object Reconstruction from a Single Image," arXiv 2024
