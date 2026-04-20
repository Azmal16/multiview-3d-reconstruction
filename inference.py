"""
inference.py — Web-ready 3D Reconstruction Module
===================================================
Wraps TripoSR into a framework-agnostic Reconstructor class.
Designed to be the model backend for a Flask / FastAPI web platform.

Usage
-----
    from inference import Reconstructor

    rec = Reconstructor(device="cuda")          # or "cpu"
    result = rec.reconstruct("photo.jpg")
    # result keys: mesh_path, voxels, n_occupied, inference_time_ms, preview_path

    # With a PIL image directly:
    from PIL import Image
    img = Image.open("photo.jpg")
    result = rec.reconstruct(img)
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Union

# ── TripoSR path bootstrap ────────────────────────────────────────────────────
# TripoSR is not a pip package — it's used as a cloned repo on sys.path.
_tsr_dir = Path(__file__).parent / "TripoSR"
if _tsr_dir.exists() and str(_tsr_dir.resolve()) not in sys.path:
    sys.path.insert(0, str(_tsr_dir.resolve()))

import numpy as np
import torch
from PIL import Image


# ── Lazy imports (only pulled in when Reconstructor is first used) ────────────

def _import_triposr():
    try:
        from tsr.system import TSR
        return TSR
    except ImportError as e:
        raise ImportError(
            "TripoSR is not installed. Run:\n"
            "  pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
        ) from e


def _import_rembg():
    try:
        from rembg import remove as rembg_remove
        return rembg_remove
    except ImportError as e:
        raise ImportError(
            "rembg is not installed. Run:\n"
            "  pip install rembg[gpu]"
        ) from e


def _import_trimesh():
    try:
        import trimesh
        return trimesh
    except ImportError as e:
        raise ImportError("trimesh is not installed. Run: pip install trimesh") from e


# ── Core Reconstructor ────────────────────────────────────────────────────────

class Reconstructor:
    """
    Single-image 3D reconstructor backed by TripoSR.

    Parameters
    ----------
    device : str
        'cuda', 'mps', or 'cpu'. Defaults to CUDA if available.
    voxel_resolution : int
        Side length of the output voxel grid (default 32 to match benchmark).
    mesh_resolution : int
        Marching-cubes resolution inside TripoSR (default 64; higher = finer mesh).
    output_dir : str | Path
        Where to save mesh and preview files. Defaults to './reconstruction_output'.
    chunk_size : int
        TripoSR renderer chunk size. Reduce if you run out of VRAM.
    """

    def __init__(
        self,
        device: str | None = None,
        voxel_resolution: int = 32,
        mesh_resolution: int = 64,
        output_dir: str | Path = "./reconstruction_output",
        chunk_size: int = 131072,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else \
                     "mps"  if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.voxel_resolution = voxel_resolution
        self.mesh_resolution = mesh_resolution
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._model = None          # loaded lazily on first call
        self._rembg = None
        self._trimesh = None
        self._chunk_size = chunk_size

    # ── Public API ────────────────────────────────────────────────────────────

    def reconstruct(
        self,
        image: Union[str, Path, "Image.Image"],
        *,
        remove_background: bool = True,
        save_mesh: bool = True,
        save_preview: bool = True,
        output_stem: str | None = None,
    ) -> dict:
        """
        Reconstruct a 3D object from a single image.

        Parameters
        ----------
        image : file path or PIL Image
        remove_background : bool
            Apply rembg background removal before reconstruction (recommended
            for real photos; set False for pre-cropped / synthetic images).
        save_mesh : bool
            Export the mesh as a .obj file in output_dir.
        save_preview : bool
            Save a matplotlib voxel preview PNG in output_dir.
        output_stem : str, optional
            Base filename (without extension) for saved files.
            Defaults to a timestamp string.

        Returns
        -------
        dict with keys:
            image_path    : str | None      — path to the input image (if file)
            mesh_path     : str | None      — path to the saved .obj mesh
            preview_path  : str | None      — path to the saved PNG preview
            voxels        : np.ndarray      — (R, R, R) float32 occupancy grid
            n_occupied    : int             — number of occupied voxels
            occupancy_pct : float           — % occupied voxels
            inference_time_ms : float       — wall-clock ms for TripoSR forward
            total_time_ms : float           — total wall-clock ms incl. voxelisation
        """
        t_total_start = time.perf_counter()
        stem = output_stem or f"recon_{int(time.time()*1000)}"

        # ── Load image ────────────────────────────────────────────────────────
        image_path = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil = Image.open(image).convert("RGB")
        else:
            pil = image.convert("RGB")

        # ── Pre-process ───────────────────────────────────────────────────────
        pil_ready = self._preprocess(pil, remove_background=remove_background)

        # ── TripoSR forward pass ──────────────────────────────────────────────
        model = self._load_model()
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            scene_codes = model([pil_ready], device=self.device)
            # Voxels: query density field directly (no rtree / trimesh.contains)
            voxels = self._density_to_voxels(model, scene_codes[0])
            # Mesh: only extracted when we need to save an .obj file
            mesh = None
            if save_mesh:
                meshes = model.extract_mesh(
                    scene_codes,
                    has_vertex_color=False,
                    resolution=self.mesh_resolution,
                )
                mesh = meshes[0]
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000

        # ── Save outputs ──────────────────────────────────────────────────────
        mesh_path = None
        if save_mesh and mesh is not None and len(mesh.vertices) > 0:
            mesh_path = str(self.output_dir / f"{stem}.obj")
            mesh.export(mesh_path)

        preview_path = None
        if save_preview:
            preview_path = self._save_preview(pil, pil_ready, voxels, stem)

        t_total_ms = (time.perf_counter() - t_total_start) * 1000
        n_occ = int(voxels.sum())
        R = self.voxel_resolution

        return {
            "image_path":        image_path,
            "mesh_path":         mesh_path,
            "preview_path":      preview_path,
            "voxels":            voxels,
            "n_occupied":        n_occ,
            "occupancy_pct":     100.0 * n_occ / (R ** 3),
            "inference_time_ms": t_infer_ms,
            "total_time_ms":     t_total_ms,
        }

    def batch_reconstruct(
        self,
        images: list,
        **kwargs,
    ) -> list[dict]:
        """Reconstruct multiple images sequentially. Returns list of result dicts."""
        return [self.reconstruct(img, **kwargs) for img in images]

    def warmup(self):
        """Run a dummy forward pass so the first real call isn't slow."""
        dummy = Image.new("RGB", (512, 512), color=(200, 200, 200))
        self.reconstruct(dummy, remove_background=False,
                         save_mesh=False, save_preview=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return self._model

        TSR = _import_triposr()
        self._model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model.renderer.set_chunk_size(self._chunk_size)
        self._model = self._model.to(self.device)
        self._model.eval()
        return self._model

    def _preprocess(self, pil: "Image.Image", *, remove_background: bool,
                    foreground_ratio: float = 0.85) -> "Image.Image":
        """
        Official TripoSR preprocessing:
          1. rembg background removal
          2. resize_foreground — crop + pad so object fills foreground_ratio of frame
          3. Composite on gray (0.5) — what TripoSR was trained with (NOT white)
        """
        from tsr.utils import remove_background as tsr_remove_bg, resize_foreground
        import rembg as _rembg

        if remove_background:
            session = _rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            pil = tsr_remove_bg(pil, rembg_session=session)    # RGBA
            pil = resize_foreground(pil, foreground_ratio)     # crop + pad
            arr   = np.array(pil).astype(np.float32) / 255.0
            rgb   = arr[:, :, :3]
            alpha = arr[:, :, 3:4]
            arr_rgb = rgb * alpha + 0.5 * (1 - alpha)          # gray fill
            pil = Image.fromarray((arr_rgb * 255).astype(np.uint8))
        else:
            pil = pil.convert("RGB")

        return pil.resize((512, 512), Image.LANCZOS)

    def _density_to_voxels(self, model, scene_code, threshold: float = 25.0) -> np.ndarray:
        """
        Query TripoSR's density field directly at a voxel_resolution³ grid.
        Avoids mesh extraction and trimesh.contains (no rtree dependency).
        """
        from tsr.utils import scale_tensor

        R = self.voxel_resolution
        lin = torch.linspace(0, 1, R)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        grid_verts = torch.stack([gx.ravel(), gy.ravel(), gz.ravel()], dim=1)

        radius = model.renderer.cfg.radius
        pts = scale_tensor(
            grid_verts.to(scene_code.device),
            (0, 1),
            (-radius, radius),
        )

        density = model.renderer.query_triplane(
            model.decoder,
            pts,
            scene_code,
        )["density_act"].squeeze(-1)   # (R³,)

        voxel = (density > threshold).reshape(R, R, R)
        return voxel.float().cpu().numpy()

    def _save_preview(
        self,
        raw_pil: "Image.Image",
        processed_pil: "Image.Image",
        voxels: np.ndarray,
        stem: str,
    ) -> str:
        import matplotlib
        matplotlib.use("Agg")                           # non-interactive backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D         # noqa: F401

        fig = plt.figure(figsize=(13, 4.5))

        ax0 = fig.add_subplot(1, 3, 1)
        ax0.imshow(raw_pil); ax0.set_title("Input"); ax0.axis("off")

        ax1 = fig.add_subplot(1, 3, 2)
        ax1.imshow(processed_pil); ax1.set_title("Processed"); ax1.axis("off")

        ax2 = fig.add_subplot(1, 3, 3, projection="3d")
        v = voxels > 0.5
        ax2.voxels(v, alpha=0.55, facecolors="#55a868", edgecolors="gray", linewidth=0.05)
        ax2.set_title(f"TripoSR ({self.voxel_resolution}³ voxels)")
        ax2.axis("off")

        plt.suptitle(stem, fontsize=12)
        plt.tight_layout()

        out = str(self.output_dir / f"{stem}_preview.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out


# ── CLI helper (quick test from terminal) ─────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [--cpu]")
        sys.exit(1)

    img_path = sys.argv[1]
    dev = "cpu" if "--cpu" in sys.argv else None

    print(f"Reconstructing: {img_path}")
    rec = Reconstructor(device=dev)
    result = rec.reconstruct(img_path)

    print(json.dumps({
        k: v for k, v in result.items() if k != "voxels"
    }, indent=2))
