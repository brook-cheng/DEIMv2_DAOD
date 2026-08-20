#!/usr/bin/env python3
"""Stage 1: Full synthetic ellipse OBB dataset generation.

Generates 3500 images (7 densities × 500 images each: 400 train + 100 val)
with DOTA-format annotations. Uses multiprocessing for speed.

Output structure:
  synthetic_ellipse/
    classes.txt
    density_001/train/  (000000.png ... 000399.png + .txt)
    density_001/val/    (000000.png ... 000099.png + .txt)
    ...
"""

import os
import sys
import time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

# Shapely may raise if GEOS is missing; surface early
try:
    from shapely.geometry import Polygon  # type: ignore[import-untyped]
except ImportError:
    sys.exit("ERROR: shapely is required. Install with: pip install shapely")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_ROOT = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse"
IMG_SIZE = 256

DENSITIES = [1, 2, 5, 10, 20, 50, 100]
TRAIN_COUNT = 400
VAL_COUNT = 100
TOTAL_PER_DENSITY = TRAIN_COUNT + VAL_COUNT

NUM_WORKERS = min(8, os.cpu_count() or 4)

MIN_ASPECT_RATIO = 1.5        # b/a
THETA_RANGE = (0.0, np.pi)    # radians

# Background gray level range
GRAY_RANGE = (50, 200)        # v ∈ [50, 200]

# ---------------------------------------------------------------------------
# Density-tiered generation parameters
# ---------------------------------------------------------------------------


def _params(density: int) -> dict:
    """Return (short_axis_range, long_axis_range, iou_threshold, max_restarts)."""
    if density <= 20:
        return {
            "short_range": (20, 60),
            "long_range": (40, 100),
            "iou_max": 0.05,
            "restarts": 15,
            "attempts": 500,
        }
    if density == 50:
        return {
            "short_range": (12, 36),
            "long_range": (24, 60),
            "iou_max": 0.10,
            "restarts": 40,
            "attempts": 500,
        }
    return {
        "short_range": (10, 24),
        "long_range": (20, 48),
        "iou_max": 0.15,
        "restarts": 60,
        "attempts": 500,
    }


# Per-category RGB base ranges (before ±5 pixel noise)
CATEGORY_COLORS_RGB: dict[str, dict[str, tuple[int, int]]] = {
    "r": {"R": (170, 230), "G": (10, 30), "B": (10, 30)},
    "g": {"R": (10, 30), "G": (170, 230), "B": (10, 30)},
    "b": {"R": (10, 30), "G": (10, 30), "B": (170, 230)},
}

CATEGORY_CYCLE = ["r", "g", "b"]

# ---------------------------------------------------------------------------
# Helpers (identical logic to generate_synthetic_ellipse_samples.py)
# ---------------------------------------------------------------------------


def random_gray_background(rng: np.random.Generator) -> tuple[int, int, int]:
    """Return an RGB gray triple (v, v, v) with v ∈ [50, 200]."""
    v = int(rng.integers(*GRAY_RANGE))
    return (v, v, v)


def random_color(rng: np.random.Generator, cat: str) -> tuple[int, int, int]:
    """Return an (R, G, B) tuple for *cat* with ±5 per-channel noise."""
    spec = CATEGORY_COLORS_RGB[cat]
    rgb: list[int] = []
    for ch in ("R", "G", "B"):
        lo, hi = spec[ch]
        val = int(rng.integers(lo, hi + 1))
        val += int(rng.integers(-5, 7))
        val = max(0, min(255, val))
        rgb.append(val)
    return (rgb[0], rgb[1], rgb[2])  # (R, G, B)


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert (R, G, B) → (B, G, R) for OpenCV drawing."""
    return (rgb[2], rgb[1], rgb[0])


def compute_obb_vertices(
    cx: float, cy: float, a: float, b: float, theta: float,
) -> np.ndarray:
    """Compute the 4 vertices of the ellipse's OBB (clockwise order)."""
    corners_local = np.array([
        [-a, -b],
        [a, -b],
        [a, b],
        [-a, b],
    ], dtype=np.float64)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                   [sin_t, cos_t]], dtype=np.float64)

    vertices = corners_local @ R.T + np.array([cx, cy], dtype=np.float64)
    return vertices


def is_contained(vertices: np.ndarray) -> bool:
    """Return True if all 4 vertices lie inside [0, IMG_SIZE-1] inclusive."""
    max_coord = float(IMG_SIZE - 1)
    return bool(np.all((vertices >= 0.0) & (vertices <= max_coord)))


def obb_to_polygon(vertices: np.ndarray) -> Polygon:
    """Build a shapely Polygon from (N, 2) vertex array."""
    return Polygon(vertices.tolist())


def passes_collision(
    vertices: np.ndarray,
    existing_polys: list[Polygon],
    threshold: float,
) -> bool:
    """Return True if the new OBB does *not* excessively overlap any existing one."""
    new_poly = Polygon(vertices.tolist())
    if not new_poly.is_valid:
        new_poly = new_poly.buffer(0)

    for ep in existing_polys:
        if not new_poly.intersects(ep):
            continue
        inter = new_poly.intersection(ep)
        if inter.is_empty:
            continue
        union = new_poly.union(ep)
        if union.area <= 0.0:
            continue
        iou = inter.area / union.area
        if iou >= threshold:
            return False
    return True


def category_sequence(density: int) -> list[str]:
    """Return the per-ellipse category assignment list."""
    if density == 1:
        return ["r"]
    return [CATEGORY_CYCLE[i % 3] for i in range(density)]


# ---------------------------------------------------------------------------
# Image generation (per-image, self-contained for multiprocessing)
# ---------------------------------------------------------------------------


def generate_image(
    rng: np.random.Generator,
    density: int,
    params: dict,
) -> tuple[np.ndarray | None, list[dict] | None]:
    """Try to create one synthetic image with *density* non-overlapping ellipses.

    Returns:
        (img_bgr, annotations) on success, (None, None) on failure.
    """
    gray = random_gray_background(rng)
    gray_bgr = (gray[2], gray[1], gray[0])
    img = np.full((IMG_SIZE, IMG_SIZE, 3), gray_bgr, dtype=np.uint8)

    short_range = params["short_range"]
    long_range = params["long_range"]
    iou_max = params["iou_max"]
    max_attempts = params["attempts"]

    cats = category_sequence(density)
    annotations: list[dict] = []
    existing_polys: list[Polygon] = []

    for ell_idx in range(density):
        cat = cats[ell_idx]
        placed = False

        for _ in range(max_attempts):
            axis_2a = float(rng.uniform(*short_range))
            axis_2b = float(rng.uniform(*long_range))
            a = axis_2a / 2.0
            b = axis_2b / 2.0

            if b / a < MIN_ASPECT_RATIO:
                continue

            theta = float(rng.uniform(*THETA_RANGE))
            cx = float(rng.uniform(0.0, float(IMG_SIZE)))
            cy = float(rng.uniform(0.0, float(IMG_SIZE)))

            vertices = compute_obb_vertices(cx, cy, a, b, theta)

            # Round to 1 decimal for consistency between collision-checking
            # and stored DOTA annotations (avoids floating-point boundary IoU violations).
            vertices_rounded = np.round(vertices, 1)

            if not is_contained(vertices_rounded):
                continue
            if not passes_collision(vertices_rounded, existing_polys, threshold=iou_max):
                continue

            # --- accepted ---
            placed = True
            rgb = random_color(rng, cat)
            bgr = rgb_to_bgr(rgb)

            verts_flat: list[float] = []
            for v in vertices_rounded:
                verts_flat.append(float(v[0]))
                verts_flat.append(float(v[1]))

            annotations.append({
                "cat": cat,
                "cx": cx,
                "cy": cy,
                "a": a,
                "b": b,
                "theta": theta,
                "rgb": rgb,
                "bgr": bgr,
                "vertices": vertices_rounded,
                "verts_flat": verts_flat,
            })

            existing_polys.append(obb_to_polygon(vertices_rounded))
            break

        if not placed:
            return None, None

    # --- draw all ellipses ---
    for ann in annotations:
        center = (int(round(ann["cx"])), int(round(ann["cy"])))
        axes = (int(round(ann["a"])), int(round(ann["b"])))
        angle_deg = float(np.degrees(ann["theta"]))
        cv2.ellipse(
            img, center, axes, angle_deg,
            0.0, 360.0, ann["bgr"], thickness=-1,
        )

    return img, annotations


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_annotation_txt(filepath: str, annotations: list[dict]) -> None:
    """Write a single DOTA-format annotation file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for ann in annotations:
            verts_str = " ".join(f"{v:.1f}" for v in ann["verts_flat"])
            f.write(f"{verts_str} {ann['cat']} 0\n")


# ---------------------------------------------------------------------------
# Worker for multiprocessing
# ---------------------------------------------------------------------------


def worker_generate(args: tuple[int, int, dict]) -> tuple[int, np.ndarray | None, list[dict] | None, int]:
    """Generate a single image in a worker process.

    Args:
        args: (seed, density, params)

    Returns:
        (idx, img_bgr, annotations, restarts_used)
    """
    idx, density, params = args
    seed = (int(time.time() * 1_000_000) + idx * 7919 + density * 6271) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)

    max_restarts = params["restarts"]
    for restart in range(max_restarts):
        img, anns = generate_image(rng, density, params)
        if img is not None and anns is not None:
            return (idx, img, anns, restart + 1)
    return (idx, None, None, max_restarts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Write classes.txt
    classes_path = os.path.join(OUTPUT_ROOT, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("r\ng\nb\n")
    print(f"[classes.txt] written: r g b\n")

    t_start = time.time()

    for density in DENSITIES:
        params = _params(density)
        dens_tag = f"density_{density:03d}"
        dens_dir = os.path.join(OUTPUT_ROOT, dens_tag)

        # Create subdirectories
        train_dir = os.path.join(dens_dir, "train")
        val_dir = os.path.join(dens_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"  {dens_tag}: 2a∈{params['short_range']}, 2b∈{params['long_range']}, "
              f"IoU<{params['iou_max']}, max_restarts={params['restarts']}")
        print(f"  Target: {TRAIN_COUNT} train + {VAL_COUNT} val = {TOTAL_PER_DENSITY} images")
        print(f"{'='*70}")

        total_idx = 0
        restarts_log: list[int] = []
        failures = 0

        # Build argument list for all images
        all_args = [(i, density, params) for i in range(TOTAL_PER_DENSITY)]

        # Results collector (order-preserving)
        results: dict[int, tuple[np.ndarray, list[dict]]] = {}

        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures_map = {}
            for args in all_args:
                f = executor.submit(worker_generate, args)
                futures_map[f] = args[0]

            completed = 0
            for future in as_completed(futures_map):
                completed += 1
                idx, img, anns, rests = future.result()
                restarts_log.append(rests)

                if img is not None and anns is not None:
                    results[idx] = (img, anns)
                else:
                    failures += 1

                # Progress every 25 images or at completion
                if completed % 25 == 0 or completed == TOTAL_PER_DENSITY:
                    avg_restarts = sum(restarts_log) / len(restarts_log) if restarts_log else 0
                    print(f"  {dens_tag} generating: {completed}/{TOTAL_PER_DENSITY} "
                          f"(avg restarts: {avg_restarts:.1f}, failures so far: {failures})",
                          flush=True)

        # --- Write results in order ---
        print(f"  {dens_tag} writing files...")
        train_written = 0
        val_written = 0
        gt_counts: list[int] = []

        for idx in range(TOTAL_PER_DENSITY):
            if idx not in results:
                continue

            img, anns = results[idx]
            is_train = idx < TRAIN_COUNT
            out_dir = train_dir if is_train else val_dir
            local_idx = idx if is_train else idx - TRAIN_COUNT
            fname = f"{local_idx:06d}"

            cv2.imwrite(os.path.join(out_dir, f"{fname}.png"), img)
            write_annotation_txt(os.path.join(out_dir, f"{fname}.txt"), anns)
            gt_counts.append(len(anns))

            if is_train:
                train_written += 1
            else:
                val_written += 1

        elapsed = time.time() - t_start
        print(f"  {dens_tag} done: train={train_written}/{TRAIN_COUNT}, "
              f"val={val_written}/{VAL_COUNT}, failures={failures}, "
              f"avg gt/img={np.mean(gt_counts):.1f}, elapsed={elapsed:.0f}s")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  FULL DATASET GENERATION COMPLETE")
    print(f"  Total elapsed: {time.time() - t_start:.0f}s")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
