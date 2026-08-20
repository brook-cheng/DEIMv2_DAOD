#!/usr/bin/env python3
"""Stage 0 sample generation: ~50 synthetic ellipse OBB images for manual review.

Covers 7 density gradients [1, 2, 5, 10, 20, 50, 100] with ~7 images each.
Outputs raw PNG, DOTA-format TXT annotations, and visualization overlays
to synthetic_ellipse/samples/.
"""

import os
import sys
import numpy as np
import cv2

# shapely may raise if GEOS is missing; surface early
try:
    from shapely.geometry import Polygon  # type: ignore[import-untyped]
except ImportError:
    sys.exit("ERROR: shapely is required. Install with: pip install shapely")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_DIR = "synthetic_ellipse/samples"
IMG_SIZE = 256

DENSITIES = [1, 2, 5, 10, 20, 50, 100]
SAMPLES_PER_DENSITY = 7

MIN_ASPECT_RATIO = 1.5        # b/a
THETA_RANGE = (0.0, np.pi)    # radians

# Background gray level range
GRAY_RANGE = (50, 200)        # v ∈ [50, 200]

# ---------------------------------------------------------------------------
# Density-tiered generation parameters
#
# High densities (50, 100) are physically impossible with the spec-default
# ellipse sizes on a 256×256 canvas.  Per the design doc §6 risk-mitigation
# table, we down-scale ellipse axes and relax IoU when density exceeds 20 so
# that Stage 0 still produces reviewable samples for every gradient.
# ---------------------------------------------------------------------------

def _params(density: int) -> dict:
    """Return (short_axis_range, long_axis_range, iou_threshold, max_restarts)."""
    if density <= 20:
        return {
            "short_range": (20, 60),
            "long_range":  (40, 100),
            "iou_max":      0.05,
            "restarts":     15,
            "attempts":     500,
        }
    if density == 50:
        return {
            "short_range": (12, 36),
            "long_range":  (24, 60),
            "iou_max":      0.10,
            "restarts":     40,
            "attempts":     500,
        }
    return {
        "short_range": (10, 24),
        "long_range":  (20, 48),
        "iou_max":      0.15,
        "restarts":     60,
        "attempts":     500,
    }

# Per-category RGB base ranges (before ±5 pixel noise)
CATEGORY_COLORS_RGB: dict[str, dict[str, tuple[int, int]]] = {
    "r": {"R": (170, 230), "G": (10, 30), "B": (10, 30)},
    "g": {"R": (10, 30), "G": (170, 230), "B": (10, 30)},
    "b": {"R": (10, 30), "G": (10, 30), "B": (170, 230)},
}

CATEGORY_CYCLE = ["r", "g", "b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def random_gray_background() -> tuple[int, int, int]:
    """Return an RGB gray triple (v, v, v) with v ∈ [50, 200]."""
    v = np.random.randint(*GRAY_RANGE)
    return (v, v, v)


def random_color(cat: str) -> tuple[int, int, int]:
    """Return an (R, G, B) tuple for *cat* with ±5 per-channel noise.

    Samples each channel uniformly from the base range, adds uniform
    jitter in [-5, +5], and clamps to [0, 255].
    """
    spec = CATEGORY_COLORS_RGB[cat]
    rgb: list[int] = []
    for ch in ("R", "G", "B"):
        lo, hi = spec[ch]
        val = np.random.randint(lo, hi + 1)
        val += np.random.randint(-5, 6)
        val = max(0, min(255, val))
        rgb.append(val)
    return (rgb[0], rgb[1], rgb[2])  # (R, G, B)


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert (R, G, B) → (B, G, R) for OpenCV drawing."""
    return (rgb[2], rgb[1], rgb[0])


def compute_obb_vertices(
    cx: float, cy: float, a: float, b: float, theta: float,
) -> np.ndarray:
    """Compute the 4 vertices of the ellipse's OBB (clockwise order).

    Args:
        cx, cy:  ellipse centre (pixel coords).
        a:       semi-short axis half-length.
        b:       semi-long axis half-length (b >= a).
        theta:   rotation angle in radians [0, π).

    Returns:
        ndarray of shape (4, 2), dtype float64.
    """
    # Un-rotated corners (clockwise in image coords: TL → TR → BR → BL)
    corners_local = np.array([
        [-a, -b],
        [ a, -b],
        [ a,  b],
        [-a,  b],
    ], dtype=np.float64)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # Spec §3.6.1 rotation matrix
    R = np.array([[cos_t, -sin_t],
                   [sin_t,  cos_t]], dtype=np.float64)

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
    """Return True if the new OBB does *not* excessively overlap any existing one.

    IoU is computed via exact polygon intersection / union.
    """
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
# Image generation
# ---------------------------------------------------------------------------

def generate_image(density: int, params: dict) -> tuple[np.ndarray | None, list[dict] | None]:
    """Try to create one synthetic image with *density* non-overlapping ellipses.

    Returns:
        (img_bgr, annotations) on success, (None, None) on failure.
        *img_bgr* is a (256, 256, 3) uint8 ndarray (BGR).
        *annotations* is a list of dicts per ellipse.
    """
    gray = random_gray_background()
    gray_bgr = (gray[2], gray[1], gray[0])  # gray so R=G=B, any order works
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
            axis_2a = np.random.uniform(*short_range)
            axis_2b = np.random.uniform(*long_range)
            a = axis_2a / 2.0  # semi-short
            b = axis_2b / 2.0  # semi-long

            # enforce aspect ratio
            if b / a < MIN_ASPECT_RATIO:
                continue

            theta = np.random.uniform(*THETA_RANGE)
            cx = np.random.uniform(0, IMG_SIZE)
            cy = np.random.uniform(0, IMG_SIZE)

            vertices = compute_obb_vertices(cx, cy, a, b, theta)

            if not is_contained(vertices):
                continue
            if not passes_collision(vertices, existing_polys, threshold=iou_max):
                continue

            # --- accepted ---
            placed = True
            rgb = random_color(cat)
            bgr = rgb_to_bgr(rgb)

            # DOTA vertex list: float with 1 decimal
            verts_flat: list[float] = []
            for v in vertices:
                verts_flat.append(round(float(v[0]), 1))
                verts_flat.append(round(float(v[1]), 1))

            annotations.append({
                "cat": cat,
                "cx": cx,
                "cy": cy,
                "a": a,
                "b": b,
                "theta": theta,
                "rgb": rgb,
                "bgr": bgr,
                "vertices": vertices,
                "verts_flat": verts_flat,
            })

            existing_polys.append(obb_to_polygon(vertices))
            break

        if not placed:
            return None, None

    # --- draw all ellipses ---
    for ann in annotations:
        center = (int(round(ann["cx"])), int(round(ann["cy"])))
        axes = (int(round(ann["a"])), int(round(ann["b"])))  # cv2: (half_w, half_h)
        angle_deg = float(np.degrees(ann["theta"]))
        cv2.ellipse(
            img, center, axes, angle_deg,
            0.0, 360.0, ann["bgr"], thickness=-1,
        )

    return img, annotations


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_viz(img_bgr: np.ndarray, annotations: list[dict]) -> np.ndarray:
    """Overlay OBB rectangles + vertices + labels on a copy of *img_bgr*."""
    viz = img_bgr.copy()

    for ann in annotations:
        verts_i32 = ann["vertices"].astype(np.int32)

        # green polyline — OBB bounding rectangle
        cv2.polylines(
            viz, [verts_i32], isClosed=True,
            color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA,
        )

        # blue filled circles — 4 vertices
        for v in verts_i32:
            cv2.circle(
                viz, (int(v[0]), int(v[1])),
                radius=3, color=(255, 0, 0), thickness=-1, lineType=cv2.LINE_AA,
            )

        # white text — category name at ellipse centre
        cx = int(round(ann["cx"]))
        cy = int(round(ann["cy"]))
        (tw, th), _ = cv2.getTextSize(
            ann["cat"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
        )
        cv2.putText(
            viz, ann["cat"],
            (cx - tw // 2, cy + th // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return viz


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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # classes.txt
    classes_path = os.path.join(OUTPUT_DIR, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("r\ng\nb\n")
    print(f"Wrote {classes_path}")

    # state
    density_counts: dict[int, int] = {d: 0 for d in DENSITIES}
    total_generated = 0
    failures: list[str] = []

    for density in DENSITIES:
        params = _params(density)
        max_restarts = params["restarts"]
        print(f"\n--- Density {density:3d} (target {SAMPLES_PER_DENSITY}, "
              f"2a∈{params['short_range']}, 2b∈{params['long_range']}, "
              f"IoU<{params['iou_max']}) ---")

        for img_idx in range(SAMPLES_PER_DENSITY):
            success = False

            for restart in range(max_restarts):
                img_bgr, annotations = generate_image(density, params)
                if img_bgr is not None and annotations is not None:
                    success = True
                    break
                if restart > 0 and restart % 10 == 0:
                    print(f"    restart attempt {restart}...")

            if not success:
                tag = f"d{density:03d}_{img_idx:06d}"
                failures.append(f"{tag}: failed after {max_restarts} restarts")
                print(f"  ✗ {tag} — FAILED")
                continue

            base = f"d{density:03d}_{img_idx:06d}"

            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}.png"), img_bgr)
            write_annotation_txt(os.path.join(OUTPUT_DIR, f"{base}.txt"), annotations)

            viz = make_viz(img_bgr, annotations)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base}_viz.png"), viz)

            density_counts[density] += 1
            total_generated += 1
            print(f"  ✓ {base}  ({len(annotations)} ellipses)")

        print(f"  Density {density:3d}: {density_counts[density]}/{SAMPLES_PER_DENSITY} done")

    # ---- summary ----
    print("\n" + "=" * 60)
    print("  GENERATION SUMMARY")
    print("=" * 60)
    total_target = len(DENSITIES) * SAMPLES_PER_DENSITY
    print(f"  Total images generated : {total_generated} / {total_target}")
    print(f"  Output directory        : {os.path.abspath(OUTPUT_DIR)}")
    print()

    for d in DENSITIES:
        bar = "█" * density_counts[d] + "░" * (SAMPLES_PER_DENSITY - density_counts[d])
        print(f"  density {d:3d}  {density_counts[d]:2d}/{SAMPLES_PER_DENSITY}  {bar}")

    if failures:
        print(f"\n  Failures ({len(failures)}):")
        for f in failures:
            print(f"    {f}")
    else:
        print("\n  No failures.")

    print("=" * 60)


if __name__ == "__main__":
    main()
