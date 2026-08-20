#!/usr/bin/env python3
"""Validate the full synthetic ellipse OBB dataset.

Checks:
  1. Directory structure and file counts (train=400, val=100, matching .txt)
  2. DOTA format correctness (8 vertex coords + category + difficulty)
  3. OBB vertex bounds [0, 255]
  4. Pairwise IoU within threshold on sampled images
  5. Outputs validation_report.json
"""

import os
import sys
import json
import random
import numpy as np

try:
    from shapely.geometry import Polygon  # type: ignore[import-untyped]
except ImportError:
    sys.exit("ERROR: shapely is required. Install with: pip install shapely")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_ROOT = "/mnt/d/project_data/model_test/deimv2_obb_train_data/synthetic_ellipse"
DENSITIES = [1, 2, 5, 10, 20, 50, 100]
EXPECTED_TRAIN = 400
EXPECTED_VAL = 100
IMG_SIZE = 256
SAMPLE_COUNT = 5  # random images per density to deep-check


def iou_threshold(density: int) -> float:
    """Return the expected max IoU threshold for a given density."""
    if density <= 20:
        return 0.05
    if density == 50:
        return 0.10
    return 0.15


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_dota_line(line: str) -> dict | None:
    """Parse a single DOTA annotation line.

    Format: x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

    Returns dict with vertices, category, difficulty, or None on parse failure.
    """
    parts = line.strip().split()
    if len(parts) < 10:
        return None

    try:
        coords = [float(x) for x in parts[:8]]
        category = parts[8]
        difficulty = int(parts[9])
    except (ValueError, IndexError):
        return None

    # Build (4, 2) vertex array
    vertices = np.array([
        [coords[0], coords[1]],
        [coords[2], coords[3]],
        [coords[4], coords[5]],
        [coords[6], coords[7]],
    ], dtype=np.float64)

    return {
        "vertices": vertices,
        "category": category,
        "difficulty": difficulty,
    }


def check_bounds(vertices: np.ndarray) -> bool:
    """Return True if all vertices are within [0, 255]."""
    return bool(np.all((vertices >= 0.0) & (vertices <= 255.0)))


def compute_iou(verts_a: np.ndarray, verts_b: np.ndarray) -> float:
    """Compute IoU between two OBB polygons."""
    poly_a = Polygon(verts_a.tolist())
    poly_b = Polygon(verts_b.tolist())

    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)

    if not poly_a.intersects(poly_b):
        return 0.0

    inter = poly_a.intersection(poly_b)
    if inter.is_empty:
        return 0.0

    union = poly_a.union(poly_b)
    if union.area <= 0.0:
        return 0.0

    return inter.area / union.area


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------


def main() -> None:
    if not os.path.isdir(OUTPUT_ROOT):
        print(f"ERROR: output root not found: {OUTPUT_ROOT}")
        sys.exit(1)

    # Check classes.txt
    classes_path = os.path.join(OUTPUT_ROOT, "classes.txt")
    if os.path.isfile(classes_path):
        with open(classes_path, "r") as f:
            classes_content = f.read().strip()
        print(f"[classes.txt] content: {repr(classes_content)}")
    else:
        print("WARNING: classes.txt not found")

    report: dict = {}
    overall_pass = True

    for density in DENSITIES:
        dens_tag = f"density_{density:03d}"
        dens_dir = os.path.join(OUTPUT_ROOT, dens_tag)
        threshold = iou_threshold(density)

        print(f"\n--- {dens_tag} (IoU threshold < {threshold}) ---")

        # ---- Check directory existence ----
        train_dir = os.path.join(dens_dir, "train")
        val_dir = os.path.join(dens_dir, "val")

        if not os.path.isdir(train_dir):
            print(f"  FAIL: missing train directory")
            report[dens_tag] = {"passed": False, "error": "missing train directory"}
            overall_pass = False
            continue
        if not os.path.isdir(val_dir):
            print(f"  FAIL: missing val directory")
            report[dens_tag] = {"passed": False, "error": "missing val directory"}
            overall_pass = False
            continue

        # ---- Count images ----
        def count_files(subdir: str) -> tuple[int, int]:
            """Return (png_count, txt_count) for a directory."""
            pngs = sorted([f for f in os.listdir(subdir) if f.endswith(".png")])
            txts = sorted([f for f in os.listdir(subdir) if f.endswith(".txt")])
            return len(pngs), len(txts)

        train_png, train_txt = count_files(train_dir)
        val_png, val_txt = count_files(val_dir)

        print(f"  train: {train_png} png, {train_txt} txt  (expected {EXPECTED_TRAIN} each)")
        print(f"  val:   {val_png} png, {val_txt} txt  (expected {EXPECTED_VAL} each)")

        count_ok = True
        if train_png != EXPECTED_TRAIN:
            print(f"  FAIL: train png count mismatch ({train_png} != {EXPECTED_TRAIN})")
            count_ok = False
        if train_txt != EXPECTED_TRAIN:
            print(f"  FAIL: train txt count mismatch ({train_txt} != {EXPECTED_TRAIN})")
            count_ok = False
        if val_png != EXPECTED_VAL:
            print(f"  FAIL: val png count mismatch ({val_png} != {EXPECTED_VAL})")
            count_ok = False
        if val_txt != EXPECTED_VAL:
            print(f"  FAIL: val txt count mismatch ({val_txt} != {EXPECTED_VAL})")
            count_ok = False

        # ---- Compute total GT counts ----
        def sum_gt(subdir: str) -> int:
            total = 0
            for fname in sorted(os.listdir(subdir)):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(subdir, fname)
                with open(fpath, "r") as f:
                    lines = [l for l in f if l.strip()]
                total += len(lines)
            return total

        train_gt_total = sum_gt(train_dir)
        val_gt_total = sum_gt(val_dir)
        print(f"  total GT boxes: train={train_gt_total}, val={val_gt_total}")

        # ---- Sample deep-check ----
        # Gather all png files from both train and val
        all_images: list[tuple[str, str]] = []
        for subdir, label in [(train_dir, "train"), (val_dir, "val")]:
            for fname in sorted(os.listdir(subdir)):
                if fname.endswith(".png"):
                    base = fname[:-4]
                    all_images.append((subdir, base))

        sample_n = min(SAMPLE_COUNT, len(all_images))
        sampled = random.Random(42).sample(all_images, sample_n)

        format_ok = True
        bounds_ok = True
        iou_ok = True
        iou_violations: list[str] = []

        for subdir, base in sampled:
            txt_path = os.path.join(subdir, f"{base}.txt")
            with open(txt_path, "r") as f:
                lines = [l for l in f if l.strip()]

            # Parse all OBBs
            obbs = []
            for line in lines:
                parsed = parse_dota_line(line)
                if parsed is None:
                    print(f"  FAIL: DOTA parse error in {subdir}/{base}.txt: {line!r}")
                    format_ok = False
                    continue
                obbs.append(parsed)

            # Check category
            for obb in obbs:
                if obb["category"] not in ("r", "g", "b"):
                    print(f"  FAIL: invalid category '{obb['category']}' in {subdir}/{base}.txt")
                    format_ok = False
                if obb["difficulty"] != 0:
                    print(f"  FAIL: non-zero difficulty {obb['difficulty']} in {subdir}/{base}.txt")
                    format_ok = False

            # Check bounds
            for i, obb in enumerate(obbs):
                if not check_bounds(obb["vertices"]):
                    print(f"  FAIL: OBB {i} out of bounds in {subdir}/{base}.txt")
                    bounds_ok = False

            # Check pairwise IoU (O(n²) but n ≤ 100 so fine)
            n = len(obbs)
            for i in range(n):
                for j in range(i + 1, n):
                    iou = compute_iou(obbs[i]["vertices"], obbs[j]["vertices"])
                    if iou >= threshold:
                        violation = (
                            f"  IoU violation: {subdir}/{base}.txt "
                            f"OBBs [{i},{j}] IoU={iou:.6f} >= threshold={threshold}"
                        )
                        iou_violations.append(violation)
                        iou_ok = False

        for v in iou_violations[:5]:  # show first 5 only
            print(v)
        if len(iou_violations) > 5:
            print(f"  ... and {len(iou_violations) - 5} more IoU violations")

        if format_ok:
            print(f"  ✓ DOTA format: OK")
        if bounds_ok:
            print(f"  ✓ Vertex bounds: OK")
        if iou_ok:
            print(f"  ✓ IoU check ({len(sampled)} sampled): OK")
        else:
            print(f"  ✗ IoU check: {len(iou_violations)} violations found")

        passed = count_ok and format_ok and bounds_ok and iou_ok
        report[dens_tag] = {
            "train_images": train_png,
            "train_gt_total": train_gt_total,
            "val_images": val_png,
            "val_gt_total": val_gt_total,
            "passed": passed,
        }

        if not passed:
            overall_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {dens_tag}")

    # ---- Write report ----
    report_path = os.path.join(OUTPUT_ROOT, "validation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[validation_report.json] written to {report_path}")

    # ---- Final summary ----
    print(f"\n{'='*50}")
    if overall_pass:
        print("  ALL DENSITIES PASSED VALIDATION")
    else:
        print("  SOME DENSITIES FAILED VALIDATION")
    print(f"{'='*50}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
